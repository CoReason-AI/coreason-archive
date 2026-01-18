# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

import asyncio
from datetime import datetime, timezone
from typing import Any, List, Optional, Set, Tuple
from uuid import uuid4

from coreason_archive.federation import FederationBroker, UserContext
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder, EntityExtractor, TaskRunner
from coreason_archive.matchmaker import MatchStrategy, SearchResult
from coreason_archive.models import CachedThought, GraphEdgeType, MemoryScope
from coreason_archive.temporal import TemporalRanker
from coreason_archive.utils.logger import logger
from coreason_archive.utils.runners import AsyncIOTaskRunner
from coreason_archive.vector_store import VectorStore


class CoreasonArchive:
    """
    Facade for the Hybrid Neuro-Symbolic Memory System.
    Orchestrates VectorStore, GraphStore, and TemporalRanker.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedder: Embedder,
        entity_extractor: Optional[EntityExtractor] = None,
        task_runner: Optional[TaskRunner] = None,
    ) -> None:
        """
        Initialize the CoreasonArchive.

        Args:
            vector_store: The vector storage engine.
            graph_store: The graph storage engine.
            embedder: Service to generate embeddings.
            entity_extractor: Service to extract entities (optional).
            task_runner: Optional custom task runner. Defaults to AsyncIOTaskRunner.
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.entity_extractor = entity_extractor
        self.temporal_ranker = TemporalRanker()
        self.task_runner = task_runner or AsyncIOTaskRunner()
        # Deprecated: _background_tasks is now managed by the default runner or custom runner
        # We keep it for backward compatibility if any test accesses it directly,
        # but internal logic uses task_runner.
        self._background_tasks: Set[asyncio.Task[Any]] = set()
        # If using default runner, expose its set for tests that inspect it
        if isinstance(self.task_runner, AsyncIOTaskRunner):
            self._background_tasks = self.task_runner._background_tasks

    def define_entity_relationship(
        self,
        source: str,
        target: str,
        relation: GraphEdgeType,
    ) -> None:
        """
        Defines a structural relationship between two entities in the GraphStore.
        Useful for ingesting organizational hierarchy (e.g., Project:Apollo -> BELONGS_TO -> Department:RnD).

        Args:
            source: The source entity string (e.g. "Project:Apollo").
            target: The target entity string (e.g. "Department:RnD").
            relation: The type of relationship.
        """
        self.graph_store.add_relationship(source, target, relation)
        logger.info(f"Defined relationship: {source} -[{relation.value}]-> {target}")

    def invalidate_source(self, urn: str) -> int:
        """
        Flags all thoughts linked to the given source URN as stale.
        This handles the integration requirement where updated source documents
        must trigger a stale flag in the archive.

        Args:
            urn: The Uniform Resource Name of the updated source document.

        Returns:
            The number of thoughts flagged as stale.
        """
        count = self.vector_store.mark_stale_by_urn(urn)
        logger.info(f"Invalidated source {urn}, affecting {count} thoughts.")
        return count

    async def add_thought(
        self,
        prompt: str,
        response: str,
        scope: MemoryScope,
        scope_id: str,
        user_id: str,
        access_roles: Optional[List[str]] = None,
        source_urns: Optional[List[str]] = None,
        ttl_seconds: int = 86400,
    ) -> CachedThought:
        """
        Ingests a new thought into the archive.
        1. Vectorizes the content.
        2. Stores in VectorStore.
        3. Extracts entities and links in GraphStore.

        Args:
            prompt: The original user prompt.
            response: The system's response/reasoning.
            scope: The memory scope (USER, DEPT, etc.).
            scope_id: The identifier for the scope.
            user_id: The user creating the thought.
            access_roles: RBAC roles required to access.
            source_urns: Links to source documents.
            ttl_seconds: Time to live for decay (default 1 day).

        Returns:
            The created CachedThought.
        """
        # 1. Vectorize
        combined_text = f"{prompt}\n{response}"
        vector = self.embedder.embed(combined_text)

        # 2. Create Object
        thought = CachedThought(
            id=uuid4(),
            vector=vector,
            entities=[],  # Will be populated async
            scope=scope,
            scope_id=scope_id,
            prompt_text=prompt,
            reasoning_trace=response,
            final_response=response,
            source_urns=source_urns or [],
            created_at=datetime.now(timezone.utc),
            ttl_seconds=ttl_seconds,
            access_roles=access_roles or [],
        )

        # 3. Store in VectorStore
        self.vector_store.add(thought)
        logger.info(f"Added thought {thought.id} to VectorStore")

        # 4. Synchronous Graph Ingestion (Metadata Linking)
        # Create structural edges: User -> CREATED -> Thought
        user_node = f"User:{user_id}"
        thought_node = f"Thought:{thought.id}"
        self.graph_store.add_relationship(user_node, thought_node, GraphEdgeType.CREATED)

        # Create structural edges: Thought -> BELONGS_TO -> ScopeEntity
        scope_prefix = {
            MemoryScope.USER: "User",
            MemoryScope.PROJECT: "Project",
            MemoryScope.DEPARTMENT: "Department",
            MemoryScope.CLIENT: "Client",
        }.get(scope, "Context")

        if scope_prefix:
            scope_node = f"{scope_prefix}:{scope_id}"
            self.graph_store.add_relationship(thought_node, scope_node, GraphEdgeType.BELONGS_TO)
            logger.debug(f"Linked thought {thought.id} to scope {scope_node}")

        # 5. Background Extraction
        if self.entity_extractor:
            self.task_runner.run(self.process_entities(thought, combined_text))

        return thought

    async def process_entities(self, thought: CachedThought, text: str) -> None:
        """
        Extracts entities and updates the GraphStore.
        This is intended to be run as a background task.

        Args:
            thought: The thought object.
            text: The text to analyze for entities.
        """
        if not self.entity_extractor:
            return

        try:
            entities = await self.entity_extractor.extract(text)
            thought.entities = entities

            # Update GraphStore
            # Node for the Thought
            thought_node = f"Thought:{thought.id}"
            self.graph_store.add_entity(thought_node)

            for entity in entities:
                # Entity format expected: "Type:Value"
                self.graph_store.add_entity(entity)
                # Link Entity -> Thought (MENTIONED_IN or RELATED_TO)
                self.graph_store.add_relationship(entity, thought_node, GraphEdgeType.RELATED_TO)
                self.graph_store.add_relationship(thought_node, entity, GraphEdgeType.RELATED_TO)

            logger.info(f"Extracted {len(entities)} entities for thought {thought.id}")

        except Exception as e:
            logger.error(f"Failed to process entities for thought {thought.id}: {e}")

    async def retrieve(
        self,
        query: str,
        context: UserContext,
        limit: int = 10,
        min_score: float = 0.0,
        graph_boost_factor: float = 1.1,
    ) -> List[Tuple[CachedThought, float, dict[str, Any]]]:
        """
        Retrieves thoughts using the Scope-Link-Rank-Retrieve Loop.
        1. Vector Search (Semantic)
        2. Federation Filter (Scope/RBAC)
        3. Graph Boost (Structural)
        4. Temporal Decay (Recency)

        Args:
            query: The search query string.
            context: The user's security context.
            limit: Max results to return.
            min_score: Minimum score threshold (pre-decay).
            graph_boost_factor: Multiplier for score if structurally linked.

        Returns:
            List of (CachedThought, final_score, metadata) tuples, sorted by score.
        """
        # 1. Vector Search
        query_vector = self.embedder.embed(query)
        # Fetch more candidates than needed to account for filtering and re-ranking
        # Returns List[(thought, score)]
        vector_results = self.vector_store.search(query_vector, limit=limit * 5, min_score=min_score)

        # Initialize candidates dictionary {thought_id: (thought, base_score)}
        candidates_map = {t.id: (t, s) for t, s in vector_results}

        # 2. Graph Sourcing (Hybrid Retrieval)
        # Extract entities from query & context, expand graph, and find linked thoughts.
        # This ensures we find thoughts that are structurally relevant even if semantically distant.

        active_projects = {f"Project:{pid}" for pid in context.project_ids}
        boost_entities = set(active_projects)

        if self.entity_extractor:
            try:
                query_entities = await self.entity_extractor.extract(query)
                if query_entities:
                    logger.debug(f"Extracted entities from query: {query_entities}")
                    boost_entities.update(query_entities)
            except Exception as e:
                logger.warning(f"Failed to extract entities from query: {e}")

        # Expand ALL seed entities with 1-hop neighbors
        seed_entities = list(boost_entities)
        for seed_entity in seed_entities:
            neighbors = self.graph_store.get_related_entities(seed_entity, direction="both")
            for neighbor, _relation in neighbors:
                boost_entities.add(neighbor)

        if len(boost_entities) > len(seed_entities):
            logger.debug(f"Expanded boost entities from {len(seed_entities)} to {len(boost_entities)}")

        # Find thoughts linked to these boost_entities
        # In the GraphStore, thoughts are nodes "Thought:<uuid>"
        # We need to find nodes connected to boost_entities that start with "Thought:"
        # Or simpler: Is the thought node ITSELF in boost_entities?
        # Since we did 1-hop expansion, if a thought is linked to a seed entity (e.g., Drug:Z),
        # then "Thought:<id>" will be in neighbors (and thus in boost_entities).
        # So we just scan boost_entities for "Thought:..." nodes.

        # We need to handle UUID conversion carefully.
        # Python's uuid.UUID(str) works.
        from uuid import UUID

        valid_ids: List[UUID] = []
        for entity in boost_entities:
            if entity.startswith("Thought:"):
                try:
                    tid_str = entity.split("Thought:", 1)[1]
                    valid_ids.append(UUID(tid_str))
                except (ValueError, IndexError):
                    continue

        if valid_ids:
            # Batch fetch from VectorStore
            graph_thoughts = self.vector_store.get_by_ids(valid_ids)
            logger.debug(f"Graph Sourcing found {len(graph_thoughts)} thoughts linked to context.")

            for thought in graph_thoughts:
                if thought.id not in candidates_map:
                    # Calculate base score since it wasn't in vector results
                    score = self.vector_store.calculate_similarity(thought, query_vector)
                    candidates_map[thought.id] = (thought, score)

        # 3. Federation Filter
        filter_fn = FederationBroker.get_filter(context)
        filtered_candidates = []

        for thought, base_score in candidates_map.values():
            if filter_fn(thought):
                filtered_candidates.append((thought, base_score))

        if not filtered_candidates:
            return []

        # 4. Graph Boost & 5. Temporal Decay
        scored_results: List[Tuple[CachedThought, float, dict[str, Any]]] = []

        for thought, base_score in filtered_candidates:
            current_score = base_score
            is_boosted = False

            # Apply Graph Boost
            # Boost if the thought contains entities related to active context (direct or 1-hop)
            if thought.entities and not boost_entities.isdisjoint(thought.entities):
                current_score *= graph_boost_factor
                is_boosted = True
                logger.debug(f"Boosted thought {thought.id} (Graph Link)")

            # Apply Temporal Decay
            decay_factor = TemporalRanker.calculate_decay_factor(thought.scope, thought.created_at)
            final_score = current_score * decay_factor

            metadata = {
                "base_score": base_score,
                "is_boosted": is_boosted,
                "decay_factor": decay_factor,
            }

            scored_results.append((thought, final_score, metadata))

        # Sort by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results[:limit]

    async def smart_lookup(
        self,
        query: str,
        context: UserContext,
        exact_threshold: float = 0.99,
        hint_threshold: float = 0.85,
        graph_boost_factor: float = 1.1,
    ) -> SearchResult:
        """
        Orchestrates the "Lookup vs. Compute" decision logic (Matchmaker).

        Args:
            query: The search query.
            context: The user context.
            exact_threshold: Score above which we return full content.
            hint_threshold: Score above which we return a hint.
            graph_boost_factor: Multiplier for score if structurally linked.

        Returns:
            A SearchResult object containing the strategy and content.
        """
        # 1. Retrieve candidates
        results = await self.retrieve(query, context, limit=5, min_score=0.0, graph_boost_factor=graph_boost_factor)

        if not results:
            return SearchResult(
                strategy=MatchStrategy.STANDARD_RETRIEVAL,
                thought=None,
                score=0.0,
                content={"message": "No relevant memories found."},
            )

        top_thought, top_score, top_metadata = results[0]

        # 2. Decide Strategy
        if top_score >= exact_threshold:
            # Exact Hit: Return full cached JSON
            return SearchResult(
                strategy=MatchStrategy.EXACT_HIT,
                thought=top_thought,
                score=top_score,
                content={
                    "prompt": top_thought.prompt_text,
                    "reasoning": top_thought.reasoning_trace,
                    "response": top_thought.final_response,
                    "source": "cache_hit",
                },
            )

        elif top_score >= hint_threshold:
            # Semantic Hint: Return Reasoning Trace only
            return SearchResult(
                strategy=MatchStrategy.SEMANTIC_HINT,
                thought=top_thought,
                score=top_score,
                content={
                    "hint": f"Similar problem solved previously. Consider this approach: {top_thought.reasoning_trace}",
                    "source": "semantic_hint",
                },
            )

        elif top_metadata.get("is_boosted", False):
            # Entity Hop: High score driven by Graph Boost
            return SearchResult(
                strategy=MatchStrategy.ENTITY_HOP,
                thought=top_thought,
                score=top_score,
                content={
                    "hint": f"Found structurally related context (Entity Hop). Consider: {top_thought.reasoning_trace}",
                    "source": "entity_hop",
                    "reasoning": top_thought.reasoning_trace,
                },
            )

        else:
            # Standard Retrieval
            return SearchResult(
                strategy=MatchStrategy.STANDARD_RETRIEVAL,
                thought=top_thought,
                score=top_score,
                content={
                    "top_thoughts": [
                        {
                            "response": t.final_response,
                            "reasoning": t.reasoning_trace,
                            "score": s,
                        }
                        for t, s, _ in results
                    ]
                },
            )
