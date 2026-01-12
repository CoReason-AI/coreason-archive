import asyncio
from datetime import datetime, timezone
from typing import Any, List, Optional, Set, Tuple
from uuid import uuid4

from coreason_archive.federation import FederationBroker, UserContext
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder, EntityExtractor
from coreason_archive.matchmaker import MatchStrategy, SearchResult
from coreason_archive.models import CachedThought, GraphEdgeType, MemoryScope
from coreason_archive.temporal import TemporalRanker
from coreason_archive.utils.logger import logger
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
    ) -> None:
        """
        Initialize the CoreasonArchive.

        Args:
            vector_store: The vector storage engine.
            graph_store: The graph storage engine.
            embedder: Service to generate embeddings.
            entity_extractor: Service to extract entities (optional).
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedder = embedder
        self.entity_extractor = entity_extractor
        self.temporal_ranker = TemporalRanker()
        self._background_tasks: Set[asyncio.Task[Any]] = set()

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

        # 4. Background Extraction
        if self.entity_extractor:
            task = asyncio.create_task(self.process_entities(thought, combined_text))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

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
        raw_candidates = self.vector_store.search(query_vector, limit=limit * 5, min_score=min_score)

        if not raw_candidates:
            return []

        # 2. Federation Filter
        filter_fn = FederationBroker.get_filter(context)
        filtered_candidates = []
        for thought, score in raw_candidates:
            if filter_fn(thought):
                filtered_candidates.append((thought, score))

        # 3. Graph Boost & 4. Temporal Decay
        scored_results: List[Tuple[CachedThought, float, dict[str, Any]]] = []

        # Pre-compute active project entities for boosting
        # We start with the projects the user is explicitly part of.
        active_projects = {f"Project:{pid}" for pid in context.project_ids}

        # Expand active_projects with 1-hop neighbors from GraphStore.
        # This implements the Neuro-Symbolic "Graph Traversal" boosting.
        # We want to boost thoughts that are LINKED to the active project, even if
        # they don't explicitly contain the Project entity itself.
        # E.g. Thought(Entity:A) --[RELATED]--> Project:Apollo
        boost_entities = set(active_projects)

        for project_entity in active_projects:
            # We check "both" directions because the relationship could be defined as:
            # 1. Project -> RELATED -> Entity (Outgoing)
            # 2. Entity -> BELONGS_TO -> Project (Incoming)
            neighbors = self.graph_store.get_related_entities(project_entity, direction="both")
            for neighbor, _relation in neighbors:
                boost_entities.add(neighbor)

        if len(boost_entities) > len(active_projects):
            logger.debug(f"Expanded boost entities from {len(active_projects)} to {len(boost_entities)}")

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
