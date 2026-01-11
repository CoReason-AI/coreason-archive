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
    ) -> List[Tuple[CachedThought, float]]:
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
            List of (CachedThought, final_score) tuples, sorted by score.
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
        scored_results: List[Tuple[CachedThought, float]] = []

        # Pre-compute active project entities for boosting
        # Assuming project IDs in context match "Project:{id}" format loosely?
        # Or we strictly expect "Project:{id}".
        # Let's handle generic case: Check if thought.entities overlaps with context-derived entities?
        # Simpler: If thought is about a Project that is in context.project_ids.
        active_projects = {f"Project:{pid}" for pid in context.project_ids}

        for thought, base_score in filtered_candidates:
            current_score = base_score

            # Apply Graph Boost
            # Boost if the thought contains entities related to active context
            # Intersection of thought.entities and active_projects
            if thought.entities and not active_projects.isdisjoint(thought.entities):
                current_score *= graph_boost_factor
                logger.debug(f"Boosted thought {thought.id} (Graph Link)")

            # Apply Temporal Decay
            final_score = TemporalRanker.adjust_score(current_score, thought.scope, thought.created_at)

            scored_results.append((thought, final_score))

        # Sort by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results[:limit]

    async def smart_lookup(
        self,
        query: str,
        context: UserContext,
        exact_threshold: float = 0.99,
        hint_threshold: float = 0.85,
    ) -> SearchResult:
        """
        Orchestrates the "Lookup vs. Compute" decision logic (Matchmaker).

        Args:
            query: The search query.
            context: The user context.
            exact_threshold: Score above which we return full content.
            hint_threshold: Score above which we return a hint.

        Returns:
            A SearchResult object containing the strategy and content.
        """
        # 1. Retrieve candidates
        results = await self.retrieve(query, context, limit=5, min_score=0.0)

        if not results:
            return SearchResult(
                strategy=MatchStrategy.STANDARD_RETRIEVAL,
                thought=None,
                score=0.0,
                content={"message": "No relevant memories found."},
            )

        top_thought, top_score = results[0]

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

        else:
            # Low Score -> Standard Retrieval / Entity Hop
            # PRD implies "Entity Hop (Graph Search)" if semantic is low?
            # Or just fall back to standard retrieval of top K?
            # Since we already boosted via Graph in `retrieve`, the top result IS the best guess.
            # If it's still low score, it means neither semantic nor graph boosted it enough.
            # However, PRD says "Entity Hop" finds relevant context that might not be semantically similar.
            # If Graph Boost worked, it should have pushed a structurally related item up.
            # So we return the top item(s) as "Standard Retrieval" or "Entity Hop" if it was boosted?
            # For MVP, let's return the top thoughts as standard context.

            # Construct list of top thoughts
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
                        for t, s in results
                    ]
                },
            )
