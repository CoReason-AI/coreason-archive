# Usage and Implementation Guide

## Core Functional Requirements

### 1. The Hybrid Storage Engine (The Brain)

**Concept:** A unified interface over a Vector DB (e.g., Qdrant/Milvus) and a lightweight Graph Store (e.g., Neo4j/Memgraph or NetworkX for local).

*   **Vector Store (Semantic):** Stores the embedding(prompt) and embedding(thought_trace).
*   **Graph Store (Symbolic):** Stores entities as nodes (User:Alice, Project:Apollo, Concept:Dosing) and edges (CREATED, BELONGS_TO, RELATED_TO).
*   **Ingestion Logic:**
    1.  **Vectorize:** Embed the prompt and response.
    2.  **Extract:** Use NLP to extract entities (e.g., "Project Apollo", "Cisplatin").
    3.  **Link:** Create graph edges between the Thought ID and these entities.

### 2. The Federation Broker (The RBAC Engine)

**Concept:** A query planner that enforces strict data sovereignty.

*   **Dynamic Context Construction:**
    *   *Input:* user_id, active_roles (from OIDC).
    *   *Process:* Constructs a **Filter Expression**.
        *   filter = (scope='USER' AND user_id='123') OR (scope='DEPT' AND dept_id IN ['Oncology', 'Safety'])
*   **The "Relocation" Manager:**
    *   Listens to coreason-identity events.
    *   **On Role Change:** Revokes access to old DEPT scopes. Migrates USER scope data if it passes a "Sanitization Check" (PII/Secret scanning).

### 3. The Temporal Ranker (The Clock)

**Concept:** Adjusts similarity scores based on recency (Recency Bias).

*   **Formula:** $S_{final} = S_{vector} \times e^{-\lambda \Delta t}$
    *   Where $\Delta t$ is the time since creation.
    *   $\lambda$ (decay rate) depends on scope (High for User scratchpad, Low for Global facts).
*   **Value:** Ensures agents don't cite obsolete protocols just because they are semantically similar.

### 4. The Matchmaker (The Cache Strategy)

**Concept:** Decides "Lookup vs. Compute."

*   **Exact Hit (Similarity > 0.99):** Returns the cached JSON. (Zero Cost).
*   **Semantic Hint (0.85 - 0.99):** Returns the **Reasoning Trace** only.
    *   *Usage:* Injects into prompt: *"Similar problem solved previously. Consider this approach: [Step 1, Step 2...]"*
    *   *SOTA:* This is "Retrieval Augmented Thought" (RAT), significantly boosting reasoning performance.
*   **Entity Hop (Graph Search):**
    *   *Query:* "What did we decide about X?"
    *   *Graph:* Finds Concept:X -> RELATED_TO -> Thought:123.
    *   *Value:* Finds relevant context that might not be *semantically* similar but is *structurally* related.

## Integration Requirements

*   **coreason-mcp (The Grounding):**
    *   Archive entries must link back to source documents in MCP (source_urn). If the source doc is updated, the Archive entry is flagged as "Stale."
*   **coreason-identity (The Key):**
    *   Provides the JWT claims that determine which Scopes are visible.
*   **coreason-economist (The Buyer):**
    *   Checks Archive before every Cortex execution.
    *   Logs "Cache Hits" as "Cost Avoidance" metrics ($ Saved).
*   **coreason-validator (The Law):**
    *   Enforces the schema of the CachedThought object.

## Data Schema

### CachedThought (The Asset)

```python
class MemoryScope(str, Enum):
    USER = "USER"
    PROJECT = "PROJECT"
    DEPARTMENT = "DEPT"
    CLIENT = "CLIENT"

class CachedThought(BaseModel):
    id: UUID

    # Neuro-Symbolic Data
    vector: List[float]          # 1536-dim embedding
    entities: List[str]          # ["Project:Apollo", "Drug:X"]

    # Hierarchy
    scope: MemoryScope           # "DEPT"
    scope_id: str                # "dept_oncology"

    # Content
    prompt_text: str
    reasoning_trace: str         # The "How"
    final_response: str          # The "What"

    # Metadata
    source_urns: List[str]       # Links to MCP docs
    created_at: datetime
    ttl_seconds: int             # Decay factor
    access_roles: List[str]      # RBAC claims required
```

## Implementation Directives

1.  **Graph Abstraction:** Do not over-engineer the graph. Use NetworkX (in-memory) serialized to JSON for the MVP, or a simple adjacency list in Postgres (node_a, node_b, relation). A full Neo4j instance is optional for v1.
2.  **Vector/Graph Hybrid Query:**
    *   *Step 1:* Vector Search -> Top 20 Candidates.
    *   *Step 2:* Graph Traversal -> Boost score if Candidate is linked to Active Project Node.
    *   *Step 3:* Decay -> Apply Time decay.
3.  **Background Worker:** Use FastAPI BackgroundTasks for the "Entity Extraction" step. Do not block the user response while parsing entities for the graph.
