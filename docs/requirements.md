# Product Requirements Document: coreason-archive

**Domain:** Enterprise Memory, Cognitive Continuity, & Knowledge Graph
**Architectural Role:** The "Hippocampus" (Long-Term Episodic & Semantic Memory)
**Core Philosophy:** "Context is King. Compute Once, Recall Forever. Truth is Relative to Scope."
**Dependencies:** coreason-identity (RBAC), coreason-economist (Arbitrage), coreason-mcp (Data Source)
**Python Libraries:** anyio, httpx, numpy, networkx, pydantic, loguru, fastapi, uvicorn

## 1. Executive Summary

coreason-archive is the persistence layer for "Cognitive State" across the CoReason ecosystem. It addresses the critical failure mode of modern AI: **"Digital Amnesia."**

Standard RAG (Retrieval Augmented Generation) only looks at static documents (coreason-mcp). coreason-archive looks at **Dynamic Experience**. It stores the *reasoning traces*, *decisions*, and *user preferences* generated during runtime.

Version 3.0 upgrades the architecture from a simple Vector Cache to a **Hybrid Neuro-Symbolic Memory System**. It combines **Vector Search** (for semantic similarity) with a **Knowledge Graph** (for structural relationships) and a **Temporal Engine** (for time-decay). This ensures that an agent doesn't just recall "similar text" but understands "who, when, and why" a decision was made, respecting strict enterprise boundaries.

## 2. Functional Philosophy

The agent must implement the **Scope-Link-Rank-Retrieve Loop**:

1.  **Hybrid Memory Structure (Neuro-Symbolic):**
    *   **Semantic (Vector):** "Find thoughts similar to 'Dosing Protocol'."
    *   **Structural (Graph):** "Find all thoughts linked to 'Project Apollo' and 'Dr. Smith'."
    *   **SOTA Best Practice:** Using vectors for fuzzy matching and graphs for explicit entity tracking prevents "Context Collapse" in complex workflows.
2.  **Federated Scoping (The Hierarchy of Truth):**
    *   Memory is not a flat bucket. It is a hierarchy: User > Project > Department > Global.
    *   A "User Preference" (e.g., "Don't use tables") overrides a "Global Default."
3.  **Active Epistemic Decay:**
    *   Knowledge has a half-life. A cached thought about "Q3 Strategy" is worthless in Q4.
    *   We implement **Time-Aware Retrieval** where older memories have lower retrieval scores unless explicitly pinned.
4.  **Memory Portability (The Digital Twin):**
    *   When a user moves departments, their *personal* cognitive state follows them, but their *former team's* secrets are left behind.

## 3. Core Functional Requirements (Component Level)

### 3.1 The Hybrid Storage Engine (The Brain)

**Concept:** A unified interface over a Vector DB (e.g., Qdrant/Milvus) and a lightweight Graph Store (e.g., Neo4j/Memgraph or NetworkX for local).

*   **Vector Store (Semantic):** Stores the embedding(prompt) and embedding(thought_trace).
*   **Graph Store (Symbolic):** Stores entities as nodes (User:Alice, Project:Apollo, Concept:Dosing) and edges (CREATED, BELONGS_TO, RELATED_TO).
*   **Ingestion Logic:**
    1.  **Vectorize:** Embed the prompt and response.
    2.  **Extract:** Use NLP to extract entities (e.g., "Project Apollo", "Cisplatin").
    3.  **Link:** Create graph edges between the Thought ID and these entities.

### 3.2 The Federation Broker (The RBAC Engine)

**Concept:** A query planner that enforces strict data sovereignty.

*   **Dynamic Context Construction:**
    *   *Input:* user_id, active_roles (from OIDC).
    *   *Process:* Constructs a **Filter Expression**.
        *   filter = (scope='USER' AND user_id='123') OR (scope='DEPT' AND dept_id IN ['Oncology', 'Safety'])
*   **The "Relocation" Manager:**
    *   Listens to coreason-identity events.
    *   **On Role Change:** Revokes access to old DEPT scopes. Migrates USER scope data if it passes a "Sanitization Check" (PII/Secret scanning).

### 3.3 The Temporal Ranker (The Clock)

**Concept:** Adjusts similarity scores based on recency (Recency Bias).

*   **Formula:** $S_{final} = S_{vector} \times e^{-\lambda \Delta t}$
    *   Where $\Delta t$ is the time since creation.
    *   $\lambda$ (decay rate) depends on scope (High for User scratchpad, Low for Global facts).
*   **Value:** Ensures agents don't cite obsolete protocols just because they are semantically similar.

### 3.4 The Matchmaker (The Cache Strategy)

**Concept:** Decides "Lookup vs. Compute."

*   **Exact Hit (Similarity > 0.99):** Returns the cached JSON. (Zero Cost).
*   **Semantic Hint (0.85 - 0.99):** Returns the **Reasoning Trace** only.
    *   *Usage:* Injects into prompt: *"Similar problem solved previously. Consider this approach: [Step 1, Step 2...]"*
    *   *SOTA:* This is "Retrieval Augmented Thought" (RAT), significantly boosting reasoning performance.
*   **Entity Hop (Graph Search):**
    *   *Query:* "What did we decide about X?"
    *   *Graph:* Finds Concept:X -> RELATED_TO -> Thought:123.
    *   *Value:* Finds relevant context that might not be *semantically* similar but is *structurally* related.

## 4. Integration Requirements (The Ecosystem)

*   **coreason-mcp (The Grounding):**
    *   Archive entries must link back to source documents in MCP (source_urn). If the source doc is updated, the Archive entry is flagged as "Stale."
*   **coreason-identity (The Key):**
    *   Provides the JWT claims that determine which Scopes are visible.
*   **coreason-economist (The Buyer):**
    *   Checks Archive before every Cortex execution.
    *   Logs "Cache Hits" as "Cost Avoidance" metrics ($ Saved).
*   **coreason-validator (The Law):**
    *   Enforces the schema of the CachedThought object.

## 5. User Stories (Behavioral Expectations)

### Story A: The "Institutional Wisdom" (Graph Search)

**Context:** A Senior Scientist (retired) once solved a rare edge case with "Drug Z."
**Trigger:** A new Junior Scientist asks: "Issues with Drug Z?"
**Vector:** Low similarity (different phrasing).
**Graph:** Archive finds Entity:Drug Z -> MENTIONED_IN -> Thought:999 (Senior Scientist).
**Result:** The system surfaces the old reasoning trace. "Warning: In 2024, we found Drug Z crystallizes at 5C."
**Value:** Critical knowledge preservation.

### Story B: The "Department Transfer" (Portability)

**Context:** User moves from R&D to Compliance.
**Event:** coreason-identity sends ROLE_UPDATE.
**Action:** Archive's Relocation Manager:

1.  **Locks** R&D Dept memories (User can no longer see them).
2.  **Migrates** User Personal memories (User preferences move with them).
3.  **Sanitizes:** Deletes one personal note tagged "Secret R&D Formula."

### Story C: The "Contextual Hint" (Cheaper Reasoning)

**Context:** Agent needs to write a SQL query for a complex schema.
**Search:** Finds a similar SQL generation task from last week.
**Action:** Returns the Thought Trace (Schema logic) but not the Final Answer (different table).
**Result:** The Agent follows the retrieved logic pattern but applies it to the new table. Success rate +40%.

## 6. Data Schema

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
    owner_id: str                # ID of the user who owns this thought
    source_urns: List[str]       # Links to MCP docs
    is_stale: bool               # Flag indicating if the source information is outdated
    created_at: datetime
    ttl_seconds: int             # Decay factor
    access_roles: List[str]      # RBAC claims required
```

## 7. Implementation Directives for the Coding Agent

1.  **Graph Abstraction:** Do not over-engineer the graph. Use NetworkX (in-memory) serialized to JSON for the MVP, or a simple adjacency list in Postgres (node_a, node_b, relation). A full Neo4j instance is optional for v1.
2.  **Vector/Graph Hybrid Query:**
    *   *Step 1:* Vector Search -> Top 20 Candidates.
    *   *Step 2:* Graph Traversal -> Boost score if Candidate is linked to Active Project Node.
    *   *Step 3:* Decay -> Apply Time decay.
3.  **Background Worker:** Use the `TaskRunner` protocol (defaulting to `AsyncIOTaskRunner` using `asyncio` or `anyio`) for the "Entity Extraction" step. Do not block the user response while parsing entities for the graph. This ensures the library remains framework-agnostic.
