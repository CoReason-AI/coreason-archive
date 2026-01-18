# The Architecture and Utility of coreason-archive

### 1. The Philosophy (The Why)

In the rapidly evolving landscape of Enterprise AI, the most critical failure mode is not a lack of intelligence, but "Digital Amnesia." Standard Retrieval Augmented Generation (RAG) systems are adept at searching static documents, but they fail to capture the dynamic, evolving context of reasoning. They treat every query as Day One.

**coreason-archive** was built to solve this by serving as the "Hippocampus" for the CoReason ecosystem. Its core philosophy is simple yet profound: **"Context is King. Compute Once, Recall Forever."**

The tool addresses the need for a **Hybrid Neuro-Symbolic Memory System**. Pure vector search suffers from "Context Collapse"—it finds similar words but misses structural relationships. Pure graph databases are too rigid for fuzzy reasoning. By fusing these approaches, `coreason-archive` enables agents to recall not just *what* information exists, but *why* a decision was made and *how* it relates to specific projects or departments, all while enforcing strict data sovereignty through its "Federated Scoping" architecture.

### 2. Under the Hood (The Dependencies & logic)

The architectural elegance of `coreason-archive` lies in its minimalistic yet powerful dependency stack, chosen to support a lightweight, portable, and robust memory system.

*   **`networkx`**: This powers the **Symbolic** side of the memory. It manages the Knowledge Graph in-memory, allowing for agile graph traversals (e.g., finding all thoughts related to "Project:Apollo") without the overhead of a heavy external graph database for the MVP.
*   **`numpy`**: The backbone of the **Semantic** layer. It handles the efficient storage and similarity calculation of the 1536-dimensional embeddings that represent the "thought vectors."
*   **`pydantic`**: Ensures rigorous data validation for the `CachedThought` schema, guaranteeing that every memory stored adheres to the strict contract required by the enterprise ecosystem.
*   **`loguru`**: Provides clear, structured observability into the memory operations.

Internally, the `CoreasonArchive` class acts as a facade, orchestrating a sophisticated **Scope-Link-Rank-Retrieve Loop**:

1.  **Federation Broker**: Before retrieval begins, it constructs a dynamic filter based on the user's identity and roles, ensuring they never see memories from scopes they don't access (e.g., a "Safety" user won't see "HR" secrets).
2.  **Hybrid Retrieval**: It combines vector similarity search with a "Graph Boost." If a memory is semantically distant but structurally linked to the user's active project, it gets promoted.
3.  **Temporal Ranker**: It applies an active epistemic decay formula ($S_{final} = S_{vector} \times e^{-\lambda \Delta t}$), ensuring that older, potentially obsolete reasoning traces naturally fade in relevance unless explicitly reinforced.
4.  **Matchmaker**: Finally, the system decides whether to return a "Semantic Hint" (to guide reasoning) or an "Exact Hit" (to skip computation entirely), optimizing both cost and latency.

### 3. In Practice (The How)

The API is designed to be intuitive for Python developers, abstracting the complexity of the hybrid store behind a clean interface.

**Initialization**
Setting up the archive involves initializing the vector and graph stores. In a production environment, these would be backed by persistent storage, but the interface remains consistent.

```python
from coreason_archive.archive import CoreasonArchive
from coreason_archive.vector_store import VectorStore
from coreason_archive.graph_store import GraphStore
from coreason_archive.utils.stubs import StubEmbedder

# Initialize the hybrid storage engine
archive = CoreasonArchive(
    vector_store=VectorStore(),
    graph_store=GraphStore(),
    embedder=StubEmbedder() # Replaced with actual embedding service in prod
)
```

**Ingesting a Thought (Neuro-Symbolic Memory)**
When an agent completes a reasoning task, the trace is stored with rich metadata. Note how we explicitly define the `scope` to ensure data sovereignty.

```python
from coreason_archive.models import MemoryScope

# The agent computes a complex result
prompt = "Calculate dosage for Patient X (Protocol Z)"
reasoning = "Protocol Z requires weight-based adjustment..."
final_answer = "Dosage: 150mg"

# Persist the cognitive state
await archive.add_thought(
    prompt=prompt,
    response=f"{reasoning}\nAnswer: {final_answer}",
    scope=MemoryScope.DEPARTMENT,
    scope_id="dept_oncology",
    user_id="dr_smith",
    ttl_seconds=86400 # Epistemic decay set to 1 day
)
```

**Smart Retrieval (The "Look before you Leap")**
Before starting a new computation, the agent consults the archive. The `smart_lookup` method intelligently determines the best strategy, potentially saving significant compute resources.

```python
from coreason_archive.federation import UserContext

# Context is crucial for the Federation Broker
context = UserContext(
    user_id="dr_jones",
    dept_ids=["dept_oncology"],
    project_ids=["protocol_z_trials"]
)

# The archive decides: Do we need to think, or do we remember?
result = await archive.smart_lookup("dosage for protocol Z?", context)

if result.strategy.value == "EXACT_HIT":
    print(f"Cache Hit! Result: {result.content['response']}")
elif result.strategy.value == "SEMANTIC_HINT":
    print(f"Guidance found: {result.content['hint']}")
    # Inject hint into the next LLM prompt
else:
    print("No relevant memory. Proceeding with fresh computation.")
```
