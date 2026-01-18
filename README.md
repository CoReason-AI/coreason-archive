# coreason-archive

**Persistence layer for "Cognitive State" across the CoReason ecosystem.**

[![CI](https://github.com/CoReason-AI/coreason_archive/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_archive/actions/workflows/ci.yml)

## Executive Summary

coreason-archive is the persistence layer for "Cognitive State" across the CoReason ecosystem. It addresses the critical failure mode of modern AI: **"Digital Amnesia."**

Standard RAG (Retrieval Augmented Generation) only looks at static documents (coreason-mcp). coreason-archive looks at **Dynamic Experience**. It stores the *reasoning traces*, *decisions*, and *user preferences* generated during runtime.

Version 3.0 upgrades the architecture from a simple Vector Cache to a **Hybrid Neuro-Symbolic Memory System**. It combines **Vector Search** (for semantic similarity) with a **Knowledge Graph** (for structural relationships) and a **Temporal Engine** (for time-decay). This ensures that an agent doesn't just recall "similar text" but understands "who, when, and why" a decision was made, respecting strict enterprise boundaries.

## Functional Philosophy

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

## Getting Started

### Prerequisites

- Python 3.12+
- Poetry

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/example/example.git
    cd my_python_project
    ```
2.  Install dependencies:
    ```sh
    poetry install
    ```

### Usage

-   Run the linter:
    ```sh
    poetry run pre-commit run --all-files
    ```
-   Run the tests:
    ```sh
    poetry run pytest
    ```

For detailed documentation, please refer to the `docs/` folder or the [MkDocs site](docs/index.md).
