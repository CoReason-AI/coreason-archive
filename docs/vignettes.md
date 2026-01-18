# User Stories (Behavioral Expectations)

These vignettes illustrate the expected behavior of the coreason-archive system in various scenarios.

## Story A: The "Institutional Wisdom" (Graph Search)

*   **Context:** A Senior Scientist (retired) once solved a rare edge case with "Drug Z."
*   **Trigger:** A new Junior Scientist asks: "Issues with Drug Z?"
*   **Vector:** Low similarity (different phrasing).
*   **Graph:** Archive finds `Entity:Drug Z` -> `MENTIONED_IN` -> `Thought:999` (Senior Scientist).
*   **Result:** The system surfaces the old reasoning trace. "Warning: In 2024, we found Drug Z crystallizes at 5C."
*   **Value:** Critical knowledge preservation.

## Story B: The "Department Transfer" (Portability)

*   **Context:** User moves from R&D to Compliance.
*   **Event:** `coreason-identity` sends `ROLE_UPDATE`.
*   **Action:** Archive's Relocation Manager:
    1.  **Locks** R&D Dept memories (User can no longer see them).
    2.  **Migrates** User Personal memories (User preferences move with them).
    3.  **Sanitizes:** Deletes one personal note tagged "Secret R&D Formula."

## Story C: The "Contextual Hint" (Cheaper Reasoning)

*   **Context:** Agent needs to write a SQL query for a complex schema.
*   **Search:** Finds a similar SQL generation task from last week.
*   **Action:** Returns the Thought Trace (Schema logic) but not the Final Answer (different table).
*   **Result:** The Agent follows the retrieved logic pattern but applies it to the new table. Success rate +40%.
