This system ranks and summarizes sections of multiple documents based on a defined **persona** and **task**, producing high-relevance, low-noise results for a specific use-case. It is designed to be fast, lightweight, and **fully local/offline**, making it suitable for constrained environments (e.g., ≤1GB model size, ≤60s execution).

---

## Approach

Given:
- A set of PDFs (already extracted into sectioned text),
- A persona (e.g., "Travel Planner"),
- A task (e.g., "Plan a 4-day trip for 10 college friends"),

The system:
1. **Embeds the persona + task as a rich query** using a Bi-Encoder (`all-MiniLM-L6-v2`).
2. **Scores each section** of the documents using semantic similarity with the query.
3. **Re-ranks top candidate sections** using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for improved accuracy.
4. **Applies filtering and deduplication**, ensuring diversity across documents.
5. **Returns structured JSON** with the most relevant sections and a ranked summary.

---

## Models and Libraries Used

- `sentence-transformers` (Bi-Encoder + Cross-Encoder):
  - `all-MiniLM-L6-v2` — fast embedding for initial ranking.
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` — precise reranking.
- `nltk` for sentence tokenization (`punkt`).
- `torch`, `numpy`, and `collections` for vector operations.
- Custom heuristics for:
  - Multi-query embedding.
  - Length penalty.
  - Document balancing and deduplication.

All models are downloaded once and stored **locally**, avoiding runtime internet dependency.

