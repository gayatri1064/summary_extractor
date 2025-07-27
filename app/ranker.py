from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def rank_sections(sections, persona, task, top_k=10):
    """
    Ranks section blocks based on semantic similarity to the given task.
    """
    query = f"{persona} needs to {task}"
    query_embedding = model.encode(query, convert_to_tensor=True)

    scored_sections = []
    for sec in sections:
        sec_embedding = model.encode(sec["text"], convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, sec_embedding).item()
        sec["importance_score"] = score
        scored_sections.append(sec)

    # Sort by score descending
    scored_sections.sort(key=lambda x: x["importance_score"], reverse=True)

    # Assign importance_rank
    for idx, sec in enumerate(scored_sections[:top_k]):
        sec["importance_rank"] = idx + 1

    return scored_sections[:top_k]