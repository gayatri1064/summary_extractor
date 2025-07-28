from sentence_transformers import SentenceTransformer, util

def rank_sections(sections, persona, task, model, top_k=10):
    """
    Ranks section blocks based on semantic similarity to the given task.
    """
    query = f"{persona} needs to {task}"
    query_embedding = model.encode(query, convert_to_tensor=True)

    texts = [sec["text"] for sec in sections]
    section_embeddings = model.encode(texts, convert_to_tensor=True)

    scored_sections = []
    for i, sec in enumerate(sections):
        score = util.pytorch_cos_sim(query_embedding, section_embeddings[i]).item()
        sec["importance_score"] = score
        scored_sections.append(sec)

    scored_sections.sort(key=lambda x: x["importance_score"], reverse=True)
    for idx, sec in enumerate(scored_sections[:top_k]):
        sec["importance_rank"] = idx + 1

    return scored_sections[:top_k]
