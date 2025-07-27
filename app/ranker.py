from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the light-weight embedding model once (good balance of size + performance)
model = SentenceTransformer("models/minilm")


def embed_text(text: str):
    return model.encode(text, convert_to_tensor=False)

def rank_sections(sections: List[Dict], persona: str, job: str, top_k: int = 5) -> List[Dict]:
    """
    Ranks sections based on similarity to the persona and job.

    Args:
        sections (List[Dict]): Each section dict contains text, title, page, doc name etc.
        persona (str): Persona description
        job (str): Job to be done
        top_k (int): Number of top sections to return

    Returns:
        List[Dict]: Top-k relevant sections sorted by score
    """
    prompt = f"{persona.strip()}. Task: {job.strip()}"
    job_embedding = embed_text(prompt)

    scored_sections = []
    for section in sections:
        combined_text = section["text"]
        section_embedding = embed_text(combined_text)

        similarity = cosine_similarity([job_embedding], [section_embedding])[0][0]
        scored_sections.append({
            **section,
            "similarity_score": float(similarity)
        })

    top_sections = sorted(scored_sections, key=lambda x: x["similarity_score"], reverse=True)[:top_k]

    # Add importance_rank
    for i, sec in enumerate(top_sections, 1):
        sec["importance_rank"] = i

    return top_sections
