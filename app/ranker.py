from sentence_transformers import SentenceTransformer, CrossEncoder, util
from collections import defaultdict
import numpy as np
import torch


def is_similar(candidate_emb, selected_embs, threshold=0.85):
    """
    Checks if a new embedding is too similar to any already selected ones.
    """
    return any(util.pytorch_cos_sim(candidate_emb, emb).item() > threshold for emb in selected_embs)


def rank_sections(
    sections, persona, task,
    bi_encoder: SentenceTransformer,
    cross_encoder: CrossEncoder,
    top_k: int = 10,
    per_doc_limit: int = 3,
    preselect_k: int = 50
):
    """
    Hybrid ranking using fast bi-encoder + accurate cross-encoder.

    Args:
        sections (List[Dict]): Section chunks from all PDFs.
        persona (str): Persona description (e.g., "Travel Planner").
        task (str): Task description (e.g., "Plan a trip of 4 days...").
        bi_encoder (SentenceTransformer): Fast embedding-based encoder.
        cross_encoder (CrossEncoder): For re-ranking best sections.
        top_k (int): Final number of sections to return.
        per_doc_limit (int): Max sections per document in final result.
        preselect_k (int): Number of top bi-encoder sections to re-rank.

    Returns:
        List[Dict]: Ranked sections with importance scores and ranks.
    """
    # Rich query context
    query_variants = [
        f"{persona} needs to {task}",
        f"The job is: {task}",
        f"A {persona}'s task: {task}",
        f"Useful content for {persona} doing {task}"
    ]
    query_embedding = bi_encoder.encode(query_variants, convert_to_tensor=True).mean(dim=0)

    doc_to_sections = defaultdict(list)
    for sec in sections:
        doc_to_sections[sec["document"]].append(sec)

    scored_sections = []
    for doc, sec_list in doc_to_sections.items():
        texts = [f"{s.get('heading', '')}. {s['text']}" for s in sec_list]
        embeddings = bi_encoder.encode(texts, convert_to_tensor=True)

        for i, s in enumerate(sec_list):
            score = util.pytorch_cos_sim(query_embedding, embeddings[i]).item()
            length_penalty = min(len(s["text"]) / 300, 1.0)
            score *= length_penalty

            s["importance_score_bi"] = score
            s["_embedding"] = embeddings[i]
            scored_sections.append(s)

    # Top preselect_k for cross-encoder reranking
    scored_sections.sort(key=lambda x: x["importance_score_bi"], reverse=True)
    preselected = scored_sections[:preselect_k]

    cross_inputs = [
        [f"{persona} needs to {task}", f"{s.get('heading', '')}. {s['text']}"]
        for s in preselected
    ]
    cross_scores = cross_encoder.predict(cross_inputs)

    for s, score in zip(preselected, cross_scores):
        s["importance_score"] = float(score)

    # Re-sort by final cross-encoder scores
    preselected.sort(key=lambda x: x["importance_score"], reverse=True)

    # Deduplicate & balance
    selected = []
    seen_docs = defaultdict(int)
    selected_embs = []

    for s in preselected:
        if seen_docs[s["document"]] >= per_doc_limit:
            continue
        if not is_similar(s["_embedding"], selected_embs):
            selected.append(s)
            selected_embs.append(s["_embedding"])
            seen_docs[s["document"]] += 1
        if len(selected) >= top_k:
            break

    for i, s in enumerate(selected):
        s["importance_rank"] = i + 1
        s.pop("_embedding", None)

    return selected
