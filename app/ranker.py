# ranker.py

import os
import torch
from sentence_transformers import SentenceTransformer, util

class SectionRanker:
    def __init__(self, model_path="models/minilm", device="cpu"):
        self.device = device
        self.model = SentenceTransformer(model_path, device=self.device)

    def rank_sections(self, persona: str, job: str, sections: list, top_k=10):
        """
        Rank sections by relevance to persona and job using cosine similarity.

        Parameters:
        - persona: Persona description string
        - job: Job-to-be-done string
        - sections: List of dicts with keys: 'document', 'section_text', 'section_title', 'page_number'

        Returns:
        - Ranked list of sections with added key 'importance_rank'
        """
        query = f"{persona}. {job}"
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        section_texts = [s["section_text"] for s in sections]
        section_embeddings = self.model.encode(section_texts, convert_to_tensor=True)

        scores = util.cos_sim(query_embedding, section_embeddings)[0]
        scores = scores.cpu().tolist()

        # Attach scores and sort
        for i, score in enumerate(scores):
            sections[i]["importance_score"] = score

        # Sort and assign rank
        sections = sorted(sections, key=lambda x: x["importance_score"], reverse=True)
        for rank, section in enumerate(sections[:top_k], start=1):
            section["importance_rank"] = rank

        return sections[:top_k]

