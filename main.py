import json
import os
from datetime import datetime
from collections import defaultdict

from app.pdf_parser import safe_extract_text_by_page as extract_text_by_page

from app.section_extractor import extract_heading_candidates
from app.ranker import rank_sections
from app.summarizer import summarize_text
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

COLLECTIONS_DIR = "collections"

def group_content_by_heading(lines, headings, max_lines=10):
    grouped = []
    headings = sorted(headings, key=lambda h: (h["page"], h["y"]))

    for i, heading in enumerate(headings):
        start_idx = lines.index(heading)
        end_idx = None

        for j in range(start_idx + 1, len(lines)):
            if lines[j]["page"] != heading["page"]:
                break
            if lines[j] in headings:
                end_idx = j
                break

        content_lines = lines[start_idx + 1:end_idx] if end_idx else lines[start_idx + 1:]
        content_text = " ".join([l["text"] for l in content_lines[:max_lines]])

        grouped.append({
            "document": heading["source"],
            "page": heading["page"],
            "section_title": heading["text"],
            "text": content_text
        })

    return grouped

def process_collection(collection_path):
    input_path = os.path.join(collection_path, "input.json")
    output_path = os.path.join(collection_path, "output.json")
    pdf_dir = os.path.join(collection_path, "pdfs")

    if not os.path.exists(input_path) or not os.path.exists(pdf_dir):
        print(f"[!] Skipping {collection_path} — missing input.json or pdfs/")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    documents = input_data["documents"]
    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]

    all_lines = []
    all_headings = []

    for doc in documents:
        file_path = os.path.join(pdf_dir, doc["filename"])
        if not os.path.exists(file_path):
            print(f"[!] Missing file: {file_path}")
            continue

        lines = extract_text_by_page(file_path, include_layout=True)

        for l in lines:
            l["source"] = doc["filename"]

        headings = extract_heading_candidates(lines)
        for h in headings:
            h["source"] = doc["filename"]

        all_lines.extend(lines)
        all_headings.extend(headings)

    section_candidates = group_content_by_heading(all_lines, all_headings)
    ranked_sections = rank_sections(section_candidates, persona, job, model)

    subsection_analysis = []
    extracted_sections = []

    for sec in ranked_sections:
        refined = summarize_text(sec["text"], max_sentences=5)

        extracted_sections.append({
            "document": sec["document"],
            "section_title": sec["section_title"],
            "importance_rank": sec["importance_rank"],
            "page_number": sec["page"]
        })

        subsection_analysis.append({
            "document": sec["document"],
            "refined_text": refined,
            "page_number": sec["page"]
        })

    output = {
        "metadata": {
            "input_documents": [d["filename"] for d in documents],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"[✓] Processed {collection_path}. Output saved to {output_path}")

def main():
    for folder in os.listdir(COLLECTIONS_DIR):
        collection_path = os.path.join(COLLECTIONS_DIR, folder)
        if os.path.isdir(collection_path):
            process_collection(collection_path)

if __name__ == "__main__":
    main()
