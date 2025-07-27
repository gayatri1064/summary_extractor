import json
import os
from datetime import datetime
from collections import defaultdict

from app.pdf_parser import extract_text_by_page
from app.section_extractor import extract_heading_candidates
from app.ranker import rank_sections
from app.summarizer import summarize_text

DATA_DIR = "data"
INPUT_FILE = "data/input.json"
OUTPUT_FILE = "output.json"

def load_input():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def group_content_by_heading(lines, headings, max_lines=10):
    """
    Assigns following lines to each heading until the next heading on the same page.
    """
    grouped = []
    headings = sorted(headings, key=lambda h: (h["page"], h["y"]))

    for i, heading in enumerate(headings):
        start_idx = lines.index(heading)
        end_idx = None

        # Stop at next heading on same or later page
        for j in range(start_idx + 1, len(lines)):
            if lines[j]["page"] != heading["page"]:
                break
            if lines[j] in headings:
                end_idx = j
                break

        content_lines = lines[start_idx+1:end_idx] if end_idx else lines[start_idx+1:]
        content_text = " ".join([l["text"] for l in content_lines[:max_lines]])
        
        grouped.append({
            "document": heading["source"],
            "page": heading["page"],
            "section_title": heading["text"],
            "text": content_text
        })

    return grouped

def main():
    input_data = load_input()
    documents = input_data["documents"]
    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]

    all_lines = []
    all_headings = []

    for doc in documents:
        file_path = os.path.join(DATA_DIR, doc["filename"])
        lines = extract_text_by_page(file_path, include_layout=True)
        
        # Attach source doc to each line
        for l in lines:
            l["source"] = doc["filename"]

        headings = extract_heading_candidates(lines)
        for h in headings:
            h["source"] = doc["filename"]

        all_lines.extend(lines)
        all_headings.extend(headings)

    # Group heading + surrounding content
    section_candidates = group_content_by_heading(all_lines, all_headings)

    # Rank by relevance
    ranked_sections = rank_sections(section_candidates, persona, job)

    # Refine content
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

    # Final Output
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

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"[✓] Done. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
