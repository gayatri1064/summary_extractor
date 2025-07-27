# app/pdf_parser.py
import pdfplumber
from typing import List, Dict


def extract_text_by_page(pdf_path: str, include_layout: bool = False) -> List[Dict]:
    """
    Extracts text from a PDF file page by page, preserving layout if needed.

    Args:
        pdf_path (str): Path to the PDF file.
        include_layout (bool): If True, extract line-wise with font/position metadata.

    Returns:
        List[Dict]: List of {'text': ..., 'page': ..., 'font_size': ..., 'x': ..., 'y': ...}
    """
    extracted_lines = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            if include_layout:
                # Extract each line's position and style
                lines = page.extract_words(use_text_flow=True, keep_blank_chars=True)
                for word in lines:
                    text = word.get("text", "").strip()
                    if text:
                        extracted_lines.append({
                            "text": text,
                            "page": page_num,
                            "x": word.get("x0", 0),
                            "y": word.get("top", 0),
                            "font_size": word.get("size", 0)
                        })
            else:
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        cleaned = line.strip()
                        if cleaned:
                            extracted_lines.append({
                                "text": cleaned,
                                "page": page_num
                            })

    return extracted_lines
