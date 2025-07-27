import pdfplumber
from typing import List, Dict
import os

def extract_text_from_pdf(file_path: str, include_layout: bool = True) -> List[Dict]:
    """
    Extracts text line-by-line from the PDF file along with page number and optional layout features.

    Args:
        file_path (str): Path to the PDF file.
        include_layout (bool): Whether to include font size, x, y etc.

    Returns:
        List[Dict]: List of lines with metadata.
    """
    extracted_lines = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                if include_layout:
                    for obj in page.extract_words(extra_attrs=["size", "fontname", "x0", "top"]):
                        extracted_lines.append({
                            "text": obj["text"],
                            "page": page_num,
                            "font_size": obj.get("size", 0),
                            "font_name": obj.get("fontname", ""),
                            "x": obj.get("x0", 0),
                            "y": obj.get("top", 0),
                        })
                else:
                    lines = page.extract_text().split('\n') if page.extract_text() else []
                    for line in lines:
                        extracted_lines.append({
                            "text": line,
                            "page": page_num,
                        })
            except Exception as e:
                print(f"Error processing page {page_num} of {file_path}: {e}")

    return extracted_lines
