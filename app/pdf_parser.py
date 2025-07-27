import pdfplumber
from typing import List, Dict


def extract_text_by_page(pdf_path: str, include_layout: bool = True) -> List[Dict]:
    """
    Extracts text from a PDF file page by page.

    Args:
        pdf_path (str): Path to the PDF file.
        include_layout (bool): If True, preserve layout by extracting line-wise text;
                               else extract full text block per page.

    Returns:
        List[Dict]: Each dict contains 'text', 'page', 'x', 'y', 'fontname', and 'size'
    """
    extracted_lines = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            if include_layout:
                for char_obj in page.extract_words(
                    keep_blank_chars=True,
                    use_text_flow=True,
                    x_tolerance=3,
                    y_tolerance=3
                ):
                    extracted_lines.append({
                        "text": char_obj.get("text", "").strip(),
                        "page": page_num,
                        "x": char_obj.get("x0", 0),
                        "y": char_obj.get("top", 0),
                        "fontname": char_obj.get("fontname", ""),
                        "size": char_obj.get("size", 0),
                    })
            else:
                text = page.extract_text()
                if text:
                    extracted_lines.append({
                        "text": text.strip(),
                        "page": page_num,
                        "x": 0,
                        "y": 0,
                        "fontname": "",
                        "size": 0
                    })

    return extracted_lines
