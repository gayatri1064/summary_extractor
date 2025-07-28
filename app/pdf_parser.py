import fitz  # PyMuPDF
import pdfplumber
import logging

def safe_extract_text_by_page(file_path, include_layout=True):
    """
    Tries to extract text using PyMuPDF first, then pdfplumber if it fails.
    """
    try:
        return extract_text_pymupdf(file_path)
    except Exception as e:
        logging.warning(f"[!] PyMuPDF failed for {file_path}: {e}")
        logging.info("⏪ Falling back to pdfplumber...")
        return extract_text_pdfplumber(file_path)


def extract_text_pymupdf(file_path):
    """
    Extracts line-level text with visual features using PyMuPDF.
    """
    lines = []
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            for l in b.get("lines", []):
                for span in l.get("spans", []):
                    lines.append({
                        "text": span["text"],
                        "x": span["bbox"][0],
                        "y": span["bbox"][1],
                        "font": span.get("font", ""),
                        "size": span.get("size", 0),
                        "page": page_num
                    })
    return lines


def extract_text_pdfplumber(file_path):
    """
    Extracts text using pdfplumber as a fallback (no font/size available).
    """
    lines = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            for word in words:
                lines.append({
                    "text": word["text"],
                    "x": float(word["x0"]),
                    "y": float(word["top"]),
                    "font": "Unknown",
                    "size": 0,
                    "page": page_num
                })
    return lines
