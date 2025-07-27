import re
from typing import List, Dict


def is_heading(line: Dict, avg_font_size: float, bold_fonts: List[str]) -> bool:
    """
    Determines if a line is a heading based on font size, font weight, and pattern.

    Args:
        line (Dict): Line dict with text, font_size, font_name, etc.
        avg_font_size (float): Average font size in the document.
        bold_fonts (List[str]): Known bold fonts.

    Returns:
        bool: True if the line is likely a heading.
    """
    text = line["text"].strip()

    if not text or len(text) < 3:
        return False

    font_size = line.get("font_size", 0)
    font_name = line.get("font_name", "").lower()

    # Heuristic rules
    is_bold = any(bold_font.lower() in font_name for bold_font in bold_fonts)
    is_large = font_size > avg_font_size + 1.5
    is_title_case = text.istitle() or text.isupper()

    # Pattern match (optional): numbered, capitalized
    has_heading_pattern = re.match(r'^([A-Z]|\d+)[\.\)]\s+\w+', text)

    return is_large or is_bold or has_heading_pattern or is_title_case


def extract_heading_candidates(lines: List[Dict], debug: bool = False) -> List[Dict]:
    """
    Extract likely headings from parsed PDF lines.

    Args:
        lines (List[Dict]): Output from `pdf_parser.py`
        debug (bool): Print debug info

    Returns:
        List[Dict]: Filtered heading lines
    """
    if not lines:
        return []

    font_sizes = [line["font_size"] for line in lines if line.get("font_size", 0) > 0]
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 11.0

    bold_fonts = ["Bold", "Black", "Heavy", "Semibold", "Arial-BoldMT", "Times-Bold"]

    headings = []
    for line in lines:
     if is_heading(line, avg_font_size, bold_fonts):
        # Defensive check to avoid KeyError
        if "page" in line and "y" in line:
            headings.append(line)
            if debug:
                print(f"[HEADING] {line['text']} (page {line['page']}, y={line['y']})")
        elif debug:
            print(f"[SKIPPED - Missing page/y] {line}")


    return headings
