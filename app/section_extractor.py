import re
import statistics
from typing import List, Dict


def title_case_density(text: str) -> float:
    """
    Measures how much of the text is in title or uppercase (indicative of headings).
    """
    words = text.split()
    if not words:
        return 0.0
    cap_words = [w for w in words if w.istitle() or w.isupper()]
    return len(cap_words) / len(words)


def is_heading(line: Dict, font_mean: float, font_std: float, bold_fonts: List[str]) -> bool:
    """
    Determines if a line is a heading based on multiple visual and textual heuristics.
    """
    text = line.get("text", "").strip()
    if not text or len(text) < 3:
        return False

    font_size = line.get("font_size", 0)
    font_name = line.get("font_name", "").lower()

    # Detect large font via z-score
    z_score = (font_size - font_mean) / font_std if font_std else 0
    is_large = z_score >= 1.0

    # Bold font detection
    is_bold = any(bold_font.lower() in font_name for bold_font in bold_fonts)

    # Title-case heuristic
    is_title_case = title_case_density(text) > 0.6

    # Common heading patterns
    heading_patterns = [
        r'^(?:[A-Z]{1,2}[\.\)]\s+)?(?:\d{1,2}[\.\)]?)+\s+\w+',  # "A.1", "1.2", "1) Intro"
        r'^(Chapter|Section|Part|Day)\s+\d+',                   # "Chapter 3", "Day 1"
        r'^[IVXLCDM]+\.\s+\w+',                                 # "I. Overview"
        r'^[A-Z][A-Za-z\s\-]{3,}$',                             # "Things to Do", "Local Cuisine"
    ]
    has_heading_pattern = any(re.match(p, text) for p in heading_patterns)

    # Noise filters
    noise_patterns = [
        r'\d{1,3}\s+\w+\s+\w+',           # Address-like: "123 Main Street"
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',   # Dates
        r'\b\w+@\w+\.\w+\b',              # Email
        r'^\d+$',                         # Pure numbers
    ]
    if any(re.search(p, text) for p in noise_patterns):
        return False

    return is_large or is_bold or has_heading_pattern or is_title_case


def extract_heading_candidates(lines: List[Dict], debug: bool = False) -> List[Dict]:
    """
    Extracts all strong heading candidates based on style, layout, and text cues.
    """
    if not lines:
        return []

    font_sizes = [line.get("font_size", 0) for line in lines if line.get("font_size", 0) > 0]
    font_mean = statistics.mean(font_sizes) if font_sizes else 11.0
    font_std = statistics.stdev(font_sizes) if len(font_sizes) > 1 else 1.0

    bold_fonts = ["Bold", "Black", "Heavy", "Semibold", "Arial-BoldMT", "Times-Bold"]

    headings = []
    for line in lines:
        if is_heading(line, font_mean, font_std, bold_fonts):
            if "page" in line and "y" in line:
                headings.append(line)
                if debug:
                    print(f"[HEADING] {line['text']} (page {line['page']}, y={line['y']})")
            elif debug:
                print(f"[SKIPPED - Missing page/y] {line}")

    return headings
