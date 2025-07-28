from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
import heapq
import re

nltk.download("punkt")

model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_sentence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)  # Normalize whitespace
    if len(s) < 30:
        return ""
    if re.match(r"^[\W\d]+$", s):  # Only symbols or numbers
        return ""
    return s

def summarize_text(text: str, max_sentences: int = 5) -> str:
    if not text or len(text.strip()) < 100:
        return ""  # Skip summarizing tiny chunks

    raw_sentences = sent_tokenize(text)
    sentences = [clean_sentence(s) for s in raw_sentences]
    sentences = [s for s in sentences if s]

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Compute embeddings
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute relevance to the document as a whole
    doc_embedding = model.encode(text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(sentence_embeddings, doc_embedding).squeeze(1)

    # Top N sentences by similarity
    top_indices = heapq.nlargest(max_sentences, range(len(scores)), scores.__getitem__)
    top_indices.sort()

    summary = " ".join([sentences[i] for i in top_indices])
    return summary
