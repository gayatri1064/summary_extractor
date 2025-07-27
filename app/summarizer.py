from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
import heapq

nltk.download('punkt')

model = SentenceTransformer('all-MiniLM-L6-v2')

def summarize_text(text: str, max_sentences: int = 5) -> str:
    if not text:
        return ""

    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Compute embeddings
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute relevance to the document itself
    doc_embedding = model.encode(text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(sentence_embeddings, doc_embedding).squeeze(1)

    # Top N sentences by relevance
    top_indices = heapq.nlargest(max_sentences, range(len(scores)), scores.__getitem__)
    top_indices.sort()  # preserve order of appearance

    return " ".join([sentences[i] for i in top_indices])
