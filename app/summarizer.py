from typing import List
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Ensure NLTK punkt tokenizer is downloaded
nltk.download('punkt', quiet=True)

def split_into_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text)

def summarize_text(text: str, max_sentences: int = 5) -> str:
    """
    Extractive summarization using TF-IDF sentence scoring.

    Args:
        text (str): Full section text
        max_sentences (int): Max number of top sentences to return

    Returns:
        str: Concise refined summary
    """
    sentences = split_into_sentences(text)

    if len(sentences) <= max_sentences:
        return text  # Already short

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Score: Sum of TF-IDF values for each sentence
    scores = tfidf_matrix.sum(axis=1).A1
    top_indices = np.argsort(scores)[::-1][:max_sentences]

    # Sort by original order (not score) for coherence
    top_sentences = [sentences[i] for i in sorted(top_indices)]

    return " ".join(top_sentences)
