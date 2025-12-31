"""Simple text embeddings for demo retrieval.

Uses TF-IDF for v1 - no ML models required.
"""

from __future__ import annotations

import re
from collections import Counter
from math import log, sqrt
from typing import List


class TextEmbedder:
    """Simple TF-IDF based text embedder.

    This is a minimal implementation for v1 that doesn't require
    any external ML libraries. Can be upgraded to sentence-transformers
    or other embedding models later.
    """

    def __init__(self) -> None:
        """Initialize the embedder."""
        self.documents: List[str] = []
        self.idf: dict[str, float] = {}
        self.vocab: set[str] = set()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase and split on non-alphanumeric.

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokens.
        """
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _compute_tf(self, tokens: List[str]) -> dict[str, float]:
        """Compute term frequency for a document.

        Args:
            tokens: List of tokens.

        Returns:
            Dictionary mapping term to frequency.
        """
        counter = Counter(tokens)
        total = len(tokens)
        if total == 0:
            return {}
        return {term: count / total for term, count in counter.items()}

    def fit(self, documents: List[str]) -> None:
        """Fit the IDF on a corpus of documents.

        Args:
            documents: List of text documents.
        """
        self.documents = documents
        self.vocab = set()

        # Count document frequency for each term
        doc_freq: dict[str, int] = {}
        for doc in documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            self.vocab.update(unique_tokens)
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Compute IDF: log(N / df)
        n_docs = len(documents)
        if n_docs == 0:
            self.idf = {}
        else:
            self.idf = {
                term: log(n_docs / df)
                for term, df in doc_freq.items()
            }

    def embed(self, text: str) -> dict[str, float]:
        """Convert text to TF-IDF vector (as sparse dict).

        Args:
            text: Input text.

        Returns:
            Dictionary mapping term to TF-IDF weight.
        """
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)

        # Multiply TF by IDF
        tfidf = {}
        for term, tf_val in tf.items():
            if term in self.idf:
                tfidf[term] = tf_val * self.idf[term]

        return tfidf

    def cosine_similarity(
        self,
        vec1: dict[str, float],
        vec2: dict[str, float],
    ) -> float:
        """Compute cosine similarity between two sparse vectors.

        Args:
            vec1: First TF-IDF vector.
            vec2: Second TF-IDF vector.

        Returns:
            Cosine similarity score in [0, 1].
        """
        # Compute dot product
        dot = 0.0
        for term in vec1:
            if term in vec2:
                dot += vec1[term] * vec2[term]

        # Compute magnitudes
        mag1 = sqrt(sum(v * v for v in vec1.values()))
        mag2 = sqrt(sum(v * v for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)
