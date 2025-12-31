"""Demo retrieval module for finding similar demonstrations."""

from openadapt_ml.retrieval.embeddings import TextEmbedder
from openadapt_ml.retrieval.index import DemoIndex
from openadapt_ml.retrieval.retriever import DemoRetriever

__all__ = ["TextEmbedder", "DemoIndex", "DemoRetriever"]
