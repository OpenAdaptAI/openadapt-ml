"""Demo retrieval module for finding similar demonstrations.

This module provides functionality for indexing and retrieving demonstrations
based on semantic similarity of task descriptions.

Main Components:
- DemoRetriever: Main class for indexing and retrieving demos
- Embedders: TFIDFEmbedder, SentenceTransformerEmbedder, OpenAIEmbedder
- DemoIndex: Legacy index class (use DemoRetriever instead)

Quick Start:
    from openadapt_ml.retrieval import DemoRetriever
    from openadapt_ml.schema import Episode

    # Create retriever (TF-IDF is default, no external dependencies)
    retriever = DemoRetriever()

    # Or use sentence-transformers for better semantic matching
    retriever = DemoRetriever(embedding_method="sentence_transformers")

    # Add demos
    retriever.add_demo(episode1)
    retriever.add_demo(episode2, app_name="Chrome", domain="github.com")

    # Build index (required before retrieval)
    retriever.build_index()

    # Retrieve similar demos
    results = retriever.retrieve("Turn off Night Shift", top_k=3)

    # Format for inclusion in a prompt
    prompt_text = retriever.format_for_prompt(results)

Embedding Methods:
    # TF-IDF (default, no dependencies)
    retriever = DemoRetriever(embedding_method="tfidf")

    # Sentence Transformers (recommended, requires: pip install sentence-transformers)
    retriever = DemoRetriever(
        embedding_method="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",  # Fast, 22MB
    )

    # OpenAI (requires: pip install openai, OPENAI_API_KEY env var)
    retriever = DemoRetriever(
        embedding_method="openai",
        embedding_model="text-embedding-3-small",
    )

See Also:
    - docs/demo_retrieval_design.md - Full design document
    - openadapt_ml/experiments/demo_prompt/ - Demo-conditioned prompting
"""

# Main retrieval class (recommended)
from openadapt_ml.retrieval.demo_retriever import (
    DemoRetriever,
    DemoMetadata,
    RetrievalResult,
)

# Embedders
from openadapt_ml.retrieval.embeddings import (
    BaseEmbedder,
    TFIDFEmbedder,
    TextEmbedder,  # Alias for TFIDFEmbedder (backward compat)
    SentenceTransformerEmbedder,
    OpenAIEmbedder,
    create_embedder,
)

# Legacy classes (for backward compatibility)
from openadapt_ml.retrieval.index import DemoIndex
from openadapt_ml.retrieval.retriever import DemoRetriever as LegacyDemoRetriever

__all__ = [
    # Main classes
    "DemoRetriever",
    "DemoMetadata",
    "RetrievalResult",
    # Embedders
    "BaseEmbedder",
    "TFIDFEmbedder",
    "TextEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "create_embedder",
    # Legacy (backward compat)
    "DemoIndex",
    "LegacyDemoRetriever",
]
