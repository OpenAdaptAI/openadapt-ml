# Demo Retrieval Module

This module provides functionality to index and retrieve similar demonstrations for few-shot prompting in GUI automation.

## Overview

The retrieval module consists of three main components:

1. **TextEmbedder** (`embeddings.py`) - Simple TF-IDF based text embeddings
2. **DemoIndex** (`index.py`) - Stores episodes with metadata and embeddings
3. **DemoRetriever** (`retriever.py`) - Retrieves top-K similar demos

## Quick Start

```python
from openadapt_ml.retrieval import DemoIndex, DemoRetriever
from openadapt_ml.schemas.sessions import Episode

# 1. Create index and add episodes
index = DemoIndex()
index.add_many(episodes)  # episodes is a list of Episode objects
index.build()  # Compute embeddings

# 2. Create retriever
retriever = DemoRetriever(index, domain_bonus=0.2)

# 3. Retrieve similar demos
task = "Turn off Night Shift on macOS"
app_context = "System Settings"
similar_demos = retriever.retrieve(task, app_context, top_k=3)

# 4. Use with prompt formatting
from openadapt_ml.experiments.demo_prompt.format_demo import format_episode_as_demo
formatted_demo = format_episode_as_demo(similar_demos[0])
```

## Features

### Text Similarity
- Uses TF-IDF with cosine similarity for v1
- No external ML libraries required
- Can be upgraded to sentence-transformers later

### Domain Matching
- Auto-extracts app name from observations
- Auto-extracts domain from URLs
- Applies bonus score for domain/app matches

### Metadata Support
- Stores arbitrary metadata with each demo
- Tracks app name, domain, and custom fields
- Efficient filtering by app/domain

## API Reference

### DemoIndex

```python
index = DemoIndex()

# Add episodes
index.add(episode, app_name="Chrome", domain="github.com")
index.add_many(episodes)

# Build index (required before retrieval)
index.build()

# Query index
index.get_apps()      # List of unique app names
index.get_domains()   # List of unique domains
len(index)            # Number of demos
index.is_fitted()     # Check if built
```

### DemoRetriever

```python
retriever = DemoRetriever(
    index,
    domain_bonus=0.2,  # Bonus score for domain match
)

# Retrieve episodes
episodes = retriever.retrieve(
    task="Description of task",
    app_context="Chrome",  # Optional
    top_k=3,
)

# Retrieve with scores (for debugging)
results = retriever.retrieve_with_scores(task, app_context, top_k=3)
for result in results:
    print(f"Score: {result.score}")
    print(f"  Text similarity: {result.text_score}")
    print(f"  Domain bonus: {result.domain_bonus}")
    print(f"  Goal: {result.demo.episode.goal}")
```

### TextEmbedder

```python
from openadapt_ml.retrieval.embeddings import TextEmbedder

embedder = TextEmbedder()

# Fit on corpus
documents = ["task 1", "task 2", "task 3"]
embedder.fit(documents)

# Embed text
vec1 = embedder.embed("new task")
vec2 = embedder.embed("another task")

# Compute similarity
similarity = embedder.cosine_similarity(vec1, vec2)
```

## Scoring

The retrieval score combines text similarity and domain matching:

```
total_score = text_similarity + domain_bonus
```

- **Text similarity**: TF-IDF cosine similarity between task descriptions (0-1)
- **Domain bonus**: Fixed bonus if app_context matches demo's app or domain (default: 0.2)

### Example Scores

```
Query: "Search GitHub for ML papers"
App context: "github.com"

Demo 1: "Search for machine learning papers on GitHub"
  - Text similarity: 0.678
  - Domain bonus: 0.200 (github.com match)
  - Total: 0.878 ⭐ Best match

Demo 2: "Create a new GitHub repository"
  - Text similarity: 0.111
  - Domain bonus: 0.200 (github.com match)
  - Total: 0.311

Demo 3: "Search for Python documentation on Google"
  - Text similarity: 0.232
  - Domain bonus: 0.000 (no match)
  - Total: 0.232
```

## Loading Real Episodes

```python
from openadapt_ml.ingest.capture import load_capture
from openadapt_ml.retrieval import DemoIndex, DemoRetriever

# Load from capture directory
capture_path = "/path/to/capture"
episodes = load_capture(capture_path)

# Build index
index = DemoIndex()
index.add_many(episodes)
index.build()

# Retrieve
retriever = DemoRetriever(index)
demos = retriever.retrieve("New task description", top_k=3)
```

## Integration with Prompting

```python
from openadapt_ml.experiments.demo_prompt.format_demo import format_episode_as_demo

# Retrieve demo
demos = retriever.retrieve(task, app_context, top_k=1)

# Format for prompt
demo_text = format_episode_as_demo(demos[0], max_steps=10)

# Inject into prompt
prompt = f"""Here is a demonstration of a similar task:

{demo_text}

Now perform this task:
Task: {task}
"""
```

## Examples

See `examples/demo_retrieval_example.py` for a complete working example.

Run it with:
```bash
uv run python examples/demo_retrieval_example.py
```

## Future Improvements

### v2: Better Embeddings
Replace TF-IDF with sentence-transformers:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### v3: Semantic Search
- Use FAISS or Qdrant for large-scale retrieval
- Add metadata filtering before similarity search
- Support multi-modal embeddings (text + screenshots)

### v4: Learning to Rank
- Train a ranking model using success/failure data
- Incorporate user feedback
- Personalized retrieval based on agent history

## Design Principles

1. **Start simple** - v1 uses no ML models, just text matching
2. **Functional over optimal** - Works out of the box, can be improved later
3. **Clear API** - Simple retrieve() interface, complex details hidden
4. **Composable** - Each component can be used independently
5. **Schema-first** - Works with Episode schema, no custom data structures
