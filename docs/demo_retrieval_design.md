# Demo Retrieval System Design (Option B)

**Goal**: Given a new task, automatically select the most relevant demo from a library.

**Target timeline**: Working prototype in 1-2 days.

**Status**: Design document. Existing implementation in `openadapt_ml/retrieval/` provides TF-IDF baseline; this document extends it for production use.

---

## 1. Demo Index Schema

### 1.1 Demo Metadata (per demo)

Each demo in the index stores:

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `demo_id` | string | Unique identifier | Auto-generated or filename |
| `goal` | string | Natural language task description | Episode.goal |
| `app_name` | string | Primary application (e.g., "Chrome", "System Settings") | Extracted from Observation.app_name |
| `domain` | string | Domain/website (e.g., "github.com") | Extracted from Observation.url |
| `platform` | string | Operating system ("macos", "windows", "web") | Inferred from observations |
| `action_types` | list[string] | Types of actions used (["click", "type", "scroll"]) | Aggregated from Action.type |
| `key_elements` | list[string] | Important UI elements touched (["Button:OK", "TextField:Search"]) | From Observation.accessibility_tree or target_name |
| `step_count` | int | Number of steps in the demo | len(Episode.steps) |
| `tags` | list[string] | User-provided tags (["settings", "display", "toggle"]) | Manual annotation |
| `created_at` | string | ISO timestamp | File creation time |
| `file_path` | string | Relative path to demo JSON | Directory structure |

### 1.2 File Format: JSONL Manifest

**Recommendation**: Use JSONL (JSON Lines) over Parquet for simplicity.

**Rationale**:
- Human-readable, easy to debug
- Append-only (can add demos without rewriting)
- No additional dependencies (Parquet requires pyarrow/fastparquet)
- Git-friendly diffs
- Fast enough for hundreds of demos (Parquet benefits only at 10K+ scale)

**index.jsonl format**:
```jsonl
{"demo_id": "navigate_settings_001", "goal": "Turn off Night Shift", "app_name": "System Settings", "domain": null, "platform": "macos", "action_types": ["click"], "key_elements": ["Button:Night Shift...", "PopUpButton:Schedule"], "step_count": 5, "tags": ["settings", "display"], "created_at": "2025-01-02T10:30:00Z", "file_path": "macos/navigate_settings_001.json"}
{"demo_id": "github_search_001", "goal": "Search for machine learning repos on GitHub", "app_name": "Chrome", "domain": "github.com", "platform": "web", "action_types": ["click", "type"], "key_elements": ["TextField:Search", "Link:Repository"], "step_count": 3, "tags": ["browser", "search", "github"], "created_at": "2025-01-02T11:00:00Z", "file_path": "browser/github_search_001.json"}
```

### 1.3 Demo Episode JSON

Each demo file contains the full Episode schema:

```json
{
  "id": "navigate_settings_001",
  "goal": "Turn off Night Shift",
  "steps": [
    {
      "t": 0.0,
      "observation": {
        "app_name": "Finder",
        "window_title": "Desktop",
        "image_path": "screenshots/step_000.png"
      },
      "action": {
        "type": "click",
        "x": 0.01,
        "y": 0.01,
        "target_name": "Apple menu"
      }
    }
  ],
  "metadata": {
    "source": "openadapt-capture",
    "capture_date": "2025-01-02T10:30:00Z"
  }
}
```

---

## 2. Retrieval Approaches

### 2.1 BM25 on Goal Text + App Context (Baseline)

**Complexity**: Low (no ML models)
**Dependencies**: None (pure Python)
**Quality**: Good for exact keyword matches, weak on semantic similarity

**Implementation**:
```python
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, demos: list[DemoMetadata]):
        # Tokenize goal + app_name + domain
        self.corpus = [
            f"{d.goal} {d.app_name or ''} {d.domain or ''}".lower().split()
            for d in demos
        ]
        self.bm25 = BM25Okapi(self.corpus)
        self.demos = demos

    def retrieve(self, query: str, top_k: int = 3) -> list[DemoMetadata]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self.demos[i] for i in top_indices]
```

**Pros**:
- Zero dependencies (rank_bm25 is optional, can use existing TF-IDF)
- Fast indexing and retrieval
- No GPU required
- Interpretable scores

**Cons**:
- Misses semantic similarity ("Turn off Night Shift" vs "Disable blue light filter")
- No understanding of action/UI similarity
- Pure lexical matching

### 2.2 Embedding Similarity (sentence-transformers)

**Complexity**: Medium
**Dependencies**: `sentence-transformers`, `torch`
**Quality**: Good semantic understanding, captures synonyms and paraphrases

**Implementation**:
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingRetriever:
    def __init__(self, demos: list[DemoMetadata], model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.demos = demos

        # Pre-compute embeddings
        texts = [f"{d.goal} [APP:{d.app_name}] [DOMAIN:{d.domain}]" for d in demos]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, app_context: str = None, top_k: int = 3) -> list[DemoMetadata]:
        query_text = query
        if app_context:
            query_text += f" [APP:{app_context}]"

        query_embedding = self.model.encode([query_text], normalize_embeddings=True)[0]

        # Cosine similarity (since normalized, just dot product)
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.demos[i] for i in top_indices]
```

**Model choices** (from fastest to best):
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` | 22MB | Fast | Good |
| `all-mpnet-base-v2` | 420MB | Medium | Better |
| `BAAI/bge-small-en-v1.5` | 130MB | Fast | Better |
| `BAAI/bge-base-en-v1.5` | 440MB | Medium | Best |

**Recommendation**: Start with `all-MiniLM-L6-v2` (22MB, CPU-friendly).

### 2.3 Hybrid (BM25 + Embeddings)

**Complexity**: Medium
**Dependencies**: Both BM25 and sentence-transformers
**Quality**: Best of both worlds - exact matches + semantic similarity

**Implementation**:
```python
class HybridRetriever:
    def __init__(self, demos: list[DemoMetadata], alpha: float = 0.5):
        self.bm25 = BM25Retriever(demos)
        self.embedding = EmbeddingRetriever(demos)
        self.alpha = alpha  # Weight for BM25 vs embeddings
        self.demos = demos

    def retrieve(self, query: str, app_context: str = None, top_k: int = 3) -> list[DemoMetadata]:
        # Get BM25 scores
        bm25_scores = self._normalize(self.bm25.get_scores(query))

        # Get embedding scores
        embed_scores = self._normalize(self.embedding.get_scores(query, app_context))

        # Combine: hybrid_score = alpha * bm25 + (1-alpha) * embed
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * embed_scores

        # Apply domain bonus
        if app_context:
            for i, demo in enumerate(self.demos):
                if demo.app_name and app_context.lower() in demo.app_name.lower():
                    hybrid_scores[i] += 0.2
                if demo.domain and app_context.lower() in demo.domain.lower():
                    hybrid_scores[i] += 0.2

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [self.demos[i] for i in top_indices]
```

**Tuning**:
- `alpha = 0.5`: Equal weight (default)
- `alpha = 0.7`: Favor exact matches (when demos have precise goal text)
- `alpha = 0.3`: Favor semantics (when tasks are described differently)

---

## 3. Demo Library Structure

```
demos/
  index.jsonl                    # Manifest with all demo metadata

  macos/                         # Platform: macOS
    settings/                    # App category: System Settings
      night_shift_off.json
      night_shift_on.json
      true_tone_toggle.json
    finder/
      create_folder.json
      delete_file.json

  windows/                       # Platform: Windows
    settings/
      dark_mode_toggle.json
    explorer/
      rename_file.json

  browser/                       # Platform: Web (cross-platform)
    github/
      search_repos.json
      create_issue.json
      fork_repo.json
    google/
      search_query.json
      navigate_gmail.json
    generic/
      fill_form.json
      download_file.json

  office/                        # Cross-platform apps
    vscode/
      open_file.json
      run_command.json
    slack/
      send_message.json
      create_channel.json
```

### Directory Conventions

1. **Top-level by platform**: `macos/`, `windows/`, `browser/`
2. **Second-level by app/domain**: `settings/`, `github/`, `vscode/`
3. **Demo files**: `{action}_{target}.json` (e.g., `toggle_night_shift.json`)

### index.jsonl Maintenance

The manifest is **append-only** during normal operation:
```bash
# Add new demo
python -m openadapt_ml.retrieval.cli add demos/macos/settings/new_demo.json

# Rebuild entire index (after bulk changes)
python -m openadapt_ml.retrieval.cli rebuild demos/
```

---

## 4. API Design

### 4.1 Core Classes

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openadapt_ml.benchmarks.base import BenchmarkTask
from openadapt_ml.schemas.sessions import Episode


@dataclass
class Demo:
    """A demonstration with its metadata."""
    demo_id: str
    goal: str
    episode: Episode
    app_name: Optional[str] = None
    domain: Optional[str] = None
    platform: Optional[str] = None
    action_types: list[str] = None
    tags: list[str] = None
    score: float = 0.0  # Retrieval score (set at query time)


class DemoRetriever:
    """Retrieves relevant demos from a library given a task."""

    def __init__(
        self,
        demo_dir: Path,
        method: str = "bm25",  # "bm25", "embedding", "hybrid"
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize retriever with demo library.

        Args:
            demo_dir: Path to demos/ directory containing index.jsonl
            method: Retrieval method ("bm25", "embedding", "hybrid")
            embedding_model: Sentence-transformers model name (if using embeddings)
        """
        self.demo_dir = demo_dir
        self.method = method
        self.embedding_model = embedding_model
        self._index: list[DemoMetadata] = []
        self._retriever = None  # Lazy initialization

    def index_demos(self) -> None:
        """Build or rebuild the retrieval index.

        Reads all demo files, extracts metadata, and builds index.jsonl.
        """
        # Scan demo_dir for *.json files
        # Extract metadata from each Episode
        # Write to index.jsonl
        # Build retrieval structures (BM25/embeddings)
        pass

    def load_index(self) -> None:
        """Load existing index from index.jsonl."""
        index_path = self.demo_dir / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"No index.jsonl found at {index_path}. Run index_demos() first.")

        # Load metadata from JSONL
        # Initialize retriever
        pass

    def retrieve(
        self,
        task: BenchmarkTask,
        top_k: int = 1,
    ) -> list[Demo]:
        """Retrieve top-K demos for a benchmark task.

        Args:
            task: The benchmark task to find demos for.
            top_k: Number of demos to retrieve.

        Returns:
            List of Demo objects, ordered by relevance (best first).
        """
        if self._retriever is None:
            self.load_index()

        # Extract query from task
        query = task.instruction
        app_context = self._extract_app_context(task)

        # Run retrieval
        results = self._retriever.retrieve(query, app_context, top_k)

        # Load full Episode for each result
        demos = []
        for meta, score in results:
            episode = self._load_episode(meta.file_path)
            demos.append(Demo(
                demo_id=meta.demo_id,
                goal=meta.goal,
                episode=episode,
                app_name=meta.app_name,
                domain=meta.domain,
                platform=meta.platform,
                action_types=meta.action_types,
                tags=meta.tags,
                score=score,
            ))

        return demos

    def retrieve_from_text(
        self,
        query: str,
        app_context: Optional[str] = None,
        top_k: int = 1,
    ) -> list[Demo]:
        """Retrieve demos from a text query (not a BenchmarkTask).

        Convenience method for interactive use.
        """
        pass

    def _extract_app_context(self, task: BenchmarkTask) -> Optional[str]:
        """Extract app/domain context from task."""
        # Check raw_config for app hints
        # Check domain field
        # Parse instruction for app names
        pass

    def _load_episode(self, file_path: str) -> Episode:
        """Load Episode from JSON file."""
        pass
```

### 4.2 CLI Interface

```bash
# Index all demos in a directory
uv run python -m openadapt_ml.retrieval.cli index demos/

# Retrieve demos for a query
uv run python -m openadapt_ml.retrieval.cli retrieve \
    --demo-dir demos/ \
    --query "Turn off Night Shift" \
    --app-context "System Settings" \
    --top-k 3

# Add a new demo to the library
uv run python -m openadapt_ml.retrieval.cli add \
    --demo-dir demos/ \
    --episode /path/to/capture/episode.json \
    --tags "settings,display"

# Validate index integrity
uv run python -m openadapt_ml.retrieval.cli validate demos/
```

### 4.3 Integration with Benchmark Runner

```python
from openadapt_ml.benchmarks import BenchmarkAgent, evaluate_agent_on_benchmark
from openadapt_ml.retrieval import DemoRetriever
from openadapt_ml.experiments.demo_prompt.format_demo import format_episode_as_demo


class DemoConditionedAgent(BenchmarkAgent):
    """Agent that retrieves and uses demos in prompts."""

    def __init__(
        self,
        base_agent: BenchmarkAgent,
        retriever: DemoRetriever,
        max_demo_steps: int = 10,
    ):
        self.base_agent = base_agent
        self.retriever = retriever
        self.max_demo_steps = max_demo_steps
        self._current_demo: Optional[Demo] = None

    def reset(self, task: BenchmarkTask) -> None:
        """Reset for new task, retrieve relevant demo."""
        self.base_agent.reset(task)

        # Retrieve best demo for this task
        demos = self.retriever.retrieve(task, top_k=1)
        self._current_demo = demos[0] if demos else None

    def act(self, observation: BenchmarkObservation) -> BenchmarkAction:
        """Generate action with demo context."""
        # Format demo for prompt
        demo_text = ""
        if self._current_demo:
            demo_text = format_episode_as_demo(
                self._current_demo.episode,
                max_steps=self.max_demo_steps,
            )

        # Inject demo into base agent's prompt
        # (Implementation depends on base agent type)
        return self.base_agent.act(observation, demo_context=demo_text)
```

---

## 5. Evaluation

### 5.1 Retrieval Quality Metrics

#### Relevance: Does the retrieved demo help?

| Metric | Definition | Target |
|--------|------------|--------|
| **Hit@1** | % of tasks where best demo is from same app/domain | > 80% |
| **Hit@3** | % of tasks where at least one of top-3 demos is relevant | > 95% |
| **MRR** | Mean Reciprocal Rank of first relevant demo | > 0.7 |

**Relevance criteria** (in order of strength):
1. **Strong**: Same goal verb + same app (e.g., "toggle Night Shift" -> "toggle True Tone")
2. **Medium**: Same app, different action (e.g., "toggle Night Shift" -> "open Display settings")
3. **Weak**: Same platform, different app (e.g., "toggle Night Shift" -> "toggle Bluetooth")
4. **None**: Different platform or completely unrelated

#### Coverage: What % of tasks have a useful demo?

| Metric | Definition | Target |
|--------|------------|--------|
| **Library coverage** | % of benchmark tasks with at least one strong/medium match | > 50% |
| **Gap analysis** | List of task categories with no matching demos | Track |

### 5.2 End-to-End Impact

| Metric | Definition | Comparison |
|--------|------------|------------|
| **Action accuracy delta** | (Demo-conditioned accuracy) - (Zero-shot accuracy) | +10pp minimum |
| **Task success delta** | (Demo-conditioned success) - (Zero-shot success) | +5pp minimum |

### 5.3 Evaluation Protocol

```python
def evaluate_retrieval(
    retriever: DemoRetriever,
    tasks: list[BenchmarkTask],
    ground_truth: dict[str, list[str]],  # task_id -> list of relevant demo_ids
) -> dict[str, float]:
    """Evaluate retrieval quality.

    Args:
        retriever: The retriever to evaluate.
        tasks: List of benchmark tasks.
        ground_truth: Manual labels of relevant demos per task.

    Returns:
        Dictionary of metrics.
    """
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []

    for task in tasks:
        demos = retriever.retrieve(task, top_k=3)
        retrieved_ids = [d.demo_id for d in demos]
        relevant_ids = ground_truth.get(task.task_id, [])

        # Hit@1
        if retrieved_ids and retrieved_ids[0] in relevant_ids:
            hits_at_1 += 1

        # Hit@3
        if any(rid in relevant_ids for rid in retrieved_ids):
            hits_at_3 += 1

        # MRR
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    n = len(tasks)
    return {
        "hit_at_1": hits_at_1 / n if n > 0 else 0.0,
        "hit_at_3": hits_at_3 / n if n > 0 else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n > 0 else 0.0,
        "coverage": len([t for t in tasks if ground_truth.get(t.task_id)]) / n if n > 0 else 0.0,
    }
```

---

## 6. Implementation Plan

### Phase 1: BM25 Baseline (Day 1)

**Goal**: Get something working end-to-end.

| Task | Time | Output |
|------|------|--------|
| Define DemoMetadata schema | 30min | `openadapt_ml/retrieval/schema.py` |
| Create demo library structure | 30min | `demos/` directory with 5-10 demos |
| Implement JSONL index read/write | 1hr | `openadapt_ml/retrieval/index.py` (extend existing) |
| Implement BM25 retriever | 1hr | `openadapt_ml/retrieval/bm25.py` |
| Add CLI for index/retrieve | 1hr | `openadapt_ml/retrieval/cli.py` |
| Integration test | 30min | `tests/test_retrieval_e2e.py` |

**Deliverable**: Can index demos and retrieve by goal text.

### Phase 2: Embedding Retriever (Day 1-2)

**Goal**: Better semantic matching.

| Task | Time | Output |
|------|------|--------|
| Add sentence-transformers dependency | 15min | `pyproject.toml` |
| Implement EmbeddingRetriever | 1hr | `openadapt_ml/retrieval/embedding.py` |
| Implement HybridRetriever | 30min | `openadapt_ml/retrieval/hybrid.py` |
| Update CLI with method flag | 30min | `cli.py` |
| Comparative eval (BM25 vs Hybrid) | 1hr | `scripts/eval_retrieval.py` |

**Deliverable**: Can choose between BM25/embedding/hybrid methods.

### Phase 3: Benchmark Integration (Day 2)

**Goal**: Use retrieval in demo-conditioned prompting.

| Task | Time | Output |
|------|------|--------|
| Create DemoConditionedAgent wrapper | 1hr | `openadapt_ml/benchmarks/demo_agent.py` |
| Run on WAA mock tasks | 1hr | Baseline metrics |
| Compare zero-shot vs demo-conditioned | 1hr | Results in `docs/experiments/` |

**Deliverable**: Retrieval integrated into benchmark pipeline.

### Phase 4: Evaluation & Iteration (Ongoing)

| Task | Time | Output |
|------|------|--------|
| Create ground truth labels for 20 tasks | 2hr | `demos/eval/ground_truth.json` |
| Compute retrieval metrics | 30min | Hit@K, MRR numbers |
| Identify gaps (tasks with no good demos) | 1hr | Gap analysis report |
| Iterate on scoring (alpha, bonus weights) | Ongoing | Tuned parameters |

---

## 7. Decision Log

| Decision | Rationale | Alternative considered |
|----------|-----------|----------------------|
| JSONL over Parquet | Simplicity, human-readable, no deps | Parquet (better compression, typing) |
| BM25 first | No ML deps, fast iteration | Start with embeddings |
| all-MiniLM-L6-v2 | Small (22MB), CPU-friendly, good quality | BGE-small (better but larger) |
| Domain bonus additive | Simple, interpretable | Multiplicative (harder to tune) |
| Episode storage in separate files | Git-friendly, lazy loading | All in index.jsonl (simpler but bloated) |

---

## 8. Open Questions

1. **How to handle multi-step demos?** Should we index sub-trajectories (steps 1-3 as one demo, steps 2-5 as another)?

2. **Cross-platform demos**: Can a macOS demo help with Windows task if UI is similar (e.g., browser)?

3. **Demo freshness**: How to deprecate outdated demos (when UI changes)?

4. **Negative examples**: Should we explicitly store "demos that don't help" to avoid them?

---

## 9. Appendix: Existing Implementation

The current `openadapt_ml/retrieval/` module provides a good foundation:

- **`embeddings.py`**: TF-IDF embeddings (can be replaced with sentence-transformers)
- **`index.py`**: DemoIndex with app/domain extraction (extend with JSONL persistence)
- **`retriever.py`**: DemoRetriever with domain bonus scoring (keep, add BM25 option)

**What to keep**:
- DemoMetadata dataclass (extend with new fields)
- Domain/app extraction logic
- Cosine similarity scoring

**What to add**:
- JSONL index persistence
- BM25 retrieval option
- Sentence-transformers embeddings
- CLI interface
- Demo library structure

---

## 10. References

- [Demo-Conditioned Prompting Results](experiments/demo_conditioned_prompting_results.md) - Validates demo-conditioning improves accuracy by 53pp
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25) - Probabilistic text retrieval
- [Sentence-Transformers](https://www.sbert.net/) - Pre-trained embedding models
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) - Python BM25 implementation
