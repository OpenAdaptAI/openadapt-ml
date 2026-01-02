# Demo Retrieval - Usage Guide

## Quick Reference

```python
# 1. Build index
from openadapt_ml.retrieval import DemoIndex, DemoRetriever
index = DemoIndex()
index.add_many(episodes)
index.build()

# 2. Retrieve
retriever = DemoRetriever(index)
similar_demos = retriever.retrieve("Turn off Night Shift", top_k=3)
```

## Complete Examples

### Example 1: Basic Usage with Synthetic Data

```python
from openadapt_ml.retrieval import DemoIndex, DemoRetriever
from openadapt_ml.schema import Action, ActionType, Episode, Observation, Step

# Create test episodes
def create_episode(instruction, app_name=None):
    obs = Observation(app_name=app_name)
    action = Action(type=ActionType.CLICK, normalized_coordinates=(0.5, 0.5))
    step = Step(step_index=0, observation=obs, action=action)
    return Episode(episode_id=f"ep_{instruction[:10]}", instruction=instruction, steps=[step])

episodes = [
    create_episode("Turn off Night Shift", app_name="System Settings"),
    create_episode("Search GitHub", app_name="Chrome"),
    create_episode("Open calculator", app_name="Calculator"),
]

# Build index
index = DemoIndex()
index.add_many(episodes)
index.build()

# Retrieve
retriever = DemoRetriever(index, domain_bonus=0.2)
results = retriever.retrieve("Disable Night Shift", top_k=2)

print(f"Found {len(results)} similar demos:")
for ep in results:
    print(f"- {ep.goal}")
```

### Example 2: Loading from Capture

```python
from openadapt_ml.ingest.capture import capture_to_episode
from openadapt_ml.retrieval import DemoIndex, DemoRetriever

# Load multiple captures
capture_paths = [
    "/path/to/capture1",
    "/path/to/capture2",
    "/path/to/capture3",
]

episodes = [
    capture_to_episode(path, include_moves=False)
    for path in capture_paths
]

# Build index
index = DemoIndex()
index.add_many(episodes)
index.build()

# Retrieve for new task
retriever = DemoRetriever(index)
task = "Turn on dark mode"
app = "System Settings"
demos = retriever.retrieve(task, app_context=app, top_k=3)
```

### Example 3: Integration with Prompting

```python
from openadapt_ml.experiments.demo_prompt.format_demo import format_episode_as_demo
from openadapt_ml.retrieval import DemoIndex, DemoRetriever

# Build index (assume episodes already loaded)
index = DemoIndex()
index.add_many(episodes)
index.build()

# Retrieve for new task
retriever = DemoRetriever(index)
task = "Turn off Night Shift"
demos = retriever.retrieve(task, top_k=1)

# Format for prompt
if demos:
    demo_text = format_episode_as_demo(demos[0], max_steps=10)

    # Create few-shot prompt
    prompt = f"""You are a GUI automation agent.

DEMONSTRATION OF SIMILAR TASK:
{demo_text}

NEW TASK:
{task}

What is your first action?"""

    print(prompt)
```

### Example 4: Retrieval with Scores (Debugging)

```python
from openadapt_ml.retrieval import DemoRetriever

retriever = DemoRetriever(index, domain_bonus=0.3)

# Retrieve with scores for analysis
results = retriever.retrieve_with_scores(
    task="Search for Python docs",
    app_context="github.com",
    top_k=5,
)

# Analyze scores
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result.demo.episode.goal}")
    print(f"   Total score: {result.score:.3f}")
    print(f"   Text similarity: {result.text_score:.3f}")
    print(f"   Domain bonus: {result.domain_bonus:.3f}")

    if result.demo.app_name:
        print(f"   App: {result.demo.app_name}")
    if result.demo.domain:
        print(f"   Domain: {result.demo.domain}")
```

### Example 5: Custom Metadata

```python
from openadapt_ml.retrieval import DemoIndex

index = DemoIndex()

# Add episodes with custom metadata
for episode in episodes:
    metadata = {
        "difficulty": "easy",
        "success_rate": 0.95,
        "duration_seconds": 30,
        "tags": ["settings", "macOS"],
    }

    index.add(
        episode,
        app_name="System Settings",
        domain=None,
        metadata=metadata,
    )

index.build()

# Access metadata after retrieval
retriever = DemoRetriever(index)
results = retriever.retrieve_with_scores("Turn off Night Shift", top_k=1)

if results:
    demo = results[0].demo
    print(f"Difficulty: {demo.metadata.get('difficulty')}")
    print(f"Tags: {demo.metadata.get('tags')}")
```

## CLI Examples

Run the provided example scripts:

```bash
# Basic demo with synthetic data
uv run python examples/demo_retrieval_example.py

# Test with real capture
uv run python examples/retrieval_with_capture.py /path/to/capture

# With custom task
uv run python examples/retrieval_with_capture.py /path/to/capture "Turn off dark mode"
```

## Common Patterns

### Pattern 1: Multi-Domain Index

```python
# Build index with episodes from multiple domains
web_episodes = load_web_captures()
desktop_episodes = load_desktop_captures()

index = DemoIndex()
index.add_many(web_episodes)
index.add_many(desktop_episodes)
index.build()

# Retrieve with domain filtering via app_context
retriever = DemoRetriever(index, domain_bonus=0.5)

# This will prefer github.com demos
web_demos = retriever.retrieve("Search code", app_context="github.com", top_k=3)

# This will prefer System Settings demos
desktop_demos = retriever.retrieve("Change settings", app_context="System Settings", top_k=3)
```

### Pattern 2: Incremental Index Updates

```python
# Build initial index
index = DemoIndex()
index.add_many(initial_episodes)
index.build()

# Add new episodes
index.add(new_episode)

# Rebuild required after adding
index.build()

# Now retriever will use updated index
retriever = DemoRetriever(index)
```

### Pattern 3: Batch Retrieval

```python
# Retrieve for multiple tasks
tasks = [
    "Turn off Night Shift",
    "Enable dark mode",
    "Adjust brightness",
]

retriever = DemoRetriever(index)

for task in tasks:
    demos = retriever.retrieve(task, top_k=3)
    print(f"\nTask: {task}")
    for demo in demos:
        print(f"  - {demo.goal}")
```

## Tuning Parameters

### Domain Bonus

Controls how much to favor domain/app matches:

```python
# No domain bonus - pure text similarity
retriever = DemoRetriever(index, domain_bonus=0.0)

# Small bonus (default)
retriever = DemoRetriever(index, domain_bonus=0.2)

# Large bonus - heavily favor same domain
retriever = DemoRetriever(index, domain_bonus=0.5)
```

**Rule of thumb:**
- `0.0-0.1`: When task text is very specific and domain doesn't matter much
- `0.2-0.3`: Good default for most cases
- `0.4-0.5`: When domain matching is critical (e.g., domain-specific workflows)

### Top-K

Number of demos to retrieve:

```python
# Single best match
demos = retriever.retrieve(task, top_k=1)

# Few-shot with 3 examples
demos = retriever.retrieve(task, top_k=3)

# Retrieve more for analysis/selection
demos = retriever.retrieve(task, top_k=10)
```

**Rule of thumb:**
- `top_k=1`: When prompt length is constrained
- `top_k=3`: Good default for few-shot learning
- `top_k=5+`: For ensemble methods or human selection

## Performance Tips

### 1. Build Once, Retrieve Many

```python
# Good: Build once
index.build()
retriever = DemoRetriever(index)
for task in many_tasks:
    retriever.retrieve(task)

# Bad: Build repeatedly
for task in many_tasks:
    index.build()  # Wasteful!
    retriever = DemoRetriever(index)
    retriever.retrieve(task)
```

### 2. Pre-extract Metadata

```python
# Good: Extract once when adding
index.add(episode, app_name="Chrome", domain="github.com")

# Less efficient: Let auto-extraction scan every episode
index.add(episode)  # Will scan steps for app_name and domain
```

### 3. Filter Before Retrieval

```python
# If you have a large index but know the domain, create a filtered index
web_demos = [d for d in index.get_all_demos() if d.domain]
web_index = DemoIndex()
for demo in web_demos:
    web_index.add(demo.episode)
web_index.build()
```

## Troubleshooting

### Issue: All scores are 0.0

**Cause:** Only one episode in index, so IDF is undefined.

**Solution:** Add more episodes or use a larger demo library.

```python
# Need at least 2-3 episodes for meaningful scores
assert len(index) >= 3, "Add more demos to the index"
```

### Issue: Domain bonus not applied

**Cause:** app_context doesn't match app_name or domain.

**Debug:**
```python
results = retriever.retrieve_with_scores(task, app_context, top_k=5)
for r in results:
    print(f"App: {r.demo.app_name}, Domain: {r.demo.domain}, Bonus: {r.domain_bonus}")
```

**Solution:** Check exact string matching (case-insensitive contains).

### Issue: Poor retrieval quality

**Causes:**
1. Task descriptions too generic
2. Demo library too small
3. TF-IDF limitations

**Solutions:**
1. Use more specific task descriptions
2. Add more diverse demos to index
3. Upgrade to sentence-transformers (see README.md § Future Improvements)

## Testing

Run unit tests:
```bash
uv run pytest tests/test_retrieval.py -v
```

Run integration tests:
```bash
uv run python test_retrieval.py
```

## Next Steps

1. **Integrate with training**: Use retrieval in data augmentation
2. **Experiment with prompting**: Test different demo counts and formats
3. **Upgrade embeddings**: Try sentence-transformers for better similarity
4. **Add filtering**: Support domain/app filtering before similarity scoring
5. **Evaluate impact**: Measure action accuracy with/without retrieval
