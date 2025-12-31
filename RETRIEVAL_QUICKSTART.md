# Demo Retrieval - Quick Start

## Installation

No additional dependencies required! The retrieval module uses only Python standard library.

```bash
# Already included in openadapt-ml
from openadapt_ml.retrieval import DemoIndex, DemoRetriever
```

## 30-Second Example

```python
from openadapt_ml.retrieval import DemoIndex, DemoRetriever
from openadapt_ml.schemas.sessions import Action, Episode, Observation, Step

# 1. Create some demo episodes
def make_episode(goal, app_name):
    obs = Observation(app_name=app_name)
    action = Action(type="click", x=0.5, y=0.5)
    step = Step(t=0.0, observation=obs, action=action)
    return Episode(id=goal[:10], goal=goal, steps=[step])

episodes = [
    make_episode("Turn off Night Shift in System Settings", "System Settings"),
    make_episode("Search GitHub for machine learning papers", "Chrome"),
    make_episode("Open calculator and compute 42 * 17", "Calculator"),
]

# 2. Build index
index = DemoIndex()
index.add_many(episodes)
index.build()

# 3. Retrieve similar demos
retriever = DemoRetriever(index)
similar = retriever.retrieve("Disable Night Shift on macOS", top_k=2)

# 4. Use results
for demo in similar:
    print(f"- {demo.goal}")
```

Output:
```
- Turn off Night Shift in System Settings
- Search GitHub for machine learning papers
```

## Running the Examples

```bash
# Basic example with synthetic data
uv run python examples/demo_retrieval_example.py

# Test with real capture data
uv run python examples/retrieval_with_capture.py /path/to/capture

# Run unit tests
uv run pytest tests/test_retrieval.py -v
```

## Integration with Prompting

```python
from openadapt_ml.experiments.demo_prompt.format_demo import format_episode_as_demo

# Retrieve demo
demos = retriever.retrieve("Turn off Night Shift", top_k=1)

# Format for prompt
demo_text = format_episode_as_demo(demos[0], max_steps=10)

# Create few-shot prompt
prompt = f"""Here is a demonstration:

{demo_text}

Now perform this task:
Task: Turn off Night Shift
What is your first action?"""
```

## Loading from Captures

```python
from openadapt_ml.ingest.capture import capture_to_episode

# Load from capture directories
captures = [
    "/Users/abrichr/oa/src/openadapt-capture/turn-off-nightshift",
    "/Users/abrichr/oa/src/openadapt-capture/another-capture",
]

episodes = [capture_to_episode(path, include_moves=False) for path in captures]

# Build index and retrieve
index = DemoIndex()
index.add_many(episodes)
index.build()

retriever = DemoRetriever(index)
demos = retriever.retrieve("Your new task", top_k=3)
```

## Key Parameters

### Domain Bonus (default: 0.2)

Controls preference for matching app/domain:

```python
# High domain preference
retriever = DemoRetriever(index, domain_bonus=0.5)
demos = retriever.retrieve("Search code", app_context="github.com", top_k=3)
# Will strongly prefer GitHub demos
```

### Top-K (default: 3)

Number of demos to return:

```python
# Single best match
demos = retriever.retrieve(task, top_k=1)

# Multiple examples for few-shot
demos = retriever.retrieve(task, top_k=5)
```

## API Reference

### DemoIndex

```python
index = DemoIndex()

# Add episodes
index.add(episode, app_name="Chrome", domain="github.com")
index.add_many(episodes)

# Build (required before retrieval)
index.build()

# Query
apps = index.get_apps()       # ["Chrome", "System Settings", ...]
domains = index.get_domains() # ["github.com", "google.com", ...]
count = len(index)            # Number of demos
```

### DemoRetriever

```python
retriever = DemoRetriever(index, domain_bonus=0.2)

# Basic retrieval
demos = retriever.retrieve(
    task="Turn off Night Shift",
    app_context="System Settings",  # Optional
    top_k=3
)

# Retrieval with scores (for debugging)
results = retriever.retrieve_with_scores(task, app_context, top_k=3)
for r in results:
    print(f"Score: {r.score}, Goal: {r.demo.episode.goal}")
```

## Common Use Cases

### 1. Few-Shot Prompting

```python
# Retrieve relevant demos
demos = retriever.retrieve(new_task, top_k=3)

# Format for prompt
demo_texts = [format_episode_as_demo(d, max_steps=10) for d in demos]

prompt = f"""Here are examples of similar tasks:

{chr(10).join(demo_texts)}

Now perform: {new_task}"""
```

### 2. Runtime Demo Selection

```python
# At inference time, retrieve demo based on user's task
user_task = "Turn on dark mode"
user_app = "System Settings"

demos = retriever.retrieve(user_task, app_context=user_app, top_k=1)
demo_prompt = format_episode_as_demo(demos[0])

# Use demo_prompt in VLM request
```

### 3. Training Data Augmentation

```python
# For each training example, retrieve similar demos
for train_episode in training_set:
    similar_demos = retriever.retrieve(
        train_episode.goal,
        app_context=extract_app(train_episode),
        top_k=2
    )
    # Use similar_demos to augment training data
```

## File Locations

- **Module**: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/retrieval/`
- **Examples**: `/Users/abrichr/oa/src/openadapt-ml/examples/`
- **Tests**: `/Users/abrichr/oa/src/openadapt-ml/tests/test_retrieval.py`

## Documentation

- **README.md**: Module overview and architecture
- **USAGE.md**: Detailed usage guide with patterns
- **This file**: Quick start guide

## Next Steps

1. Try the examples: `uv run python examples/demo_retrieval_example.py`
2. Test with your captures: `uv run python examples/retrieval_with_capture.py /path/to/capture`
3. Read USAGE.md for advanced patterns
4. Integrate with your prompting pipeline

## Getting Help

- Check `openadapt_ml/retrieval/README.md` for architecture details
- See `openadapt_ml/retrieval/USAGE.md` for usage patterns
- Run tests: `uv run pytest tests/test_retrieval.py -v`
- Look at example scripts in `examples/`
