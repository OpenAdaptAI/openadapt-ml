# Benchmark Viewer Integration Design

## Overview

This document describes how to integrate benchmark evaluation results (WAA, WebArena, OSWorld) into the unified viewer, enabling side-by-side comparison of model performance across benchmark tasks.

## Current State

### Viewer Capabilities
- Step-by-step screenshot playback
- Human vs model action comparison per step
- Checkpoint switching (None, Epoch 1, Epoch 2, etc.)
- Metrics: accuracy percentage, step count

### Benchmark Module (`openadapt_ml/benchmarks/`)
- `BenchmarkAdapter` interface for different benchmarks
- `WAAAdapter` for Windows Agent Arena
- `AzureWAAOrchestrator` for parallel VM execution
- Produces per-task success/failure results

## Design Goals

1. **Unified Experience**: Same viewer UI for captures and benchmark tasks
2. **Model Comparison**: Compare multiple models on identical tasks
3. **Drill-Down**: From aggregate metrics → task list → step-by-step replay
4. **Actionable Insights**: Identify failure patterns, common errors

## Proposed Architecture

### Data Model

```
benchmark_results/
├── waa_eval_20241214/
│   ├── metadata.json          # Benchmark config, models evaluated
│   ├── tasks/
│   │   ├── task_001/
│   │   │   ├── task.json      # Task definition, success criteria
│   │   │   ├── screenshots/   # Execution screenshots
│   │   │   ├── model_a.json   # Model A's execution trace
│   │   │   └── model_b.json   # Model B's execution trace
│   │   └── task_002/
│   │       └── ...
│   └── summary.json           # Aggregate results
```

### Schema Extensions

```python
@dataclass
class BenchmarkTask:
    task_id: str
    name: str
    domain: str  # e.g., "browser", "file_manager", "settings"
    description: str
    success_criteria: str
    max_steps: int

@dataclass
class BenchmarkExecution:
    task_id: str
    model_id: str  # e.g., "qwen3vl-2b-epoch5", "gpt-5.1"
    success: bool
    steps_taken: int
    execution_time: float
    error_message: Optional[str]
    trace: List[ExecutionStep]  # Screenshots + actions

@dataclass
class ExecutionStep:
    step_idx: int
    screenshot_path: str
    action: Action  # From existing schema
    reasoning: Optional[str]
    timestamp: float
```

### Viewer Integration

#### 1. Benchmark Dashboard Tab
Add a third tab alongside "Training" and "Viewer":

```
[Training] [Viewer] [Benchmarks]
```

The Benchmarks tab shows:
- Dropdown: Select benchmark run (by date/name)
- Summary metrics: Overall success rate, by-domain breakdown
- Task list with pass/fail status
- Model comparison table

#### 2. Task Drill-Down View
Clicking a task opens step-by-step view:
- Same UI as current Viewer tab
- Model selector: Switch between model executions
- Side-by-side mode: Compare two models simultaneously
- Failure analysis: Highlight where execution diverged

#### 3. Model Comparison Mode
New comparison layout:

```
┌─────────────────┬─────────────────┐
│   Model A       │    Model B      │
│  [Screenshot]   │  [Screenshot]   │
│  Action: CLICK  │  Action: TYPE   │
│  Step 3 of 12   │  Step 3 of 8    │
│  ✓ Succeeded    │  ✗ Failed       │
└─────────────────┴─────────────────┘
```

### Implementation Plan

#### Phase 1: Data Collection
1. Extend `BenchmarkRunner` to save execution traces
2. Save screenshots at each step during benchmark runs
3. Store structured results in `benchmark_results/` directory

#### Phase 2: Viewer Backend
1. Add `load_benchmark_results()` function to trainer.py
2. Generate `benchmark.html` from results
3. Reuse existing viewer components where possible

#### Phase 3: UI Components
1. Benchmark summary dashboard (success rates, charts)
2. Task list with filtering (by domain, status, model)
3. Step-by-step replay with model comparison
4. Export capabilities (CSV, JSON)

#### Phase 4: Analysis Features
1. Failure clustering: Group similar failures
2. Step-level accuracy: Where do models commonly fail?
3. Difficulty estimation: Rank tasks by model success rate
4. Regression detection: Compare across training runs

## Integration with Existing Code

### Viewer Generation
The consolidated `_generate_unified_viewer_from_extracted_data()` function can be extended:

```python
def generate_benchmark_viewer(
    benchmark_dir: Path,
    output_path: Path,
) -> None:
    """Generate viewer for benchmark results."""
    # Load benchmark metadata
    metadata = load_benchmark_metadata(benchmark_dir)

    # Load all task results
    tasks = load_task_results(benchmark_dir)

    # Generate HTML using same template patterns
    html = _generate_benchmark_viewer_html(metadata, tasks)
    output_path.write_text(html)
```

### Shared Components
Reuse from existing viewer:
- Header/nav component (`_get_shared_header_css()`, `_generate_shared_header_html()`)
- Screenshot display with click markers
- Action comparison boxes
- Playback controls

### CLI Integration
```bash
# Run benchmark and generate viewer
uv run python -m openadapt_ml.benchmarks.cli run-azure --tasks 10 --viewer

# Generate viewer from existing results
uv run python -m openadapt_ml.benchmarks.cli viewer benchmark_results/waa_eval_20241214/

# Serve benchmark viewer
uv run python -m openadapt_ml.cloud.local serve --benchmark benchmark_results/waa_eval_20241214/
```

## Open Questions

1. **Screenshot Storage**: Benchmark runs may produce thousands of screenshots. Cloud storage (S3/Azure Blob) or local with lazy loading?

2. **Real-time Updates**: Should viewer update live during benchmark runs, or only after completion?

3. **Comparison Granularity**: Compare at task level, step level, or both?

4. **Historical Tracking**: How to track model improvement across multiple benchmark runs?

## Dependencies

- Existing viewer consolidation (DONE)
- WAA benchmark adapter (implemented)
- Azure orchestration (implemented)
- Screenshot capture during benchmark runs (needs work)

## Success Metrics

- Users can view benchmark results in browser
- Side-by-side model comparison works
- Drill-down from summary to step-level works
- Performance: Handles 100+ tasks without lag
