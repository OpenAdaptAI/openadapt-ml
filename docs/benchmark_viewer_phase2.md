# Benchmark Viewer Phase 2 - Implementation Complete

This document describes the completed Phase 2 implementation of the Benchmark Viewer integration.

## Overview

Phase 2 implements the viewer backend that generates interactive HTML dashboards from benchmark evaluation results. This allows you to:

- View benchmark results in a web browser with a unified interface
- See summary metrics (success rate, average steps, average time)
- Filter tasks by status (all/success/fail) and domain
- Expand tasks to see step-by-step execution details with screenshots
- Navigate between Training, Viewer, and Benchmarks tabs seamlessly

## Implementation

### Files Created/Modified

**New Files:**
- `openadapt_ml/training/benchmark_viewer.py` - Benchmark viewer generation functions
- `test_benchmark_viewer.py` - Test script for benchmark viewer

**Modified Files:**
- `openadapt_ml/training/trainer.py` - Added "Benchmarks" tab to shared header
- `openadapt_ml/cloud/local.py` - Added `benchmark-viewer` command and `--benchmark` flag to `serve`

### Key Components

#### 1. Benchmark Viewer Generation (`benchmark_viewer.py`)

The main function `generate_benchmark_viewer()` loads benchmark results and generates an interactive HTML dashboard:

```python
from openadapt_ml.training.benchmark_viewer import generate_benchmark_viewer

# Generate viewer from benchmark results
viewer_path = generate_benchmark_viewer("benchmark_results/test_run_phase1")
```

**Features:**
- Loads metadata, summary, and task results from the benchmark directory
- Generates a complete standalone HTML file with embedded data
- Reuses shared header components for consistent UI across all viewers
- Includes filters for status and domain
- Shows step-by-step execution with screenshots

#### 2. Shared Header Update (`trainer.py`)

Updated `_generate_shared_header_html()` to include a third "Benchmarks" tab:

```python
def _generate_shared_header_html(
    active_page: str,  # "training", "viewer", or "benchmarks"
    controls_html: str = "",
    meta_html: str = "",
) -> str:
    # Generates header with Training | Viewer | Benchmarks tabs
```

All three viewers (Training Dashboard, Capture Viewer, Benchmark Viewer) now have the same header with navigation tabs.

#### 3. CLI Commands (`local.py`)

**New `benchmark-viewer` command:**
```bash
# Generate benchmark viewer from a benchmark results directory
python -m openadapt_ml.cloud.local benchmark-viewer benchmark_results/test_run_phase1 --open
```

**Updated `serve` command:**
```bash
# Serve benchmark results (auto-regenerates benchmark.html)
python -m openadapt_ml.cloud.local serve --benchmark benchmark_results/test_run_phase1 --open

# Serve training output (default behavior)
python -m openadapt_ml.cloud.local serve --open
```

The `serve` command now supports a `--benchmark` flag to serve benchmark results instead of training output.

## Usage

### Step 1: Generate Benchmark Data (Phase 1)

First, run a benchmark evaluation that collects execution traces:

```bash
# Test with mock data (no actual Windows environment needed)
python -m openadapt_ml.benchmarks.cli test-collection --tasks 5 --run-name my_benchmark

# This creates:
# benchmark_results/my_benchmark/
#   ├── metadata.json
#   ├── summary.json
#   └── tasks/
#       ├── task_001/
#       │   ├── task.json
#       │   ├── execution.json
#       │   └── screenshots/
#       └── ...
```

### Step 2: Generate Benchmark Viewer

```bash
# Generate the viewer HTML
python -m openadapt_ml.cloud.local benchmark-viewer benchmark_results/my_benchmark --open

# Or use the test script
python test_benchmark_viewer.py
```

This creates `benchmark_results/my_benchmark/benchmark.html`.

### Step 3: View Results

```bash
# Option 1: Serve with auto-regeneration
python -m openadapt_ml.cloud.local serve --benchmark benchmark_results/my_benchmark --open

# Option 2: Open the HTML file directly
open benchmark_results/my_benchmark/benchmark.html
```

## Viewer Features

### Summary Dashboard

The viewer displays 4 key metrics at the top:
- **Total Tasks**: Number of benchmark tasks evaluated
- **Success Rate**: Percentage of tasks that passed (with count: X / Y passed)
- **Avg Steps**: Average number of steps taken per task
- **Avg Time**: Average execution time per task in seconds

### Task List

Tasks are displayed in a filterable list showing:
- Status indicator (✓ for success, ✗ for failure)
- Task ID and instruction
- Domain badge (browser, office, etc.)
- Number of steps and execution time

**Filters:**
- **Status**: All Tasks | Success Only | Failure Only
- **Domain**: All Domains | Browser | Office | etc.

### Task Details (Expandable)

Click on any task to expand and view step-by-step details:
- Screenshot at each step (if available)
- Action type (CLICK, TYPE, SCROLL, etc.)
- Action details (coordinates, text, keys, etc.)
- Reasoning (if recorded by the agent)

## Design Decisions

### 1. Separate Module for Benchmark Viewer

We created `benchmark_viewer.py` as a separate module instead of adding functions directly to `trainer.py` because:
- Keeps trainer.py focused on training-related functionality
- Makes benchmark viewer code easier to find and maintain
- Allows for future expansion without cluttering trainer.py

### 2. Reuse Shared Header Components

The benchmark viewer imports `_get_shared_header_css()` and `_generate_shared_header_html()` from trainer.py to ensure:
- Visual consistency across all viewers
- Single source of truth for header styling
- Automatic updates when header components change

### 3. Embedded Data vs. Separate JSON

We embed all benchmark data directly in the HTML as JavaScript constants because:
- Creates a single standalone file (easier to share/archive)
- No CORS issues when opening locally
- Simpler deployment (just copy one HTML file)
- Still allows for large datasets via JavaScript array loading

### 4. Serve Command Enhancement

The `serve` command now supports both training output and benchmark results:
- `--benchmark` flag switches to benchmark serving mode
- Auto-regenerates the appropriate viewer before serving
- Uses the same server infrastructure for consistency

## Dark Theme Styling

The benchmark viewer uses the same dark theme as the training dashboard and capture viewer:

```css
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-tertiary: #1a1a24;
    --border-color: rgba(255, 255, 255, 0.06);
    --text-primary: #f0f0f0;
    --text-secondary: #888;
    --text-muted: #555;
    --accent: #00d4aa;
    --success: #00d4aa;
    --failure: #ff4444;
}
```

## Testing

Run the test script to verify the implementation:

```bash
python test_benchmark_viewer.py
```

This will:
1. Check if test data exists (from Phase 1)
2. Generate the benchmark viewer
3. Display success message with instructions to view

## Next Steps (Future Phases)

### Phase 3: Advanced UI Components
- Model comparison mode (side-by-side)
- Success/failure pattern analysis
- Domain-specific metrics
- Historical trend charts

### Phase 4: Analysis Features
- Failure clustering and categorization
- Step-level accuracy analysis
- Difficulty estimation per task
- Regression detection across runs

## Examples

### Example 1: Generate and Serve

```bash
# Generate benchmark data
python -m openadapt_ml.benchmarks.cli test-collection --tasks 10 --run-name waa_test

# Generate and serve the viewer
python -m openadapt_ml.cloud.local serve --benchmark benchmark_results/waa_test --open
```

### Example 2: Regenerate Viewer

```bash
# If you've updated the viewer code, regenerate without re-running the benchmark:
python -m openadapt_ml.cloud.local benchmark-viewer benchmark_results/waa_test
```

### Example 3: Serve Without Regeneration

```bash
# If the benchmark.html is already up-to-date, skip regeneration for faster startup:
python -m openadapt_ml.cloud.local serve --benchmark benchmark_results/waa_test --no-regenerate --open
```

## Troubleshooting

### "Benchmark directory not found"

Ensure you've run Phase 1 data collection first:
```bash
python -m openadapt_ml.benchmarks.cli test-collection --tasks 5
```

### "metadata.json not found"

The benchmark directory is missing required files. Re-run the benchmark evaluation to regenerate them.

### "No tasks match the current filters"

Try changing the filter settings:
- Set Status to "All Tasks"
- Set Domain to "All Domains"

### Screenshots not displaying

Check that:
1. Screenshots were collected during benchmark run (Phase 1)
2. Screenshot paths in `execution.json` are correct
3. You're serving the benchmark directory (not just opening the HTML file)

## File Structure

After generating the benchmark viewer, your directory structure will look like:

```
benchmark_results/
└── my_benchmark/
    ├── metadata.json          # Benchmark metadata
    ├── summary.json           # Aggregate metrics
    ├── benchmark.html         # ← Generated viewer (new!)
    └── tasks/
        ├── browser_1/
        │   ├── task.json
        │   ├── execution.json
        │   └── screenshots/
        │       ├── step_000.png
        │       ├── step_001.png
        │       └── ...
        └── ...
```

## API Reference

### `generate_benchmark_viewer(benchmark_dir, output_path=None)`

Generate benchmark viewer HTML from benchmark results directory.

**Parameters:**
- `benchmark_dir` (str | Path): Path to benchmark results directory
- `output_path` (str | Path | None): Optional output path (default: benchmark_dir/benchmark.html)

**Returns:**
- `Path`: Path to generated benchmark.html file

**Raises:**
- `FileNotFoundError`: If benchmark directory or metadata.json not found

**Example:**
```python
from openadapt_ml.training.benchmark_viewer import generate_benchmark_viewer

viewer_path = generate_benchmark_viewer(
    benchmark_dir="benchmark_results/waa_eval_20241214",
    output_path="custom_output/benchmark.html"
)
print(f"Generated: {viewer_path}")
```

## Summary

Phase 2 is complete! The benchmark viewer:
- ✅ Generates HTML dashboards from benchmark results
- ✅ Reuses shared header for unified UI
- ✅ Supports filtering by status and domain
- ✅ Shows step-by-step execution with screenshots
- ✅ Integrates with existing CLI commands
- ✅ Uses consistent dark theme styling
- ✅ Works with Phase 1 data collection output

The viewer is ready for testing and can be extended with Phase 3/4 features in the future.
