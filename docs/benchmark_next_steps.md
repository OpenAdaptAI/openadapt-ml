# Benchmark Integration - Next Steps

## Current State

The benchmark viewer infrastructure is complete (Phases 1-2), but displays an empty state because:
- All existing benchmark data is mock/test data (RandomAgent on WAAMockAdapter)
- Mock data is automatically filtered out to show only real evaluations
- No real benchmark evaluations have been run yet

## Priority Alignment (from roadmap.md)

The roadmap identifies these as **critical blocking priorities**:

| Priority | Description | Status |
|----------|-------------|--------|
| 0 | Fix Episode Success Rate | Parsing fixed, but 0% success |
| 0.1 | Validate prompts on known benchmark | NOT DONE |
| 0.2 | **Establish upper bound with larger models** | NOT DONE |
| 0.3 | Achieve >0% episode success | BLOCKING |

**Key insight**: "Without task completion, all other metrics are noise"

## Recommended Next Steps

### 1. Run Claude/GPT-5.1 API Baseline (Priority 0.2)

This establishes the "upper bound" - what frontier VLMs can achieve on our benchmarks.

```bash
# Claude baseline (requires ANTHROPIC_API_KEY in .env)
uv run python -m openadapt_ml.benchmarks.cli run-api \
  --provider anthropic --tasks 10

# GPT-5.1 baseline (requires OPENAI_API_KEY in .env)
uv run python -m openadapt_ml.benchmarks.cli run-api \
  --provider openai --tasks 10
```

**Why this matters:**
- If Claude/GPT-5.1 also fail, the problem is in prompts/action format
- If they succeed, smaller models need more training data or better architecture
- Provides comparison point for fine-tuned model improvement

### 2. Convert Capture to Benchmark Format (Human Baseline)

The `turn-off-nightshift` capture represents a human demonstration. Converting it to benchmark format provides a "human baseline" for comparison.

```bash
# This would need implementation:
uv run python -m openadapt_ml.benchmarks.cli convert-capture \
  --capture ~/oa/src/openadapt-capture/turn-off-nightshift \
  --output benchmark_results/human-nightshift
```

**Why this matters:**
- Shows what successful task completion looks like
- Enables human vs model comparison in the viewer
- Validates the benchmark viewer can display real execution traces

### 3. Run Real WAA Evaluation (on Windows/Azure)

Once prompts are validated:

```bash
# On Windows with WAA setup:
uv run python -m openadapt_ml.benchmarks.cli run-local \
  --waa-path /path/to/WindowsAgentArena

# On Azure (parallel VMs):
uv run python -m openadapt_ml.benchmarks.cli run-azure --workers 4
```

## Parallel Agent Tasks

These tasks can run simultaneously as background agents:

### Agent 1: API Baseline Evaluation
- Run `run-api` with Claude on mock tasks
- Measure success rate, action accuracy
- Compare to RandomAgent baseline

### Agent 2: Capture Conversion Script
- Create `openadapt_ml/benchmarks/capture_converter.py`
- Convert capture.db + screenshots to benchmark format
- Test with `turn-off-nightshift` capture

### Agent 3: Prompt Engineering
- Review prompts in `APIBenchmarkAgent.SYSTEM_PROMPT`
- Compare to TTI repo prompts (`scripts/prompts/create_prompt_json.py`)
- Test variations on synthetic benchmark

## Files Changed

- `openadapt_ml/cloud/local.py` - Added `_is_mock_benchmark()`, updated `_regenerate_benchmark_viewer_if_available()`
- `openadapt_ml/training/benchmark_viewer.py` - Added `generate_empty_benchmark_viewer()`
- `openadapt_ml/benchmarks/agent.py` - Added `APIBenchmarkAgent` class
- `openadapt_ml/benchmarks/__init__.py` - Exported `APIBenchmarkAgent`
- `openadapt_ml/benchmarks/cli.py` - Added `run-api` command

## Success Metrics

- [ ] Claude/GPT-5.1 API baselines show >0% task completion
- [ ] Human baseline visible in benchmark viewer
- [ ] At least one fine-tuned model evaluated on same benchmark
- [ ] Clear comparison between baselines and fine-tuned models
