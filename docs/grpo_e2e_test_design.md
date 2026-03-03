# GRPO E2E Test Design

## Date: 2026-03-02

## Problem

The GRPO trainer was recently rewritten. The existing tests in `tests/test_grpo.py` are
unit tests that mock everything and only verify individual functions in isolation. We need
end-to-end tests that exercise the full training loop and produce artifacts a human can
inspect to verify correctness.

## What a human reviewer needs to see

1. **Did the training loop run without errors?** -- test report with pass/fail, duration,
   error traces.
2. **Did the model weights change?** -- LoRA parameter diff (L2 norm of delta) before vs
   after training. If weights did not change, training is broken.
3. **Were rollouts collected and rewards computed?** -- rollout traces showing the sequence
   of (screenshot, action, reward) for each rollout.
4. **Is the loss signal reasonable?** -- per-step metrics: loss, reward_mean,
   advantage stats, gradient norm.
5. **Can the checkpoint be saved and reloaded?** -- verify the saved LoRA adapter can be
   loaded back.
6. **Does the GRPO loss function actually drive policy toward high-reward actions?** --
   synthetic convergence test with controlled log-probs and rewards.

## Design Options Considered

### Option A: pytest with artifact directory
- Standard pytest tests write artifacts to `test_artifacts/grpo_e2e/`.
- Pros: CI integration, no extra dependencies, familiar.
- Cons: artifacts are just files on disk; need separate step to view.

### Option B: Standalone script
- `scripts/run_e2e_test.py` with HTML report.
- Pros: rich output, self-contained.
- Cons: does not integrate with CI.

### Option C: pytest + HTML report plugin (pytest-html)
- Best of both worlds but adds a dependency.

### Option D: pytest + artifact directory + separate summary script
- pytest writes artifacts; `scripts/grpo_e2e_report.py` reads them and prints a
  formatted summary (or generates HTML).
- Pros: separation of concerns, can re-run report without re-running tests, CI-friendly.
- Cons: two invocations.

### Chosen: Option D

Reasoning:
- The user wants to "look at" results -- a summary script can print a clean, readable
  report without adding pytest-html as a dependency.
- Tests work in CI (pytest) and locally (run report script after).
- Artifacts tell the full story: JSON metrics, PNG screenshots, rollout traces.
- Report script can be extended later to generate HTML without changing tests.

## Test Architecture

### Mock Strategy

We do NOT load a real Qwen2.5-VL model (too slow, too large). Instead:

1. **Mock model**: A tiny `nn.Module` with a single linear layer + LoRA-like trainable
   params. It accepts "input_ids" and returns logits. This lets us test that gradients
   flow and weights update without needing a 7B model.
2. **Mock processor**: Returns pre-built tensors. Has `apply_chat_template`,
   `decode`, and `__call__` methods.
3. **Mock environment**: Generates synthetic screenshots (colored rectangles with text
   via PIL), returns mock `RolloutStep` objects with realistic `BenchmarkObservation`
   and `BenchmarkAction` data. Reward is deterministic based on the action.
4. **Mock rollout collector**: Replaces `GRPORolloutCollector` -- returns pre-built
   `Rollout` objects with mock steps that contain PNG screenshot bytes.

This way:
- The training loop (optimizer, loss computation, checkpointing) is exercised for real.
- Artifacts contain visually meaningful screenshots.
- Tests run in < 60s on CPU.

### Tests

1. **`test_e2e_training_loop_mock`** -- Full loop: 2 training steps, 2 rollouts each.
   Verifies weights change, loss is computed, checkpoint is saved and loadable.

2. **`test_e2e_rollout_collection_mock`** -- Collects rollouts from mock environment,
   saves traces (JSON) and screenshots (PNG) as artifacts.

3. **`test_e2e_grpo_loss_convergence`** -- Synthetic test: creates fake log-probs
   (as trainable parameters) and rewards, runs GRPO loss + optimizer for 50 steps,
   verifies the "policy" shifts probability toward high-reward actions.

### Artifacts Written

```
test_artifacts/grpo_e2e/<timestamp>/
  test_report.json           -- overall pass/fail, timing, errors
  training_log.json          -- per-step metrics from the training loop
  rollout_traces/
    step_0_rollout_0.json    -- per-rollout trace
    step_0_rollout_0_screenshot_0.png
    ...
  model_diff.json            -- LoRA weight delta stats
  checkpoint/                -- saved LoRA adapter
  convergence/
    loss_history.json         -- loss values over 50 synthetic steps
    advantage_policy.json     -- policy probabilities over time
  summary.txt                -- human-readable summary
```

### Report Script

`scripts/grpo_e2e_report.py` reads the artifact directory and prints:
- Test status (pass/fail per test)
- Training metrics summary
- Model weight change (did LoRA params move?)
- Convergence check (did loss decrease in synthetic test?)
- File listing of all artifacts

Uses `fire` for CLI: `python scripts/grpo_e2e_report.py <artifact_dir>`
