# Training Pipeline Gaps & Plan

## Current State (Feb 23, 2026)

First successful JSONL training launch on Lambda Labs (Qwen3-VL-8B, A10).
Pipeline works end-to-end: `convert_demos --bundle` -> `train.py --jsonl` -> Lambda launch.

### What Works
- Bundle creation (JSONL + images with relative paths)
- JSONL loader in TRL trainer
- Lambda Labs instance launch, code sync, bundle upload
- Checkpoint save (TRL SFTTrainer auto-saves per epoch + final)
- Dashboard with Lambda cloud badge, cost display, setup progress
- Checkpoint download via rsync

### What's Missing

## Gap 1: TRL Training Callback (logging + early stopping)

**Problem**: `train_from_jsonl()` uses TRL's SFTTrainer which logs to stdout/TensorBoard
but NOT to our `training_log.json`. The Lambda polling loop in `lambda_labs.py` reads
`training_log.json` for dashboard updates — so the dashboard shows 0 progress.

Also, `early_stop_loss` is in YAML config but not wired to a TRL callback.

**Fix**: Add a `TrainerCallback` subclass to `trl_trainer.py`:
- `on_log()`: Write step/epoch/loss/lr to `training_log.json`
- `on_log()`: Check `early_stop_loss` threshold, set `trainer.state.should_training_stop = True`
- ~40 lines

**Files**: `openadapt_ml/training/trl_trainer.py`

## Gap 2: Cost Persistence

**Problem**: Cost is only computed in dashboard JavaScript. Not saved to `training_log.json`.
After instance terminates, cost is lost.

**Fix**: Write `cost_per_hour`, `total_cost`, `instance_type` to `training_log.json` in the
TRL callback (same one from Gap 1). The callback can read instance_type from an env var
or config field.

**Files**: `openadapt_ml/training/trl_trainer.py` (extend callback)

## Gap 3: Post-Training Eval Automation

**Problem**: After training completes on Lambda, there's no automatic evaluation. User must
manually run `eval_policy.py` or `compare.py`.

**Fix**: Add `--eval` flag to `train.py` that runs eval after training completes:
1. Load the saved checkpoint
2. Run prediction on each training sample (sanity check: can it reproduce training data?)
3. Write results to `eval_results.json` alongside checkpoint
4. ~30 lines in `train.py`, reusing `eval_policy.py` logic

For Lambda: add eval step to `run_training()` command string when `--eval` is passed.

**Files**: `openadapt_ml/scripts/train.py`, `openadapt_ml/cloud/lambda_labs.py`

## Gap 4: Training Speed (Unsloth on Lambda)

**Problem**: 8B model without Unsloth = 569s/step on A10. With Unsloth should be ~100-200s/step.
Current Lambda setup installs deps with `uv sync` which doesn't include Unsloth (it's optional
and needs special install: `pip install unsloth`).

**Fix**: Add Unsloth install to Lambda setup script. Remove `--no-unsloth` from default
Lambda train command. Test Unsloth + Qwen3-VL-8B on A10.

Alternative: Use 2B model (`Qwen/Qwen3-VL-2B-Instruct`) which is 4x faster without Unsloth.

**Files**: `openadapt_ml/cloud/lambda_labs.py` (setup_instance), `configs/qwen3vl_demo.yaml`

## Gap 5: Auto-Terminate After Training

**Problem**: When training is launched manually (SSH nohup), the Lambda instance keeps running
after training completes. No auto-terminate.

**Fix**: The automated `train` command already handles this when it's used end-to-end.
The issue is only when we SSH in manually. Not a code gap — operational.

## Priority Order

1. **Gap 4** (speed) — blocking: current run takes 8 hours, wastes money
2. **Gap 1** (callback) — high: dashboard shows nothing during training
3. **Gap 3** (eval) — medium: need eval to know if training worked
4. **Gap 2** (cost) — low: nice-to-have for tracking spend
5. **Gap 5** (terminate) — low: operational

## Immediate Action

Kill current slow run, switch to 2B model, relaunch. Fix Gaps 1+4 first.
