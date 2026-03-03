# GRPO Architecture Analysis: Custom vs TRL-Based Approach

## The Problem: 26 Issues From One Root Cause

After a comprehensive review of our custom GRPO trainer (~809 lines), we identified
26 issues (7 critical, 8 important, 7 medium, 4 low). The sheer count is a code smell
pointing to an architectural problem rather than implementation bugs.

**Root cause**: We wrote a custom GRPO trainer that reimplements what TRL now provides
natively, while also tightly coupling RL math with WAA-specific glue code.

## Breakdown of Our 809-Line Trainer

| Category | Lines | What It Does |
|----------|-------|-------------|
| GRPO Math | ~190 | Advantage computation, KL penalty, policy gradient loss, reference policy |
| Infrastructure/Glue | ~180 | Model loading, LoRA setup, optimizer, checkpointing, training loop |
| Unique to Our Use Case | ~400+ | Multi-turn rollout processing, DSL parsing, prompt formatting, observation handling |

## What TRL v0.29.0 Now Provides

TRL's GRPOTrainer (as of Feb 2026) supports:

1. **Multi-turn rollouts** via `rollout_func` (v0.29.0) — you provide a custom function
   that replaces TRL's generation loop. Returns `prompt_ids`, `completion_ids`, `logprobs`.
   A Wordle example shows 6-turn interactive loops.

2. **`environment_factory`** (v0.29.0) — stateful environments with `reset()` and
   arbitrary methods as tools. One instance per rollout.

3. **Multimodal VLMs** including Qwen2.5-VL — natively supported since v0.20.0.

4. **Custom reward functions** — pass a callable, supports async, multiple functions,
   environment access, extra rollout fields forwarded as kwargs.

5. **LoRA + quantization** — standard PEFT integration.

6. **Gradient accumulation** — standard HF Trainer mechanisms + `steps_per_generation`.

7. **Advanced loss variants** — `dapo`, `dr_grpo`, `bnpo`, asymmetric clipping, Liger kernel fusion.

## Which Issues Vanish With TRL?

~14 of 26 issues are eliminated by delegating to TRL:

- **CR-03** (custom GRPO duplicates TRL): Eliminated by definition
- **CR-07** (untested training loop): TRL is battle-tested
- **IM-03** (no error handling in rollouts): TRL handles generation errors
- **IM-05** (prompt misalignment risk): TRL manages tokenization
- **IM-06** (monkey-patch Unsloth loading): Use TRL's standard model loading
- **IM-07** (LoRA param capture fragile): TRL handles reference policy
- **MD-01** (no gradient clipping): TRL includes it
- **MD-02** (no LR scheduler): TRL includes standard schedulers
- **MD-03** (no WandB logging): TRL integrates with all HF loggers
- **MD-04** (hardcoded AdamW): TRL supports all optimizers
- **MD-05** (no multi-GPU): TRL + accelerate/DeepSpeed handles this
- **MD-06** (no mixed precision): TRL handles bf16/fp16
- **LO-01** (verbose step logging): TRL's logging is configurable
- **LO-02** (no TensorBoard): TRL integrates natively

## The Key Gap: Multi-Turn Interactive Rollouts

TRL is fundamentally **single-turn**: prompt -> completion -> reward. Even with
`rollout_func`, the advantage is computed at the trajectory level (one reward per
complete rollout), not per-step.

But this actually **matches our use case**:
- WebAgent-R1 uses binary task-success rewards (0 or 1)
- GRPO computes group-relative advantages across N trajectories of the same task
- We don't need per-step credit assignment — trajectory-level reward is sufficient

The `rollout_func` approach lets us:
1. Call our `RLEnvironment.collect_rollout()` to get interactive multi-step trajectories
2. Return the concatenated token IDs and log-probs to TRL
3. Let TRL handle advantage computation, clipping, KL penalty, optimization

## Proposed Architecture

```
TRL GRPOTrainer              <- standard, maintained, tested (0 lines from us)
  |
  +-- rollout_func:           <- ~100 lines (our custom rollout function)
  |     Uses RLEnvironment to collect interactive multi-step trajectories
  |     Returns prompt_ids, completion_ids, logprobs
  |
  +-- reward_func:            <- ~20 lines (already exists in reward.py)
  |     binary_task_success() + compute_group_advantages()
  |
  +-- RolloutCollector        <- ~150 lines (already exists)
  |     collect_group() orchestrates N rollouts per task
  |
  +-- RLEnvironment           <- openadapt-evals (already exists, PR #73)
        reset() / step() / observe() / evaluate()
```

**Our code shrinks from ~800 lines to ~200 lines** of genuine domain-specific logic:
- `rollout_func`: Bridges TRL's generation loop with our interactive environment
- Action DSL parsing (CLICK/TYPE/WAIT/DONE)
- Prompt construction for multi-turn VLM interaction
- Reward function (already exists)

## What About WebAgent-R1 and Agent-R1?

Both build on **veRL** (ByteDance's RL framework), NOT TRL. They implement their own
multi-turn GRPO from scratch. Key results:
- WebAgent-R1: Qwen-2.5-3B went 6.1% -> 33.9% on WebArena-Lite
- Agent-R1: Supports PPO, GRPO, REINFORCE++ with per-tool-call process rewards

We could also consider veRL, but TRL has better ecosystem integration (HF Hub, PEFT,
quantization, vLLM) and the `rollout_func` API is flexible enough for our needs.

## Standalone GRPO Math (Fallback Option)

If TRL's `rollout_func` proves too constraining, the GRPO math is ~30 lines of PyTorch:

```python
# Advantage (group-normalized)
mean_r = rewards.reshape(-1, G).mean(dim=1, keepdim=True)
std_r = rewards.reshape(-1, G).std(dim=1, keepdim=True)
advantages = (rewards - mean_r.repeat(1, G).flatten()) / (std_r.repeat(1, G).flatten() + 1e-4)

# KL penalty (Schulman 2020 approximation)
x = ref_logps - current_logps
kl = torch.exp(x) - x - 1

# Clipped surrogate loss
ratio = torch.exp(current_logps - old_logps)
clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
loss = -torch.min(ratio * advantages, clipped * advantages) + beta * kl
```

This gives us full control while still eliminating the infrastructure/glue code
by using HF Trainer for the training loop.

## Recommendation

1. **Merge PR #73** (openadapt-evals RL environment) — stable foundation, CI passing
2. **Don't merge PR #34 as-is** — the custom trainer has too many issues
3. **Rewrite GRPO module** as thin TRL adapter using `rollout_func`:
   - Keep: rollout_collector.py, reward.py, config.py, cot_warmup.py
   - Replace: trainer.py (800 lines -> ~200 lines)
   - Delete: All custom GRPO math, model loading, optimizer, checkpointing
4. **Close ~14 GitHub issues** that become N/A with TRL delegation

## TRL Version Compatibility Note

TRL v0.29.0 `rollout_func` requires `transformers>=5.2.0`. Verify this works with
Unsloth and our quantization setup before committing to this path.

## References

- [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [TRL OpenEnv integration](https://huggingface.co/docs/trl/main/en/openenv)
- [TRL v0.29.0 release](https://github.com/huggingface/trl/releases/tag/v0.29.0)
- [WebAgent-R1 paper](https://arxiv.org/abs/2505.16421)
- [Agent-R1 (veRL-based)](https://github.com/0russwest0/Agent-R1)
- GitHub issues: openadapt-ml #35-#50, #42 (tracking)
- GitHub issues: openadapt-evals #76-#78
