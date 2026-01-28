# WAA Speedup Options

## Summary Table

| Option | Speedup | Cost Impact | Recommended |
|--------|---------|-------------|-------------|
| `--fast` flag | ~30% install, ~40% eval | +$0.19/hr | YES for dev |
| Deallocated VM | Skip 25min install | ~$1.50/mo | YES for repeat runs |
| Parallelization | 45x (154 tasks) | -78% cost | YES for large benchmarks |

## Option 1: `--fast` Flag (Double Hardware)

Use larger VM with more CPU/RAM allocated to QEMU.

**Usage:**
```bash
# Create fast VM
uv run python -m openadapt_ml.benchmarks.cli create --fast

# Start with fast QEMU allocation
uv run python -m openadapt_ml.benchmarks.cli start --fast
```

**Specs:**

| Mode | VM Size | vCPU | RAM | QEMU Cores | QEMU RAM | Cost/hr |
|------|---------|------|-----|------------|----------|---------|
| Standard | D4ds_v4 | 4 | 16GB | 4 | 8GB | $0.19 |
| Fast | D8ds_v5 | 8 | 32GB | 6 | 16GB | $0.38 |

**Expected Speedups:**
- Windows installation: ~30% faster (25min → ~18min)
- Task evaluation: ~40% faster (navi agent ML inference benefits from more CPU)
- Total benchmark (30 tasks): ~35% faster

**When to use:**
- Development/debugging when you don't want to wait
- Time-sensitive evaluations
- Cost difference is negligible (~$0.19/hr extra)

## Option 2: Deallocated "Golden" VM

Keep a VM deallocated after WAA is fully installed. Restart when needed.

**How it works:**
1. First run: Create VM, install WAA fully (~25 min)
2. After use: `deallocate` (stops billing, keeps disk)
3. Next time: `vm-start` → boots in ~2-3 min with WAA ready

**Cost:**
- Deallocated VM: $0 compute
- Disk storage: ~$0.05/GB/month = ~$1.50/month for 30GB

**Commands:**
```bash
# After first successful run
uv run python -m openadapt_ml.benchmarks.cli deallocate

# Next time
uv run python -m openadapt_ml.benchmarks.cli vm-start
uv run python -m openadapt_ml.benchmarks.cli start  # Container starts, Windows boots in 2-3 min
```

## Option 3: Parallelization (Best for Large Benchmarks)

Run multiple VMs in parallel for large task sets.

**Speedup for 154 tasks:**

| Workers | Time | Cost | vs Single VM |
|---------|------|------|--------------|
| 1 (sequential) | ~15 hours | $2.88 | baseline |
| 5 | ~3 hours | $1.14 | 5x faster, 60% cheaper |
| 10 | ~1.5 hours | $0.63 | 10x faster, 78% cheaper |

**Implementation:** See `docs/waa_parallelization_plan.md`

## Quick Reference

```bash
# Standard mode (default)
uv run python -m openadapt_ml.benchmarks.cli create
uv run python -m openadapt_ml.benchmarks.cli build
uv run python -m openadapt_ml.benchmarks.cli start

# Fast mode (double hardware)
uv run python -m openadapt_ml.benchmarks.cli create --fast
uv run python -m openadapt_ml.benchmarks.cli build
uv run python -m openadapt_ml.benchmarks.cli start --fast

# Reuse deallocated VM
uv run python -m openadapt_ml.benchmarks.cli vm-start
uv run python -m openadapt_ml.benchmarks.cli start
```
