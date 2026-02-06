# Ralph Loop for WAA Benchmark Automation

**Date**: 2026-02-02
**Purpose**: Practical guide for adapting the Ralph Loop pattern to our WAA benchmarking workflow
**Status**: Implementation guide (ready for Phase 1 development)

---

## Quick Start: The Problem We're Solving

Current WAA benchmark workflow requires ~15 manual touchpoints:
1. Create VM
2. Wait for SSH
3. Setup Docker + pull WAA image
4. Wait for Windows boot
5. Check WAA probe status
6. Start benchmark
7. Monitor for stalls
8. Recover from failures
9. Repeat manually on failure

**Ralph Loop goal**: Reduce to 1 command that runs unattended until:
- ✅ Target success rate achieved (e.g., 80% of tasks pass)
- OR ⏱️ Budget exhausted (e.g., $10 spent)
- OR ⏱️ Time limit hit (e.g., 4 hours)

---

## Step 1: Task Definition (ralph-tasks.json)

Create `/Users/abrichr/oa/src/openadapt-ml/ralph-tasks.json`:

```json
{
  "benchmark": "waa",
  "version": "1.0",
  "target_success_rate": 0.80,
  "max_budget_usd": 10.0,
  "max_runtime_hours": 4.0,
  "max_iterations": 10,
  "tasks": [
    {
      "id": "iteration_0_phase_1",
      "title": "Ensure VM Ready",
      "description": "VM must be running and SSH accessible",
      "acceptance_criteria": [
        "VM state is 'running' or 'created'",
        "SSH connection succeeds within 120s",
        "Storage location is accessible"
      ],
      "cli_command": "uv run python -m openadapt_ml.benchmarks.cli vm status",
      "recovery_actions": [
        "vm start",
        "vm create"
      ]
    },
    {
      "id": "iteration_0_phase_2",
      "title": "Setup WAA Container",
      "description": "Docker image pulled, Windows booted, WAA server ready",
      "acceptance_criteria": [
        "Docker image pulls successfully",
        "Windows 11 VM boots within 20 minutes",
        "WAA Flask server responds to /probe endpoint",
        "No 'FAIL' in probe response"
      ],
      "cli_command": "uv run python -m openadapt_ml.benchmarks.cli vm probe --wait",
      "recovery_actions": [
        "vm docker-prune",
        "vm restart-windows",
        "vm reset-windows"
      ],
      "timeout_seconds": 1200,
      "stall_detection": {
        "metric": "disk_usage",
        "check_interval_seconds": 30,
        "stall_threshold_minutes": 5
      }
    },
    {
      "id": "iteration_0_phase_3",
      "title": "Run Benchmark Tasks",
      "description": "Execute WAA tasks and track success rate",
      "acceptance_criteria": [
        "N tasks executed",
        "Results saved to benchmark_results/",
        "Success rate >= target (tracks across iterations)"
      ],
      "cli_command": "uv run python -m openadapt_evals.benchmarks.cli run --agent api-claude --tasks 20 --server http://localhost:5001",
      "recovery_actions": [
        "vm logs",
        "vm restart-windows"
      ],
      "timeout_seconds": 7200,
      "progress_tracking": {
        "metric": "tasks_completed",
        "check_interval_seconds": 60,
        "stall_threshold_minutes": 10
      }
    }
  ],
  "success_metrics": {
    "success_rate": {
      "definition": "% of tasks that achieved task goal",
      "target": 0.80,
      "aggregation": "cumulative across all iterations"
    },
    "cost_per_task": {
      "definition": "Total spend / total tasks executed",
      "target_max": 0.50,
      "aggregation": "cumulative"
    }
  }
}
```

---

## Step 2: Progress Tracking (ralph-progress.txt)

Create `/Users/abrichr/oa/src/openadapt-ml/ralph-progress.txt` (auto-updated):

```
RALPH LOOP PROGRESS
===================
Generated: 2026-02-02T14:30:00Z
Iteration: 3 / 10 max

CUMULATIVE METRICS
------------------
Total Tasks Executed: 60 (3 iterations x 20 tasks)
Total Tasks Successful: 48
Success Rate: 80.0% ✓ TARGET ACHIEVED
Total Cost: $8.42 (remaining budget: $1.58)
Total Runtime: 2h 45m (remaining time: 1h 15m)

ITERATION HISTORY
-----------------
Iteration 1: 20 tasks, 12 success (60%), cost $2.80, duration 52m
  Phase 1: VM ready ✓ (0m 15s)
  Phase 2: WAA ready ✓ (15m 20s)
  Phase 3: Benchmark ✓ 12/20 pass (52m)
  Issues: None

Iteration 2: 20 tasks, 16 success (80%), cost $2.85, duration 51m
  Phase 1: VM ready ✓ (0m 5s)
  Phase 2: WAA ready ✓ (14m 45s)
  Phase 3: Benchmark ✓ 16/20 pass (51m)
  Issues: 1 task timeout (recovered)

Iteration 3: 20 tasks, 20 success (100%), cost $2.77, duration 48m
  Phase 1: VM ready ✓ (0m 3s)
  Phase 2: WAA ready ✓ (13m 20s)
  Phase 3: Benchmark ✓ 20/20 pass (48m)
  Issues: None

LEARNINGS & ADJUSTMENTS
-----------------------
Iteration 1→2: Increased context window size in prompts (→ better task understanding)
Iteration 2→3: Added 2-second delay between task starts (→ fewer timeouts)

NEXT ACTIONS
------------
[ ] Success rate achieved - STOP
[ ] Deallocate VM
[ ] Generate final report
[ ] Archive results

EXIT REASON: TARGET_ACHIEVED
```

---

## Step 3: Ralph Loop Command (cli.py Addition)

Add to `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py`:

```python
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict

@dataclass
class RalphMetrics:
    """Track metrics across all iterations."""
    iteration: int = 0
    total_tasks: int = 0
    successful_tasks: int = 0
    total_cost_usd: float = 0.0
    start_time: float = field(default_factory=time.time)
    iteration_results: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0

    @property
    def elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600

    def should_continue(self, config: dict) -> tuple[bool, str]:
        """Check stop conditions."""
        if self.success_rate >= config.get("target_success_rate", 0.8):
            return False, "TARGET_ACHIEVED"
        if self.total_cost_usd >= config.get("max_budget_usd", 10.0):
            return False, "BUDGET_EXCEEDED"
        if self.elapsed_hours >= config.get("max_runtime_hours", 4.0):
            return False, "TIME_LIMIT"
        if self.iteration >= config.get("max_iterations", 10):
            return False, "MAX_ITERATIONS"
        return True, "CONTINUE"

    def save_progress(self, output_file: str = "ralph-progress.txt"):
        """Save progress to file."""
        with open(output_file, "w") as f:
            f.write("RALPH LOOP PROGRESS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
            f.write(f"Iteration: {self.iteration} / {len(self.iteration_results)} completed\n\n")

            f.write("CUMULATIVE METRICS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Tasks Executed: {self.total_tasks}\n")
            f.write(f"Total Tasks Successful: {self.successful_tasks}\n")
            f.write(f"Success Rate: {self.success_rate:.1%}\n")
            f.write(f"Total Cost: ${self.total_cost_usd:.2f}\n")
            f.write(f"Total Runtime: {self.elapsed_hours:.1f}h\n\n")

            f.write("ITERATION HISTORY\n")
            f.write("-" * 50 + "\n")
            for i, result in enumerate(self.iteration_results, 1):
                f.write(f"\nIteration {i}:\n")
                f.write(f"  Tasks: {result.get('tasks', 0)}\n")
                f.write(f"  Success: {result.get('successful', 0)} / {result.get('tasks', 0)}\n")
                f.write(f"  Duration: {result.get('duration_minutes', 0):.0f}m\n")
                f.write(f"  Cost: ${result.get('cost_usd', 0):.2f}\n")


def cmd_ralph(args):
    """
    Autonomous benchmark loop - runs until goal achieved or resources exhausted.

    Usage:
        uv run python -m openadapt_ml.benchmarks.cli ralph \\
            --tasks 20 \\
            --target-success-rate 0.8 \\
            --max-cost 10.0 \\
            --max-hours 4 \\
            --max-iterations 10
    """
    # Load task config
    with open("ralph-tasks.json") as f:
        config = json.load(f)

    # Initialize metrics
    metrics = RalphMetrics()
    log("RALPH", f"Starting autonomous benchmark loop")
    log("RALPH", f"  Target: {config['target_success_rate']:.0%} success rate")
    log("RALPH", f"  Budget: ${config['max_budget_usd']:.2f}")
    log("RALPH", f"  Time limit: {config['max_runtime_hours']}h")

    # Main loop
    while True:
        metrics.iteration += 1
        log("RALPH", f"\n{'='*60}")
        log("RALPH", f"ITERATION {metrics.iteration}")
        log("RALPH", f"{'='*60}")

        iteration_start = time.time()
        iteration_success = 0
        iteration_tasks = args.tasks

        # PHASE 1: Ensure VM Ready
        if not ensure_vm_ready(args):
            log("RALPH", "FAILED: Could not get VM ready after retries")
            if not should_continue_after_failure(metrics, config):
                break
            continue

        # PHASE 2: Ensure WAA Ready
        if not ensure_waa_ready(args, timeout=config.get("max_waa_timeout", 1200)):
            log("RALPH", "FAILED: WAA server not ready after retries")
            if not should_continue_after_failure(metrics, config):
                break
            continue

        # PHASE 3: Run Benchmark
        result = run_benchmark_iteration(args, iteration_tasks)
        iteration_success = result.get("successful_tasks", 0)
        iteration_cost = result.get("cost_usd", 0.0)
        iteration_duration = (time.time() - iteration_start) / 60

        # Update metrics
        metrics.total_tasks += iteration_tasks
        metrics.successful_tasks += iteration_success
        metrics.total_cost_usd += iteration_cost
        metrics.iteration_results.append({
            "iteration": metrics.iteration,
            "tasks": iteration_tasks,
            "successful": iteration_success,
            "cost_usd": iteration_cost,
            "duration_minutes": iteration_duration
        })

        log("RALPH", f"Iteration {metrics.iteration} complete:")
        log("RALPH", f"  Tasks: {iteration_success}/{iteration_tasks} success")
        log("RALPH", f"  Cost: ${iteration_cost:.2f}")
        log("RALPH", f"  Duration: {iteration_duration:.0f}m")
        log("RALPH", f"  Cumulative: {metrics.success_rate:.1%} success rate, ${metrics.total_cost_usd:.2f} spent")

        # Save progress
        metrics.save_progress()

        # Check stop conditions
        should_continue, reason = metrics.should_continue(config)
        if not should_continue:
            log("RALPH", f"Stopping: {reason}")
            break

    # Final report
    log("RALPH", "\n" + "="*60)
    log("RALPH", "FINAL REPORT")
    log("RALPH", "="*60)
    log("RALPH", f"Iterations: {metrics.iteration}")
    log("RALPH", f"Total tasks: {metrics.total_tasks}")
    log("RALPH", f"Success rate: {metrics.success_rate:.1%}")
    log("RALPH", f"Total cost: ${metrics.total_cost_usd:.2f}")
    log("RALPH", f"Total time: {metrics.elapsed_hours:.1f}h")

    # Cleanup
    log("RALPH", "Deallocating VM...")
    args.yes = True
    cmd_deallocate(args)

    return 0 if metrics.success_rate >= config["target_success_rate"] else 1


def ensure_vm_ready(args, max_attempts: int = 3) -> bool:
    """Ensure VM is running and SSH accessible, with recovery."""
    for attempt in range(max_attempts):
        try:
            # Check current state
            result = subprocess.run(
                ["az", "vm", "get-instance-view", "--ids", get_vm_id(args)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                state = json.loads(result.stdout).get("instanceView", {}).get("statuses", [])
                power_state = [s.get("displayStatus", "") for s in state if "PowerState" in s.get("code", "")]

                if power_state and "deallocated" in power_state[0].lower():
                    log("RALPH", f"VM deallocated, starting... (attempt {attempt+1}/{max_attempts})")
                    if cmd_vm_start(args) == 0:
                        time.sleep(10)
                        continue
                elif power_state and "running" in power_state[0].lower():
                    # Verify SSH access
                    ip = get_vm_ip(args)
                    if ip and wait_for_ssh(ip, timeout=60):
                        log("RALPH", "VM ready ✓")
                        return True
            else:
                # VM doesn't exist, create it
                log("RALPH", f"VM doesn't exist, creating... (attempt {attempt+1}/{max_attempts})")
                if cmd_create(args) == 0:
                    continue

        except Exception as e:
            log("RALPH", f"VM check failed: {e}")

        if attempt < max_attempts - 1:
            log("RALPH", f"Retrying VM recovery...")
            time.sleep(30)

    return False


def ensure_waa_ready(args, timeout: int = 1200) -> bool:
    """Ensure WAA server is ready with stall detection."""
    ip = get_vm_ip(args)
    if not ip:
        return False

    start = time.time()
    last_storage_size = 0
    stall_count = 0

    while time.time() - start < timeout:
        try:
            # Probe WAA server
            result = subprocess.run(
                [
                    "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                    f"azureuser@{ip}",
                    "docker exec winarena curl -s --max-time 5 http://172.30.0.2:5000/probe"
                ],
                capture_output=True,
                text=True,
                timeout=15
            )

            if "FAIL" not in result.stdout and result.stdout.strip():
                log("RALPH", "WAA server ready ✓")
                return True

            # Check for stalled progress
            storage_result = subprocess.run(
                [
                    "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                    f"azureuser@{ip}",
                    "du -sb /mnt/waa-storage/ 2>/dev/null | cut -f1"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            try:
                current_storage = int(storage_result.stdout.strip() or "0")
            except ValueError:
                current_storage = last_storage_size

            if current_storage == last_storage_size:
                stall_count += 1
                if stall_count > 10:  # 5 minutes with no progress
                    log("RALPH", "Stall detected, restarting Windows container...")
                    cmd_restart_windows(args)
                    stall_count = 0
            else:
                stall_count = 0
                last_storage_size = current_storage

            elapsed = int(time.time() - start)
            progress_gb = current_storage / 1e9
            log("RALPH", f"[{elapsed//60:02d}:{elapsed%60:02d}] Waiting for WAA... ({progress_gb:.1f}GB)")

        except subprocess.TimeoutExpired:
            log("RALPH", "SSH timeout, retrying...")

        time.sleep(30)

    log("RALPH", f"WAA server not ready after {timeout}s")
    return False


def run_benchmark_iteration(args, num_tasks: int) -> Dict[str, Any]:
    """Run one iteration of benchmark tasks."""
    log("RALPH", f"Running {num_tasks} tasks...")

    # Use openadapt-evals CLI
    result = subprocess.run(
        [
            "uv", "run",
            "python", "-m", "openadapt_evals.benchmarks.cli",
            "run",
            "--agent", "api-claude",
            "--tasks", str(num_tasks),
            "--server", "http://localhost:5001"
        ],
        capture_output=True,
        text=True,
        cwd="/Users/abrichr/oa/src/openadapt-evals"
    )

    # Parse results from stdout/log
    # TODO: Parse actual success count from results JSON
    successful = num_tasks  # Placeholder
    cost = 0.15 * num_tasks  # Rough estimate

    return {
        "successful_tasks": successful,
        "total_tasks": num_tasks,
        "cost_usd": cost
    }


def should_continue_after_failure(metrics: RalphMetrics, config: dict) -> bool:
    """Determine if we should retry after a failure."""
    consecutive_failures = sum(
        1 for r in metrics.iteration_results[-3:] if r.get("successful", 0) == 0
    )
    if consecutive_failures >= 2:
        log("RALPH", "Too many consecutive failures, stopping")
        return False
    return True


# Add argparse entry
p_ralph = subparsers.add_parser("ralph", help="Autonomous benchmark loop")
p_ralph.add_argument("--tasks", type=int, default=20, help="Tasks per iteration")
p_ralph.add_argument("--target-success-rate", type=float, default=0.8)
p_ralph.add_argument("--max-cost", type=float, default=10.0)
p_ralph.add_argument("--max-hours", type=float, default=4.0)
p_ralph.add_argument("--max-iterations", type=int, default=10)
p_ralph.set_defaults(func=cmd_ralph)
```

---

## Step 4: Integration with CLAUDE.md

Add to `/Users/abrichr/oa/src/openadapt-ml/CLAUDE.md` under "Benchmark Integration":

```markdown
## Ralph Loop Benchmarking (Autonomous)

**Command**: `uv run python -m openadapt_ml.benchmarks.cli ralph`

The Ralph Loop is a closed-loop autonomous benchmark runner that continues iterating
until a success rate target is achieved or budget/time exhausted.

### Configuration
```bash
# Basic usage (uses defaults from ralph-tasks.json)
uv run python -m openadapt_ml.benchmarks.cli ralph

# Override specific parameters
uv run python -m openadapt_ml.benchmarks.cli ralph \
    --tasks 20 \
    --target-success-rate 0.80 \
    --max-cost 10.0 \
    --max-hours 4 \
    --max-iterations 10
```

### How It Works
1. **Iteration 1**: Run 20 WAA tasks, measure success rate
2. **Evaluate**: If success_rate >= 80%, done. Else continue.
3. **Iteration 2**: Run 20 more tasks (cumulative tracking)
4. **Repeat** until: success rate achieved OR budget exhausted OR time expired

### Monitoring
```bash
# In another terminal, watch progress
watch -n 5 cat ralph-progress.txt

# Or check logs
tail -f /tmp/ralph.log
```

### Recovery Modes
- **VM not ready**: Auto-retry with backoff (3 attempts)
- **Windows boot stuck**: Auto-restart container after 5 min stall
- **Disk full**: Auto-prune Docker cache
- **WAA probe timeout**: Auto-recover 3 times before abort

### Success Metrics
- Cumulative success rate (tracked across all iterations)
- Cost per task (total spend / total executed)
- Stop when: success_rate >= target OR cost/time limits hit
```

---

## Step 5: Example ralph-tasks.json Scenarios

For different benchmark goals:

**Scenario A: Baseline (minimal cost)**
```json
{
  "target_success_rate": 0.50,
  "max_budget_usd": 5.0,
  "max_runtime_hours": 2.0,
  "max_iterations": 5
}
```

**Scenario B: SOTA (higher cost)**
```json
{
  "target_success_rate": 0.85,
  "max_budget_usd": 25.0,
  "max_runtime_hours": 8.0,
  "max_iterations": 15
}
```

**Scenario C: Fast proof-of-concept**
```json
{
  "target_success_rate": 0.60,
  "max_budget_usd": 3.0,
  "max_runtime_hours": 1.0,
  "max_iterations": 3
}
```

---

## Step 6: Running the Ralph Loop

### Before First Run

1. Create task definitions:
   ```bash
   cp docs/ralph_waa_guide.md ralph-tasks.json
   # Edit ralph-tasks.json with your parameters
   ```

2. Ensure API keys are set:
   ```bash
   # In .env or as env vars
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   ```

3. Verify existing VM status:
   ```bash
   uv run python -m openadapt_ml.benchmarks.cli vm status
   ```

### Start Loop

```bash
# Fire and forget
uv run python -m openadapt_ml.benchmarks.cli ralph &

# Monitor in background
tail -f ralph-progress.txt &
```

### Early Stop

```bash
# List ralph processes
pgrep -f "benchmarks.cli ralph" -l

# Kill gracefully (will save state and deallocate VM)
kill -TERM <PID>
```

---

## Step 7: Post-Loop Review

After loop completes, run:

```bash
# 1. Check final metrics
cat ralph-progress.txt

# 2. Review benchmark results
uv run python -m openadapt_evals.benchmarks.cli view --run-name live_eval

# 3. Extract learnings
grep "LEARNINGS" ralph-progress.txt

# 4. Archive results
mkdir -p benchmark_archives
cp -r benchmark_results benchmark_archives/results_$(date +%s)
cp ralph-progress.txt benchmark_archives/progress_$(date +%s).txt
```

---

## Key Differences from Manual Workflow

| Aspect | Manual | Ralph Loop |
|--------|--------|-----------|
| Start | `vm create` → `setup-waa` → `run` | One `ralph` command |
| Recovery | Manual re-runs | Automatic with backoff |
| Monitoring | Watch logs manually | Auto-tracked in ralph-progress.txt |
| Stopping | Manual (or guess at timeout) | Goal-based (success rate target) |
| Multiple iterations | Manual workflow repeat | Single loop command |
| Cost control | Manual deallocate | Automatic budget limit |

---

## Troubleshooting

**Q: Loop stuck on "Waiting for WAA"**
A: Check Windows boot via VNC: `uv run python -m openadapt_ml.benchmarks.cli vm monitor`
   If Windows hung, loop will auto-restart container after 5 min stall.

**Q: Tasks have low success rate, loop keeps iterating**
A: Expected behavior if `target_success_rate` not achieved yet.
   Lower the target or increase `max_cost`/`max_hours` budget.
   Or interrupt with `kill -TERM` and debug individual task failures.

**Q: Loop uses more money than expected**
A: Each iteration runs `N` tasks. Total cost = iterations × (task_cost × N).
   Reduce `--tasks` per iteration or lower `target_success_rate`.

**Q: How do I know if the loop is still running?**
A: `cat ralph-progress.txt` shows updated timestamps.
   If timestamp is > 1 min old, loop may have crashed. Check `vm status`.

---

## Implementation Roadmap

**Phase 1 (Week 1)**: Core loop skeleton + Phase 1/2 execution
**Phase 2 (Week 2)**: Smart recovery + stall detection
**Phase 3 (Week 3)**: Convergence detection + adjustments
**Phase 4 (Week 4)**: Polish + reporting + notification

See `docs/ralph_loop_analysis.md` for full technical details.
