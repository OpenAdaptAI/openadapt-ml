# VM Monitor Dashboard Screenshot Analysis

**Date**: 2026-01-17
**Goal**: Generate terminal screenshots automatically for README without manual work
**Target Command**: `uv run python -m openadapt_ml.benchmarks.cli vm monitor`

---

## Executive Summary

**RECOMMENDATION: Semi-Automated with asciinema + agg (Option B)**

- **Effort**: 2-3 hours implementation + 30 min per screenshot session
- **Quality**: High - authentic terminal output, reproducible, no VM costs
- **Maintenance**: Low - script can be reused for future updates
- **ROI**: Positive - worth automating given documentation importance and reusability

**Key Insight**: The `vm monitor` command already outputs beautifully formatted terminal output with box drawing, status icons, and structured sections. We should capture this REAL output, not mock it.

---

## Analysis of Terminal Screenshot Tools

### Option A: asciinema + agg (RECOMMENDED)

**Tools**:
- `asciinema` - Records terminal sessions as JSON
- `agg` (asciinema-to-gif) - Converts recordings to PNG/GIF

**Pros**:
- ✅ Captures REAL terminal output with all formatting
- ✅ Reproducible - can regenerate screenshots from recordings
- ✅ No VM costs - mock data works
- ✅ High quality output
- ✅ Can edit recordings before rendering
- ✅ Open source, actively maintained

**Cons**:
- ❌ Requires two tools (asciinema + agg)
- ❌ JSON recordings are verbose (not human-editable)
- ❌ No built-in editing UI

**Example Workflow**:
```bash
# 1. Record terminal session (with mock data)
asciinema rec vm_monitor.cast --command "uv run python -m openadapt_ml.benchmarks.cli vm monitor --mock"

# 2. Convert to PNG
agg vm_monitor.cast vm_monitor.png --font-family "Monaco" --font-size 14

# 3. Trim/crop if needed
# Use ImageMagick: convert vm_monitor.png -crop 800x600+0+0 vm_monitor_cropped.png
```

**Installation**:
```bash
brew install asciinema
brew install agg
```

**Cost**: Free, $0

---

### Option B: termshot

**Tool**: `termshot` - Direct terminal screenshot utility

**Pros**:
- ✅ One-step screenshot generation
- ✅ SVG output (scalable, crisp on all displays)
- ✅ Simple command-line interface
- ✅ Can style with CSS

**Cons**:
- ❌ Requires command to complete quickly (not ideal for long-running monitors)
- ❌ Less control over timing/frames
- ❌ Harder to reproduce exact output

**Example Workflow**:
```bash
# Direct screenshot
termshot --command "uv run python -m openadapt_ml.benchmarks.cli vm status" \
  --output vm_status.svg \
  --font Monaco \
  --columns 120
```

**Installation**:
```bash
brew install termshot
```

**Cost**: Free, $0

---

### Option C: carbon-now-cli

**Tool**: `carbon-now-cli` - Generate beautiful code screenshots via Carbon

**Pros**:
- ✅ Beautiful, stylized output
- ✅ Good for marketing/documentation
- ✅ Many themes and customization options
- ✅ Embeddable images

**Cons**:
- ❌ NOT for terminal output - designed for code
- ❌ Doesn't preserve terminal formatting (box drawing, colors)
- ❌ Overkill for our use case

**Verdict**: NOT SUITABLE for terminal screenshots

---

### Option D: Manual Screenshots

**Method**: Run command, take screenshot with macOS/Windows tools

**Pros**:
- ✅ Zero setup
- ✅ WYSIWYG - exactly what user sees
- ✅ Can show real VM data

**Cons**:
- ❌ Not reproducible
- ❌ Hard to update when output changes
- ❌ Requires VM running ($$$)
- ❌ Manual cropping/editing needed
- ❌ Different on every platform

**Effort**: 30-60 min per screenshot session

**Verdict**: ACCEPTABLE but not ideal

---

## Workflow Design

### Approach A: Record Real `vm monitor` Session (NOT RECOMMENDED)

**Flow**:
1. Start Azure VM
2. Wait for VM to be ready
3. Run `vm monitor` command
4. Record terminal output
5. Convert to screenshot
6. Stop VM

**Problems**:
- Requires VM running ($0.192/hour)
- 15-30 min setup time
- Real IP addresses, Azure IDs (need sanitizing)
- Not reproducible without VM

**Verdict**: TOO EXPENSIVE for documentation screenshots

---

### Approach B: Mock Output with Test Data (RECOMMENDED)

**Flow**:
1. Add `--mock` flag to `vm monitor` command
2. Mock returns fake but realistic data:
   - VM IP: `172.171.112.41` (example)
   - Activity: "WAA benchmark ready"
   - Cost: `$1.23` (example)
   - Azure ML jobs: 2-3 fake jobs
3. Record with asciinema
4. Convert to PNG with agg
5. Save to `docs/screenshots/`

**Why This Works**:
- The `vm monitor` command already has great terminal output formatting
- We just need to mock the data sources
- Zero VM costs
- Reproducible
- Fast iteration

**Implementation**:
```python
# In cli.py, add --mock flag to vm monitor:
if args.mock:
    # Mock VM status
    vm_name = "azure-waa-vm"
    ip = "172.171.112.41"
    vm_size = "Standard_D4ds_v5"
    power_state = "VM running"

    # Mock activity
    activity = VMActivity(
        is_active=True,
        activity_type="benchmark_running",
        description="WAA benchmark ready (154 tasks)",
    )

    # Mock costs
    uptime_hours = 2.5
    costs = calculate_vm_costs(vm_size, uptime_hours)

    # Mock Azure ML jobs
    jobs = [
        AzureMLJob(job_id="abc123", display_name="waa-eval-run-1",
                   status="completed", created_at="2026-01-15T10:30:00Z"),
        AzureMLJob(job_id="def456", display_name="waa-eval-run-2",
                   status="running", created_at="2026-01-17T08:15:00Z"),
    ]
```

**Effort**: 2-3 hours to implement mock flag + recording script

---

### Approach C: Hybrid (Real Command, Sanitized Data)

**Flow**:
1. Run real `vm monitor` against VM
2. Record with asciinema
3. Post-process .cast file to replace sensitive data
4. Convert to PNG

**Problems**:
- Still requires VM
- Post-processing JSON is fragile
- Not much better than Approach B

**Verdict**: NOT WORTH IT

---

## Screenshots Needed

Based on README requirements and `vm monitor` command output:

### Priority 1: Core Screenshots (3-4)

1. **VM Monitor Dashboard (Full)**
   - Shows all 6 sections:
     1. VM Status (name, IP, size, state)
     2. Current Activity (idle/benchmark_running)
     3. Cost Tracking (uptime, rate, cost)
     4. Recent Azure ML Jobs (last 7 days)
     5. Evaluation History (optional with --details)
     6. Dashboard & Access (server URL, tunnels)
   - Terminal: ~70 cols x 40 rows
   - File: `vm_monitor_dashboard_full.png`

2. **VM Monitor - Active Benchmark**
   - Activity shows "BENCHMARK_RUNNING"
   - Cost shows realistic uptime
   - Recent jobs show 2-3 jobs
   - File: `vm_monitor_active.png`

3. **VM Monitor - Idle State**
   - Activity shows "IDLE"
   - Lower cost (shorter uptime)
   - File: `vm_monitor_idle.png`

4. **VM Monitor with --details Flag**
   - Shows section 5 (Evaluation History)
   - Shows daily/weekly cost breakdown
   - File: `vm_monitor_details.png`

### Priority 2: Supplementary Screenshots (2-3)

5. **VM Setup Command**
   - Output of `vm setup-waa` command
   - Shows Docker installation, image pull
   - File: `vm_setup_output.png`

6. **VM Run WAA Command**
   - Output of `vm run-waa --num-tasks 5`
   - Shows benchmark progress
   - File: `vm_run_waa_output.png`

7. **VM Status/Diag Commands**
   - Combined view of `vm status` and `vm diag`
   - Shows quick health check
   - File: `vm_status_diag.png`

---

## Implementation Plan

### Phase 1: Mock Data Implementation (2-3 hours)

**File**: `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py`

Add `--mock` flag to `vm monitor` subcommand:

```python
# In vm_parser.add_parser("monitor", ...)
monitor_parser.add_argument(
    "--mock",
    action="store_true",
    help="Use mock data (no VM required, for documentation/testing)",
)
```

Implement mock data generator:

```python
def get_mock_vm_data():
    """Generate realistic mock data for vm monitor screenshots."""
    return {
        "vm_name": "azure-waa-vm",
        "ip": "172.171.112.41",
        "vm_size": "Standard_D4ds_v5",
        "power_state": "VM running",
        "activity": VMActivity(
            is_active=True,
            activity_type="benchmark_running",
            description="WAA benchmark ready (154 tasks)",
        ),
        "uptime_hours": 2.5,
        "jobs": [
            AzureMLJob(
                job_id="abc123def456",
                display_name="waa-eval-20-tasks",
                status="completed",
                created_at="2026-01-15T10:30:00Z",
            ),
            AzureMLJob(
                job_id="ghi789jkl012",
                display_name="waa-eval-50-tasks",
                status="running",
                created_at="2026-01-17T08:15:00Z",
            ),
        ],
        "eval_history": [
            EvaluationRun(
                run_id="20260115_103045",
                started_at="2026-01-15T10:30:45Z",
                completed_at="2026-01-15T12:15:30Z",
                num_tasks=20,
                success_rate=0.65,
                agent_type="api-claude",
                status="completed",
            ),
        ],
    }
```

Modify `vm monitor` action to use mock data:

```python
elif args.action == "monitor":
    if args.mock:
        mock_data = get_mock_vm_data()
        # Use mock_data instead of Azure queries
        vm_name = mock_data["vm_name"]
        ip = mock_data["ip"]
        # ... etc
```

### Phase 2: Recording Script (30 min)

**File**: `/Users/abrichr/oa/src/openadapt-ml/scripts/generate_vm_screenshots.py`

```python
#!/usr/bin/env python3
"""Generate VM monitor screenshots for documentation.

Usage:
    python scripts/generate_vm_screenshots.py

Output:
    docs/screenshots/vm_monitor_*.png
"""

import subprocess
import time
from pathlib import Path

SCREENSHOTS_DIR = Path(__file__).parent.parent / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

def record_and_convert(command: list[str], output_name: str, width: int = 120, height: int = 40):
    """Record terminal command and convert to PNG.

    Args:
        command: Command to run (list of strings)
        output_name: Output filename (without extension)
        width: Terminal width in columns
        height: Terminal height in rows
    """
    cast_file = SCREENSHOTS_DIR / f"{output_name}.cast"
    png_file = SCREENSHOTS_DIR / f"{output_name}.png"

    print(f"Recording: {' '.join(command)}")

    # Record with asciinema
    subprocess.run(
        [
            "asciinema",
            "rec",
            str(cast_file),
            "--overwrite",
            "--command",
            " ".join(command),
        ],
        env={**os.environ, "COLUMNS": str(width), "LINES": str(height)},
        check=True,
    )

    print(f"Converting to PNG: {png_file}")

    # Convert to PNG with agg
    subprocess.run(
        [
            "agg",
            str(cast_file),
            str(png_file),
            "--font-family",
            "Monaco",
            "--font-size",
            "14",
        ],
        check=True,
    )

    print(f"✓ Saved: {png_file}")

    # Clean up cast file
    cast_file.unlink()


def main():
    """Generate all VM monitor screenshots."""

    # Screenshot 1: Full monitor dashboard
    record_and_convert(
        ["uv", "run", "python", "-m", "openadapt_ml.benchmarks.cli", "vm", "monitor", "--mock"],
        "vm_monitor_dashboard_full",
        width=120,
        height=45,
    )

    # Screenshot 2: Monitor with --details
    record_and_convert(
        ["uv", "run", "python", "-m", "openadapt_ml.benchmarks.cli", "vm", "monitor", "--mock", "--details"],
        "vm_monitor_details",
        width=120,
        height=50,
    )

    # Screenshot 3: VM status (quick check)
    record_and_convert(
        ["uv", "run", "python", "-m", "openadapt_ml.benchmarks.cli", "vm", "status", "--mock"],
        "vm_status",
        width=100,
        height=20,
    )

    print("\n✓ All screenshots generated!")
    print(f"   Location: {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
```

### Phase 3: README Integration (15 min)

Update README to include screenshots:

```markdown
## VM Monitoring

The `vm monitor` command provides a comprehensive dashboard for tracking VM usage:

![VM Monitor Dashboard](docs/screenshots/vm_monitor_dashboard_full.png)

Features:
- **VM Status**: Real-time state, size, and IP
- **Activity Detection**: What the VM is currently doing
- **Cost Tracking**: Current uptime and total cost
- **Azure ML Jobs**: Recent jobs from last 7 days
- **Evaluation History**: Past benchmark runs (with --details flag)
- **Dashboard & Tunnels**: Auto-starts web dashboard and SSH/VNC tunnels

### Detailed View

Use `--details` flag to see extended information:

![VM Monitor with Details](docs/screenshots/vm_monitor_details.png)
```

---

## Cost-Benefit Analysis

### Option 1: Fully Automated (Recommended)

**Investment**:
- Initial implementation: 2-3 hours
- Per-screenshot session: 10-15 min (run script)
- Maintenance: Low (script is reusable)

**Benefits**:
- High quality, authentic output
- Reproducible
- No VM costs
- Easy to update when command changes
- Can generate variations (idle, active, details)

**ROI**: HIGH - script is reusable for future updates and other CLI commands

---

### Option 2: Manual Screenshots

**Investment**:
- Per-screenshot session: 30-60 min
- Requires VM running: $0.20-0.50 per session
- Manual cropping/editing: 15-30 min

**Benefits**:
- Shows real data
- Zero code changes

**Drawbacks**:
- Not reproducible
- Higher ongoing cost
- Harder to update

**ROI**: MEDIUM - acceptable for one-time use, but not ideal for maintenance

---

## Recommendation

**Implement Option 1: Semi-Automated with asciinema + agg + Mock Data**

### Why This is Best

1. **Quality**: Captures real terminal output with all formatting intact
2. **Cost**: $0 - no VM required with mock data
3. **Reproducibility**: Can regenerate anytime
4. **Maintainability**: Script can be reused for updates
5. **Authenticity**: Shows actual command output (just with fake data)
6. **Time Investment**: 2-3 hours upfront, 10-15 min per update

### 80/20 MVP

**Minimum for Maximum Impact**:
- Implement `--mock` flag for `vm monitor` command
- Create recording script for 2-3 key screenshots
- Update README with images

**Skip for Now**:
- Screenshot editing UI
- Animated GIFs (PNG is sufficient)
- Screenshots for every CLI command (focus on `vm monitor`)

### Next Steps

1. ✅ **Phase 1**: Implement `--mock` flag in cli.py (2 hours)
2. ✅ **Phase 2**: Create recording script (30 min)
3. ✅ **Phase 3**: Generate screenshots (15 min)
4. ✅ **Phase 4**: Update README (15 min)

**Total Estimated Time**: 3-3.5 hours

---

## Alternative: Quick Manual Approach (If Automated Not Worth It)

**If you decide automation isn't worth it** (disagree with recommendation):

### Manual Process (30 min total)

1. Start VM: `uv run python -m openadapt_ml.benchmarks.cli vm start`
2. Wait for ready: `uv run python -m openadapt_ml.benchmarks.cli vm monitor`
3. Take screenshot:
   - macOS: `Cmd+Shift+4` → Select terminal window
   - Windows: Snipping Tool
   - Linux: `gnome-screenshot --window`
4. Crop/edit screenshot in Preview/GIMP
5. Save to `docs/screenshots/`
6. Stop VM: `uv run python -m openadapt_ml.benchmarks.cli vm deallocate`

**Cost**: $0.10-0.20 + 30 min manual work

**When to Use This**:
- One-time documentation need
- No plans to update screenshots frequently
- VM is already running for other work

---

## Conclusion

**RECOMMENDED: Implement Semi-Automated Approach (Option 1)**

- **Effort**: 3-3.5 hours
- **Quality**: High
- **ROI**: Positive (reusable script, no VM costs, easy updates)
- **Next Action**: Implement `--mock` flag and recording script

The automated approach is worth the investment because:
1. The script is reusable for future updates
2. Zero ongoing VM costs
3. Easy to generate variations (idle, active, details)
4. Can be applied to other CLI commands later
5. Screenshots will need updating as features evolve

**Final Verdict**: AUTOMATE IT
