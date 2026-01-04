# Benchmark Viewer UX Improvements

## Current State Analysis

The benchmark viewer (`benchmark.html`) has several confusing elements that need improvement.

### Issue 1: Conflicting Status Indicators in Background Tasks

**Current behavior:**
```
Background Tasks
Updated 11:04:34 PM
Azure VM Host
⏳ Starting
Linux host running at 172.171.112.41
Complete
completed
```

**Problems:**
- Shows both "Starting" AND "Complete" and "completed" simultaneously
- Unclear what "Azure VM Host" status actually represents
- Redundant labels ("Complete" vs "completed")
- The emoji (⏳) conflicts with "Complete" text below

**Proposed fix:**
- Single, clear status per item: `Running`, `Stopped`, `Error`
- Remove redundant labels
- Use consistent terminology

**Mockup:**
```
Background Tasks
Azure VM Host                    ● Running
172.171.112.41                   Last ping: 2s ago
```

---

### Issue 2: VM Discovery Shows "Not Responding" When VM IS Responding

**Current behavior:**
```
VM Discovery
azure-waa-vm                     offline
SSH: azureuser@172.171.112.41
Container: winarena
✗ WAA Not Responding
Last checked: 1/1/2026, 11:04:24 PM
```

**Problems:**
- Shows "offline" and "WAA Not Responding" but benchmark IS running
- The probe is checking wrong endpoint or not updating
- Container name shows "winarena" but actual container is "waa-container"
- Confusing mix of VM status vs WAA server status

**Root cause investigation needed:**
1. Is the probe hitting the right endpoint? (`http://172.30.0.2:5000/probe` inside container)
2. Is the probe result being cached/not refreshed?
3. Is the VM registry out of sync with actual state?

**Proposed fix:**
- Separate VM status from WAA server status
- Add actual probe result display
- Show container status from `docker ps`

**Mockup:**
```
VM Status
┌─────────────────────────────────────────────────┐
│ azure-waa-vm                        ● Connected │
│ SSH: azureuser@172.171.112.41                   │
│ Container: waa-container            ● Running   │
│ Windows VM:                         ● Booted    │
│ WAA Server:                         ● Ready     │
│ Last probe: {"status": "Probe successful"}      │
└─────────────────────────────────────────────────┘
```

---

### Issue 3: Azure Jobs Shows Stale Historical Data

**Current behavior:**
```
Azure Jobs
Live from Azure • 11:04:36 PM
Canceled  waa-worker-0  12/17/2025, 11:30:01 AM
Failed    waa-worker-0  12/17/2025, 11:01:01 AM
Failed    waa-worker-0  12/17/2025, 12:27:31 AM
...
```

**Problems:**
- Shows jobs from December 2025 (2+ weeks old)
- No indication of current running benchmark
- "Live from Azure" is misleading - this is Azure ML jobs, not our direct VM run
- Creates confusion: user thinks nothing is running

**Proposed fix:**
- Separate "Azure ML Jobs" (historical) from "Current Benchmark" (active)
- Add prominent "Currently Running" section at top
- Filter/collapse old jobs by default

**Mockup:**
```
Current Benchmark
┌─────────────────────────────────────────────────┐
│ ● RUNNING  GPT-4o on WAA (30 tasks)             │
│ Started: 11:20 PM • Elapsed: 43 min             │
│ Progress: 5/30 tasks (17%)                      │
│ Current: libreoffice_calc - 21ab7b40...         │
│ [View VNC] [View Logs] [Stop]                   │
└─────────────────────────────────────────────────┘

Historical Azure ML Jobs (collapsed by default)
▶ 6 jobs from December 2025
```

---

### Issue 4: Live Evaluation Says "No evaluation running"

**Current behavior:**
```
Live Evaluation
Updated 11:04:30 PM
No evaluation running
```

**Problems:**
- A benchmark IS running on the Azure VM
- The viewer doesn't know about it because it's not using our tracking system
- Disconnect between "direct SSH benchmark" vs "tracked evaluation"

**Root cause:**
The benchmark was started via direct `docker exec` command, not through our CLI's tracking system. The viewer looks for `benchmark_live.json` which doesn't exist for this run.

**Proposed fixes:**

Option A: Make direct runs trackable
- When starting benchmark via SSH, also create tracking file
- Poll the VM's benchmark log and update local tracking file

Option B: Auto-detect running benchmarks
- Check if VM has running benchmark process
- Parse the log file for progress
- Display even without explicit tracking

**Recommended: Option B** - more robust, doesn't require behavior change

**Mockup:**
```
Live Evaluation
┌─────────────────────────────────────────────────┐
│ ● Detected: WAA benchmark running on Azure VM   │
│                                                 │
│ Model: gpt-4o                                   │
│ Tasks: 127 total (all domains)                  │
│ Progress: ~5 completed                          │
│                                                 │
│ Current task: 21ab7b40-77c2-4ae6-8321-...       │
│ Domain: libreoffice_calc                        │
│ Step: 15/15                                     │
│                                                 │
│ [Auto-detected from /tmp/waa_benchmark.log]     │
└─────────────────────────────────────────────────┘
```

---

### Issue 5: No Live Updates Visible

**Current behavior:**
- Page shows static data
- No visual indication of polling/updates
- Timestamps don't update frequently

**Problems:**
- User can't tell if page is actively monitoring
- No feedback on connection status
- Stale data looks the same as fresh data

**Proposed fixes:**
1. Add pulsing indicator when actively polling
2. Show "Last updated: Xs ago" that counts up
3. Flash/highlight sections when data changes
4. Show connection status prominently

**Mockup:**
```
┌──────────────────────────────────────────────────────────┐
│ 🟢 Connected to Azure VM • Polling every 5s • Updated 2s │
└──────────────────────────────────────────────────────────┘
```

---

### Issue 6: Benchmark Run Shows Old Test Data

**Current behavior:**
```
Benchmark Run:
openai-api - 0% (waa_eval_20251217_test_real)
✓ REAL - Actual Windows Agent Arena evaluation
Total Tasks: 10
Success Rate: 0.0%
```

**Problems:**
- Shows run from December 17th, not current run
- "0%" and "0.0%" success rate is discouraging/unclear
- No indication this is historical vs current

**Proposed fix:**
- Clearly label as "Last Completed Run" vs "Current Run"
- Show current run progress prominently
- Archive old runs in collapsible section

---

## Implementation Priority

### P0 - Critical (blocks understanding of current state)
1. Fix "No evaluation running" - detect running benchmarks
2. Fix VM status showing "Not Responding" when it IS responding
3. Add "Currently Running" section at top

### P1 - Important (reduces confusion)
4. Clean up Background Tasks status display
5. Separate current vs historical Azure jobs
6. Add live update indicators

### P2 - Nice to have
7. Polish status card designs
8. Add task-level progress details
9. Improve mobile responsiveness

---

## Technical Implementation Notes

### Detecting Running Benchmarks (P0 #1)

The viewer should poll the VM to detect running benchmarks:

```javascript
async function detectRunningBenchmark(vmConfig) {
  // 1. Check if benchmark process is running
  const psResult = await sshExec(vmConfig,
    "docker exec waa-container pgrep -f 'python.*run.py'");

  if (!psResult.success) return null;

  // 2. Get log file stats
  const logStat = await sshExec(vmConfig,
    "docker exec waa-container stat /tmp/waa_benchmark.log");

  // 3. Parse recent log entries for progress
  const logTail = await sshExec(vmConfig,
    "docker exec waa-container tail -50 /tmp/waa_benchmark.log");

  return {
    isRunning: true,
    logModified: parseStatTime(logStat),
    currentTask: parseCurrentTask(logTail),
    progress: parseProgress(logTail)
  };
}
```

### Fixing VM Probe (P0 #2)

Current probe may be hitting wrong endpoint. Should be:

```javascript
async function probeWAA(vmConfig) {
  // Probe from INSIDE the container (where Windows VM is accessible)
  const result = await sshExec(vmConfig,
    "docker exec waa-container curl -s --connect-timeout 5 http://172.30.0.2:5000/probe");

  return JSON.parse(result);
}
```

### Adding Current Run Section (P0 #3)

New HTML section:

```html
<div id="current-run" class="status-card prominent">
  <h3>🔴 Currently Running</h3>
  <div class="run-info">
    <div class="model">GPT-4o</div>
    <div class="benchmark">Windows Agent Arena</div>
    <div class="progress">
      <div class="progress-bar" style="width: 17%"></div>
      <span>5/30 tasks</span>
    </div>
    <div class="current-task">
      <span class="domain">libreoffice_calc</span>
      <span class="task-id">21ab7b40...</span>
      <span class="step">Step 12/15</span>
    </div>
    <div class="elapsed">Elapsed: 43 min</div>
  </div>
  <div class="actions">
    <button onclick="openVNC()">View VNC</button>
    <button onclick="viewLogs()">View Logs</button>
    <button onclick="stopBenchmark()" class="danger">Stop</button>
  </div>
</div>
```

---

## Questions for User

1. Should we prioritize fixing the detection issue (so current run shows up) or cleaning up the UI layout first?

2. For historical runs, should we:
   - A) Show last 5 runs with details
   - B) Show only last run, collapse older
   - C) Move all historical to separate "History" tab

3. Should the viewer auto-refresh, or require manual refresh?
   - A) Auto-refresh every 5s (more resource usage)
   - B) Auto-refresh every 30s
   - C) Manual refresh only with "Refresh" button
