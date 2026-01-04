# SSE Frontend Integration Design

## Overview

This document describes how to replace the current polling-based approach in the benchmark viewer frontend with Server-Sent Events (SSE) for real-time updates.

**Backend endpoint**: `/api/benchmark-sse` (implemented in `openadapt_ml/cloud/local.py:1501-1624`)
**Frontend target**: `openadapt_ml/training/benchmark_viewer.py`

---

## 1. Current State (Polling Approach)

The benchmark viewer currently uses multiple `setInterval` polling loops:

| Panel | Function | Interval | Endpoint |
|-------|----------|----------|----------|
| Background Tasks | `fetchBackgroundTasks()` | 10s | `/api/tasks` |
| Live Evaluation | `fetchLiveEvaluation()` | 2s | `/api/benchmark-live` |
| Azure Jobs | `fetchAzureJobs()` | 10s | `/api/azure-jobs` |
| Azure Job Logs | `fetchJobLogs()` | 5s | `/api/azure-job-logs` |
| VM Discovery | `fetchVMs()` | 10s | `/api/vms` |

### Current Code Pattern (benchmark_viewer.py)

```javascript
// Background Tasks Panel (lines 505-507)
fetchBackgroundTasks();
setInterval(fetchBackgroundTasks, 10000);

// Live Evaluation Panel (lines 757-759)
fetchLiveEvaluation();
setInterval(fetchLiveEvaluation, 2000);

// Azure Jobs Panel (lines 1072-1075)
fetchAzureJobs();
setInterval(fetchAzureJobs, 10000);
setInterval(fetchJobLogs, 5000);

// VM Discovery Panel (lines 1465-1467)
fetchVMs();
setInterval(fetchVMs, 10000);
```

### Problems with Current Approach

1. **High request volume**: 6+ requests/minute per panel, most return unchanged data
2. **Latency**: Updates arrive up to 10s after actual state changes
3. **Resource waste**: Empty HTTP responses consume bandwidth and server resources
4. **Inconsistent intervals**: 2s-10s depending on panel, leading to staggered updates
5. **No connection management**: Polling continues even when browser tab is inactive

---

## 2. Target State (SSE with Fallback)

### Architecture

```
Browser                          Server (local.py)
   |                                   |
   |--EventSource(/api/benchmark-sse)->|
   |                                   |
   |<--event: status------------------|  VM/container status
   |<--event: progress-----------------|  Benchmark progress
   |<--event: task_complete------------|  Task finished
   |<--event: heartbeat----------------|  Keep-alive (new)
   |<--event: error--------------------|  Error messages
   |                                   |
```

### Unified SSE Connection

Replace all 5+ polling loops with a single EventSource connection:

```javascript
// Single SSE connection replaces all polling
const eventSource = new EventSource('/api/benchmark-sse?interval=2');

eventSource.addEventListener('status', handleStatus);
eventSource.addEventListener('progress', handleProgress);
eventSource.addEventListener('task_complete', handleTaskComplete);
eventSource.addEventListener('heartbeat', handleHeartbeat);
eventSource.addEventListener('error', handleError);
```

### Benefits

| Metric | Polling (Current) | SSE (Target) | Improvement |
|--------|-------------------|--------------|-------------|
| Requests/min | ~18 | 1 | 94% reduction |
| Latency | 2-10s | ~instant | Real-time |
| Empty responses | Most | None | 100% reduction |
| Connections | 5+ per page | 1 | Simplified |

---

## 3. JavaScript Code Changes

### 3.1 New SSE Manager Module

Add to `_get_background_tasks_panel_html()` or create new function `_get_sse_manager_js()`:

```javascript
/**
 * SSE Manager - Replaces all polling with unified SSE connection
 * Add this at the start of the page's <script> section
 */

let benchmarkSSE = null;
let sseReconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 3000;

function initBenchmarkSSE(interval = 2) {
    // Check for EventSource support
    if (typeof EventSource === 'undefined') {
        console.warn('SSE not supported, falling back to polling');
        initPollingFallback();
        return;
    }

    // Close existing connection if any
    if (benchmarkSSE) {
        benchmarkSSE.close();
    }

    const url = `/api/benchmark-sse?interval=${interval}`;
    benchmarkSSE = new EventSource(url);

    // Connection opened
    benchmarkSSE.onopen = function() {
        console.log('SSE connection established');
        sseReconnectAttempts = 0;
        updateConnectionStatus('connected');
    };

    // Generic error handler (connection issues)
    benchmarkSSE.onerror = function(e) {
        console.error('SSE connection error:', e);
        updateConnectionStatus('disconnected');

        // EventSource auto-reconnects, but we track attempts
        sseReconnectAttempts++;
        if (sseReconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
            console.warn('Max reconnect attempts reached, falling back to polling');
            benchmarkSSE.close();
            initPollingFallback();
        }
    };

    // Event handlers
    benchmarkSSE.addEventListener('status', handleStatusEvent);
    benchmarkSSE.addEventListener('progress', handleProgressEvent);
    benchmarkSSE.addEventListener('task_complete', handleTaskCompleteEvent);
    benchmarkSSE.addEventListener('heartbeat', handleHeartbeatEvent);
    benchmarkSSE.addEventListener('error', handleErrorEvent);
}

function closeBenchmarkSSE() {
    if (benchmarkSSE) {
        benchmarkSSE.close();
        benchmarkSSE = null;
        updateConnectionStatus('closed');
    }
}

// Clean up on page unload
window.addEventListener('beforeunload', closeBenchmarkSSE);

// Pause SSE when tab is hidden (optional optimization)
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        closeBenchmarkSSE();
    } else {
        initBenchmarkSSE();
    }
});
```

### 3.2 Event Handler Implementations

```javascript
/**
 * Handle 'status' event - VM and container status
 * Replaces: fetchBackgroundTasks() and fetchVMs()
 */
function handleStatusEvent(event) {
    try {
        const data = JSON.parse(event.data);
        console.log('SSE status:', data);

        // Update Background Tasks panel
        updateBackgroundTasksFromSSE(data);

        // Update VM Discovery panel if applicable
        if (data.probe && data.probe.vnc_url) {
            updateVMDiscoveryFromSSE(data);
        }

        // Update last refresh timestamp
        updateRefreshTime('tasks-refresh-time');
        updateRefreshTime('vm-refresh-time');

    } catch (e) {
        console.error('Error parsing status event:', e);
    }
}

function updateBackgroundTasksFromSSE(vmData) {
    // Convert SSE status to background tasks format
    const tasks = [];

    if (vmData.type === 'vm_status') {
        tasks.push({
            task_id: 'docker_container',
            task_type: 'docker_container',
            title: 'WAA Container',
            description: vmData.waa_ready ? 'WAA server ready' : 'Waiting for WAA server',
            status: vmData.connected ? 'running' : 'pending',
            phase: vmData.phase,
            progress_percent: vmData.waa_ready ? 100 : 50,
            metadata: {
                vnc_url: vmData.probe?.vnc_url,
                probe_response: vmData.probe?.status,
                waa_server_ready: vmData.waa_ready
            }
        });
    }

    renderBackgroundTasks(tasks);
}

/**
 * Handle 'progress' event - Benchmark execution progress
 * Replaces: fetchLiveEvaluation()
 */
function handleProgressEvent(event) {
    try {
        const data = JSON.parse(event.data);
        console.log('SSE progress:', data);

        // Update Live Evaluation panel
        updateLiveEvaluationFromSSE(data);

        // Update progress bar if exists
        updateProgressBar(data.tasks_completed, data.total_tasks);

        updateRefreshTime('live-eval-refresh-time');

    } catch (e) {
        console.error('Error parsing progress event:', e);
    }
}

function updateLiveEvaluationFromSSE(progressData) {
    const state = {
        status: 'running',
        tasks_completed: progressData.tasks_completed,
        total_tasks: progressData.total_tasks,
        current_task: {
            task_id: progressData.current_task,
            instruction: `Step ${progressData.current_step || 0}`,
            domain: 'unknown',
            steps: []
        }
    };
    renderLiveEvaluation(state);
}

/**
 * Handle 'task_complete' event - Individual task finished
 * New: Previously not tracked in real-time
 */
function handleTaskCompleteEvent(event) {
    try {
        const data = JSON.parse(event.data);
        console.log('SSE task_complete:', data);

        // Add to results list
        addTaskResult(data);

        // Show notification
        showTaskNotification(data);

        // Update summary metrics
        updateTaskMetrics(data);

    } catch (e) {
        console.error('Error parsing task_complete event:', e);
    }
}

function addTaskResult(taskData) {
    // Find or create results container
    let resultsContainer = document.getElementById('task-results-list');
    if (!resultsContainer) {
        // Create if doesn't exist
        const panel = document.getElementById('live-eval-panel');
        if (panel) {
            resultsContainer = document.createElement('div');
            resultsContainer.id = 'task-results-list';
            resultsContainer.className = 'task-results-list';
            panel.appendChild(resultsContainer);
        }
    }

    if (resultsContainer) {
        const resultClass = taskData.success ? 'success' : 'failure';
        const resultIcon = taskData.success ? 'check-circle' : 'x-circle';
        const scoreText = taskData.score !== null ? ` (${taskData.score.toFixed(2)})` : '';

        const resultHtml = `
            <div class="task-result ${resultClass}">
                <span class="result-icon ${resultClass}">${taskData.success ? 'OK' : 'FAIL'}</span>
                <span class="result-task-id">${taskData.task_id}</span>
                <span class="result-score">${scoreText}</span>
            </div>
        `;
        resultsContainer.insertAdjacentHTML('afterbegin', resultHtml);
    }
}

function showTaskNotification(taskData) {
    // Browser notification (if permitted)
    if (Notification.permission === 'granted') {
        new Notification(`Task ${taskData.success ? 'passed' : 'failed'}: ${taskData.task_id}`, {
            body: taskData.score !== null ? `Score: ${taskData.score}` : undefined,
            icon: taskData.success ? '/static/check.png' : '/static/x.png'
        });
    }
}

/**
 * Handle 'heartbeat' event - Keep-alive signal
 * New: Prevents proxy/LB timeouts
 */
function handleHeartbeatEvent(event) {
    try {
        const data = JSON.parse(event.data);
        // Update connection indicator
        updateConnectionStatus('connected');

        // Optional: log for debugging
        if (data.timestamp) {
            console.debug('SSE heartbeat:', new Date(data.timestamp * 1000).toLocaleTimeString());
        }
    } catch (e) {
        // Heartbeat may be empty, that's OK
        updateConnectionStatus('connected');
    }
}

/**
 * Handle 'error' event - Server-side errors
 * Note: Different from onerror (connection errors)
 */
function handleErrorEvent(event) {
    try {
        const data = JSON.parse(event.data);
        console.error('SSE server error:', data.message);

        // Show error in UI
        showErrorBanner(data.message);

    } catch (e) {
        console.error('Error parsing error event:', e);
    }
}

function showErrorBanner(message) {
    let banner = document.getElementById('sse-error-banner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'sse-error-banner';
        banner.className = 'error-banner';
        document.body.insertBefore(banner, document.body.firstChild);
    }
    banner.textContent = message;
    banner.style.display = 'block';

    // Auto-hide after 10 seconds
    setTimeout(() => {
        banner.style.display = 'none';
    }, 10000);
}
```

### 3.3 UI Helper Functions

```javascript
/**
 * Update connection status indicator
 */
function updateConnectionStatus(status) {
    let indicator = document.getElementById('sse-connection-indicator');
    if (!indicator) {
        // Create indicator in header
        indicator = document.createElement('span');
        indicator.id = 'sse-connection-indicator';
        indicator.className = 'connection-indicator';
        const header = document.querySelector('.tasks-header') || document.querySelector('header');
        if (header) {
            header.appendChild(indicator);
        }
    }

    indicator.className = `connection-indicator ${status}`;
    indicator.title = `SSE: ${status}`;

    const icons = {
        connected: 'Live',
        disconnected: 'Reconnecting...',
        closed: 'Offline'
    };
    indicator.textContent = icons[status] || status;
}

/**
 * Update progress bar
 */
function updateProgressBar(completed, total) {
    const progressFill = document.querySelector('.task-progress-fill');
    if (progressFill && total > 0) {
        const percent = (completed / total) * 100;
        progressFill.style.width = `${percent}%`;
    }

    const progressText = document.querySelector('.task-meta span:first-child');
    if (progressText) {
        progressText.textContent = `${completed}/${total} tasks completed`;
    }
}

/**
 * Update refresh timestamp
 */
function updateRefreshTime(elementId) {
    const el = document.getElementById(elementId);
    if (el) {
        el.textContent = 'Live ' + new Date().toLocaleTimeString();
    }
}

/**
 * Update aggregate task metrics
 */
let taskMetrics = { passed: 0, failed: 0, total: 0 };

function updateTaskMetrics(taskData) {
    taskMetrics.total++;
    if (taskData.success) {
        taskMetrics.passed++;
    } else {
        taskMetrics.failed++;
    }

    const metricsEl = document.getElementById('task-metrics');
    if (metricsEl) {
        const rate = taskMetrics.total > 0 ?
            ((taskMetrics.passed / taskMetrics.total) * 100).toFixed(1) : 0;
        metricsEl.innerHTML = `
            <span class="metric passed">${taskMetrics.passed} passed</span>
            <span class="metric failed">${taskMetrics.failed} failed</span>
            <span class="metric rate">${rate}% success rate</span>
        `;
    }
}
```

---

## 4. Event Handlers Summary

| Event Type | Handler | Updates | Replaces |
|------------|---------|---------|----------|
| `status` | `handleStatusEvent()` | Background Tasks, VM Discovery | `fetchBackgroundTasks()`, `fetchVMs()` |
| `progress` | `handleProgressEvent()` | Live Evaluation, Progress Bar | `fetchLiveEvaluation()` |
| `task_complete` | `handleTaskCompleteEvent()` | Results List, Metrics, Notification | New functionality |
| `heartbeat` | `handleHeartbeatEvent()` | Connection Indicator | New functionality |
| `error` | `handleErrorEvent()` | Error Banner | Implicit in fetch catch |

---

## 5. Fallback Strategy for Browsers Without SSE

### 5.1 Feature Detection

```javascript
function initBenchmarkUpdates(interval = 2) {
    if (typeof EventSource !== 'undefined') {
        console.log('Using SSE for real-time updates');
        initBenchmarkSSE(interval);
    } else {
        console.log('SSE not supported, using polling fallback');
        initPollingFallback(interval);
    }
}
```

### 5.2 Polling Fallback Implementation

```javascript
let pollingIntervals = [];

function initPollingFallback(interval = 5) {
    // Clear any existing intervals
    pollingIntervals.forEach(clearInterval);
    pollingIntervals = [];

    // Background Tasks (10s)
    pollingIntervals.push(setInterval(fetchBackgroundTasks, 10000));
    fetchBackgroundTasks();

    // Live Evaluation (2s for responsiveness)
    pollingIntervals.push(setInterval(fetchLiveEvaluation, interval * 1000));
    fetchLiveEvaluation();

    // Azure Jobs (10s)
    pollingIntervals.push(setInterval(fetchAzureJobs, 10000));
    fetchAzureJobs();

    // VM Discovery (10s)
    pollingIntervals.push(setInterval(fetchVMs, 10000));
    fetchVMs();
}

function stopPollingFallback() {
    pollingIntervals.forEach(clearInterval);
    pollingIntervals = [];
}
```

### 5.3 Browser Compatibility

| Browser | EventSource Support | Strategy |
|---------|---------------------|----------|
| Chrome 6+ | Yes | SSE |
| Firefox 6+ | Yes | SSE |
| Safari 5+ | Yes | SSE |
| Edge 79+ | Yes | SSE |
| IE 11 | No | Polling fallback |
| Opera 11+ | Yes | SSE |

---

## 6. Bug Fixes Needed

### 6.1 Task Success Hardcoded to True

**Location**: `local.py:1593`

**Current Code**:
```python
complete_data = {
    "task_id": last_task,
    "success": True,  # Would need to parse from logs
    "score": None,
}
```

**Problem**: Task success is always reported as `True`, regardless of actual outcome.

**Fix**: Parse success/failure from WAA benchmark log:

```python
def _parse_task_result(self, log_lines: list[str], task_id: str) -> dict:
    """Parse task success/failure from log output.

    WAA log patterns:
    - Success: "Task task_001 completed successfully"
    - Success: "Result: PASS"
    - Failure: "Task task_001 failed"
    - Failure: "Result: FAIL"
    - Score: "Score: 0.85"
    """
    success = None
    score = None

    # Search backwards from most recent
    for line in reversed(log_lines):
        # Check for explicit result
        if 'Result: PASS' in line or 'completed successfully' in line:
            success = True
        elif 'Result: FAIL' in line or 'failed' in line:
            success = False

        # Check for score
        score_match = re.search(r'Score:\s*([\d.]+)', line)
        if score_match:
            try:
                score = float(score_match.group(1))
            except ValueError:
                pass

        # Check for task-specific completion
        if task_id in line:
            if 'success' in line.lower() or 'pass' in line.lower():
                success = True
            elif 'fail' in line.lower() or 'error' in line.lower():
                success = False

    # Default to True if no explicit failure found (backwards compatible)
    if success is None:
        success = True

    return {"success": success, "score": score}

# Usage in _stream_benchmark_updates:
if last_task is not None:
    result = self._parse_task_result(log_lines, last_task)
    complete_data = {
        "task_id": last_task,
        "success": result["success"],
        "score": result["score"],
    }
```

### 6.2 Missing Heartbeat Events

**Problem**: Long-running SSE connections may be closed by proxies/load balancers if no data is sent for extended periods.

**Fix**: Add heartbeat events in `_stream_benchmark_updates()`:

```python
import time

HEARTBEAT_INTERVAL = 30  # seconds

last_heartbeat = time.time()

while True:
    # ... existing event logic ...

    # Send heartbeat every 30 seconds
    current_time = time.time()
    if current_time - last_heartbeat >= HEARTBEAT_INTERVAL:
        if not send_event("heartbeat", {"timestamp": current_time}):
            break
        last_heartbeat = current_time

    time.sleep(interval)
```

### 6.3 Missing Connection Status Events

**Problem**: Frontend has no way to know if backend is still running but has no updates.

**Fix**: Send a "connected" event on initial connection:

```python
def _stream_benchmark_updates(self, interval: int):
    # ... headers ...

    # Send initial connection event
    send_event("connected", {
        "timestamp": time.time(),
        "interval": interval,
        "version": "1.0"
    })

    # ... rest of loop ...
```

---

## 7. Implementation Checklist

### Phase 1: Backend Fixes (local.py)

- [ ] Add `_parse_task_result()` method for log parsing
- [ ] Update `task_complete` event to use parsed success/score
- [ ] Add heartbeat event every 30 seconds
- [ ] Add `connected` event on initial connection
- [ ] Add retry logic for SSH connection failures
- [ ] Add timeout handling for log parsing

### Phase 2: Frontend SSE Manager (benchmark_viewer.py)

- [ ] Add `_get_sse_manager_js()` function
- [ ] Add `initBenchmarkSSE()` with EventSource setup
- [ ] Add event handlers for all 5 event types
- [ ] Add connection status indicator CSS
- [ ] Add error banner CSS and HTML
- [ ] Add task results list CSS

### Phase 3: Replace Polling

- [ ] Update `_get_background_tasks_panel_html()` to use SSE
- [ ] Update `_get_live_evaluation_panel_html()` to use SSE
- [ ] Update `_get_azure_jobs_panel_html()` to use SSE
- [ ] Update `_get_vm_discovery_panel_html()` to use SSE
- [ ] Remove individual `setInterval` calls
- [ ] Add fallback detection for old browsers

### Phase 4: Testing

- [ ] Test SSE connection establishment
- [ ] Test reconnection after disconnect
- [ ] Test fallback to polling in IE11/old browsers
- [ ] Test task_complete events with real WAA runs
- [ ] Test heartbeat prevents proxy timeouts
- [ ] Test page visibility handling (pause on hidden)

### Phase 5: Documentation Updates

- [ ] Update `docs/sse_benchmark_endpoint.md` with new event types
- [ ] Update `docs/sse_quick_reference.md` with heartbeat info
- [ ] Add troubleshooting section for common SSE issues

---

## 8. CSS Additions

Add to shared CSS or `_get_background_tasks_panel_css()`:

```css
/* Connection status indicator */
.connection-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 500;
}

.connection-indicator.connected {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
}

.connection-indicator.connected::before {
    content: '';
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse-connected 2s infinite;
}

.connection-indicator.disconnected {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
}

.connection-indicator.closed {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
}

@keyframes pulse-connected {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Error banner */
.error-banner {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    padding: 12px 20px;
    background: linear-gradient(90deg, #ef4444, #dc2626);
    color: white;
    text-align: center;
    font-size: 0.9rem;
    font-weight: 500;
    z-index: 9999;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

/* Task results list */
.task-results-list {
    max-height: 200px;
    overflow-y: auto;
    margin-top: 12px;
    border-top: 1px solid var(--border-color);
    padding-top: 12px;
}

.task-result {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: 4px;
    margin-bottom: 4px;
    font-size: 0.8rem;
}

.task-result.success {
    background: rgba(16, 185, 129, 0.1);
    border-left: 3px solid #10b981;
}

.task-result.failure {
    background: rgba(239, 68, 68, 0.1);
    border-left: 3px solid #ef4444;
}

.result-icon {
    font-weight: 600;
    font-size: 0.7rem;
    padding: 2px 6px;
    border-radius: 3px;
}

.result-icon.success {
    background: #10b981;
    color: white;
}

.result-icon.failure {
    background: #ef4444;
    color: white;
}

.result-task-id {
    font-family: 'SF Mono', Monaco, monospace;
    color: var(--text-primary);
}

.result-score {
    margin-left: auto;
    color: var(--text-muted);
}

/* Metrics bar */
#task-metrics {
    display: flex;
    gap: 16px;
    padding: 12px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
    margin-top: 12px;
}

.metric {
    font-size: 0.85rem;
    font-weight: 500;
}

.metric.passed { color: #10b981; }
.metric.failed { color: #ef4444; }
.metric.rate { color: #6366f1; margin-left: auto; }
```

---

## 9. Related Files

| File | Purpose |
|------|---------|
| `openadapt_ml/cloud/local.py` | SSE endpoint implementation (lines 1501-1624) |
| `openadapt_ml/training/benchmark_viewer.py` | Frontend polling code to replace |
| `docs/sse_benchmark_endpoint.md` | Full SSE endpoint documentation |
| `docs/sse_quick_reference.md` | Quick reference for event types |
| `docs/sse_architecture.md` | System architecture diagram |

---

## 10. Migration Strategy

### Gradual Rollout

1. **Phase A**: Add SSE manager alongside existing polling
   - SSE runs in parallel with polling
   - Log both sources for comparison
   - No user-visible changes

2. **Phase B**: Enable SSE by default, keep polling as fallback
   - SSE is primary data source
   - Polling activates only on SSE failure
   - Monitor error rates

3. **Phase C**: Remove polling code
   - Delete `setInterval` calls
   - Keep only SSE + feature detection fallback
   - Update documentation

### Rollback Plan

If SSE causes issues, revert by:
1. Set `USE_SSE = false` in config
2. `initBenchmarkUpdates()` will use polling fallback
3. No code changes needed for emergency rollback
