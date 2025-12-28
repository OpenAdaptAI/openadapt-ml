# Background Task Visibility Design

## Overview

This document describes the Background Task Visibility feature, which provides users with real-time visibility into long-running background tasks within the benchmark viewer. Currently, users lack insight into operations like Docker image pulls, VM provisioning, and benchmark evaluations that can take minutes to hours.

## Problem Statement

Users experience uncertainty and frustration when:
- Waiting for Docker images to pull (5-15 minutes)
- VM provisioning in Azure (3-10 minutes)
- Benchmark evaluations running (30-120+ minutes for full WAA suite)
- Model training on cloud GPUs (20-60 minutes per run)

Without visibility, users don't know:
- If their task is still running or stuck
- How much longer they need to wait
- Whether errors have occurred
- What operation is currently executing

The existing Azure Jobs panel in the benchmark viewer shows only Azure ML job status, but doesn't capture:
- Local Docker operations
- Pre-Azure setup tasks (ACR pulls, workspace initialization)
- Benchmark runner coordination logic
- Model inference during evaluation

## Proposed Solution

Add a unified "Tasks" panel to the benchmark viewer that displays all active background tasks with real-time status updates, progress indicators, and completion estimates.

### Key Features

1. **Unified Task View**: Single panel showing all background operations
2. **Real-time Updates**: Live status via API polling (10-second intervals)
3. **Progress Tracking**: Percentage complete, elapsed time, ETA
4. **Task History**: Recently completed tasks with success/failure status
5. **Expandable Details**: Click task to see logs, substeps, error messages
6. **Visual Indicators**: Color-coded status badges, pulsing animations for active tasks

## Task Types to Track

### 1. Docker Operations
- **Image Pull**: Downloading WAA Docker image from ACR/Docker Hub
  - Progress: Layer download percentage
  - Metadata: Image size, layers remaining
  - Typical duration: 5-15 minutes

### 2. VM Provisioning
- **Azure ML Compute**: Creating/starting compute instance
  - Progress: Provisioning → Starting → Running
  - Metadata: VM size, region, estimated cost/hour
  - Typical duration: 3-10 minutes

### 3. Benchmark Runs
- **WAA Evaluation**: Running full benchmark suite
  - Progress: Tasks completed / total tasks
  - Metadata: Model ID, worker count, success rate so far
  - Typical duration: 30-120 minutes
  - Substeps: Individual task execution with pass/fail

### 4. Model Training
- **Cloud GPU Training**: Lambda Labs or Azure ML training jobs
  - Progress: Current epoch / total epochs
  - Metadata: Loss, instance type, GPU utilization
  - Typical duration: 20-60 minutes

### 5. Data Operations
- **Upload/Download**: Rsync of captures, checkpoints, results
  - Progress: Bytes transferred / total bytes
  - Metadata: Transfer speed, files remaining
  - Typical duration: 1-10 minutes

## Data Model

### Core Schema

```python
@dataclass
class BackgroundTask:
    """Represents a long-running background operation."""
    task_id: str  # UUID for tracking
    task_type: str  # "docker_pull" | "vm_provision" | "benchmark_run" | "training" | "data_transfer"
    status: str  # "pending" | "running" | "completed" | "failed" | "cancelled"

    # Display information
    title: str  # Short human-readable name
    description: str  # Detailed description

    # Progress tracking
    progress_percent: float  # 0.0 to 100.0
    elapsed_seconds: float
    estimated_total_seconds: Optional[float]

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    # Task-specific metadata
    metadata: Dict[str, Any]  # Flexible storage for task-specific data

    # Substeps (optional, for multi-stage tasks)
    substeps: List[TaskSubstep]

    # Error handling
    error_message: Optional[str]
    retry_count: int
    max_retries: int

    # Logs
    log_lines: List[str]  # Recent log output
    log_file_path: Optional[Path]

@dataclass
class TaskSubstep:
    """Represents a substep within a task."""
    step_id: str
    name: str
    status: str
    progress_percent: float
    error_message: Optional[str]
```

### Example Task Instances

**Docker Pull:**
```python
BackgroundTask(
    task_id="docker-pull-waa-001",
    task_type="docker_pull",
    status="running",
    title="Pulling WAA Docker Image",
    description="Downloading openadaptacr.azurecr.io/winarena:latest",
    progress_percent=45.2,
    elapsed_seconds=180.5,
    estimated_total_seconds=420.0,
    metadata={
        "image_name": "winarena:latest",
        "total_layers": 24,
        "completed_layers": 11,
        "total_size_mb": 8500,
        "downloaded_mb": 3842
    }
)
```

**Benchmark Run:**
```python
BackgroundTask(
    task_id="waa-eval-20241217-001",
    task_type="benchmark_run",
    status="running",
    title="WAA Benchmark Evaluation",
    description="Running 154 tasks with qwen3vl-2b-epoch5",
    progress_percent=32.5,
    elapsed_seconds=1840.0,
    estimated_total_seconds=5200.0,
    metadata={
        "model_id": "qwen3vl-2b-epoch5",
        "total_tasks": 154,
        "completed_tasks": 50,
        "successful_tasks": 38,
        "failed_tasks": 12,
        "workers": 4,
        "success_rate": 0.76
    },
    substeps=[
        TaskSubstep(
            step_id="task-001",
            name="Open Notepad and type text",
            status="completed",
            progress_percent=100.0
        ),
        TaskSubstep(
            step_id="task-002",
            name="Navigate to Settings",
            status="failed",
            error_message="Timeout waiting for Settings window"
        ),
        TaskSubstep(
            step_id="task-003",
            name="Create new folder",
            status="running",
            progress_percent=60.0
        )
    ]
)
```

## UI Design

### Tasks Panel Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Background Tasks                      🔄 Auto-refresh: 10s │
├─────────────────────────────────────────────────────────────┤
│  ● Running (2)    ✓ Completed (5)    ✗ Failed (1)          │
├─────────────────────────────────────────────────────────────┤
│  ┌─ Docker Pull: WAA Image ─────────────────────────┐      │
│  │  [████████████░░░░░░░░░░░░] 45%  •  3m 0s / 7m 0s  │      │
│  │  ↓ 3.8 GB / 8.5 GB  •  11/24 layers                │      │
│  │  [▼ Show Details]                                  │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
│  ┌─ WAA Benchmark: qwen3vl-2b-epoch5 ───────────────┐      │
│  │  [████████░░░░░░░░░░░░░░░░] 32%  •  30m 40s / 1h 26m │   │
│  │  ✓ 38  ✗ 12  ⏱ 1 / 154 tasks  •  76% success      │      │
│  │  [▼ Show Details]                                  │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
│  ┌─ Training: turn-off-nightshift ──────────────────┐      │
│  │  ✓ Completed  •  23m 15s                          │      │
│  │  Final loss: 0.12  •  4 epochs                     │      │
│  └────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Expanded Task View

```
┌─ WAA Benchmark: qwen3vl-2b-epoch5 ───────────────────────┐
│  [████████░░░░░░░░░░░░░░░░] 32%  •  30m 40s / 1h 26m    │
│  ✓ 38  ✗ 12  ⏱ 1 / 154 tasks  •  76% success            │
│                                                           │
│  [▲ Hide Details]                                        │
│                                                           │
│  Recent Activity:                                         │
│  ✓ task-048: "Copy files between folders" (8.2s)        │
│  ✓ task-049: "Set desktop wallpaper" (12.1s)            │
│  ⏱ task-050: "Install program from web" (running 45s)   │
│                                                           │
│  Logs (last 10 lines):                                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │ [14:32:15] Starting task-050: Install program      │  │
│  │ [14:32:18] Screenshot captured (step 1)            │  │
│  │ [14:32:22] Action: CLICK(x=0.42, y=0.31)           │  │
│  │ [14:32:28] Screenshot captured (step 2)            │  │
│  │ [14:32:31] Action: TYPE("example.exe")             │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  [View Full Logs]  [Cancel Task]                         │
└───────────────────────────────────────────────────────────┘
```

### Integration with Existing Viewer

The Tasks panel appears:
1. **In Benchmark Viewer**: Top of page, before Azure Jobs panel
2. **In Training Dashboard**: Sidebar or collapsible panel
3. **Standalone**: Accessible via `/api/tasks` endpoint for external monitoring

### Visual Design Elements

**Status Indicators:**
- Pending: ⏸ Gray circle
- Running: ● Pulsing blue circle
- Completed: ✓ Green checkmark
- Failed: ✗ Red X
- Cancelled: ⏹ Orange square

**Progress Bars:**
- Active tasks: Animated gradient (blue → cyan)
- Completed: Solid green
- Failed: Solid red at failure point
- Indeterminate: Animated stripes (when % unknown)

**Time Display:**
- Elapsed: "3m 20s" format
- ETA: "~15m remaining"
- Completion: "Finished at 14:35:22"

## Implementation Approach

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Benchmark Viewer                       │
│  (HTML/CSS/JS in benchmark.html)                         │
│                                                           │
│  JavaScript polls /api/tasks every 10 seconds            │
│  Renders task cards with progress bars                   │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTP GET
                  ↓
┌─────────────────────────────────────────────────────────┐
│          Local HTTP Server (local.py)                    │
│                                                           │
│  GET /api/tasks → task_manager.get_active_tasks()       │
│  POST /api/tasks/{id}/cancel → task_manager.cancel()    │
└─────────────────┬───────────────────────────────────────┘
                  │ Python API
                  ↓
┌─────────────────────────────────────────────────────────┐
│         TaskManager (new module)                         │
│  openadapt_ml/tasks/manager.py                          │
│                                                           │
│  - In-memory task registry (dict by task_id)             │
│  - Writes to tasks.json for persistence                  │
│  - Thread-safe with locks                                │
│  - Garbage collection of old tasks                       │
└─────────────────┬───────────────────────────────────────┘
                  │ Used by
                  ↓
┌─────────────────────────────────────────────────────────┐
│       Background Operation Modules                       │
│                                                           │
│  - benchmarks/runner.py → registers benchmark tasks     │
│  - cloud/lambda_labs.py → registers training tasks      │
│  - cloud/azure.py → registers Azure tasks               │
│  - (future) docker client → registers pull tasks        │
└─────────────────────────────────────────────────────────┘
```

### File Structure

```
openadapt_ml/
├── tasks/
│   ├── __init__.py
│   ├── manager.py       # TaskManager singleton
│   ├── models.py        # BackgroundTask, TaskSubstep dataclasses
│   └── tracker.py       # ProgressTracker helper (context manager)
├── benchmarks/
│   └── runner.py        # Modified to register tasks
├── cloud/
│   ├── local.py         # Add /api/tasks endpoints
│   ├── lambda_labs.py   # Register training tasks
│   └── azure.py         # Register Azure tasks
└── training/
    └── benchmark_viewer.py  # Add tasks panel HTML/CSS/JS
```

### Task State Storage

**In-Memory (Primary):**
- `TaskManager._tasks: Dict[str, BackgroundTask]`
- Fast access for API queries
- Thread-safe with `threading.Lock`

**Persistent (Backup):**
- `training_output/tasks.json` written on task updates
- Used for recovery if server restarts
- Pruned to keep only recent tasks (last 24 hours)

**Why not Database?**
- Lightweight: No dependencies on SQLite/Postgres
- Simplicity: JSON serialization is straightforward
- Portability: Works in Docker, cloud, local environments
- Scale: Max ~100 concurrent tasks, JSON is sufficient

### Progress Tracking API

```python
from openadapt_ml.tasks import task_manager, ProgressTracker

# Context manager for automatic registration
with ProgressTracker(
    task_type="docker_pull",
    title="Pulling WAA Image",
    description="Downloading winarena:latest from ACR"
) as tracker:

    # Update progress (0.0-100.0)
    tracker.update(progress=25.0, metadata={"layers": "6/24"})

    # Add log lines
    tracker.log("Started layer download: sha256:abc123...")

    # Report substep progress
    tracker.add_substep("layer-1", "Download base image")
    tracker.update_substep("layer-1", progress=100.0, status="completed")

    # Automatic completion on exit
    # Or explicit: tracker.complete()
    # Or explicit: tracker.fail("Error message")
```

### Integration Points

**1. Benchmark Runner** (`benchmarks/runner.py`)

```python
def evaluate_agent_on_benchmark(
    agent: BenchmarkAgent,
    adapter: BenchmarkAdapter,
    num_tasks: int = None
) -> BenchmarkResults:

    # Register task
    tracker = task_manager.create_task(
        task_type="benchmark_run",
        title=f"WAA Evaluation: {agent.model_id}",
        metadata={
            "model_id": agent.model_id,
            "total_tasks": num_tasks or len(tasks)
        }
    )

    try:
        for i, task in enumerate(tasks):
            # Update progress
            tracker.update(
                progress=(i / len(tasks)) * 100,
                metadata={
                    "completed_tasks": i,
                    "successful_tasks": sum(1 for r in results if r.success)
                }
            )

            # Add substep for each task
            tracker.add_substep(task.task_id, task.instruction)
            result = agent.execute_task(task, adapter)
            tracker.update_substep(
                task.task_id,
                status="completed" if result.success else "failed"
            )

        tracker.complete()
    except Exception as e:
        tracker.fail(str(e))
        raise
```

**2. Azure Orchestrator** (`cloud/azure.py`)

```python
def submit_benchmark_job(self, ...):
    # Register VM provisioning task
    vm_tracker = task_manager.create_task(
        task_type="vm_provision",
        title=f"Provisioning Azure VM: {vm_size}",
        metadata={"vm_size": vm_size, "region": region}
    )

    vm_tracker.update(progress=10.0, status="creating")
    job = ml_client.jobs.create_or_update(job_config)
    vm_tracker.update(progress=50.0, status="starting")

    # Poll until running
    while job.status != "Running":
        time.sleep(10)
        job = ml_client.jobs.get(job.name)

    vm_tracker.complete(metadata={"job_id": job.name})
```

**3. Local Server** (`cloud/local.py`)

```python
@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    """Return all active and recent tasks."""
    from openadapt_ml.tasks import task_manager

    tasks = task_manager.get_tasks(
        include_completed=True,
        max_age_hours=24
    )

    return jsonify([task.to_dict() for task in tasks])

@app.route("/api/tasks/<task_id>/cancel", methods=["POST"])
def cancel_task(task_id: str):
    """Request task cancellation."""
    from openadapt_ml.tasks import task_manager

    success = task_manager.cancel_task(task_id)
    return jsonify({"success": success})
```

**4. Benchmark Viewer** (`training/benchmark_viewer.py`)

Add new function:

```python
def _get_background_tasks_panel_css() -> str:
    """Return CSS for background tasks panel."""
    return '''
        .tasks-panel {
            background: linear-gradient(135deg, rgba(100, 100, 255, 0.1) 0%, rgba(100, 100, 255, 0.05) 100%);
            border: 1px solid rgba(100, 100, 255, 0.3);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
        }

        .task-card {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }

        .task-progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }

        .task-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #06b6d4);
            transition: width 0.5s ease;
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% { opacity: 0.8; }
            50% { opacity: 1.0; }
            100% { opacity: 0.8; }
        }

        .task-status-running {
            color: #3b82f6;
            animation: pulse 2s infinite;
        }

        .task-status-completed {
            color: #10b981;
        }

        .task-status-failed {
            color: #ef4444;
        }
    '''

def _get_background_tasks_panel_html() -> str:
    """Return HTML for background tasks panel with polling."""
    return '''
    <div class="tasks-panel" id="tasks-panel">
        <div class="tasks-header">
            <h3>Background Tasks</h3>
            <span id="tasks-refresh-time">Checking...</span>
        </div>
        <div id="tasks-list">
            <div class="no-tasks">No active tasks</div>
        </div>
    </div>

    <script>
        async function fetchBackgroundTasks() {
            try {
                const response = await fetch('/api/tasks?' + Date.now());
                if (response.ok) {
                    const tasks = await response.json();
                    renderBackgroundTasks(tasks);
                    document.getElementById('tasks-refresh-time').textContent =
                        'Updated ' + new Date().toLocaleTimeString();
                }
            } catch (e) {
                console.log('Tasks API unavailable:', e);
            }
        }

        function renderBackgroundTasks(tasks) {
            const container = document.getElementById('tasks-list');

            if (!tasks || tasks.length === 0) {
                container.innerHTML = '<div class="no-tasks">No active tasks</div>';
                return;
            }

            const html = tasks.map(task => {
                const statusClass = `task-status-${task.status}`;
                const progressPercent = task.progress_percent || 0;

                const elapsedText = formatDuration(task.elapsed_seconds);
                const etaText = task.estimated_total_seconds
                    ? `~${formatDuration(task.estimated_total_seconds - task.elapsed_seconds)} remaining`
                    : '';

                return `
                    <div class="task-card">
                        <div class="task-header">
                            <span class="${statusClass}">${getStatusIcon(task.status)}</span>
                            <strong>${task.title}</strong>
                        </div>
                        <div class="task-description">${task.description}</div>
                        <div class="task-progress-bar">
                            <div class="task-progress-fill" style="width: ${progressPercent}%"></div>
                        </div>
                        <div class="task-meta">
                            ${progressPercent.toFixed(1)}% • ${elapsedText} ${etaText}
                        </div>
                    </div>
                `;
            }).join('');

            container.innerHTML = html;
        }

        function getStatusIcon(status) {
            const icons = {
                'pending': '⏸',
                'running': '●',
                'completed': '✓',
                'failed': '✗',
                'cancelled': '⏹'
            };
            return icons[status] || '?';
        }

        function formatDuration(seconds) {
            if (!seconds) return '0s';
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
        }

        // Poll every 10 seconds
        fetchBackgroundTasks();
        setInterval(fetchBackgroundTasks, 10000);
    </script>
    '''
```

## Technical Considerations

### Thread Safety

**Issue**: Multiple threads may update task state concurrently
- Benchmark runner in main thread
- HTTP server in Flask thread
- Azure polling in background thread

**Solution**: Use `threading.Lock` in TaskManager

```python
class TaskManager:
    def __init__(self):
        self._tasks = {}
        self._lock = threading.Lock()

    def update_task(self, task_id: str, **kwargs):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                for key, value in kwargs.items():
                    setattr(task, key, value)
                self._persist()
```

### Performance

**Polling Frequency**: 10-second intervals balances:
- Responsiveness: Users see updates within 10 seconds
- Server load: ~6 requests/minute is negligible
- Network: <1 KB per request

**Task Pruning**: Automatically remove old completed tasks
- Keep completed tasks for 24 hours
- Keep failed tasks for 7 days (for debugging)
- Limits in-memory size to ~100 tasks

**Pagination** (future): If >50 tasks, paginate API response

### Error Handling

**Task Failures:**
- Capture exception message in `task.error_message`
- Set `status = "failed"`
- Keep task in history for debugging

**Server Restart:**
- Load tasks from `tasks.json` on startup
- Mark all "running" tasks as "cancelled" (server died mid-execution)

**Network Errors:**
- Frontend shows "Last updated X seconds ago"
- Retries with exponential backoff

### Security

**Task Cancellation:**
- Only allow cancellation of tasks owned by current session
- Add `owner_id` field if multi-user support needed
- For now: Local-only server = implicit auth

**Log Exposure:**
- Sanitize sensitive data (API keys, passwords) from logs
- Truncate logs to last 100 lines in API response
- Link to full log file for detailed inspection

## Migration Path

### Phase 1: Core Infrastructure (Week 1)
1. Create `tasks/` module with models and manager
2. Add `/api/tasks` endpoint to local server
3. Write unit tests for TaskManager

### Phase 2: UI Integration (Week 1)
1. Add tasks panel to benchmark viewer
2. Implement polling and rendering logic
3. Test with mock tasks

### Phase 3: Background Operation Integration (Week 2)
1. Integrate with benchmark runner
2. Integrate with Azure orchestrator
3. Integrate with Lambda Labs training

### Phase 4: Polish & Features (Week 2)
1. Add substep tracking
2. Add log viewing
3. Add task cancellation
4. Performance testing with 50+ concurrent tasks

## Future Enhancements

1. **WebSocket Support**: Real-time push instead of polling
2. **Historical Analytics**: Task duration trends, failure rates over time
3. **Notifications**: Browser notifications when tasks complete
4. **Mobile View**: Responsive design for phone monitoring
5. **Multi-user**: Task ownership and permissions
6. **External Monitoring**: Standalone dashboard page at `/tasks`
7. **Prometheus Metrics**: Export task stats for Grafana dashboards

## Success Metrics

**User Experience:**
- Users can see Docker pull progress within 10 seconds of starting
- Users know how long benchmark will take (ETA accuracy ±20%)
- Users can identify stuck tasks and cancel them

**Technical:**
- API responds in <100ms for task list
- UI renders 50+ tasks without lag
- Zero task state corruption after server restart

**Adoption:**
- 90% of long-running operations (>1 minute) are tracked
- Users report reduced frustration with waiting times
- Support tickets about "is it stuck?" decrease by 50%

## References

- Existing Azure Jobs panel: `openadapt_ml/training/benchmark_viewer.py` lines 13-326
- Similar feature in GitHub Actions: Real-time job logs with expandable steps
- Similar feature in Docker Desktop: Container logs with auto-scroll
- Similar feature in Kubernetes Dashboard: Pod status with event stream
