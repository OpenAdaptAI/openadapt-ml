# SSE Benchmark Endpoint - Usage Examples

This document provides practical examples for using the SSE benchmark endpoint.

## Quick Start

### 1. Start the Server

```bash
uv run python -m openadapt_ml.cloud.local serve --port 8765
```

### 2. Test with curl

```bash
# Basic connection
curl -N http://localhost:8765/api/benchmark-sse

# With custom interval (2 seconds)
curl -N http://localhost:8765/api/benchmark-sse?interval=2
```

### 3. Test with Python Script

```bash
python test_sse_endpoint.py --interval 5
```

## Example Outputs

### VM Status Event

When the VM is ready:

```
event: status
data: {"type": "vm_status", "connected": true, "phase": "ready", "waa_ready": true, "probe": {"status": "ready", "vnc_url": "http://20.123.45.67:8006"}}
```

When VM is still starting:

```
event: status
data: {"type": "vm_status", "connected": true, "phase": "oobe", "waa_ready": false, "probe": {"status": "Not responding", "vnc_url": "http://20.123.45.67:8006"}}
```

### Progress Event

During benchmark execution:

```
event: progress
data: {"tasks_completed": 5, "total_tasks": 30, "current_task": "task_001_notepad", "current_step": 12}
```

### Task Complete Event

When a task finishes:

```
event: task_complete
data: {"task_id": "task_001_notepad", "success": true, "score": 1.0}
```

### Error Event

When something goes wrong:

```
event: error
data: {"message": "SSH connection failed"}
```

## JavaScript Integration

### Basic Connection

```javascript
// Connect to SSE endpoint
const eventSource = new EventSource('/api/benchmark-sse?interval=5');

// Handle connection open
eventSource.onopen = () => {
  console.log('Connected to benchmark updates');
};

// Handle generic errors
eventSource.onerror = (error) => {
  console.error('SSE connection error:', error);
  eventSource.close();
};
```

### Status Updates

```javascript
eventSource.addEventListener('status', (e) => {
  const status = JSON.parse(e.data);

  // Update VM status indicator
  const statusEl = document.getElementById('vm-status');
  if (status.waa_ready) {
    statusEl.className = 'status-ready';
    statusEl.textContent = 'VM Ready';
  } else if (status.connected) {
    statusEl.className = 'status-starting';
    statusEl.textContent = `Starting (${status.phase})`;
  } else {
    statusEl.className = 'status-offline';
    statusEl.textContent = 'Offline';
  }

  // Show VNC link if available
  if (status.probe.vnc_url) {
    document.getElementById('vnc-link').href = status.probe.vnc_url;
    document.getElementById('vnc-link').style.display = 'block';
  }
});
```

### Progress Updates

```javascript
eventSource.addEventListener('progress', (e) => {
  const progress = JSON.parse(e.data);

  // Update progress bar
  const percent = (progress.tasks_completed / progress.total_tasks) * 100;
  document.getElementById('progress-bar').style.width = `${percent}%`;

  // Update task counter
  document.getElementById('task-counter').textContent =
    `${progress.tasks_completed} / ${progress.total_tasks} tasks`;

  // Show current task
  document.getElementById('current-task').textContent =
    `Task: ${progress.current_task} (Step ${progress.current_step})`;
});
```

### Task Completion

```javascript
eventSource.addEventListener('task_complete', (e) => {
  const result = JSON.parse(e.data);

  // Add to results table
  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${result.task_id}</td>
    <td>${result.success ? '✓' : '✗'}</td>
    <td>${result.score || 'N/A'}</td>
  `;
  document.getElementById('results-table').appendChild(row);
});
```

### Complete Example

```html
<!DOCTYPE html>
<html>
<head>
  <title>Benchmark Monitor</title>
  <style>
    .status-ready { color: green; }
    .status-starting { color: orange; }
    .status-offline { color: red; }
    .progress-bar {
      width: 0%;
      height: 20px;
      background: green;
      transition: width 0.3s;
    }
  </style>
</head>
<body>
  <h1>Benchmark Monitor</h1>

  <div id="status-panel">
    <h2>VM Status: <span id="vm-status">Connecting...</span></h2>
    <a id="vnc-link" target="_blank" style="display: none;">Open VNC</a>
  </div>

  <div id="progress-panel">
    <h2>Progress</h2>
    <div id="progress-bar" class="progress-bar"></div>
    <p id="task-counter">0 / 0 tasks</p>
    <p id="current-task">Idle</p>
  </div>

  <div id="results-panel">
    <h2>Results</h2>
    <table id="results-table">
      <thead>
        <tr>
          <th>Task ID</th>
          <th>Success</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

  <script>
    const eventSource = new EventSource('/api/benchmark-sse?interval=5');

    eventSource.addEventListener('status', (e) => {
      const status = JSON.parse(e.data);
      const statusEl = document.getElementById('vm-status');

      if (status.waa_ready) {
        statusEl.className = 'status-ready';
        statusEl.textContent = 'Ready';
      } else if (status.connected) {
        statusEl.className = 'status-starting';
        statusEl.textContent = `Starting (${status.phase})`;
      } else {
        statusEl.className = 'status-offline';
        statusEl.textContent = 'Offline';
      }

      if (status.probe.vnc_url) {
        document.getElementById('vnc-link').href = status.probe.vnc_url;
        document.getElementById('vnc-link').style.display = 'block';
      }
    });

    eventSource.addEventListener('progress', (e) => {
      const progress = JSON.parse(e.data);
      const percent = (progress.tasks_completed / progress.total_tasks) * 100;

      document.getElementById('progress-bar').style.width = `${percent}%`;
      document.getElementById('task-counter').textContent =
        `${progress.tasks_completed} / ${progress.total_tasks} tasks`;
      document.getElementById('current-task').textContent =
        `Task: ${progress.current_task} (Step ${progress.current_step})`;
    });

    eventSource.addEventListener('task_complete', (e) => {
      const result = JSON.parse(e.data);
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${result.task_id}</td>
        <td>${result.success ? '✓' : '✗'}</td>
        <td>${result.score || 'N/A'}</td>
      `;
      document.querySelector('#results-table tbody').appendChild(row);
    });

    eventSource.addEventListener('error', (e) => {
      const error = JSON.parse(e.data);
      console.error('Benchmark error:', error.message);
    });

    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
      eventSource.close();
    });
  </script>
</body>
</html>
```

## Python Client Example

### Basic Client

```python
import requests
import json
from datetime import datetime


class BenchmarkMonitor:
    def __init__(self, base_url: str = "http://localhost:8765", interval: int = 5):
        self.url = f"{base_url}/api/benchmark-sse?interval={interval}"
        self.running = False

    def start(self):
        """Start monitoring benchmark updates."""
        self.running = True
        response = requests.get(self.url, stream=True, timeout=None)

        event_type = None
        for line in response.iter_lines():
            if not self.running:
                break

            if line:
                line = line.decode('utf-8')

                if line.startswith('event:'):
                    event_type = line[6:].strip()
                elif line.startswith('data:'):
                    data = json.loads(line[5:].strip())
                    self._handle_event(event_type, data)

    def _handle_event(self, event_type: str, data: dict):
        """Handle incoming SSE event."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if event_type == 'status':
            self._on_status(timestamp, data)
        elif event_type == 'progress':
            self._on_progress(timestamp, data)
        elif event_type == 'task_complete':
            self._on_task_complete(timestamp, data)
        elif event_type == 'error':
            self._on_error(timestamp, data)

    def _on_status(self, timestamp: str, data: dict):
        """Handle VM status event."""
        status = "Ready" if data['waa_ready'] else f"Starting ({data['phase']})"
        print(f"[{timestamp}] VM Status: {status}")

    def _on_progress(self, timestamp: str, data: dict):
        """Handle progress event."""
        completed = data['tasks_completed']
        total = data['total_tasks']
        task = data['current_task']
        step = data['current_step']
        print(f"[{timestamp}] Progress: {completed}/{total} tasks | Current: {task} (step {step})")

    def _on_task_complete(self, timestamp: str, data: dict):
        """Handle task completion event."""
        task_id = data['task_id']
        success = data['success']
        score = data.get('score', 'N/A')
        print(f"[{timestamp}] Task Complete: {task_id} | Success: {success} | Score: {score}")

    def _on_error(self, timestamp: str, data: dict):
        """Handle error event."""
        print(f"[{timestamp}] ERROR: {data['message']}")

    def stop(self):
        """Stop monitoring."""
        self.running = False


if __name__ == "__main__":
    monitor = BenchmarkMonitor(interval=5)
    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\nStopped by user")
        monitor.stop()
```

### Advanced Client with Callbacks

```python
from typing import Callable, Optional
import requests
import json


class BenchmarkClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        interval: int = 5,
        on_status: Optional[Callable] = None,
        on_progress: Optional[Callable] = None,
        on_task_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        self.url = f"{base_url}/api/benchmark-sse?interval={interval}"
        self.on_status = on_status
        self.on_progress = on_progress
        self.on_task_complete = on_task_complete
        self.on_error = on_error

    def connect(self):
        """Connect and start receiving events."""
        response = requests.get(self.url, stream=True, timeout=None)

        event_type = None
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')

                if line.startswith('event:'):
                    event_type = line[6:].strip()
                elif line.startswith('data:'):
                    data = json.loads(line[5:].strip())

                    # Call appropriate callback
                    if event_type == 'status' and self.on_status:
                        self.on_status(data)
                    elif event_type == 'progress' and self.on_progress:
                        self.on_progress(data)
                    elif event_type == 'task_complete' and self.on_task_complete:
                        self.on_task_complete(data)
                    elif event_type == 'error' and self.on_error:
                        self.on_error(data)


# Usage example
def handle_status(data):
    print(f"VM is {'ready' if data['waa_ready'] else 'starting'}")

def handle_progress(data):
    print(f"Progress: {data['tasks_completed']}/{data['total_tasks']}")

def handle_completion(data):
    print(f"Task {data['task_id']} completed: {data['success']}")

def handle_error(data):
    print(f"Error: {data['message']}")


client = BenchmarkClient(
    interval=5,
    on_status=handle_status,
    on_progress=handle_progress,
    on_task_complete=handle_completion,
    on_error=handle_error,
)

try:
    client.connect()
except KeyboardInterrupt:
    print("Disconnected")
```

## Troubleshooting

### Connection Refused

If you get "Connection refused":

1. Check server is running: `curl http://localhost:8765/api/tasks`
2. Check port matches: `--port 8765`
3. Check firewall settings

### No Events Received

If connected but no events:

1. Check VM is running: `az vm get-instance-view --name waa-eval-vm ...`
2. Check interval is reasonable: `?interval=5` (not too high)
3. Check server logs for errors

### SSH Connection Failed

If you see "SSH connection failed" errors:

1. Verify SSH key exists: `ls ~/.ssh/id_rsa`
2. Test SSH manually: `ssh azureuser@<vm-ip>`
3. Check VM security group allows SSH (port 22)

### Events Stop After Some Time

If events stop coming:

1. Check connection timeout (proxies may close long connections)
2. Reconnect automatically on error
3. Use shorter intervals to keep connection alive

### Browser Compatibility

If EventSource not working:

```javascript
if (typeof EventSource === 'undefined') {
  alert('Your browser does not support Server-Sent Events');
  // Fall back to polling
}
```

## Performance Tips

1. **Use appropriate intervals**: 5-10s for most cases, 2-3s for fast updates
2. **Close connections**: Always call `eventSource.close()` when done
3. **Handle reconnection**: Browser will auto-reconnect on network errors
4. **Limit event handlers**: Don't update DOM too frequently (use throttling)

## Security Considerations

For production use:

1. **Add authentication**: Check API keys in SSE endpoint
2. **Rate limiting**: Prevent abuse by limiting connections per IP
3. **HTTPS**: Always use TLS in production
4. **CORS**: Configure proper CORS headers for allowed origins
