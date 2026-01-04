# SSE Benchmark Endpoint

## Overview

The `/api/benchmark-sse` endpoint provides real-time updates for benchmark status using Server-Sent Events (SSE). This is more efficient than polling for getting live updates during benchmark execution.

## Endpoint

```
GET /api/benchmark-sse?interval=5
```

### Query Parameters

- `interval` (optional, default: 5): Poll interval in seconds (min: 1, max: 60)

### Response Format

SSE stream with the following event types:

#### Event: `status`

VM status and connection information.

```json
{
  "type": "vm_status",
  "connected": true,
  "phase": "ready",
  "waa_ready": true,
  "probe": {
    "status": "ready",
    "vnc_url": "http://20.123.45.67:8006"
  }
}
```

#### Event: `progress`

Benchmark execution progress.

```json
{
  "tasks_completed": 5,
  "total_tasks": 30,
  "current_task": "task_001_notepad",
  "current_step": 12
}
```

#### Event: `task_complete`

Fired when a task finishes.

```json
{
  "task_id": "task_001_notepad",
  "success": true,
  "score": 1.0
}
```

#### Event: `error`

Error messages.

```json
{
  "message": "SSH connection failed"
}
```

## Usage

### JavaScript (Browser)

```javascript
const eventSource = new EventSource('/api/benchmark-sse?interval=5');

eventSource.addEventListener('status', (e) => {
  const data = JSON.parse(e.data);
  console.log('VM Status:', data);
  updateVMStatus(data);
});

eventSource.addEventListener('progress', (e) => {
  const data = JSON.parse(e.data);
  console.log('Progress:', data);
  updateProgressBar(data);
});

eventSource.addEventListener('task_complete', (e) => {
  const data = JSON.parse(e.data);
  console.log('Task completed:', data);
  addTaskResult(data);
});

eventSource.addEventListener('error', (e) => {
  const data = JSON.parse(e.data);
  console.error('Error:', data.message);
});

// Close connection when done
eventSource.close();
```

### Python (requests)

```python
import requests
import json

url = "http://localhost:8765/api/benchmark-sse?interval=5"
response = requests.get(url, stream=True)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')

        if line.startswith('event:'):
            event_type = line[6:].strip()
        elif line.startswith('data:'):
            data = json.loads(line[5:].strip())
            print(f"{event_type}: {data}")
```

### curl

```bash
curl -N http://localhost:8765/api/benchmark-sse?interval=5
```

## How It Works

1. **VM Status Detection**: Checks Azure VM and Docker container status via existing `_fetch_background_tasks()` method

2. **Benchmark Detection**: If VM is ready, SSH into VM and:
   - Check for running process: `docker exec winarena pgrep -f 'python.*run.py'`
   - Parse log file: `tail -100 /tmp/waa_benchmark.log`
   - Extract progress using regex patterns

3. **Event Emission**: Stream events at configured interval

4. **Graceful Degradation**:
   - Falls back to empty data if SSH fails
   - Closes connection cleanly on client disconnect
   - Handles missing log files

## Implementation

The SSE endpoint is implemented in `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/cloud/local.py`:

- `_stream_benchmark_updates()`: Main SSE streaming loop
- `_detect_running_benchmark()`: SSH-based benchmark detection
- `_fetch_background_tasks()`: Existing VM/Docker status checks

## Log Parsing

The endpoint looks for these patterns in `/tmp/waa_benchmark.log`:

- **Task progress**: `Task 5/30`
- **Current task**: `Running task: task_001_notepad`
- **Current step**: `Step 12`

## Testing

Run the test script:

```bash
# Start server (in one terminal)
uv run python -m openadapt_ml.cloud.local serve --port 8765

# Test SSE endpoint (in another terminal)
python test_sse_endpoint.py --interval 5
```

Expected output:
```
[14:23:45] Event: status
{
  "type": "vm_status",
  "connected": true,
  "phase": "ready",
  ...
}
--------------------------------------------------------------------------------
[14:23:50] Event: progress
{
  "tasks_completed": 5,
  "total_tasks": 30,
  ...
}
--------------------------------------------------------------------------------
```

## Frontend Integration (Future)

The frontend can be updated to use SSE instead of polling:

```javascript
// Replace this polling approach
setInterval(() => {
  fetch('/api/tasks').then(r => r.json()).then(updateTasks);
}, 5000);

// With SSE
const eventSource = new EventSource('/api/benchmark-sse?interval=5');
eventSource.addEventListener('status', (e) => {
  updateVMStatus(JSON.parse(e.data));
});
eventSource.addEventListener('progress', (e) => {
  updateProgress(JSON.parse(e.data));
});
```

## Benefits Over Polling

1. **Efficiency**: Server pushes updates only when needed, no empty responses
2. **Real-time**: Lower latency, updates as soon as state changes
3. **Bandwidth**: Reduced HTTP overhead (one connection vs many requests)
4. **Simplicity**: Browser native EventSource API, automatic reconnection

## Fallback Strategy

If SSE is not supported (old browsers, proxies blocking streaming):

```javascript
if (typeof EventSource !== 'undefined') {
  // Use SSE
  const eventSource = new EventSource('/api/benchmark-sse');
  // ...
} else {
  // Fall back to polling
  setInterval(pollBenchmarkStatus, 5000);
}
```

## Limitations

- **SSH Dependency**: Requires SSH access to VM for benchmark detection
- **Log Format**: Assumes specific log patterns (may need updates for different benchmark tools)
- **Connection Timeout**: Long-running connections may be closed by proxies/load balancers
- **Browser Limit**: Most browsers limit to 6 concurrent SSE connections per domain

## Future Enhancements

1. **Task Success Detection**: Parse log output to determine task success/failure
2. **Score Extraction**: Extract score from benchmark output
3. **Multiple Benchmarks**: Support multiple simultaneous benchmark runs
4. **Heartbeat**: Send periodic keepalive events to prevent timeout
5. **Compression**: Enable gzip for SSE stream
6. **Authentication**: Add token-based auth for production use
