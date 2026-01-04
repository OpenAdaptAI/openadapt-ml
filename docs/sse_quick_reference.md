# SSE Benchmark Endpoint - Quick Reference

## Endpoint

```
GET /api/benchmark-sse?interval=5
```

## Query Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `interval` | int | 5 | 1-60 | Poll interval in seconds |

## Event Types

| Event | Description | Frequency |
|-------|-------------|-----------|
| `status` | VM status and probe | Every interval |
| `progress` | Benchmark progress | When benchmark running |
| `task_complete` | Task finished | When task completes |
| `error` | Error message | When error occurs |

## Event Schemas

### status
```json
{
  "type": "vm_status",
  "connected": boolean,
  "phase": "ready" | "oobe" | "booting" | "building" | "unknown",
  "waa_ready": boolean,
  "probe": {
    "status": string,
    "vnc_url": string
  }
}
```

### progress
```json
{
  "tasks_completed": int,
  "total_tasks": int,
  "current_task": string,
  "current_step": int
}
```

### task_complete
```json
{
  "task_id": string,
  "success": boolean,
  "score": float | null
}
```

### error
```json
{
  "message": string
}
```

## JavaScript Usage

```javascript
const eventSource = new EventSource('/api/benchmark-sse?interval=5');

eventSource.addEventListener('status', (e) => {
  const data = JSON.parse(e.data);
  // Handle VM status
});

eventSource.addEventListener('progress', (e) => {
  const data = JSON.parse(e.data);
  // Update progress bar
});

eventSource.close(); // When done
```

## Python Usage

```python
import requests, json

response = requests.get('http://localhost:8765/api/benchmark-sse?interval=5', stream=True)
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('event:'):
            event_type = line[6:].strip()
        elif line.startswith('data:'):
            data = json.loads(line[5:].strip())
            print(f"{event_type}: {data}")
```

## curl Testing

```bash
# Basic connection
curl -N http://localhost:8765/api/benchmark-sse

# With custom interval
curl -N http://localhost:8765/api/benchmark-sse?interval=2
```

## Test Script

```bash
# Start server
uv run python -m openadapt_ml.cloud.local serve --port 8765

# Test SSE
python test_sse_endpoint.py --interval 5
```

## Implementation Details

**File**: `openadapt_ml/cloud/local.py`

**Methods**:
- `_stream_benchmark_updates(interval)` - SSE streaming loop
- `_detect_running_benchmark(vm_ip, container)` - SSH-based detection

**Dependencies**:
- Existing `_fetch_background_tasks()` method
- SSH access to Azure VM
- Benchmark log at `/tmp/waa_benchmark.log`

## Log Patterns

The endpoint parses these patterns from `/tmp/waa_benchmark.log`:

| Pattern | Regex | Extracts |
|---------|-------|----------|
| Task progress | `Task\s+(\d+)/(\d+)` | tasks_completed, total_tasks |
| Current task | `(?:Running\|Processing) task:\s*(\S+)` | current_task |
| Current step | `Step\s+(\d+)` | current_step |

## Response Headers

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Access-Control-Allow-Origin: *
Connection: keep-alive
```

## Error Handling

| Error | Cause | Behavior |
|-------|-------|----------|
| SSH timeout | VM unreachable | Returns empty progress, continues |
| Missing log | Benchmark not started | Returns empty progress, continues |
| Client disconnect | Browser closed | Breaks loop, closes connection |
| Invalid interval | Out of range (1-60) | Clamped to valid range |

## Performance

| Metric | Value |
|--------|-------|
| Overhead per event | ~200 bytes |
| Latency | ~interval seconds |
| Connections per domain | Browser limited (usually 6) |
| Memory per connection | ~1KB |

## Comparison: SSE vs Polling

| Aspect | SSE | Polling |
|--------|-----|---------|
| Requests/min (5s interval) | 1 | 12 |
| Empty responses | 0 | Most |
| Latency | ~interval | ~interval |
| Browser overhead | Low | High |
| Server overhead | Low | Medium |
| Real-time capability | High | Medium |

## Troubleshooting

| Issue | Check | Fix |
|-------|-------|-----|
| No connection | Server running? | Start server |
| No events | VM running? | Check Azure VM status |
| SSH errors | SSH key exists? | Verify `~/.ssh/id_rsa` |
| Disconnects | Proxy timeout? | Use shorter interval |

## Browser Support

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome | ✓ | Full support |
| Firefox | ✓ | Full support |
| Safari | ✓ | Full support |
| Edge | ✓ | Full support |
| IE 11 | ✗ | Use polyfill or polling |

## Security Checklist

- [ ] HTTPS in production
- [ ] API key authentication
- [ ] Rate limiting per IP
- [ ] CORS whitelist (not `*`)
- [ ] Input validation (interval param)
- [ ] SSH key permissions (600)

## Quick Debugging

```bash
# Check server endpoint exists
curl -I http://localhost:8765/api/benchmark-sse

# Check VM is reachable
ssh azureuser@<vm-ip> 'echo OK'

# Check log file exists
ssh azureuser@<vm-ip> 'ls -la /tmp/waa_benchmark.log'

# Check process running
ssh azureuser@<vm-ip> 'docker exec winarena pgrep -f python.*run.py'

# Tail log manually
ssh azureuser@<vm-ip> 'tail -f /tmp/waa_benchmark.log'
```

## Files

| File | Purpose |
|------|---------|
| `openadapt_ml/cloud/local.py` | SSE endpoint implementation |
| `test_sse_endpoint.py` | Test client script |
| `docs/sse_benchmark_endpoint.md` | Full documentation |
| `docs/sse_usage_examples.md` | Usage examples |
| `SSE_IMPLEMENTATION_SUMMARY.md` | Implementation summary |

## Next Steps

1. Test with real VM and benchmark
2. Update frontend to use SSE (replace polling)
3. Add task success/failure detection
4. Implement authentication
5. Add monitoring/metrics

## Related Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/tasks` | GET | Background tasks (polling) |
| `/api/benchmark-progress` | GET | Local benchmark progress (polling) |
| `/api/benchmark-live` | GET | Live evaluation state (polling) |
| `/api/vms` | GET | VM registry status (polling) |
| `/api/azure-jobs` | GET | Azure ML jobs (polling) |

**Note**: SSE endpoint can replace all of the above for real-time monitoring.
