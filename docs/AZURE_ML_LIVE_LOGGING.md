# Azure ML Live Logging Design Document

## Problem Statement

When running WAA benchmark evaluations on Azure ML compute instances, we need to see real-time logs from inside the Docker container to:
1. Verify patches are being applied correctly (e.g., IP address fix)
2. Monitor Windows VM boot progress
3. Debug issues without waiting for job completion
4. Confirm the WAA server is responding

**Current pain point**: Jobs run for 15-30+ minutes, and we can't see what's happening inside until they complete or fail.

## Architecture

```
Azure ML Job
└── Compute Instance (Standard_D8ds_v4)
    └── Docker Container (windowsarena/winarena)
        └── run_entry.py (entry point)
            └── /entry_setup.sh
                └── QEMU (Windows 11 VM)
                    └── WAA Flask Server (port 5000)
```

## Approaches Tried

### 1. Azure ML SDK `jobs.stream()` ❌
```python
client.jobs.stream('job_name')
```
**Result**: Appears to hang indefinitely. May be waiting for job to complete before returning any output.

**Why it doesn't work**:
- `stream()` may only stream *control plane* logs (job start, provisioning), not container stdout
- Container logs need different endpoint/mechanism

### 2. Azure ML SDK `jobs.download()` ❌
```python
client.jobs.download(name='job_name', download_path='/tmp', output_name='default')
```
**Result**: Error - "Download is allowed only in states ['Completed', 'Failed', 'Canceled', ...]"

**Why it doesn't work**: By design, artifacts are only available after job termination.

### 3. SSH to Compute Instance ⚠️ (Partial)
```bash
ssh azureuser@104.43.139.26 -p 50000
docker exec <container_id> cat /path/to/logs
```
**Result**: Works but requires:
- Finding the compute instance IP (not easily available from SDK)
- Finding the SSH port (default 50000 for Azure ML)
- Finding the container ID inside the instance
- Navigating into container to find logs

**Challenge**: Compute instance IPs aren't exposed in the standard SDK. Need to use lower-level Azure APIs or parse from job metadata.

### 4. Reading Blob Storage Directly ⚠️ (Investigating)
Azure ML writes job artifacts to blob storage at path: `ExperimentRun/dcid.<job_name>/`
```bash
az storage blob list --container-name azureml --prefix "ExperimentRun/dcid.job_name"
```
**Result**: Logs may not be written in real-time. Standard output goes to `user_logs/std_log.txt` but sync timing is unclear.

## Potential Solutions

### Option A: Write Logs to Blob Storage in Real-Time
**Approach**: Modify `run_entry.py` to write logs directly to mounted blob storage.

```python
# In run_entry.py
import sys

# Azure ML mounts output paths - write logs there
log_path = "/mnt/azureml/cr/j/.../user_logs/live_log.txt"
with open(log_path, 'a') as f:
    f.write("Starting VM...\n")
    f.flush()  # Ensure immediate write
```

**Pros**: Uses existing infrastructure
**Cons**: Need to determine exact mount path; may still have sync delays

### Option B: Configure Application Insights/Log Analytics
**Approach**: Stream container logs to Azure Monitor.

```python
# In job environment
from applicationinsights import TelemetryClient
tc = TelemetryClient('<instrumentation_key>')
tc.track_trace('Starting VM...')
tc.flush()
```

**Pros**: Centralized logging, queryable, real-time
**Cons**: Additional Azure resource, more complexity

### Option C: HTTP Webhook for Status Updates
**Approach**: Have `run_entry.py` POST status updates to an external endpoint.

```python
import requests

def log_status(message):
    requests.post('https://your-endpoint.com/log', json={
        'job': job_name,
        'message': message,
        'timestamp': time.time()
    })
```

**Pros**: Real-time, flexible
**Cons**: Requires external endpoint, network connectivity from Azure ML

### Option D: Write to Mounted Output Directory (RECOMMENDED)
**Approach**: Azure ML automatically mounts output directories. Write logs there.

```python
# run_entry.py
import os
import sys
from datetime import datetime

# Azure ML sets this to the mounted output path
output_dir = os.environ.get('AZUREML_OUTPUT_PATH', '/mnt/azureml/outputs')
log_file = os.path.join(output_dir, 'live_log.txt')

class TeeLogger:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Redirect stdout to both console and file
log_handle = open(log_file, 'w')
sys.stdout = TeeLogger(sys.__stdout__, log_handle)
```

Then read from blob storage:
```bash
az storage blob download \
    --account-name <storage_account> \
    --container-name azureml \
    --name "ExperimentRun/dcid.<job_name>/outputs/live_log.txt" \
    --file /tmp/live_log.txt
```

**Pros**: Uses existing infrastructure, no additional setup
**Cons**: Still have blob sync delays (typically 30-60 seconds)

### Option E: Azure ML Run Logging API
**Approach**: Use Azure ML's built-in logging that's designed for real-time streaming.

```python
from azureml.core import Run
run = Run.get_context()

run.log('status', 'Starting VM...')
run.log('ip_patch', 'Applied')
run.log('windows_boot', 'In progress')
```

Then query via SDK:
```python
run = client.runs.get(job_name)
metrics = run.get_metrics()
```

**Pros**: Built-in, designed for this use case
**Cons**: Requires azureml-core package in container; may conflict with existing setup

### Option F: Compute Instance SSH with Automation (SIMPLEST)
**Approach**: Automate the SSH process to extract container logs.

```python
def get_live_logs(job_name):
    # 1. Get compute instance from job
    job = client.jobs.get(job_name)
    compute_name = job.compute  # e.g., "w0Expe02041220"

    # 2. Get compute instance IP via Azure Resource Manager API
    # (Need to call ARM directly as SDK doesn't expose this)

    # 3. SSH and get logs
    ssh_cmd = f"ssh -p 50000 azureuser@{ip} 'docker logs $(docker ps -q)'"
    logs = subprocess.run(ssh_cmd, capture_output=True)
    return logs.stdout
```

**Pros**: Works with existing setup, real-time
**Cons**: Requires parsing ARM API for IP; SSH key must be configured

## Recommendation

**Short-term (now)**: Use **Option F** - SSH automation. We already have SSH access working; just need to automate finding the IP and container ID.

**Medium-term**: Use **Option D** - Write to mounted output + poll blob storage. More reliable and doesn't require SSH.

**Long-term**: Use **Option E** - Azure ML Run logging. Most integrated solution.

## Implementation Plan

### Phase 1: SSH Automation (This Week)

1. Add CLI command to get compute instance IP:
```bash
uv run python -m openadapt_ml.benchmarks.cli azure-logs --job quiet_pasta_byvjklj2q8
```

2. Implementation:
```python
def cmd_azure_logs(job_name: str, follow: bool = False):
    """Get live logs from Azure ML job via SSH."""
    # Get job
    job = client.jobs.get(job_name)
    compute_name = job.compute

    # Get compute IP via ARM API
    ip = get_compute_ip(compute_name)

    # SSH command
    if follow:
        cmd = f"ssh -p 50000 azureuser@{ip} 'docker logs -f $(docker ps -q)'"
    else:
        cmd = f"ssh -p 50000 azureuser@{ip} 'docker logs $(docker ps -q)'"

    subprocess.run(cmd, shell=True)
```

### Phase 2: Blob Storage Logging (Next Week)

1. Modify `run_entry.py` to tee stdout to output directory
2. Add CLI command to poll blob storage for live logs
3. Add background agent that polls and streams logs to terminal

## Open Questions

1. Does Azure ML SDK have undocumented methods for live log streaming?
2. What's the actual blob sync delay for output directories?
3. Can we use Azure SignalR for real-time log push?
4. Is there a REST API endpoint for log streaming (used by Azure Portal)?

## References

- [Azure ML Job Logging](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics)
- [Azure ML Run Context](https://learn.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run)
- [Streaming Logs in Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-debug-visual-studio-code)

---
*Last updated: 2026-02-04*
