# WAA Benchmark Evaluation Results - January 2026

## Summary

| Metric | Value |
|--------|-------|
| Date | January 6, 2026 |
| Tasks Attempted | 8 of 19 (42%) |
| Tasks Passed | 1 of 8 (12.5%) |
| SSH Timeout | ~1.5 hours into run |
| VM Platform | Azure Standard_D4ds_v5 |
| Agent | Navi (GPT-4o) via WAA |

**Baseline comparison**: SOTA on WAA is ~19.5% (GPT-5.1 + OmniParser). Our run achieved 12.5%, below SOTA.

## Task Results

| # | Task | Domain | Result | Notes |
|---|------|--------|--------|-------|
| 1 | Open Edge browser | Edge | FAIL | Navi agent bug: TypeError on plan_result |
| 2 | Navigate to URL in Edge | Edge | FAIL | Agent got stuck, timeout |
| 3 | Open File Explorer | File Explorer | FAIL | NoneType error in parse_action |
| 4 | Create new folder | File Explorer | FAIL | Command parsing failure |
| 5 | Open Settings app | Settings | FAIL | Agent confusion, wrong actions |
| 6 | Navigate to Display settings | Settings | FAIL | Timeout after failed navigation |
| 7 | Open Details view in Explorer | File Explorer | **PASS** | Correctly clicked View > Details |
| 8 | Change desktop background | Personalization | FAIL | SSH timeout before completion |

## Key Findings

### 1. Navi Agent Bugs (Critical)

The WAA Navi agent has fundamental bugs that cause most failures:

```python
# Error seen repeatedly:
TypeError: expected string or bytes-like object, got 'NoneType'

# Occurs when plan_result is None in:
# navi/agent.py line ~287: re.search(pattern, plan_result)
```

**Root cause**: The agent's planning step sometimes returns `None`, but the regex matching doesn't handle this case.

### 2. Command Parsing Fragility

The agent frequently generates malformed action strings that fail to parse:

- Missing quotes around text arguments
- Invalid coordinate formats
- Actions that don't match expected DSL

### 3. SSH Connection Stability

The benchmark was interrupted at ~1.5 hours due to SSH timeout:
```
Read from remote host 172.171.112.41: Operation timed out
```

**Mitigation**: The dashboard now has auto-reconnect for SSH tunnels, but long-running benchmarks may still be affected.

### 4. Single Success Analysis

The only passing task ("Open Details view") succeeded because:
- It was a simple 2-click action sequence
- The UI elements were clearly visible
- No complex planning was required

This suggests the agent can handle simple, direct UI interactions but struggles with multi-step tasks.

## Infrastructure Performance

### What Worked

1. **Custom waa-auto Docker image**: Auto-downloaded Windows 11, auto-installed dependencies
2. **SSH tunnel management**: VNC accessible at localhost:8006 throughout
3. **Live log tailing**: Real-time visibility into benchmark progress via dashboard
4. **Dashboard monitoring**: Full visibility into VM status, tunnel status, task progress

### What Needs Improvement

1. **SSH keepalive**: Need more aggressive keepalive settings for long-running benchmarks
2. **Task resume**: Ability to resume benchmark from where it stopped after connection drop
3. **Agent fallback**: Consider swapping to Claude/GPT-5.1 API agent when Navi fails

## Recommendations

### Short-term (This Session)

1. **Deallocate Azure VM** - Stop billing while we analyze results
2. **File issue on WAA repo** - The Navi agent `NoneType` bug should be reported upstream

### Medium-term

1. **Implement API-based agent** - Use Claude Sonnet 4.5 or GPT-5.1 directly instead of Navi
2. **Add SSH keepalive** - Configure more aggressive SSH keepalive to prevent timeouts
3. **Add checkpoint/resume** - Save benchmark progress so runs can resume after failures

### Long-term

1. **Custom agent development** - Build openadapt-ml native agent using our fine-tuned models
2. **Compare against baselines** - Run same tasks with Claude/GPT-5.1 API agents for fair comparison
3. **Focus on training data** - Use successful WAA traces for training data collection

## Cost Analysis

| Resource | Duration | Cost |
|----------|----------|------|
| Azure VM (D4ds_v5) | ~2 hours | ~$0.38 |
| Estimated if full run | ~6 hours | ~$1.14 |

VM remains running until deallocated. Recommend deallocating to stop charges.

## Raw Log Highlights

### Successful Task (Details view)
```
[06:17:48] Task: Open the Details view in the current File Explorer window
[06:17:52] Action: CLICK View menu
[06:17:58] Action: CLICK Details option
[06:18:02] Result: PASS - Details view activated
```

### Typical Failure (NoneType error)
```
[06:12:34] Task: Open Edge browser
[06:12:38] Planning step returned None
[06:12:38] TypeError: expected string or bytes-like object, got 'NoneType'
[06:12:38] Result: FAIL - Agent error
```

## Appendix: VM Configuration

```
Instance: Standard_D4ds_v5
vCPUs: 4
RAM: 16 GB
Disk: 128 GB (with /mnt for Docker)
Nested Virtualization: Enabled
Docker Image: waa-auto (custom, based on dockurr/windows:latest)
Windows: Windows 11 (auto-downloaded)
```

## Files

- Dashboard: `http://localhost:5000/benchmark.html`
- Full logs: `/tmp/waa_benchmark.log`
- This document: `docs/experiments/waa_benchmark_results_jan2026.md`
