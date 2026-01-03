# WAA Network Architecture

This document describes the deterministic network architecture for accessing WAA services.

## Network Topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Local Machine (developer laptop)                                         │
│                                                                          │
│   localhost:8006 ──────┐                                                │
│   localhost:5000 ──────┼── SSH Tunnel ──┐                               │
└────────────────────────┼────────────────┼───────────────────────────────┘
                         │                │
                         ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Azure VM (Standard_D4ds_v5)              azureuser@<public-ip>          │
│                                                                          │
│   localhost:8006 ◄─────────────────┐    Port 22 (SSH) ◄─── NSG open    │
│   localhost:5000 ◄─────────────────┼── Docker port forwarding          │
│                                    │                                     │
│   ┌────────────────────────────────┼────────────────────────────────┐   │
│   │ Docker Container (winarena)    │                                │   │
│   │                                │                                │   │
│   │   Port 8006 ◄──────────────────┤    (VNC/noVNC)                │   │
│   │   Port 5000 ◄──────────────────┘    (WAA Flask)                │   │
│   │                                                                 │   │
│   │   ┌──────────────────────────────────────────────────────────┐ │   │
│   │   │ QEMU Windows 11 VM                                       │ │   │
│   │   │                                                          │ │   │
│   │   │   Internal IP: 172.30.0.2 (dockurr default)             │ │   │
│   │   │              or 20.20.20.21 (official WAA)               │ │   │
│   │   │                                                          │ │   │
│   │   │   WAA Server (Flask) listening on 0.0.0.0:5000          │ │   │
│   │   │   noVNC Server listening on 0.0.0.0:8006                │ │   │
│   │   └──────────────────────────────────────────────────────────┘ │   │
│   └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Principle: Always Use localhost

**The internal QEMU IP (172.30.0.2 or 20.20.20.21) is an implementation detail.**

Docker port forwarding (`-p 8006:8006 -p 5000:5000`) ensures that:
- `localhost:8006` on the Azure VM routes to VNC
- `localhost:5000` on the Azure VM routes to WAA Flask server

This works regardless of:
- Which Docker image is used (official `windowsarena/winarena` or custom `waa-auto`)
- What internal IP QEMU assigns to Windows
- Whether the image was built with patched IPs or not

## Access Patterns

### 1. From Local Machine (via SSH Tunnel)

SSH tunnels forward local ports to the Azure VM's localhost:

```bash
# Manual tunnel
ssh -L 8006:localhost:8006 -L 5000:localhost:5000 azureuser@<vm-ip>

# Then access:
# - VNC: http://localhost:8006
# - WAA: http://localhost:5000/probe
```

The SSHTunnelManager automates this (see `openadapt_ml/cloud/ssh_tunnel.py`).

### 2. From Azure VM (probing WAA health)

When SSH'd into the Azure VM or running commands via SSH:

```bash
# CORRECT: Use localhost (Docker port forwarding)
curl http://localhost:5000/probe

# WRONG: Don't use internal IPs directly
curl http://172.30.0.2:5000/probe  # May not work depending on image
curl http://20.20.20.21:5000/probe  # May not work depending on image
```

### 3. From Code (SSE/API probes)

```python
# CORRECT: Always probe localhost from inside Azure VM
ssh_cmd = f"curl -s http://localhost:5000/probe"
result = subprocess.run(["ssh", ..., f"azureuser@{vm_ip}", ssh_cmd])

# The SSH tunnel manager handles local machine → Azure VM routing
```

## Why This Works

1. **Docker's `-p` flag** creates a port mapping from the host to the container
2. **QEMU's port forwarding** inside the container routes to the Windows VM
3. **WAA Flask server** binds to `0.0.0.0:5000` inside Windows

The chain is:
```
localhost:5000 (Azure VM)
    → Docker forwards to container:5000
        → QEMU forwards to Windows:5000
            → Flask server responds
```

## Configuration

### VM Registry

The VM registry (`vm_registry.json`) stores:

```json
{
  "name": "azure-waa-vm",
  "ssh_host": "172.171.112.41",    // Azure VM public IP
  "vnc_port": 8006,                 // Local tunnel port
  "waa_port": 5000,                 // Local tunnel port
  "internal_ip": "172.30.0.2"       // DEPRECATED: Don't use for probing
}
```

The `internal_ip` field is kept for backwards compatibility but should NOT be used for probing. Always use localhost via Docker port forwarding.

### Environment Variables

```bash
WAA_VM_IP=172.171.112.41      # Azure VM public IP (for SSH)
WAA_INTERNAL_IP=172.30.0.2    # DEPRECATED: Use localhost instead
```

## Troubleshooting

### "WAA not responding" but VNC shows Windows desktop

1. Check if WAA server is running inside Windows (via VNC)
2. Verify Docker port forwarding: `docker ps` should show `0.0.0.0:5000->5000/tcp`
3. Test from Azure VM: `curl http://localhost:5000/probe`

### Connection refused on localhost:5000

1. Docker container may not be running: `docker ps`
2. WAA server may not have started yet: Check VNC for setup progress
3. Port conflict: Another service using port 5000

## Summary

| Location | Target | Method |
|----------|--------|--------|
| Local machine | WAA API | `http://localhost:5000` via SSH tunnel |
| Local machine | VNC | `http://localhost:8006` via SSH tunnel |
| Azure VM | WAA API | `http://localhost:5000` (Docker forwarding) |
| Azure VM | VNC | `http://localhost:8006` (Docker forwarding) |
| Inside container | WAA API | `http://localhost:5000` or internal IP |

**Rule of thumb:** If you're not inside the Docker container, use `localhost`.
