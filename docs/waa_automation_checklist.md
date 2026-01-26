# WAA End-to-End Automation Checklist

**Purpose**: Track all steps required for fully automated WAA deployment. If any step fails, automation is broken.

---

## Pre-flight

| Step | What | How to Verify |
|------|------|---------------|
| 1 | Azure credentials valid | `az account show` succeeds |
| 2 | SSH key exists | `~/.ssh/id_rsa.pub` exists |
| 3 | Nested virt VM size available | D4ds_v4 has quota |

---

## VM Setup Phase

| Step | What | How to Verify | Automated By |
|------|------|---------------|--------------|
| 4 | VM creates | `az vm show` succeeds | `vm create` |
| 5 | VM has nested virt | KVM device exists | VM size selection |
| 6 | Docker installs | `docker --version` | `vm setup-waa` |
| 7 | Docker uses /mnt storage | `docker info` shows /mnt | daemon.json |

---

## Image Build Phase

| Step | What | How to Verify | Automated By |
|------|------|---------------|--------------|
| 8 | Vendor files copied to VM | Files exist in ~/waa-build | `vm setup-waa` |
| 9 | Image builds | `docker images` shows image | `docker build` |
| 10 | Image size reasonable | < 2GB (not 45GB) | Standalone approach |

---

## Container Start Phase

| Step | What | How to Verify | Automated By |
|------|------|---------------|--------------|
| 11 | Container starts | `docker ps` shows running | `docker run` |
| 12 | VNC accessible | Port 8006 responds | Container ports |
| 13 | Samba share created | /tmp/smb/ exists | dockurr/windows |
| 14 | WAA files in Samba | /tmp/smb/install.bat exists | samba.sh injection |

---

## Windows Install Phase

| Step | What | How to Verify | Automated By |
|------|------|---------------|--------------|
| 15 | Windows ISO downloads | VNC shows "Downloading" | VERSION=11e |
| 16 | No ISO selection prompt | VNC shows progress bar | Enterprise Eval |
| 17 | Windows installs | VNC shows setup screens | Unattend XML |
| 18 | No license key prompt | Setup continues | Enterprise GVLK |
| 19 | Windows boots to desktop | VNC shows desktop | Unattend XML |

---

## WAA Setup Phase

| Step | What | How to Verify | Automated By |
|------|------|---------------|--------------|
| 20 | FirstLogonCommands run | Log file created | Unattend XML |
| 21 | install.bat finds scripts | No "file not found" in log | Dev mode + samba.sh |
| 22 | setup.ps1 runs | Log shows "Running setup.ps1" | install.bat |
| 23 | Python installs | Log shows Python version | setup.ps1 |
| 24 | Dependencies install | Log shows pip install | setup.ps1 |
| 25 | WAA server starts | Port 5000 responds | on-logon.ps1 |

---

## Connectivity Phase

| Step | What | How to Verify | Automated By |
|------|------|---------------|--------------|
| 26 | Port 5000 forwarded | Container:5000 → VM:5000 | port_forward.sh |
| 27 | Probe returns 200 | `curl localhost:5000/probe` | WAA server |
| 28 | SSH tunnels work | Localhost ports accessible | ssh_tunnel.py |

---

## Benchmark Phase

| Step | What | How to Verify | Automated By |
|------|------|---------------|--------------|
| 29 | Task executes | Client sends action | start_client.sh |
| 30 | Screenshot captured | Response includes image | WAA server |
| 31 | Task completes | Result logged | Evaluation |

---

## Current Status

**Last tested**: (date)
**Result**: (pass/fail)
**Failed at step**: (if applicable)

---

## Approach Summary

```
                    Container                          Windows VM
                   ┌──────────────────────────────────────────────────────┐
                   │                                                      │
 Build time:       │  /waa-setup/  ──────────────────────────────────────│
                   │  (staging)                                           │
                   │                                                      │
 Runtime:          │  samba.sh     ──────┐                                │
                   │  copies to          │                                │
                   │                     ▼                                │
                   │  /tmp/smb/   ◄──────┬───────────► \\host.lan\Data    │
                   │  (Samba)            │                                │
                   │                     │                                │
                   │  port 5000  ◄───────┼───────────► WAA Server :5000   │
                   │  (forwarder)        │                                │
                   └─────────────────────┴────────────────────────────────┘
```

**Key insight**: Use vanilla WAA's "dev mode" which expects files at `\\host.lan\Data` (Samba share). No script patching required except IP address fix for modern dockurr/windows.
