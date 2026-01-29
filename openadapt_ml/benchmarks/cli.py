#!/usr/bin/env python3
"""
WAA Benchmark CLI - Windows Agent Arena evaluation toolkit

Uses custom waa_deploy/Dockerfile with dockurr/windows:latest base and
Python 3.9 from vanilla windowsarena/winarena for GroundingDINO compatibility.

See waa_deploy/Dockerfile for details.

Usage:
    uv run python -m openadapt_ml.benchmarks.cli <command> [options]

Commands:
    create      Create Azure VM with nested virtualization
    delete      Delete VM and ALL associated resources
    status      Show VM state and IP
    build       Build WAA image from waa_deploy/Dockerfile
    start       Start WAA container (Windows boots + WAA server)
    probe       Check if WAA server is ready
    run         Run benchmark tasks
    deallocate  Stop VM (preserves disk, stops billing)
    logs        Show WAA status and logs

Workflow:
    1. create    - Create Azure VM (~5 min)
    2. build     - Build custom WAA image (~10 min)
    3. start     - Start container, Windows downloads+boots (~15-20 min first time)
    4. probe --wait - Wait for WAA server
    5. run       - Run benchmark
    6. deallocate - Stop billing
"""

import argparse
import json
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# Constants (single source of truth)
# =============================================================================

# VM sizes with nested virtualization support
# Standard: $0.19/hr, 4 vCPU, 16GB RAM - baseline
# Fast: $0.38/hr, 8 vCPU, 32GB RAM - ~30% faster install, ~40% faster eval
VM_SIZE_STANDARD = "Standard_D4ds_v4"
VM_SIZE_FAST = "Standard_D8ds_v5"
VM_SIZE = VM_SIZE_STANDARD  # Default, can be overridden by --fast flag

# Fallback sizes for --fast mode (in order of preference)
# D8ds_v5: First choice (v5 with local SSD)
# D8s_v5: v5 without local SSD
# D8ds_v4: v4 with local SSD
# D8as_v5: AMD version
VM_SIZE_FAST_FALLBACKS = [
    ("Standard_D8ds_v5", 0.38),
    ("Standard_D8s_v5", 0.36),
    ("Standard_D8ds_v4", 0.38),
    ("Standard_D8as_v5", 0.34),
]
VM_REGIONS = ["centralus", "eastus", "westus2", "eastus2"]
VM_NAME = "waa-eval-vm"
RESOURCE_GROUP = "openadapt-agents"
# Custom image built from waa_deploy/Dockerfile
# Uses dockurr/windows:latest (proper ISO download) + WAA components
DOCKER_IMAGE = "waa-auto:latest"
LOG_DIR = Path.home() / ".openadapt" / "waa"
SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "ConnectTimeout=10",
]


def setup_vnc_tunnel_and_browser(ip: str) -> Optional[subprocess.Popen]:
    """Set up SSH tunnel for VNC and open browser.

    Returns the tunnel process on success, None on failure.
    """
    # Kill any existing tunnel on port 8006
    subprocess.run(["pkill", "-f", "ssh.*8006:localhost:8006"], capture_output=True)

    # Start SSH tunnel in background
    tunnel_proc = subprocess.Popen(
        ["ssh", *SSH_OPTS, "-N", "-L", "8006:localhost:8006", f"azureuser@{ip}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for tunnel to establish
    time.sleep(2)

    # Check if tunnel is running
    if tunnel_proc.poll() is not None:
        return None

    # Open browser
    vnc_url = "http://localhost:8006"
    webbrowser.open(vnc_url)

    return tunnel_proc


# Dockerfile location (relative to this file)
DOCKERFILE_PATH = Path(__file__).parent / "waa_deploy" / "Dockerfile"

# =============================================================================
# Logging
# =============================================================================

_log_file: Optional[Path] = None
_session_id: Optional[str] = None


def init_logging() -> Path:
    """Initialize logging for this session."""
    global _log_file, _session_id

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create session ID
    _session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    session_dir = LOG_DIR / "sessions" / _session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Session log file
    _log_file = session_dir / "full.log"

    # Update current session pointer
    (LOG_DIR / "session_id.txt").write_text(_session_id)

    # Symlink for easy access
    current_link = LOG_DIR / "current"
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()
    current_link.symlink_to(session_dir)

    return _log_file


def log(step: str, message: str, end: str = "\n"):
    """Log message to file and stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [{step}] {message}"

    # Print to stdout
    print(formatted, end=end, flush=True)

    # Write to log file
    if _log_file:
        with open(_log_file, "a") as f:
            f.write(formatted + end)


def log_stream(step: str, process: subprocess.Popen):
    """Stream process output to log and stdout."""
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            if line:
                log(step, line.rstrip())


# =============================================================================
# Azure Helpers
# =============================================================================


def get_vm_ip() -> Optional[str]:
    """Get VM public IP if it exists."""
    result = subprocess.run(
        [
            "az",
            "vm",
            "show",
            "-d",
            "-g",
            RESOURCE_GROUP,
            "-n",
            VM_NAME,
            "--query",
            "publicIps",
            "-o",
            "tsv",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def get_vm_state() -> Optional[str]:
    """Get VM power state."""
    result = subprocess.run(
        [
            "az",
            "vm",
            "get-instance-view",
            "-g",
            RESOURCE_GROUP,
            "-n",
            VM_NAME,
            "--query",
            "instanceView.statuses[1].displayStatus",
            "-o",
            "tsv",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def ssh_run(
    ip: str, cmd: str, stream: bool = False, step: str = "SSH"
) -> subprocess.CompletedProcess:
    """Run command on VM via SSH.

    When stream=True:
    1. Runs command on VM with output redirected to a persistent log file
    2. Streams that log file locally in real-time
    3. Log file persists on VM even if connection breaks

    Remote logs are stored at: /home/azureuser/cli_logs/{step}.log
    """
    if stream:
        # Remote log directory and file (persistent across sessions)
        remote_log_dir = "/home/azureuser/cli_logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_log = f"{remote_log_dir}/{step.lower()}_{timestamp}.log"

        # Ensure log directory exists
        subprocess.run(
            ["ssh", *SSH_OPTS, f"azureuser@{ip}", f"mkdir -p {remote_log_dir}"],
            capture_output=True,
        )

        log(step, f"Remote log: {remote_log}")

        # Run command with output to log file, capturing exit code
        # Using script to capture terminal output including \r progress updates
        # The command runs in foreground but output goes to file AND stdout
        wrapped_cmd = f"""
set -o pipefail
{{
  {cmd}
  echo $? > {remote_log}.exit
}} 2>&1 | tee {remote_log}
"""
        full_cmd = ["ssh", *SSH_OPTS, f"azureuser@{ip}", wrapped_cmd]

        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output to local log
        try:
            for line in iter(process.stdout.readline, ""):
                if line:
                    # Handle carriage returns (Docker progress)
                    clean_line = line.rstrip()
                    if "\r" in clean_line:
                        # Take the last part after \r
                        parts = clean_line.split("\r")
                        clean_line = parts[-1].strip()
                    if clean_line:
                        log(step, clean_line)
            process.wait()
        except KeyboardInterrupt:
            log(step, "Interrupted - command continues on VM")
            log(step, f"View full log: ssh azureuser@{ip} 'cat {remote_log}'")
            process.terminate()
            return subprocess.CompletedProcess(cmd, 130, "", "")

        # Get exit code
        result = subprocess.run(
            [
                "ssh",
                *SSH_OPTS,
                f"azureuser@{ip}",
                f"cat {remote_log}.exit 2>/dev/null || echo 1",
            ],
            capture_output=True,
            text=True,
        )
        exit_code = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 1

        if exit_code != 0:
            log(step, f"Command failed (exit {exit_code})")
            log(step, f"Full log: ssh azureuser@{ip} 'cat {remote_log}'")

        return subprocess.CompletedProcess(cmd, exit_code, "", "")
    else:
        full_cmd = ["ssh", *SSH_OPTS, f"azureuser@{ip}", cmd]
        return subprocess.run(full_cmd, capture_output=True, text=True)


def wait_for_ssh(ip: str, timeout: int = 120) -> bool:
    """Wait for SSH to become available."""
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            ["ssh", *SSH_OPTS, f"azureuser@{ip}", "echo ok"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return True
        time.sleep(5)
    return False


# =============================================================================
# Commands
# =============================================================================


def cmd_create(args):
    """Create Azure VM with nested virtualization."""
    init_logging()

    # Check if VM already exists
    ip = get_vm_ip()
    if ip:
        log("CREATE", f"VM already exists: {ip}")
        log("CREATE", "Use 'delete' first if you want to recreate")
        return 0

    # Determine which sizes to try
    use_fast = getattr(args, "fast", False)
    if use_fast:
        # Try multiple fast sizes with fallbacks
        sizes_to_try = VM_SIZE_FAST_FALLBACKS
        log(
            "CREATE",
            f"Creating VM '{VM_NAME}' with --fast (trying multiple D8 sizes)...",
        )
    else:
        # Standard mode: single size
        sizes_to_try = [(VM_SIZE_STANDARD, 0.19)]
        log("CREATE", f"Creating VM '{VM_NAME}' ({VM_SIZE_STANDARD}, $0.19/hr)...")

    # Try size+region combinations until one works
    vm_created = False
    successful_size = None
    successful_cost = None

    for vm_size, cost_per_hour in sizes_to_try:
        log("CREATE", f"Trying size {vm_size} (${cost_per_hour:.2f}/hr)...")

        for region in VM_REGIONS:
            log("CREATE", f"  {region}...", end=" ")

            result = subprocess.run(
                [
                    "az",
                    "vm",
                    "create",
                    "--resource-group",
                    RESOURCE_GROUP,
                    "--name",
                    VM_NAME,
                    "--location",
                    region,
                    "--image",
                    "Ubuntu2204",
                    "--size",
                    vm_size,
                    "--admin-username",
                    "azureuser",
                    "--generate-ssh-keys",
                    "--public-ip-sku",
                    "Standard",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                vm_info = json.loads(result.stdout)
                ip = vm_info.get("publicIpAddress", "")
                log("CREATE", f"created ({ip})")
                vm_created = True
                successful_size = vm_size
                successful_cost = cost_per_hour
                break
            else:
                log("CREATE", "unavailable")

        if vm_created:
            break

    if not vm_created:
        log("CREATE", "ERROR: Could not create VM in any region with any size")
        if use_fast:
            log("CREATE", "Tried sizes: " + ", ".join(s[0] for s in sizes_to_try))
        return 1

    log(
        "CREATE",
        f"Successfully created {successful_size} (${successful_cost:.2f}/hr) in {region}",
    )

    # Wait for SSH
    log("CREATE", "Waiting for SSH...")
    if not wait_for_ssh(ip):
        log("CREATE", "ERROR: SSH not available after 2 minutes")
        return 1
    log("CREATE", "SSH ready")

    # Install Docker with /mnt storage
    log("CREATE", "Installing Docker with /mnt storage...")
    docker_setup = """
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Configure Docker to use /mnt (larger temp disk)
sudo systemctl stop docker
sudo mkdir -p /mnt/docker
sudo bash -c 'echo "{\\"data-root\\": \\"/mnt/docker\\"}" > /etc/docker/daemon.json'
sudo systemctl start docker

# Verify
docker --version
df -h /mnt
"""
    result = ssh_run(ip, docker_setup, stream=True, step="CREATE")
    if result.returncode != 0:
        log("CREATE", "ERROR: Docker setup failed")
        return 1

    log("CREATE", f"VM ready: {ip}")
    return 0


def cmd_delete(args):
    """Delete VM and ALL associated resources."""
    init_logging()
    log("DELETE", f"Deleting VM '{VM_NAME}' and all associated resources...")

    # Delete VM
    log("DELETE", "Deleting VM...")
    result = subprocess.run(
        [
            "az",
            "vm",
            "delete",
            "-g",
            RESOURCE_GROUP,
            "-n",
            VM_NAME,
            "--yes",
            "--force-deletion",
            "true",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log("DELETE", "VM deleted")
    else:
        log("DELETE", "VM not found or already deleted")

    # Delete NICs
    log("DELETE", "Deleting NICs...")
    result = subprocess.run(
        [
            "az",
            "network",
            "nic",
            "list",
            "-g",
            RESOURCE_GROUP,
            "--query",
            "[?contains(name, 'waa')].name",
            "-o",
            "tsv",
        ],
        capture_output=True,
        text=True,
    )
    for nic in result.stdout.strip().split("\n"):
        if nic:
            subprocess.run(
                ["az", "network", "nic", "delete", "-g", RESOURCE_GROUP, "-n", nic],
                capture_output=True,
            )
            log("DELETE", f"  Deleted NIC: {nic}")

    # Delete public IPs
    log("DELETE", "Deleting public IPs...")
    result = subprocess.run(
        [
            "az",
            "network",
            "public-ip",
            "list",
            "-g",
            RESOURCE_GROUP,
            "--query",
            "[?contains(name, 'waa')].name",
            "-o",
            "tsv",
        ],
        capture_output=True,
        text=True,
    )
    for pip in result.stdout.strip().split("\n"):
        if pip:
            subprocess.run(
                [
                    "az",
                    "network",
                    "public-ip",
                    "delete",
                    "-g",
                    RESOURCE_GROUP,
                    "-n",
                    pip,
                ],
                capture_output=True,
            )
            log("DELETE", f"  Deleted IP: {pip}")

    # Delete disks
    log("DELETE", "Deleting disks...")
    result = subprocess.run(
        [
            "az",
            "disk",
            "list",
            "-g",
            RESOURCE_GROUP,
            "--query",
            "[?contains(name, 'waa')].name",
            "-o",
            "tsv",
        ],
        capture_output=True,
        text=True,
    )
    for disk in result.stdout.strip().split("\n"):
        if disk:
            subprocess.run(
                ["az", "disk", "delete", "-g", RESOURCE_GROUP, "-n", disk, "--yes"],
                capture_output=True,
            )
            log("DELETE", f"  Deleted disk: {disk}")

    # Delete NSGs
    log("DELETE", "Deleting NSGs...")
    result = subprocess.run(
        [
            "az",
            "network",
            "nsg",
            "list",
            "-g",
            RESOURCE_GROUP,
            "--query",
            "[?contains(name, 'waa')].name",
            "-o",
            "tsv",
        ],
        capture_output=True,
        text=True,
    )
    for nsg in result.stdout.strip().split("\n"):
        if nsg:
            subprocess.run(
                ["az", "network", "nsg", "delete", "-g", RESOURCE_GROUP, "-n", nsg],
                capture_output=True,
            )
            log("DELETE", f"  Deleted NSG: {nsg}")

    log("DELETE", "Cleanup complete")
    return 0


def cmd_status(args):
    """Show VM status."""
    ip = get_vm_ip()
    state = get_vm_state()

    if not ip:
        print(f"VM '{VM_NAME}' not found")
        return 1

    print(f"VM: {VM_NAME}")
    print(f"  State: {state or 'unknown'}")
    print(f"  IP: {ip}")
    print(f"  Size: {VM_SIZE}")
    print(f"  SSH: ssh azureuser@{ip}")
    return 0


def cmd_build(args):
    """Build WAA image from waa_deploy/Dockerfile.

    This builds our custom image that:
    - Uses dockurr/windows:latest (has working ISO auto-download)
    - Copies WAA components from windowsarena/winarena:latest
    - Patches IP addresses and adds automation
    """
    init_logging()

    ip = get_vm_ip()
    if not ip:
        log("BUILD", "ERROR: VM not found. Run 'create' first.")
        return 1

    log("BUILD", "Building WAA image from waa_deploy/Dockerfile...")

    # Check Dockerfile exists
    if not DOCKERFILE_PATH.exists():
        log("BUILD", f"ERROR: Dockerfile not found: {DOCKERFILE_PATH}")
        return 1

    # Copy Dockerfile and supporting files to VM
    log("BUILD", "Copying build files to VM...")
    ssh_run(ip, "mkdir -p ~/build")

    waa_deploy_dir = DOCKERFILE_PATH.parent
    files_to_copy = ["Dockerfile", "start_waa_server.bat", "api_agent.py"]
    for filename in files_to_copy:
        src = waa_deploy_dir / filename
        if src.exists():
            result = subprocess.run(
                ["scp", *SSH_OPTS, str(src), f"azureuser@{ip}:~/build/"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                log("BUILD", f"ERROR: Failed to copy {filename}: {result.stderr}")
                return 1

    # Pre-build cleanup
    log("BUILD", "Cleaning up dangling images before build...")
    ssh_run(ip, "docker image prune -f 2>/dev/null")

    # Build image (streams output)
    log("BUILD", "Running docker build (this takes ~10-15 minutes)...")
    build_cmd = f"cd ~/build && docker build --pull -t {DOCKER_IMAGE} . 2>&1"
    result = ssh_run(ip, build_cmd, stream=True, step="BUILD")

    if result.returncode != 0:
        log("BUILD", "ERROR: Docker build failed")
        return 1

    # Post-build cleanup
    log("BUILD", "Cleaning up dangling images after build...")
    ssh_run(ip, "docker image prune -f 2>/dev/null")

    log("BUILD", f"Image built: {DOCKER_IMAGE}")
    return 0


def cmd_start(args):
    """Start WAA container."""
    init_logging()

    ip = get_vm_ip()
    if not ip:
        log("START", "ERROR: VM not found. Run 'create' first.")
        return 1

    log("START", "Starting WAA container...")

    # Stop existing container
    log("START", "Stopping any existing container...")
    ssh_run(ip, "docker stop winarena 2>/dev/null; docker rm -f winarena 2>/dev/null")

    # Clean storage if --fresh
    if args.fresh:
        log("START", "Cleaning storage for fresh Windows install...")
        ssh_run(ip, "sudo rm -rf /mnt/waa-storage/*")

    # Create storage directory
    ssh_run(
        ip,
        "sudo mkdir -p /mnt/waa-storage && sudo chown azureuser:azureuser /mnt/waa-storage",
    )

    # Start container
    # Our custom image has ENTRYPOINT that handles everything:
    # - Downloads Windows 11 Enterprise if not present
    # - Boots QEMU VM
    # - Runs WAA server automatically via FirstLogonCommands
    # QEMU resource allocation (--fast uses more resources on D8ds_v5)
    if getattr(args, "fast", False):
        ram_size = "16G"
        cpu_cores = 6
        log(
            "START",
            "Starting container with VERSION=11e (FAST mode: 6 cores, 16GB RAM)...",
        )
    else:
        ram_size = "8G"
        cpu_cores = 4
        log("START", "Starting container with VERSION=11e...")

    docker_cmd = f"""docker run -d \\
  --name winarena \\
  --device=/dev/kvm \\
  --cap-add NET_ADMIN \\
  -p 8006:8006 \\
  -p 5000:5000 \\
  -p 7200:7200 \\
  -v /mnt/waa-storage:/storage \\
  -e VERSION=11e \\
  -e RAM_SIZE={ram_size} \\
  -e CPU_CORES={cpu_cores} \\
  -e DISK_SIZE=64G \\
  {DOCKER_IMAGE}"""

    result = ssh_run(ip, docker_cmd)
    if result.returncode != 0:
        log("START", f"ERROR: Failed to start container: {result.stderr}")
        return 1

    log("START", "Container started")
    log("START", "Windows will boot and install (15-20 min on first run)")

    # Auto-launch VNC unless --no-vnc specified
    if not getattr(args, "no_vnc", False):
        log("START", "Auto-launching VNC viewer...")
        tunnel_proc = setup_vnc_tunnel_and_browser(ip)
        if tunnel_proc:
            log(
                "START",
                f"VNC auto-launched at http://localhost:8006 (tunnel PID: {tunnel_proc.pid})",
            )
        else:
            log("START", "WARNING: VNC tunnel failed to start")
            log("START", f"Manual VNC: ssh -L 8006:localhost:8006 azureuser@{ip}")
    else:
        log("START", f"VNC (via SSH tunnel): ssh -L 8006:localhost:8006 azureuser@{ip}")

    return 0


def cmd_stop(args):
    """Stop and remove WAA container."""
    ip = get_vm_ip()
    if not ip:
        print("ERROR: VM not found")
        return 1

    print(f"Stopping container on VM ({ip})...")

    # Stop container
    result = ssh_run(
        ip, "docker stop winarena 2>/dev/null && echo STOPPED || echo NOT_RUNNING"
    )
    if "STOPPED" in result.stdout:
        print("  Container stopped")
    else:
        print("  Container was not running")

    # Remove container
    result = ssh_run(
        ip, "docker rm -f winarena 2>/dev/null && echo REMOVED || echo NOT_FOUND"
    )
    if "REMOVED" in result.stdout:
        print("  Container removed")
    else:
        print("  Container already removed")

    # Optionally clean storage
    if hasattr(args, "clean") and args.clean:
        print("  Cleaning Windows storage...")
        ssh_run(ip, "sudo rm -rf /mnt/waa-storage/*")
        print("  Storage cleaned")

    print("Done")
    return 0


def cmd_probe(args):
    """Check if WAA server is ready."""
    ip = get_vm_ip()
    if not ip:
        print("ERROR: VM not found")
        return 1

    timeout = args.timeout
    start = time.time()
    last_storage = None

    while True:
        # Check via SSH - must run curl INSIDE container to reach Docker network
        result = ssh_run(
            ip,
            "docker exec winarena curl -s --max-time 5 http://172.30.0.2:5000/probe 2>/dev/null || echo FAIL",
        )

        if "FAIL" not in result.stdout and result.stdout.strip():
            print("\nWAA server is READY")
            print(f"  Response: {result.stdout.strip()[:100]}")
            return 0

        if not args.wait:
            print("WAA server is NOT ready")
            return 1

        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"\nTIMEOUT: WAA server not ready after {timeout}s")
            return 1

        # Get detailed status for progress display
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)

        # Get storage in bytes for detailed view
        storage_result = ssh_run(
            ip, "docker exec winarena du -sb /storage/ 2>/dev/null | cut -f1"
        )
        storage_bytes = storage_result.stdout.strip()
        if storage_bytes.isdigit():
            storage_mb = int(storage_bytes) / (1024 * 1024)
            storage_str = f"{storage_mb:,.1f} MB"
            # Show delta if we have previous value
            if last_storage is not None:
                delta = int(storage_bytes) - last_storage
                if delta > 0:
                    delta_mb = delta / (1024 * 1024)
                    storage_str += f" (+{delta_mb:,.1f} MB)"
            last_storage = int(storage_bytes)
        else:
            storage_str = "unknown"

        # Get QEMU uptime
        qemu_result = ssh_run(
            ip,
            'docker exec winarena sh -c \'QPID=$(pgrep -f qemu-system 2>/dev/null | head -1); [ -n "$QPID" ] && ps -o etime= -p $QPID 2>/dev/null | tr -d " " || echo N/A\'',
        )
        qemu_uptime = qemu_result.stdout.strip() or "N/A"

        # Get container uptime
        container_result = ssh_run(
            ip, "docker ps --filter name=winarena --format '{{.Status}}' 2>/dev/null"
        )
        container_status = container_result.stdout.strip() or "unknown"

        print(
            f"[{elapsed_min:02d}:{elapsed_sec:02d}] Waiting... | Storage: {storage_str} | QEMU: {qemu_uptime} | Container: {container_status}"
        )
        time.sleep(30)


def cmd_run(args):
    """Run benchmark tasks using vanilla WAA's navi agent.

    Note: For API-based agents (Claude, GPT-4 direct), use openadapt-evals
    which communicates with WAA's Flask API externally.
    """
    init_logging()

    ip = get_vm_ip()
    if not ip:
        log("RUN", "ERROR: VM not found")
        return 1

    # Check WAA is ready
    log("RUN", "Checking WAA server...")
    result = ssh_run(
        ip,
        "docker exec winarena curl -s --max-time 5 http://172.30.0.2:5000/probe 2>/dev/null || echo FAIL",
    )
    if "FAIL" in result.stdout or not result.stdout.strip():
        log("RUN", "ERROR: WAA server not ready. Run 'probe --wait' first.")
        return 1

    log("RUN", "WAA server is ready")

    # Get API key (navi uses GPT-4o for reasoning)
    api_key = args.api_key
    if not api_key:
        try:
            from openadapt_ml.config import settings

            api_key = settings.openai_api_key or ""
        except ImportError:
            api_key = ""

    if not api_key:
        log("RUN", "ERROR: OpenAI API key required (navi uses GPT-4o)")
        log("RUN", "  Set OPENAI_API_KEY in .env file or pass --api-key")
        return 1

    # Build task selection
    domain = args.domain
    task = args.task
    model = args.model

    task_info = []
    if task:
        task_info.append(f"task={task}")
    elif domain != "all":
        task_info.append(f"domain={domain}")
    else:
        task_info.append(f"{args.num_tasks} task(s)")

    log("RUN", f"Starting benchmark: {', '.join(task_info)}, model={model}")

    # Build run.py arguments
    run_args = [
        "--agent_name navi",
        f"--model {model}",
        f"--domain {domain}",
    ]

    # Add parallelization flags if specified (argparse converts hyphens to underscores)
    worker_id = getattr(args, "worker_id", 0)
    num_workers = getattr(args, "num_workers", 1)
    if num_workers > 1:
        run_args.append(f"--worker_id {worker_id}")
        run_args.append(f"--num_workers {num_workers}")
        log("RUN", f"Parallel mode: worker {worker_id}/{num_workers}")

    # If specific task requested, create custom test config
    if task:
        create_custom_test_cmd = f'''
cat > /client/evaluation_examples_windows/test_custom.json << 'CUSTOMEOF'
["{task}"]
CUSTOMEOF
'''
        run_args.append(
            "--test_all_meta_path evaluation_examples_windows/test_custom.json"
        )
        pre_cmd = create_custom_test_cmd
    elif args.num_tasks and args.num_tasks < 154:
        # Limit tasks by creating custom test config with first N tasks
        num = args.num_tasks
        # Write a temp Python script then run it (avoids quote escaping hell)
        # test_all.json is a dict {{domain: [task_ids...]}} - preserve domain structure
        create_limited_test_cmd = f"""cat > /tmp/limit_tasks.py << LIMITEOF
import json
d = json.load(open("/client/evaluation_examples_windows/test_all.json"))
# Collect (domain, task_id) pairs to preserve domain info
all_tasks = []
for domain, tasks in d.items():
    for task in tasks:
        all_tasks.append((domain, task))
# Limit total tasks
limited = all_tasks[:{num}]
# Rebuild dict preserving original domain structure
result = {{}}
for domain, task in limited:
    if domain not in result:
        result[domain] = []
    result[domain].append(task)
json.dump(result, open("/client/evaluation_examples_windows/test_limited.json", "w"))
print("Limited to", len(limited), "tasks from", len(result), "domains")
LIMITEOF
python /tmp/limit_tasks.py && """
        run_args.append(
            "--test_all_meta_path evaluation_examples_windows/test_limited.json"
        )
        pre_cmd = create_limited_test_cmd
    else:
        pre_cmd = ""

    # Run the benchmark inside the container
    run_cmd = (
        f'export OPENAI_API_KEY="{api_key}" && '
        f"docker exec -e OPENAI_API_KEY winarena "
        f"bash -c '{pre_cmd}cd /client && python run.py {' '.join(run_args)}'"
    )

    log("RUN", "Executing benchmark...")
    log("RUN", f"  Model: {model}")
    log("RUN", f"  Tasks: {task_info[0]}")
    log("RUN", "-" * 60)

    # Run with streaming output
    result = ssh_run(ip, run_cmd, stream=True, step="RUN")

    if result.returncode != 0:
        log("RUN", f"Benchmark failed with exit code {result.returncode}")
    else:
        log("RUN", "Benchmark completed!")

    # Download results unless --no-download
    if not args.no_download:
        log("RUN", "Downloading results...")
        download_benchmark_results(ip)

    return result.returncode


def download_benchmark_results(ip: str) -> str:
    """Download benchmark results from the container.

    Results are saved to benchmark_results/waa_results_TIMESTAMP/
    Returns the path to the results directory, or None if failed.
    """
    from pathlib import Path

    # Create local results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("benchmark_results") / f"waa_results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    log("RUN", f"Saving results to {results_dir}/")

    # Create tarball of results inside container
    log("RUN", "Creating results archive...")
    tar_cmd = "docker exec winarena tar -czvf /tmp/results.tar.gz -C /client/results . 2>/dev/null"
    result = subprocess.run(
        ["ssh", *SSH_OPTS, f"azureuser@{ip}", tar_cmd], capture_output=True, text=True
    )

    if result.returncode != 0:
        log(
            "RUN",
            f"Warning: Failed to create archive: {result.stderr[:200] if result.stderr else 'unknown'}",
        )
        log("RUN", "Trying direct copy...")

        # Try copying results directory directly
        copy_cmd = "docker cp winarena:/client/results/. /tmp/waa-results/"
        subprocess.run(
            [
                "ssh",
                *SSH_OPTS,
                f"azureuser@{ip}",
                f"rm -rf /tmp/waa-results && mkdir -p /tmp/waa-results && {copy_cmd}",
            ],
            capture_output=True,
        )

        # Download via scp
        scp_result = subprocess.run(
            [
                "scp",
                "-r",
                *SSH_OPTS,
                f"azureuser@{ip}:/tmp/waa-results/*",
                str(results_dir),
            ],
            capture_output=True,
            text=True,
        )
        if scp_result.returncode == 0:
            log("RUN", f"Results saved to: {results_dir}")
            return str(results_dir)
        else:
            log(
                "RUN",
                f"Warning: Failed to download results: {scp_result.stderr[:200] if scp_result.stderr else 'unknown'}",
            )
            return None

    # Copy tarball from container to VM host
    copy_tar_cmd = "docker cp winarena:/tmp/results.tar.gz /tmp/results.tar.gz"
    subprocess.run(
        ["ssh", *SSH_OPTS, f"azureuser@{ip}", copy_tar_cmd], capture_output=True
    )

    # Download tarball
    local_tar = results_dir / "results.tar.gz"
    scp_result = subprocess.run(
        ["scp", *SSH_OPTS, f"azureuser@{ip}:/tmp/results.tar.gz", str(local_tar)],
        capture_output=True,
        text=True,
    )

    if scp_result.returncode != 0:
        log(
            "RUN",
            f"Warning: Failed to download tarball: {scp_result.stderr[:200] if scp_result.stderr else 'unknown'}",
        )
        return None

    # Extract tarball
    log("RUN", "Extracting results...")
    import tarfile

    try:
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=results_dir)
        local_tar.unlink()  # Remove tarball after extraction
    except Exception as e:
        log("RUN", f"Warning: Failed to extract: {e}")
        log("RUN", f"Tarball saved at: {local_tar}")

    # Clean up remote tarball
    subprocess.run(
        ["ssh", *SSH_OPTS, f"azureuser@{ip}", "rm -f /tmp/results.tar.gz"],
        capture_output=True,
    )

    # List what we downloaded
    result_files = list(results_dir.glob("**/*"))
    log("RUN", f"Downloaded {len(result_files)} files to {results_dir}/")

    # Show summary if available
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        import json

        try:
            with open(summary_file) as f:
                summary = json.load(f)
            log("RUN", f"Summary: {json.dumps(summary, indent=2)[:500]}")
        except Exception:
            pass

    return str(results_dir)


def cmd_download(args):
    """Download benchmark results from VM."""
    init_logging()

    ip = get_vm_ip()
    if not ip:
        log("DOWNLOAD", "ERROR: VM not found")
        return 1

    log("DOWNLOAD", "Downloading benchmark results...")
    result_path = download_benchmark_results(ip)

    if result_path:
        log("DOWNLOAD", f"Results saved to: {result_path}")
        return 0
    else:
        log("DOWNLOAD", "Failed to download results")
        return 1


def cmd_analyze(args):
    """Analyze benchmark results from downloaded logs."""
    import re
    from collections import defaultdict

    results_dir = (
        Path(args.results_dir) if args.results_dir else Path("benchmark_results")
    )

    # Find most recent results if no specific dir given
    if args.results_dir:
        target_dir = Path(args.results_dir)
    else:
        dirs = sorted(results_dir.glob("waa_results_*"), reverse=True)
        if not dirs:
            print("No results found in benchmark_results/")
            print("Run 'cli download' first to get results from VM")
            return 1
        target_dir = dirs[0]

    print(f"Analyzing: {target_dir}")
    print("=" * 60)

    # Find log files
    log_files = list(target_dir.glob("logs/normal-*.log"))
    if not log_files:
        print("No log files found")
        return 1

    # Parse results
    tasks = []
    current_task = None
    pending_domain = None

    for log_file in sorted(log_files):
        with open(log_file) as f:
            for line in f:
                # Strip ANSI codes
                clean = re.sub(r"\x1b\[[0-9;]*m", "", line)

                # Domain comes before Example ID
                if "[Domain]:" in clean:
                    match = re.search(r"\[Domain\]: (.+)", clean)
                    if match:
                        pending_domain = match.group(1).strip()

                # Task start (Example ID comes after Domain)
                if "[Example ID]:" in clean:
                    match = re.search(r"\[Example ID\]: (.+)", clean)
                    if match:
                        current_task = {
                            "id": match.group(1).strip(),
                            "domain": pending_domain,
                            "reward": None,
                            "error": None,
                        }
                        pending_domain = None

                # Task result
                if "Reward:" in clean and current_task:
                    match = re.search(r"Reward: ([0-9.]+)", clean)
                    if match:
                        current_task["reward"] = float(match.group(1))
                        tasks.append(current_task)
                        current_task = None

                # Task error
                if "Exception in" in clean and current_task:
                    match = re.search(r"Exception in .+: (.+)", clean)
                    if match:
                        current_task["error"] = match.group(1).strip()
                        current_task["reward"] = 0.0
                        tasks.append(current_task)
                        current_task = None

    # Summary
    print(f"\nTotal tasks attempted: {len(tasks)}")

    if not tasks:
        print("No completed tasks found")
        return 0

    # Success rate
    successes = sum(1 for t in tasks if t["reward"] and t["reward"] > 0)
    print(f"Successful: {successes} ({100 * successes / len(tasks):.1f}%)")

    # By domain
    by_domain = defaultdict(list)
    for t in tasks:
        by_domain[t["domain"] or "unknown"].append(t)

    print("\nBy domain:")
    for domain in sorted(by_domain.keys()):
        domain_tasks = by_domain[domain]
        domain_success = sum(1 for t in domain_tasks if t["reward"] and t["reward"] > 0)
        print(
            f"  {domain}: {domain_success}/{len(domain_tasks)} ({100 * domain_success / len(domain_tasks):.1f}%)"
        )

    # Errors
    errors = [t for t in tasks if t.get("error")]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for t in errors[:5]:  # Show first 5
            print(f"  {t['id']}: {t['error'][:50]}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    return 0


def cmd_tasks(args):
    """List available WAA benchmark tasks."""
    ip = get_vm_ip()
    if not ip:
        print("ERROR: VM not found")
        return 1

    print("Fetching available tasks from WAA container...")
    print("-" * 60)

    # Get list of domains (subdirectories in examples/)
    result = subprocess.run(
        [
            "ssh",
            *SSH_OPTS,
            f"azureuser@{ip}",
            "docker exec winarena ls /client/evaluation_examples_windows/examples/",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("ERROR: Could not fetch domain list")
        return 1

    domains = result.stdout.strip().split("\n")

    # Count tasks per domain
    domain_tasks = {}
    total_tasks = 0

    for domain in domains:
        if not domain:
            continue
        count_result = subprocess.run(
            [
                "ssh",
                *SSH_OPTS,
                f"azureuser@{ip}",
                f"docker exec winarena ls /client/evaluation_examples_windows/examples/{domain}/ 2>/dev/null | wc -l",
            ],
            capture_output=True,
            text=True,
        )
        count = (
            int(count_result.stdout.strip())
            if count_result.stdout.strip().isdigit()
            else 0
        )
        domain_tasks[domain] = count
        total_tasks += count

    # Print summary
    print(f"Total tasks: {total_tasks}")
    print(f"Domains: {len(domains)}")
    print()

    # Print by domain
    for domain in sorted(domain_tasks.keys()):
        count = domain_tasks[domain]
        print(f"  {domain}: {count} tasks")

        if args.verbose and count > 0:
            # List actual task IDs
            tasks_result = subprocess.run(
                [
                    "ssh",
                    *SSH_OPTS,
                    f"azureuser@{ip}",
                    f"docker exec winarena ls /client/evaluation_examples_windows/examples/{domain}/",
                ],
                capture_output=True,
                text=True,
            )
            for task_file in tasks_result.stdout.strip().split("\n")[:5]:  # Limit to 5
                task_id = task_file.replace(".json", "")
                print(f"    - {task_id}")
            if count > 5:
                print(f"    ... and {count - 5} more")

    print()
    print("Usage examples:")
    print("  Run all notepad tasks:  cli_v2 run --domain notepad")
    print("  Run all chrome tasks:   cli_v2 run --domain chrome")
    print(
        "  Run specific task:      cli_v2 run --task 366de66e-cbae-4d72-b042-26390db2b145-WOS"
    )

    return 0


def cmd_deallocate(args):
    """Stop VM (preserves disk, stops billing)."""
    init_logging()
    log("DEALLOCATE", f"Deallocating VM '{VM_NAME}'...")

    result = subprocess.run(
        ["az", "vm", "deallocate", "-g", RESOURCE_GROUP, "-n", VM_NAME],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        log("DEALLOCATE", "VM deallocated (billing stopped)")
        log("DEALLOCATE", "Use 'vm-start' to resume")
        return 0
    else:
        log("DEALLOCATE", f"ERROR: {result.stderr}")
        return 1


def cmd_vm_start(args):
    """Start a deallocated VM."""
    init_logging()
    log("VM-START", f"Starting VM '{VM_NAME}'...")

    result = subprocess.run(
        ["az", "vm", "start", "-g", RESOURCE_GROUP, "-n", VM_NAME],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        ip = get_vm_ip()
        log("VM-START", f"VM started: {ip}")
        log("VM-START", "Run 'build' then 'start' to launch WAA container")
        return 0
    else:
        log("VM-START", f"ERROR: {result.stderr}")
        return 1


def cmd_exec(args):
    """Run command on VM host."""
    ip = get_vm_ip()
    if not ip:
        print("ERROR: VM not found or not running")
        return 1

    cmd = args.cmd
    if not cmd:
        print("ERROR: --cmd is required")
        return 1

    result = ssh_run(ip, cmd, stream=True)
    return result.returncode


def cmd_docker_exec(args):
    """Run command inside winarena container."""
    ip = get_vm_ip()
    if not ip:
        print("ERROR: VM not found or not running")
        return 1

    cmd = args.cmd
    if not cmd:
        print("ERROR: --cmd is required")
        return 1

    docker_cmd = f"docker exec winarena {cmd}"
    result = ssh_run(ip, docker_cmd, stream=True)
    return result.returncode


def cmd_vnc(args):
    """Open VNC to view Windows desktop via SSH tunnel."""
    ip = get_vm_ip()
    if not ip:
        print("ERROR: VM not found or not running")
        return 1

    print(f"Setting up SSH tunnel to VM ({ip})...")
    print("VNC will be available at: http://localhost:8006")
    print("-" * 60)

    # Kill any existing tunnel on port 8006
    subprocess.run(["pkill", "-f", "ssh.*8006:localhost:8006"], capture_output=True)

    # Start SSH tunnel in background
    tunnel_proc = subprocess.Popen(
        ["ssh", *SSH_OPTS, "-N", "-L", "8006:localhost:8006", f"azureuser@{ip}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Give tunnel a moment to establish
    time.sleep(2)

    # Check if tunnel is running
    if tunnel_proc.poll() is not None:
        print("ERROR: SSH tunnel failed to start")
        return 1

    print(f"SSH tunnel established (PID: {tunnel_proc.pid})")

    # Open browser
    import webbrowser

    vnc_url = "http://localhost:8006"
    print(f"Opening {vnc_url} in browser...")
    webbrowser.open(vnc_url)

    print()
    print("VNC is now accessible at: http://localhost:8006")
    print("Press Ctrl+C to close the tunnel")
    print("-" * 60)

    try:
        # Keep tunnel alive
        tunnel_proc.wait()
    except KeyboardInterrupt:
        print("\nClosing SSH tunnel...")
        tunnel_proc.terminate()

    return 0


def _show_benchmark_progress(ip: str) -> int:
    """Show benchmark progress with estimated completion time.

    Parses the run log to count completed tasks and estimate remaining time.
    """
    # Find the most recent run log
    result = ssh_run(
        ip, "ls -t /home/azureuser/cli_logs/run_*.log 2>/dev/null | head -1"
    )
    log_file = result.stdout.strip()

    if not log_file:
        print("No benchmark running. Start one with: run --num-tasks N")
        return 1

    # Get task count and timestamps
    result = ssh_run(
        ip,
        f"""
        echo "=== WAA Benchmark Progress ==="
        echo ""

        # Count completed tasks (each "Result:" line = 1 task done)
        COMPLETED=$(grep -c "Result:" {log_file} 2>/dev/null || echo 0)
        # Count total tasks from task list (sum of all domain counts)
        TOTAL=$(grep -A20 "Left tasks:" {log_file} | grep -E "^[a-z_]+: [0-9]+" | awk -F': ' '{{sum+=$2}} END {{print sum}}')
        [ -z "$TOTAL" ] || [ "$TOTAL" -eq 0 ] && TOTAL=154

        # Get timestamps
        FIRST_TS=$(grep -oE '\\[2026-[0-9-]+ [0-9:]+' {log_file} | head -1 | tr -d '[')
        LAST_TS=$(grep -oE '\\[2026-[0-9-]+ [0-9:]+' {log_file} | tail -1 | tr -d '[')

        echo "Log: {log_file}"
        echo "Started: $FIRST_TS"
        echo "Latest:  $LAST_TS"
        echo ""
        echo "Tasks completed: $COMPLETED / $TOTAL"

        # Calculate elapsed minutes
        if [ -n "$FIRST_TS" ] && [ -n "$LAST_TS" ]; then
            START_H=$(echo "$FIRST_TS" | awk '{{print $2}}' | cut -d: -f1)
            START_M=$(echo "$FIRST_TS" | awk '{{print $2}}' | cut -d: -f2)
            NOW_H=$(echo "$LAST_TS" | awk '{{print $2}}' | cut -d: -f1)
            NOW_M=$(echo "$LAST_TS" | awk '{{print $2}}' | cut -d: -f2)

            ELAPSED_MIN=$(( (NOW_H - START_H) * 60 + (NOW_M - START_M) ))
            echo "Elapsed: $ELAPSED_MIN minutes"

            if [ "$COMPLETED" -gt 0 ] && [ "$ELAPSED_MIN" -gt 0 ]; then
                MIN_PER_TASK=$((ELAPSED_MIN / COMPLETED))
                REMAINING=$((TOTAL - COMPLETED))
                EST_MIN=$((REMAINING * MIN_PER_TASK))
                EST_H=$((EST_MIN / 60))
                EST_M=$((EST_MIN % 60))

                echo ""
                echo "Avg time per task: ~$MIN_PER_TASK min"
                echo "Remaining tasks: $REMAINING"
                echo "Estimated remaining: ~${{EST_H}}h ${{EST_M}}m"

                # Progress bar
                PCT=$((COMPLETED * 100 / TOTAL))
                echo ""
                echo "Progress: $PCT% [$COMPLETED/$TOTAL]"
            fi
        fi
        """,
    )
    print(result.stdout)
    return 0


def _show_run_logs(ip: str, follow: bool = False, tail: Optional[int] = None) -> int:
    """Show the most recent run command log file.

    Args:
        ip: VM IP address
        follow: If True, use tail -f to stream the log
        tail: Number of lines to show (default: entire file or 100 for follow)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Find the most recent run log file
    result = ssh_run(
        ip, "ls -t /home/azureuser/cli_logs/run_*.log 2>/dev/null | head -1"
    )
    log_file = result.stdout.strip()

    if not log_file:
        print("No run logs found at /home/azureuser/cli_logs/run_*.log")
        print("Run a benchmark first: cli_v2 run --task <task_id>")
        return 1

    print(f"Run log: {log_file}")
    print("-" * 60)

    if follow:
        # Stream the log file
        print("Streaming log (Ctrl+C to stop)...")
        subprocess.run(["ssh", *SSH_OPTS, f"azureuser@{ip}", f"tail -f {log_file}"])
    else:
        # Show the log file contents
        if tail:
            cmd = f"tail -n {tail} {log_file}"
        else:
            # Check file size first - if small, cat it; if large, use tail
            size_result = ssh_run(ip, f"wc -l < {log_file}")
            line_count = (
                int(size_result.stdout.strip())
                if size_result.stdout.strip().isdigit()
                else 0
            )

            if line_count <= 200:
                cmd = f"cat {log_file}"
            else:
                print(
                    f"(Showing last 100 of {line_count} lines, use --tail N for more)"
                )
                cmd = f"tail -n 100 {log_file}"

        subprocess.run(["ssh", *SSH_OPTS, f"azureuser@{ip}", cmd])

    return 0


def cmd_logs(args):
    """Show comprehensive logs from the WAA container.

    Default behavior shows all relevant logs (docker, storage, probe status).
    Use --follow to stream docker logs continuously.
    Use --run to show run command output instead of container logs.
    Use --progress to show benchmark progress and ETA.
    """
    ip = get_vm_ip()
    if not ip:
        print("ERROR: VM not found")
        return 1

    # Handle --progress flag: show benchmark progress
    if getattr(args, "progress", False):
        return _show_benchmark_progress(ip)

    # Handle --run flag: show run command output
    if args.run:
        return _show_run_logs(ip, args.follow, args.tail)

    # Check if container exists
    result = ssh_run(ip, "docker ps -a --filter name=winarena --format '{{.Status}}'")
    container_status = result.stdout.strip()
    container_exists = bool(container_status)

    # If --follow, stream the most relevant logs
    if args.follow:
        # Priority 1: If container is running, stream container logs
        if container_exists and "Up" in container_status:
            print(f"Streaming container logs from VM ({ip}):")
            print("Press Ctrl+C to stop")
            print("-" * 60)
            subprocess.run(
                ["ssh", *SSH_OPTS, f"azureuser@{ip}", "docker logs -f winarena 2>&1"]
            )
            return 0

        # Priority 2: Check for active docker build
        result = ssh_run(
            ip,
            "pgrep -f 'docker build' >/dev/null && echo BUILD_RUNNING || echo NO_BUILD",
        )
        if "BUILD_RUNNING" in result.stdout:
            print(f"Docker build in progress on VM ({ip})")
            print("Streaming build logs (Ctrl+C to stop):")
            print("-" * 60)
            # Find and tail the most recent build log
            subprocess.run(
                [
                    "ssh",
                    *SSH_OPTS,
                    f"azureuser@{ip}",
                    "tail -f $(ls -t ~/cli_logs/build_*.log 2>/dev/null | head -1) 2>/dev/null || "
                    "tail -f ~/build.log 2>/dev/null || "
                    "echo 'No build logs found - build may have just started'",
                ]
            )
            return 0

        # Priority 3: No container, no build - show helpful message
        print(f"Container 'winarena' not running on VM ({ip})")
        print()
        # Check if image exists
        result = ssh_run(
            ip, "docker images waa-auto:latest --format '{{.Repository}}:{{.Tag}}'"
        )
        if result.stdout.strip():
            print("Image 'waa-auto:latest' is ready.")
            print("Run: uv run python -m openadapt_ml.benchmarks.cli_v2 start")
        else:
            print("Image not yet built.")
            print("Run: uv run python -m openadapt_ml.benchmarks.cli_v2 build")
        return 1

    # Default: show comprehensive status
    import sys

    print(f"WAA Status ({ip})")
    print("=" * 60)
    sys.stdout.flush()

    # Docker images
    print("\n[Docker Images]", flush=True)
    subprocess.run(
        [
            "ssh",
            *SSH_OPTS,
            f"azureuser@{ip}",
            "docker images --format 'table {{.Repository}}\\t{{.Tag}}\\t{{.Size}}' 2>/dev/null | head -5",
        ]
    )

    # Container status
    print("\n[Container]", flush=True)
    if container_exists:
        print(f"  Status: {container_status}", flush=True)
    else:
        print("  Container 'winarena' not created yet", flush=True)
        # Check for active build
        result = ssh_run(
            ip,
            "pgrep -f 'docker build' >/dev/null && echo BUILD_RUNNING || echo NO_BUILD",
        )
        if "BUILD_RUNNING" in result.stdout:
            print("  Docker build in progress...", flush=True)

    # Only show these sections if container exists
    if container_exists and "Up" in container_status:
        # Storage info
        print("\n[Storage]", flush=True)
        subprocess.run(
            [
                "ssh",
                *SSH_OPTS,
                f"azureuser@{ip}",
                "docker exec winarena sh -c '"
                'echo "  Total: $(du -sh /storage/ 2>/dev/null | cut -f1)"; '
                'ls -lh /storage/*.img 2>/dev/null | awk "{print \\"  Disk image: \\" \\$5}" || true'
                "'",
            ]
        )

        # QEMU VM status
        print("\n[QEMU VM]", flush=True)
        subprocess.run(
            [
                "ssh",
                *SSH_OPTS,
                f"azureuser@{ip}",
                "docker exec winarena sh -c '"
                "QPID=$(pgrep -f qemu-system 2>/dev/null | head -1); "
                'if [ -n "$QPID" ]; then '
                '  echo "  Status: Running (PID $QPID)"; '
                '  ps -o %cpu,%mem,etime -p $QPID 2>/dev/null | tail -1 | awk "{print \\"  CPU: \\" \\$1 \\"%, MEM: \\" \\$2 \\"%, Uptime: \\" \\$3}"; '
                "else "
                '  echo "  Status: Not running"; '
                "fi"
                "'",
            ]
        )

        # WAA server probe
        print("\n[WAA Server]", flush=True)
        subprocess.run(
            [
                "ssh",
                *SSH_OPTS,
                f"azureuser@{ip}",
                "docker exec winarena curl -s --max-time 5 http://172.30.0.2:5000/probe 2>/dev/null && echo ' (READY)' || echo 'Not ready (Windows installing - check VNC for progress)'",
            ]
        )

        # Windows install log (written by install.bat to Samba share at Z:\install_log.txt)
        # The Samba share \\host.lan\Data maps to /tmp/smb inside the container
        result = ssh_run(
            ip, "docker exec winarena cat /tmp/smb/install_log.txt 2>/dev/null | wc -l"
        )
        install_log_lines = result.stdout.strip()
        if install_log_lines and install_log_lines != "0":
            print("\n[Windows Install Log]", flush=True)
            # Show last 10 lines of the install log (shows current step like [5/14] Installing Git...)
            subprocess.run(
                [
                    "ssh",
                    *SSH_OPTS,
                    f"azureuser@{ip}",
                    "docker exec winarena tail -10 /tmp/smb/install_log.txt 2>/dev/null",
                ]
            )

        # Recent docker logs
        tail_lines = args.tail if args.tail else 20
        print(f"\n[Recent Logs (last {tail_lines} lines)]", flush=True)
        print("-" * 60, flush=True)
        subprocess.run(
            [
                "ssh",
                *SSH_OPTS,
                f"azureuser@{ip}",
                f"docker logs --tail {tail_lines} winarena 2>&1",
            ]
        )

        print("\n" + "=" * 60, flush=True)
        print("VNC: ssh -L 8006:localhost:8006 azureuser@" + ip, flush=True)
        print("     Then open http://localhost:8006", flush=True)
        print("     (Windows installation % visible on VNC screen)", flush=True)
    else:
        # Show next steps
        print("\n[Next Steps]")
        result = ssh_run(ip, "docker images waa-auto:latest --format '{{.Repository}}'")
        if result.stdout.strip():
            print("  Image ready. Run: cli_v2 start")
        else:
            print("  Build image first. Run: cli_v2 build")

    return 0


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="WAA Benchmark CLI v2 - Minimal working CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full setup workflow (vanilla WAA)
  %(prog)s create          # Create Azure VM
  %(prog)s pull            # Pull vanilla WAA image
  %(prog)s start           # Start container + Windows
  %(prog)s probe --wait    # Wait for WAA server
  %(prog)s run --num-tasks 1 --agent navi   # Run benchmark
  %(prog)s deallocate      # Stop billing

  # Monitor in separate terminal
  %(prog)s logs --docker   # Docker container logs
  %(prog)s vnc             # View Windows desktop

  # Cleanup
  %(prog)s delete
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = subparsers.add_parser("create", help="Create Azure VM")
    p_create.add_argument(
        "--fast",
        action="store_true",
        help="Use larger VM (D8ds_v5, $0.38/hr) for ~30%% faster install, ~40%% faster eval",
    )
    p_create.set_defaults(func=cmd_create)

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete VM and all resources")
    p_delete.set_defaults(func=cmd_delete)

    # status
    p_status = subparsers.add_parser("status", help="Show VM status")
    p_status.set_defaults(func=cmd_status)

    # build
    p_build = subparsers.add_parser(
        "build", help="Build WAA image from waa_deploy/Dockerfile"
    )
    p_build.set_defaults(func=cmd_build)

    # start
    p_start = subparsers.add_parser("start", help="Start WAA container")
    p_start.add_argument(
        "--fresh", action="store_true", help="Clean storage for fresh Windows install"
    )
    p_start.add_argument(
        "--no-vnc", action="store_true", help="Don't auto-launch VNC viewer"
    )
    p_start.add_argument(
        "--fast",
        action="store_true",
        help="Allocate more CPU/RAM to QEMU (use with D8ds_v5 VM)",
    )
    p_start.set_defaults(func=cmd_start)

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop and remove WAA container")
    p_stop.add_argument(
        "--clean", action="store_true", help="Also clean Windows storage"
    )
    p_stop.set_defaults(func=cmd_stop)

    # probe
    p_probe = subparsers.add_parser("probe", help="Check if WAA server is ready")
    p_probe.add_argument("--wait", action="store_true", help="Wait until ready")
    p_probe.add_argument(
        "--timeout", type=int, default=1200, help="Timeout in seconds (default: 1200)"
    )
    p_probe.set_defaults(func=cmd_probe)

    # run
    p_run = subparsers.add_parser(
        "run", help="Run benchmark tasks (uses vanilla WAA navi agent)"
    )
    p_run.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Number of tasks to run (ignored if --task specified)",
    )
    p_run.add_argument("--task", help="Specific task ID to run")
    p_run.add_argument(
        "--domain",
        default="all",
        help="Domain filter (e.g., 'notepad', 'chrome', 'all')",
    )
    p_run.add_argument(
        "--model", default="gpt-4o", help="Model for navi agent (default: gpt-4o)"
    )
    p_run.add_argument(
        "--api-key", help="OpenAI API key (or set OPENAI_API_KEY in .env)"
    )
    p_run.add_argument(
        "--no-download", action="store_true", help="Skip downloading results"
    )
    p_run.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Worker ID for parallel execution (0-indexed)",
    )
    p_run.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of parallel workers",
    )
    p_run.set_defaults(func=cmd_run)

    # download
    p_download = subparsers.add_parser(
        "download", help="Download benchmark results from VM"
    )
    p_download.set_defaults(func=cmd_download)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze benchmark results")
    p_analyze.add_argument(
        "--results-dir",
        help="Results directory (default: most recent in benchmark_results/)",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # tasks
    p_tasks = subparsers.add_parser("tasks", help="List available WAA benchmark tasks")
    p_tasks.add_argument(
        "--verbose", "-v", action="store_true", help="Show all task IDs"
    )
    p_tasks.set_defaults(func=cmd_tasks)

    # deallocate
    p_dealloc = subparsers.add_parser("deallocate", help="Stop VM (preserves disk)")
    p_dealloc.set_defaults(func=cmd_deallocate)

    # vm-start
    p_vmstart = subparsers.add_parser("vm-start", help="Start a deallocated VM")
    p_vmstart.set_defaults(func=cmd_vm_start)

    # logs
    p_logs = subparsers.add_parser("logs", help="Show WAA status and logs")
    p_logs.add_argument(
        "--follow", "-f", action="store_true", help="Stream docker logs continuously"
    )
    p_logs.add_argument(
        "--tail", "-n", type=int, help="Number of log lines to show (default: 20)"
    )
    p_logs.add_argument(
        "--run",
        action="store_true",
        help="Show run command output instead of container logs",
    )
    p_logs.add_argument(
        "--progress",
        "-p",
        action="store_true",
        help="Show benchmark progress and estimated completion time",
    )
    p_logs.set_defaults(func=cmd_logs)

    # exec
    p_exec = subparsers.add_parser("exec", help="Run command on VM host")
    p_exec.add_argument("--cmd", required=True, help="Command to run")
    p_exec.set_defaults(func=cmd_exec)

    # docker-exec
    p_dexec = subparsers.add_parser(
        "docker-exec", help="Run command inside winarena container"
    )
    p_dexec.add_argument("--cmd", required=True, help="Command to run")
    p_dexec.set_defaults(func=cmd_docker_exec)

    # vnc
    p_vnc = subparsers.add_parser(
        "vnc", help="Open VNC to view Windows desktop via SSH tunnel"
    )
    p_vnc.set_defaults(func=cmd_vnc)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
