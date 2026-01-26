# =============================================================================
# WAA (Windows Agent Arena) - Standalone Dockerfile
# =============================================================================
#
# DEFINITIVE APPROACH - No circular dependencies, no script patching
#
# This uses vanilla WAA's "dev mode" which expects files at \\host.lan\Data
# (the Samba share at /tmp/smb/). No patching of script paths required.
#
# Key features:
# - Uses modern dockurr/windows:latest (auto-downloads Windows 11)
# - VERSION=11e = Enterprise Evaluation (no license key required)
# - Fully unattended installation
# - No 15GB image pull required
# - Uses vanilla WAA scripts without modification (except IP address)
#
# Build from repo root:
#   docker build -f openadapt_ml/benchmarks/waa_deploy/Dockerfile -t waa:latest .
#
# Run:
#   docker run --rm --device=/dev/kvm --cap-add NET_ADMIN \
#     -p 8006:8006 -p 5000:5000 \
#     -v /path/to/storage:/storage \
#     waa:latest
#
# =============================================================================

FROM dockurr/windows:latest

# -----------------------------------------------------------------------------
# Copy WAA components from LOCAL vendor submodule
# Build context must be repo root for these paths to work
# -----------------------------------------------------------------------------

# Entry scripts (for benchmark client on Linux side)
COPY vendor/WindowsAgentArena/src/win-arena-container/entry.sh /entry.sh
COPY vendor/WindowsAgentArena/src/win-arena-container/entry_setup.sh /entry_setup.sh
COPY vendor/WindowsAgentArena/src/win-arena-container/start_client.sh /start_client.sh
COPY vendor/WindowsAgentArena/src/win-arena-container/start_vm.sh /start_vm.sh

# Python benchmark client (runs on Linux side)
COPY vendor/WindowsAgentArena/src/win-arena-container/client /client

# Windows setup scripts → /waa-setup/ (staging location)
# These get copied to /tmp/smb/ (Samba share) at container startup
# Windows accesses them at \\host.lan\Data
COPY vendor/WindowsAgentArena/src/win-arena-container/vm/setup /waa-setup

# Our custom startup script (more robust than vanilla)
COPY openadapt_ml/benchmarks/waa_deploy/start_waa_server.bat /waa-setup/start_waa_server.bat

# -----------------------------------------------------------------------------
# Use DEV unattend XML (expects files at \\host.lan\Data, not C:\oem)
# This is vanilla WAA's approach for local development
# -----------------------------------------------------------------------------

COPY vendor/WindowsAgentArena/src/win-arena-container/vm/unattend-files/dev_win11x64-enterprise-eval.xml /run/assets/win11x64-enterprise-eval.xml

# -----------------------------------------------------------------------------
# Patch IP addresses: WAA uses 20.20.20.21, dockurr/windows uses 172.30.0.2
# This is the ONLY patching we need to do
# -----------------------------------------------------------------------------

RUN sed -i 's|20\.20\.20\.21|172.30.0.2|g' /entry.sh /entry_setup.sh /start_client.sh && \
    find /client -name "*.py" -exec sed -i 's|20\.20\.20\.21|172.30.0.2|g' {} \; && \
    chmod +x /*.sh && \
    echo "Patched IP addresses to 172.30.0.2"

# -----------------------------------------------------------------------------
# Copy WAA setup files to Samba share when container starts
# /tmp/smb/ is created by dockurr/windows at runtime
# Windows sees this as \\host.lan\Data
# -----------------------------------------------------------------------------

RUN sed -i '/^return 0$/i \
# Copy WAA setup files to Samba share\n\
cp -r /waa-setup/* /tmp/smb/ 2>/dev/null || true\n\
chmod -R 777 /tmp/smb/ 2>/dev/null || true\n\
echo "WAA setup files copied to Samba share"' /run/samba.sh && \
    echo "Added WAA file copy to samba.sh"

# -----------------------------------------------------------------------------
# Port forwarding: Forward port 5000 from container to Windows VM
# dockurr/windows doesn't auto-forward ports to the QEMU VM
# -----------------------------------------------------------------------------

RUN printf '#!/bin/bash\n\
# Wait for Windows VM to get IP\n\
while ! grep -q "172.30.0.2" /var/lib/misc/dnsmasq.leases 2>/dev/null; do\n\
    sleep 5\n\
done\n\
echo "Windows VM is up, starting port forwarder"\n\
# Forward port 5000 to Windows VM\n\
while true; do\n\
    nc -lp 5000 -c "nc 172.30.0.2 5000" 2>/dev/null || sleep 1\n\
done\n\
' > /port_forward.sh && chmod +x /port_forward.sh

# Inject port forwarder into samba.sh (runs after network is up)
RUN sed -i '/^return 0$/i nohup /port_forward.sh >/dev/null 2>\&1 \&' /run/samba.sh && \
    echo "Added port forwarder to samba.sh"

# -----------------------------------------------------------------------------
# Install Python and dependencies for benchmark client (runs on Linux host)
# -----------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    tesseract-ocr libgl1 libglib2.0-0 ffmpeg netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install minimal Python deps for client (full deps installed in Windows)
RUN pip3 install --no-cache-dir --break-system-packages \
    requests flask pyyaml pillow

# -----------------------------------------------------------------------------
# Add API agent support (Claude / GPT)
# -----------------------------------------------------------------------------

COPY openadapt_ml/benchmarks/waa_deploy/api_agent.py /client/mm_agents/api_agent.py

# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

# VERSION=11e downloads Windows 11 Enterprise Evaluation
# This has a built-in GVLK key - NO LICENSE KEY REQUIRED
ENV VERSION="11e"

# VM resources
ENV RAM_SIZE="8G"
ENV CPU_CORES="4"
ENV DISK_SIZE="30G"

# Display resolution
ENV XRES="1440"
ENV YRES="900"

# QEMU arguments (expose QMP for monitoring)
ENV ARGUMENTS="-qmp tcp:0.0.0.0:7200,server,nowait"

# Expose ports
# 8006: noVNC web interface
# 5000: WAA Flask server (forwarded to Windows VM)
# 7200: QEMU QMP (monitoring)
# 3389: RDP (if needed)
EXPOSE 8006 5000 7200 3389

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

# Use dockurr/windows default entry which handles:
# - Windows ISO download (automatic with VERSION=11e)
# - QEMU/KVM setup
# - VNC server on 8006
# - Samba share at /tmp/smb/ (\\host.lan\Data)
ENTRYPOINT ["/usr/bin/tini", "-s", "/run/entry.sh"]
