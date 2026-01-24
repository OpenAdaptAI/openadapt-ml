# =============================================================================
# WAA (Windows Agent Arena) Docker Image - Simplified
# =============================================================================
#
# This image follows vanilla WAA's Azure mode approach:
# - Uses native dockurr/windows OEM mechanism (copies /oem → C:\OEM automatically)
# - Patches scripts to use C:\oem instead of \\host.lan\Data (like vanilla WAA Azure mode)
# - No custom FirstLogonCommands or samba.sh patching needed
#
# =============================================================================

FROM dockurr/windows:latest

# -----------------------------------------------------------------------------
# Copy official WAA components from windowsarena/winarena
# -----------------------------------------------------------------------------

# Copy benchmark client scripts
COPY --from=windowsarena/winarena:latest /entry.sh /entry.sh
COPY --from=windowsarena/winarena:latest /entry_setup.sh /entry_setup.sh
COPY --from=windowsarena/winarena:latest /start_client.sh /start_client.sh

# Copy the Python benchmark client code
COPY --from=windowsarena/winarena:latest /client /client

# Copy Windows setup scripts (install.bat, setup.ps1, etc.) to /oem
# dockurr/windows will automatically copy /oem to C:\OEM and run install.bat
COPY --from=windowsarena/winarena:latest /oem /oem

# Copy our WAA server startup script
COPY start_waa_server.bat /oem/start_waa_server.bat

# -----------------------------------------------------------------------------
# DO NOT replace dockurr/windows's autounattend.xml - it handles fresh OOBE properly
# Instead, we rely on dockurr/windows's OEM mechanism:
# 1. It copies /oem → C:\OEM automatically during Windows setup
# 2. It runs C:\OEM\install.bat via autounattend.xml FirstLogonCommands
# We just need to ensure our install.bat in /oem sets up WAA correctly
# -----------------------------------------------------------------------------

# Patch autounattend.xml to add InstallFrom element (prevents "Select OS" dialog)
# This modifies dockurr/windows's existing XML rather than replacing it
RUN for xml in /run/assets/win11x64.xml /run/assets/win11x64-enterprise-eval.xml; do \
        if [ -f "$xml" ] && ! grep -q "InstallFrom" "$xml"; then \
            sed -i 's|<InstallTo>|<InstallFrom>\n            <MetaData wcm:action="add">\n              <Key>/IMAGE/INDEX</Key>\n              <Value>1</Value>\n            </MetaData>\n          </InstallFrom>\n          <InstallTo>|' "$xml"; \
        fi; \
    done && echo "Added InstallFrom element for automatic image selection"

# -----------------------------------------------------------------------------
# Azure mode: Patch scripts to use C:\oem instead of \\host.lan\Data
# This matches vanilla WAA's Dockerfile-WinArena Azure mode behavior
# -----------------------------------------------------------------------------

RUN WINDOWS_DATA_FOLDER='\\\\host.lan\\Data' && \
    WINDOWS_OEM_FOLDER='C:\\oem' && \
    sed -i "s|${WINDOWS_DATA_FOLDER}|${WINDOWS_OEM_FOLDER}|g" /oem/install.bat && \
    sed -i "s|${WINDOWS_DATA_FOLDER}|${WINDOWS_OEM_FOLDER}|g" /oem/on-logon.ps1 2>/dev/null || true && \
    sed -i "s|${WINDOWS_DATA_FOLDER}|${WINDOWS_OEM_FOLDER}|g" /oem/setup.ps1 && \
    echo "Patched scripts to use C:\\oem (Azure mode)"

# Also patch start_waa_server.bat if it has UNC paths
RUN sed -i 's|\\\\host.lan\\Data|C:\\oem|g' /oem/start_waa_server.bat 2>/dev/null || true

# -----------------------------------------------------------------------------
# Port forwarding: Forward port 5000 from container to Windows VM (172.30.0.2)
# dockurr/windows doesn't auto-forward ports to the Windows VM inside QEMU
# -----------------------------------------------------------------------------

RUN printf '#!/bin/bash\n\
while ! grep -q "172.30.0.2" /var/lib/misc/dnsmasq.leases 2>/dev/null; do sleep 5; done\n\
while true; do nc -lp 5000 -c "nc 172.30.0.2 5000" 2>/dev/null || sleep 1; done\n\
' > /port_forward.sh && chmod +x /port_forward.sh

# Inject port forwarder into samba.sh (runs after network is up)
RUN sed -i '/^return 0$/i nohup /port_forward.sh >/dev/null 2>\&1 \&' /run/samba.sh && \
    echo "Added port forwarder"

# -----------------------------------------------------------------------------
# Create start_vm.sh that uses dockurr/windows entrypoint
# -----------------------------------------------------------------------------

RUN printf '#!/bin/bash\n/usr/bin/tini -s /run/entry.sh\n' > /start_vm.sh && chmod +x /start_vm.sh

# -----------------------------------------------------------------------------
# Patch IP addresses: official WAA uses 20.20.20.21, dockurr/windows uses 172.30.0.2
# (Same as vanilla WAA's Dockerfile-WinArena lines 48-49)
# -----------------------------------------------------------------------------

RUN sed -i 's|20\.20\.20\.21|172.30.0.2|g' /entry_setup.sh /entry.sh /start_client.sh && \
    find /client -name "*.py" -exec sed -i 's|20\.20\.20\.21|172.30.0.2|g' {} \; && \
    echo "Patched IP addresses"

# -----------------------------------------------------------------------------
# Install Python dependencies for benchmark client (runs on Linux host, not Windows)
# NOTE: Must be installed BEFORE any python3 commands
# -----------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip tesseract-ocr libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir --break-system-packages \
    gymnasium openai anthropic tiktoken pyyaml tenacity httpx \
    pillow pytesseract requests flask numpy pandas

# -----------------------------------------------------------------------------
# Add API-backed agent support (Claude / GPT)
# -----------------------------------------------------------------------------

COPY api_agent.py /client/mm_agents/api_agent.py

# Patch run.py to support api-claude and api-openai agents
RUN python3 -c "import re; \
f = open('/client/run.py', 'r'); c = f.read(); f.close(); \
patch = '''    elif cfg_args[\"agent_name\"] in [\"api-claude\", \"api-openai\"]:\n        from mm_agents.api_agent import ApiAgent\n        provider = \"anthropic\" if cfg_args[\"agent_name\"] == \"api-claude\" else \"openai\"\n        agent = ApiAgent(provider=provider, temperature=args.temperature)\n'''; \
c = c.replace('    else:\\n        raise ValueError', patch + '    else:\\n        raise ValueError'); \
f = open('/client/run.py', 'w'); f.write(c); f.close(); \
print('Patched run.py for API agents')"

# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

ENV YRES="900"
ENV XRES="1440"
ENV RAM_SIZE="6G"
ENV CPU_CORES="4"
ENV DISK_SIZE="30G"
ENV VERSION="11e"
ENV ARGUMENTS="-qmp tcp:0.0.0.0:7200,server,nowait"

EXPOSE 8006 5000 7200 3389

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/entry.sh --start-client false"]
