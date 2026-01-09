# =============================================================================
# WAA (Windows Agent Arena) Docker Image
# =============================================================================
#
# This image combines:
# 1. dockurr/windows:latest - Modern base that auto-downloads Windows 11
# 2. windowsarena/winarena:latest - Official WAA benchmark client and scripts
#
# The official windowsarena/winarena uses an outdated dockurr/windows (v0.00)
# that doesn't auto-download Windows. This image fixes that while keeping
# full compatibility with the official WAA benchmark.
#
# Usage:
#   # Build the image
#   docker build -t waa-auto:latest .
#
#   # Run benchmark (after Windows is set up)
#   docker run --rm --device=/dev/kvm --cap-add NET_ADMIN \
#     -p 8006:8006 -p 5000:5000 -p 7200:7200 \
#     -v /path/to/storage:/storage \
#     -e OPENAI_API_KEY="your-key" \
#     waa-auto:latest \
#     "/entry.sh --start-client true --model gpt-4o --num-tasks 5"
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

# Copy our WAA server startup script
COPY start_waa_server.bat /oem/start_waa_server.bat

# Copy model weights (GroundingDINO, OmniParser, etc.)
COPY --from=windowsarena/winarena:latest /models /models

# Copy Windows setup scripts (install.bat, setup.ps1, etc.)
COPY --from=windowsarena/winarena:latest /oem /oem

# Copy OEM files AFTER dockurr/samba starts (which wipes /tmp/smb)
# Copy IMMEDIATELY (no delay) and SYNCHRONOUSLY (not backgrounded) to ensure
# files are available before Windows boots and runs FirstLogonCommands
RUN sed -i '/^return 0$/i cp -r /oem/* /tmp/smb/ 2>/dev/null || true' /run/samba.sh && \
    echo "Inserted OEM copy before return in samba.sh"

# Copy unattend.xml for automated Windows installation
COPY --from=windowsarena/winarena:latest /run/assets/win11x64-enterprise-eval.xml /run/assets/win11x64.xml

# -----------------------------------------------------------------------------
# Create start_vm.sh that uses our dockurr/windows entrypoint
# -----------------------------------------------------------------------------

RUN printf '#!/bin/bash\n/usr/bin/tini -s /run/entry.sh\n' > /start_vm.sh && chmod +x /start_vm.sh

# -----------------------------------------------------------------------------
# Patch IP addresses: official uses 20.20.20.21, dockurr/windows uses 172.30.0.2
# -----------------------------------------------------------------------------

# Patch entry scripts (must work - these files were just copied)
RUN sed -i 's|20.20.20.21|172.30.0.2|g' /entry_setup.sh && \
    sed -i 's|20.20.20.21|172.30.0.2|g' /entry.sh && \
    sed -i 's|20.20.20.21|172.30.0.2|g' /start_client.sh && \
    echo "Patched entry scripts"

# Patch client Python files
RUN find /client -name "*.py" -exec sed -i 's|20.20.20.21|172.30.0.2|g' {} \; && \
    echo "Patched client Python files"

# -----------------------------------------------------------------------------
# Add API-backed agent support (Claude Sonnet 4.5 / GPT-5.1)
# This allows using --agent api-claude or --agent api-openai instead of navi
# -----------------------------------------------------------------------------

# Copy api_agent.py to the client mm_agents directory
COPY api_agent.py /client/mm_agents/api_agent.py

# Patch run.py to support api-claude and api-openai agents
# This adds elif blocks after the "navi" agent handling
# Using Python to insert the patch with proper indentation
RUN python3 -c "import re; \
f = open('/client/run.py', 'r'); c = f.read(); f.close(); \
patch = '''    elif cfg_args[\"agent_name\"] in [\"api-claude\", \"api-openai\"]:\n        from mm_agents.api_agent import ApiAgent\n        provider = \"anthropic\" if cfg_args[\"agent_name\"] == \"api-claude\" else \"openai\"\n        agent = ApiAgent(provider=provider, temperature=args.temperature)\n'''; \
c = c.replace('raise ValueError(f\"Unknown agent name: {cfg_args', patch + '    raise ValueError(f\"Unknown agent name: {cfg_args'); \
f = open('/client/run.py', 'w'); f.write(c); f.close(); \
print('Patched run.py for API agents')"

# -----------------------------------------------------------------------------
# Fix Windows setup for automation
# -----------------------------------------------------------------------------

# Set password for AutoLogon (Windows 11 requires password for login)
RUN sed -i 's|<Value></Value>|<Value>docker</Value>|g' /run/assets/win11x64.xml 2>/dev/null || true
RUN sed -i 's|<Value />|<Value>docker</Value>|g' /run/assets/win11x64.xml 2>/dev/null || true

# Add firewall disable and other automation commands to FirstLogonCommands
# CRITICAL: Also create a scheduled task so WAA server starts on EVERY boot, not just first logon
RUN if grep -q "</FirstLogonCommands>" /run/assets/win11x64.xml; then \
    LAST_ORDER=$(grep -oP "Order>\K[0-9]+" /run/assets/win11x64.xml | sort -n | tail -1) && \
    N1=$((LAST_ORDER + 1)) && \
    N2=$((LAST_ORDER + 2)) && \
    N3=$((LAST_ORDER + 3)) && \
    N4=$((LAST_ORDER + 4)) && \
    N5=$((LAST_ORDER + 5)) && \
    N6=$((LAST_ORDER + 6)) && \
    sed -i "s|</FirstLogonCommands>|\
        <SynchronousCommand wcm:action=\"add\">\n\
          <Order>$N1</Order>\n\
          <CommandLine>netsh advfirewall set allprofiles state off</CommandLine>\n\
          <Description>Disable Windows Firewall</Description>\n\
        </SynchronousCommand>\n\
        <SynchronousCommand wcm:action=\"add\">\n\
          <Order>$N2</Order>\n\
          <CommandLine>powercfg /change standby-timeout-ac 0</CommandLine>\n\
          <Description>Disable sleep</Description>\n\
        </SynchronousCommand>\n\
        <SynchronousCommand wcm:action=\"add\">\n\
          <Order>$N3</Order>\n\
          <CommandLine>powercfg /change monitor-timeout-ac 0</CommandLine>\n\
          <Description>Disable monitor timeout</Description>\n\
        </SynchronousCommand>\n\
        <SynchronousCommand wcm:action=\"add\">\n\
          <Order>$N4</Order>\n\
          <CommandLine>reg add \"HKLM\\\\SOFTWARE\\\\Policies\\\\Microsoft\\\\Windows\\\\Personalization\" /v NoLockScreen /t REG_DWORD /d 1 /f</CommandLine>\n\
          <Description>Disable lock screen</Description>\n\
        </SynchronousCommand>\n\
        <SynchronousCommand wcm:action=\"add\">\n\
          <Order>$N5</Order>\n\
          <CommandLine>cmd /c start /wait \\\\\\\\host.lan\\\\Data\\\\install.bat</CommandLine>\n\
          <Description>Run WAA setup script to install Python, Chrome, etc.</Description>\n\
        </SynchronousCommand>\n\
        <SynchronousCommand wcm:action=\"add\">\n\
          <Order>$N6</Order>\n\
          <CommandLine>schtasks /create /tn \"WAAServer\" /tr \"\\\\\\\\host.lan\\\\Data\\\\start_waa_server.bat\" /sc onlogon /rl highest /f</CommandLine>\n\
          <Description>Create scheduled task for WAA server auto-start on every boot</Description>\n\
        </SynchronousCommand>\n\
        <SynchronousCommand wcm:action=\"add\">\n\
          <Order>$((N6 + 1))</Order>\n\
          <CommandLine>reg add \"HKCU\\\\SOFTWARE\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run\" /v WAAServer /t REG_SZ /d \"cmd /c \\\\\\\\host.lan\\\\Data\\\\start_waa_server.bat\" /f</CommandLine>\n\
          <Description>Add registry entry for WAA server auto-start (backup)</Description>\n\
        </SynchronousCommand>\n\
        <SynchronousCommand wcm:action=\"add\">\n\
          <Order>$((N6 + 2))</Order>\n\
          <CommandLine>\\\\\\\\host.lan\\\\Data\\\\start_waa_server.bat</CommandLine>\n\
          <Description>Start WAA server immediately</Description>\n\
        </SynchronousCommand>\n\
      </FirstLogonCommands>|" /run/assets/win11x64.xml; \
    fi

# -----------------------------------------------------------------------------
# Install Python and dependencies directly
# dockurr/windows base is Debian trixie which has Python 3.12
# -----------------------------------------------------------------------------

# Install Python 3 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install Python dependencies for WAA client
# Using --break-system-packages since we're in a container
# Full dependency list from: github.com/microsoft/WindowsAgentArena/blob/main/src/win-arena-container/client/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir --break-system-packages \
    gymnasium farama-notifications cloudpickle packaging typer rich tqdm colorama \
    openai anthropic google-generativeai groq tiktoken \
    pyyaml jsonschema tenacity httpx backoff toml func-timeout wrapt-timeout-decorator \
    psutil pyperclip screeninfo mss pyautogui fabric \
    easyocr pillow pytesseract opencv-python-headless scikit-image ImageHash \
    requests flask beautifulsoup4 lxml cssselect xmltodict playwright requests-toolbelt \
    pydrive openpyxl python-docx python-pptx odfpy pypdf PyPDF2 pdfplumber pymupdf borb \
    xlrd xlwt xlsxwriter mammoth pdf2image \
    google-api-python-client google-auth-httplib2 google-auth-oauthlib gdown \
    numpy pandas scipy formulas rapidfuzz anytree addict \
    transformers accelerate "timm>=0.9.0,<1.0.0" ultralytics supervision pycocotools einops \
    mutagen pyacoustid chardet librosa fastdtw \
    py7zr LnkParse3 \
    matplotlib wandb yapf

# Install Playwright browsers
RUN playwright install chromium

# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------

ENV YRES="900"
ENV XRES="1440"
ENV RAM_SIZE="8G"
ENV CPU_CORES="4"
ENV DISK_SIZE="30G"
ENV VERSION="11e"
ENV ARGUMENTS="-qmp tcp:0.0.0.0:7200,server,nowait"

# Expose ports
EXPOSE 8006 5000 7200 3389

# Default entrypoint - copy OEM files then run entry.sh
# Use: /entry.sh --start-client true --model gpt-4o
# Or:  /entry.sh --start-client false (just start Windows, no benchmark)
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/copy-oem.sh /entry.sh --start-client false"]
