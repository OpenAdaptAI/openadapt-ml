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

# Copy Windows setup scripts (install.bat, setup.ps1, etc.)
COPY --from=windowsarena/winarena:latest /oem /oem

# Copy unattend.xml for automated Windows installation
COPY --from=windowsarena/winarena:latest /run/assets/win11x64-enterprise-eval.xml /run/assets/win11x64.xml

# -----------------------------------------------------------------------------
# Create start_vm.sh that uses our dockurr/windows entrypoint
# -----------------------------------------------------------------------------

RUN printf '#!/bin/bash\n/usr/bin/tini -s /run/entry.sh\n' > /start_vm.sh && chmod +x /start_vm.sh

# -----------------------------------------------------------------------------
# Patch IP addresses: official uses 20.20.20.21, dockurr/windows uses 172.30.0.2
# -----------------------------------------------------------------------------

RUN sed -i 's|20.20.20.21|172.30.0.2|g' /entry_setup.sh /entry.sh /start_client.sh /client/*.py 2>/dev/null || true

# Also patch the client Python files
RUN find /client -name "*.py" -exec sed -i 's|20.20.20.21|172.30.0.2|g' {} \; 2>/dev/null || true

# -----------------------------------------------------------------------------
# Fix Windows setup for automation
# -----------------------------------------------------------------------------

# Set password for AutoLogon (Windows 11 requires password for login)
RUN sed -i 's|<Value></Value>|<Value>docker</Value>|g' /run/assets/win11x64.xml 2>/dev/null || true
RUN sed -i 's|<Value />|<Value>docker</Value>|g' /run/assets/win11x64.xml 2>/dev/null || true

# Add firewall disable and other automation commands to FirstLogonCommands
RUN if grep -q "</FirstLogonCommands>" /run/assets/win11x64.xml; then \
    LAST_ORDER=$(grep -oP "Order>\K[0-9]+" /run/assets/win11x64.xml | sort -n | tail -1) && \
    N1=$((LAST_ORDER + 1)) && \
    N2=$((LAST_ORDER + 2)) && \
    N3=$((LAST_ORDER + 3)) && \
    N4=$((LAST_ORDER + 4)) && \
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
      </FirstLogonCommands>|" /run/assets/win11x64.xml; \
    fi

# -----------------------------------------------------------------------------
# Install Python dependencies for benchmark client (runs on Linux host)
# -----------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for benchmark client
# Use --break-system-packages for Debian Trixie (Python 3.13 with PEP 668)
RUN pip3 install --no-cache-dir --break-system-packages \
    openai \
    requests \
    pillow \
    numpy \
    easyocr \
    pyautogui \
    tenacity

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

# Default entrypoint (can be overridden)
# Use: /entry.sh --start-client true --model gpt-4o
# Or:  /entry.sh --start-client false (just start Windows, no benchmark)
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/entry.sh --start-client false"]
