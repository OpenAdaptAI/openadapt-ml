# OpenAdapt Desktop App Distribution Plan

## Executive Summary

This document outlines the strategy for distributing OpenAdapt as a downloadable desktop application. The goal is to provide a seamless installation experience where users can download, install, and start recording their first automation within minutes.

## 1. Current State Analysis

### 1.1 OpenAdapt.web Repository

**Repository**: https://github.com/openadaptai/openadapt.web

**Tech Stack**:
- Next.js v12 with Tailwind CSS
- Deployed on Netlify
- JavaScript (87.6%), CSS (7.4%), TypeScript (4.7%)

**Current Download Strategy**:
- **No download button exists**. The website currently has:
  - "Learn How" button -> scrolls to #industries section
  - "Get Started" button -> scrolls to #start section
  - Links to GitHub, Discord, X (Twitter), LinkedIn
  - Email signup form
  - Download statistics graph (fetches from GitHub releases API)

**Key Components Analyzed**:
- `MastHead.js`: Hero section with CTAs, no download functionality
- `Footer.js`: Social links, legal pages, no app distribution
- `DownloadGraph.js`: Visualizes GitHub release download counts
- `pages/index.js`: Homepage layout with sections but no download page

### 1.2 Current OpenAdapt Distribution

**Repository**: https://github.com/OpenAdaptAI/OpenAdapt

**Current Installation Methods**:
1. **Scripted Installation** (recommended):
   - Windows: PowerShell script with admin elevation
   - macOS: Bash script via curl requiring Git and Python 3.10
2. **Manual Installation**: Python 3.10, Git, Tesseract, nvm, Poetry

**GitHub Releases**:
- Semantic versioning (v0.46.0, v0.45.0, etc.)
- 6 assets per release (source archives + zip files)
- No pre-built binaries or platform-specific installers

**Dependencies** (from pyproject.toml):
- Python >=3.10,<3.12
- Heavy ML dependencies: torch, transformers, openai, anthropic
- GUI: pywebview, fastapi, uvicorn
- Vision: pytesseract, easyocr, segment-anything, ultralytics

### 1.3 Related Package Ecosystem

| Package | Purpose | Status |
|---------|---------|--------|
| `openadapt` | Core recording/replay engine | Production |
| `openadapt-ml` | ML training and inference | Active development |
| `openadapt-capture` | Screen recording library | Published on PyPI |
| `openadapt-retrieval` | Demo retrieval system | Active development |
| `openadapt-evals` | Benchmark evaluation | Active development |

## 2. Proposed App Architecture

### 2.1 Recommended Approach: pywebview + PyInstaller

**Rationale**:
- OpenAdapt already uses `pywebview` for its GUI
- Python-native approach aligns with existing codebase
- Lighter weight than Electron (no bundled Chromium)
- Uses native WebView (WinForms on Windows, Cocoa on macOS, GTK/QT on Linux)

**Architecture**:
```
OpenAdapt Desktop App
├── Python Runtime (bundled via PyInstaller)
├── openadapt (core)
├── openadapt-ml (ML engine)
├── openadapt-capture (recording)
├── FastAPI backend (local server)
└── pywebview frontend (native WebView)
```

### 2.2 Alternative Approaches Considered

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Electron** | Cross-platform, rich ecosystem, auto-updates | Heavy (~200MB), requires rewrite, duplicate bundling | Not recommended |
| **Nuitka** | Better code protection, slight performance gains | Slower builds, less mature ecosystem | Consider for production |
| **BeeWare/Briefcase** | Native look, Python-native | Less mature, requires specific project structure | Future consideration |
| **Flet** | Modern Flutter UI, fast packaging | Different UI paradigm, learning curve | Not recommended |

### 2.3 What to Include in the Download

**Core Bundle (Required)**:
- Python 3.10+ runtime (embedded)
- `openadapt` core package
- `openadapt-capture` for screen recording
- `pywebview` for GUI
- `fastapi` + `uvicorn` for local API server
- Tesseract OCR (bundled)
- Database (SQLite, bundled)

**Optional ML Components** (separate download or on-demand):
- `openadapt-ml` (training, fine-tuning)
- PyTorch (large, ~2GB)
- Transformers
- Cloud inference option (API-based, no local ML)

**Recommended Bundle Strategy**:
1. **Lite Bundle** (~200-300MB): Core recording/replay, cloud inference
2. **Full Bundle** (~2-3GB): Includes local ML capabilities

## 3. Platform Support

### 3.1 Windows

**Installer Format**: MSI (recommended) or NSIS
- MSI: Better for enterprise deployment, silent install, Group Policy
- NSIS: More customizable, smaller installer

**Requirements**:
- Windows 10/11 (64-bit)
- Code signing certificate (EV recommended for SmartScreen reputation)
- Elevated permissions for initial install

**Build Tool**: PyInstaller with `--onedir` mode

### 3.2 macOS

**Installer Format**: DMG with signed .app bundle

**Requirements**:
- macOS 10.15 (Catalina) or later
- Apple Developer ID certificate
- Notarization with Apple

**Special Considerations**:
- Accessibility permissions required for screen recording
- Need to guide users through System Preferences permissions
- Universal binary (arm64 + x86_64) for M1/M2/Intel support

**Build Tool**: PyInstaller + `create-dmg`

### 3.3 Linux

**Installer Formats**:
- AppImage (recommended for broad compatibility)
- .deb for Debian/Ubuntu
- .rpm for Fedora/RHEL (optional)

**Requirements**:
- GTK3 or QT for pywebview
- Tesseract system package

**Build Tool**: PyInstaller + `appimagetool`

## 4. Build and Release Pipeline

### 4.1 CI/CD Workflow (GitHub Actions)

```yaml
# .github/workflows/build-desktop.yml
name: Build Desktop App

on:
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install pyinstaller
          pip install -e .[desktop]
      - name: Build with PyInstaller
        run: pyinstaller openadapt.spec --clean
      - name: Sign executable
        run: # Code signing with EV certificate
      - name: Create MSI installer
        run: # WiX toolset or Inno Setup
      - uses: actions/upload-artifact@v4
        with:
          name: windows-installer
          path: dist/*.msi

  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Build with PyInstaller
        run: pyinstaller openadapt.spec --clean
      - name: Sign and notarize
        run: |
          codesign --deep --force --sign "$DEVELOPER_ID" dist/OpenAdapt.app
          xcrun notarytool submit dist/OpenAdapt.app --wait
          xcrun stapler staple dist/OpenAdapt.app
      - name: Create DMG
        run: create-dmg dist/OpenAdapt.app --out dist/
      - uses: actions/upload-artifact@v4
        with:
          name: macos-installer
          path: dist/*.dmg

  build-linux:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Build with PyInstaller
        run: pyinstaller openadapt.spec --clean
      - name: Create AppImage
        run: |
          # Create AppDir structure
          # Package with appimagetool
      - uses: actions/upload-artifact@v4
        with:
          name: linux-installer
          path: dist/*.AppImage

  release:
    needs: [build-windows, build-macos, build-linux]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            windows-installer/*.msi
            macos-installer/*.dmg
            linux-installer/*.AppImage
```

### 4.2 Version Management

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Tag releases trigger automated builds
- Pre-release builds for beta testing
- Use `python-semantic-release` for automated changelog

## 5. Auto-Update Mechanism

### 5.1 Recommended: PyUpdater

[PyUpdater](https://github.com/Digital-Sapphire/PyUpdater) is designed specifically for PyInstaller applications.

**Features**:
- Delta updates (only download changed files)
- Code signing verification
- Multiple release channels (stable, beta)
- Self-contained update client

**Integration**:
```python
from pyupdater.client import Client
from openadapt.config import APP_NAME, APP_VERSION

def check_for_updates():
    client = Client(
        ClientConfig(),
        refresh=True,
    )
    client.add_progress_hook(print_status)

    app_update = client.update_check(APP_NAME, APP_VERSION)
    if app_update:
        app_update.download()
        if app_update.is_downloaded():
            app_update.extract_restart()
```

### 5.2 Update Server

**Options**:
1. **GitHub Releases** (simplest): Host releases on GitHub
2. **S3/CloudFront**: For large files and CDN distribution
3. **Self-hosted**: Control over update server

**Recommended**: Start with GitHub Releases, migrate to S3 if needed.

### 5.3 Update Channels

| Channel | Purpose | Auto-update |
|---------|---------|-------------|
| Stable | Production releases | Yes (with confirmation) |
| Beta | Pre-release testing | Yes (opt-in) |
| Nightly | Development builds | No (manual only) |

## 6. Code Signing Requirements

### 6.1 Windows Code Signing

**Certificate Types**:
1. **Standard Code Signing** (~$200-500/year): Basic signing, may show SmartScreen warning initially
2. **EV Code Signing** (~$300-600/year): Instant SmartScreen reputation, hardware token required

**Recommended**: Start with Standard, upgrade to EV when download volume increases.

**Providers**: DigiCert, Sectigo, SSL.com

### 6.2 macOS Code Signing & Notarization

**Requirements**:
1. Apple Developer Program membership ($99/year)
2. Developer ID Application certificate
3. Notarization through Apple

**Process**:
1. Sign app bundle with Developer ID
2. Submit to Apple for notarization (automated malware scan)
3. Staple notarization ticket to app
4. Create DMG with signed app

**Tools**: `codesign`, `xcrun notarytool`, `xcrun stapler`

## 7. User Journey

### 7.1 Discovery and Download

```
1. User visits openadapt.ai
2. Clicks prominent "Download" button
3. Auto-detects OS, offers correct installer
4. Shows manual platform selection if needed
5. Download starts (~200MB lite, ~2GB full)
```

**Website Changes Required**:
- Add Download page/section to openadapt.web
- Platform detection JavaScript
- Download statistics tracking
- Clear system requirements

### 7.2 Installation

**Windows**:
```
1. Run .msi installer
2. UAC prompt for admin permissions
3. Choose install location (default: Program Files)
4. Optional: Create desktop shortcut
5. Installation completes (~30 seconds)
```

**macOS**:
```
1. Open .dmg file
2. Drag OpenAdapt.app to Applications
3. First launch: Right-click -> Open (Gatekeeper)
4. Grant accessibility permissions (guided)
5. Grant screen recording permissions (guided)
```

**Linux**:
```
1. Download .AppImage
2. chmod +x OpenAdapt.AppImage
3. Run ./OpenAdapt.AppImage
4. Optional: Integrate with desktop
```

### 7.3 First Run Experience

```
1. Launch OpenAdapt
2. Welcome screen with quick tour
3. Permissions check (macOS: accessibility, screen recording)
4. Optional: Sign in / create account (for cloud features)
5. Tutorial: Record your first automation
   a. Click "Record" button
   b. Perform actions to automate
   c. Click "Stop" button
   d. Review recording in viewer
6. Tutorial: Replay automation
   a. Select recording
   b. Click "Replay" button
   c. Watch automation execute
7. Prompt: Train AI model (optional, requires ML bundle)
```

### 7.4 Ongoing Use

```
- System tray icon for quick access
- Hotkey to start/stop recording
- Automatic updates (with user confirmation)
- Cloud sync for recordings (optional)
- Model training progress dashboard
```

## 8. OpenAdapt.web Changes Required

### 8.1 New Components

```javascript
// components/DownloadSection.js
const DownloadSection = () => {
  const [detectedOS, setDetectedOS] = useState(null);

  useEffect(() => {
    // Detect user's OS
    const platform = navigator.platform;
    if (platform.includes('Win')) setDetectedOS('windows');
    else if (platform.includes('Mac')) setDetectedOS('macos');
    else if (platform.includes('Linux')) setDetectedOS('linux');
  }, []);

  return (
    <section id="download">
      <h2>Download OpenAdapt</h2>
      <DownloadButton os={detectedOS} />
      <PlatformSelector />
      <SystemRequirements />
    </section>
  );
};
```

### 8.2 Download Button Component

```javascript
// components/DownloadButton.js
const DOWNLOAD_URLS = {
  windows: 'https://github.com/OpenAdaptAI/OpenAdapt/releases/latest/download/OpenAdapt-Setup.msi',
  macos: 'https://github.com/OpenAdaptAI/OpenAdapt/releases/latest/download/OpenAdapt.dmg',
  linux: 'https://github.com/OpenAdaptAI/OpenAdapt/releases/latest/download/OpenAdapt.AppImage',
};

const DownloadButton = ({ os }) => (
  <a
    href={DOWNLOAD_URLS[os]}
    className="btn btn-primary btn-large"
    download
  >
    Download for {getPlatformName(os)}
  </a>
);
```

### 8.3 Pages to Add

1. `/download` - Dedicated download page with all platforms
2. `/install` - Installation guide with troubleshooting
3. `/getting-started` - First-run tutorial

## 9. Implementation Phases

### Phase 1: MVP (4-6 weeks)
- [ ] Create PyInstaller spec file
- [ ] Build Windows MSI installer
- [ ] Build macOS DMG (signed and notarized)
- [ ] Build Linux AppImage
- [ ] Set up GitHub Actions workflow
- [ ] Add download section to website
- [ ] Basic auto-update check (no delta updates)

### Phase 2: Polish (2-4 weeks)
- [ ] Implement PyUpdater for delta updates
- [ ] Add update channels (stable/beta)
- [ ] First-run onboarding wizard
- [ ] System tray integration
- [ ] Keyboard shortcuts

### Phase 3: Advanced (4-6 weeks)
- [ ] Lite vs Full bundle options
- [ ] On-demand ML model download
- [ ] Cloud account integration
- [ ] Usage analytics (opt-in)
- [ ] Crash reporting

### Phase 4: App Store Distribution (Optional)
- [ ] Microsoft Store submission
- [ ] Mac App Store submission (requires sandbox)
- [ ] Snap/Flatpak for Linux

## 10. Cost Estimates

### One-Time Costs
| Item | Cost |
|------|------|
| Windows Code Signing (EV, 3-year) | $1,000-1,500 |
| Apple Developer Program (1 year) | $99 |
| Build infrastructure setup | Internal time |

### Ongoing Costs
| Item | Cost/Year |
|------|-----------|
| Apple Developer Program | $99 |
| Windows Code Signing renewal | $300-500 |
| GitHub Actions (free tier sufficient) | $0 |
| S3/CloudFront (if used for updates) | ~$50-100 |

## 11. Security Considerations

1. **Code Signing**: All executables must be signed
2. **Update Verification**: Cryptographic verification of updates
3. **Sandbox Mode**: Consider for Mac App Store
4. **Privacy**: No telemetry without explicit consent
5. **API Keys**: Securely store user API keys (OS keychain)
6. **Permissions**: Request minimum necessary permissions

## 12. Success Metrics

| Metric | Target |
|--------|--------|
| Download-to-first-recording time | < 5 minutes |
| Installation success rate | > 95% |
| Auto-update adoption rate | > 80% |
| Crash-free sessions | > 99% |
| User retention (7-day) | > 40% |

## 13. References

### Python Desktop Packaging
- [PyInstaller Documentation](https://pyinstaller.org/en/stable/)
- [Nuitka User Manual](https://nuitka.net/doc/user-manual.html)
- [pywebview Documentation](https://pywebview.flowrl.com/)
- [BeeWare Briefcase](https://briefcase.beeware.org/)

### Auto-Update Solutions
- [PyUpdater](https://www.pyupdater.org/)
- [Updater4pyi](https://updater4pyi.readthedocs.io/)

### Code Signing
- [Apple Developer ID](https://developer.apple.com/developer-id/)
- [Windows Code Signing](https://learn.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools)
- [DigiCert Code Signing](https://comparecheapssl.com/digicert-code-signing-for-mac-developers-a-complete-guide/)

### Build Automation
- [GitHub Actions](https://docs.github.com/en/actions)
- [electron-builder (for reference)](https://www.electron.build/)

---

*Document Version: 1.0*
*Last Updated: January 2026*
