# OpenAdapt Desktop App Distribution Plan

## Executive Summary

This document outlines the strategy for distributing OpenAdapt as a downloadable desktop application. The goal is to provide a seamless installation experience where users can download, install, and start recording their first automation within minutes.

**Key insight**: The "end user app" doesn't need PyInstaller if uv can install Python, manage dependencies, and openadapt is published to PyPI. The question is whether we can make installing uv itself seamless enough for non-technical users.

## 1. Installation Tiers (Priority Order)

### Tier 1: Developer Install (uv-based) - PRIMARY

This is the recommended installation method for most users.

**macOS/Linux:**
```bash
# Install uv (one-liner)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install openadapt as CLI tool
uv tool install openadapt

# Run
openadapt record
```

**Windows:**
```powershell
# Install uv (one-liner)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install openadapt as CLI tool
uv tool install openadapt

# Run
openadapt record
```

**Why uv-first?**
1. **uv can install Python automatically**: `uv python install 3.12` - no manual Python installation needed
2. **uv can install tools globally**: `uv tool install openadapt` creates a managed virtual environment
3. **Single command install**: No need for complex installers or bundled runtimes
4. **Already using uv**: All new packages (openadapt-ml, evals, viewer, retrieval) use uv
5. **Standardization**: One tool for everything (development and end-user installation)
6. **Cross-platform**: Works on Windows, macOS, and Linux with identical commands
7. **Automatic updates**: `uv tool upgrade openadapt` handles updates cleanly

### Tier 2: End-User Install (uv bundled)

For users who want a "click to install" experience without running terminal commands.

**Option A: Installer Script**
Small installer that:
1. Downloads and installs uv
2. Runs `uv tool install openadapt`
3. Creates desktop shortcut

**macOS/Linux:**
```bash
#!/bin/bash
# install-openadapt.sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
uv tool install openadapt
echo "OpenAdapt installed! Run 'openadapt' to start."
```

**Windows:**
```powershell
# install-openadapt.ps1
irm https://astral.sh/uv/install.ps1 | iex
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","User")
uv tool install openadapt
Write-Host "OpenAdapt installed! Run 'openadapt' to start."
```

**Option B: Lightweight Native Installer**
- Small native installer (MSI/DMG) that:
  - Bundles uv binary (~15MB)
  - Runs post-install to `uv tool install openadapt`
  - Creates Start Menu / Applications entry
- Much smaller than PyInstaller bundle (~15MB vs ~200MB+)

### Tier 3: Full Bundle (PyInstaller) - DEFERRED

Only pursue if Tier 1/2 don't meet user needs:
- Enterprise deployments with strict requirements
- Air-gapped environments without internet access
- Environments where users cannot install any software

**Note**: This tier is deferred to Phase 2 and should only be implemented if there's demonstrated user demand.

## 2. Current State Analysis

### 2.1 openadapt-web Repository

**Repository**: https://github.com/openadaptai/openadapt-web

**Tech Stack**:
- Next.js v12 with Tailwind CSS
- Deployed on Netlify
- JavaScript (87.6%), CSS (7.4%), TypeScript (4.7%)

**Current Download Strategy**:
- **Download buttons exist** (dynamically generated via JavaScript showing download counts):
  - Platform-specific download buttons with counts from GitHub releases API
  - Version info: "Current Version: v0.46.0" with release date
- Additional elements:
  - "Learn How" button -> scrolls to #industries section
  - "Get Started" button -> scrolls to #start section
  - Links to GitHub, Discord, X (Twitter), LinkedIn
  - Email signup form (collecting leads - KEEP)
  - Industry use case sections (HR, Law, Insurance, etc.)
  - Bounties section (REMOVE - no longer active)
  - Gitbook documentation links (UPDATE or HIDE)

### 2.2 Current OpenAdapt Distribution

**Repository**: https://github.com/OpenAdaptAI/OpenAdapt

**Current Installation Methods** (to be replaced):
1. ~~Scripted Installation~~: PowerShell/Bash scripts requiring Git, Python, Poetry
2. ~~Manual Installation~~: Python 3.10, Git, Tesseract, nvm, Poetry

**Target Installation Method** (uv-based):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install uv
uv tool install openadapt                         # Install openadapt
openadapt record                                  # Run
```

### 2.3 Related Package Ecosystem

| Package | Purpose | Package Manager | Status |
|---------|---------|-----------------|--------|
| `openadapt` | Core recording/replay engine | uv (migration pending) | Production |
| `openadapt-ml` | ML training and inference | uv | Active development |
| `openadapt-capture` | Screen recording library | uv | Published on PyPI |
| `openadapt-retrieval` | Demo retrieval system | uv | Active development |
| `openadapt-evals` | Benchmark evaluation | uv | Active development |

## 3. Architecture for uv-based Distribution

### 3.1 Package Structure

```
openadapt (PyPI package)
├── openadapt/
│   ├── __init__.py
│   ├── __main__.py          # Entry point for `python -m openadapt`
│   ├── cli.py               # CLI commands (record, replay, etc.)
│   ├── core/                # Core functionality
│   ├── capture/             # Recording (or depends on openadapt-capture)
│   └── gui/                 # pywebview GUI
├── pyproject.toml           # uv-compatible with entry points
└── README.md
```

### 3.2 Entry Points (pyproject.toml)

```toml
[project.scripts]
openadapt = "openadapt.cli:main"

[project.gui-scripts]
openadapt-gui = "openadapt.gui:main"
```

### 3.3 Dependencies

**Core (always installed):**
- `openadapt-capture` - Screen recording
- `pywebview` - Native GUI
- `fastapi` + `uvicorn` - Local API server
- `sqlalchemy` - Database (SQLite bundled)

**Optional ML (install on demand):**
```bash
uv tool install openadapt[ml]  # Includes torch, transformers, etc.
```

## 4. Website Changes Required

### 4.1 New Download Section

```javascript
// components/DownloadSection.js
const DownloadSection = () => {
  const [selectedOS, setSelectedOS] = useState('auto');

  const installCommands = {
    macos: `curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install openadapt
openadapt`,
    linux: `curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install openadapt
openadapt`,
    windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv tool install openadapt
openadapt`,
  };

  return (
    <section id="download">
      <h2>Install OpenAdapt</h2>
      <p>Install in seconds with a single command.</p>

      <PlatformTabs
        selected={selectedOS}
        onSelect={setSelectedOS}
      />

      <CodeBlock
        code={installCommands[selectedOS]}
        copyable
      />

      <p className="text-muted">
        <a href="/install">Need help?</a> |
        <a href="https://docs.astral.sh/uv/">What is uv?</a>
      </p>
    </section>
  );
};
```

### 4.2 Pages to Add

1. `/install` - Step-by-step installation guide with troubleshooting
2. `/getting-started` - First-run tutorial after installation
3. `/update` - How to update (`uv tool upgrade openadapt`)

## 5. Implementation Phases

### Phase 1: uv Migration (2-3 weeks)
- [ ] Migrate openadapt core from Poetry to uv
- [ ] Add proper entry points in pyproject.toml
- [ ] Test `uv tool install openadapt` from PyPI
- [ ] Update website with installation commands
- [ ] Create install troubleshooting guide

### Phase 2: User Experience Polish (2-3 weeks)
- [ ] Create installer scripts for one-click setup
- [ ] First-run onboarding wizard in app
- [ ] System tray integration
- [ ] Auto-update check (prompt user to run `uv tool upgrade`)

### Phase 3: PyInstaller Fallback (DEFERRED - 4-6 weeks if needed)
- [ ] Only if Tier 1/2 don't meet user needs
- [ ] Create PyInstaller spec file
- [ ] Build platform-specific installers
- [ ] Set up GitHub Actions workflow
- [ ] Code signing (Windows EV, macOS notarization)

## 6. Platform Support

### 6.1 All Platforms (via uv)

uv supports:
- **Windows**: 10/11 (64-bit)
- **macOS**: 10.12+ (Intel and Apple Silicon)
- **Linux**: glibc 2.17+ (most modern distros)

uv automatically:
- Downloads and manages Python versions
- Creates isolated tool environments
- Handles PATH configuration

### 6.2 Platform-Specific Considerations

**macOS:**
- Accessibility permissions required for screen recording
- First launch may require right-click -> Open (Gatekeeper)
- Guide users through System Preferences permissions

**Windows:**
- May need to allow PowerShell script execution
- Windows Defender SmartScreen warning (can be bypassed)
- UAC prompt if permissions needed

**Linux:**
- May need to install system packages for GUI (GTK/QT)
- Some distros may need Tesseract installed separately

## 7. User Journey

### 7.1 Discovery and Install

```
1. User visits openadapt.ai
2. Sees "Install in 30 seconds" section
3. Copies installation command for their OS
4. Pastes in terminal and runs
5. OpenAdapt is ready to use
```

### 7.2 First Run Experience

```
1. Run `openadapt` command
2. GUI launches with welcome screen
3. Permissions check (macOS: accessibility, screen recording)
4. Quick tutorial: Record your first automation
5. User is productive in < 5 minutes
```

### 7.3 Updates

```bash
# User runs:
uv tool upgrade openadapt

# App can also prompt and run this for user
```

## 8. Comparison: uv vs PyInstaller

| Aspect | uv-based | PyInstaller |
|--------|----------|-------------|
| Download size | ~15MB (uv) + packages on demand | ~200MB-2GB bundle |
| Installation | Single command | Download installer, run wizard |
| Updates | `uv tool upgrade` | Download new installer |
| Python version | Managed by uv | Bundled, static |
| Dependencies | Resolved at install | Frozen at build time |
| Build complexity | None (uses PyPI) | CI/CD pipeline, signing |
| Cross-platform | Same commands | Separate builds per OS |
| Maintenance | Low (PyPI handles distribution) | High (build infra, signing certs) |

**Conclusion**: uv-based installation is simpler, lighter, and easier to maintain.

## 9. Cost Estimates

### uv-based Approach (Tier 1/2)

| Item | Cost |
|------|------|
| PyPI hosting | Free |
| GitHub Actions (testing) | Free tier sufficient |
| Website updates | Internal time |
| **Total** | **$0/year** |

### PyInstaller Approach (Tier 3, if needed)

| Item | Cost/Year |
|------|-----------|
| Apple Developer Program | $99 |
| Windows Code Signing (EV) | $300-500 |
| GitHub Actions (builds) | Free tier likely sufficient |
| S3/CloudFront (updates) | ~$50-100 |
| **Total** | **~$500-700/year** |

## 10. Security Considerations

1. **Package Signing**: PyPI packages are signed via Trusted Publishing
2. **Script Verification**: Installation scripts from astral.sh are signed
3. **Dependency Pinning**: uv.lock ensures reproducible installs
4. **Permissions**: Request minimum necessary OS permissions
5. **API Keys**: Store in OS keychain (via keyring library)

## 11. Success Metrics

| Metric | Target |
|--------|--------|
| Install-to-first-recording time | < 5 minutes |
| Installation success rate | > 95% |
| Command simplicity | 3 lines or less |
| Support tickets per 100 installs | < 5 |

## 12. References

### uv Documentation
- [uv Installation](https://docs.astral.sh/uv/getting-started/installation/)
- [uv Tool Management](https://docs.astral.sh/uv/guides/tools/)
- [uv Python Management](https://docs.astral.sh/uv/guides/install-python/)

### Python Packaging
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [pywebview Documentation](https://pywebview.flowrl.com/)

### PyInstaller (for Tier 3 if needed)
- [PyInstaller Documentation](https://pyinstaller.org/en/stable/)
- [Apple Developer ID](https://developer.apple.com/developer-id/)
- [Windows Code Signing](https://learn.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools)

---

*Document Version: 2.0*
*Last Updated: January 2026*
*Major revision: Shifted from PyInstaller-first to uv-first approach*
