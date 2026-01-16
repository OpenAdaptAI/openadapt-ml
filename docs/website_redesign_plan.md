# OpenAdapt.ai Website Redesign Plan

**Date**: January 2026
**Status**: Planning
**Context**: Transition from monolithic openadapt repo to modular ecosystem

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Page Structure](#2-page-structure)
3. [Hero Section](#3-hero-section)
4. [Installation Section](#4-installation-section)
5. [Industry Sections](#5-industry-sections)
6. [Documentation Strategy](#6-documentation-strategy)
7. [Getting Started](#7-getting-started)
8. [Footer and Links](#8-footer-and-links)
9. [Technical Considerations](#9-technical-considerations)
10. [Content to Remove](#10-content-to-remove)
11. [Migration Checklist](#11-migration-checklist)

---

## 1. Executive Summary

### Goals

1. **Update for new architecture**: Reflect the modular ecosystem (openadapt-ml, openadapt-capture, etc.)
2. **Simplify installation**: uv-first approach replaces complex scripted installation
3. **Preserve what works**: Keep email signup, download stats, industry sections, professional branding
4. **Remove outdated content**: Bounties, legacy installation scripts, outdated Gitbook links

### Key Changes

| Element | Current | New |
|---------|---------|-----|
| Installation | PowerShell/bash scripts calling old repo | `uv tool install openadapt` |
| Package | Single monolithic `openadapt` | Modular: `openadapt-capture`, `openadapt-ml`, etc. |
| Documentation | Gitbook (outdated) | GitHub READMEs + potential new docs site |
| Download stats | Single repo | Aggregated from multiple packages |

---

## 2. Page Structure

### Recommended Section Order

```
1. Navigation Bar
   - Logo
   - Nav links: Features | Docs | GitHub | Get Started
   - Email signup CTA (compact)

2. Hero Section
   - Tagline emphasizing AI GUI automation
   - Primary CTA: Get Started
   - Secondary CTA: View on GitHub
   - Download stats badge (PyPI total)

3. Installation Section (NEW - prominent)
   - uv-first installation with platform tabs
   - Quick start code block
   - "Works on macOS, Linux, Windows"

4. Value Proposition / How It Works
   - 3-column feature highlights
   - Visual diagram of capture -> train -> deploy flow

5. Industry Use Cases (KEEP)
   - HR, Legal, Insurance, Finance, etc.
   - Expandable cards with examples

6. Ecosystem Overview (NEW)
   - Visual showing modular packages
   - Brief description of each component

7. Getting Started for Developers
   - Simplified flow
   - Links to package READMEs

8. Social Proof / Stats
   - GitHub stars (aggregated)
   - PyPI downloads
   - Community size

9. Footer
   - Quick links
   - GitHub org link
   - Contact/email signup
   - MIT License badge
```

---

## 3. Hero Section

### Current Issues

- References old installation method
- May mention features no longer in scope
- Download stats point to single repo

### Recommended Updates

#### Tagline Options

```
Primary: "AI-Powered GUI Automation"
Subtitle: "Record. Learn. Automate."

Alternative: "Teach AI to use your software"
Subtitle: "Capture human workflows, train domain-specific agents"
```

#### Hero CTAs

```html
<div class="hero-ctas">
  <a href="#install" class="btn-primary">Get Started</a>
  <a href="https://github.com/OpenAdaptAI" class="btn-secondary">View on GitHub</a>
</div>
```

#### Stats Badge

Display aggregated stats:
- Combined PyPI downloads across all packages
- GitHub stars from openadapt-ml (primary repo)
- "v0.2.0" (latest openadapt-ml version)

---

## 4. Installation Section

### Design Principles

1. **uv-first**: Lead with uv (modern, fast, reliable)
2. **Platform tabs**: macOS/Linux and Windows tabs
3. **Copy buttons**: Easy clipboard copy for commands
4. **Progressive disclosure**: Basic install first, advanced options below

### Recommended Layout

```html
<section id="install" class="installation-section">
  <h2>Install in 30 Seconds</h2>

  <div class="platform-tabs">
    <button class="tab active" data-platform="unix">macOS / Linux</button>
    <button class="tab" data-platform="windows">Windows</button>
  </div>

  <div class="tab-content unix active">
    <pre><code># Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install OpenAdapt
uv tool install openadapt</code></pre>
  </div>

  <div class="tab-content windows">
    <pre><code># Install uv (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install OpenAdapt
uv tool install openadapt</code></pre>
  </div>

  <div class="requirements">
    <p>Requirements: Python 3.12+</p>
  </div>
</section>
```

### Package Options

Show optional packages for power users:

```html
<details class="advanced-install">
  <summary>Advanced Installation Options</summary>

  <h4>Individual Packages (for development)</h4>
  <pre><code># Clone and sync any package
git clone https://github.com/OpenAdaptAI/openadapt-ml.git
cd openadapt-ml
uv sync
uv run python -m openadapt_ml.scripts.train --help

# Or install as tools
uv tool install openadapt-capture
uv tool install openadapt-evals</code></pre>
</details>
```

---

## 5. Industry Sections

### Assessment: KEEP

The industry sections (HR, Legal, Insurance, Finance, etc.) are valuable because:
- They help visitors understand use cases
- They demonstrate domain versatility
- They support lead generation (email signup)
- They differentiate from general-purpose automation tools

### Recommended Modifications

1. **Update any screenshots** that show old UI
2. **Refresh example workflows** to reflect new capture/train flow
3. **Keep the expandable card pattern** - it works well
4. **Consider adding new industries**: Healthcare, Manufacturing, Retail

### Example Industry Card Structure

```html
<div class="industry-card">
  <div class="industry-header">
    <span class="industry-icon"><!-- icon --></span>
    <h3>Legal</h3>
  </div>
  <p class="industry-summary">Automate document processing and case management</p>
  <div class="industry-examples">
    <h4>Example Workflows:</h4>
    <ul>
      <li>Contract review and extraction</li>
      <li>Court filing automation</li>
      <li>Client intake processing</li>
    </ul>
  </div>
</div>
```

---

## 6. Documentation Strategy

### Current State

- **Gitbook**: Outdated, points to old repo
- **GitHub READMEs**: Up-to-date, comprehensive
- **Decision needed**: Hide Gitbook or update?

### Recommended Approach

#### Phase 1: Immediate (Website Update)

1. **Remove Gitbook links** from main navigation
2. **Point "Docs" link** to GitHub org README or openadapt-ml README
3. **Keep Gitbook alive but unlisted** (for old link preservation)

#### Phase 2: Future (If Resources Allow)

Options for new documentation:
1. **GitHub Wiki** on openadapt-ml repo
2. **MkDocs or Docusaurus** deployed to docs.openadapt.ai
3. **Update Gitbook** with new content

### Documentation Link Targets

| Link | Target (Immediate) | Target (Future) |
|------|-------------------|-----------------|
| "Docs" nav | github.com/OpenAdaptAI/openadapt-ml#readme | docs.openadapt.ai |
| "API Reference" | github.com/OpenAdaptAI/openadapt-ml/tree/main/docs | docs.openadapt.ai/api |
| "Examples" | github.com/OpenAdaptAI/openadapt-ml/tree/main/examples | docs.openadapt.ai/examples |

---

## 7. Getting Started

### Current Issues

- Complex multi-step process
- References Poetry and old repo structure
- Confusing for new users

### Simplified Developer Flow

```
1. Install (30 seconds)
   └── uv tool install openadapt

2. Capture (2 minutes)
   └── openadapt-capture record --output my-workflow
   └── Perform task, press Enter to stop

3. Train (5 minutes)
   └── openadapt train --capture my-workflow --open
   └── Dashboard shows training progress

4. Evaluate (optional)
   └── openadapt compare --capture my-workflow --checkpoint model
```

### Getting Started Section HTML

```html
<section class="getting-started">
  <h2>Getting Started for Developers</h2>

  <div class="steps">
    <div class="step">
      <span class="step-number">1</span>
      <h3>Install</h3>
      <pre><code>uv tool install openadapt</code></pre>
    </div>

    <div class="step">
      <span class="step-number">2</span>
      <h3>Capture a Workflow</h3>
      <pre><code>openadapt capture --output my-task
# Perform the task, then press Enter to stop</code></pre>
    </div>

    <div class="step">
      <span class="step-number">3</span>
      <h3>Train</h3>
      <pre><code>openadapt train --capture my-task --open</code></pre>
    </div>
  </div>

  <a href="https://github.com/OpenAdaptAI/openadapt-ml" class="btn-secondary">
    Full Documentation
  </a>
</section>
```

---

## 8. Footer and Links

### Footer Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  OPENADAPT                                                       │
│                                                                  │
│  Packages           Resources          Connect                   │
│  ───────────        ─────────          ───────                   │
│  openadapt-ml       Documentation      GitHub                    │
│  openadapt-capture  Examples           Discord (if exists)       │
│  openadapt-evals    Blog (if exists)   Twitter/X                 │
│  openadapt-grounding                   Email signup              │
│  openadapt-retrieval                                             │
│  openadapt-viewer                                                │
│                                                                  │
│  ─────────────────────────────────────────────────────────────── │
│  MIT License  |  Made by MLDSAI Inc.  |  v0.2.0                  │
└─────────────────────────────────────────────────────────────────┘
```

### External Links

| Link | URL | Notes |
|------|-----|-------|
| GitHub Org | https://github.com/OpenAdaptAI | Primary |
| Main Repo | https://github.com/OpenAdaptAI/openadapt-ml | Star count source |
| PyPI | https://pypi.org/project/openadapt-ml/ | Download stats |
| Discord | (if exists) | Community |

---

## 9. Technical Considerations

### 9.1 Download Statistics

#### Current Implementation

The site currently fetches download stats from GitHub releases API.

#### New Implementation

Aggregate stats from multiple sources:

```javascript
// Fetch PyPI download stats for all packages
async function fetchDownloadStats() {
  const packages = [
    'openadapt-ml',
    'openadapt-capture',
    'openadapt-evals',
    'openadapt-grounding',
    'openadapt-retrieval',
    'openadapt-viewer',
  ];

  let totalDownloads = 0;

  for (const pkg of packages) {
    // Use pypistats.org API or similar
    const response = await fetch(`https://pypistats.org/api/packages/${pkg}/recent`);
    const data = await response.json();
    totalDownloads += data.data.last_month;
  }

  return totalDownloads;
}
```

#### Alternative: Single Primary Package

If aggregation is complex, focus on `openadapt-ml` as the primary package:
- Most downloads will be `openadapt-ml` anyway
- Simpler to implement and maintain
- Can show "X downloads of openadapt-ml"

### 9.2 Email Signup Integration

**Keep existing implementation** - this is working and valuable for lead generation.

Ensure:
- Form remains prominent (hero section or sticky footer)
- Integration with existing email service continues
- Confirmation flow works correctly

### 9.3 Version Display

Show latest version dynamically:

```javascript
async function fetchLatestVersion() {
  const response = await fetch('https://pypi.org/pypi/openadapt-ml/json');
  const data = await response.json();
  return data.info.version;
}
```

### 9.4 GitHub Stars

Fetch from OpenAdaptAI org or specific repos:

```javascript
async function fetchGitHubStars() {
  const response = await fetch('https://api.github.com/repos/OpenAdaptAI/openadapt-ml');
  const data = await response.json();
  return data.stargazers_count;
}
```

---

## 10. Content to Remove

### 10.1 Bounties Section

**Action**: Remove completely

**Reason**: No longer active, creates confusion

### 10.2 Legacy Installation Scripts

**Action**: Remove references to:
- `install.sh` from old openadapt repo
- PowerShell installation script for old repo
- Poetry-based installation instructions
- `pip install openadapt` for the old package

**Replace with**: uv-first installation (see Section 4)

### 10.3 Gitbook Links

**Action**: Remove from main navigation

**Options**:
1. **Hide completely**: Remove all Gitbook links
2. **Redirect**: Set up redirects from old Gitbook URLs to GitHub READMEs
3. **Update Gitbook** (if time permits): Refresh content to match new architecture

**Recommended**: Option 1 (Hide completely) for immediate update, with potential for Option 3 later.

### 10.4 Old Repository References

**Action**: Update all links pointing to `OpenAdaptAI/openadapt` to point to `OpenAdaptAI/openadapt-ml` or the GitHub org page.

---

## 11. Migration Checklist

### Phase 1: Content Updates (1-2 days)

- [ ] Update hero section tagline and CTAs
- [ ] Replace installation instructions with uv-first approach
- [ ] Update "Getting Started" to simplified flow
- [ ] Remove bounties section
- [ ] Remove/hide Gitbook links from navigation
- [ ] Update footer links to new package structure
- [ ] Update any screenshots showing old UI

### Phase 2: Technical Updates (1-2 days)

- [ ] Update download stats fetching (PyPI instead of GitHub releases)
- [ ] Update version display to fetch from PyPI
- [ ] Update GitHub stars to point to new repo
- [ ] Test email signup still works
- [ ] Add platform tabs for installation (macOS/Linux/Windows)
- [ ] Add copy buttons to code blocks

### Phase 3: New Content (2-3 days)

- [ ] Add ecosystem overview section
- [ ] Create visual diagram of modular architecture
- [ ] Refresh industry section examples
- [ ] Add "How It Works" section with capture->train->deploy flow
- [ ] Consider adding a "Why OpenAdapt?" section

### Phase 4: Testing & Launch

- [ ] Test on all major browsers (Chrome, Firefox, Safari, Edge)
- [ ] Test on mobile devices
- [ ] Verify all links work
- [ ] Check SEO metadata is correct
- [ ] Deploy to Netlify staging
- [ ] Get stakeholder review
- [ ] Deploy to production

### Post-Launch

- [ ] Monitor analytics for any issues
- [ ] Set up redirects from old URLs if needed
- [ ] Update any external documentation pointing to old URLs
- [ ] Archive old openadapt repo with note pointing to new ecosystem

---

## Appendix A: Ecosystem Package Summary

| Package | Purpose | PyPI | GitHub |
|---------|---------|------|--------|
| **openadapt-ml** | Core ML engine for training and policy runtime | [Link](https://pypi.org/project/openadapt-ml/) | [Link](https://github.com/OpenAdaptAI/openadapt-ml) |
| **openadapt-capture** | Platform-agnostic GUI recording | [Link](https://pypi.org/project/openadapt-capture/) | [Link](https://github.com/OpenAdaptAI/openadapt-capture) |
| **openadapt-evals** | Benchmark evaluation infrastructure | [Link](https://pypi.org/project/openadapt-evals/) | [Link](https://github.com/OpenAdaptAI/openadapt-evals) |
| **openadapt-grounding** | UI element localization | [Link](https://pypi.org/project/openadapt-grounding/) | [Link](https://github.com/OpenAdaptAI/openadapt-grounding) |
| **openadapt-retrieval** | Multimodal demo retrieval | [Link](https://pypi.org/project/openadapt-retrieval/) | [Link](https://github.com/OpenAdaptAI/openadapt-retrieval) |
| **openadapt-viewer** | Visualization components | [Link](https://pypi.org/project/openadapt-viewer/) | [Link](https://github.com/OpenAdaptAI/openadapt-viewer) |

---

## Appendix B: Architecture Diagram (for website)

```
                    OpenAdapt Ecosystem
                    ===================

     User Workflow                      ML Pipeline
     ─────────────                      ───────────

  ┌─────────────────┐              ┌─────────────────┐
  │                 │              │                 │
  │   Record Task   │───────────►│   Fine-tune     │
  │   (capture)     │   Episodes   │   VLM           │
  │                 │              │   (ml)          │
  └─────────────────┘              └────────┬────────┘
                                            │
                                            │ Model
                                            ▼
  ┌─────────────────┐              ┌─────────────────┐
  │                 │              │                 │
  │   Run Agent     │◄─────────────│   Evaluate      │
  │   (deploy)      │   Policy     │   (evals)       │
  │                 │              │                 │
  └─────────────────┘              └─────────────────┘


  Supporting Components:
  ─────────────────────

  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │   Grounding   │  │   Retrieval   │  │   Viewer      │
  │   (grounding) │  │   (retrieval) │  │   (viewer)    │
  └───────────────┘  └───────────────┘  └───────────────┘
     UI element         Demo search        Visualization
     localization       and matching       components
```

---

## Appendix C: Sample Email Signup Copy

**Headline**: Stay Updated

**Body**: Get notified about new releases, features, and GUI automation best practices.

**Form Fields**:
- Email (required)
- Company (optional)
- Use case: [Dropdown: Personal, Startup, Enterprise, Research, Other]

**CTA Button**: Subscribe

**Privacy Note**: We respect your privacy. Unsubscribe anytime.
