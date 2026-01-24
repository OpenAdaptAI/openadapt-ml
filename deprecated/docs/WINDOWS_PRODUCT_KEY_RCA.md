# Root Cause Analysis: Windows Product Key Prompt Recurring Issue

**Document Created**: 2026-01-20
**Author**: Claude Code Agent
**Status**: ACTIVE - Requires Immediate Fix

---

## 1. Problem Statement

### What Happens
The Windows installer shows interactive dialogs that block unattended installation:
1. **"Select the operating system you want to install"** - Edition picker dialog
2. **"Enter your product key"** - Product key prompt (less common)

### When It Happens
- On first Windows installation inside waa-auto container
- After deleting cached Windows disk image (`/data/waa-storage/data.img`)
- Seemingly randomly after "fixes" are applied

### How Often
This issue has recurred **at least 3-4 times** despite being "fixed" each time.

### Impact
- 30-45 minute delays per occurrence
- Requires manual VNC intervention
- Breaks fully-automated WAA benchmark runs
- Wastes Azure VM billing ($0.19/hr)

---

## 2. Timeline of "Fixes"

| Date | Fix Attempted | Outcome | Why It Failed |
|------|---------------|---------|---------------|
| ~Jan 15 | Added `<InstallFrom>` element to autounattend.xml | Partial | Only applied to one XML file |
| ~Jan 17 | Set `VERSION=11` for Windows 11 Pro | Worked briefly | CLI later overridden with `VERSION=11e` in some places |
| ~Jan 18 | Applied sed patch to BOTH win11x64.xml files | Should work | Dockerfile ENV sets VERSION=11e, CLI docker run sets VERSION=11 - MISMATCH |
| ~Jan 19 | Documented fix in CLAUDE.md | Documentation only | Actual code still has contradictions |
| Jan 20 | Copied windowsarena XML to both file paths | Should work | But cached storage may have old Windows installed |

---

## 3. Root Cause Analysis

### THE CORE PROBLEM: Configuration Contradiction

**There are TWO different VERSION values being used:**

1. **Dockerfile (line 275)**: `ENV VERSION="11e"` (Enterprise Evaluation)
2. **CLI docker run commands (cli.py)**: `-e VERSION=11` (Windows 11 Pro)

This creates a race condition:
- The Dockerfile builds with `VERSION=11e` which uses `win11x64-enterprise-eval.xml`
- But docker run overrides with `VERSION=11` which uses `win11x64.xml`
- The XML files patched during build may not be the ones used at runtime!

### Why Each Fix Failed

#### Fix 1: Add `<InstallFrom>` element
**What it did**: Added IMAGE/INDEX selector to autounattend.xml
**Why it failed**: Only patched one XML file, but dockurr/windows selects XML based on VERSION at runtime

#### Fix 2: Use VERSION=11 for Windows 11 Pro
**What it did**: Changed CLI to use VERSION=11
**Why it failed**: Dockerfile still has VERSION=11e, creating mismatch
**Documentation says**: "VERSION=11 downloads Windows 11 Pro - fully unattended, no dialogs"
**Reality**: This is only true IF the XML file for that version is properly patched

#### Fix 3: Patch BOTH XML files
**What it did**: Applied sed patches to both win11x64.xml and win11x64-enterprise-eval.xml
**Why it should work**: Covers both VERSION=11 and VERSION=11e
**Why it might still fail**:
1. Cached Windows installation (`data.img`) was created before patches
2. dockurr/windows may regenerate XML at runtime in some scenarios

#### Fix 4: Copy windowsarena XML to both paths
**What it did**: `COPY --from=windowsarena/winarena:latest /run/assets/win11x64-enterprise-eval.xml /run/assets/win11x64.xml`
**Why it might fail**: The windowsarena XML may also lack InstallFrom element

### Secondary Issues

#### Issue A: Cached Windows Installation
- Once Windows is installed, it's cached in `/data/waa-storage/data.img`
- A fix to XML has NO EFFECT on existing installation
- Must delete `data.img` to force reinstallation with new XML

#### Issue B: Upstream XML Changes
- dockurr/windows updates may overwrite our patches
- windowsarena/winarena updates may have different XML format

#### Issue C: Enterprise Evaluation vs Pro
- `VERSION=11e` (Enterprise Evaluation) - Uses GVLK key, no activation needed
- `VERSION=11` (Pro) - May prompt for product key if XML not correct
- CLAUDE.md says "VERSION=11e shows edition picker dialog" - WRONG, it's the opposite
- The confusion between these has led to incorrect "fixes"

---

## 4. Current State Analysis

### Dockerfile (waa_deploy/Dockerfile)

```dockerfile
# Line 85-96: Copies XML and patches InstallFrom
COPY --from=windowsarena/winarena:latest /run/assets/win11x64-enterprise-eval.xml /run/assets/win11x64.xml
COPY --from=windowsarena/winarena:latest /run/assets/win11x64-enterprise-eval.xml /run/assets/win11x64-enterprise-eval.xml

RUN sed -i 's|<InstallTo>|<InstallFrom>\n...\n</InstallFrom>\n<InstallTo>|' /run/assets/win11x64.xml
RUN sed -i 's|<InstallTo>|<InstallFrom>\n...\n</InstallFrom>\n<InstallTo>|' /run/assets/win11x64-enterprise-eval.xml

# Line 275: ENV sets VERSION=11e
ENV VERSION="11e"
```

### CLI (cli.py) - Multiple Contradictions

```python
# Line 3262: Comment says to use VERSION=11
# Note: VERSION=11e downloads Enterprise Evaluation which shows edition picker dialog

# Line 3273: Docker run uses VERSION=11
-e VERSION=11 \

# Line 6110: Another docker run uses VERSION=11
-e VERSION=11 \

# Line 6180: Yet another docker run uses VERSION=11
"-e VERSION=11 "
```

### CLAUDE.md - Contradictory Documentation

```markdown
# Line 769: Says VERSION=11 is correct
- Setting `VERSION=11` downloads Windows 11 Pro (~6.6 GB) - **fully unattended, no dialogs**
- Note: `VERSION=11e` downloads Enterprise Evaluation which shows an edition picker dialog

# Line 816: Says VERSION=11 is used
1. Uses `dockurr/windows:latest` (auto-downloads Windows Pro via `VERSION=11`)
```

### waa_setup.md - Says Opposite

```markdown
# Line 77: Says VERSION=11e is recommended
- `VERSION=11e` - Windows 11 Enterprise (6.6 GB, recommended)

# Line 91: Says Enterprise accepts GVLK (no product key needed)
- Accepts GVLK keys (no "product key" dialog during setup)
```

---

## 5. THE ACTUAL ROOT CAUSE

**The documentation is WRONG about which VERSION causes the edition picker.**

Based on dockurr/windows behavior:
- `VERSION=11` (Pro) - Uses `win11x64.xml`, may show product key dialog if XML wrong
- `VERSION=11e` (Enterprise Eval) - Uses `win11x64-enterprise-eval.xml`, GVLK key built-in

The edition picker ("Select operating system") is caused by:
1. Missing `<InstallFrom>` element with IMAGE/INDEX
2. An install.wim with multiple editions where Windows can't auto-detect which to use

This is INDEPENDENT of VERSION - both Pro and Enterprise can show the picker if their XML lacks InstallFrom.

**The real fix:**
1. Ensure BOTH XML files have `<InstallFrom>` element (DONE in Dockerfile)
2. Use consistent VERSION everywhere (currently inconsistent)
3. Delete cached data.img after any XML changes (NOT automated)

---

## 6. Permanent Solution

### Step 1: Standardize on VERSION=11e (Enterprise Evaluation)

**Why**: Enterprise Evaluation has built-in GVLK key, never prompts for product key.

**Changes Required**:

1. **cli.py**: Change ALL `VERSION=11` to `VERSION=11e`
   - Line 3273
   - Line 6110
   - Line 6180

2. **CLAUDE.md**: Fix incorrect documentation
   - Line 769-770: Remove claim that 11e shows edition picker
   - Line 816: Update to say VERSION=11e

3. **waa_setup.md**: Already correct, keep as-is

### Step 2: Ensure InstallFrom is Added to windowsarena's XML

The current approach copies windowsarena's XML then patches. But we should verify the source XML.

**Add verification step to Dockerfile**:
```dockerfile
# After patching, verify InstallFrom exists
RUN grep -q "InstallFrom" /run/assets/win11x64.xml || (echo "ERROR: InstallFrom patch failed" && exit 1)
RUN grep -q "InstallFrom" /run/assets/win11x64-enterprise-eval.xml || (echo "ERROR: InstallFrom patch failed" && exit 1)
```

### Step 3: Automate Cache Invalidation

Add a check that detects XML/Dockerfile changes and forces reinstall:

```python
# In cli.py run_waa or start_windows
def needs_reinstall():
    """Check if Windows disk image predates Dockerfile changes."""
    image_path = "/data/waa-storage/data.img"
    dockerfile_path = "openadapt_ml/benchmarks/waa_deploy/Dockerfile"

    if not os.path.exists(image_path):
        return False  # No image, will install fresh

    image_mtime = os.path.getmtime(image_path)
    dockerfile_mtime = os.path.getmtime(dockerfile_path)

    if dockerfile_mtime > image_mtime:
        print("WARNING: Dockerfile changed since Windows was installed.")
        print("Run with --reinstall to apply XML changes.")
        return True
    return False
```

### Step 4: Add Pre-Flight Check

Before starting Windows, verify the container has correct XML:

```bash
# In start_windows or run_waa
docker run --rm waa-auto:latest grep -c "InstallFrom" /run/assets/win11x64-enterprise-eval.xml
# Should return 1, not 0
```

---

## 7. Verification Checklist

After implementing fixes, verify:

- [ ] `grep VERSION Dockerfile` returns only `VERSION="11e"`
- [ ] `grep "VERSION=11" cli.py` returns 0 matches (all should be VERSION=11e)
- [ ] `docker run --rm waa-auto:latest grep InstallFrom /run/assets/win11x64-enterprise-eval.xml` returns match
- [ ] `docker run --rm waa-auto:latest grep InstallFrom /run/assets/win11x64.xml` returns match
- [ ] Delete `/data/waa-storage/data.img` and run fresh install
- [ ] Windows installs without any dialogs (verify via VNC)
- [ ] WAA server starts automatically

---

## 8. Prevention

### Documentation Hygiene
- Add VERSION/XML info to README in waa_deploy folder
- Update CLAUDE.md with correct information about VERSION behavior

### Code Review Checklist
When modifying WAA/Windows code:
- [ ] Check all VERSION= references are consistent
- [ ] Check all XML patches apply to both files
- [ ] Test with fresh install (delete data.img)

### Automated Testing
Add CI step to:
1. Build waa-auto image
2. Run container and check XML has InstallFrom
3. Check VERSION consistency across files

---

## 9. Files to Modify

### Immediate (P0)

1. `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/cli.py`
   - Change lines 3273, 6110, 6180 from `VERSION=11` to `VERSION=11e`

2. `/Users/abrichr/oa/src/openadapt-ml/CLAUDE.md`
   - Line 769-770: Fix incorrect claim about VERSION=11e showing picker
   - Line 816: Update to VERSION=11e

### Documentation Update (P1)

3. Create `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/README.md`
   - Document VERSION behavior correctly
   - Document XML patching strategy

### Automation (P2)

4. Add cache invalidation check to cli.py
5. Add pre-flight XML verification to cli.py

---

## 10. Summary

The Windows product key prompt keeps recurring because:

1. **Configuration mismatch**: Dockerfile uses VERSION=11e, CLI uses VERSION=11
2. **Documentation confusion**: CLAUDE.md incorrectly states VERSION=11e shows picker
3. **Cache persistence**: XML fixes don't apply to existing Windows installations
4. **No verification**: No automated check that XML patches were applied

**The fix is simple**: Standardize on VERSION=11e everywhere, ensure InstallFrom is patched, and delete cached installations after changes.

---

## Appendix A: dockurr/windows VERSION Behavior

From dockurr/windows source code:

| VERSION | ISO Downloaded | XML File Used |
|---------|----------------|---------------|
| `11` | Windows 11 Pro | `win11x64.xml` |
| `11e` | Windows 11 Enterprise Eval | `win11x64-enterprise-eval.xml` |
| `11p` | Windows 11 Pro | `win11x64.xml` |
| `win11x64` | Windows 11 Pro | `win11x64.xml` |
| `win11x64-enterprise-eval` | Windows 11 Enterprise Eval | `win11x64-enterprise-eval.xml` |

The Enterprise Evaluation ISO includes a GVLK key, so it NEVER prompts for product key.
The Pro ISO may prompt if the XML doesn't specify a key or skip the prompt.

## Appendix B: XML Patching Commands

To verify XML has InstallFrom:
```bash
docker run --rm waa-auto:latest cat /run/assets/win11x64-enterprise-eval.xml | grep -A5 InstallFrom
```

Expected output:
```xml
<InstallFrom>
  <MetaData wcm:action="add">
    <Key>/IMAGE/INDEX</Key>
    <Value>1</Value>
  </MetaData>
</InstallFrom>
```

If missing, the XML patch failed during Docker build.
