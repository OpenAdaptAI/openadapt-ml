# API Agent Integration Approaches for WAA

## Problem Statement

Windows Agent Arena (WAA) uses a modular agent architecture where `run.py` instantiates agents based on the `--agent` flag. The default agent is `navi`, which has known bugs (NoneType errors). We want to add support for API-backed agents (`api-claude` and `api-openai`) that use Claude Sonnet 4.5 or GPT-5.1 directly via their hosted APIs.

### Current Architecture

WAA's `run.py` agent selection (lines 163-191):

```python
if cfg_args["agent_name"] == "navi":
    # ... NaviAgent setup
    agent = NaviAgent(...)
elif cfg_args["agent_name"] == "claude":
    from mm_agents.claude.agent import ClaudeAgent
    agent = ClaudeAgent()
else:
    raise ValueError(f"Unknown agent name: {cfg_args['agent_name']}")
```

The agent is invoked via shell scripts:
1. `entry.sh` parses CLI args (`--agent api-claude`)
2. Calls `start_client.sh` which runs `python run.py --agent api-claude ...`

### What We Need to Add

```python
elif cfg_args["agent_name"] in ["api-claude", "api-openai"]:
    from mm_agents.api_agent import ApiAgent
    provider = "anthropic" if cfg_args["agent_name"] == "api-claude" else "openai"
    agent = ApiAgent(provider=provider, temperature=args.temperature)
```

---

## Current Approach and Why It Fails

### What the Dockerfile Tries to Do

The current `Dockerfile.simplified` (lines 101-107) attempts build-time patching:

```dockerfile
# Patch run.py to support api-claude and api-openai agents
RUN python3 -c "import re; \
f = open('/client/run.py', 'r'); c = f.read(); f.close(); \
patch = '''    elif cfg_args[\"agent_name\"] in [\"api-claude\", \"api-openai\"]:\n        from mm_agents.api_agent import ApiAgent\n        provider = \"anthropic\" if cfg_args[\"agent_name\"] == \"api-claude\" else \"openai\"\n        agent = ApiAgent(provider=provider, temperature=args.temperature)\n'''; \
c = c.replace('    else:\\n        raise ValueError', patch + '    else:\\n        raise ValueError'); \
f = open('/client/run.py', 'w'); f.write(c); f.close(); \
print('Patched run.py for API agents')"
```

### Why It Fails

1. **Order dependency**: The patch runs during Docker build, but Python 3 installation happens later in the Dockerfile (or sometimes fails silently).

2. **String matching fragility**: The `c.replace()` approach depends on exact whitespace matching. If WAA's `run.py` has any formatting differences (tabs vs spaces, trailing whitespace), the patch silently fails.

3. **No verification**: The patch prints success but doesn't verify the replacement actually occurred.

4. **Multi-line string escaping**: Getting proper indentation through the shell -> Docker -> Python chain is error-prone.

5. **WAA upstream changes**: Any change to the `else: raise ValueError` line (e.g., different error message, added comment) breaks the patch.

---

## Proposed Solutions

### Approach 1: Runtime Patching via docker exec

Patch `run.py` at runtime when starting the benchmark, not during Docker build.

**Implementation**:

```bash
# In CLI's run-waa command, after container starts but before benchmark runs:
docker exec winarena python3 -c "
import sys
sys.path.insert(0, '/client')

# Read run.py
with open('/client/run.py', 'r') as f:
    content = f.read()

# Check if already patched
if 'api-claude' in content:
    print('Already patched')
    sys.exit(0)

# Find insertion point
marker = \"    else:\\n        raise ValueError(f\\\"Unknown agent name\"
if marker not in content:
    print('ERROR: Could not find insertion point')
    sys.exit(1)

# Insert our elif block
patch = '''    elif cfg_args[\"agent_name\"] in [\"api-claude\", \"api-openai\"]:
        from mm_agents.api_agent import ApiAgent
        provider = \"anthropic\" if cfg_args[\"agent_name\"] == \"api-claude\" else \"openai\"
        agent = ApiAgent(provider=provider, temperature=args.temperature)
'''

content = content.replace(marker, patch + marker)

with open('/client/run.py', 'w') as f:
    f.write(content)

print('Patched successfully')
"
```

**Pros**:
- Python is definitely installed when this runs (container is fully built)
- Can be easily debugged (run interactively)
- CLI can verify patch succeeded before running benchmark
- No Docker rebuild required to iterate

**Cons**:
- Adds latency to benchmark startup (~1 second)
- Patch must be idempotent (check if already applied)
- Still fragile to WAA upstream changes

**Complexity**: Low
**Risk**: Medium

---

### Approach 2: Ship Pre-Patched run.py

Instead of patching, ship a complete modified `run.py` and copy it during build.

**Implementation**:

1. Copy `vendor/WindowsAgentArena/src/win-arena-container/client/run.py` to `waa_deploy/run_patched.py`
2. Apply the patch manually (one-time)
3. In Dockerfile:
   ```dockerfile
   # Replace run.py with our patched version
   COPY run_patched.py /client/run.py
   ```

**Pros**:
- No runtime patching complexity
- Guaranteed to work (no string matching)
- Easy to verify by inspecting `run_patched.py`
- Fast (no runtime overhead)

**Cons**:
- **Must sync with WAA upstream**: When WAA updates `run.py`, we must manually merge changes
- Increases maintenance burden
- Easy to forget to sync, causing subtle bugs

**Complexity**: Very Low
**Risk**: Medium (maintenance burden)

---

### Approach 3: Volume Mount at Runtime

Mount a patched `run.py` from the host into the container at runtime.

**Implementation**:

```bash
# Create patched run.py on host (once)
cp /client/run.py /tmp/run_patched.py
# Apply patch to /tmp/run_patched.py

# When starting container, mount it:
docker run -v /tmp/run_patched.py:/client/run.py:ro ...
```

Or using docker-compose:
```yaml
volumes:
  - ./waa_deploy/run_patched.py:/client/run.py:ro
```

**Pros**:
- No Docker rebuild required to change patch
- Can test different agent implementations quickly
- Easy to debug (edit file on host)

**Cons**:
- Requires the patched file to exist on the VM
- More complex deployment (file must be synced to VM)
- Adds complexity to CLI's Docker run command
- File must be recreated if WAA upstream changes

**Complexity**: Medium
**Risk**: Low

---

### Approach 4: Fork WAA and Add Agents Upstream

Create a fork of WindowsAgentArena and add `api-claude`/`api-openai` agents properly.

**Implementation**:

1. Fork `microsoft/WindowsAgentArena` to `OpenAdapt/WindowsAgentArena`
2. Add `mm_agents/api_agent.py` to the fork
3. Modify `run.py` in the fork to support new agents
4. Update our Dockerfile to use our fork's image or build from our fork:
   ```dockerfile
   FROM openadapt/winarena:latest
   # or
   FROM ghcr.io/openadapt/windowsagentarena:latest
   ```

**Pros**:
- Clean, maintainable solution
- Can contribute upstream (PR to Microsoft)
- Other OpenAdapt users benefit
- No patching fragility

**Cons**:
- Requires maintaining a fork
- Fork must track upstream changes (rebasing)
- Takes longer to set up initially
- Microsoft may not accept PR (their prerogative)

**Complexity**: High (initially)
**Risk**: Low (long-term)

---

### Approach 5: Monkey-Patch via Python Import Hook (EXPERIMENTAL)

Create a Python module that patches `run.py` when it's imported.

**Implementation**:

Create `api_agent_loader.py`:
```python
"""Import hook that patches run.py to support API agents."""
import sys
import importlib.abc
import importlib.util

class ApiAgentPatcher(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == '__main__' and 'run.py' in sys.argv[0]:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def exec_module(self, module):
        # Inject our agent handlers before run.py's main executes
        import mm_agents.api_agent as api_agent
        # Patch the agent selection at runtime
        ...

sys.meta_path.insert(0, ApiAgentPatcher())
```

**Pros**:
- No file modification
- Dynamic patching

**Cons**:
- Complex, hard to understand
- Fragile (depends on Python import internals)
- Harder to debug
- Overkill for this use case

**Complexity**: High
**Risk**: High

---

### Approach 6: Wrapper Script (RECOMMENDED)

Instead of patching `run.py`, create a wrapper script that handles API agents separately.

**Implementation**:

Create `run_with_api_agents.py`:
```python
#!/usr/bin/env python3
"""Wrapper for WAA run.py that adds API agent support."""
import sys
import os

# Check if using API agent
agent_name = None
for i, arg in enumerate(sys.argv):
    if arg == '--agent' and i + 1 < len(sys.argv):
        agent_name = sys.argv[i + 1]
        break

if agent_name in ['api-claude', 'api-openai']:
    # Run our custom agent handler
    from run_api_agent import main
    main()
else:
    # Fall through to original run.py
    exec(open('/client/run.py').read())
```

Then create `run_api_agent.py` that:
1. Imports the ApiAgent
2. Sets up the DesktopEnv
3. Runs the evaluation loop (copied from run.py's test() function)

**Pros**:
- No modification to WAA's run.py
- Clear separation of concerns
- Easy to maintain and test independently
- Can add features without touching upstream code

**Cons**:
- Some code duplication (evaluation loop)
- Must keep `run_api_agent.py` in sync with WAA's test() function
- Two entry points to maintain

**Complexity**: Medium
**Risk**: Low

---

## Comparison Matrix

| Approach | Complexity | Fragility | Maintenance | Deployment | Recommended? |
|----------|-----------|-----------|-------------|------------|--------------|
| 1. Runtime patch | Low | High | Low | Easy | Maybe |
| 2. Pre-patched file | Very Low | Medium | High | Easy | Short-term |
| 3. Volume mount | Medium | Medium | Medium | Medium | No |
| 4. Fork upstream | High (init) | Low | Medium | Medium | Long-term |
| 5. Import hook | High | Very High | High | Easy | No |
| **6. Wrapper script** | **Medium** | **Low** | **Low** | **Easy** | **YES** |

---

## Recommended Approach: Wrapper Script (Approach 6)

### Why This Approach

1. **Zero modification to WAA code**: We don't touch `run.py`, eliminating all patching fragility.

2. **Clear separation**: API agent logic is self-contained and easy to test.

3. **Future-proof**: WAA upstream changes won't break our integration (unless they change the evaluation loop API, which is rare).

4. **Easy to extend**: Adding new agent types (e.g., `api-gemini`) only requires updating the wrapper.

### Implementation Plan

1. **Create `/client/run_api_agent.py`**:
   - Copy the core evaluation loop from `run.py`
   - Import and use `ApiAgent` for agent creation
   - Keep CLI argument parsing compatible

2. **Create `/client/run_wrapper.py`** (or modify `start_client.sh`):
   - Check `--agent` value
   - Route to `run_api_agent.py` or original `run.py`

3. **Update Dockerfile**:
   ```dockerfile
   COPY api_agent.py /client/mm_agents/api_agent.py
   COPY run_api_agent.py /client/run_api_agent.py
   COPY run_wrapper.py /client/run_wrapper.py

   # Patch start_client.sh to use wrapper
   RUN sed -i 's|python run.py|python run_wrapper.py|' /start_client.sh
   ```

4. **Update CLI** (optional):
   - The CLI can continue using `--agent api-claude` as before
   - No changes needed if using wrapper approach

### Fallback: Approach 2 (Pre-patched file)

If the wrapper approach is deemed too much work, use Approach 2 as a quick fix:

1. Copy WAA's `run.py` to `waa_deploy/run_patched.py`
2. Add the API agent elif block manually
3. COPY in Dockerfile

This is simple but requires watching for WAA upstream changes.

---

## Files Referenced

- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/Dockerfile`
- `/Users/abrichr/oa/src/openadapt-ml/openadapt_ml/benchmarks/waa_deploy/api_agent.py`
- `/Users/abrichr/oa/src/openadapt-ml/vendor/WindowsAgentArena/src/win-arena-container/client/run.py`
- `/Users/abrichr/oa/src/openadapt-ml/vendor/WindowsAgentArena/src/win-arena-container/entry.sh`
- `/Users/abrichr/oa/src/openadapt-ml/vendor/WindowsAgentArena/src/win-arena-container/start_client.sh`
