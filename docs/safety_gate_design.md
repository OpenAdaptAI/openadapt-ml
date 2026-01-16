# Safety Gate Design

## Overview

The Safety Gate is a **deterministic** (non-learned) runtime module that intercepts actions between policy decision and execution. It provides a critical safety layer for GUI automation agents, preventing destructive or irreversible actions without explicit human confirmation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent Runtime                                │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │   Policy    │───>│  Safety Gate │───>│  Action Executor      │  │
│  │  (VLM/LLM)  │    │  (this doc)  │    │  (OS/Browser hooks)   │  │
│  └─────────────┘    └──────────────┘    └───────────────────────┘  │
│        │                   │                       │                │
│        │                   ▼                       │                │
│        │            ┌──────────────┐               │                │
│        │            │   ALLOW /    │               │                │
│        │            │   BLOCK /    │               │                │
│        │            │   REQUIRE    │               │                │
│        │            │ CONFIRMATION │               │                │
│        │            └──────────────┘               │                │
└────────│───────────────────────────────────────────│────────────────┘
         │                                           │
         ▼                                           ▼
    Observation                                 Action
    (screenshot,                               (click, type,
     a11y tree,                                 key, scroll,
     window info)                               etc.)
```

## Design Principles

1. **Deterministic**: All safety rules are explicit, regex-based, or threshold-based. No ML inference involved.
2. **Conservative**: When in doubt, require confirmation rather than allow.
3. **Configurable**: All thresholds, patterns, and behaviors are configurable via `SafetyConfig`.
4. **Transparent**: Every decision includes detailed reasoning via `SafetyAssessment`.
5. **Stateful for Loop Detection**: Maintains history of visited states to detect infinite loops.
6. **Non-blocking by Default**: Returns assessments synchronously; confirmation handling is caller's responsibility.

## Safety Checks

### 1. Blocklist (Destructive Keywords)

**Purpose**: Prevent actions that could cause irreversible damage to data or systems.

**Mechanism**: Regex pattern matching against action text content.

**Default Patterns**:
```python
blocklist_patterns = [
    # File/data destruction
    r"\bdelete\b",
    r"\bremove\b",
    r"\bformat\b",
    r"\breset\b",
    r"\bbroadcast\b",

    # Database destruction
    r"\bdrop\s+table\b",
    r"\btruncate\b",

    # Shell destruction
    r"\brm\s+-rf\b",
    r"\bsudo\s+rm\b",
]
```

**Decision**: `BLOCK` - Action is outright blocked, not just requiring confirmation.

### 2. Irreversibility Detection

**Purpose**: Require human confirmation before committing irreversible actions.

**Mechanism**: Regex pattern matching for commitment/submission keywords.

**Default Patterns**:
```python
irreversible_patterns = [
    # Submission actions
    r"\bsubmit\b",
    r"\bsend\b",
    r"\bapply\b",
    r"\bconfirm\b",

    # Document closure
    r"\bclos(?:e|ing)\b.*\bunsaved\b",

    # Financial actions
    r"\bpurchase\b",
    r"\bcheckout\b",
    r"\bpay\b",
]
```

**Decision**: `REQUIRE_CONFIRMATION` - Human must approve before execution.

### 3. Confidence Threshold

**Purpose**: Request human oversight when the policy model is uncertain.

**Mechanism**: Compare action confidence score against configurable threshold.

**Default Threshold**: `0.7` (70%)

**Logic**:
```python
if action.confidence is not None and action.confidence < threshold:
    return SafetyDecision.REQUIRE_CONFIRMATION
```

**Note**: Confidence must be provided by the policy. If not provided, this check is skipped.

### 4. Loop Detection

**Purpose**: Halt agents stuck in repetitive loops (visiting same state repeatedly).

**Mechanism**: Track state hashes across execution trace; trigger if same state visited N times.

**Default Loop Threshold**: `3` visits

**State Hash Computation**:
```python
def compute_state_hash(observation: Observation) -> str:
    """Compute hash from key observation attributes."""
    components = [
        observation.window_title or "",
        observation.app_name or "",
        observation.url or "",
        # Optionally: screenshot perceptual hash
    ]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]
```

**Decision**: `BLOCK` - Infinite loop detection triggers a hard stop.

### 5. Credential Guard

**Purpose**: Prevent accidental exposure of credentials to automation.

**Mechanism**: Detect TYPE actions targeting password/secret fields.

**Detection Logic**:
1. Check if action is `TYPE`
2. Check if target element's label/name contains sensitive keywords:
   - `password`
   - `token`
   - `secret`
   - `api_key`
   - `apikey`
   - `credential`

**Default**: Warn and require confirmation.

**Configuration**: Can be set to `allow` specific patterns via allowlist.

### 6. App/Window Mismatch

**Purpose**: Detect when agent attempts to act on unexpected application.

**Mechanism**: Compare action's target window/app against expected context.

**Use Case**: Agent is automating Chrome but focus shifted to Terminal.

**Decision**: `REQUIRE_CONFIRMATION` with detailed warning.

## Interface

```python
from enum import Enum
from dataclasses import dataclass, field


class SafetyDecision(Enum):
    """Possible safety gate decisions."""
    ALLOW = "allow"              # Action can proceed
    BLOCK = "block"              # Action must not proceed
    REQUIRE_CONFIRMATION = "require_confirmation"  # Human must approve


@dataclass
class SafetyAssessment:
    """Result of safety gate evaluation."""
    decision: SafetyDecision
    reason: str                  # Human-readable explanation
    triggered_rules: list[str]   # Which safety rules triggered
    confidence: float            # How certain the gate is (1.0 = certain)


@dataclass
class SafetyConfig:
    """Configuration for safety gate behavior."""

    # Blocklist patterns (BLOCK decision)
    blocklist_patterns: list[str] = field(default_factory=lambda: [...])

    # Irreversible action patterns (REQUIRE_CONFIRMATION)
    irreversible_patterns: list[str] = field(default_factory=lambda: [...])

    # Confidence threshold (below this = REQUIRE_CONFIRMATION)
    confidence_threshold: float = 0.7

    # Loop detection threshold (same state N times = BLOCK)
    loop_threshold: int = 3

    # Credential field patterns (TYPE to these = REQUIRE_CONFIRMATION)
    credential_patterns: list[str] = field(default_factory=lambda: [...])

    # Allowed credential field patterns (override credential_patterns)
    credential_allowlist: list[str] = field(default_factory=list)

    # Expected app/window context (None = don't check)
    expected_app: str | None = None
    expected_window_pattern: str | None = None


class SafetyGate:
    """Deterministic safety gate for GUI automation actions."""

    def __init__(self, config: SafetyConfig | None = None):
        """Initialize with optional custom configuration."""
        ...

    def assess(
        self,
        action: Action,
        observation: Observation,
        trace: list[Step] | None = None,
    ) -> SafetyAssessment:
        """Evaluate an action against all safety rules.

        Args:
            action: The action proposed by the policy
            observation: Current state observation
            trace: Execution history (for loop detection)

        Returns:
            SafetyAssessment with decision and reasoning
        """
        ...

    def reset(self) -> None:
        """Clear internal state (loop detection history)."""
        ...
```

## Usage Example

```python
from openadapt_ml.runtime import SafetyGate, SafetyConfig, SafetyDecision
from openadapt_ml.runtime.policy import AgentPolicy

# Configure safety gate
config = SafetyConfig(
    confidence_threshold=0.8,  # Stricter than default
    loop_threshold=2,           # More aggressive loop detection
    expected_app="Chrome",      # Only allow Chrome actions
)
gate = SafetyGate(config)

# In agent loop
policy = AgentPolicy(adapter)

for observation in environment:
    # Policy decides action
    action, thought, state, raw = policy.predict_action_from_sample(sample)

    # Safety gate evaluates
    assessment = gate.assess(action, observation, trace=execution_history)

    if assessment.decision == SafetyDecision.ALLOW:
        # Execute action
        environment.execute(action)
    elif assessment.decision == SafetyDecision.REQUIRE_CONFIRMATION:
        # Prompt human for confirmation
        if human_confirms(assessment.reason):
            environment.execute(action)
        else:
            # Handle rejection (retry, abort, etc.)
            pass
    else:  # BLOCK
        # Log and skip action
        logger.warning(f"Action blocked: {assessment.reason}")
```

## Check Priority

Safety checks are evaluated in this order (first match wins):

1. **Blocklist** - Immediate BLOCK, no further checks
2. **Loop Detection** - BLOCK if loop detected
3. **Credential Guard** - REQUIRE_CONFIRMATION if typing to sensitive field
4. **Irreversibility** - REQUIRE_CONFIRMATION for irreversible actions
5. **App/Window Mismatch** - REQUIRE_CONFIRMATION if context mismatch
6. **Confidence Threshold** - REQUIRE_CONFIRMATION if low confidence
7. **Default** - ALLOW if no rules triggered

## State Management

The SafetyGate maintains internal state for loop detection:

```python
class SafetyGate:
    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self._state_visit_counts: dict[str, int] = {}  # state_hash -> count

    def reset(self) -> None:
        """Clear loop detection history. Call between episodes."""
        self._state_visit_counts.clear()
```

**Important**: Call `gate.reset()` between episodes to clear loop detection state.

## Extension Points

### Custom Safety Rules

The SafetyGate can be extended with custom rules via subclassing:

```python
class CustomSafetyGate(SafetyGate):
    def assess(self, action, observation, trace=None):
        # Custom check first
        if self._is_dangerous_time():
            return SafetyAssessment(
                decision=SafetyDecision.BLOCK,
                reason="Actions blocked during maintenance window",
                triggered_rules=["maintenance_window"],
                confidence=1.0,
            )

        # Fall back to standard checks
        return super().assess(action, observation, trace)
```

### Integration with Confirmation Systems

The SafetyGate returns assessments but does not handle confirmation UI. Integrators should implement their own confirmation flow:

- CLI: Interactive prompt
- Web: Modal dialog with action preview
- API: Webhook to approval system
- Slack/Teams: Bot message with approve/reject buttons

## Testing

Safety rules should be tested with:

1. **Unit tests**: Each rule tested in isolation
2. **Integration tests**: Full assess() flow with mock actions/observations
3. **Regression tests**: Known dangerous action patterns should always trigger

Example test cases:
- `TYPE(text="rm -rf /")` -> BLOCK (blocklist)
- `CLICK` on "Submit Order" button -> REQUIRE_CONFIRMATION (irreversible)
- Low-confidence action (0.5) -> REQUIRE_CONFIRMATION (threshold)
- Same state visited 3 times -> BLOCK (loop)
- `TYPE` to password field -> REQUIRE_CONFIRMATION (credential)

## Limitations

1. **Text-based detection**: Relies on pattern matching; sophisticated evasion possible.
2. **No semantic understanding**: Cannot reason about action intent or consequences.
3. **Screenshot analysis not included**: Does not analyze screenshot content for sensitive data.
4. **Confirmation is caller's responsibility**: Gate only returns decisions, not UI.

## Future Enhancements

- **Screenshot PII detection**: OCR + regex for visible sensitive data
- **Rollback capability**: Snapshot state before risky actions
- **Confidence calibration**: Learn appropriate thresholds from user feedback
- **Action simulation**: Preview action effects before execution
- **Audit logging**: Persistent log of all safety decisions for compliance
