"""
Safety Gate Module - Deterministic safety checks for GUI automation actions.

The Safety Gate runs AFTER the policy decides but BEFORE execution, providing
a critical safety layer that prevents destructive or irreversible actions
without explicit human confirmation.

Example Usage:
    from openadapt_ml.runtime import SafetyGate, SafetyConfig, SafetyDecision

    # Create with default config
    gate = SafetyGate()

    # Or customize
    config = SafetyConfig(
        confidence_threshold=0.8,
        loop_threshold=2,
        expected_app="Chrome",
    )
    gate = SafetyGate(config)

    # Evaluate action
    assessment = gate.assess(action, observation, trace=history)

    if assessment.decision == SafetyDecision.ALLOW:
        execute(action)
    elif assessment.decision == SafetyDecision.REQUIRE_CONFIRMATION:
        if user_confirms(assessment.reason):
            execute(action)
    else:  # BLOCK
        log_warning(assessment.reason)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from openadapt_ml.schema import Action, ActionType, Observation, Step


class SafetyDecision(str, Enum):
    """Possible safety gate decisions."""

    ALLOW = "allow"  # Action can proceed without intervention
    BLOCK = "block"  # Action must not proceed under any circumstances
    REQUIRE_CONFIRMATION = "require_confirmation"  # Human must approve


@dataclass
class SafetyAssessment:
    """Result of safety gate evaluation.

    Attributes:
        decision: The safety decision (ALLOW, BLOCK, or REQUIRE_CONFIRMATION)
        reason: Human-readable explanation of the decision
        triggered_rules: List of safety rule names that triggered this decision
        confidence: How certain the gate is about this decision (1.0 = certain)
    """

    decision: SafetyDecision
    reason: str
    triggered_rules: list[str]
    confidence: float = 1.0

    def __str__(self) -> str:
        return f"SafetyAssessment({self.decision.value}: {self.reason})"


# Default blocklist patterns - actions that are ALWAYS blocked
DEFAULT_BLOCKLIST_PATTERNS = [
    # File/data destruction keywords
    r"\bdelete\b",
    r"\bremove\b",
    r"\bformat\b",
    r"\breset\b",
    r"\bbroadcast\b",
    # Database destruction
    r"\bdrop\s+table\b",
    r"\btruncate\b",
    # Shell destruction commands
    r"\brm\s+-rf\b",
    r"\bsudo\s+rm\b",
]

# Default irreversible action patterns - require confirmation
DEFAULT_IRREVERSIBLE_PATTERNS = [
    # Submission actions
    r"\bsubmit\b",
    r"\bsend\b",
    r"\bapply\b",
    r"\bconfirm\b",
    # Document closure (potentially with unsaved changes)
    r"\bclos(?:e|ing)\b",
    # Financial actions
    r"\bpurchase\b",
    r"\bcheckout\b",
    r"\bpay\b",
    r"\bbuy\b",
    r"\border\b",
]

# Default credential field patterns - typing to these requires confirmation
DEFAULT_CREDENTIAL_PATTERNS = [
    r"password",
    r"token",
    r"secret",
    r"api[_-]?key",
    r"apikey",
    r"credential",
    r"auth",
    r"private[_-]?key",
]


@dataclass
class SafetyConfig:
    """Configuration for safety gate behavior.

    All patterns are case-insensitive regular expressions.

    Attributes:
        blocklist_patterns: Patterns that trigger BLOCK decision
        irreversible_patterns: Patterns that trigger REQUIRE_CONFIRMATION
        confidence_threshold: Actions below this confidence require confirmation
        loop_threshold: Same state visited this many times triggers BLOCK
        credential_patterns: Field name patterns that indicate sensitive input
        credential_allowlist: Override patterns to allow typing in specific fields
        expected_app: Expected application name (None = don't check)
        expected_window_pattern: Regex for expected window title (None = don't check)
    """

    blocklist_patterns: list[str] = field(
        default_factory=lambda: DEFAULT_BLOCKLIST_PATTERNS.copy()
    )
    irreversible_patterns: list[str] = field(
        default_factory=lambda: DEFAULT_IRREVERSIBLE_PATTERNS.copy()
    )
    confidence_threshold: float = 0.7
    loop_threshold: int = 3
    credential_patterns: list[str] = field(
        default_factory=lambda: DEFAULT_CREDENTIAL_PATTERNS.copy()
    )
    credential_allowlist: list[str] = field(default_factory=list)
    expected_app: Optional[str] = None
    expected_window_pattern: Optional[str] = None


class SafetyGate:
    """Deterministic safety gate for GUI automation actions.

    The SafetyGate evaluates proposed actions against a set of safety rules
    and returns a SafetyAssessment indicating whether the action should be
    allowed, blocked, or require human confirmation.

    Safety checks are evaluated in priority order:
    1. Blocklist (immediate BLOCK)
    2. Loop Detection (BLOCK if loop detected)
    3. Credential Guard (REQUIRE_CONFIRMATION if typing to sensitive field)
    4. Irreversibility (REQUIRE_CONFIRMATION for irreversible actions)
    5. App/Window Mismatch (REQUIRE_CONFIRMATION if context mismatch)
    6. Confidence Threshold (REQUIRE_CONFIRMATION if low confidence)
    7. Default (ALLOW if no rules triggered)
    """

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        """Initialize the safety gate with optional custom configuration.

        Args:
            config: Safety configuration. If None, uses default SafetyConfig.
        """
        self.config = config or SafetyConfig()
        self._state_visit_counts: dict[str, int] = {}

        # Pre-compile regex patterns for efficiency
        self._blocklist_re = self._compile_patterns(self.config.blocklist_patterns)
        self._irreversible_re = self._compile_patterns(
            self.config.irreversible_patterns
        )
        self._credential_re = self._compile_patterns(self.config.credential_patterns)
        self._credential_allowlist_re = self._compile_patterns(
            self.config.credential_allowlist
        )
        self._window_pattern_re = (
            re.compile(self.config.expected_window_pattern, re.IGNORECASE)
            if self.config.expected_window_pattern
            else None
        )

    def _compile_patterns(self, patterns: list[str]) -> list[re.Pattern]:
        """Compile a list of regex patterns (case-insensitive)."""
        return [re.compile(p, re.IGNORECASE) for p in patterns]

    def _matches_any(self, text: str, patterns: list[re.Pattern]) -> Optional[str]:
        """Check if text matches any pattern. Returns matching pattern or None."""
        for pattern in patterns:
            if pattern.search(text):
                return pattern.pattern
        return None

    def _compute_state_hash(self, observation: Observation) -> str:
        """Compute a hash representing the current state for loop detection.

        Uses window title, app name, and URL to identify state. This is a
        lightweight approach that doesn't require image comparison.
        """
        components = [
            observation.window_title or "",
            observation.app_name or "",
            observation.url or "",
        ]
        state_str = "|".join(components)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    def _get_action_text(self, action: Action) -> str:
        """Extract text content from an action for pattern matching."""
        parts = []

        # Include typed text
        if action.text:
            parts.append(action.text)

        # Include key presses
        if action.key:
            parts.append(action.key)

        # Include URL for navigation
        if action.url:
            parts.append(action.url)

        # Include app name for open/close
        if action.app_name:
            parts.append(action.app_name)

        # Include window title for focus
        if action.window_title:
            parts.append(action.window_title)

        # Include raw action data if present
        if action.raw:
            for value in action.raw.values():
                if isinstance(value, str):
                    parts.append(value)

        return " ".join(parts)

    def _get_target_field_name(self, action: Action, observation: Observation) -> str:
        """Get the name/label of the field targeted by a TYPE action."""
        parts = []

        # Check action's target element
        if action.element:
            if action.element.name:
                parts.append(action.element.name)
            if action.element.role:
                parts.append(action.element.role)
            if action.element.automation_id:
                parts.append(action.element.automation_id)

        # Check observation's focused element
        if observation.focused_element:
            if observation.focused_element.name:
                parts.append(observation.focused_element.name)
            if observation.focused_element.role:
                parts.append(observation.focused_element.role)
            if observation.focused_element.automation_id:
                parts.append(observation.focused_element.automation_id)

        return " ".join(parts)

    def _get_action_confidence(self, action: Action) -> Optional[float]:
        """Extract confidence score from action's raw metadata."""
        if action.raw and "confidence" in action.raw:
            try:
                return float(action.raw["confidence"])
            except (TypeError, ValueError):
                pass
        return None

    def _check_blocklist(self, action: Action) -> Optional[SafetyAssessment]:
        """Check if action text matches any blocklist pattern."""
        action_text = self._get_action_text(action)
        matched = self._matches_any(action_text, self._blocklist_re)

        if matched:
            return SafetyAssessment(
                decision=SafetyDecision.BLOCK,
                reason=f"Action contains blocked keyword pattern: {matched}",
                triggered_rules=["blocklist"],
                confidence=1.0,
            )
        return None

    def _check_loop_detection(
        self, observation: Observation
    ) -> Optional[SafetyAssessment]:
        """Check if we're stuck in a loop visiting the same state."""
        state_hash = self._compute_state_hash(observation)
        self._state_visit_counts[state_hash] = (
            self._state_visit_counts.get(state_hash, 0) + 1
        )

        visit_count = self._state_visit_counts[state_hash]
        if visit_count >= self.config.loop_threshold:
            return SafetyAssessment(
                decision=SafetyDecision.BLOCK,
                reason=f"Loop detected: same state visited {visit_count} times "
                f"(threshold: {self.config.loop_threshold})",
                triggered_rules=["loop_detection"],
                confidence=1.0,
            )
        return None

    def _check_credential_guard(
        self, action: Action, observation: Observation
    ) -> Optional[SafetyAssessment]:
        """Check if TYPE action targets a credential/sensitive field."""
        if action.type != ActionType.TYPE:
            return None

        field_name = self._get_target_field_name(action, observation)
        if not field_name:
            return None

        # Check if field is in allowlist (override)
        if self._matches_any(field_name, self._credential_allowlist_re):
            return None

        # Check if field matches credential patterns
        matched = self._matches_any(field_name, self._credential_re)
        if matched:
            return SafetyAssessment(
                decision=SafetyDecision.REQUIRE_CONFIRMATION,
                reason=f"Typing into credential field matching '{matched}': {field_name}",
                triggered_rules=["credential_guard"],
                confidence=1.0,
            )
        return None

    def _check_irreversibility(self, action: Action) -> Optional[SafetyAssessment]:
        """Check if action appears to be irreversible (submit, send, etc.)."""
        action_text = self._get_action_text(action)
        matched = self._matches_any(action_text, self._irreversible_re)

        if matched:
            return SafetyAssessment(
                decision=SafetyDecision.REQUIRE_CONFIRMATION,
                reason=f"Action may be irreversible (matched pattern: {matched})",
                triggered_rules=["irreversibility"],
                confidence=0.9,
            )
        return None

    def _check_app_window_mismatch(
        self, observation: Observation
    ) -> Optional[SafetyAssessment]:
        """Check if current app/window doesn't match expected context."""
        triggered_rules = []
        reasons = []

        # Check app name
        if self.config.expected_app:
            current_app = observation.app_name or ""
            if current_app.lower() != self.config.expected_app.lower():
                triggered_rules.append("app_mismatch")
                reasons.append(
                    f"Expected app '{self.config.expected_app}', got '{current_app}'"
                )

        # Check window title pattern
        if self._window_pattern_re:
            current_title = observation.window_title or ""
            if not self._window_pattern_re.search(current_title):
                triggered_rules.append("window_mismatch")
                reasons.append(
                    f"Window title '{current_title}' doesn't match pattern "
                    f"'{self.config.expected_window_pattern}'"
                )

        if triggered_rules:
            return SafetyAssessment(
                decision=SafetyDecision.REQUIRE_CONFIRMATION,
                reason="; ".join(reasons),
                triggered_rules=triggered_rules,
                confidence=0.95,
            )
        return None

    def _check_confidence_threshold(self, action: Action) -> Optional[SafetyAssessment]:
        """Check if action confidence is below threshold."""
        confidence = self._get_action_confidence(action)

        if confidence is not None and confidence < self.config.confidence_threshold:
            return SafetyAssessment(
                decision=SafetyDecision.REQUIRE_CONFIRMATION,
                reason=f"Action confidence ({confidence:.2f}) below threshold "
                f"({self.config.confidence_threshold})",
                triggered_rules=["confidence_threshold"],
                confidence=1.0,
            )
        return None

    def assess(
        self,
        action: Action,
        observation: Observation,
        trace: Optional[list[Step]] = None,
    ) -> SafetyAssessment:
        """Evaluate an action against all safety rules.

        Checks are evaluated in priority order. First triggered rule wins.

        Args:
            action: The action proposed by the policy
            observation: Current state observation
            trace: Execution history (currently unused, reserved for future use)

        Returns:
            SafetyAssessment with decision and reasoning
        """
        # Priority 1: Blocklist (BLOCK)
        assessment = self._check_blocklist(action)
        if assessment:
            return assessment

        # Priority 2: Loop Detection (BLOCK)
        assessment = self._check_loop_detection(observation)
        if assessment:
            return assessment

        # Priority 3: Credential Guard (REQUIRE_CONFIRMATION)
        assessment = self._check_credential_guard(action, observation)
        if assessment:
            return assessment

        # Priority 4: Irreversibility (REQUIRE_CONFIRMATION)
        assessment = self._check_irreversibility(action)
        if assessment:
            return assessment

        # Priority 5: App/Window Mismatch (REQUIRE_CONFIRMATION)
        assessment = self._check_app_window_mismatch(observation)
        if assessment:
            return assessment

        # Priority 6: Confidence Threshold (REQUIRE_CONFIRMATION)
        assessment = self._check_confidence_threshold(action)
        if assessment:
            return assessment

        # Default: ALLOW
        return SafetyAssessment(
            decision=SafetyDecision.ALLOW,
            reason="All safety checks passed",
            triggered_rules=[],
            confidence=1.0,
        )

    def reset(self) -> None:
        """Clear internal state (loop detection history).

        Call this method between episodes to reset the state visit counts.
        """
        self._state_visit_counts.clear()

    def get_state_visit_counts(self) -> dict[str, int]:
        """Get current state visit counts (for debugging/monitoring)."""
        return self._state_visit_counts.copy()
