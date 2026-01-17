"""
Runtime module for GUI automation agents.

This module provides:
- AgentPolicy: Runtime policy wrapper for VLM-based action prediction
- SafetyGate: Deterministic safety checks for action validation

Example usage:
    from openadapt_ml.runtime import AgentPolicy, SafetyGate, SafetyConfig, SafetyDecision

    # Create policy and safety gate
    policy = AgentPolicy(adapter)
    gate = SafetyGate(SafetyConfig(confidence_threshold=0.8))

    # In agent loop
    action, thought, state, raw = policy.predict_action_from_sample(sample)
    assessment = gate.assess(action, observation)

    if assessment.decision == SafetyDecision.ALLOW:
        execute(action)
    elif assessment.decision == SafetyDecision.REQUIRE_CONFIRMATION:
        if user_confirms():
            execute(action)
    else:  # BLOCK
        log_blocked(assessment.reason)
"""

from openadapt_ml.runtime.policy import (
    AgentPolicy,
    PolicyOutput,
    parse_thought_state_action,
)
from openadapt_ml.runtime.safety_gate import (
    SafetyAssessment,
    SafetyConfig,
    SafetyDecision,
    SafetyGate,
)

__all__ = [
    # Policy
    "AgentPolicy",
    "PolicyOutput",
    "parse_thought_state_action",
    # Safety Gate
    "SafetyGate",
    "SafetyConfig",
    "SafetyDecision",
    "SafetyAssessment",
]
