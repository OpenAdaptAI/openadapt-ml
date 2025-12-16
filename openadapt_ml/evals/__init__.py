"""Evaluation modules for openadapt-ml.

This package provides evaluation metrics and utilities for measuring
model performance on GUI automation tasks.

Modules:
    - grounding: Grounding-specific metrics (IoU, hit rate, latency)
    - trajectory_matching: Trajectory comparison metrics (existing)
"""

from openadapt_ml.evals.grounding import (
    GroundingMetrics,
    GroundingResult,
    evaluate_grounder,
    evaluate_grounder_on_episode,
)

__all__ = [
    "GroundingMetrics",
    "GroundingResult",
    "evaluate_grounder",
    "evaluate_grounder_on_episode",
]
