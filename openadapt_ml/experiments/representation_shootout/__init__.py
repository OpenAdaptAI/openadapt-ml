"""Representation Shootout Experiment.

Compares three approaches for GUI action prediction under distribution drift:

- Condition A: Raw Coordinates - Direct coordinate regression
- Condition B: Coordinates + Visual Cues - Enhanced with markers and zoom
- Condition C: Marks (Element IDs) - Element classification using SoM

Usage:
    # Run full experiment
    python -m openadapt_ml.experiments.representation_shootout.runner run

    # Run specific condition
    python -m openadapt_ml.experiments.representation_shootout.runner run --condition marks

    # Evaluate under specific drift
    python -m openadapt_ml.experiments.representation_shootout.runner eval --drift resolution

See docs/experiments/representation_shootout_design.md for full documentation.
"""

from openadapt_ml.experiments.representation_shootout.config import (
    ConditionConfig,
    ConditionName,
    DriftConfig,
    DriftType,
    ExperimentConfig,
    MetricName,
)
from openadapt_ml.experiments.representation_shootout.conditions import (
    ConditionBase,
    CoordsCuesCondition,
    MarksCondition,
    RawCoordsCondition,
    create_condition,
)
from openadapt_ml.experiments.representation_shootout.evaluator import (
    DriftEvaluator,
    EvaluationResult,
    compute_metrics,
    make_recommendation,
)
from openadapt_ml.experiments.representation_shootout.runner import (
    ExperimentRunner,
    run_experiment,
)

__all__ = [
    # Config
    "ExperimentConfig",
    "ConditionConfig",
    "ConditionName",
    "DriftConfig",
    "DriftType",
    "MetricName",
    # Conditions
    "ConditionBase",
    "RawCoordsCondition",
    "CoordsCuesCondition",
    "MarksCondition",
    "create_condition",
    # Evaluator
    "DriftEvaluator",
    "EvaluationResult",
    "compute_metrics",
    "make_recommendation",
    # Runner
    "ExperimentRunner",
    "run_experiment",
]
