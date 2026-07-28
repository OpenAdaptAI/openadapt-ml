"""Demo-conditioned prompt experiment.

Tests whether including a human demonstration in the prompt
improves VLM agent performance on similar tasks.
"""

from openadapt_ml.experiments.demo_prompt.format_demo import (
    format_action,
    format_episode_as_demo,
)
from openadapt_ml.experiments.demo_prompt.run_experiment import (
    DemoPromptExperiment,
)

__all__ = [
    "format_episode_as_demo",
    "format_action",
    "DemoPromptExperiment",
]
