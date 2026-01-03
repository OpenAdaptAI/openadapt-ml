"""WAA demo-conditioned experiment module.

This module contains demonstrations and task definitions for the
Windows Agent Arena demo-conditioned prompting experiment.
"""

from openadapt_ml.experiments.waa_demo.demos import DEMOS, get_demo
from openadapt_ml.experiments.waa_demo.tasks import TASKS, get_task

__all__ = ["DEMOS", "TASKS", "get_demo", "get_task"]
