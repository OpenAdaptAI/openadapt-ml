"""Refuse Modal GPU work on the production workspace.

The production Modal workspace hosts OpenAdapt's managed runner, compiler and
recorder. On 2026-10-16 GPU training used up that workspace's monthly spend
limit, and Modal stopped the production services with it. GPU training now
goes to Kaggle (``openadapt_ml.cloud.kaggle_cloud``).

Modal GPU work runs only when the caller opts in on purpose:

* ``OPENADAPT_ALLOW_MODAL_GPU=1``, and
* ``MODAL_ENVIRONMENT`` names a non-production environment (not ``main``,
  ``prod`` or ``production``), and
* the active Modal profile/workspace is not listed in
  ``OPENADAPT_MODAL_PROD_WORKSPACES`` (default: ``abrichr``, the workspace that
  serves app.openadapt.ai).

Modal Starter budgets are workspace-wide, so a separate environment in the
production workspace still spends the production budget. Use a separate
workspace (``modal profile activate <other>``) for GPU work.
"""

from __future__ import annotations

import os
import subprocess

OPT_IN_ENV = "OPENADAPT_ALLOW_MODAL_GPU"
PROD_WORKSPACES_ENV = "OPENADAPT_MODAL_PROD_WORKSPACES"
DEFAULT_PROD_WORKSPACES = ("abrichr",)
PROD_ENVIRONMENTS = frozenset({"", "main", "prod", "production"})


class ModalGpuRefused(RuntimeError):
    """Raised when Modal GPU work would run on the production workspace."""


def _active_profile() -> str:
    explicit = os.environ.get("MODAL_PROFILE", "").strip()
    if explicit:
        return explicit
    try:
        out = subprocess.run(
            ["modal", "profile", "current"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def modal_gpu_refusal(
    environ: dict[str, str] | None = None, profile: str | None = None
) -> str | None:
    """Return why Modal GPU work is refused, or None when it is allowed."""
    env = os.environ if environ is None else environ
    if env.get(OPT_IN_ENV) != "1":
        return (
            f"Modal GPU work is off by default. Train on Kaggle "
            f"(python -m openadapt_ml.cloud.kaggle_cloud), or set {OPT_IN_ENV}=1 "
            "with a non-production Modal workspace and environment."
        )
    environment = env.get("MODAL_ENVIRONMENT", "").strip().lower()
    if environment in PROD_ENVIRONMENTS:
        return (
            "MODAL_ENVIRONMENT must name a non-production Modal environment "
            f"(got {environment or 'unset'!r}; main/prod/production are refused)."
        )
    prod = {
        w.strip().lower()
        for w in env.get(PROD_WORKSPACES_ENV, ",".join(DEFAULT_PROD_WORKSPACES)).split(
            ","
        )
        if w.strip()
    }
    active = (profile if profile is not None else _active_profile()).strip().lower()
    if not active:
        return "Could not determine the active Modal profile; refusing GPU work."
    if active in prod:
        return (
            f"Active Modal profile {active!r} is the production workspace. "
            "Activate a separate workspace (modal profile activate <name>) for GPU work."
        )
    return None


def require_nonprod_modal_gpu() -> None:
    """Raise ModalGpuRefused unless Modal GPU work is explicitly allowed."""
    reason = modal_gpu_refusal()
    if reason:
        raise ModalGpuRefused(reason)
