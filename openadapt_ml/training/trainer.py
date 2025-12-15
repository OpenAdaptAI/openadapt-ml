from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from openadapt_ml.models.base_adapter import BaseVLMAdapter
from openadapt_ml.schemas.sessions import Episode
from openadapt_ml.training.shared_ui import (
    get_shared_header_css as _get_shared_header_css,
    generate_shared_header_html as _generate_shared_header_html,
    build_nav_links as _build_nav_links,
)


def setup_job_directory(base_dir: str | Path, job_id: str) -> Path:
    """Set up job-scoped directory structure with symlink.

    Creates:
        {base_dir}/{job_id}/     - Job-specific directory
        {base_dir}/current       - Symlink to current job directory

    Args:
        base_dir: Base output directory (e.g., "training_output")
        job_id: Unique job identifier (e.g., "20251214_200417")

    Returns:
        Path to the job-specific directory
    """
    base_dir = Path(base_dir)
    job_dir = base_dir / job_id
    current_link = base_dir / "current"

    # Create base and job directories
    base_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Atomically update the 'current' symlink
    # Use a temp link then rename for atomic operation
    temp_link = base_dir / f".current_temp_{job_id}"
    try:
        # Remove temp link if it exists from a previous failed attempt
        if temp_link.exists() or temp_link.is_symlink():
            temp_link.unlink()

        # Create temp symlink pointing to job_id (relative path)
        temp_link.symlink_to(job_id)

        # Atomically replace current with temp
        temp_link.rename(current_link)
    except Exception as e:
        # Clean up temp link on failure
        if temp_link.exists() or temp_link.is_symlink():
            temp_link.unlink()
        raise RuntimeError(f"Failed to create current symlink: {e}")

    return job_dir


def get_current_job_directory(base_dir: str | Path) -> Path | None:
    """Get the current job directory from symlink.

    Returns:
        Path to current job directory, or None if no current symlink
    """
    base_dir = Path(base_dir)
    current_link = base_dir / "current"

    if current_link.is_symlink():
        return current_link.resolve()
    return None


@dataclass
class TrainingConfig:
    # Model / LoRA-related fields are handled elsewhere; this covers loop hyperparams.
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    # Early stopping: stop when loss is below threshold for patience consecutive steps
    early_stop_loss: float = 1e-4
    early_stop_patience: int = 10
    # Output directory for logs and visualizations
    output_dir: str = "training_output"
    # Checkpoint saving
    save_checkpoint_every_epoch: bool = True
    checkpoint_dir: str = "checkpoints"
    # Evaluation during training
    eval_every_epoch: bool = True
    eval_samples: int = 3  # Number of samples to evaluate per epoch


@dataclass
class TrainingState:
    """Tracks training progress for visualization."""
    # Job identification
    job_id: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))
    hostname: str = field(default_factory=lambda: __import__('socket').gethostname())
    capture_path: str = ""
    config_path: str = ""
    # Training progress
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    total_epochs: int = 1  # Set by logger from config
    loss: float = 0.0
    learning_rate: float = 0.0
    samples_seen: int = 0
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0  # For historical data loaded from JSON
    losses: List[Dict[str, Any]] = field(default_factory=list)
    evaluations: List[Dict[str, Any]] = field(default_factory=list)
    # Cloud info (optional)
    instance_type: str = ""
    instance_ip: str = ""
    # Cloud provider info (for dashboard link)
    cloud_provider: str = ""  # e.g. "lambda", "azure"
    cloud_dashboard_url: str = ""  # e.g. "https://cloud.lambda.ai/instances"
    cloud_instance_id: str = ""  # Provider-specific instance ID
    # Setup status tracking
    setup_status: str = ""  # e.g. "booting", "installing", "training", "complete"
    setup_logs: List[str] = field(default_factory=list)  # Setup progress messages
    # Termination tracking
    termination_status: str = ""  # e.g. "auto_low_loss", "auto_complete", "user_stop", "running"
    termination_message: str = ""  # Human-readable termination reason

    def log_step(self, epoch: int, step: int, loss: float, lr: float = 0.0) -> None:
        """Log a training step."""
        self.epoch = epoch
        self.step = step
        self.loss = loss
        self.learning_rate = lr
        self.losses.append({
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "lr": lr,
            "time": time.time() - self.start_time,
        })

    def log_evaluation(self, epoch: int, sample_idx: int, image_path: str,
                       human_action: Dict, predicted_action: Dict) -> None:
        """Log an evaluation sample."""
        # Calculate distance for click actions
        distance = 0.0
        if human_action.get("type") == "click" and predicted_action.get("type") == "click":
            hx, hy = human_action.get("x", 0), human_action.get("y", 0)
            px, py = predicted_action.get("x", 0), predicted_action.get("y", 0)
            distance = ((hx - px) ** 2 + (hy - py) ** 2) ** 0.5

        self.evaluations.append({
            "epoch": epoch,
            "sample_idx": sample_idx,
            "image_path": image_path,
            "human_action": human_action,
            "predicted_action": predicted_action,
            "distance": distance,
            "correct": distance < 50,  # Within 50 pixels is "correct"
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to serializable dict."""
        return {
            # Job metadata
            "job_id": self.job_id,
            "hostname": self.hostname,
            "capture_path": self.capture_path,
            "config_path": self.config_path,
            "instance_type": self.instance_type,
            "instance_ip": self.instance_ip,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.start_time)),
            # Cloud provider info
            "cloud_provider": self.cloud_provider,
            "cloud_dashboard_url": self.cloud_dashboard_url,
            "cloud_instance_id": self.cloud_instance_id,
            "setup_status": self.setup_status,
            "setup_logs": self.setup_logs,
            # Training progress
            "epoch": self.epoch,
            "step": self.step,
            "total_steps": self.total_steps,
            "total_epochs": self.total_epochs,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "samples_seen": self.samples_seen,
            "elapsed_time": time.time() - self.start_time,
            "losses": self.losses,
            "evaluations": self.evaluations,
            # Termination tracking
            "termination_status": self.termination_status,
            "termination_message": self.termination_message,
        }


class TrainingLogger:
    """Logs training progress and generates visualization."""

    def __init__(
        self,
        output_dir: str | Path,
        config: TrainingConfig,
        capture_path: str = "",
        config_path: str = "",
        instance_ip: str = "",
        instance_type: str = "",
        cloud_provider: str = "",
        cloud_dashboard_url: str = "",
        cloud_instance_id: str = "",
        job_id: str = "",
    ):
        # Generate job_id if not provided
        if not job_id:
            job_id = time.strftime("%Y%m%d_%H%M%S")

        # Set up job-scoped directory with symlink
        base_dir = Path(output_dir)
        self.base_dir = base_dir
        self.output_dir = setup_job_directory(base_dir, job_id)
        self.config = config
        self.state = TrainingState(
            job_id=job_id,
            capture_path=capture_path,
            config_path=config_path,
            instance_ip=instance_ip,
            instance_type=instance_type,
            total_epochs=config.num_train_epochs,
            cloud_provider=cloud_provider,
            cloud_dashboard_url=cloud_dashboard_url,
            cloud_instance_id=cloud_instance_id,
        )
        self.log_file = self.output_dir / "training_log.json"
        self.terminal_log_file = self.output_dir / "training.log"
        self.terminal_log_handle = None

    def _log_to_terminal(self, message: str):
        """Write message to training.log file.

        Args:
            message: Message to log
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"

        # Open file on first write (line buffered)
        if self.terminal_log_handle is None:
            self.terminal_log_handle = open(self.terminal_log_file, "w", buffering=1)

        self.terminal_log_handle.write(log_line + "\n")
        self.terminal_log_handle.flush()

    def on_step(self, epoch: int, step: int, loss: float, lr: float = 0.0) -> None:
        """Called after each training step."""
        self.state.log_step(epoch, step, loss, lr)
        self._save_log()

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch."""
        self.state.epoch = epoch
        self._save_log()
        self._generate_dashboard()

    def on_train_end(self) -> None:
        """Called at the end of training."""
        self._save_log()
        self._generate_dashboard()
        print(f"Training dashboard: {self.output_dir / 'dashboard.html'}")

        # Close terminal log file
        if self.terminal_log_handle:
            self.terminal_log_handle.close()
            self.terminal_log_handle = None

    def _save_log(self) -> None:
        """Save training log to JSON."""
        with open(self.log_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def _generate_dashboard(self) -> None:
        """Generate HTML training dashboard."""
        dashboard_path = self.output_dir / "dashboard.html"
        html = generate_training_dashboard(self.state, self.config)
        dashboard_path.write_text(html)


def _generate_termination_status_html(state: TrainingState, is_training_complete: bool) -> str:
    """Generate HTML for termination status section."""
    # Check if we have termination info
    if state.termination_status:
        # Map termination status to colors and icons
        status_styles = {
            "auto_complete": {"color": "#22c55e", "icon": "✓", "label": "Training Complete"},
            "auto_low_loss": {"color": "#22c55e", "icon": "✓", "label": "Auto-Stopped (Low Loss)"},
            "user_stop": {"color": "#f59e0b", "icon": "■", "label": "Stopped by User"},
        }
        style = status_styles.get(state.termination_status, {"color": "#22c55e", "icon": "✓", "label": "Complete"})

        return f'''<div style="display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px; color: {style['color']};">
                <span style="font-size: 1.2rem;">{style['icon']}</span>
                <span style="font-weight: 600;">{style['label']}</span>
            </div>
            {f'<div style="font-size: 0.85rem; color: var(--text-muted); margin-left: 28px;">{state.termination_message}</div>' if state.termination_message else ''}
        </div>'''
    elif is_training_complete:
        return '''<div style="display: flex; align-items: center; gap: 8px; color: #22c55e;">
            <span style="font-size: 1.2rem;">✓</span>
            <span style="font-weight: 600;">Training Complete</span>
        </div>'''
    else:
        return '''<button id="stop-training-btn" onclick="stopTraining()" style="
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
        ">
            <span style="font-size: 1.1rem;">■</span> Stop Training
        </button>
        <p id="stop-status" style="margin-top: 8px; font-size: 0.75rem; color: var(--text-muted);"></p>'''


def generate_training_dashboard(state: TrainingState, config: TrainingConfig) -> str:
    """Generate an HTML dashboard for training visualization."""
    losses_json = json.dumps(state.losses)
    # Use stored elapsed_time if available (historical data), otherwise calculate
    elapsed = state.elapsed_time if state.elapsed_time > 0 else time.time() - state.start_time
    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    # Calculate stats
    if state.losses:
        min_loss = min(l["loss"] for l in state.losses)
        avg_loss = sum(l["loss"] for l in state.losses) / len(state.losses)
        recent_losses = state.losses[-10:] if len(state.losses) >= 10 else state.losses
        recent_avg = sum(l["loss"] for l in recent_losses) / len(recent_losses)
        # Calculate step times
        step_times = []
        for i in range(1, len(state.losses)):
            step_times.append(state.losses[i]["time"] - state.losses[i-1]["time"])
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        # Loss by epoch
        epoch_losses: dict = {}
        for l in state.losses:
            ep = l["epoch"]
            if ep not in epoch_losses:
                epoch_losses[ep] = []
            epoch_losses[ep].append(l["loss"])
        epoch_avg = {ep: sum(losses)/len(losses) for ep, losses in epoch_losses.items()}
        # Estimate ETA
        # Steps per epoch = steps in completed epochs / completed epochs
        completed_epochs = state.epoch
        steps_in_completed = sum(1 for l in state.losses if l["epoch"] < completed_epochs)
        if completed_epochs > 0 and steps_in_completed > 0:
            steps_per_epoch = steps_in_completed / completed_epochs
        else:
            # Estimate from current epoch progress
            steps_per_epoch = len(state.losses) / (state.epoch + 1) if state.epoch >= 0 else len(state.losses)

        total_epochs = state.total_epochs if state.total_epochs > 0 else config.num_train_epochs
        total_steps_estimate = steps_per_epoch * total_epochs
        remaining_steps = max(0, total_steps_estimate - len(state.losses))
        eta_seconds = remaining_steps * avg_step_time if avg_step_time > 0 else 0
        # Check if training is complete (all steps done)
        is_training_complete = remaining_steps == 0 and len(state.losses) > 0
    else:
        min_loss = avg_loss = recent_avg = avg_step_time = 0.0
        epoch_avg = {}
        eta_seconds = 0
        steps_per_epoch = 0
        total_steps_estimate = 0
        remaining_steps = 0
        is_training_complete = False

    epoch_avg_json = json.dumps(list(epoch_avg.items()))

    # Generate comparison viewer preview if capture path available
    comparison_viewer_path = ""
    if state.capture_path:
        try:
            from openadapt_ml.scripts.compare import generate_comparison_html, generate_comparison_data
            from openadapt_ml.ingest.capture import capture_to_episode

            capture_path = Path(state.capture_path)
            if capture_path.exists():
                # Load episode from capture
                episode = capture_to_episode(capture_path)

                # Generate comparison data with null predictions (shows "— No prediction")
                comparison_data = []
                for i, step in enumerate(episode.steps):
                    step_data = {
                        "index": i,
                        "time": step.t,
                        "image_path": step.observation.image_path,
                        "human_action": {
                            "type": step.action.type,
                            "x": step.action.x,
                            "y": step.action.y,
                            "text": step.action.text,
                        },
                        "predicted_action": None,  # Shows "— No prediction" in viewer
                        "match": None,
                    }
                    comparison_data.append(step_data)

                # Generate comparison HTML
                output_dir = Path(config.output_dir) if hasattr(config, 'output_dir') else Path("training_output")
                output_dir.mkdir(parents=True, exist_ok=True)
                comparison_output = output_dir / "comparison_preview.html"
                generate_comparison_html(capture_path, episode, comparison_data, comparison_output)
                comparison_viewer_path = str(comparison_output.name)  # Relative path
        except Exception as e:
            pass  # Fail silently if comparison viewer can't be generated

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a24;
            --border-color: rgba(255, 255, 255, 0.06);
            --text-primary: #f0f0f0;
            --text-secondary: #888;
            --accent: #00d4aa;
            --accent-secondary: #a78bfa;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 24px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 24px;
        }}
        header h1 {{ font-size: 1.3rem; font-weight: 600; }}
        .job-info {{
            display: flex;
            gap: 16px;
            margin-top: 4px;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        .job-id {{
            font-family: "SF Mono", Monaco, monospace;
            color: var(--accent);
        }}
        .job-host {{
            font-family: "SF Mono", Monaco, monospace;
        }}
        .job-config {{
            font-family: "SF Mono", Monaco, monospace;
            opacity: 0.7;
        }}
        .cloud-link {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 0.75rem;
            color: var(--text-primary);
            text-decoration: none;
            transition: all 0.2s;
        }}
        .cloud-link:hover {{
            border-color: var(--accent);
            background: rgba(0, 212, 170, 0.1);
        }}
        .cloud-link svg {{
            width: 14px;
            height: 14px;
        }}
        .cloud-badge {{
            background: linear-gradient(135deg, rgba(167, 139, 250, 0.2), rgba(0, 212, 170, 0.1));
            border-color: rgba(167, 139, 250, 0.3);
            margin-left: 12px;
        }}
        .setup-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .setup-panel.hidden {{
            display: none;
        }}
        .setup-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .setup-header h2 {{
            font-size: 0.9rem;
        }}
        .setup-status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}
        .setup-status-badge.booting {{
            background: rgba(255, 149, 0, 0.2);
            color: #ff9500;
        }}
        .setup-status-badge.installing {{
            background: rgba(167, 139, 250, 0.2);
            color: #a78bfa;
        }}
        .setup-status-badge.training {{
            background: rgba(0, 212, 170, 0.2);
            color: #00d4aa;
        }}
        .setup-status-badge.complete {{
            background: rgba(52, 211, 153, 0.2);
            color: #34d399;
        }}
        .setup-logs {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
            max-height: 200px;
            overflow-y: auto;
            font-family: "SF Mono", Monaco, monospace;
            font-size: 0.7rem;
            line-height: 1.6;
        }}
        .setup-log-line {{
            color: var(--text-secondary);
            padding: 2px 0;
        }}
        .setup-log-line.current {{
            color: var(--accent);
        }}
        .status {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--accent);
        }}
        .status-dot {{
            width: 10px;
            height: 10px;
            background: var(--accent);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        .status.complete .status-dot {{
            animation: none;
            background: #34d399;
        }}
        .status.stale {{
            color: #ff9500;
        }}
        .status.stale .status-dot {{
            animation: none;
            background: #ff9500;
        }}
        .stale-warning {{
            font-size: 0.7rem;
            color: #ff9500;
            margin-top: 2px;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
        }}
        .stat-card.updating {{
            border-color: var(--accent);
            box-shadow: 0 0 20px rgba(0, 212, 170, 0.1);
        }}
        .stat-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        .stat-detail {{
            font-size: 0.65rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        .eta-card {{
            background: linear-gradient(135deg, rgba(167, 139, 250, 0.1), rgba(0, 212, 170, 0.05));
            border-color: rgba(167, 139, 250, 0.3);
        }}
        .stat-value {{
            font-size: 1.6rem;
            font-weight: 600;
            font-family: "SF Mono", Monaco, monospace;
            transition: all 0.3s ease;
        }}
        .stat-value.accent {{ color: var(--accent); }}
        .stat-delta {{
            font-size: 0.75rem;
            margin-top: 4px;
            font-family: "SF Mono", Monaco, monospace;
        }}
        .stat-delta.positive {{ color: #34d399; }}
        .stat-delta.negative {{ color: #ff5f5f; }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }}
        @media (max-width: 900px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
        .chart-container {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
        }}
        .chart-title {{
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .chart-subtitle {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-weight: normal;
        }}
        .chart-wrapper {{
            height: 300px;
            position: relative;
        }}
        .config-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }}
        .config-panel h2 {{
            font-size: 0.9rem;
            margin-bottom: 16px;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
        }}
        .config-item {{
            font-size: 0.8rem;
        }}
        .config-item .key {{
            color: var(--text-secondary);
        }}
        .config-item .value {{
            font-family: "SF Mono", Monaco, monospace;
            color: var(--accent);
        }}
        .progress-bar {{
            height: 4px;
            background: var(--bg-tertiary);
            border-radius: 2px;
            margin-top: 8px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--accent-secondary));
            border-radius: 2px;
            transition: width 0.5s ease;
        }}
        .update-indicator {{
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-align: right;
            margin-top: 16px;
        }}
        /* Shared header styles (injected from _get_shared_header_css) */
        {_get_shared_header_css()}
        .eval-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-top: 16px;
        }}
        .eval-panel h2 {{
            font-size: 0.9rem;
            margin-bottom: 16px;
        }}
        .eval-metrics {{
            display: flex;
            gap: 24px;
            margin-bottom: 16px;
            font-size: 0.85rem;
        }}
        .eval-metrics .metric {{
            display: flex;
            flex-direction: column;
        }}
        .eval-metrics .metric-value {{
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--accent);
        }}
        .eval-filters {{
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .eval-filters .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .eval-filters label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .eval-filters select {{
            padding: 8px 32px 8px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            background: rgba(0,0,0,0.4);
            color: var(--text-primary);
            border: 1px solid rgba(255,255,255,0.1);
            cursor: pointer;
            appearance: none;
            background-image: url('data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%2712%27 height=%278%27%3E%3Cpath fill=%27%23888%27 d=%27M0 0l6 8 6-8z%27/%3E%3C/svg%3E');
            background-repeat: no-repeat;
            background-position: right 10px center;
            transition: all 0.2s;
        }}
        .eval-filters select:hover {{
            border-color: var(--accent);
            background-color: rgba(0,212,170,0.1);
        }}
        .eval-filters select:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(0,212,170,0.2);
        }}
        .eval-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}
        .eval-sample {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 0;
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        .eval-sample.hidden {{
            display: none;
        }}
        .eval-sample .image-container {{
            position: relative;
            background: #000;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .eval-sample img {{
            width: 100%;
            height: auto;
            display: block;
            max-height: 400px;
            object-fit: contain;
        }}
        .eval-sample .overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
        }}
        .eval-sample .marker {{
            position: absolute;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            border: 3px solid white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: bold;
            color: white;
            z-index: 10;
        }}
        .eval-sample .marker.human {{
            background: rgba(0, 212, 170, 0.4);
            border-color: #00d4aa;
        }}
        .eval-sample .marker.human::after {{
            content: 'H';
            color: #00d4aa;
        }}
        .eval-sample .marker.predicted {{
            background: rgba(167, 139, 250, 0.4);
            border-color: #a78bfa;
        }}
        .eval-sample .marker.predicted::after {{
            content: 'AI';
            font-size: 9px;
            color: #a78bfa;
        }}
        .eval-sample .line {{
            position: absolute;
            height: 2px;
            background: rgba(255, 255, 255, 0.5);
            transform-origin: left center;
        }}
        .eval-sample .content {{
            padding: 12px;
        }}
        .eval-sample .info {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-color);
        }}
        .eval-sample .info .correct {{
            color: #34d399;
            font-weight: 600;
        }}
        .eval-sample .info .incorrect {{
            color: #ff5f5f;
            font-weight: 600;
        }}
        .eval-sample .details {{
            font-size: 0.7rem;
            color: var(--text-secondary);
        }}
        .eval-sample .coords {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-bottom: 8px;
        }}
        .eval-sample .coords .human-coord {{
            color: #34d399;
        }}
        .eval-sample .coords .pred-coord {{
            color: #a78bfa;
        }}
        .eval-sample .thinking {{
            margin-top: 8px;
            padding: 8px;
            background: rgba(0,0,0,0.3);
            border-radius: 4px;
            font-size: 0.65rem;
            color: var(--text-secondary);
            max-height: 150px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
            font-family: "SF Mono", Monaco, monospace;
            line-height: 1.4;
        }}
        .eval-sample .thinking.collapsed {{
            max-height: 60px;
            overflow: hidden;
            position: relative;
        }}
        .eval-sample .thinking.collapsed::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 30px;
            background: linear-gradient(to bottom, transparent, rgba(0,0,0,0.5));
        }}
        .eval-sample .thinking-toggle {{
            cursor: pointer;
            color: var(--accent);
            font-size: 0.7rem;
            margin-top: 4px;
            display: inline-block;
        }}
        .eval-sample .thinking-toggle:hover {{
            text-decoration: underline;
        }}
        .terminal-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-top: 16px;
        }}
        .terminal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .terminal-header h2 {{
            font-size: 0.9rem;
            margin: 0;
        }}
        .terminal-toggle {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .terminal-toggle:hover {{
            border-color: var(--accent);
            background: rgba(0, 212, 170, 0.1);
        }}
        .terminal-container {{
            background: #000;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 16px;
            max-height: 400px;
            overflow-y: auto;
            font-family: "SF Mono", Monaco, "Courier New", monospace;
            font-size: 0.7rem;
            line-height: 1.5;
            color: #0f0;
            position: relative;
        }}
        .terminal-container.collapsed {{
            max-height: 200px;
        }}
        .terminal-output {{
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .terminal-line {{
            padding: 2px 0;
        }}
        .terminal-line.timestamp {{
            color: #888;
        }}
        .terminal-line.error {{
            color: #ff5f5f;
        }}
        .terminal-line.success {{
            color: #34d399;
        }}
        .terminal-line.warning {{
            color: #ff9500;
        }}
        .terminal-empty {{
            color: #888;
            font-style: italic;
            text-align: center;
            padding: 40px;
        }}
        .terminal-controls {{
            display: flex;
            gap: 8px;
            margin-bottom: 8px;
            font-size: 0.7rem;
        }}
        .terminal-control-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .terminal-control-btn:hover {{
            border-color: var(--accent);
            color: var(--text-primary);
        }}
        .terminal-control-btn.active {{
            border-color: var(--accent);
            color: var(--accent);
        }}
    </style>
</head>
<body>
    {_generate_shared_header_html("training", meta_html=f"Job: {state.job_id}")}

    <div class="container">
        <header>
            <div>
                <h1>Training Dashboard{f' <a href="{state.cloud_dashboard_url}" target="_blank" class="cloud-link cloud-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>{state.cloud_provider.title()} Cloud</a>' if state.cloud_dashboard_url else ''}</h1>
                <div class="job-info" id="job-info">
                    <span class="job-host">{state.hostname or 'stub-local'} @ {state.instance_ip or '127.0.0.1'}</span>
                    {f'<span class="job-config">{state.instance_type}</span>' if state.instance_type else ''}
                </div>
            </div>
            <div class="status" id="status">
                <div class="status-dot"></div>
                <span id="status-text">Training in progress</span>
            </div>
        </header>

        <div class="setup-panel{' hidden' if not state.setup_logs else ''}" id="setup-panel">
            <div class="setup-header">
                <h2>Setup Progress</h2>
                <span class="setup-status-badge {state.setup_status}" id="setup-status-badge">{state.setup_status or 'initializing'}</span>
            </div>
            <div class="setup-logs" id="setup-logs">
                {''.join(f'<div class="setup-log-line{" current" if i == len(state.setup_logs) - 1 else ""}">{log}</div>' for i, log in enumerate(state.setup_logs)) if state.setup_logs else '<div class="setup-log-line">Waiting for setup logs...</div>'}
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card" id="card-epoch">
                <div class="stat-label">Epoch Progress</div>
                <div class="stat-value" id="stat-epoch">{min(state.epoch + 1, config.num_train_epochs)} / {config.num_train_epochs}</div>
                <div class="progress-bar"><div class="progress-fill" id="epoch-progress" style="width: {(min(state.epoch + 1, config.num_train_epochs) / config.num_train_epochs) * 100}%"></div></div>
            </div>
            <div class="stat-card" id="card-step">
                <div class="stat-label">Steps</div>
                <div class="stat-value" id="stat-step">{state.step}</div>
            </div>
            <div class="stat-card" id="card-loss">
                <div class="stat-label">Current Loss</div>
                <div class="stat-value accent" id="stat-loss">{state.loss:.4f}</div>
                <div class="stat-delta" id="loss-delta"></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Min Loss</div>
                <div class="stat-value" id="stat-min-loss">{min_loss:.4f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg (last 10)</div>
                <div class="stat-value" id="stat-avg-loss">{recent_avg:.4f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Step Time</div>
                <div class="stat-value" id="stat-step-time">{avg_step_time:.1f}s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Elapsed</div>
                <div class="stat-value" id="stat-elapsed">{elapsed_str}</div>
            </div>
            <div class="stat-card eta-card">
                <div class="stat-label">ETA</div>
                <div class="stat-value" id="stat-eta">{f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if eta_seconds > 0 else ("Complete" if is_training_complete else "calculating...")}</div>
                <div class="stat-detail" id="eta-detail">{f"~{int(remaining_steps)} steps @ {avg_step_time:.1f}s/step" if remaining_steps > 0 else ""}</div>
            </div>
            <div class="stat-card" id="card-cost" style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05)); border-color: rgba(239, 68, 68, 0.3);">
                <div class="stat-label">Cloud Cost</div>
                <div class="stat-value" id="stat-running-cost" style="color: #ef4444;">$0.00</div>
                <div class="stat-detail" id="stat-est-total">Est. Total: $0.00</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">
                    Loss Curve
                    <span class="chart-subtitle" id="loss-trend"></span>
                </div>
                <div class="chart-wrapper">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Loss by Epoch</div>
                <div class="chart-wrapper">
                    <canvas id="epochChart"></canvas>
                </div>
            </div>
        </div>

        <div class="config-panel">
            <h2>Training Configuration</h2>
            <div class="config-grid">
                <div class="config-item"><span class="key">Epochs:</span> <span class="value">{config.num_train_epochs}</span></div>
                <div class="config-item"><span class="key">Batch size:</span> <span class="value">{config.per_device_train_batch_size}</span></div>
                <div class="config-item"><span class="key">Learning rate:</span> <span class="value">{config.learning_rate}</span></div>
                <div class="config-item"><span class="key">Grad accum:</span> <span class="value">{config.gradient_accumulation_steps}</span></div>
                <div class="config-item"><span class="key">Max grad norm:</span> <span class="value">{config.max_grad_norm}</span></div>
                <div class="config-item"><span class="key">Early stop:</span> <span class="value">{config.early_stop_loss}</span></div>
            </div>
            <div id="stop-training-section" class="stop-training-section" style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                {_generate_termination_status_html(state, is_training_complete)}
            </div>
        </div>

        <div class="eval-panel" id="eval-panel" style="display: none;">
            <h2>Evaluation Samples</h2>
            <div class="eval-metrics" id="eval-metrics"></div>
            <div class="eval-filters">
                <div class="filter-group">
                    <label for="epoch-filter">Epoch:</label>
                    <select id="epoch-filter">
                        <option value="all">All Epochs</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="correctness-filter">Status:</label>
                    <select id="correctness-filter">
                        <option value="all">All</option>
                        <option value="correct">Correct Only</option>
                        <option value="incorrect">Incorrect Only</option>
                    </select>
                </div>
                <div style="margin-left: auto; font-size: 0.75rem; color: var(--text-muted);">
                    <span id="filter-count"></span>
                </div>
            </div>
            <div class="eval-gallery" id="eval-gallery"></div>
        </div>

        <div class="terminal-panel" id="terminal-panel">
            <div class="terminal-header">
                <h2>Training Output</h2>
                <button class="terminal-toggle" id="terminal-toggle" onclick="toggleTerminal()">
                    <span id="terminal-toggle-text">Collapse</span>
                </button>
            </div>
            <div class="terminal-controls">
                <button class="terminal-control-btn active" id="auto-scroll-btn" onclick="toggleAutoScroll()">Auto-scroll</button>
                <button class="terminal-control-btn" id="wrap-btn" onclick="toggleWrap()">Wrap text</button>
                <span style="margin-left: auto; color: var(--text-secondary); font-size: 0.7rem;">
                    <span id="terminal-line-count">0</span> lines
                </span>
            </div>
            <div class="terminal-container" id="terminal-container">
                <div class="terminal-output" id="terminal-output">
                    <div class="terminal-empty">Waiting for training output...</div>
                </div>
            </div>
        </div>

        <div class="update-indicator" id="update-indicator">Last updated: just now</div>
    </div>

    <script>
        let losses = {losses_json};
        let epochAvg = {epoch_avg_json};
        let lossChart, epochChart;
        let lastStep = {state.step};
        let lastLoss = {state.loss};

        // Cloud cost tracking
        const instanceType = '{state.instance_type}';
        const COST_RATES = {{
            'gpu_1x_a10': 0.75,      // Lambda Labs A10
            'gpu_8x_a100': 1.29,     // Lambda Labs A100 (per GPU)
            'a10': 0.75,             // Generic A10
            'a100': 1.29,            // Generic A100
        }};

        function getHourlyRate(instanceType) {{
            // Try exact match first
            if (COST_RATES[instanceType.toLowerCase()]) {{
                return COST_RATES[instanceType.toLowerCase()];
            }}
            // Try partial match
            const typeStr = instanceType.toLowerCase();
            if (typeStr.includes('a100')) return COST_RATES['a100'];
            if (typeStr.includes('a10')) return COST_RATES['a10'];
            // Default to A10 rate
            return COST_RATES['a10'];
        }}

        function updateCostDisplay() {{
            // Only show costs for actual cloud training (not stub/local)
            if (!instanceType || instanceType === '' || instanceType === 'stub') {{
                document.getElementById('card-cost').style.display = 'none';
                return;
            }}

            const hourlyRate = getHourlyRate(instanceType);

            // Calculate running cost based on elapsed time
            const timeSinceSync = (Date.now() - lastSyncTime) / 1000;
            const liveElapsed = baseElapsedTime + timeSinceSync;
            const elapsedHours = liveElapsed / 3600;
            const runningCost = elapsedHours * hourlyRate;

            // Calculate estimated total cost
            let estimatedTotal = runningCost;
            if (etaSeconds > 0) {{
                const totalTimeSeconds = liveElapsed + etaSeconds;
                const totalHours = totalTimeSeconds / 3600;
                estimatedTotal = totalHours * hourlyRate;
            }}

            // Update display
            document.getElementById('stat-running-cost').textContent = `$${{runningCost.toFixed(2)}}`;
            document.getElementById('stat-est-total').textContent = `Est. Total: $${{estimatedTotal.toFixed(2)}}`;
        }}

        async function stopTraining() {{
            const btn = document.getElementById('stop-training-btn');
            const status = document.getElementById('stop-status');

            btn.disabled = true;
            btn.innerHTML = '<span style="font-size: 1.1rem;">⏳</span> Stopping...';
            btn.style.background = '#666';

            try {{
                // Try to create stop signal via API
                const response = await fetch('/api/stop', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }}
                }});

                if (response.ok) {{
                    btn.innerHTML = '<span style="font-size: 1.1rem;">✓</span> Stop Signal Sent';
                    btn.style.background = '#22c55e';
                    status.textContent = 'Training will stop after current step. Checkpoints will be downloaded.';
                    status.style.color = '#22c55e';
                }} else {{
                    throw new Error('Server returned ' + response.status);
                }}
            }} catch (e) {{
                // Fallback: show manual command
                btn.innerHTML = '<span style="font-size: 1.1rem;">!</span> Manual Stop Required';
                btn.style.background = '#f59e0b';
                status.innerHTML = 'Run this command to stop training:<br><code style="background: #1a1a24; padding: 4px 8px; border-radius: 4px; font-family: monospace;">touch training_output/STOP_TRAINING</code>';
                status.style.color = '#f59e0b';
            }}
        }}

        function updateTerminationStatus(data) {{
            const stopSection = document.getElementById('stop-training-section');
            if (!stopSection) return;

            const termStatus = data.termination_status || 'auto_complete';
            const termMessage = data.termination_message || '';

            const statusStyles = {{
                'auto_complete': {{ color: '#22c55e', icon: '✓', label: 'Training Complete' }},
                'auto_low_loss': {{ color: '#22c55e', icon: '✓', label: 'Auto-Stopped (Low Loss)' }},
                'user_stop': {{ color: '#f59e0b', icon: '■', label: 'Stopped by User' }},
            }};

            const style = statusStyles[termStatus] || statusStyles['auto_complete'];

            let html = `<div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 8px; color: ${{style.color}};">
                    <span style="font-size: 1.2rem;">${{style.icon}}</span>
                    <span style="font-weight: 600;">${{style.label}}</span>
                </div>`;

            if (termMessage) {{
                html += `<div style="font-size: 0.85rem; color: var(--text-muted); margin-left: 28px;">${{termMessage}}</div>`;
            }}

            html += '</div>';
            stopSection.innerHTML = html;
        }}

        function initCharts() {{
            const lossCtx = document.getElementById('lossChart').getContext('2d');
            lossChart = new Chart(lossCtx, {{
                type: 'line',
                data: {{
                    labels: losses.map(l => l.step),
                    datasets: [{{
                        label: 'Loss',
                        data: losses.map(l => l.loss),
                        borderColor: '#00d4aa',
                        backgroundColor: 'rgba(0, 212, 170, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: losses.length > 50 ? 0 : 3,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {{ duration: 500 }},
                    scales: {{
                        x: {{
                            title: {{ display: true, text: 'Step', color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.05)' }},
                            ticks: {{ color: '#888' }}
                        }},
                        y: {{
                            title: {{ display: true, text: 'Loss', color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.05)' }},
                            ticks: {{ color: '#888' }}
                        }}
                    }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});

            const epochCtx = document.getElementById('epochChart').getContext('2d');
            epochChart = new Chart(epochCtx, {{
                type: 'bar',
                data: {{
                    labels: epochAvg.map(e => `Epoch ${{e[0] + 1}}`),
                    datasets: [{{
                        label: 'Avg Loss',
                        data: epochAvg.map(e => e[1]),
                        backgroundColor: 'rgba(167, 139, 250, 0.6)',
                        borderColor: '#a78bfa',
                        borderWidth: 1,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {{ duration: 500 }},
                    scales: {{
                        y: {{
                            beginAtZero: false,
                            grid: {{ color: 'rgba(255,255,255,0.05)' }},
                            ticks: {{ color: '#888' }}
                        }},
                        x: {{
                            grid: {{ display: false }},
                            ticks: {{ color: '#888' }}
                        }}
                    }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});

            updateTrend();
        }}

        function updateTrend() {{
            if (losses.length >= 10) {{
                const recent = losses.slice(-10);
                const first = recent[0].loss;
                const last = recent[recent.length - 1].loss;
                const change = ((last - first) / first * 100).toFixed(1);
                const trendEl = document.getElementById('loss-trend');
                if (change < 0) {{
                    trendEl.textContent = `↓ ${{Math.abs(change)}}% (last 10)`;
                    trendEl.style.color = '#34d399';
                }} else {{
                    trendEl.textContent = `↑ ${{change}}% (last 10)`;
                    trendEl.style.color = '#ff5f5f';
                }}
            }}
        }}

        // Live elapsed timer variables
        let baseElapsedTime = {elapsed};  // Last known elapsed time from server
        let lastSyncTime = Date.now();    // When we last synced with server
        let lastSuccessfulFetch = Date.now();  // When we last got a successful response
        let currentJobId = '{state.job_id}';   // Current job ID
        const STALE_THRESHOLD_SECONDS = 30;    // Consider stale after 30s without updates

        // ETA tracking
        let etaSeconds = {eta_seconds};
        let avgStepTime = {avg_step_time};
        let remainingSteps = {remaining_steps};
        let isTrainingComplete = {'true' if is_training_complete else 'false'};

        function updateElapsedDisplay() {{
            // Don't update elapsed if training is complete
            if (isTrainingComplete) {{
                return;
            }}

            // Calculate live elapsed: base time + time since last sync
            const timeSinceSync = (Date.now() - lastSyncTime) / 1000;
            const liveElapsed = baseElapsedTime + timeSinceSync;
            const mins = Math.floor(liveElapsed / 60);
            const secs = Math.floor(liveElapsed % 60);
            document.getElementById('stat-elapsed').textContent = `${{mins}}m ${{secs}}s`;

            // Update ETA countdown
            if (etaSeconds > 0) {{
                const liveEta = Math.max(0, etaSeconds - timeSinceSync);
                const etaMins = Math.floor(liveEta / 60);
                const etaSecs = Math.floor(liveEta % 60);
                document.getElementById('stat-eta').textContent = `${{etaMins}}m ${{etaSecs}}s`;
            }}

            // Update cost display
            updateCostDisplay();
        }}

        function updateStatusIndicator() {{
            const timeSinceUpdate = (Date.now() - lastSuccessfulFetch) / 1000;
            const statusEl = document.getElementById('status');
            const statusText = document.getElementById('status-text');

            if (timeSinceUpdate > STALE_THRESHOLD_SECONDS) {{
                statusEl.className = 'status stale';
                const staleMins = Math.floor(timeSinceUpdate / 60);
                const staleSecs = Math.floor(timeSinceUpdate % 60);
                if (staleMins > 0) {{
                    statusText.innerHTML = `STALE <span class="stale-warning">(no update for ${{staleMins}}m ${{staleSecs}}s)</span>`;
                }} else {{
                    statusText.innerHTML = `STALE <span class="stale-warning">(no update for ${{staleSecs}}s)</span>`;
                }}
            }} else {{
                statusEl.className = 'status';
                statusText.textContent = 'LIVE';
            }}
        }}

        async function fetchAndUpdate() {{
            try {{
                const response = await fetch('training_log.json?t=' + Date.now());
                if (!response.ok) return;

                const data = await response.json();
                lastSuccessfulFetch = Date.now();

                // Check if job_id has changed - if so, reload to get fresh data
                if (data.job_id && data.job_id !== currentJobId) {{
                    console.log(`Job changed from ${{currentJobId}} to ${{data.job_id}}, reloading...`);
                    location.reload();
                    return;
                }}

                // Update job info display
                if (data.job_id) {{
                    const jobIdEl = document.querySelector('.job-id');
                    const jobHostEl = document.querySelector('.job-host');
                    const jobConfigEl = document.querySelector('.job-config');
                    if (jobIdEl) jobIdEl.textContent = `Job: ${{data.job_id}}`;
                    if (jobHostEl) {{
                        let hostText = data.hostname || 'local';
                        if (data.instance_ip) hostText += ` @ ${{data.instance_ip}}`;
                        jobHostEl.textContent = hostText;
                    }}
                    if (jobConfigEl && data.config_path) {{
                        jobConfigEl.textContent = data.config_path;
                    }}
                }}

                // Update setup panel if setup logs present
                if (data.setup_logs && data.setup_logs.length > 0) {{
                    const setupPanel = document.getElementById('setup-panel');
                    const setupLogs = document.getElementById('setup-logs');
                    const setupBadge = document.getElementById('setup-status-badge');

                    setupPanel.classList.remove('hidden');

                    // Update status badge
                    if (data.setup_status) {{
                        setupBadge.textContent = data.setup_status;
                        setupBadge.className = `setup-status-badge ${{data.setup_status}}`;
                    }}

                    // Update logs
                    setupLogs.innerHTML = data.setup_logs.map((log, i) =>
                        `<div class="setup-log-line${{i === data.setup_logs.length - 1 ? ' current' : ''}}">${{log}}</div>`
                    ).join('');

                    // Auto-scroll to bottom
                    setupLogs.scrollTop = setupLogs.scrollHeight;

                    // Hide setup panel when training starts
                    if (data.setup_status === 'training' || data.setup_status === 'complete') {{
                        setTimeout(() => setupPanel.classList.add('hidden'), 3000);
                    }}
                }}

                // Always update elapsed time base
                if (data.elapsed_time) {{
                    baseElapsedTime = data.elapsed_time;
                    lastSyncTime = Date.now();
                }}

                // Check for termination status (handles completed/stopped states)
                if (data.termination_status && !isTrainingComplete) {{
                    isTrainingComplete = true;
                    document.getElementById('stat-eta').textContent = 'Complete';
                    document.getElementById('eta-detail').textContent = '';
                    updateTerminationStatus(data);
                    updateCostDisplay();
                }}

                // Only update other stats if step changed
                if (data.step !== lastStep) {{
                    // Update with animation
                    const cards = document.querySelectorAll('.stat-card');
                    cards.forEach(c => c.classList.add('updating'));
                    setTimeout(() => cards.forEach(c => c.classList.remove('updating')), 300);

                    // Update stats
                    const totalEpochs = data.total_epochs || {config.num_train_epochs};
                    const displayEpoch = Math.min(data.epoch + 1, totalEpochs);  // Cap at max
                    document.getElementById('stat-epoch').textContent = `${{displayEpoch}} / ${{totalEpochs}}`;
                    document.getElementById('epoch-progress').style.width = `${{(displayEpoch / totalEpochs) * 100}}%`;
                    document.getElementById('stat-step').textContent = data.step;
                    document.getElementById('stat-loss').textContent = data.loss.toFixed(4);

                    // Loss delta
                    const delta = data.loss - lastLoss;
                    const deltaEl = document.getElementById('loss-delta');
                    if (delta < 0) {{
                        deltaEl.textContent = `↓ ${{Math.abs(delta).toFixed(4)}}`;
                        deltaEl.className = 'stat-delta positive';
                    }} else {{
                        deltaEl.textContent = `↑ ${{delta.toFixed(4)}}`;
                        deltaEl.className = 'stat-delta negative';
                    }}

                    // Other stats
                    if (data.losses && data.losses.length > 0) {{
                        const minLoss = Math.min(...data.losses.map(l => l.loss));
                        document.getElementById('stat-min-loss').textContent = minLoss.toFixed(4);

                        const recentLosses = data.losses.slice(-10);
                        const avgLoss = recentLosses.reduce((a, b) => a + b.loss, 0) / recentLosses.length;
                        document.getElementById('stat-avg-loss').textContent = avgLoss.toFixed(4);

                        // Calculate avg step time and update ETA
                        if (data.losses.length > 1) {{
                            let stepTimes = [];
                            for (let i = 1; i < data.losses.length; i++) {{
                                stepTimes.push(data.losses[i].time - data.losses[i-1].time);
                            }}
                            avgStepTime = stepTimes.reduce((a,b) => a+b, 0) / stepTimes.length;
                            document.getElementById('stat-step-time').textContent = avgStepTime.toFixed(1) + 's';

                            // Recalculate ETA
                            const totalEpochs = data.total_epochs || {config.num_train_epochs};
                            const currentEpoch = data.epoch;
                            const stepsInCompletedEpochs = data.losses.filter(l => l.epoch < currentEpoch).length;
                            const stepsPerEpoch = currentEpoch > 0 && stepsInCompletedEpochs > 0
                                ? stepsInCompletedEpochs / currentEpoch
                                : data.losses.length / (currentEpoch + 1);
                            const totalStepsEstimate = stepsPerEpoch * totalEpochs;
                            remainingSteps = Math.max(0, totalStepsEstimate - data.losses.length);
                            etaSeconds = remainingSteps * avgStepTime;

                            // Update ETA display
                            if (etaSeconds > 0) {{
                                const etaMins = Math.floor(etaSeconds / 60);
                                const etaSecs = Math.floor(etaSeconds % 60);
                                document.getElementById('stat-eta').textContent = `${{etaMins}}m ${{etaSecs}}s`;
                                document.getElementById('eta-detail').textContent = `~${{Math.round(remainingSteps)}} steps @ ${{avgStepTime.toFixed(1)}}s/step`;
                            }} else if (data.losses.length > 0) {{
                                // Training complete - stop elapsed timer and update UI
                                isTrainingComplete = true;
                                document.getElementById('stat-eta').textContent = 'Complete';
                                document.getElementById('eta-detail').textContent = '';
                                // Update cost display one final time
                                updateCostDisplay();
                                // Replace stop button with termination status
                                updateTerminationStatus(data);
                            }} else {{
                                // No data yet
                                document.getElementById('stat-eta').textContent = 'calculating...';
                            }}
                        }}

                        // Update charts
                        losses = data.losses;
                        lossChart.data.labels = losses.map(l => l.step);
                        lossChart.data.datasets[0].data = losses.map(l => l.loss);
                        lossChart.data.datasets[0].pointRadius = losses.length > 50 ? 0 : 3;
                        lossChart.update('none');

                        // Recalculate epoch averages
                        const epochLosses = {{}};
                        losses.forEach(l => {{
                            if (!epochLosses[l.epoch]) epochLosses[l.epoch] = [];
                            epochLosses[l.epoch].push(l.loss);
                        }});
                        epochAvg = Object.entries(epochLosses).map(([ep, arr]) => [parseInt(ep), arr.reduce((a,b) => a+b, 0) / arr.length]);
                        epochChart.data.labels = epochAvg.map(e => `Epoch ${{e[0] + 1}}`);
                        epochChart.data.datasets[0].data = epochAvg.map(e => e[1]);
                        epochChart.update('none');

                        updateTrend();
                    }}

                    lastStep = data.step;
                    lastLoss = data.loss;
                }}

                // Update evaluations if present
                if (data.evaluations && data.evaluations.length > 0) {{
                    renderEvaluations(data.evaluations);
                }}

                document.getElementById('update-indicator').textContent = 'Last updated: just now';
            }} catch (e) {{
                console.log('Update failed:', e);
            }}
        }}

        function renderEvaluations(evaluations) {{
            const panel = document.getElementById('eval-panel');
            const gallery = document.getElementById('eval-gallery');
            const metrics = document.getElementById('eval-metrics');

            if (evaluations.length === 0) {{
                panel.style.display = 'none';
                return;
            }}

            panel.style.display = 'block';

            // Calculate metrics
            const correctCount = evaluations.filter(e => e.correct).length;
            const avgDistance = evaluations.reduce((a, e) => a + e.distance, 0) / evaluations.length;
            const accuracy = (correctCount / evaluations.length * 100).toFixed(1);

            metrics.innerHTML = `
                <div class="metric">
                    <span class="metric-label">Accuracy</span>
                    <span class="metric-value">${{accuracy}}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Distance</span>
                    <span class="metric-value">${{avgDistance.toFixed(1)}}px</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Samples</span>
                    <span class="metric-value">${{evaluations.length}}</span>
                </div>
                <div class="legend" style="display: flex; gap: 16px; margin-left: auto; font-size: 0.75rem; align-items: center;">
                    <span style="display: flex; align-items: center; gap: 4px;">
                        <span style="width: 12px; height: 12px; border-radius: 50%; background: rgba(52, 211, 153, 0.8);"></span>
                        Human
                    </span>
                    <span style="display: flex; align-items: center; gap: 4px;">
                        <span style="width: 12px; height: 12px; border-radius: 50%; background: rgba(167, 139, 250, 0.8);"></span>
                        Predicted
                    </span>
                </div>
            `;

            // Render gallery (show last 9 evaluations)
            const recentEvals = evaluations.slice(-9);
            gallery.innerHTML = recentEvals.map((ev, i) => {{
                const statusClass = ev.correct ? 'correct' : 'incorrect';
                const statusText = ev.correct ? '✓ Correct' : '✗ Off by ' + (ev.distance * 100).toFixed(1) + '%';
                const humanX = (ev.human_action.x || 0).toFixed(3);
                const humanY = (ev.human_action.y || 0).toFixed(3);
                const predX = (ev.predicted_action.x || 0).toFixed(3);
                const predY = (ev.predicted_action.y || 0).toFixed(3);
                const rawOutput = ev.predicted_action.raw_output || '';
                const thoughtMatch = rawOutput.match(/Thought:([\\s\\S]*?)(?:Action:|$)/);
                const thought = thoughtMatch ? thoughtMatch[1].trim().substring(0, 200) : '';
                const sampleId = 'eval-' + ev.epoch + '-' + ev.sample_idx;
                return `
                    <div class="eval-sample">
                        <div style="position: relative;">
                            <img src="${{ev.image_path}}" alt="Sample ${{ev.sample_idx}}" onerror="this.style.display='none'">
                            <div class="overlay" style="width: 100%; height: 100%;">
                                <div class="marker human" style="left: ${{(ev.human_action.x || 0) * 100}}%; top: ${{(ev.human_action.y || 0) * 100}}%;" title="Human"></div>
                                <div class="marker predicted" style="left: ${{(ev.predicted_action.x || 0) * 100}}%; top: ${{(ev.predicted_action.y || 0) * 100}}%;" title="Predicted"></div>
                            </div>
                        </div>
                        <div class="info">
                            <span class="${{statusClass}}">${{statusText}}</span>
                            <span> | Epoch ${{ev.epoch + 1}}</span>
                        </div>
                        <div class="details">
                            <div class="coords">
                                <span class="human-coord">Human: (${{humanX}}, ${{humanY}})</span>
                                <span class="pred-coord">Pred: (${{predX}}, ${{predY}})</span>
                            </div>
                        </div>
                        ${{thought ? `
                            <div class="thinking">${{thought}}${{thought.length >= 200 ? '...' : ''}}</div>
                        ` : ''}}
                    </div>
                `;
            }}).join('');
        }}

        // Terminal output management
        let terminalAutoScroll = true;
        let terminalWrap = false;
        let terminalCollapsed = false;
        let lastTerminalSize = 0;
        const MAX_TERMINAL_LINES = 500;

        function toggleTerminal() {{
            const container = document.getElementById('terminal-container');
            const toggleText = document.getElementById('terminal-toggle-text');
            terminalCollapsed = !terminalCollapsed;

            if (terminalCollapsed) {{
                container.classList.add('collapsed');
                toggleText.textContent = 'Expand';
            }} else {{
                container.classList.remove('collapsed');
                toggleText.textContent = 'Collapse';
            }}
        }}

        function toggleAutoScroll() {{
            terminalAutoScroll = !terminalAutoScroll;
            const btn = document.getElementById('auto-scroll-btn');
            if (terminalAutoScroll) {{
                btn.classList.add('active');
                scrollTerminalToBottom();
            }} else {{
                btn.classList.remove('active');
            }}
        }}

        function toggleWrap() {{
            terminalWrap = !terminalWrap;
            const btn = document.getElementById('wrap-btn');
            const output = document.getElementById('terminal-output');
            if (terminalWrap) {{
                btn.classList.add('active');
                output.style.whiteSpace = 'pre-wrap';
            }} else {{
                btn.classList.remove('active');
                output.style.whiteSpace = 'pre';
            }}
        }}

        function scrollTerminalToBottom() {{
            const container = document.getElementById('terminal-container');
            container.scrollTop = container.scrollHeight;
        }}

        async function fetchTerminalOutput() {{
            try {{
                const response = await fetch('training.log?t=' + Date.now());
                if (!response.ok) {{
                    // File doesn't exist yet
                    return;
                }}

                const text = await response.text();
                const lines = text.trim().split('\\n');

                // Keep only last MAX_TERMINAL_LINES
                const displayLines = lines.slice(-MAX_TERMINAL_LINES);

                const output = document.getElementById('terminal-output');
                const lineCount = document.getElementById('terminal-line-count');

                // Update line count
                lineCount.textContent = lines.length;

                // Only update if content changed
                if (displayLines.length === 0) {{
                    output.innerHTML = '<div class="terminal-empty">Waiting for training output...</div>';
                    return;
                }}

                // Format lines with basic syntax highlighting
                const formattedLines = displayLines.map(line => {{
                    let className = 'terminal-line';

                    // Detect line type
                    if (line.match(/^\\d{{4}}-\\d{{2}}-\\d{{2}}/)) {{
                        className += ' timestamp';
                    }} else if (line.toLowerCase().includes('error') || line.toLowerCase().includes('failed')) {{
                        className += ' error';
                    }} else if (line.toLowerCase().includes('success') || line.toLowerCase().includes('complete')) {{
                        className += ' success';
                    }} else if (line.toLowerCase().includes('warning')) {{
                        className += ' warning';
                    }}

                    // Escape HTML
                    const escaped = line
                        .replace(/&/g, '&amp;')
                        .replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;');

                    return `<div class="${{className}}">${{escaped}}</div>`;
                }}).join('');

                output.innerHTML = formattedLines;

                // Auto-scroll if enabled and new content arrived
                if (terminalAutoScroll && lines.length > lastTerminalSize) {{
                    scrollTerminalToBottom();
                }}

                lastTerminalSize = lines.length;
            }} catch (err) {{
                console.error('Failed to fetch terminal output:', err);
            }}
        }}

        initCharts();
        updateCostDisplay();  // Initialize cost display
        fetchAndUpdate();  // Initial fetch on page load
        fetchTerminalOutput();  // Initial terminal fetch
        setInterval(fetchAndUpdate, 3000);
        setInterval(fetchTerminalOutput, 2000);  // Poll terminal output every 2 seconds
        setInterval(updateElapsedDisplay, 1000);  // Update elapsed time every second
        setInterval(updateStatusIndicator, 1000);  // Update LIVE/STALE indicator every second
    </script>
</body>
</html>'''
    return html


def regenerate_all_dashboards(output_dir: str | Path) -> list[Path]:
    """Regenerate all dashboards in a directory with static navigation.

    This updates dashboard.html and generates the unified viewer.html.
    Old comparison_*.html files are left in place but no longer linked.

    Args:
        output_dir: Directory containing dashboard files

    Returns:
        List of paths to regenerated files
    """
    output_dir = Path(output_dir)
    regenerated = []

    # Nav links are now fixed (Training + Viewer)
    nav_links = _build_nav_links()

    # Regenerate main dashboard
    if (output_dir / "training_log.json").exists():
        try:
            path = regenerate_local_dashboard(output_dir, nav_links=nav_links)
            regenerated.append(path)
        except Exception as e:
            print(f"Warning: Failed to regenerate dashboard: {e}")

    # Generate unified viewer if we have capture data
    try:
        viewer_path = generate_unified_viewer_from_output_dir(output_dir)
        if viewer_path:
            regenerated.append(viewer_path)
    except Exception as e:
        print(f"Warning: Failed to generate unified viewer: {e}")
        import traceback
        traceback.print_exc()

    return regenerated


def _copy_transcript_and_audio(capture_path: Path | None, output_dir: Path) -> None:
    """Copy transcript.json and convert audio to mp3 for viewer playback.

    Args:
        capture_path: Path to the capture directory (may be None)
        output_dir: Output directory for the viewer
    """
    import shutil
    import subprocess

    if capture_path is None or not capture_path.exists():
        return

    # Copy transcript.json if it exists
    transcript_src = capture_path / "transcript.json"
    transcript_dst = output_dir / "transcript.json"
    if transcript_src.exists() and not transcript_dst.exists():
        shutil.copy2(transcript_src, transcript_dst)
        print(f"  Copied transcript.json from capture")

    # Convert audio to mp3 if it exists (ffmpeg required)
    audio_dst = output_dir / "audio.mp3"
    if not audio_dst.exists():
        # Try common audio formats
        for audio_ext in [".flac", ".wav", ".m4a", ".aac", ".ogg"]:
            audio_src = capture_path / f"audio{audio_ext}"
            if audio_src.exists():
                try:
                    result = subprocess.run(
                        ["ffmpeg", "-i", str(audio_src), "-y", "-q:a", "2", str(audio_dst)],
                        capture_output=True,
                        timeout=60,
                    )
                    if result.returncode == 0:
                        print(f"  Converted {audio_src.name} to audio.mp3")
                    else:
                        print(f"  Warning: ffmpeg conversion failed for {audio_src.name}")
                except FileNotFoundError:
                    print("  Warning: ffmpeg not found, cannot convert audio")
                except subprocess.TimeoutExpired:
                    print("  Warning: ffmpeg timed out")
                break


def generate_unified_viewer_from_output_dir(output_dir: Path) -> Path | None:
    """Generate the unified viewer.html from existing data in output_dir.

    Collects predictions from any comparison_epoch*.html or comparison_*.html files
    and consolidates them into a single viewer with checkpoint dropdown.

    If the original capture is not available locally, extracts all data from
    existing comparison HTML files.
    """
    import re

    output_dir = Path(output_dir)

    # Try to load training log to get capture path and goal
    training_log_path = output_dir / "training_log.json"
    capture_path = None
    goal = "Complete the recorded workflow"
    capture_id = "unknown"

    if training_log_path.exists():
        with open(training_log_path) as f:
            log_data = json.load(f)

        capture_path_str = log_data.get("capture_path", "")
        if capture_path_str:
            capture_path = Path(capture_path_str)
            capture_id = capture_path.name
            if not capture_path.exists():
                print(f"Capture path not found locally: {capture_path}")
                capture_path = None  # Will extract from HTML files

    # Collect predictions and base data from JSON files or HTML files
    predictions_by_checkpoint: dict[str, list[dict]] = {"None": []}
    base_data: list[dict] | None = None

    # First, try to load from JSON files (preferred)
    for json_file in sorted(output_dir.glob("predictions_*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Determine checkpoint name from filename
            name_match = re.search(r'predictions_(.+)\.json', json_file.name)
            if name_match:
                raw_name = name_match.group(1)
                if raw_name.startswith('epoch'):
                    checkpoint_name = f"Epoch {raw_name[5:]}"
                elif raw_name == 'preview':
                    checkpoint_name = "Preview"
                else:
                    checkpoint_name = raw_name.title()
            else:
                checkpoint_name = json_file.stem

            # Extract base data from first file
            if base_data is None and 'base_data' in data:
                base_data = data['base_data']

            # Store predictions
            if 'predictions' in data:
                predictions_by_checkpoint[checkpoint_name] = data['predictions']
                print(f"  Loaded predictions from {json_file.name}")
        except Exception as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")

    # Fallback: look for comparison_epoch*.html files and extract their data
    for comp_file in sorted(output_dir.glob("comparison_epoch*.html")):
        match = re.search(r'epoch(\d+)', comp_file.name)
        if not match:
            continue

        epoch_num = match.group(1)
        checkpoint_name = f"Epoch {epoch_num}"

        # Extract comparisonData from the HTML
        try:
            html_content = comp_file.read_text()
            # Look for const comparisonData = [...];
            data_match = re.search(
                r'const\s+comparisonData\s*=\s*(\[.*?\]);',
                html_content,
                re.DOTALL
            )
            if data_match:
                comparison_data = json.loads(data_match.group(1))

                # Extract base data from the first file we find
                if base_data is None:
                    base_data = []
                    for item in comparison_data:
                        base_data.append({
                            "index": item.get("index", 0),
                            "time": item.get("time", 0),
                            "image_path": item.get("image_path", ""),
                            "human_action": item.get("human_action", {}),
                        })

                # Extract predictions
                predictions = []
                for item in comparison_data:
                    predictions.append({
                        "predicted_action": item.get("predicted_action"),
                        "match": item.get("match"),
                    })
                predictions_by_checkpoint[checkpoint_name] = predictions
                print(f"  Loaded predictions from {comp_file.name}")
        except Exception as e:
            print(f"  Warning: Could not extract data from {comp_file.name}: {e}")

    # Also check comparison_preview.html
    preview_file = output_dir / "comparison_preview.html"
    if preview_file.exists():
        try:
            html_content = preview_file.read_text()
            data_match = re.search(
                r'const\s+comparisonData\s*=\s*(\[.*?\]);',
                html_content,
                re.DOTALL
            )
            if data_match:
                comparison_data = json.loads(data_match.group(1))

                # Extract base data if we haven't yet
                if base_data is None:
                    base_data = []
                    for item in comparison_data:
                        base_data.append({
                            "index": item.get("index", 0),
                            "time": item.get("time", 0),
                            "image_path": item.get("image_path", ""),
                            "human_action": item.get("human_action", {}),
                        })

                predictions = []
                for item in comparison_data:
                    predictions.append({
                        "predicted_action": item.get("predicted_action"),
                        "match": item.get("match"),
                    })
                # Only add if it has actual predictions
                has_predictions = any(p.get("predicted_action") for p in predictions)
                if has_predictions and "Preview" not in predictions_by_checkpoint:
                    predictions_by_checkpoint["Preview"] = predictions
                    print(f"  Loaded predictions from comparison_preview.html")
        except Exception as e:
            print(f"  Warning: Could not extract data from comparison_preview.html: {e}")

    # If we still don't have base data, we can't generate the viewer
    if base_data is None:
        print("No comparison data found, cannot generate unified viewer")
        return None

    # Copy transcript and audio files from capture if available
    _copy_transcript_and_audio(capture_path, output_dir)

    # Generate the unified viewer using standalone HTML template
    # (Consolidated approach - always use standalone for reliability)
    viewer_path = output_dir / "viewer.html"

    _generate_unified_viewer_from_extracted_data(
        base_data=base_data,
        predictions_by_checkpoint=predictions_by_checkpoint,
        output_path=viewer_path,
        capture_id=capture_id,
        goal=goal,
    )

    return viewer_path


def _generate_unified_viewer_from_extracted_data(
    base_data: list[dict],
    predictions_by_checkpoint: dict[str, list[dict]],
    output_path: Path,
    capture_id: str = "unknown",
    goal: str = "Untitled",
) -> None:
    """Generate unified viewer from extracted comparison data.

    This is used when the original capture isn't available locally
    but we have comparison HTML files to extract from.
    """
    # Get shared header components for consistent nav
    shared_header_css = _get_shared_header_css()
    shared_header_html = _generate_shared_header_html("viewer")

    # Build base HTML from extracted data (standalone, no openadapt-capture dependency)
    base_data_json = json.dumps(base_data)
    predictions_json = json.dumps(predictions_by_checkpoint)
    captures_json = json.dumps([{
        "id": capture_id,
        "name": goal,
        "steps": len(base_data),
    }])
    current_capture_json = json.dumps(capture_id)

    # Find first image to get dimensions (for display)
    first_image_path = base_data[0].get("image_path", "") if base_data else ""

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified Viewer - {capture_id}</title>
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a24;
            --border-color: rgba(255, 255, 255, 0.06);
            --text-primary: #f0f0f0;
            --text-secondary: #888;
            --text-muted: #555;
            --accent: #00d4aa;
            --accent-dim: rgba(0, 212, 170, 0.15);
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px;
        }}
        {shared_header_css}
        .nav-bar {{
            display: flex;
            gap: 8px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .nav-link {{
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.8rem;
            text-decoration: none;
            color: var(--text-secondary);
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            transition: all 0.2s;
        }}
        .nav-link:hover {{ border-color: var(--accent); color: var(--text-primary); }}
        .nav-link.active {{
            background: var(--accent);
            color: var(--bg-primary);
            border-color: var(--accent);
            font-weight: 600;
        }}
        .nav-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-right: 8px;
        }}
        .viewer-controls {{
            display: flex;
            gap: 16px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .control-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .control-select {{
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            cursor: pointer;
            min-width: 200px;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23888' d='M3 4.5L6 7.5L9 4.5'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 12px center;
            padding-right: 32px;
            transition: all 0.2s;
        }}
        .control-select:hover {{ border-color: var(--accent); background-color: var(--bg-secondary); }}
        .control-select:focus {{ outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-dim); }}
        .control-hint {{ font-size: 0.7rem; color: var(--text-muted); }}
        .comparison-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 16px;
        }}
        .comparison-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 12px 18px;
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
        }}
        .comparison-panel h2 {{ font-size: 0.9rem; font-weight: 600; margin: 0; }}
        .comparison-content {{
            padding: 14px 18px;
            display: grid;
            grid-template-columns: 1fr 1fr auto;
            gap: 16px;
            align-items: start;
        }}
        .action-box {{ padding: 12px; border-radius: 8px; }}
        .action-box.human {{
            background: rgba(0, 212, 170, 0.1);
            border: 1px solid rgba(0, 212, 170, 0.3);
        }}
        .action-box.predicted {{
            background: rgba(167, 139, 250, 0.1);
            border: 1px solid rgba(167, 139, 250, 0.3);
        }}
        .action-box.predicted.disabled {{ opacity: 0.5; }}
        .action-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .action-details {{ font-family: "SF Mono", Monaco, monospace; font-size: 0.85rem; }}
        .match-indicator {{
            text-align: center;
            padding: 8px;
            border-radius: 6px;
            font-weight: 600;
            min-width: 80px;
        }}
        .match-indicator.match {{ background: rgba(52, 211, 153, 0.2); color: #34d399; }}
        .match-indicator.mismatch {{ background: rgba(255, 95, 95, 0.2); color: #ff5f5f; }}
        .match-indicator.pending {{ background: var(--bg-tertiary); color: var(--text-muted); }}
        .metrics-summary {{
            display: flex;
            gap: 16px;
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }}
        .metric-item {{ display: flex; align-items: center; gap: 6px; }}
        .metric-value {{ font-size: 0.9rem; font-weight: 600; color: var(--accent); }}
        .metric-label {{ font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; }}
        .overlay-toggles {{ display: flex; gap: 6px; margin-left: auto; }}
        .toggle-btn {{
            padding: 6px 12px;
            border: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.75rem;
        }}
        .toggle-btn.active {{ background: var(--accent); color: var(--bg-primary); border-color: var(--accent); }}
        .main-content {{ display: grid; grid-template-columns: 1fr 340px; gap: 24px; }}
        .viewer-section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }}
        .frame-container {{
            position: relative;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 420px;
        }}
        .frame-container img {{ max-width: 100%; max-height: 70vh; object-fit: contain; }}
        .click-marker {{
            position: absolute;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            pointer-events: none;
            z-index: 100;
        }}
        .click-marker.human {{
            background: rgba(0, 212, 170, 0.3);
            border: 3px solid #00d4aa;
            color: #00d4aa;
        }}
        .click-marker.predicted {{
            background: rgba(167, 139, 250, 0.3);
            border: 3px solid #a78bfa;
            color: #a78bfa;
        }}
        .click-marker.human::after {{ content: 'H'; }}
        .click-marker.predicted::after {{ content: 'AI'; font-size: 10px; }}
        .sidebar {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .step-list {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            max-height: 500px;
            overflow-y: auto;
        }}
        .step-item {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            transition: background 0.2s;
        }}
        .step-item:hover {{ background: var(--bg-tertiary); }}
        .step-item.active {{ background: var(--accent-dim); border-left: 3px solid var(--accent); }}
        .step-index {{ font-weight: 600; color: var(--accent); }}
        .step-action {{ font-size: 0.85rem; color: var(--text-secondary); }}
        .playback-controls {{
            display: flex;
            gap: 8px;
            padding: 12px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .playback-btn {{
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            min-width: 40px;
            text-align: center;
        }}
        .playback-btn:hover {{ border-color: var(--accent); }}
        .playback-btn.active {{ background: var(--accent); color: var(--bg-primary); border-color: var(--accent); }}
        .playback-btn.primary {{ flex: 1; min-width: 60px; }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-left: auto;
        }}
        .speed-control label {{
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        .speed-control select {{
            padding: 4px 8px;
            border-radius: 4px;
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            font-size: 0.8rem;
            cursor: pointer;
        }}
        .progress-bar {{
            width: 100%;
            height: 4px;
            background: var(--bg-tertiary);
            border-radius: 2px;
            margin-top: 8px;
            overflow: hidden;
            cursor: pointer;
        }}
        .progress-bar .progress {{
            height: 100%;
            background: var(--accent);
            transition: width 0.1s ease;
        }}
        .details-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-top: 16px;
        }}
        .details-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        .details-content {{
            padding: 12px 16px;
            font-size: 0.82rem;
            max-height: 400px;
            overflow-y: auto;
        }}
        .detail-row {{
            display: flex;
            margin-bottom: 6px;
        }}
        .detail-key {{
            color: var(--text-muted);
            min-width: 70px;
            font-size: 0.75rem;
            text-transform: uppercase;
        }}
        .detail-value {{
            font-family: "SF Mono", Monaco, monospace;
            color: var(--text-secondary);
        }}
        .copy-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.7rem;
            text-transform: uppercase;
        }}
        .copy-btn:hover {{ background: var(--bg-secondary); color: var(--text-primary); }}
        .copy-btn.copied {{ background: var(--accent-dim); color: var(--accent); border-color: var(--accent); }}
        .cost-panel {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.05));
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
            display: none;
        }}
        .cost-panel.visible {{ display: flex; }}
        .cost-panel .cost-items {{
            display: flex;
            gap: 24px;
            align-items: center;
            flex: 1;
        }}
        .cost-panel .cost-item {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}
        .cost-panel .cost-label {{
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .cost-panel .cost-value {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #ef4444;
            font-family: "SF Mono", Monaco, monospace;
        }}
        .cost-panel .cost-info {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-left: auto;
        }}
        .transcript-panel {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }}
        .transcript-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 18px;
            border-bottom: 1px solid var(--border-color);
        }}
        .transcript-panel h2 {{
            font-size: 0.9rem;
            font-weight: 600;
            margin: 0;
        }}
        .transcript-follow-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.7rem;
            transition: all 0.2s;
        }}
        .transcript-follow-btn:hover {{
            border-color: var(--accent);
            color: var(--text-secondary);
        }}
        .transcript-follow-btn.active {{
            background: var(--accent-dim);
            border-color: var(--accent);
            color: var(--accent);
        }}
        .transcript-content {{
            padding: 14px 18px;
            font-size: 0.85rem;
            line-height: 1.9;
            color: var(--text-secondary);
            max-height: 150px;
            overflow-y: auto;
        }}
        .transcript-segment {{
            display: inline;
            cursor: pointer;
            padding: 2px 6px;
            border-radius: 4px;
            transition: all 0.15s ease;
        }}
        .transcript-segment:hover {{
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }}
        .transcript-segment.active {{
            background: var(--accent-dim);
            color: var(--accent);
        }}
        .transcript-time {{
            color: var(--text-muted);
            font-size: 0.7rem;
            font-family: "SF Mono", Monaco, monospace;
            margin-right: 4px;
        }}
        .transcript-empty {{
            color: var(--text-muted);
            font-style: italic;
            text-align: center;
            padding: 16px;
        }}
        .step-list-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        .step-list-header h3 {{
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0;
        }}
        .copy-all-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.7rem;
            text-transform: uppercase;
        }}
        .copy-all-btn:hover {{ background: var(--bg-secondary); color: var(--text-primary); }}
        .copy-all-btn.copied {{ background: var(--accent-dim); color: var(--accent); border-color: var(--accent); }}
    </style>
</head>
<body>
    {shared_header_html}

    <div class="container">
        <div class="viewer-controls">
            <div class="control-group">
                <span class="control-label">Training Example:</span>
                <select class="control-select" id="capture-select"></select>
                <span class="control-hint" id="capture-hint"></span>
            </div>
            <div class="control-group">
                <span class="control-label">Checkpoint:</span>
                <select class="control-select" id="checkpoint-select"></select>
            </div>
        </div>

        <div class="cost-panel" id="cost-panel">
            <div class="cost-items">
                <div class="cost-item">
                    <div class="cost-label">Running Cost</div>
                    <div class="cost-value" id="cost-running">$0.00</div>
                </div>
                <div class="cost-item">
                    <div class="cost-label">Total Cost</div>
                    <div class="cost-value" id="cost-total">$0.00</div>
                </div>
                <div class="cost-info" id="cost-info"></div>
            </div>
        </div>

        <div class="comparison-panel">
            <div class="comparison-header">
                <h2>Action Comparison</h2>
                <div class="metrics-summary" id="metrics-summary"></div>
                <div class="overlay-toggles" id="overlay-toggles"></div>
            </div>
            <div class="comparison-content">
                <div class="action-box human">
                    <div class="action-label">Human Action</div>
                    <div class="action-details" id="human-action"></div>
                </div>
                <div class="action-box predicted" id="predicted-box">
                    <div class="action-label">Model Prediction</div>
                    <div class="action-details" id="predicted-action"></div>
                </div>
                <div class="match-indicator" id="match-indicator"></div>
            </div>
        </div>

        <div class="main-content">
            <div class="viewer-section">
                <div class="frame-container" id="frame-container">
                    <img id="frame-image" src="" alt="Screenshot">
                    <div id="image-placeholder" style="display:none;flex-direction:column;align-items:center;justify-content:center;min-height:300px;width:100%;"></div>
                </div>
            </div>
            <div class="sidebar">
                <div class="playback-controls">
                    <button class="playback-btn" id="rewind-btn" title="Rewind (Home)">⏮</button>
                    <button class="playback-btn" id="prev-btn" title="Previous (←)">◀</button>
                    <button class="playback-btn primary" id="play-btn" title="Play/Pause (Space)">▶ Play</button>
                    <button class="playback-btn" id="next-btn" title="Next (→)">▶</button>
                    <button class="playback-btn" id="end-btn" title="End (End)">⏭</button>
                    <div class="speed-control">
                        <label>Speed</label>
                        <select id="speed-select">
                            <option value="2000">0.5x</option>
                            <option value="1000" selected>1x</option>
                            <option value="500">2x</option>
                            <option value="250">4x</option>
                        </select>
                    </div>
                    <div class="progress-bar" id="progress-bar">
                        <div class="progress" id="progress"></div>
                    </div>
                </div>
                <div class="step-list" id="step-list">
                    <div class="step-list-header">
                        <h3>Steps</h3>
                        <button class="copy-all-btn" id="copy-all-btn">Copy All</button>
                    </div>
                    <div id="step-list-items"></div>
                </div>
                <div class="transcript-panel" id="transcript-panel">
                    <div class="transcript-header">
                        <h2>Transcript</h2>
                        <button class="transcript-follow-btn active" id="transcript-follow-btn" title="Auto-scroll to active segment">Follow</button>
                    </div>
                    <div class="transcript-content" id="transcript-content"></div>
                </div>
                <audio id="audio" style="display:none;"></audio>
                <div class="details-panel" id="details-panel">
                    <div class="details-header">
                        <span style="font-size:0.9rem;font-weight:600;">Step Details</span>
                        <button class="copy-btn" id="copy-btn">Copy</button>
                    </div>
                    <div class="details-content" id="details-content"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
    const baseData = {base_data_json};
    const predictionsByCheckpoint = {predictions_json};
    const availableCaptures = {captures_json};
    const currentCaptureId = {current_capture_json};

    let currentIndex = 0;
    let currentCheckpoint = 'None';
    let showHumanOverlay = true;
    let showPredictedOverlay = true;
    let isPlaying = false;
    let playInterval = null;
    let playSpeed = 1000;  // ms per step

    // Cloud cost tracking
    const COST_RATES = {{
        'gpu_1x_a10': 0.75,      // Lambda Labs A10
        'gpu_8x_a100': 1.29,     // Lambda Labs A100 (per GPU)
        'a10': 0.75,             // Generic A10
        'a100': 1.29,            // Generic A100
    }};

    function getHourlyRate(instanceType) {{
        if (!instanceType) return 0;
        // Try exact match first
        const lowerType = instanceType.toLowerCase();
        if (COST_RATES[lowerType]) {{
            return COST_RATES[lowerType];
        }}
        // Try partial match
        if (lowerType.includes('a100')) return COST_RATES['a100'];
        if (lowerType.includes('a10')) return COST_RATES['a10'];
        // Default to A10 rate
        return COST_RATES['a10'];
    }}

    async function loadAndDisplayCosts() {{
        try {{
            const response = await fetch('training_log.json?t=' + Date.now());
            if (!response.ok) return;

            const data = await response.json();
            const instanceType = data.instance_type || '';

            // Only show costs for actual cloud training (not stub/local)
            if (!instanceType || instanceType === '' || instanceType === 'stub') {{
                document.getElementById('cost-panel').style.display = 'none';
                return;
            }}

            const hourlyRate = getHourlyRate(instanceType);
            const elapsedTime = data.elapsed_time || 0;
            const elapsedHours = elapsedTime / 3600;
            const totalCost = elapsedHours * hourlyRate;

            // Update display
            document.getElementById('cost-running').textContent = `$${{totalCost.toFixed(2)}}`;
            document.getElementById('cost-total').textContent = `$${{totalCost.toFixed(2)}}`;
            document.getElementById('cost-info').textContent = `${{instanceType}} @ $${{hourlyRate.toFixed(2)}}/hr`;
            document.getElementById('cost-panel').classList.add('visible');
        }} catch (e) {{
            // Silently fail if training_log.json not available
            console.log('Could not load training costs:', e);
        }}
    }}

    function getMergedData() {{
        const predictions = predictionsByCheckpoint[currentCheckpoint] || [];
        return baseData.map((base, i) => {{
            const pred = predictions[i] || {{}};
            return {{
                ...base,
                predicted_action: pred.predicted_action || null,
                match: pred.match !== undefined ? pred.match : null,
            }};
        }});
    }}

    function parseModelOutput(rawOutput) {{
        // Parse model output for structured action commands
        let action = null;
        let thinking = '';

        // Try to extract SoM actions: CLICK([N]), TYPE([N], "text"), TYPE("text")
        const clickSomMatch = rawOutput.match(/CLICK\\s*\\(\\s*\\[\\s*(\\d+)\\s*\\]\\s*\\)/);
        const typeSomMatch = rawOutput.match(/TYPE\\s*\\(\\s*\\[\\s*(\\d+)\\s*\\]\\s*,\\s*["']([^"']*)["']\\s*\\)/);
        const typeSimpleMatch = rawOutput.match(/TYPE\\s*\\(\\s*["']([^"']*)["']\\s*\\)/);

        // Try coordinate-based: CLICK(x=0.5, y=0.5)
        const clickCoordMatch = rawOutput.match(/CLICK\\s*\\(\\s*x\\s*=\\s*([\\d.]+)\\s*,\\s*y\\s*=\\s*([\\d.]+)\\s*\\)/);

        // Try to extract thinking/reasoning
        const thinkMatch = rawOutput.match(/(?:Thought|Thinking|Reasoning|Analysis):\\s*([\\s\\S]*?)(?:Action:|$)/i);
        const actionMatch = rawOutput.match(/Action:\\s*([^\\n]+)/i);

        if (thinkMatch) thinking = thinkMatch[1].trim().substring(0, 150);

        if (clickSomMatch) {{
            action = {{ type: 'click', element: `[${{clickSomMatch[1]}}]` }};
        }} else if (typeSomMatch) {{
            action = {{ type: 'type', element: `[${{typeSomMatch[1]}}]`, text: typeSomMatch[2] }};
        }} else if (typeSimpleMatch) {{
            action = {{ type: 'type', text: typeSimpleMatch[1] }};
        }} else if (clickCoordMatch) {{
            action = {{ type: 'click', x: parseFloat(clickCoordMatch[1]), y: parseFloat(clickCoordMatch[2]) }};
        }} else if (actionMatch) {{
            // Extract the action line for cleaner display
            action = {{ type: 'raw', text: actionMatch[1].trim() }};
        }}

        // Generate HTML
        let html = '';
        if (action) {{
            if (action.type === 'click' && action.element) {{
                html = `<div style="font-weight:600;color:var(--accent);">CLICK(${{action.element}})</div>`;
            }} else if (action.type === 'click' && action.x !== undefined) {{
                html = `<div style="font-weight:600;color:var(--accent);">CLICK(x=${{action.x.toFixed(2)}}, y=${{action.y.toFixed(2)}})</div>`;
            }} else if (action.type === 'type') {{
                const elem = action.element ? `${{action.element}}, ` : '';
                html = `<div style="font-weight:600;color:var(--accent);">TYPE(${{elem}}"${{action.text}}")</div>`;
            }} else if (action.type === 'raw') {{
                html = `<div style="color:var(--accent);">${{action.text}}</div>`;
            }}
            if (thinking) {{
                html += `<div style="font-size:0.8rem;color:var(--text-muted);margin-top:4px;max-height:60px;overflow:hidden;">${{thinking}}...</div>`;
            }}
        }} else {{
            // No parseable action - show truncated raw output
            const truncated = rawOutput.substring(0, 200).replace(/\\n/g, ' ');
            html = `<div style="font-size:0.85rem;color:var(--text-muted);max-height:80px;overflow:hidden;">${{truncated}}${{rawOutput.length > 200 ? '...' : ''}}</div>`;
        }}

        return {{ action, thinking, html }};
    }}

    function initDropdowns() {{
        const captureSelect = document.getElementById('capture-select');
        const checkpointSelect = document.getElementById('checkpoint-select');
        const captureHint = document.getElementById('capture-hint');

        captureSelect.innerHTML = '';
        availableCaptures.forEach(cap => {{
            const opt = document.createElement('option');
            opt.value = cap.id;
            opt.textContent = `${{cap.name}} (${{cap.steps}} steps)`;
            opt.selected = cap.id === currentCaptureId;
            captureSelect.appendChild(opt);
        }});
        captureHint.textContent = `(${{availableCaptures.length}} available)`;

        checkpointSelect.innerHTML = '';
        const checkpointNames = Object.keys(predictionsByCheckpoint);
        checkpointNames.sort((a, b) => {{
            if (a === 'None') return -1;
            if (b === 'None') return 1;
            const aNum = parseInt(a.match(/\\d+/)?.[0] || '999');
            const bNum = parseInt(b.match(/\\d+/)?.[0] || '999');
            return aNum - bNum;
        }});
        checkpointNames.forEach(name => {{
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name === 'None' ? 'None (Capture Only)' : name;
            checkpointSelect.appendChild(opt);
        }});
        const latestCheckpoint = checkpointNames.filter(n => n !== 'None').pop();
        if (latestCheckpoint) {{
            checkpointSelect.value = latestCheckpoint;
            currentCheckpoint = latestCheckpoint;
        }}
        checkpointSelect.addEventListener('change', (e) => {{
            currentCheckpoint = e.target.value;
            updateMetrics();
            updateDisplay();
        }});
    }}

    function computeMetrics() {{
        const data = getMergedData();
        let matches = 0, total = 0;
        data.forEach(d => {{
            if (d.match !== null) {{ total++; if (d.match) matches++; }}
        }});
        return {{
            accuracy: total > 0 ? (matches / total * 100).toFixed(1) : 'N/A',
            total: data.length,
            hasPredictions: total > 0,
        }};
    }}

    function updateMetrics() {{
        const metricsEl = document.getElementById('metrics-summary');
        const metrics = computeMetrics();
        if (!metrics.hasPredictions) {{
            metricsEl.innerHTML = `<div class="metric-item"><span class="metric-label">Steps:</span><span class="metric-value">${{metrics.total}}</span></div>`;
        }} else {{
            metricsEl.innerHTML = `<div class="metric-item"><span class="metric-label">Accuracy:</span><span class="metric-value">${{metrics.accuracy}}%</span></div><div class="metric-item"><span class="metric-label">Steps:</span><span class="metric-value">${{metrics.total}}</span></div>`;
        }}
    }}

    function updateDisplay() {{
        const data = getMergedData()[currentIndex];
        if (!data) return;

        // Update image - handle both local and remote paths
        const imgEl = document.getElementById('frame-image');
        const placeholderEl = document.getElementById('image-placeholder');

        // Check if image path is remote (Lambda Labs path)
        const imagePath = data.image_path || '';
        const isRemote = imagePath.startsWith('/home/ubuntu/') || imagePath.startsWith('/root/');

        // Try local screenshots folder first
        const localPath = isRemote ? 'screenshots/' + imagePath.split('/').pop() : imagePath;

        imgEl.src = localPath;
        imgEl.style.display = 'block';
        if (placeholderEl) placeholderEl.style.display = 'none';

        imgEl.onerror = () => {{
            imgEl.style.display = 'none';
            if (placeholderEl) {{
                placeholderEl.style.display = 'flex';
                placeholderEl.innerHTML = `
                    <div style="text-align:center;padding:40px;color:var(--text-muted);">
                        <div style="font-size:2rem;margin-bottom:12px;">📷</div>
                        <div style="margin-bottom:8px;color:var(--text-secondary);">Screenshots not downloaded</div>
                        <div style="font-size:0.8rem;margin-bottom:12px;">
                            Run: <code style="background:var(--bg-tertiary);padding:4px 8px;border-radius:4px;">uv run python -m openadapt_ml.cloud.lambda_labs rsync remote:/home/ubuntu/capture/screenshots/ training_output/screenshots/</code>
                        </div>
                        <div style="font-size:0.75rem;color:var(--text-muted);">Step ${{currentIndex + 1}} of ${{baseData.length}}</div>
                    </div>
                `;
            }}
        }};

        // Update human action
        const humanEl = document.getElementById('human-action');
        humanEl.innerHTML = `<div>Type: ${{data.human_action.type || 'unknown'}}</div>${{data.human_action.x !== null && data.human_action.x !== undefined ? `<div>Position: (${{(data.human_action.x * 100).toFixed(1)}}%, ${{(data.human_action.y * 100).toFixed(1)}}%)</div>` : ''}}${{data.human_action.text ? `<div>Text: ${{data.human_action.text}}</div>` : ''}}`;

        // Update predicted action
        const predictedEl = document.getElementById('predicted-action');
        const predictedBox = document.getElementById('predicted-box');
        const hasPredictions = currentCheckpoint !== 'None';
        predictedBox.classList.toggle('disabled', !hasPredictions);
        if (!hasPredictions) {{
            predictedEl.innerHTML = '<em style="color:var(--text-muted);">Select a checkpoint</em>';
        }} else if (data.predicted_action) {{
            const pred = data.predicted_action;
            if (pred.x !== undefined) {{
                predictedEl.innerHTML = `<div>Type: ${{pred.type || 'click'}}</div><div>Position: (${{(pred.x * 100).toFixed(1)}}%, ${{(pred.y * 100).toFixed(1)}}%)</div>`;
            }} else {{
                // Parse raw_output for actions
                const rawOutput = pred.raw_output || JSON.stringify(pred);
                const parsed = parseModelOutput(rawOutput);
                predictedEl.innerHTML = parsed.html;
            }}
        }} else {{
            predictedEl.innerHTML = '<em style="color:var(--text-muted);">No prediction</em>';
        }}

        // Update match indicator
        const matchEl = document.getElementById('match-indicator');
        if (!hasPredictions) {{
            matchEl.className = 'match-indicator pending'; matchEl.textContent = '—';
        }} else if (data.match === true) {{
            matchEl.className = 'match-indicator match'; matchEl.textContent = '✓ Match';
        }} else if (data.match === false) {{
            matchEl.className = 'match-indicator mismatch'; matchEl.textContent = '✗ Mismatch';
        }} else {{
            matchEl.className = 'match-indicator pending'; matchEl.textContent = '—';
        }}

        // Update click overlays
        updateClickOverlays();

        // Update step list active state
        document.querySelectorAll('.step-item').forEach((el, i) => {{
            el.classList.toggle('active', i === currentIndex);
        }});

        // Update details panel
        updateDetailsPanel(data);

        // Update progress bar
        updateProgressBar();
    }}

    function updateDetailsPanel(data) {{
        const detailsEl = document.getElementById('details-content');
        const action = data.human_action;

        // Build human action section
        let html = `
            <div style="font-weight:600;font-size:0.8rem;color:var(--accent);margin-bottom:8px;text-transform:uppercase;">Human Action</div>
            <div class="detail-row"><span class="detail-key">Step</span><span class="detail-value">${{currentIndex + 1}} of ${{baseData.length}}</span></div>
            <div class="detail-row"><span class="detail-key">Time</span><span class="detail-value">${{data.time ? data.time.toFixed(2) + 's' : '—'}}</span></div>
            <div class="detail-row"><span class="detail-key">Type</span><span class="detail-value">${{action.type}}</span></div>
        `;
        if (action.x !== null && action.x !== undefined) {{
            html += `<div class="detail-row"><span class="detail-key">Position</span><span class="detail-value">(${{(action.x * 100).toFixed(2)}}%, ${{(action.y * 100).toFixed(2)}}%)</span></div>`;
        }}
        if (action.text) {{
            html += `<div class="detail-row"><span class="detail-key">Text</span><span class="detail-value">"${{action.text}}"</span></div>`;
        }}

        // Build prediction section if available
        if (data.predicted_action && currentCheckpoint !== 'None') {{
            const pred = data.predicted_action;
            html += `<div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border-color);">`;
            html += `<div style="font-weight:600;font-size:0.8rem;color:#a78bfa;margin-bottom:8px;text-transform:uppercase;display:flex;justify-content:space-between;">
                <span>Model Prediction</span>
                <span style="color:${{data.match === true ? '#34d399' : data.match === false ? '#ff5f5f' : 'var(--text-muted)'}};">${{data.match === true ? '✓ Match' : data.match === false ? '✗ Mismatch' : '—'}}</span>
            </div>`;

            // Show predicted position if available
            if (pred.x !== undefined && pred.y !== undefined) {{
                html += `<div class="detail-row"><span class="detail-key">Type</span><span class="detail-value">${{pred.type || 'click'}}</span></div>`;
                html += `<div class="detail-row"><span class="detail-key">Position</span><span class="detail-value">(${{(pred.x * 100).toFixed(2)}}%, ${{(pred.y * 100).toFixed(2)}}%)</span></div>`;
            }}

            // Show raw output (model reasoning)
            if (pred.raw_output) {{
                const rawOutput = pred.raw_output;
                html += `<div class="detail-row" style="flex-direction:column;margin-top:8px;">
                    <span class="detail-key" style="margin-bottom:4px;">Model Output</span>
                    <div class="detail-value" style="font-size:0.75rem;max-height:150px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;background:var(--bg-tertiary);padding:8px;border-radius:4px;">${{rawOutput.replace(/</g, '&lt;').replace(/>/g, '&gt;')}}</div>
                </div>`;
            }} else {{
                // Show whatever fields are present
                const predStr = JSON.stringify(pred, null, 2);
                html += `<div class="detail-row" style="flex-direction:column;margin-top:8px;">
                    <span class="detail-key" style="margin-bottom:4px;">Prediction Data</span>
                    <div class="detail-value" style="font-size:0.75rem;max-height:100px;overflow-y:auto;white-space:pre;background:var(--bg-tertiary);padding:8px;border-radius:4px;">${{predStr}}</div>
                </div>`;
            }}
            html += `</div>`;
        }}

        detailsEl.innerHTML = html;
    }}

    function setupCopyButton() {{
        document.getElementById('copy-btn').onclick = function() {{
            const data = getMergedData()[currentIndex];
            const text = JSON.stringify(data, null, 2);
            navigator.clipboard.writeText(text);
            this.textContent = 'Copied!';
            this.classList.add('copied');
            setTimeout(() => {{
                this.textContent = 'Copy';
                this.classList.remove('copied');
            }}, 1500);
        }};
    }}

    function setupCopyAllButton() {{
        const btn = document.getElementById('copy-all-btn');
        if (!btn) return;

        btn.onclick = function() {{
            const allData = getMergedData();
            const text = JSON.stringify(allData, null, 2);
            navigator.clipboard.writeText(text);
            this.textContent = 'Copied!';
            this.classList.add('copied');
            setTimeout(() => {{
                this.textContent = 'Copy All';
                this.classList.remove('copied');
            }}, 1500);
        }};
    }}

    function updateClickOverlays() {{
        document.querySelectorAll('.click-marker').forEach(el => el.remove());
        const data = getMergedData()[currentIndex];
        if (!data) return;
        const container = document.getElementById('frame-container');

        if (showHumanOverlay && data.human_action.x !== null && data.human_action.x !== undefined) {{
            const marker = document.createElement('div');
            marker.className = 'click-marker human';
            marker.style.left = (data.human_action.x * 100) + '%';
            marker.style.top = (data.human_action.y * 100) + '%';
            container.appendChild(marker);
        }}
        if (showPredictedOverlay && data.predicted_action && data.predicted_action.x !== undefined) {{
            const marker = document.createElement('div');
            marker.className = 'click-marker predicted';
            marker.style.left = (data.predicted_action.x * 100) + '%';
            marker.style.top = (data.predicted_action.y * 100) + '%';
            container.appendChild(marker);
        }}
    }}

    function buildStepList() {{
        const listEl = document.getElementById('step-list');
        listEl.innerHTML = '';
        const typeColors = {{
            click: '#ff5f5f',
            double_click: '#ff5f5f',
            type: '#34d399',
            scroll: '#a78bfa',
            drag: '#00d4aa',
            done: '#888',
        }};
        baseData.forEach((step, i) => {{
            const item = document.createElement('div');
            item.className = 'step-item' + (i === currentIndex ? ' active' : '');
            const action = step.human_action;
            const time = step.time ? step.time.toFixed(1) + 's' : '';
            const typeColor = typeColors[action.type] || 'var(--text-secondary)';
            const actionDetail = action.type === 'type' && action.text
                ? `"${{action.text.length > 15 ? action.text.slice(0,15) + '...' : action.text}}"`
                : (action.x !== null && action.x !== undefined ? `(${{(action.x*100).toFixed(0)}}%, ${{(action.y*100).toFixed(0)}}%)` : '');
            item.innerHTML = `
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-family:monospace;font-size:0.7rem;color:var(--text-muted);min-width:40px;">${{time}}</span>
                    <span style="font-weight:600;color:${{typeColor}};text-transform:uppercase;font-size:0.75rem;">${{action.type}}</span>
                </div>
                <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:2px;font-family:monospace;">${{actionDetail}}</div>
            `;
            item.onclick = () => {{ currentIndex = i; updateDisplay(); }};
            listEl.appendChild(item);
        }});
    }}

    function setupOverlayToggles() {{
        const container = document.getElementById('overlay-toggles');
        container.innerHTML = `<button class="toggle-btn active" id="toggle-human" title="Toggle human overlay (H)">Human</button><button class="toggle-btn active" id="toggle-predicted" title="Toggle AI overlay (A)">AI</button>`;
        document.getElementById('toggle-human').onclick = function() {{
            showHumanOverlay = !showHumanOverlay;
            this.classList.toggle('active', showHumanOverlay);
            updateClickOverlays();
        }};
        document.getElementById('toggle-predicted').onclick = function() {{
            showPredictedOverlay = !showPredictedOverlay;
            this.classList.toggle('active', showPredictedOverlay);
            updateClickOverlays();
        }};
    }}

    function updateProgressBar() {{
        const progress = document.getElementById('progress');
        if (progress) {{
            const pct = (currentIndex / (baseData.length - 1)) * 100;
            progress.style.width = pct + '%';
        }}
    }}

    function stopPlayback() {{
        isPlaying = false;
        if (playInterval) {{
            clearInterval(playInterval);
            playInterval = null;
        }}
        const playBtn = document.getElementById('play-btn');
        if (playBtn) {{
            playBtn.textContent = '▶ Play';
            playBtn.classList.remove('active');
        }}
        // Pause audio if playing
        if (audioElement && !audioElement.paused) {{
            audioElement.pause();
        }}
    }}

    function startPlayback() {{
        isPlaying = true;
        const playBtn = document.getElementById('play-btn');
        if (playBtn) {{
            playBtn.textContent = '⏸ Pause';
            playBtn.classList.add('active');
        }}
        // Start audio if available
        if (audioElement && audioElement.src) {{
            audioElement.play().catch(e => console.log('Audio play failed:', e));
        }}
        playInterval = setInterval(() => {{
            if (currentIndex < baseData.length - 1) {{
                currentIndex++;
                updateDisplay();
            }} else {{
                stopPlayback();
            }}
        }}, playSpeed);
    }}

    function togglePlayback() {{
        if (isPlaying) {{
            stopPlayback();
        }} else {{
            startPlayback();
        }}
    }}

    function setupPlaybackControls() {{
        // Rewind
        document.getElementById('rewind-btn').onclick = () => {{
            stopPlayback();
            currentIndex = 0;
            updateDisplay();
        }};

        // Previous
        document.getElementById('prev-btn').onclick = () => {{
            stopPlayback();
            if (currentIndex > 0) {{ currentIndex--; updateDisplay(); }}
        }};

        // Play/Pause
        document.getElementById('play-btn').onclick = togglePlayback;

        // Next
        document.getElementById('next-btn').onclick = () => {{
            stopPlayback();
            if (currentIndex < baseData.length - 1) {{ currentIndex++; updateDisplay(); }}
        }};

        // End
        document.getElementById('end-btn').onclick = () => {{
            stopPlayback();
            currentIndex = baseData.length - 1;
            updateDisplay();
        }};

        // Speed control
        document.getElementById('speed-select').onchange = (e) => {{
            playSpeed = parseInt(e.target.value);
            if (isPlaying) {{
                stopPlayback();
                startPlayback();
            }}
        }};

        // Progress bar click to seek
        document.getElementById('progress-bar').onclick = (e) => {{
            const rect = e.currentTarget.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            currentIndex = Math.round(pct * (baseData.length - 1));
            updateDisplay();
        }};

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            // Ignore if focused on an input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

            switch(e.key) {{
                case 'ArrowLeft':
                    document.getElementById('prev-btn').click();
                    break;
                case 'ArrowRight':
                    document.getElementById('next-btn').click();
                    break;
                case ' ':  // Space
                    e.preventDefault();
                    togglePlayback();
                    break;
                case 'Home':
                    document.getElementById('rewind-btn').click();
                    break;
                case 'End':
                    document.getElementById('end-btn').click();
                    break;
                case 'h':
                case 'H':
                    document.getElementById('toggle-human').click();
                    break;
                case 'a':
                case 'A':
                    document.getElementById('toggle-predicted').click();
                    break;
            }}
        }});
    }}

    // Transcript/audio sync variables
    let transcriptSegments = [];
    let audioElement = null;
    let lastActiveSegmentIndex = -1;
    let autoScrollTranscript = true;

    async function loadTranscript() {{
        // Try to load transcript.json
        try {{
            const response = await fetch('transcript.json?t=' + Date.now());
            if (response.ok) {{
                const data = await response.json();
                if (data.segments && data.segments.length > 0) {{
                    transcriptSegments = data.segments;
                    renderTranscript();
                    setupAudioSync();
                    return;
                }}
            }}
        }} catch (e) {{
            console.log('No transcript.json found');
        }}

        // Check if any base data has transcript info
        const hasTranscript = baseData.some(d => d.transcript_text || d.audio_start !== undefined);
        if (!hasTranscript) {{
            document.getElementById('transcript-content').innerHTML = '<div class="transcript-empty">No transcript available</div>';
            return;
        }}

        // Build segments from base data
        baseData.forEach((step, i) => {{
            if (step.transcript_text) {{
                transcriptSegments.push({{
                    start: step.audio_start || step.time || 0,
                    end: step.audio_end || (baseData[i + 1]?.time || step.time + 5),
                    text: step.transcript_text,
                    stepIndex: i
                }});
            }}
        }});

        if (transcriptSegments.length > 0) {{
            renderTranscript();
            setupAudioSync();
        }} else {{
            document.getElementById('transcript-content').innerHTML = '<div class="transcript-empty">No transcript available</div>';
        }}
    }}

    function renderTranscript() {{
        const container = document.getElementById('transcript-content');
        if (transcriptSegments.length === 0) {{
            container.innerHTML = '<div class="transcript-empty">No transcript available</div>';
            return;
        }}

        container.innerHTML = transcriptSegments.map((seg, i) => {{
            const timeStr = formatTime(seg.start);
            return `<span class="transcript-segment" data-index="${{i}}" data-start="${{seg.start}}" data-end="${{seg.end}}">` +
                   `<span class="transcript-time">${{timeStr}}</span>${{seg.text}} </span>`;
        }}).join('');

        // Add click handlers for seek
        container.querySelectorAll('.transcript-segment').forEach(el => {{
            el.onclick = () => {{
                const start = parseFloat(el.dataset.start);
                seekAudio(start);

                // Also jump to corresponding step if available
                const segIndex = parseInt(el.dataset.index);
                if (transcriptSegments[segIndex]?.stepIndex !== undefined) {{
                    currentIndex = transcriptSegments[segIndex].stepIndex;
                    updateDisplay();
                }}
            }};
        }});
    }}

    function formatTime(seconds) {{
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${{mins}}:${{secs.toString().padStart(2, '0')}}`;
    }}

    function seekAudio(time) {{
        if (!audioElement) {{
            audioElement = document.getElementById('audio');
        }}
        if (audioElement && audioElement.src) {{
            audioElement.currentTime = time;
            if (audioElement.paused) {{
                audioElement.play().catch(e => console.log('Audio play failed:', e));
            }}
        }}
    }}

    function setupAudioSync() {{
        audioElement = document.getElementById('audio');

        // Try to load audio file
        const audioSrc = 'audio.mp3';
        audioElement.src = audioSrc;
        audioElement.load();

        // Auto-highlight during playback
        audioElement.ontimeupdate = () => {{
            const currentTime = audioElement.currentTime;
            highlightCurrentSegment(currentTime);
        }};

        audioElement.onerror = () => {{
            console.log('Audio file not available');
        }};

        // Setup follow toggle button
        const followBtn = document.getElementById('transcript-follow-btn');
        if (followBtn) {{
            followBtn.onclick = () => {{
                autoScrollTranscript = !autoScrollTranscript;
                followBtn.classList.toggle('active', autoScrollTranscript);
            }};
        }}
    }}

    function highlightCurrentSegment(currentTime) {{
        const segments = document.querySelectorAll('.transcript-segment');
        let newActiveIndex = -1;

        segments.forEach((el, i) => {{
            const start = parseFloat(el.dataset.start);
            const end = parseFloat(el.dataset.end);
            const isActive = currentTime >= start && currentTime < end;
            el.classList.toggle('active', isActive);

            if (isActive) {{
                newActiveIndex = i;
            }}
        }});

        // Only scroll when active segment changes (not on every timeupdate)
        if (newActiveIndex !== lastActiveSegmentIndex && newActiveIndex !== -1) {{
            lastActiveSegmentIndex = newActiveIndex;
            if (autoScrollTranscript) {{
                segments[newActiveIndex].scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
            }}
        }}
    }}

    // Initialize
    initDropdowns();
    buildStepList();
    setupOverlayToggles();
    setupPlaybackControls();
    setupCopyButton();
    setupCopyAllButton();
    updateMetrics();
    updateDisplay();
    loadAndDisplayCosts();
    loadTranscript();  // Load transcript and setup audio sync
    </script>
</body>
</html>'''

    output_path.write_text(html, encoding='utf-8')
    print(f"Generated unified viewer: {output_path}")


def _enhance_comparison_to_unified_viewer(
    base_html_file: Path,
    predictions_by_checkpoint: dict[str, list[dict]],
    output_path: Path,
    capture_id: str = "unknown",
    goal: str = "Untitled",
) -> None:
    """Enhance an existing comparison HTML file into a unified viewer.

    DEPRECATED: This function uses script injection which is fragile.
    Use _generate_unified_viewer_from_extracted_data() instead for a
    standalone viewer that doesn't depend on the comparison.html structure.

    Takes the nice openadapt-capture viewer and adds:
    - Simplified nav (Training + Viewer only)
    - Checkpoint dropdown to switch between predictions
    - Training example dropdown (stub for future)
    """
    import re

    html = base_html_file.read_text()

    # Extract base data from the existing comparisonData
    data_match = re.search(
        r'const\s+comparisonData\s*=\s*(\[.*?\]);',
        html,
        re.DOTALL
    )
    if not data_match:
        print(f"Could not find comparisonData in {base_html_file}")
        return

    base_comparison_data = json.loads(data_match.group(1))

    # Build base data (human actions only) and ensure predictions dict has base data
    base_data = []
    for item in base_comparison_data:
        base_data.append({
            "index": item.get("index", 0),
            "time": item.get("time", 0),
            "image_path": item.get("image_path", ""),
            "human_action": item.get("human_action", {}),
        })

    # JSON encode predictions
    predictions_json = json.dumps(predictions_by_checkpoint)
    captures_json = json.dumps([{
        "id": capture_id,
        "name": goal,
        "steps": len(base_data),
    }])

    # 1. Replace nav bar with unified header combining nav + controls
    # Use shared header CSS and HTML for consistency with training dashboard
    header_css = f'<style>{_get_shared_header_css()}</style>'

    # Build the controls HTML for the viewer (example + checkpoint dropdowns)
    controls_html = f'''
            <div class="control-group">
                <span class="control-label">Example</span>
                <select id="capture-select">
                    <option value="{capture_id}">{goal[:40]}{'...' if len(goal) > 40 else ''} ({len(base_data)})</option>
                </select>
            </div>
            <div class="control-group">
                <span class="control-label">Checkpoint</span>
                <select id="checkpoint-select"></select>
            </div>
    '''

    unified_header = header_css + _generate_shared_header_html(
        "viewer",
        controls_html=controls_html,
        meta_html=f"ID: {capture_id}"
    )

    # Remove any old viewer-controls div if it exists (from previous runs)
    html = re.sub(
        r'<div class="viewer-controls"[^>]*>.*?</div>\s*(?=<)',
        '',
        html,
        flags=re.DOTALL
    )

    # Try to replace existing nav with unified header
    nav_replaced = False
    if re.search(r'<nav class="nav-bar"', html):
        html = re.sub(
            r'<nav class="nav-bar"[^>]*>.*?</nav>\s*',
            unified_header,
            html,
            flags=re.DOTALL
        )
        nav_replaced = True

    # Remove the old <header> element - unified header already contains all info
    html = re.sub(
        r'<header[^>]*>.*?</header>\s*',
        '',
        html,
        flags=re.DOTALL
    )

    # If no nav was found/replaced, insert unified header after <body>
    if not nav_replaced:
        html = re.sub(
            r'(<body[^>]*>)',
            r'\1\n' + unified_header,
            html,
            count=1
        )

    # 3. Replace the comparisonData with multi-checkpoint system
    # We need to modify the JavaScript to use our checkpoint system

    checkpoint_script = f'''
    <script>
    // Unified viewer: multi-checkpoint support
    // Bridge local comparisonData to window scope for cross-script access
    if (typeof comparisonData !== 'undefined' && typeof window.comparisonData === 'undefined') {{
        window.comparisonData = comparisonData;
    }}

    // Parse model output for SoM actions
    window.parseModelOutput = function(rawOutput) {{
        if (!rawOutput) return {{ html: '<em style="color:var(--text-muted);">No prediction</em>' }};

        // Try to extract SoM actions: CLICK([N]), TYPE([N], "text"), TYPE("text")
        const clickSomMatch = rawOutput.match(/CLICK\\s*\\(\\s*\\[\\s*(\\d+)\\s*\\]\\s*\\)/);
        const typeSomMatch = rawOutput.match(/TYPE\\s*\\(\\s*\\[\\s*(\\d+)\\s*\\]\\s*,\\s*["']([^"']*)["']\\s*\\)/);
        const typeSimpleMatch = rawOutput.match(/TYPE\\s*\\(\\s*["']([^"']*)["']\\s*\\)/);
        const clickCoordMatch = rawOutput.match(/CLICK\\s*\\(\\s*x\\s*=\\s*([\\d.]+)\\s*,\\s*y\\s*=\\s*([\\d.]+)\\s*\\)/);

        let html = '';

        if (clickSomMatch) {{
            html = `<div style="font-weight:600;color:#00d4aa;">CLICK([${{clickSomMatch[1]}}])</div>`;
        }} else if (typeSomMatch) {{
            html = `<div style="font-weight:600;color:#00d4aa;">TYPE([${{typeSomMatch[1]}}], "${{typeSomMatch[2]}}")</div>`;
        }} else if (typeSimpleMatch) {{
            html = `<div style="font-weight:600;color:#00d4aa;">TYPE("${{typeSimpleMatch[1]}}")</div>`;
        }} else if (clickCoordMatch) {{
            html = `<div style="font-weight:600;color:#00d4aa;">CLICK(x=${{clickCoordMatch[1]}}, y=${{clickCoordMatch[2]}})</div>`;
        }} else {{
            // No structured action - show truncated output
            const truncated = rawOutput.replace(/\\n/g, ' ').substring(0, 150);
            html = `<div style="font-size:0.85rem;color:var(--text-muted);max-height:60px;overflow:hidden;">${{truncated}}${{rawOutput.length > 150 ? '...' : ''}}</div>`;
        }}

        return {{ html }};
    }};

    // Override prediction display in comparison viewer
    window.formatPrediction = function(pred) {{
        if (!pred) return '<em style="color:var(--text-muted);">No prediction</em>';
        if (pred.x !== undefined) {{
            return `<div>Type: ${{pred.type || 'click'}}</div><div>Position: (${{(pred.x * 100).toFixed(1)}}%, ${{(pred.y * 100).toFixed(1)}}%)</div>`;
        }}
        return window.parseModelOutput(pred.raw_output || JSON.stringify(pred)).html;
    }};

    // Use window. prefix for cross-script variable access
    window.predictionsByCheckpoint = {predictions_json};
    window.availableCaptures = {captures_json};
    window.currentCheckpoint = 'None';

    // Initialize checkpoint dropdown
    window.initCheckpointDropdown = function() {{
        const select = document.getElementById('checkpoint-select');
        if (!select) return;

        const checkpointNames = Object.keys(window.predictionsByCheckpoint);
        checkpointNames.sort((a, b) => {{
            if (a === 'None') return -1;
            if (b === 'None') return 1;
            const aNum = parseInt(a.match(/\\d+/)?.[0] || '999');
            const bNum = parseInt(b.match(/\\d+/)?.[0] || '999');
            return aNum - bNum;
        }});

        select.innerHTML = '';
        checkpointNames.forEach(name => {{
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name === 'None' ? 'None (Capture Only)' : name;
            select.appendChild(opt);
        }});

        // Default to latest epoch checkpoint (prefer actual trained checkpoints over Preview)
        const epochCheckpoints = checkpointNames.filter(n => n.toLowerCase().includes('epoch'));
        const latestCheckpoint = epochCheckpoints.length > 0
            ? epochCheckpoints.pop()
            : checkpointNames.filter(n => n !== 'None').pop();
        if (latestCheckpoint) {{
            select.value = latestCheckpoint;
            window.currentCheckpoint = latestCheckpoint;
            window.applyCheckpointPredictions(latestCheckpoint);
        }}

        select.addEventListener('change', (e) => {{
            window.currentCheckpoint = e.target.value;
            window.applyCheckpointPredictions(window.currentCheckpoint);
        }});
    }};

    // Apply predictions from selected checkpoint to comparisonData
    window.applyCheckpointPredictions = function(checkpointName) {{
        const predictions = window.predictionsByCheckpoint[checkpointName] || [];

        // Update comparisonData with new predictions (access from window)
        if (typeof window.comparisonData !== 'undefined') {{
            window.comparisonData.forEach((item, i) => {{
                const pred = predictions[i] || {{}};
                item.predicted_action = pred.predicted_action || null;
                item.match = pred.match !== undefined ? pred.match : null;
            }});
        }}

        // Refresh display if updateComparison exists (check both window and global scope)
        const idx = typeof window.currentIndex !== 'undefined' ? window.currentIndex :
                    (typeof currentIndex !== 'undefined' ? currentIndex : 0);
        if (typeof window.updateComparison === 'function') {{
            window.updateComparison(idx);
        }} else if (typeof updateComparison === 'function') {{
            updateComparison(idx);
        }}

        // Reformat prediction display after original updateComparison runs
        setTimeout(() => {{
            const predEl = document.getElementById('predicted-action') ||
                          document.querySelector('.action-box.predicted .action-details');
            if (predEl && window.comparisonData && window.comparisonData[idx]) {{
                const pred = window.comparisonData[idx].predicted_action;
                if (pred) {{
                    predEl.innerHTML = window.formatPrediction(pred);
                }}
            }}
        }}, 50);

        // Update metrics if setupMetricsSummary exists
        if (typeof window.setupMetricsSummary === 'function') {{
            window.setupMetricsSummary();
        }}
    }};

    // Initialize on load
    setTimeout(window.initCheckpointDropdown, 200);

    // Smart auto-scroll: scroll while playing, but stop if user scrolls up
    (function() {{
        let autoScrollEnabled = true;
        let lastScrollTop = 0;

        // Find the events list element
        const eventsList = document.querySelector('.events-list');
        if (!eventsList) return;

        // Detect user scroll - disable auto-scroll if scrolling up
        eventsList.addEventListener('scroll', function() {{
            const currentScrollTop = eventsList.scrollTop;

            // If user scrolled up, disable auto-scroll
            if (currentScrollTop < lastScrollTop - 10) {{
                autoScrollEnabled = false;
            }}

            // If user scrolled to bottom (within 50px), re-enable auto-scroll
            const isAtBottom = eventsList.scrollHeight - eventsList.scrollTop - eventsList.clientHeight < 50;
            if (isAtBottom) {{
                autoScrollEnabled = true;
            }}

            lastScrollTop = currentScrollTop;
        }});

        // Override scrollIntoView behavior for event items
        const originalScrollIntoView = Element.prototype.scrollIntoView;
        Element.prototype.scrollIntoView = function(options) {{
            // Only block scroll for event items when auto-scroll is disabled
            if (!autoScrollEnabled && this.classList && this.classList.contains('event-item')) {{
                return; // Skip scrollIntoView when user has scrolled up
            }}
            return originalScrollIntoView.call(this, options);
        }};

        // Add scroll lock indicator
        const indicator = document.createElement('div');
        indicator.id = 'scroll-lock-indicator';
        indicator.style.cssText = 'position:fixed;bottom:20px;right:20px;padding:8px 12px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:4px;font-size:0.75rem;color:var(--text-muted);opacity:0;transition:opacity 0.3s;pointer-events:none;z-index:1000;';
        indicator.textContent = '⏸ Auto-scroll paused (scroll to bottom to resume)';
        document.body.appendChild(indicator);

        // Show/hide indicator based on scroll state
        setInterval(() => {{
            indicator.style.opacity = autoScrollEnabled ? '0' : '1';
        }}, 200);
    }})();
    </script>
    '''

    # Insert checkpoint script before </body>
    html = html.replace('</body>', checkpoint_script + '</body>')

    # 4. Disable the old discoverDashboards that creates wrong nav
    html = html.replace(
        'discoverDashboards();',
        '// discoverDashboards disabled - using unified viewer nav'
    )

    # Write output
    output_path.write_text(html, encoding='utf-8')
    print(f"Generated unified viewer from {base_html_file.name}: {output_path}")


def _add_static_nav_to_comparison(
    comparison_path: Path,
    output_dir: Path,
    nav_links: list[tuple[str, str]] | None = None,
) -> None:
    """Add or update static navigation in a comparison HTML file.

    Also moves the Action Comparison panel to main-content (above screenshot) if needed.

    Args:
        comparison_path: Path to the comparison HTML file
        output_dir: Directory containing all dashboard files
        nav_links: Pre-built list of (filename, label) tuples for consistency
    """
    import re

    html = comparison_path.read_text()

    # Move comparison panel to be a full-width sibling BEFORE main-content (not inside it)
    if '<div class="comparison-panel"' in html:
        # Check if panel is NOT already right before main-content
        if '<div class="comparison-panel"' in html and 'class="comparison-panel"' in html:
            # Check if it's in the wrong place (inside sidebar or main-content)
            in_sidebar = '<div class="sidebar">' in html and html.index('<div class="comparison-panel"') > html.index('<div class="sidebar">')
            in_main = '<div class="main-content">' in html and '<div class="main-content">\n' in html and '<div class="main-content">\n        <div class="comparison-panel"' in html

            if in_sidebar or in_main:
                # Extract comparison panel from wherever it is
                panel_match = re.search(
                    r'(\s*<div class="comparison-panel"[^>]*>.*?</div>\s*</div>\s*</div>)',
                    html,
                    re.DOTALL
                )
                if panel_match:
                    panel_html = panel_match.group(1)
                    # Remove from current location
                    html = html.replace(panel_html, '')
                    # Insert as sibling BEFORE main-content
                    html = html.replace(
                        '<div class="main-content">',
                        panel_html.strip() + '\n        <div class="main-content">'
                    )
                    print(f"  Moved Action Comparison above screenshot in {comparison_path.name}")

    # Build nav links if not provided
    if nav_links is None:
        nav_links = _build_nav_links()

    # Build nav HTML with active state for current file
    # NOTE: No "Dashboards:" label to match training dashboard nav
    current_file = comparison_path.name
    nav_html = '''
    <nav class="nav-bar" style="display:flex;gap:8px;padding:12px 16px;background:#12121a;border:1px solid rgba(255,255,255,0.06);border-radius:8px;margin-bottom:16px;flex-wrap:wrap;">
'''
    for filename, label in nav_links:
        is_active = filename == current_file
        active_style = "background:#00d4aa;color:#0a0a0f;border-color:#00d4aa;font-weight:600;" if is_active else ""
        nav_html += f'        <a href="{filename}" style="padding:8px 16px;border-radius:6px;font-size:0.8rem;text-decoration:none;color:#888;background:#1a1a24;border:1px solid rgba(255,255,255,0.06);{active_style}">{label}</a>\n'
    nav_html += '    </nav>\n'

    # ALWAYS replace existing nav or add new one (for consistency)
    if '<nav class="nav-bar"' in html:
        # Replace existing nav
        html = re.sub(
            r'<nav class="nav-bar"[^>]*>.*?</nav>\s*',
            nav_html,
            html,
            flags=re.DOTALL
        )
        print(f"  Updated navigation in {comparison_path.name}")
    elif '<div class="container">' in html:
        html = html.replace(
            '<div class="container">',
            '<div class="container">\n' + nav_html
        )
        print(f"  Added navigation to {comparison_path.name}")
    elif '<body>' in html:
        html = html.replace('<body>', '<body>\n' + nav_html)
        print(f"  Added navigation to {comparison_path.name}")

    comparison_path.write_text(html)


def regenerate_local_dashboard(
    output_dir: str | Path,
    capture_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    nav_links: list[tuple[str, str]] | None = None,
) -> Path:
    """Regenerate dashboard.html with correct local paths and static navigation.

    This should be called after downloading training results from a remote instance.
    It fixes:
    - Training status (COMPLETED/STOPPED instead of always LIVE)
    - Navigation links to sibling dashboards (comparison, viewer)
    - Local capture path for comparison preview

    Args:
        output_dir: Directory containing training_log.json and dashboard files
        capture_path: Local path to capture directory (for comparison preview)
        checkpoint_path: Local path to checkpoint directory
        nav_links: Pre-built list of (filename, label) tuples for consistency

    Returns:
        Path to generated dashboard.html
    """
    output_dir = Path(output_dir)
    log_file = output_dir / "training_log.json"

    if not log_file.exists():
        raise FileNotFoundError(f"No training_log.json found in {output_dir}")

    # Load training state from log
    with open(log_file) as f:
        data = json.load(f)

    # Create state from log data
    state = TrainingState(
        job_id=data.get("job_id", "unknown"),
        hostname=data.get("hostname", ""),
        capture_path=str(capture_path) if capture_path else data.get("capture_path", ""),
        config_path=data.get("config_path", ""),
        epoch=data.get("epoch", 0),
        step=data.get("step", 0),
        loss=data.get("loss", 0),
        learning_rate=data.get("learning_rate", 0),
        total_epochs=data.get("total_epochs", 5),
        instance_type=data.get("instance_type", ""),
        instance_ip=data.get("instance_ip", ""),
        elapsed_time=data.get("elapsed_time", 0.0),
        cloud_provider=data.get("cloud_provider", ""),
    )
    state.losses = data.get("losses", [])
    state.evaluations = data.get("evaluations", [])

    # Determine training status
    total_epochs = data.get("total_epochs", 5)
    current_epoch = data.get("epoch", 0)

    if current_epoch + 1 >= total_epochs:
        training_status = "COMPLETED"
    elif len(state.losses) > 0:
        training_status = "STOPPED"
    else:
        training_status = "NOT_STARTED"

    # Use provided nav_links or build them
    if nav_links is None:
        nav_links = _build_nav_links()

    # Create config
    config = TrainingConfig(
        num_train_epochs=total_epochs,
        learning_rate=data.get("learning_rate", 5e-5),
    )

    # Generate dashboard HTML with modifications
    html = generate_training_dashboard(state, config)

    # Replace dynamic status with static status
    if training_status == "COMPLETED":
        html = html.replace(
            '<div class="status" id="status">',
            '<div class="status complete" id="status">'
        )
        html = html.replace(
            '<span id="status-text">Training in progress</span>',
            '<span id="status-text">COMPLETED</span>'
        )
    elif training_status == "STOPPED":
        html = html.replace(
            '<div class="status" id="status">',
            '<div class="status stale" id="status">'
        )
        html = html.replace(
            '<span id="status-text">Training in progress</span>',
            '<span id="status-text">STOPPED (Epoch {}/{})'.format(current_epoch + 1, total_epochs) + '</span>'
        )

    # Fix ETA display for completed/stopped training
    import re
    if training_status in ("COMPLETED", "STOPPED"):
        # Replace "calculating..." with appropriate status
        html = re.sub(
            r'(<div class="stat-value" id="stat-eta">)[^<]*(</div>)',
            r'\1—\2' if training_status == "STOPPED" else r'\1complete\2',
            html
        )

    # Replace dynamic nav with static unified header
    # The dashboard now uses the shared unified-header, so we just need to ensure
    # the header HTML is present (it's already generated by generate_training_dashboard)

    # Disable the JS polling and dynamic discovery (training is done, no need to fetch updates)
    # This is critical for file:// protocol where fetch() doesn't work
    html = html.replace(
        "setInterval(fetchAndUpdate, 3000);",
        "// fetchAndUpdate disabled for static dashboard"
    )
    html = html.replace(
        "setInterval(updateElapsedDisplay, 1000);",
        "// updateElapsedDisplay disabled for static dashboard"
    )
    html = html.replace(
        "setInterval(updateStatusIndicator, 1000);",
        "// updateStatusIndicator disabled for static dashboard"
    )
    # CRITICAL: Disable discoverDashboards() - it overwrites static nav on file:// protocol
    html = html.replace(
        "discoverDashboards();",
        "// discoverDashboards disabled - using static nav for file:// protocol"
    )

    # Write output
    dashboard_path = output_dir / "dashboard.html"
    dashboard_path.write_text(html)
    print(f"Regenerated dashboard: {dashboard_path}")

    return dashboard_path


def run_epoch_evaluation(
    adapter: BaseVLMAdapter,
    episode: Episode,
    epoch: int,
    config: TrainingConfig,
    logger: "TrainingLogger",
    sample_indices: Optional[List[int]] = None,
) -> Path:
    """Run inference evaluation on sample steps after an epoch.

    This generates a comparison_epoch{N}.html file showing human vs predicted actions.

    Args:
        adapter: Trained adapter to use for inference
        episode: Episode with steps to evaluate
        epoch: Current epoch number
        config: Training configuration
        logger: Training logger for state tracking
        sample_indices: Specific step indices to evaluate (default: evenly spaced)

    Returns:
        Path to generated comparison HTML file
    """
    from openadapt_ml.scripts.compare import generate_comparison_html, predict_action, format_action

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select sample indices if not provided
    num_samples = min(config.eval_samples, len(episode.steps))
    if sample_indices is None:
        if num_samples >= len(episode.steps):
            sample_indices = list(range(len(episode.steps)))
        else:
            # Evenly space samples across the episode
            step_size = len(episode.steps) // num_samples
            sample_indices = [i * step_size for i in range(num_samples)]

    print(f"  Running inference on {len(sample_indices)} sample steps...")

    # Switch adapter to eval mode
    adapter.eval()

    comparison_data = []
    action_history: List[str] = []
    total_steps = len(episode.steps)

    for i, step in enumerate(episode.steps):
        step_data = {
            "index": i,
            "time": step.t,
            "image_path": step.observation.image_path,
            "human_action": {
                "type": step.action.type,
                "x": step.action.x,
                "y": step.action.y,
                "text": step.action.text,
            },
            "predicted_action": None,
            "match": None,
        }

        # Only run inference on selected samples (for speed)
        if i in sample_indices and step.observation.image_path:
            try:
                predicted = predict_action(
                    adapter,
                    step.observation.image_path,
                    episode.goal,
                    step_index=i,
                    total_steps=total_steps,
                    action_history=action_history.copy(),
                )
                step_data["predicted_action"] = predicted

                # Check match and calculate distance
                if predicted and predicted.get("type") == step.action.type:
                    step_data["match"] = True

                    # Calculate distance for click actions
                    if step.action.type == "click":
                        hx, hy = step.action.x or 0, step.action.y or 0
                        px, py = predicted.get("x", 0), predicted.get("y", 0)
                        distance = ((hx - px) ** 2 + (hy - py) ** 2) ** 0.5

                        # Log evaluation to training state
                        logger.state.log_evaluation(
                            epoch=epoch,
                            sample_idx=i,
                            image_path=step.observation.image_path,
                            human_action=step_data["human_action"],
                            predicted_action=predicted,
                        )
                else:
                    step_data["match"] = False

                print(f"    Step {i}: {step.action.type} -> {predicted.get('type') if predicted else 'none'}")

            except Exception as e:
                print(f"    Step {i}: inference failed - {e}")

        # Build action history for context
        action_history.append(format_action(step.action, use_som=False))
        comparison_data.append(step_data)

    # Switch back to train mode
    adapter.train()

    # Generate comparison HTML
    output_path = output_dir / f"comparison_epoch{epoch}.html"
    capture_path = Path(logger.state.capture_path) if logger.state.capture_path else Path(".")

    generate_comparison_html(capture_path, episode, comparison_data, output_path)
    print(f"  Comparison saved: {output_path}")

    # Also regenerate all dashboards to update navigation
    regenerate_all_dashboards(output_dir)

    return output_path


def _create_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    # Use an identity collate_fn so that each batch is a List[Dict], matching
    # the expectations of adapters that operate on SFT-style samples.
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)


def train_supervised(
    adapter: BaseVLMAdapter,
    dataset: Dataset,
    config: TrainingConfig,
    optimizer: Optional[Optimizer] = None,
    logger: Optional[TrainingLogger] = None,
    episode: Optional[Episode] = None,
) -> bool:
    """Minimal supervised training loop skeleton.

    This assumes that `adapter.prepare_inputs` and `adapter.compute_loss` are
    implemented. It will raise if those methods are not implemented.

    Args:
        adapter: VLM adapter to train.
        dataset: Training dataset.
        config: Training configuration.
        optimizer: Optional optimizer (default: AdamW).
        logger: Optional training logger for visualization.
        episode: Optional episode for periodic evaluation (generates comparison_epoch{N}.html).

    Returns:
        True if training completed successfully, False if aborted due to NaN/Inf loss.
    """

    device = adapter.device  # type: ignore[attr-defined]
    dataloader = _create_dataloader(dataset, batch_size=config.per_device_train_batch_size)

    if optimizer is None:
        optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=config.learning_rate)  # type: ignore[arg-type]

    # Create logger if not provided
    if logger is None:
        logger = TrainingLogger(config.output_dir, config)

    total_steps = 0
    adapter.train()

    # Early stopping tracking
    consecutive_low_loss = 0
    early_stopped = False
    user_stopped = False

    for epoch in range(config.num_train_epochs):
        if early_stopped or user_stopped:
            break

        for _, batch in enumerate(dataloader):
            # Check for stop signal from dashboard
            stop_file = Path(config.output_dir) / "STOP_TRAINING"
            if stop_file.exists():
                msg = "Stop signal received from dashboard. Stopping training..."
                print(msg)
                logger._log_to_terminal(msg)
                user_stopped = True
                stop_file.unlink()  # Remove signal file
                break

            # Batch is a List[Dict[str, Any]] of SFT-style samples; adapter is
            # responsible for converting it into model inputs.
            samples: List[Dict[str, Any]] = batch

            inputs = adapter.prepare_inputs(samples)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            loss = adapter.compute_loss(inputs)

            # Guard against invalid losses to avoid propagating NaNs/Infs
            if torch.isnan(loss) or torch.isinf(loss):
                msg = f"Encountered invalid loss at epoch={epoch} step={total_steps + 1}: {loss.item()}"
                print(msg)
                logger._log_to_terminal(msg)
                logger.on_train_end()
                return False

            loss.backward()

            if (total_steps + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), config.max_grad_norm)  # type: ignore[arg-type]
                optimizer.step()
                optimizer.zero_grad()

            total_steps += 1
            loss_val = loss.item()

            # Log step
            logger.on_step(epoch, total_steps, loss_val, config.learning_rate)

            if config.logging_steps and total_steps % config.logging_steps == 0:
                msg = f"epoch={epoch} step={total_steps} loss={loss_val:.4f}"
                print(msg)
                logger._log_to_terminal(msg)

            # Early stopping check
            if loss_val < config.early_stop_loss:
                consecutive_low_loss += 1
                if consecutive_low_loss >= config.early_stop_patience:
                    msg = (
                        f"Early stopping: loss ({loss_val:.6f}) below threshold "
                        f"({config.early_stop_loss}) for {config.early_stop_patience} consecutive steps"
                    )
                    print(msg)
                    logger._log_to_terminal(msg)
                    early_stopped = True
                    break
            else:
                consecutive_low_loss = 0

        # End of epoch
        logger.on_epoch_end(epoch)

        # Save checkpoint at end of each epoch
        if config.save_checkpoint_every_epoch:
            checkpoint_path = Path(config.checkpoint_dir) / f"epoch_{epoch}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            try:
                adapter.save_checkpoint(str(checkpoint_path))
                msg = f"Checkpoint saved to {checkpoint_path}"
                print(msg)
                logger._log_to_terminal(msg)
            except Exception as e:
                msg = f"Warning: Failed to save checkpoint: {e}"
                print(msg)
                logger._log_to_terminal(msg)

        # Run evaluation after each epoch (generates comparison_epoch{N}.html)
        if config.eval_every_epoch and episode is not None:
            try:
                print(f"Running epoch {epoch} evaluation...")
                run_epoch_evaluation(
                    adapter=adapter,
                    episode=episode,
                    epoch=epoch,
                    config=config,
                    logger=logger,
                )
            except Exception as e:
                print(f"Warning: Epoch evaluation failed: {e}")
                import traceback
                traceback.print_exc()

    logger.on_train_end()
    return True
