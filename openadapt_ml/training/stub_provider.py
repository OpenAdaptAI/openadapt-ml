"""Stub training provider for rapid UI testing without actual training."""

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path


class StubTrainingProvider:
    """Simulates training without actual computation.

    Use this to test dashboard, viewer, stop button, etc. without
    waiting for real training on GPU or Lambda.
    """

    def __init__(
        self,
        output_dir: Path,
        epochs: int = 5,
        steps_per_epoch: int = 10,
        step_delay: float = 0.5,
        early_stop_loss: float = 0.0,
        early_stop_patience: int = 3,
    ):
        """Initialize stub provider.

        Args:
            output_dir: Directory to write training_log.json
            epochs: Number of epochs to simulate
            steps_per_epoch: Steps per epoch
            step_delay: Delay between steps in seconds (for realistic feel)
            early_stop_loss: Stop if loss drops below this threshold
            early_stop_patience: Number of consecutive steps below threshold before stopping
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.step_delay = step_delay
        self.early_stop_loss = early_stop_loss
        self.early_stop_patience = early_stop_patience

        self.current_epoch = 0
        self.current_step = 0
        self.losses = []
        self.evaluations = []
        self.start_time = time.time()
        self.job_id = f"stub_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.consecutive_low_loss = 0
        self.termination_status = None
        self.termination_message = None

        # Set up logging to file
        self.log_file = self.output_dir / "training.log"
        self.log_handle = None

    def _log(self, message: str, to_stdout: bool = True):
        """Write message to both log file and stdout.

        Args:
            message: Message to log
            to_stdout: If True, also print to stdout (default: True)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"

        # Write to file
        if self.log_handle is None:
            self.log_handle = open(self.log_file, "w", buffering=1)  # Line buffered

        self.log_handle.write(log_line + "\n")
        self.log_handle.flush()

        # Print to stdout
        if to_stdout:
            print(message)

    def simulate_step(self) -> dict:
        """Simulate one training step.

        Returns:
            Current training status dict
        """
        # Generate decreasing loss with noise
        progress = self.current_step / (self.epochs * self.steps_per_epoch)
        base_loss = 2.5 * (1 - progress * 0.8)  # Decrease from 2.5 to ~0.5
        noise = random.uniform(-0.15, 0.15)
        loss = max(0.1, base_loss + noise)

        elapsed = time.time() - self.start_time

        self.losses.append({
            "epoch": self.current_epoch,
            "step": self.current_step + 1,
            "loss": loss,
            "lr": 5e-5,
            "time": elapsed,
        })

        self.current_step += 1

        # Check for epoch completion
        if self.current_step % self.steps_per_epoch == 0:
            self._generate_epoch_evaluation()
            self.current_epoch += 1
            # Cap at max epochs for display
            if self.current_epoch > self.epochs:
                self.current_epoch = self.epochs

        return self.get_status()

    def _generate_epoch_evaluation(self):
        """Generate fake evaluation for completed epoch."""
        # Improve accuracy as training progresses
        progress = self.current_epoch / self.epochs
        accuracy_boost = progress * 0.3  # Up to 30% improvement

        # Use real screenshot if available, otherwise placeholder
        sample_path = self.output_dir / "screenshots" / "sample.png"
        if not sample_path.exists():
            # Try to copy from common capture location
            import shutil
            capture_screenshots = Path.home() / "oa/src/openadapt-capture/turn-off-nightshift/screenshots"
            if capture_screenshots.exists():
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                for img in capture_screenshots.glob("*.png"):
                    shutil.copy(img, sample_path)
                    break  # Just copy the first one

        self.evaluations.append({
            "epoch": self.current_epoch,
            "sample_idx": 7,  # Match the real training sample
            "image_path": "screenshots/sample.png",
            "human_action": {
                "type": "click",
                "x": 0.65,
                "y": 0.65,
                "text": None,
            },
            "predicted_action": {
                "type": "click",
                "x": 0.65 + random.uniform(-0.15, 0.15) * (1 - accuracy_boost),
                "y": 0.65 + random.uniform(-0.15, 0.15) * (1 - accuracy_boost),
                "raw_output": f"Thought: [Stub] Epoch {self.current_epoch} - analyzing screenshot to find target element. The model is learning to identify UI components.\nAction: CLICK(x=0.65, y=0.65)",
            },
            "distance": random.uniform(0.05, 0.2) * (1 - accuracy_boost),
            "correct": random.random() > (0.5 - accuracy_boost),
        })

    def get_status(self) -> dict:
        """Return current training status.

        Returns:
            Status dict compatible with training_log.json format
        """
        current_loss = self.losses[-1]["loss"] if self.losses else 0
        elapsed = time.time() - self.start_time

        # Determine status
        if self.termination_status:
            status = "completed" if self.termination_status == "auto_complete" else self.termination_status
        elif self.is_complete():
            status = "completed"
        else:
            status = "training"

        return {
            "job_id": self.job_id,
            "hostname": "stub-local",
            "capture_path": "/stub/capture",
            "config_path": "configs/stub.yaml",
            "instance_type": "stub",
            "instance_ip": "127.0.0.1",
            "started_at": datetime.fromtimestamp(self.start_time).isoformat() + "Z",
            "cloud_provider": "stub",
            "cloud_dashboard_url": "",
            "cloud_instance_id": "stub",
            "setup_status": "training",
            "setup_logs": ["[Stub] Simulated training in progress..."],
            "epoch": self.current_epoch,
            "step": self.current_step,
            "total_steps": self.epochs * self.steps_per_epoch,
            "total_epochs": self.epochs,
            "loss": current_loss,
            "learning_rate": 5e-5,
            "samples_seen": self.current_step,
            "elapsed_time": elapsed,
            "losses": self.losses,
            "evaluations": self.evaluations,
            "status": status,
            "termination_status": self.termination_status,
            "termination_message": self.termination_message,
        }

    def write_status(self):
        """Write current status to training_log.json."""
        log_path = self.output_dir / "training_log.json"
        log_path.write_text(json.dumps(self.get_status(), indent=2))

    def is_complete(self) -> bool:
        """Check if training simulation is complete."""
        return self.current_epoch >= self.epochs

    def check_stop_signal(self) -> bool:
        """Check if stop signal file exists."""
        stop_file = self.output_dir / "STOP_TRAINING"
        return stop_file.exists()

    def run(self, callback=None):
        """Run the full training simulation.

        Args:
            callback: Optional function called after each step with status dict
        """
        self._log(f"[Stub] Starting simulated training: {self.epochs} epochs, {self.steps_per_epoch} steps/epoch")
        self._log(f"[Stub] Output: {self.output_dir}")
        self._log(f"[Stub] Step delay: {self.step_delay}s (total ~{self.epochs * self.steps_per_epoch * self.step_delay:.0f}s)")
        if self.early_stop_loss > 0:
            self._log(f"[Stub] Early stop: loss < {self.early_stop_loss} for {self.early_stop_patience} steps")
        self._log("")

        while not self.is_complete():
            # Check for user stop signal
            if self.check_stop_signal():
                self._log("\n[Stub] Stop signal received from user!")
                (self.output_dir / "STOP_TRAINING").unlink(missing_ok=True)
                self.termination_status = "user_stop"
                self.termination_message = f"Stopped at epoch {self.current_epoch + 1}, step {self.current_step}"
                self.write_status()
                break

            status = self.simulate_step()

            # Check for early stop loss
            loss = status["loss"]
            if self.early_stop_loss > 0 and loss < self.early_stop_loss:
                self.consecutive_low_loss += 1
                if self.consecutive_low_loss >= self.early_stop_patience:
                    self._log(f"\n[Stub] Auto-stopped: loss ({loss:.4f}) < {self.early_stop_loss} for {self.early_stop_patience} steps")
                    self.termination_status = "auto_low_loss"
                    self.termination_message = f"Loss reached {loss:.4f} (< {self.early_stop_loss})"
                    self.write_status()
                    break
            else:
                self.consecutive_low_loss = 0

            self.write_status()

            # Progress output
            epoch = status["epoch"]
            step = status["step"]
            display_epoch = min(epoch + 1, self.epochs)  # Cap at max for display
            self._log(f"  Epoch {display_epoch}/{self.epochs} | Step {step} | Loss: {loss:.4f}")

            if callback:
                callback(status)

            time.sleep(self.step_delay)

        # Set completion status if not already set
        if self.termination_status is None:
            self.termination_status = "auto_complete"
            self.termination_message = f"Completed {self.epochs} epochs"
            self.write_status()

        self._log(f"\n[Stub] Training complete: {self.termination_status}")

        # Close log file
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None

        return self.get_status()
