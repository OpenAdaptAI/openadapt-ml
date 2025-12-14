"""Azure async inference queue for live training feedback.

This module implements Phase 2 of the live inference design:
- Training instance uploads checkpoints to Azure Blob Storage
- Triggers Azure Queue Storage message with checkpoint info
- Inference worker polls queue and runs inference
- Results uploaded to Blob Storage for dashboard to display

Architecture:
    Training GPU → Blob (checkpoints) → Queue (jobs) → Inference GPU → Blob (comparisons)

Usage:
    # Submit checkpoint for async inference (called by trainer)
    uv run python -m openadapt_ml.cloud.azure_inference inference-submit \
        --checkpoint checkpoints/epoch_1 \
        --capture /path/to/capture

    # Start inference worker (runs on separate instance)
    uv run python -m openadapt_ml.cloud.azure_inference inference-worker \
        --model Qwen/Qwen2.5-VL-3B

    # Watch for new comparison results
    uv run python -m openadapt_ml.cloud.azure_inference inference-watch
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceJob:
    """Inference job metadata."""

    job_id: str
    checkpoint_blob: str  # Path in blob storage
    checkpoint_epoch: int
    capture_path: str  # Path to capture data
    submitted_at: float
    status: str = "pending"  # pending, running, completed, failed
    output_blob: str | None = None  # Path to comparison HTML in blob
    error: str | None = None


class AzureInferenceQueue:
    """Manages async inference jobs via Azure Queue Storage.

    This class provides the core infrastructure for async inference during training:
    1. submit_checkpoint() - Upload checkpoint to blob, queue inference job
    2. poll_and_process() - Worker loop that processes jobs from queue
    3. watch_comparisons() - Poll for new comparison results

    Authentication uses existing AzureConfig pattern from benchmarks/azure.py.
    """

    def __init__(
        self,
        storage_connection_string: str | None = None,
        queue_name: str = "inference-jobs",
        checkpoints_container: str = "checkpoints",
        comparisons_container: str = "comparisons",
    ):
        """Initialize Azure inference queue.

        Args:
            storage_connection_string: Azure Storage connection string (from .env)
            queue_name: Queue name for inference jobs
            checkpoints_container: Blob container for checkpoints
            comparisons_container: Blob container for comparison results

        Raises:
            ImportError: If azure-storage-blob or azure-storage-queue not installed
            ValueError: If connection string not provided and not in settings
        """
        # Lazy import Azure SDK
        try:
            from azure.storage.blob import BlobServiceClient
            from azure.storage.queue import QueueClient

            self._BlobServiceClient = BlobServiceClient
            self._QueueClient = QueueClient
        except ImportError as e:
            raise ImportError(
                "Azure Storage SDK not installed. Install with: "
                "pip install azure-storage-blob azure-storage-queue"
            ) from e

        # Get connection string from settings if not provided
        if not storage_connection_string:
            from openadapt_ml.config import settings

            storage_connection_string = settings.azure_storage_connection_string
            if not storage_connection_string:
                raise ValueError(
                    "AZURE_STORAGE_CONNECTION_STRING not set. "
                    "Run 'python scripts/setup_azure.py' to configure Azure storage."
                )

        self.connection_string = storage_connection_string
        self.queue_name = queue_name
        self.checkpoints_container = checkpoints_container
        self.comparisons_container = comparisons_container

        # Initialize clients
        self.blob_service = self._BlobServiceClient.from_connection_string(
            storage_connection_string
        )
        self.queue_client = self._QueueClient.from_connection_string(
            storage_connection_string, queue_name
        )

        logger.info(f"Initialized Azure inference queue: {queue_name}")

    def submit_checkpoint(
        self, checkpoint_path: str | Path, capture_path: str | Path, epoch: int = 0
    ) -> InferenceJob:
        """Upload checkpoint and queue inference job.

        This is called by the trainer after saving a checkpoint.
        It uploads the checkpoint to blob storage and adds a message to the queue.

        Args:
            checkpoint_path: Local path to checkpoint directory
            capture_path: Path to capture data (for inference)
            epoch: Epoch number for this checkpoint

        Returns:
            InferenceJob with job metadata
        """
        checkpoint_path = Path(checkpoint_path)
        capture_path = Path(capture_path)

        # Generate unique job ID
        job_id = f"inference_{int(time.time())}_{epoch}"

        # Upload checkpoint to blob storage
        blob_name = f"checkpoints/epoch_{epoch}/{checkpoint_path.name}"
        logger.info(f"Uploading checkpoint to {blob_name}...")

        checkpoint_blob_client = self.blob_service.get_blob_client(
            container=self.checkpoints_container, blob=blob_name
        )

        # Upload all files in checkpoint directory
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(checkpoint_path)
                file_blob_name = f"checkpoints/epoch_{epoch}/{relative_path}"
                file_blob_client = self.blob_service.get_blob_client(
                    container=self.checkpoints_container, blob=file_blob_name
                )

                with open(file_path, "rb") as f:
                    file_blob_client.upload_blob(f, overwrite=True)
                    logger.debug(f"  Uploaded {file_blob_name}")

        # Create job metadata
        job = InferenceJob(
            job_id=job_id,
            checkpoint_blob=blob_name,
            checkpoint_epoch=epoch,
            capture_path=str(capture_path),
            submitted_at=time.time(),
        )

        # Queue inference job
        job_message = {
            "job_id": job.job_id,
            "checkpoint_blob": job.checkpoint_blob,
            "checkpoint_epoch": job.checkpoint_epoch,
            "capture_path": job.capture_path,
            "submitted_at": job.submitted_at,
        }

        self.queue_client.send_message(json.dumps(job_message))
        logger.info(f"Queued inference job: {job_id}")

        return job

    def poll_and_process(
        self,
        adapter,
        max_messages: int = 1,
        visibility_timeout: int = 3600,
    ) -> None:
        """Worker: poll queue and run inference.

        This is the main worker loop that runs on a separate GPU instance.
        It continuously polls the queue for new jobs, downloads checkpoints,
        runs inference, and uploads results.

        Args:
            adapter: VLM adapter for inference (e.g., Qwen adapter)
            max_messages: Maximum messages to process per iteration
            visibility_timeout: How long to hide message while processing (seconds)
        """
        logger.info("Starting inference worker...")

        while True:
            try:
                # Poll for messages
                messages = self.queue_client.receive_messages(
                    messages_per_page=max_messages,
                    visibility_timeout=visibility_timeout,
                )

                for msg in messages:
                    try:
                        # Parse job metadata
                        job_data = json.loads(msg.content)
                        job = InferenceJob(**job_data)

                        logger.info(f"Processing job: {job.job_id}")
                        job.status = "running"

                        # Download checkpoint from blob
                        checkpoint_dir = Path(f"/tmp/checkpoints/{job.job_id}")
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)

                        self._download_checkpoint(job.checkpoint_blob, checkpoint_dir)

                        # Run inference
                        output_path = Path(f"/tmp/comparisons/{job.job_id}.html")
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        self._run_inference(
                            adapter=adapter,
                            checkpoint_path=checkpoint_dir,
                            capture_path=job.capture_path,
                            output_path=output_path,
                        )

                        # Upload result to blob
                        output_blob = (
                            f"comparisons/epoch_{job.checkpoint_epoch}_comparison.html"
                        )
                        self._upload_comparison(output_path, output_blob)

                        job.status = "completed"
                        job.output_blob = output_blob

                        logger.info(f"Job completed: {job.job_id}")

                        # Delete message from queue
                        self.queue_client.delete_message(msg)

                    except Exception as e:
                        logger.error(f"Job failed: {e}")
                        job.status = "failed"
                        job.error = str(e)
                        # Don't delete message - let it become visible again for retry

            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(10)  # Back off on errors

            # Poll interval
            time.sleep(5)

    def watch_comparisons(self, poll_interval: int = 10) -> None:
        """Poll for new comparison results.

        This can be used by the dashboard to discover new comparison files.

        Args:
            poll_interval: How often to check for new files (seconds)
        """
        logger.info("Watching for new comparison results...")
        seen_blobs = set()

        while True:
            try:
                # List blobs in comparisons container
                container_client = self.blob_service.get_container_client(
                    self.comparisons_container
                )
                blobs = container_client.list_blobs()

                for blob in blobs:
                    if blob.name not in seen_blobs:
                        logger.info(f"New comparison: {blob.name}")
                        seen_blobs.add(blob.name)

                        # Optionally download and open in browser
                        # self._download_and_open(blob.name)

            except Exception as e:
                logger.error(f"Watch error: {e}")

            time.sleep(poll_interval)

    def _download_checkpoint(self, blob_name: str, local_dir: Path) -> None:
        """Download checkpoint from blob storage."""
        # List all files under the checkpoint blob prefix
        container_client = self.blob_service.get_container_client(
            self.checkpoints_container
        )
        blob_prefix = "/".join(blob_name.split("/")[:-1])  # Get directory prefix

        blobs = container_client.list_blobs(name_starts_with=blob_prefix)

        for blob in blobs:
            # Download each file
            local_path = local_dir / Path(blob.name).relative_to(blob_prefix)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            blob_client = self.blob_service.get_blob_client(
                container=self.checkpoints_container, blob=blob.name
            )

            with open(local_path, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())

            logger.debug(f"Downloaded {blob.name} to {local_path}")

    def _run_inference(
        self,
        adapter: Any,
        checkpoint_path: Path,
        capture_path: str,
        output_path: Path,
    ) -> None:
        """Run inference and generate comparison HTML.

        This wraps the comparison generation logic from scripts/compare.py.
        """
        # Import here to avoid circular dependencies
        from openadapt_ml.scripts.compare import generate_comparison

        logger.info(f"Running inference on {capture_path}...")

        # Load checkpoint into adapter
        adapter.load_lora_weights(str(checkpoint_path))

        # Generate comparison
        generate_comparison(
            capture_path=capture_path,
            adapter=adapter,
            output_path=str(output_path),
        )

        logger.info(f"Comparison saved to {output_path}")

    def _upload_comparison(self, local_path: Path, blob_name: str) -> None:
        """Upload comparison HTML to blob storage."""
        blob_client = self.blob_service.get_blob_client(
            container=self.comparisons_container, blob=blob_name
        )

        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)

        logger.info(f"Uploaded comparison to {blob_name}")


def main():
    """CLI for Azure async inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Azure async inference queue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Submit checkpoint for inference
    submit_parser = subparsers.add_parser(
        "inference-submit", help="Submit checkpoint for async inference"
    )
    submit_parser.add_argument(
        "--checkpoint", "-c", required=True, help="Path to checkpoint directory"
    )
    submit_parser.add_argument(
        "--capture", required=True, help="Path to capture data"
    )
    submit_parser.add_argument(
        "--epoch", "-e", type=int, default=0, help="Epoch number"
    )

    # Start inference worker
    worker_parser = subparsers.add_parser(
        "inference-worker", help="Start inference worker process"
    )
    worker_parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-VL-3B",
        help="VLM model to use for inference",
    )

    # Watch for new comparisons
    watch_parser = subparsers.add_parser(
        "inference-watch", help="Watch for new comparison results"
    )
    watch_parser.add_argument(
        "--interval", "-i", type=int, default=10, help="Poll interval in seconds"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize queue
    queue = AzureInferenceQueue()

    if args.command == "inference-submit":
        # Submit checkpoint for inference
        print(f"Submitting checkpoint for inference...")
        job = queue.submit_checkpoint(
            checkpoint_path=args.checkpoint,
            capture_path=args.capture,
            epoch=args.epoch,
        )
        print(f"Job submitted: {job.job_id}")
        print(f"Checkpoint uploaded to: {job.checkpoint_blob}")

    elif args.command == "inference-worker":
        # Start inference worker
        from openadapt_ml.adapters.qwen import QwenVLAdapter

        print(f"Starting inference worker with model: {args.model}")
        adapter = QwenVLAdapter(model_name=args.model)
        queue.poll_and_process(adapter)

    elif args.command == "inference-watch":
        # Watch for new comparisons
        queue.watch_comparisons(poll_interval=args.interval)


if __name__ == "__main__":
    main()
