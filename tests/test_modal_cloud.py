"""Tests for Modal cloud GPU integration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------


class TestCLIParsing:
    """Test CLI argument parsing without needing modal installed."""

    def test_train_help(self):
        """Test that train --help exits cleanly."""
        from openadapt_ml.cloud.modal_cloud import cli_main

        with pytest.raises(SystemExit) as exc_info:
            cli_main(["train", "--help"])
        assert exc_info.value.code == 0

    def test_status_help(self):
        """Test that status --help exits cleanly."""
        from openadapt_ml.cloud.modal_cloud import cli_main

        with pytest.raises(SystemExit) as exc_info:
            cli_main(["status", "--help"])
        assert exc_info.value.code == 0

    def test_download_help(self):
        """Test that download --help exits cleanly."""
        from openadapt_ml.cloud.modal_cloud import cli_main

        with pytest.raises(SystemExit) as exc_info:
            cli_main(["download", "--help"])
        assert exc_info.value.code == 0

    def test_no_command_returns_1(self):
        """Test that no command prints help and returns 1."""
        from openadapt_ml.cloud.modal_cloud import cli_main

        result = cli_main([])
        assert result == 1

    def test_train_requires_bundle_or_demo_dir(self):
        """Test that train without --bundle or --demo-dir fails."""
        from openadapt_ml.cloud.modal_cloud import cli_main

        # Mock _get_modal so we don't need modal installed
        mock_modal = MagicMock()
        with patch(
            "openadapt_ml.cloud.modal_cloud._get_modal", return_value=mock_modal
        ):
            result = cli_main(["train"])
        assert result == 1

    def test_train_demo_dir_requires_captures_dir(self):
        """Test that --demo-dir without --captures-dir fails."""
        from openadapt_ml.cloud.modal_cloud import cli_main

        mock_modal = MagicMock()
        with patch(
            "openadapt_ml.cloud.modal_cloud._get_modal", return_value=mock_modal
        ):
            result = cli_main(["train", "--demo-dir", "/fake/demos"])
        assert result == 1


# ---------------------------------------------------------------------------
# Bundle upload tests
# ---------------------------------------------------------------------------


class TestBundleUpload:
    """Test bundle upload logic."""

    def test_upload_missing_bundle_raises(self):
        """Test that uploading a non-existent bundle raises FileNotFoundError."""
        from openadapt_ml.cloud.modal_cloud import upload_bundle_to_volume

        with pytest.raises(FileNotFoundError, match="Bundle not found"):
            upload_bundle_to_volume("/nonexistent/path")

    def test_upload_bundle_without_jsonl_raises(self):
        """Test that bundle without training_data.jsonl raises FileNotFoundError."""
        from openadapt_ml.cloud.modal_cloud import upload_bundle_to_volume

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No training_data.jsonl"):
                upload_bundle_to_volume(tmpdir)

    def test_upload_bundle_calls_modal_volume_put(self):
        """Test that upload invokes 'modal volume create' then 'modal volume put'."""
        from openadapt_ml.cloud.modal_cloud import VOLUME_NAME, upload_bundle_to_volume

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the required file
            (Path(tmpdir) / "training_data.jsonl").write_text('{"test": true}\n')

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""

            with patch(
                "openadapt_ml.cloud.modal_cloud.subprocess.run",
                return_value=mock_result,
            ) as mock_run:
                upload_bundle_to_volume(tmpdir)

                # Two calls: create volume + put
                assert mock_run.call_count == 2
                # Second call is the put
                put_cmd = mock_run.call_args_list[1][0][0]
                assert put_cmd[0] == "modal"
                assert put_cmd[1] == "volume"
                assert put_cmd[2] == "put"
                assert put_cmd[3] == VOLUME_NAME
                assert put_cmd[4] == tmpdir
                assert put_cmd[5] == "/bundle"

    def test_upload_bundle_failure_raises(self):
        """Test that a failed volume put raises RuntimeError."""
        from openadapt_ml.cloud.modal_cloud import upload_bundle_to_volume

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "training_data.jsonl").write_text('{"test": true}\n')

            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "volume not found"
            mock_result.stdout = ""

            with patch(
                "openadapt_ml.cloud.modal_cloud.subprocess.run",
                return_value=mock_result,
            ):
                with pytest.raises(RuntimeError, match="Volume upload failed"):
                    upload_bundle_to_volume(tmpdir)


# ---------------------------------------------------------------------------
# Download results tests
# ---------------------------------------------------------------------------


class TestDownloadResults:
    """Test results download logic."""

    def test_download_calls_modal_volume_get(self):
        """Test that download invokes 'modal volume get' with correct args."""
        from openadapt_ml.cloud.modal_cloud import (
            VOLUME_NAME,
            download_results_from_volume,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""

            with patch(
                "openadapt_ml.cloud.modal_cloud.subprocess.run",
                return_value=mock_result,
            ) as mock_run:
                download_results_from_volume(tmpdir)

                mock_run.assert_called_once()
                cmd = mock_run.call_args[0][0]
                assert cmd[0] == "modal"
                assert cmd[1] == "volume"
                assert cmd[2] == "get"
                assert cmd[3] == VOLUME_NAME
                assert cmd[4] == "/results"
                assert cmd[5] == tmpdir

    def test_download_failure_raises(self):
        """Test that a failed volume get raises RuntimeError."""
        from openadapt_ml.cloud.modal_cloud import download_results_from_volume

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "volume not found"
        mock_result.stdout = ""

        with patch(
            "openadapt_ml.cloud.modal_cloud.subprocess.run", return_value=mock_result
        ):
            with pytest.raises(RuntimeError, match="Volume download failed"):
                download_results_from_volume("/tmp/test_output")

    def test_download_creates_output_dir(self):
        """Test that download creates the output directory if missing."""
        from openadapt_ml.cloud.modal_cloud import download_results_from_volume

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "output"

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""

            with patch(
                "openadapt_ml.cloud.modal_cloud.subprocess.run",
                return_value=mock_result,
            ):
                download_results_from_volume(output_dir)
                assert output_dir.exists()


# ---------------------------------------------------------------------------
# Status check tests
# ---------------------------------------------------------------------------


class TestStatus:
    """Test status checking."""

    def test_status_no_data(self):
        """Test status when no training log exists."""
        from openadapt_ml.cloud.modal_cloud import check_status

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "not found"
        mock_result.stdout = ""

        with patch(
            "openadapt_ml.cloud.modal_cloud.subprocess.run", return_value=mock_result
        ):
            status = check_status()
            assert status["running"] is False
            assert status["status"] == "no_data"

    def test_status_running(self):
        """Test status when training is running."""
        from openadapt_ml.cloud.modal_cloud import check_status

        log_data = {
            "status": "running",
            "epoch": 2,
            "elapsed_time": 120.5,
            "losses": [{"loss": 1.0}, {"loss": 0.5}],
        }

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            # Write the log file to the tmpdir (last arg in the command)
            output_dir = cmd[-1]
            log_path = Path(output_dir) / "training_log.json"
            log_path.write_text(json.dumps(log_data))
            return result

        with patch(
            "openadapt_ml.cloud.modal_cloud.subprocess.run", side_effect=mock_run
        ):
            status = check_status()
            assert status["running"] is True
            assert status["status"] == "running"
            assert status["epoch"] == 2

    def test_status_completed(self):
        """Test status when training is completed."""
        from openadapt_ml.cloud.modal_cloud import check_status

        log_data = {
            "status": "completed",
            "epoch": 5,
            "elapsed_time": 3600.0,
            "losses": [{"loss": 1.0}, {"loss": 0.1}],
        }

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            output_dir = cmd[-1]
            log_path = Path(output_dir) / "training_log.json"
            log_path.write_text(json.dumps(log_data))
            return result

        with patch(
            "openadapt_ml.cloud.modal_cloud.subprocess.run", side_effect=mock_run
        ):
            status = check_status()
            assert status["running"] is False
            assert status["status"] == "completed"
            assert status["elapsed_time"] == 3600.0


# ---------------------------------------------------------------------------
# Train command auto-convert flow tests
# ---------------------------------------------------------------------------


class TestTrainAutoConvert:
    """Test the train command's auto-convert demo flow."""

    def test_train_auto_converts_demos(self):
        """Test that --demo-dir triggers prepare_bundle and proceeds."""
        from openadapt_ml.cloud.modal_cloud import cli_main

        mock_modal = MagicMock()
        mock_modal.enable_output = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_dir.mkdir()
            (bundle_dir / "training_data.jsonl").write_text('{"test": true}\n')
            (bundle_dir / "images").mkdir()

            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("model_name: test\nnum_epochs: 1\n")

            mock_upload_result = MagicMock()
            mock_upload_result.returncode = 0
            mock_upload_result.stdout = ""
            mock_upload_result.stderr = ""

            with (
                patch(
                    "openadapt_ml.cloud.modal_cloud._get_modal", return_value=mock_modal
                ),
                patch(
                    "openadapt_ml.cloud.modal_cloud.upload_bundle_to_volume"
                ) as mock_upload,
                patch(
                    "openadapt_ml.training.convert_demos.prepare_bundle",
                    return_value=bundle_dir,
                ) as mock_prepare,
                patch(
                    "openadapt_ml.cloud.modal_cloud._register_train_function"
                ) as mock_register,
                patch("openadapt_ml.cloud.modal_cloud.get_app") as mock_get_app,
                patch("openadapt_ml.cloud.modal_cloud.download_results_from_volume"),
            ):
                # Set up the mock app context manager
                mock_app = MagicMock()
                mock_app.run.return_value.__enter__ = MagicMock()
                mock_app.run.return_value.__exit__ = MagicMock(return_value=False)
                mock_get_app.return_value = mock_app

                # Set up the mock train function
                mock_train_fn = MagicMock()
                mock_train_fn.remote.return_value = json.dumps(
                    {"status": "completed", "elapsed_time": 100.0}
                )
                mock_register.return_value = mock_train_fn

                result = cli_main(
                    [
                        "train",
                        "--demo-dir",
                        "/fake/demos",
                        "--captures-dir",
                        "/fake/captures",
                        "--config",
                        str(config_path),
                    ]
                )

                # prepare_bundle should have been called
                mock_prepare.assert_called_once_with(
                    demo_dir="/fake/demos",
                    captures_dir="/fake/captures",
                    mapping_path=None,
                )

                # Upload should have been called with the bundle dir
                mock_upload.assert_called_once_with(str(bundle_dir))

                assert result == 0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    """Test that Modal config fields exist in Settings."""

    def test_modal_token_fields_exist(self):
        """Verify modal_token_id and modal_token_secret are in Settings."""
        from openadapt_ml.config import Settings

        s = Settings()
        assert hasattr(s, "modal_token_id")
        assert hasattr(s, "modal_token_secret")
        # Defaults should be empty strings
        assert s.modal_token_id == ""
        assert s.modal_token_secret == ""


# ---------------------------------------------------------------------------
# Module constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Test module-level constants are sensible."""

    def test_constants(self):
        """Verify constants are set correctly."""
        from openadapt_ml.cloud import modal_cloud

        assert modal_cloud.MODAL_APP_NAME == "openadapt-training"
        assert modal_cloud.INFERENCE_APP_NAME == "openadapt-inference"
        assert modal_cloud.VOLUME_NAME == "openadapt-training-data"
        assert modal_cloud.VOLUME_MOUNT == "/training"
        assert modal_cloud.BUNDLE_REMOTE_PATH == "/training/bundle"
        assert modal_cloud.RESULTS_REMOTE_PATH == "/training/results"


# ---------------------------------------------------------------------------
# Serve / inference tests
# ---------------------------------------------------------------------------


class TestServeCLI:
    """Test serve command CLI parsing."""

    def test_serve_help(self):
        """Test that serve --help exits cleanly."""
        from openadapt_ml.cloud.modal_cloud import cli_main

        with pytest.raises(SystemExit) as exc_info:
            cli_main(["serve", "--help"])
        assert exc_info.value.code == 0

    def test_serve_no_adapter_flag(self):
        """Test that serve --no-adapter uses base model only."""
        from openadapt_ml.cloud.modal_cloud import cli_main

        mock_modal = MagicMock()
        mock_modal.enable_output = MagicMock()

        mock_app = MagicMock()
        mock_infer = MagicMock()

        with (
            patch(
                "openadapt_ml.cloud.modal_cloud._get_modal", return_value=mock_modal
            ),
            patch(
                "openadapt_ml.cloud.modal_cloud._build_inference_app",
                return_value=(mock_app, mock_infer),
            ) as mock_build,
            patch("openadapt_ml.cloud.modal_cloud.json") as mock_json,
        ):
            mock_json.dumps.return_value = '[]'
            mock_json.loads.return_value = {"response": "ready"}
            mock_infer.remote.return_value = '{"response": "ready"}'
            mock_app.run.return_value.__enter__ = MagicMock()
            mock_app.run.return_value.__exit__ = MagicMock(return_value=False)

            # The serve command blocks on the while loop, so we use KeyboardInterrupt
            call_count = [0]

            def mock_sleep(t):
                call_count[0] += 1
                if call_count[0] > 1:
                    raise KeyboardInterrupt()

            with patch("time.sleep", side_effect=mock_sleep):
                result = cli_main(["serve", "--no-adapter"])

            # Should have called _build_inference_app with no adapter
            mock_build.assert_called_once_with(
                adapter_path=None,
                base_model="Qwen/Qwen3-VL-2B-Instruct",
                gpu="A10G",
            )
            assert result == 0


class TestUploadAdapter:
    """Test adapter upload logic."""

    def test_upload_missing_adapter_raises(self):
        """Test that uploading a non-existent adapter raises."""
        from openadapt_ml.cloud.modal_cloud import upload_adapter_to_volume

        with pytest.raises(FileNotFoundError, match="Adapter not found"):
            upload_adapter_to_volume("/nonexistent/path")

    def test_upload_adapter_without_config_raises(self):
        """Test that adapter without adapter_config.json raises."""
        from openadapt_ml.cloud.modal_cloud import upload_adapter_to_volume

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No adapter_config.json"):
                upload_adapter_to_volume(tmpdir)

    def test_upload_adapter_calls_volume_put(self):
        """Test that upload invokes volume put with correct args."""
        from openadapt_ml.cloud.modal_cloud import upload_adapter_to_volume

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "adapter_config.json").write_text('{"test": true}')
            (Path(tmpdir) / "adapter_model.safetensors").write_text("fake")

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""

            with patch(
                "openadapt_ml.cloud.modal_cloud.subprocess.run",
                return_value=mock_result,
            ) as mock_run:
                remote = upload_adapter_to_volume(tmpdir)

                # Two calls: create + put
                assert mock_run.call_count == 2
                put_cmd = mock_run.call_args_list[1][0][0]
                assert "put" in put_cmd
                assert "/adapter" in put_cmd
                assert "--force" in put_cmd
                assert remote == "/training/adapter"
