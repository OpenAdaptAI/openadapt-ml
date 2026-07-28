"""Tests for Vast.ai cloud GPU integration."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from openadapt_ml.cloud.vast_ai import (
    API_BASE,
    DEFAULT_DISK_GB,
    DEFAULT_IMAGE,
    VastAIClient,
    VastInstance,
    VastOffer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_OFFER = {
    "id": 12345,
    "gpu_name": "A10",
    "num_gpus": 1,
    "gpu_ram": 24576,  # MB
    "cpu_cores_effective": 8,
    "cpu_ram": 32768,  # MB
    "disk_space": 100.0,
    "dph_total": 0.17,
    "dlperf": 15.2,
    "reliability2": 0.98,
    "inet_down": 500.0,
    "inet_up": 200.0,
    "geolocation": "US",
}

SAMPLE_INSTANCE = {
    "id": 99999,
    "intended_status": "running",
    "actual_status": "running",
    "gpu_name": "A10",
    "ssh_host": "ssh5.vast.ai",
    "ssh_port": 22222,
    "public_ipaddr": "1.2.3.4",
    "image_uuid": "pytorch/pytorch:2.4.0",
    "num_gpus": 1,
    "gpu_ram": 24576,
    "dph_total": 0.17,
    "disk_space": 50.0,
}


@pytest.fixture
def mock_settings():
    """Mock settings to provide API key."""
    with patch("openadapt_ml.cloud.vast_ai.os.environ", {"VAST_API_KEY": ""}):
        with patch("openadapt_ml.config.Settings") as mock:
            mock.return_value.vast_api_key = "test-api-key-123"
            yield mock


def make_client() -> VastAIClient:
    """Create a VastAIClient with a test API key (bypasses settings)."""
    return VastAIClient(api_key="test-api-key-123")


# ---------------------------------------------------------------------------
# Client initialization tests
# ---------------------------------------------------------------------------


class TestClientInit:
    """Test VastAIClient initialization."""

    def test_init_with_explicit_key(self):
        """Client initializes with explicitly provided API key."""
        client = VastAIClient(api_key="my-key")
        assert client.api_key == "my-key"
        assert client.session.headers["Authorization"] == "Bearer my-key"

    def test_init_from_env_var(self):
        """Client picks up API key from environment variable."""
        mock_settings = MagicMock()
        mock_settings.vast_api_key = None
        with patch(
            "openadapt_ml.cloud.vast_ai.os.environ", {"VAST_API_KEY": "env-key"}
        ):
            with patch("openadapt_ml.config.settings", mock_settings):
                client = VastAIClient()
                assert client.api_key == "env-key"

    def test_init_from_settings(self):
        """Client picks up API key from config settings."""
        mock_settings = MagicMock()
        mock_settings.vast_api_key = "settings-key"
        with patch("openadapt_ml.cloud.vast_ai.os.environ", {}):
            with patch("openadapt_ml.config.settings", mock_settings):
                client = VastAIClient()
                assert client.api_key == "settings-key"

    def test_init_no_key_raises(self):
        """Client raises ValueError when no API key is available."""
        mock_settings = MagicMock()
        mock_settings.vast_api_key = None
        with patch("openadapt_ml.cloud.vast_ai.os.environ", {}):
            with patch("openadapt_ml.config.settings", mock_settings):
                with pytest.raises(ValueError, match="Vast.ai API key required"):
                    VastAIClient()


# ---------------------------------------------------------------------------
# Offer parsing tests
# ---------------------------------------------------------------------------


class TestOfferParsing:
    """Test parsing of marketplace offer data."""

    def test_parse_single_offer(self):
        """VastOffer correctly parses API response fields."""
        offer = VastOffer(
            id=SAMPLE_OFFER["id"],
            gpu_name=SAMPLE_OFFER["gpu_name"],
            num_gpus=SAMPLE_OFFER["num_gpus"],
            gpu_ram_gb=SAMPLE_OFFER["gpu_ram"] / 1024,
            cpu_cores=SAMPLE_OFFER["cpu_cores_effective"],
            ram_gb=SAMPLE_OFFER["cpu_ram"] / 1024,
            disk_gb=SAMPLE_OFFER["disk_space"],
            dph_total=SAMPLE_OFFER["dph_total"],
            dlperf=SAMPLE_OFFER["dlperf"],
            reliability=SAMPLE_OFFER["reliability2"],
            inet_down=SAMPLE_OFFER["inet_down"],
            inet_up=SAMPLE_OFFER["inet_up"],
            geolocation=SAMPLE_OFFER["geolocation"],
        )

        assert offer.id == 12345
        assert offer.gpu_name == "A10"
        assert offer.gpu_ram_gb == 24.0
        assert offer.price_per_hour == 0.17
        assert "A10" in str(offer)
        assert "$0.170/hr" in str(offer)

    def test_list_offers_api_call(self):
        """list_offers makes correct API call and parses response."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"offers": [SAMPLE_OFFER]}

        with patch.object(
            client.session, "put", return_value=mock_response
        ) as mock_put:
            offers = client.list_offers(gpu_type="A10", max_price=0.30)

            # Verify API was called
            mock_put.assert_called_once()
            call_url = mock_put.call_args[0][0]
            assert call_url == f"{API_BASE}/search/asks/"

            # Verify payload
            call_data = mock_put.call_args[1]["json"]
            assert call_data["q"]["gpu_name"] == {"eq": "A10"}
            assert call_data["q"]["dph_total"] == {"lte": 0.30}

            # Verify parsed offer
            assert len(offers) == 1
            assert offers[0].gpu_name == "A10"
            assert offers[0].dph_total == 0.17

    def test_list_offers_empty_result(self):
        """list_offers returns empty list when no offers match."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"offers": []}

        with patch.object(client.session, "put", return_value=mock_response):
            offers = client.list_offers(gpu_type="H100", max_price=0.10)
            assert offers == []

    def test_list_offers_vram_filter(self):
        """list_offers correctly converts min_vram GB to MB for the API."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"offers": []}

        with patch.object(
            client.session, "put", return_value=mock_response
        ) as mock_put:
            client.list_offers(min_vram=24.0)

            call_data = mock_put.call_args[1]["json"]
            # 24 GB = 24576 MB
            assert call_data["q"]["gpu_ram"] == {"gte": 24576.0}


# ---------------------------------------------------------------------------
# Instance lifecycle tests
# ---------------------------------------------------------------------------


class TestInstanceLifecycle:
    """Test instance create, list, destroy."""

    def test_list_instances(self):
        """list_instances parses API response correctly."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"instances": [SAMPLE_INSTANCE]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_response):
            instances = client.list_instances()

            assert len(instances) == 1
            inst = instances[0]
            assert inst.id == 99999
            assert inst.gpu_name == "A10"
            assert inst.ssh_host == "ssh5.vast.ai"
            assert inst.ssh_port == 22222
            assert inst.actual_status == "running"
            assert inst.dph_total == 0.17
            assert "A10" in str(inst)

    def test_list_instances_empty(self):
        """list_instances returns empty list when no instances."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"instances": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.session, "get", return_value=mock_response):
            instances = client.list_instances()
            assert instances == []

    def test_destroy_instance_success(self):
        """destroy_instance returns True on success."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.text = '{"success": true}'
        mock_response.json.return_value = {"success": True}

        with patch.object(client.session, "delete", return_value=mock_response):
            assert client.destroy_instance(99999) is True

    def test_destroy_instance_failure(self):
        """destroy_instance returns False on API error."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_response.text = "Not found"

        with patch.object(client.session, "delete", return_value=mock_response):
            assert client.destroy_instance(99999) is False

    def test_create_instance_success(self):
        """create_instance creates instance and waits for SSH info."""
        client = make_client()

        # Mock the PUT to create instance
        mock_create_response = MagicMock()
        mock_create_response.ok = True
        mock_create_response.json.return_value = {
            "success": True,
            "new_contract": 99999,
        }

        # Mock list_instances to return the running instance
        running_instance = VastInstance(
            id=99999,
            status="running",
            gpu_name="A10",
            ssh_host="ssh5.vast.ai",
            ssh_port=22222,
            ip="ssh5.vast.ai",
            actual_status="running",
            dph_total=0.17,
        )

        with patch.object(client.session, "put", return_value=mock_create_response):
            with patch.object(
                client, "list_instances", return_value=[running_instance]
            ):
                instance = client.create_instance(
                    offer_id=12345, image=DEFAULT_IMAGE, disk_gb=50
                )
                assert instance.id == 99999
                assert instance.ssh_host == "ssh5.vast.ai"
                assert instance.ssh_port == 22222

    def test_create_instance_api_failure(self):
        """create_instance raises RuntimeError when API returns failure."""
        client = make_client()

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "success": False,
            "msg": "Insufficient funds",
        }

        with patch.object(client.session, "put", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Insufficient funds"):
                client.create_instance(offer_id=12345)


# ---------------------------------------------------------------------------
# SSH and rsync tests
# ---------------------------------------------------------------------------


class TestSSHOperations:
    """Test SSH command execution and file transfer."""

    def _make_instance(self) -> VastInstance:
        return VastInstance(
            id=99999,
            status="running",
            gpu_name="A10",
            ssh_host="ssh5.vast.ai",
            ssh_port=22222,
            ip="ssh5.vast.ai",
            actual_status="running",
        )

    def test_ssh_run_success(self):
        """ssh_run executes command and returns result."""
        client = make_client()
        instance = self._make_instance()

        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "hello\n"
        mock_result.stderr = ""

        with patch(
            "openadapt_ml.cloud.vast_ai.subprocess.run", return_value=mock_result
        ) as mock_run:
            result = client.ssh_run(instance, "echo hello", timeout=30)

            assert result.returncode == 0
            assert result.stdout == "hello\n"

            # Verify SSH command includes correct port and host
            call_args = mock_run.call_args[0][0]
            assert "-p" in call_args
            port_idx = call_args.index("-p")
            assert call_args[port_idx + 1] == "22222"
            assert "root@ssh5.vast.ai" in call_args

    def test_ssh_run_retry_on_timeout(self):
        """ssh_run retries on timeout."""
        client = make_client()
        instance = self._make_instance()

        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise subprocess.TimeoutExpired(cmd="ssh", timeout=30)
            return mock_result

        with patch(
            "openadapt_ml.cloud.vast_ai.subprocess.run", side_effect=side_effect
        ):
            result = client.ssh_run(instance, "echo ok", timeout=30, retries=3)
            assert result.stdout == "ok"
            assert call_count == 2

    def test_ssh_run_no_host_raises(self):
        """ssh_run raises when instance has no SSH info."""
        client = make_client()
        instance = VastInstance(
            id=1,
            status="loading",
            gpu_name="A10",
            ssh_host="",
            ssh_port=0,
        )

        with pytest.raises(RuntimeError, match="no SSH info"):
            client.ssh_run(instance, "echo test")

    def test_rsync_to_builds_correct_command(self):
        """rsync_to builds command with correct port and paths."""
        client = make_client()
        instance = self._make_instance()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch(
            "openadapt_ml.cloud.vast_ai.subprocess.run", return_value=mock_result
        ) as mock_run:
            result = client.rsync_to(instance, "/local/data", "/root/data")

            assert result is True
            call_args = mock_run.call_args[0][0]
            assert "rsync" in call_args
            # SSH options should include port
            ssh_opts = call_args[call_args.index("-e") + 1]
            assert "-p 22222" in ssh_opts
            # Source and dest
            assert "/local/data/" in call_args
            assert "root@ssh5.vast.ai:/root/data/" in call_args

    def test_rsync_from_creates_local_dir(self):
        """rsync_from creates local directory if it doesn't exist."""
        import tempfile

        client = make_client()
        instance = self._make_instance()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = f"{tmpdir}/nested/output"

            with patch(
                "openadapt_ml.cloud.vast_ai.subprocess.run", return_value=mock_result
            ):
                client.rsync_from(instance, "/root/results", local_path)
                from pathlib import Path

                assert Path(local_path).exists()


# ---------------------------------------------------------------------------
# Training status tests
# ---------------------------------------------------------------------------


class TestTrainingStatus:
    """Test training status polling."""

    def _make_instance(self) -> VastInstance:
        return VastInstance(
            id=99999,
            status="running",
            gpu_name="A10",
            ssh_host="ssh5.vast.ai",
            ssh_port=22222,
        )

    def test_get_training_status_parses_json(self):
        """get_training_status parses training_log.json from remote."""
        client = make_client()
        instance = self._make_instance()

        log_data = {"step": 42, "epoch": 2, "loss": 0.35, "elapsed_time": 120.0}

        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(log_data)
        mock_result.stderr = ""

        with patch.object(client, "ssh_run", return_value=mock_result):
            status = client.get_training_status(instance)
            assert status["step"] == 42
            assert status["epoch"] == 2
            assert status["loss"] == 0.35

    def test_get_training_status_empty(self):
        """get_training_status returns empty dict when no log exists."""
        client = make_client()
        instance = self._make_instance()

        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "{}"
        mock_result.stderr = ""

        with patch.object(client, "ssh_run", return_value=mock_result):
            status = client.get_training_status(instance)
            assert status == {}

    def test_get_training_status_invalid_json(self):
        """get_training_status returns empty dict on invalid JSON."""
        client = make_client()
        instance = self._make_instance()

        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "not json at all"
        mock_result.stderr = ""

        with patch.object(client, "ssh_run", return_value=mock_result):
            status = client.get_training_status(instance)
            assert status == {}


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------


class TestCLIParsing:
    """Test CLI argument parsing."""

    def test_no_command_prints_help(self):
        """Running with no command prints help and returns."""
        from openadapt_ml.cloud.vast_ai import main

        with patch("openadapt_ml.cloud.vast_ai.VastAIClient") as mock_cls:
            mock_cls.return_value = make_client()
            with patch("sys.argv", ["vast_ai"]):
                # main() with no args should print help and return
                main()

    def test_list_subcommand(self):
        """List subcommand calls list_offers."""
        client = make_client()

        with patch("openadapt_ml.cloud.vast_ai.VastAIClient", return_value=client):
            with patch.object(client, "list_offers", return_value=[]) as mock_list:
                with patch("sys.argv", ["vast_ai", "list", "--gpu", "A10"]):
                    from openadapt_ml.cloud.vast_ai import main

                    main()
                    mock_list.assert_called_once()

    def test_status_subcommand(self):
        """Status subcommand calls list_instances."""
        client = make_client()

        with patch("openadapt_ml.cloud.vast_ai.VastAIClient", return_value=client):
            with patch.object(client, "list_instances", return_value=[]) as mock_list:
                with patch("sys.argv", ["vast_ai", "status"]):
                    from openadapt_ml.cloud.vast_ai import main

                    main()
                    mock_list.assert_called_once()

    def test_terminate_subcommand(self):
        """Terminate subcommand calls destroy_instance."""
        client = make_client()

        with patch("openadapt_ml.cloud.vast_ai.VastAIClient", return_value=client):
            with patch.object(
                client, "destroy_instance", return_value=True
            ) as mock_destroy:
                with patch("sys.argv", ["vast_ai", "terminate", "99999"]):
                    from openadapt_ml.cloud.vast_ai import main

                    main()
                    mock_destroy.assert_called_once_with(99999)


# ---------------------------------------------------------------------------
# Train auto-convert flow tests
# ---------------------------------------------------------------------------


class TestTrainAutoConvert:
    """Test the train command's auto-convert demo flow."""

    def test_train_demo_dir_requires_captures_dir(self):
        """--demo-dir without --captures-dir prints error."""
        client = make_client()

        with patch("openadapt_ml.cloud.vast_ai.VastAIClient", return_value=client):
            with patch(
                "sys.argv",
                ["vast_ai", "train", "--demo-dir", "/fake/demos"],
            ):
                from openadapt_ml.cloud.vast_ai import main

                # Should print error and return (not crash)
                main()

    def test_train_demo_dir_calls_prepare_bundle(self):
        """--demo-dir triggers prepare_bundle call."""
        import tempfile

        client = make_client()

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = f"{tmpdir}/bundle"
            from pathlib import Path

            Path(bundle_dir).mkdir()
            (Path(bundle_dir) / "training_data.jsonl").write_text('{"test": true}\n')

            instance = VastInstance(
                id=99999,
                status="running",
                gpu_name="A10",
                ssh_host="ssh5.vast.ai",
                ssh_port=22222,
                ip="ssh5.vast.ai",
                actual_status="running",
                dph_total=0.17,
            )

            with (
                patch("openadapt_ml.cloud.vast_ai.VastAIClient", return_value=client),
                patch.object(client, "list_instances", return_value=[instance]),
                patch.object(client, "wait_for_ssh", return_value=True),
                patch.object(client, "setup_instance", return_value=True),
                patch.object(client, "upload_capture", return_value=True),
                patch.object(client, "run_training"),
                patch.object(
                    client,
                    "get_training_status",
                    side_effect=KeyboardInterrupt,
                ),
                patch(
                    "openadapt_ml.training.convert_demos.prepare_bundle",
                    return_value=Path(bundle_dir),
                ) as mock_prepare,
                patch(
                    "sys.argv",
                    [
                        "vast_ai",
                        "train",
                        "--demo-dir",
                        "/fake/demos",
                        "--captures-dir",
                        "/fake/captures",
                    ],
                ),
            ):
                from openadapt_ml.cloud.vast_ai import main

                main()

                mock_prepare.assert_called_once_with(
                    demo_dir="/fake/demos",
                    captures_dir="/fake/captures",
                    mapping_path=None,
                )


# ---------------------------------------------------------------------------
# VastInstance dataclass tests
# ---------------------------------------------------------------------------


class TestVastInstance:
    """Test VastInstance dataclass."""

    def test_str_representation(self):
        """String representation includes key info."""
        inst = VastInstance(
            id=12345,
            status="running",
            gpu_name="A10",
            ssh_host="ssh5.vast.ai",
            ssh_port=22222,
            ip="1.2.3.4",
            actual_status="running",
            dph_total=0.17,
        )
        s = str(inst)
        assert "12345" in s
        assert "A10" in s
        assert "running" in s
        assert "$0.170/hr" in s

    def test_default_fields(self):
        """Optional fields have correct defaults."""
        inst = VastInstance(
            id=1,
            status="loading",
            gpu_name="A10",
            ssh_host="",
            ssh_port=0,
        )
        assert inst.ip is None
        assert inst.actual_status is None
        assert inst.image is None
        assert inst.num_gpus == 1
        assert inst.gpu_ram_gb == 0.0
        assert inst.dph_total == 0.0
        assert inst.disk_gb == 0.0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    """Test that Vast.ai config fields exist in Settings."""

    def test_vast_api_key_field_exists(self):
        """Verify vast_api_key is in Settings."""
        from openadapt_ml.config import Settings

        s = Settings()
        assert hasattr(s, "vast_api_key")
        # Default should be None
        assert s.vast_api_key is None


# ---------------------------------------------------------------------------
# Module constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Test module-level constants."""

    def test_api_base(self):
        assert API_BASE == "https://console.vast.ai/api/v0"

    def test_default_image(self):
        assert "pytorch" in DEFAULT_IMAGE
        assert "cuda" in DEFAULT_IMAGE

    def test_default_disk(self):
        assert DEFAULT_DISK_GB == 50
