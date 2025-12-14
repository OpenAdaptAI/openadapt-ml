from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file.

    Priority order for configuration values:
    1. Environment variables
    2. .env file
    3. Default values (None for API keys)
    """

    # VLM API Keys
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None

    # Azure credentials (for WAA benchmark on Azure)
    # These are used by DefaultAzureCredential for Service Principal auth
    azure_client_id: str | None = None
    azure_client_secret: str | None = None
    azure_tenant_id: str | None = None

    # Azure ML workspace config
    azure_subscription_id: str | None = None
    azure_ml_resource_group: str | None = None
    azure_ml_workspace_name: str | None = None

    # Azure VM settings (optional overrides)
    # D2_v3 = 2 vCPUs, 8GB RAM (fits free trial with existing usage)
    # D4_v3 = 4 vCPUs, 16GB RAM (needs 4 free vCPUs)
    # D8_v3 = 8 vCPUs, 32GB RAM (requires quota increase)
    azure_vm_size: str = "Standard_D2_v3"
    # Docker image for WAA agent container
    # Default is Docker Hub; setup_azure.py will set this to ACR image
    azure_docker_image: str = "docker.io/windowsarena/winarena:latest"

    # Azure Storage for async inference queue (Phase 2)
    azure_storage_connection_string: str | None = None
    azure_inference_queue_name: str = "inference-jobs"
    azure_checkpoints_container: str = "checkpoints"
    azure_comparisons_container: str = "comparisons"

    # Lambda Labs (cloud GPU for training)
    lambda_api_key: str | None = None

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # ignore extra env vars
    }


settings = Settings()
