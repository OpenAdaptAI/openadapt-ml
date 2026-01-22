#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
WORKSPACE_ROOT=$(cd "${ROOT_DIR}/.." && pwd)

WAA_PATH=""
DO_CLONE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --waa-path)
            WAA_PATH="$2"
            shift 2
            ;;
        --clone)
            DO_CLONE="true"
            shift 1
            ;;
        --help)
            echo "Usage: $0 [--waa-path <path>] [--clone]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$WAA_PATH" ]]; then
    if [[ -d "${WORKSPACE_ROOT}/openadapt-evals/vendor/WindowsAgentArena" ]]; then
        WAA_PATH="${WORKSPACE_ROOT}/openadapt-evals/vendor/WindowsAgentArena"
    elif [[ -d "${WORKSPACE_ROOT}/openadapt-evals/external/WindowsAgentArena" ]]; then
        WAA_PATH="${WORKSPACE_ROOT}/openadapt-evals/external/WindowsAgentArena"
    elif [[ -d "${ROOT_DIR}/vendor/WindowsAgentArena" ]]; then
        WAA_PATH="${ROOT_DIR}/vendor/WindowsAgentArena"
    elif [[ -d "${ROOT_DIR}/external/WindowsAgentArena" ]]; then
        WAA_PATH="${ROOT_DIR}/external/WindowsAgentArena"
    elif [[ -d "${HOME}/WindowsAgentArena" ]]; then
        WAA_PATH="${HOME}/WindowsAgentArena"
    else
        if [[ -d "${WORKSPACE_ROOT}/openadapt-evals" ]]; then
            WAA_PATH="${WORKSPACE_ROOT}/openadapt-evals/vendor/WindowsAgentArena"
        else
            WAA_PATH="${ROOT_DIR}/vendor/WindowsAgentArena"
        fi
    fi
fi

if [[ ! -d "$WAA_PATH" ]]; then
    if [[ "$DO_CLONE" == "true" ]]; then
        echo "Cloning WindowsAgentArena into ${WAA_PATH}..."
        git clone --depth 1 https://github.com/microsoft/WindowsAgentArena.git "$WAA_PATH"
    else
        echo "WAA repo not found at ${WAA_PATH}"
        echo "Run with --clone to create it."
        exit 1
    fi
fi

ISO_DEST="$WAA_PATH/src/win-arena-container/vm/image/setup.iso"
CONFIG_PATH="$WAA_PATH/config.json"
RUN_LOCAL="$WAA_PATH/scripts/run-local.sh"

echo "WAA path: $WAA_PATH"
echo "run-local.sh: $RUN_LOCAL"
echo "setup.iso: $ISO_DEST"
echo "config.json: $CONFIG_PATH"

if [[ ! -x "$RUN_LOCAL" ]]; then
    echo "Missing or non-executable run-local.sh"
    exit 1
fi

if [[ ! -f "$ISO_DEST" ]]; then
    echo "Missing setup.iso (place at $ISO_DEST)"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Missing config.json (create in $WAA_PATH)"
fi

echo "Helper check complete."
