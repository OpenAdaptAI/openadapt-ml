#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
WORKSPACE_ROOT=$(cd "${ROOT_DIR}/.." && pwd)

WAA_PATH=""
ISO_URL=""
ISO_PATH=""
USE_SUDO="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --waa-path)
            WAA_PATH="$2"
            shift 2
            ;;
        --iso-url)
            ISO_URL="$2"
            shift 2
            ;;
        --iso-path)
            ISO_PATH="$2"
            shift 2
            ;;
        --sudo)
            USE_SUDO="true"
            shift 1
            ;;
        --help)
            echo "Usage: $0 [--waa-path <path>] [--iso-url <url> | --iso-path <path>] [--sudo]"
            echo ""
            echo "Optional:"
            echo "  --waa-path <path>     Path to WindowsAgentArena repo (auto-detected if omitted)"
            echo "  --iso-url <url>       Download Windows 11 Enterprise ISO to setup.iso"
            echo "  --iso-path <path>     Copy ISO from local path to setup.iso"
            echo "  --sudo                Run run-local.sh with sudo"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
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
        echo "Cloning WindowsAgentArena into ${WAA_PATH}..."
        git clone --depth 1 https://github.com/microsoft/WindowsAgentArena.git "$WAA_PATH"
    fi
fi

if [[ ! -d "$WAA_PATH" ]]; then
    echo "Error: WAA path does not exist: $WAA_PATH"
    exit 1
fi

IMAGE_DIR="$WAA_PATH/src/win-arena-container/vm/image"
ISO_DEST="$IMAGE_DIR/setup.iso"

mkdir -p "$IMAGE_DIR"

if [[ -f "$ISO_DEST" ]]; then
    echo "ISO already present: $ISO_DEST"
else
    if [[ -z "$ISO_PATH" && -z "$ISO_URL" ]]; then
        candidates=(
            "${HOME}/Downloads/Windows11_Enterprise_Eval.iso"
            "${HOME}/Downloads/Windows11_Enterprise.iso"
            "${HOME}/Downloads/Windows11.iso"
            "${HOME}/Downloads/Win11*.iso"
            "${HOME}/Downloads/*Windows*11*Enterprise*.iso"
        )
        matches=()
        for pattern in "${candidates[@]}"; do
            for file in $pattern; do
                if [[ -f "$file" ]]; then
                    matches+=("$file")
                fi
            done
        done
        if [[ ${#matches[@]} -eq 1 ]]; then
            ISO_PATH="${matches[0]}"
        elif [[ ${#matches[@]} -gt 1 ]]; then
            echo "Error: multiple ISO candidates found. Specify --iso-path."
            printf '  - %s\n' "${matches[@]}"
            exit 1
        fi
    fi

    if [[ -n "$ISO_PATH" ]]; then
        if [[ ! -f "$ISO_PATH" ]]; then
            echo "Error: ISO path not found: $ISO_PATH"
            exit 1
        fi
        cp "$ISO_PATH" "$ISO_DEST"
    elif [[ -n "$ISO_URL" ]]; then
        curl -L "$ISO_URL" -o "$ISO_DEST"
    else
        echo "Error: provide --iso-url or --iso-path to place setup.iso"
        exit 1
    fi
fi

if [[ ! -x "$WAA_PATH/scripts/run-local.sh" ]]; then
    echo "Error: run-local.sh not found at $WAA_PATH/scripts/run-local.sh"
    exit 1
fi

if [[ "$USE_SUDO" == "true" ]]; then
    (cd "$WAA_PATH/scripts" && sudo ./run-local.sh --prepare-image true)
else
    (cd "$WAA_PATH/scripts" && ./run-local.sh --prepare-image true)
fi
