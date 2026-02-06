#!/bin/bash
# Check Azure resources on session start
# This script is designed to be used as a Claude Code SessionStart hook
#
# Setup: Add to ~/.claude/settings.json or .claude/settings.local.json:
# {
#   "hooks": {
#     "SessionStart": [
#       {
#         "type": "command",
#         "command": "/Users/abrichr/oa/src/openadapt-ml/scripts/check_azure_resources.sh"
#       }
#     ]
#   }
# }

cd "$(dirname "$0")/.." || exit 0

# Run the Python resource tracker, suppress errors
uv run python -m openadapt_ml.benchmarks.resource_tracker 2>/dev/null || true

exit 0
