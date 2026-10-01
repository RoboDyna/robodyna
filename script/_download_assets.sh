#!/usr/bin/env bash
# Download the minimal asset package needed by the shipped RoboDyna tasks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/assets/_download.py" "$@"
echo "Configuring local embodiment paths ..."
python "$ROOT/script/update_embodiment_config_path.py"
