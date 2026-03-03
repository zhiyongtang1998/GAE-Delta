#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

HDF5_PATH="${1:?Usage: $0 <hdf5_path> [fi_network_path]}"
FI_PATH="${2:-data/example/toy_fi_network.txt}"

echo "==> Running GAE-Δ pipeline"
echo "    Data:       ${HDF5_PATH}"
echo "    FI Network: ${FI_PATH}"
echo ""

python -m gae_delta.pipeline.runner \
    data.hdf5_path="${HDF5_PATH}" \
    data.fi_network_path="${FI_PATH}" \
    "$@"
