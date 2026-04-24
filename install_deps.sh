#!/usr/bin/env bash
set -euo pipefail

USE_GPU=false
if [[ "${1:-}" == "--gpu" ]]; then
    USE_GPU=true
    shift
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--gpu]"
    exit 1
fi

# Install uv.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install modern Python and create venv.
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install deps.
uv pip install -U pip
if [[ "${USE_GPU}" == "true" ]]; then
    uv pip install -U "jax[cuda12]"
else
    uv pip install -U "jax"
fi
uv pip install -U numpy pandas matplotlib pytest
