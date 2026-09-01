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

# Install uv if not already present.
if ! command -v uv &>/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Create venv if it doesn't already exist.
if [[ ! -d .venv ]]; then
    uv python install 3.12
    uv venv --python 3.12 .venv
fi
source .venv/bin/activate

# Install deps only when missing.
uv pip install pip
if [[ "${USE_GPU}" == "true" ]]; then
    uv pip install "jax[cuda12]"
else
    uv pip install "jax"
fi
uv pip install numpy pandas matplotlib pytest

echo "Running train.py smoke test..."
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/nn-python-smoke.XXXXXX")"
cleanup_smoke() {
    rm -rf "$SMOKE_ROOT"
}
trap cleanup_smoke EXIT

JAX_PLATFORMS=cpu python train.py config/test_single.toml \
    --path "$SMOKE_ROOT/results" \
    --num_updates 1 \
    --num_envs 4 \
    --rollout_length 4
