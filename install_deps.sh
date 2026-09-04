#!/usr/bin/env bash
set -euo pipefail

USE_GPU=false
JAX_VERSION="0.10.0"
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
PYTHON_BIN=".venv/bin/python"

# Install deps only when missing.
uv pip install --python "$PYTHON_BIN" pip
if [[ "${USE_GPU}" == "true" ]]; then
    uv pip install --python "$PYTHON_BIN" "jax[cuda12]==${JAX_VERSION}"
else
    uv pip install --python "$PYTHON_BIN" "jax==${JAX_VERSION}"
fi
uv pip install --python "$PYTHON_BIN" numpy pandas matplotlib pytest

echo "Running train.py smoke test..."
SMOKE_TMP_ROOT="${TMPDIR:-./tmp}"
mkdir -p "$SMOKE_TMP_ROOT"
SMOKE_ROOT="$(mktemp -d "${SMOKE_TMP_ROOT%/}/nn-python-smoke.XXXXXX")"
cleanup_smoke() {
    rm -rf "$SMOKE_ROOT"
}
trap cleanup_smoke EXIT

JAX_PLATFORMS=cpu "$PYTHON_BIN" train.py config/test_single.toml \
    --path "$SMOKE_ROOT/results" \
    --num_updates 1 \
    --num_envs 4 \
    --rollout_length 4
