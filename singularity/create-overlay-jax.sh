#!/usr/bin/env bash
set -euo pipefail

CURR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_IMAGE=${BASE_IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}
OVERLAY_NAME=${OVERLAY_NAME:-jax-overlay.ext3}
OVERLAY_SIZE_MB=${OVERLAY_SIZE_MB:-16384}
DEST_DIR=${DEST_DIR:-${SCRATCH:-$HOME}/img}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
OVERLAY="$TMP/$OVERLAY_NAME"

if command -v apptainer >/dev/null 2>&1; then
  SING_CMD=apptainer
elif command -v singularity >/dev/null 2>&1; then
  SING_CMD=singularity
else
  echo "Neither apptainer nor singularity is available on PATH."
  exit 1
fi

"$SING_CMD" overlay create --size "$OVERLAY_SIZE_MB" "$OVERLAY"

"$SING_CMD" exec --fakeroot --pwd "$CURR_DIR" --overlay "$OVERLAY":rw "$BASE_IMAGE" /bin/bash <<'EOS'
set -euo pipefail

wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
rm Miniforge3-Linux-x86_64.sh

cat > /ext3/env.sh <<'EOM'
#!/bin/bash
unset -f which
export HOME=/ext3/home
source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
EOM

source /ext3/env.sh
mkdir -p "$HOME"

conda update -n base conda -y
conda clean --all --yes
conda install pip -y

python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda12]"
python -m pip install --upgrade numpy pandas matplotlib pytest
EOS

mkdir -p "$DEST_DIR"
rm -f "$DEST_DIR/$OVERLAY_NAME"
mv "$OVERLAY" "$DEST_DIR/$OVERLAY_NAME"

echo "Created overlay: $DEST_DIR/$OVERLAY_NAME"
