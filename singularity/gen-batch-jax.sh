#!/usr/bin/env bash
set -euo pipefail

CURR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_IMAGE=${BASE_IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}
OVERLAY_NAME=${OVERLAY_NAME:-jax-overlay.ext3}
OVERLAY_PATH=${OVERLAY_PATH:-${SCRATCH:-$HOME}/img/$OVERLAY_NAME}

cat <<EOT
#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

module purge

exec singularity exec --nv \\
  --writable-tmpfs \\
  --pwd $CURR_DIR \\
  --overlay $OVERLAY_PATH:ro \\
  $BASE_IMAGE \\
  /bin/bash -c "source /ext3/env.sh; exec \$@"
EOT
