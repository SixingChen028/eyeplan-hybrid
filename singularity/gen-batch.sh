CURR_DIR=$(dirname $(realpath -s $0))
BASE_IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY_NAME=overlay.ext3
OVERLAY=$SCRATCH/img/$OVERLAY_NAME

cat <<EOT
#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

module purge

exec singularity exec --nv \
    --writable-tmpfs \
    --pwd $CURR_DIR \
    --overlay $OVERLAY:ro \
    $BASE_IMAGE \
    /bin/bash -c "source /ext3/env.sh; exec $@"
EOT
