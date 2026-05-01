CURR_DIR=$(dirname $(realpath -s $0))
BASE_IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
# Creating in a temp dir since it might be faster to avoid networked filesystem
TMP=$(mktemp -d)

OVERLAY_NAME=overlay.ext3
OVERLAY=$TMP/$OVERLAY_NAME

singularity overlay create --size 16384 $OVERLAY

singularity exec --pwd=$CURR_DIR --overlay $OVERLAY:rw $BASE_IMAGE /bin/bash <<- 'EOF'

# Install miniforge
wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
rm Miniforge3-Linux-x86_64.sh

# Set path variables
cat > /ext3/env.sh <<- 'EOM'
    export HOME=/ext3/home
    source /ext3/miniforge3/etc/profile.d/conda.sh
    export PATH=/ext3/miniforge3/bin:$PATH
EOM
source /ext3/env.sh
mkdir -p $HOME

# Install conda & pip
conda update -n base conda -y
conda clean --all --yes
conda install pip -y

# Should be in a directory with requirements
pip install --no-cache-dir -r requirements.txt

EOF

# Make destination directory
mkdir -p $SCRATCH/img

# Remove existing overlay in destination
rm -f $SCRATCH/img/$OVERLAY_NAME

# Copy overlay
mv $OVERLAY $SCRATCH/img/$OVERLAY_NAME
