# JAX + GPU Singularity Setup (NYU Torch style)

This uses a writable ext3 overlay + Miniforge, then installs `jax[cuda12]` via pip.

## 1) Build the overlay (run on a compute node)

```bash
sbatch --output=install-jax.log --cpus-per-task=4 --mem=20GB --time=01:00:00 --wrap='bash singularity/create-overlay-jax.sh'
```

Defaults:
- base image: `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif`
- output overlay: `${SCRATCH}/img/jax-overlay.ext3`

Optional environment overrides:
- `BASE_IMAGE`
- `OVERLAY_NAME`
- `OVERLAY_SIZE_MB`
- `DEST_DIR`

## 2) Test JAX + GPU in a batch job

```bash
sbatch <(bash singularity/gen-batch-jax.sh "python singularity/test_jax.py")
```

Expected output includes:
- `default_backend=gpu`
- at least one `CudaDevice` in `devices`
- `has_gpu=True`
