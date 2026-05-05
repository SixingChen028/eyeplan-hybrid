#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-$(ls -t config/* 2>/dev/null | head -n 1)}"

if [[ -z "${CONFIG}" ]]; then
  echo "No config provided and no files found in config/"
  exit 1
fi

CMD=(
  sbatch
  --account=torch_pr_279_general
  --cpus-per-task=1
  --mem=3G
  --output=./log/%j
  --time=2:00:00
  --gres=gpu:1
  "--wrap=.venv/bin/python train_parallel.py ${CONFIG}"
)

echo "Command to run:"
printf ' %q' "${CMD[@]}"
echo

read -r -p "Submit job? [y/N] " CONFIRM
if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
  echo "Cancelled."
  exit 0
fi

"${CMD[@]}"
