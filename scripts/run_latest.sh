#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git pull

latest_config=$(
  git ls-files 'config/*.toml' | while read -r f; do
    git log -1 --format="%ct $f" -- "$f"
  done | sort -t' ' -k1,1nr | head -1 | cut -d' ' -f2-
)

if [[ -z "${latest_config}" ]]; then
  echo "No tracked config files found under config/" >&2
  exit 1
fi

NAME="$(basename "${latest_config}" .toml)"
echo "Using config: ${NAME}"

.venv/bin/python train.py "${NAME}"
.venv/bin/python simulate.py "${NAME}"
