#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${DEVICE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

if [[ -z "${DTYPE:-}" ]]; then
  if [[ "$DEVICE" == cuda* ]]; then
    DTYPE="bfloat16"
  else
    DTYPE="float32"
  fi
fi

python3 train_class.py \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --no-distributed \
  "$@"
