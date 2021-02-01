#!/bin/bash
# Infer with the most-common-label algorithm.
# Usage: infer_common.sh DATASET N_JOBS
set -ex

python3 arranger/common/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/common/default" \
  -d "$1" -j "${2:-1}"

python3 arranger/common/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/common/default" \
  -d "$1" -or -j "${2:-1}"
