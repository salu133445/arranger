#!/bin/bash
# Learn the most common label.
# Usage: learn_common.sh DATASET N_JOBS
set -ex

python3 arranger/common/learn.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/common/default" \
  -d "$1" -j "${2:-1}"
