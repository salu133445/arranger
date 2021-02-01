#!/bin/bash
# Learn the optimal zone setting.
# Usage: infer_zone.sh DATASET N_JOBS
set -ex

python3 arranger/zone/learn.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/zone/default" \
  -d "$1" -j "${2:-1}"

python3 arranger/zone/learn.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/zone/permutation" \
  -d "$1" -p -j "${2:-1}"
