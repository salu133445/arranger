#!/bin/bash
# Infer with the closest-pitch algorithm.
# Usage: infer_closest.sh DATASET N_JOBS
set -ex

python3 arranger/closest/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/closest/default" \
  -d "$1" -j "${2:-1}"

python3 arranger/closest/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/closest/states" \
  -d "$1" -s -j "${2:-1}"
