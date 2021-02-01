#!/bin/bash
# Infer with the zone-based algorithm.
# Usage: infer_zone.sh DATASET N_JOBS
set -ex

python3 arranger/zone/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/zone/default" \
  -d "$1" -j "${2:-1}"

python3 arranger/zone/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/zone/default" \
  -d "$1" -or -j "${2:-1}"

python3 arranger/zone/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/zone/permutation" \
  -d "$1" -p -j "${2:-1}"

python3 arranger/zone/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/zone/permutation" \
  -d "$1" -p -or -j "${2:-1}"
