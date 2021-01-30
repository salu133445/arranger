#!/bin/bash
set -ex

python3 arranger/zone/inference.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/zone/default" \
  -d "$1" -j "${2:-1}"

python3 arranger/zone/inference.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/zone/default_oracle" \
  -d "$1" -or -j "${2:-1}"

python3 arranger/zone/inference.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/zone/permutation" \
  -d "$1" -p -j "${2:-1}"

python3 arranger/zone/inference.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/$1/zone/permutation_oracle" \
  -d "$1" -p -or -j "${2:-1}"
