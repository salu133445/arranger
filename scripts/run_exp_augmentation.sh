#!/bin/bash
set -ex

if [[ "$1" =~ ^(bach|musicnet|nes|lmd)$ ]]
then
  mkdir -p "$HOME/data/arranger/exp/ablation/$1/lstm"

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_embedding_no_augmentation/" \
    -d "$1" -s 500 -g 3 -pe -fi -bp -be -na

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_embedding_no_augmentation/" \
    -d "$1" -s 500 -g 3 -bi -pe -fi -bp -be -na

  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_embedding_no_augmentation/" \
    -d "$1" -g 3 -pe -fi -bp -be -na

  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_embedding_no_augmentation/" \
    -d "$1" -g 3 -bi -pe -fi -bp -be -na
else
  echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
fi
