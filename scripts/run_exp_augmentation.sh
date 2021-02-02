#!/bin/bash
set -ex

case "$1" in
  bach|musicnet|nes)
    STEPS_PER_EPOCH=500
    ;;
  lmd)
    STEPS_PER_EPOCH=5000
    ;;
  *)
    echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
    ;;
esac

mkdir -p "$HOME/data/arranger/exp/ablation/$1/lstm"

python3 arranger/lstm/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_embedding_no_augmentation/" \
  -d "$1" -s $STEPS_PER_EPOCH -g "$2" -pe -fi -bp -be -na

python3 arranger/lstm/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_embedding_no_augmentation/" \
  -d "$1" -s $STEPS_PER_EPOCH -g "$2" -bi -pe -fi -bp -be -na

python3 arranger/lstm/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_embedding_light_augmentation/" \
  -d "$1" -s $STEPS_PER_EPOCH -g "$2" -pe -fi -bp -be -pr 1 1

python3 arranger/lstm/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_embedding_light_augmentation/" \
  -d "$1" -s $STEPS_PER_EPOCH -g "$2" -bi -pe -fi -bp -be -pr 1 1

python3 arranger/lstm/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_embedding_no_augmentation/" \
  -d "$1" -g "$2" -pe -fi -bp -be

python3 arranger/lstm/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_embedding_no_augmentation/" \
  -d "$1" -g "$2" -bi -pe -fi -bp -be

python3 arranger/lstm/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_embedding_light_augmentation/" \
  -d "$1" -g "$2" -pe -fi -bp -be

python3 arranger/lstm/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_embedding_light_augmentation/" \
  -d "$1" -g "$2" -bi -pe -fi -bp -be
