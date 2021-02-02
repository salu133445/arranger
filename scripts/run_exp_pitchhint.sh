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

if [[ "$1" =~ ^(bach|musicnet|nes|lmd)$ ]]
then
  mkdir -p "$HOME/data/arranger/exp/ablation/$1/lstm"

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_embedding_pitchhint/" \
    -d "$1" -s $STEPS_PER_EPOCH -g "$2" -pe -bp -be -fi -ph

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_embedding_pitchhint/" \
    -d "$1" -s $STEPS_PER_EPOCH -g "$2" -bi -pe -bp -be -fi -ph

  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_embedding_pitchhint/" \
    -d "$1" -g "$2" -pe -bp -be -fi -ph

  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_embedding_pitchhint/" \
    -d "$1" -g "$2" -bi -pe -bp -be -fi -ph
else
  echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
fi
