#!/bin/bash
set -ex

case "$1" in
  bach|musicnet)
    PATIENCE=10
    ;;
  nes|lmd)
    PATIENCE=3
    ;;
  *)
    echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
    ;;
esac

python3 arranger/transformer/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/$1/transformer/plain/" \
  -d "$1" -p $PATIENCE -g "$2" -q

python3 arranger/transformer/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/$1/transformer/embedding/" \
  -d "$1" -p $PATIENCE -g "$2" -q -pe -te -fi

python3 arranger/transformer/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/$1/transformer/embedding_onsethint/" \
  -d "$1" -p $PATIENCE -g "$2" -q -pe -te -fi -oh

python3 arranger/transformer/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/$1/transformer/embedding_onsethint_duration/" \
  -d "$1" -p $PATIENCE -g "$2" -q -pe -te -fi -oh -di -de

python3 arranger/transformer/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/$1/transformer/autoregressive/" \
  -d "$1" -p $PATIENCE -g "$2" -q -ar

python3 arranger/transformer/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding/" \
  -d "$1" -p $PATIENCE -g "$2" -q -ar -pe -te -fi

python3 arranger/transformer/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint/" \
  -d "$1" -p $PATIENCE -g "$2" -q -ar -pe -te -fi -oh

python3 arranger/transformer/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint_duration/" \
  -d "$1" -p $PATIENCE -g "$2" -q -ar -pe -te -fi -oh
