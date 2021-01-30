#!/bin/bash
# Train the Transformer models
# Usage: train_transformer.sh DATASET GPU_NUM GROUP [GROUP ...]
#
# DATASET : {'bach', 'musicnet', 'nes', 'lmd'}
# GROUP : list of {'default', 'autoregressive'}
set -ex

case "$1" in
  bach|musicnet)
    PATIENCE=10
    ;;
  nes|lmd)
    PATIENCE=5
    ;;
  *)
    echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
    ;;
esac

if [[ -z "$3" ]]; then
  echo "Please provide the group."
  exit
fi

for GROUP in "${@:3}"
do
  case "$GROUP" in
    de|default)
      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/default/" \
        -d "$1" -p $PATIENCE -g "$2"

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/default_embedding/" \
        -d "$1" -p $PATIENCE -g "$2" -pe -te -fi

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/default_embedding_onsethint/" \
        -d "$1" -p $PATIENCE -g "$2" -pe -te -fi -oh

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/default_embedding_onsethint_duration/" \
        -d "$1" -p $PATIENCE -g "$2" -pe -te -fi -oh -di -de
      ;;

    ar|autoregressive)
      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/autoregressive/" \
        -d "$1" -p $PATIENCE -g "$2" -ar

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding/" \
        -d "$1" -p $PATIENCE -g "$2" -ar -pe -te -fi

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint/" \
        -d "$1" -p $PATIENCE -g "$2" -ar -pe -te -fi -oh

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint_duration/" \
        -d "$1" -p $PATIENCE -g "$2" -ar -pe -te -fi -oh -di -de
      ;;

    *)
      echo "Skip unrecognized group : $GROUP"
      ;;
  esac
done
