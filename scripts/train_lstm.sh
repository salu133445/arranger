#!/bin/bash
# Train the LSTMs
# Usage: train_lstms.sh DATASET GPU_NUM GROUP
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

for GROUP in "${@:3}"
do
  case "$GROUP" in
    default)
      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/default/" \
        -d "$1" -p $PATIENCE -g "$2"

      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/default_embedding/" \
        -d "$1" -p $PATIENCE -g "$2" -pe -te -fi

      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/default_embedding_onsethint/" \
        -d "$1" -p $PATIENCE -g "$2" -pe -te -fi -oh

      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/default_embedding_onsethint_duration/" \
        -d "$1" -p $PATIENCE -g "$2" -pe -te -fi -oh -di -de
      ;;

    ar|autoregressive)
      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/autoregressive/" \
        -d "$1" -p $PATIENCE -g "$2" -ar

      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding/" \
        -d "$1" -p $PATIENCE -g "$2" -ar -pe -te -fi

      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint/" \
        -d "$1" -p $PATIENCE -g "$2" -ar -pe -te -fi -oh

      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint_duration/" \
        -d "$1" -p $PATIENCE -g "$2" -ar -pe -te -fi -oh -di -de
      ;;

    bi|bidirectional)
      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/bidirectional/" \
        -d "$1" -p $PATIENCE -g "$2" -bi

      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding/" \
        -d "$1" -p $PATIENCE -g "$2" -bi -pe -te -fi

      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding_onsethint/" \
        -d "$1" -p $PATIENCE -g "$2" -bi -pe -te -fi -oh

      python3 arranger/lstm/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding_onsethint_duration/" \
        -d "$1" -p $PATIENCE -g "$2" -bi -pe -te -fi -oh -di -de
      ;;

    *)
      echo "Skip unrecognized group : $GROUP"
      ;;
  esac
done
