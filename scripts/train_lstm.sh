#!/bin/bash
# Train the LSTM models
# Usage: train_lstms.sh DATASET GPU_NUM GROUP [GROUP ...]
#
# DATASET : {'bach', 'musicnet', 'nes', 'lmd'}
# GROUP : list of {'default', 'autoregressive', 'bidirectional'}
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
