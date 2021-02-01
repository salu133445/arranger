#!/bin/bash
# Train the Transformer models.
# Usage: train_transformer.sh DATASET GPU_NUM GROUP [GROUP ...]
#
# DATASET : {'bach', 'musicnet', 'nes', 'lmd'}
# GROUP : list of {'default', 'autoregressive'}
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

if [[ -z "$3" ]]; then
  echo "Please specify the group(s)."
  exit
fi

for GROUP in "${@:3}"
do
  case "$GROUP" in
    de|default)
      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/default/" \
        -d "$1" -s $STEPS_PER_EPOCH -g "$2"

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/default_embedding/" \
        -d "$1" -s $STEPS_PER_EPOCH -g "$2" -pe -be -fi

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/default_embedding_onsethint/" \
        -d "$1" -s $STEPS_PER_EPOCH -g "$2" -pe -be -fi -oh

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/default_embedding_onsethint_duration/" \
        -d "$1" -s $STEPS_PER_EPOCH -g "$2" -pe -be -fi -oh -di -de
      ;;

    ar|autoregressive)
      # python3 arranger/transformer/train.py \
      #   -i "$HOME/data/arranger/$1/preprocessed/" \
      #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive/" \
      #   -d "$1" -s $STEPS_PER_EPOCH -g "$2" -ar

      # python3 arranger/transformer/train.py \
      #   -i "$HOME/data/arranger/$1/preprocessed/" \
      #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding/" \
      #   -d "$1" -s $STEPS_PER_EPOCH -g "$2" -ar -pe -be -fi

      # python3 arranger/transformer/train.py \
      #   -i "$HOME/data/arranger/$1/preprocessed/" \
      #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint/" \
      #   -d "$1" -s $STEPS_PER_EPOCH -g "$2" -ar -pe -be -fi -oh

      # python3 arranger/transformer/train.py \
      #   -i "$HOME/data/arranger/$1/preprocessed/" \
      #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint_duration/" \
      #   -d "$1" -s $STEPS_PER_EPOCH -g "$2" -ar -pe -be -fi -oh -di -de
      ;;

    la|lookahead)
      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/lookahead/" \
        -d "$1" -s $STEPS_PER_EPOCH -g "$2" -lm

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/lookahead_embedding/" \
        -d "$1" -s $STEPS_PER_EPOCH -g "$2" -lm -pe -be -fi

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/lookahead_embedding_onsethint/" \
        -d "$1" -s $STEPS_PER_EPOCH -g "$2" -lm -pe -be -fi -oh

      python3 arranger/transformer/train.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/lookahead_embedding_onsethint_duration/" \
        -d "$1" -s $STEPS_PER_EPOCH -g "$2" -lm -pe -be -fi -oh -di -de
      ;;

    *)
      echo "Skip unrecognized group : $GROUP"
      ;;
  esac
done
