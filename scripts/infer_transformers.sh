#!/bin/bash
# Infer with the Transformers
# Usage: infer_lstms.sh DATASET GPU_NUM GROUP
set -ex

if [[ "$1" =~ ^(bach|musicnet|nes|lmd)$ ]]
then
  for GROUP in "${@:3}"
  do
    case "$GROUP" in
      default)
        python3 arranger/transformer/infer.py \
        -i "$HOME/data/arranger/$1/preprocessed/" \
        -o "$HOME/data/arranger/exp/$1/transformer/default/" \
        -d "$1" -g "$2"

        python3 arranger/transformer/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/transformer/default_embedding/" \
          -d "$1" -g "$2" -pe -te -fi

        python3 arranger/transformer/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/transformer/default_embedding_onsethint/" \
          -d "$1" -g "$2" -pe -te -fi -oh

        python3 arranger/transformer/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/transformer/default_embedding_onsethint_duration/" \
          -d "$1" -g "$2" -pe -te -fi -oh -di -de
        ;;

      ar|autoregressive)
        python3 arranger/transformer/train.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/transformer/autoregressive/" \
          -d "$1" -g "$2" -ar -or

        python3 arranger/transformer/train.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding/" \
          -d "$1" -g "$2" -ar -pe -te -fi -or

        python3 arranger/transformer/train.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint/" \
          -d "$1" -g "$2" -ar -pe -te -fi -oh -or

        python3 arranger/transformer/train.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint_duration/" \
          -d "$1" -g "$2" -ar -pe -te -fi -oh -di -de -or

        python3 arranger/transformer/train.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/transformer/autoregressive/" \
          -d "$1" -g "$2" -ar

        python3 arranger/transformer/train.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding/" \
          -d "$1" -g "$2" -ar -pe -te -fi

        python3 arranger/transformer/train.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint/" \
          -d "$1" -g "$2" -ar -pe -te -fi -oh

        python3 arranger/transformer/train.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint_duration/" \
          -d "$1" -g "$2" -ar -pe -te -fi -oh -di -de
        ;;

      *)
        echo "Skip unrecognized group : $GROUP"
        ;;
    esac
  done
else
  echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
fi
