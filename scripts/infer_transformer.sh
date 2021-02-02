#!/bin/bash
# Infer with the Transformer models.
# Usage: infer_transformer.sh DATASET GPU_NUM GROUP [GROUP ...]
#
# DATASET : {'bach', 'musicnet', 'nes', 'lmd'}
# GROUP : list of {'default', 'autoregressive'}
set -ex

if [[ -z "$3" ]]; then
  echo "Please provide the group."
  exit
fi

if [[ "$1" =~ ^(bach|musicnet|nes|lmd)$ ]]
then
  for GROUP in "${@:3}"
  do
    case "$GROUP" in
      de|default)
        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/default/" \
        #   -d "$1" -g "$2"

        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/default_embedding/" \
        #   -d "$1" -g "$2" -pe -te -fi

        python3 arranger/transformer/infer.py \
          -i "$HOME/data/arranger/$1/json/" \
          -o "$HOME/data/arranger/exp/$1/transformer/default_embedding_onsethint/" \
          -d "$1" -g "$2" -pe -te -fi -oh

        python3 arranger/transformer/infer.py \
          -i "$HOME/data/arranger/$1/json/" \
          -o "$HOME/data/arranger/exp/$1/transformer/default_embedding_onsethint_duration/" \
          -d "$1" -g "$2" -pe -te -fi -oh -di -de
        ;;

      ar|autoregressive)
        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive/" \
        #   -d "$1" -g "$2" -ar -or

        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi -or

        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi -oh -or

        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint_duration/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi -oh -di -de -or

        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive/" \
        #   -d "$1" -g "$2" -ar

        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi

        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi -oh

        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/autoregressive_embedding_onsethint_duration/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi -oh -di -de
        ;;

      la|lookahead)
        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/lookahead/" \
        #   -d "$1" -g "$2" -lm

        # python3 arranger/transformer/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/transformer/lookahead_embedding/" \
        #   -d "$1" -g "$2" -lm -pe -te -fi

        python3 arranger/transformer/infer.py \
          -i "$HOME/data/arranger/$1/json/" \
          -o "$HOME/data/arranger/exp/$1/transformer/lookahead_embedding_onsethint/" \
          -d "$1" -g "$2" -lm -pe -te -fi -oh

        python3 arranger/transformer/infer.py \
          -i "$HOME/data/arranger/$1/json/" \
          -o "$HOME/data/arranger/exp/$1/transformer/lookahead_embedding_onsethint_duration/" \
          -d "$1" -g "$2" -lm -pe -te -fi -oh -di -de
        ;;

      *)
        echo "Skip unrecognized group : $GROUP"
        ;;
    esac
  done
else
  echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
fi
