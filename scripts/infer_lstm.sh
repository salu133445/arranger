#!/bin/bash
# Infer with the LSTM models.
# Usage: infer_lstms.sh DATASET GPU_NUM GROUP [GROUP ...]
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
        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/default/" \
        #   -d "$1" -g "$2"

        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/default_embedding/" \
        #   -d "$1" -g "$2" -pe -bp -be -fi

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/json/" \
          -o "$HOME/data/arranger/exp/$1/lstm/default_embedding_onsethint/" \
          -d "$1" -g "$2" -pe -bp -be -fi -oh

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/json/" \
          -o "$HOME/data/arranger/exp/$1/lstm/default_embedding_onsethint_duration/" \
          -d "$1" -g "$2" -pe -bp -be -fi -oh -di -de
        ;;

      ar|autoregressive)
        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/autoregressive/" \
        #   -d "$1" -g "$2" -ar -or

        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi -or

        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi -oh -or

        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint_duration/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi -oh -di -de -or

        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/autoregressive/" \
        #   -d "$1" -g "$2" -ar

        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi

        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi -oh

        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint_duration/" \
        #   -d "$1" -g "$2" -ar -pe -be -fi -oh -di -de
        ;;

      bi|bidirectional)
        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/bidirectional/" \
        #   -d "$1" -g "$2" -bi

        # python3 arranger/lstm/infer.py \
        #   -i "$HOME/data/arranger/$1/json/" \
        #   -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding/" \
        #   -d "$1" -g "$2" -bi -pe -bp -be -fi

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/json/" \
          -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding_onsethint/" \
          -d "$1" -g "$2" -bi -pe -bp -be -fi -oh

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/json/" \
          -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding_onsethint_duration/" \
          -d "$1" -g "$2" -bi -pe -bp -be -fi -oh -di -de
        ;;

      *)
        echo "Skip unrecognized group : $GROUP"
        ;;
    esac
  done
else
  echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
fi
