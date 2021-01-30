#!/bin/bash
# Infer with the LSTMs
# Usage: infer_lstms.sh DATASET GPU_NUM GROUP
set -ex

if [[ "$1" =~ ^(bach|musicnet|nes|lmd)$ ]]
then
  for GROUP in "${@:3}"
  do
    case "$GROUP" in
      default)
        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/default/" \
          -d "$1" -p $PATIENCE -g "$2" -q

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/default_embedding/" \
          -d "$1" -p $PATIENCE -g "$2" -q -pe -te -fi

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/default_embedding_onsethint/" \
          -d "$1" -p $PATIENCE -g "$2" -q -pe -te -fi -oh

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/default_embedding_onsethint_duration/" \
          -d "$1" -p $PATIENCE -g "$2" -q -pe -te -fi -oh -di -de
        ;;

      ar|autoregressive)
        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/autoregressive/" \
          -d "$1" -p $PATIENCE -g "$2" -q -ar -or

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding/" \
          -d "$1" -p $PATIENCE -g "$2" -q -ar -pe -te -fi -or

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint/" \
          -d "$1" -p $PATIENCE -g "$2" -q -ar -pe -te -fi -oh -or

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint_duration/" \
          -d "$1" -p $PATIENCE -g "$2" -q -ar -pe -te -fi -oh -di -de -or

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/autoregressive/" \
          -d "$1" -p $PATIENCE -g "$2" -q -ar

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding/" \
          -d "$1" -p $PATIENCE -g "$2" -q -ar -pe -te -fi

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint/" \
          -d "$1" -p $PATIENCE -g "$2" -q -ar -pe -te -fi -oh

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint_duration/" \
          -d "$1" -p $PATIENCE -g "$2" -q -ar -pe -te -fi -oh -di -de
        ;;

      bi|bidirectional)
        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/bidirectional/" \
          -d "$1" -p $PATIENCE -g "$2" -q -bi

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding/" \
          -d "$1" -p $PATIENCE -g "$2" -q -bi -pe -te -fi

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding_onsethint/" \
          -d "$1" -p $PATIENCE -g "$2" -q -bi -pe -te -fi -oh

        python3 arranger/lstm/infer.py \
          -i "$HOME/data/arranger/$1/preprocessed/" \
          -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding_onsethint_duration/" \
          -d "$1" -p $PATIENCE -g "$2" -q -bi -pe -te -fi -oh -di -de
        ;;

      *)
        echo "Skip unrecognized group : $GROUP"
        ;;
    esac
  done
else
  echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
fi
