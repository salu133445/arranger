#!/bin/bash
set -ex

if [[ "$1" =~ ^(bach|musicnet|nes|lmd)$ ]]
then
  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/plain/" \
    -d "$1" -g "$2" -q

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/embedding/" \
    -d "$1" -g "$2" -q -pe -te -fi

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/embedding_onsethint/" \
    -d "$1" -g "$2" -q -pe -te -fi -oh

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/embedding_onsethint_duration/" \
    -d "$1" -g "$2" -q -pe -te -fi -oh -di -de

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding/" \
    -d "$1" -g "$2" -q -ar -pe -te -fi -or

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint/" \
    -d "$1" -g "$2" -q -ar -pe -te -fi -oh -or

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint_duration/" \
    -d "$1" -g "$2" -q -ar -pe -te -fi -oh -or

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/autoregressive/" \
    -d "$1" -g "$2" -q -ar -or

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/bidirectional/" \
    -d "$1" -g "$2" -q -bi

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding/" \
    -d "$1" -g "$2" -q -bi -pe -te -fi

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding_onsethint/" \
    -d "$1" -g "$2" -q -bi -pe -te -fi -oh

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/bidirectional_embedding_onsethint_duration/" \
    -d "$1" -g "$2" -q -bi -pe -te -fi -oh -di -de

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/autoregressive/" \
    -d "$1" -g "$2" -q -ar

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding/" \
    -d "$1" -g "$2" -q -ar -pe -te -fi

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint/" \
    -d "$1" -g "$2" -q -ar -pe -te -fi -oh

  python3 arranger/lstm/inference.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/$1/lstm/autoregressive_embedding_onsethint_duration/" \
    -d "$1" -g "$2" -q -ar -pe -te -fi -oh
else
  echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
fi
