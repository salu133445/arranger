#!/bin/bash
set -ex

if [[ "$1" =~ ^(bach|musicnet|nes|lmd)$ ]]
then
  python3 arranger/transformer/inference.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/$1/transformer/default/" \
    -d "$1" -g "$2" -q

  python3 arranger/transformer/inference.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/$1/transformer/embedding/" \
    -d "$1" -g "$2" -q -pe -te -fi

  python3 arranger/transformer/inference.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/$1/transformer/embedding_hints/" \
    -d "$1" -g "$2" -q -pe -te -fi -oh -ph

  python3 arranger/transformer/inference.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/$1/transformer/embedding_hints_augmentation/" \
    -d "$1" -g "$2" -q -pe -te -fi -oh -ph -au -go

  python3 arranger/transformer/inference.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/$1/transformer/embedding_hints_augmentation/" \
    -d "$1" -g "$2" -q -pe -te -fi -oh -ph -au

  python3 arranger/transformer/inference.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/$1/transformer/embedding_hints_duration/" \
    -d "$1" -g "$2" -q -pe -te -fi -oh -ph -du -de

  python3 arranger/transformer/inference.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/$1/transformer/embedding_hints_duration_bidirectional/" \
    -d "$1" -g "$2" -q -pe -te -fi -oh -ph -du -de -bi

  python3 arranger/transformer/inference.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/$1/transformer/embedding_hints_duration_bidirectional_augmentation/" \
    -d "$1" -g "$2" -q -pe -te -fi -oh -ph -du -de -bi -go

  python3 arranger/transformer/inference.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/$1/transformer/embedding_hints_duration_bidirectional_augmentation/" \
    -d "$1" -g "$2" -q -pe -te -fi -oh -ph -du -de -bi -au
else
  echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
fi
