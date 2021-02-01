#!/bin/bash
set -ex

if [[ "$1" =~ ^(bach|musicnet|nes|lmd)$ ]]
then
  mkdir -p "$HOME/data/arranger/exp/ablation/$1/lstm"

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_raw_time/" \
    -d "$1" -s 500 -g 3 -pe -fi

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_time_embedding/" \
    -d "$1" -s 500 -g 3 -pe -fi -te

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_raw_beat_position/" \
    -d "$1" -s 500 -g 3 -pe -fi -bp

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_beat_position_embedding/" \
    -d "$1" -s 500 -g 3 -pe -fi -bp -be

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_raw_time/" \
    -d "$1" -s 500 -g 3 -bi -pe -fi

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_time_embedding/" \
    -d "$1" -s 500 -g 3 -bi -pe -fi -te

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_raw_beat_position/" \
    -d "$1" -s 500 -g 3 -bi -pe -fi -bp

  python3 arranger/lstm/train.py \
    -i "$HOME/data/arranger/$1/preprocessed/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_beat_position_embedding/" \
    -d "$1" -s 500 -g 3 -bi -pe -fi -bp -be


  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/infer/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_raw_time/" \
    -d "$1" -g 3 -pe -fi

  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_time_embedding/" \
    -d "$1" -g 3 -pe -fi -te

  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_raw_beat_position/" \
    -d "$1" -g 3 -pe -fi -bp

  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_beat_position_embedding/" \
    -d "$1" -g 3 -pe -fi -bp -be

  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_raw_time/" \
    -d "$1" -g 3 -bi -pe -fi

  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_time_embedding/" \
    -d "$1" -g 3 -bi -pe -fi -te

  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_raw_beat_position/" \
    -d "$1" -g 3 -bi -pe -fi -bp

  python3 arranger/lstm/infer.py \
    -i "$HOME/data/arranger/$1/json/" \
    -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_beat_position_embedding/" \
    -d "$1" -g 3 -bi -pe -fi -bp -be
else
  echo "Dataset must be one of 'bach', 'musicnet', 'nes' or 'lmd'."
fi
