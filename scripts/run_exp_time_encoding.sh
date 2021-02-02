#!/bin/bash
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

mkdir -p "$HOME/data/arranger/exp/ablation/$1/lstm"

python3 arranger/lstm/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_raw_time/" \
  -d "$1" -s $STEPS_PER_EPOCH -g "$2" -pe -fi

python3 arranger/lstm/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_time_embedding/" \
  -d "$1" -s $STEPS_PER_EPOCH -g "$2" -pe -fi -te

python3 arranger/lstm/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_raw_beat_position/" \
  -d "$1" -s $STEPS_PER_EPOCH -g "$2" -pe -fi -bp

python3 arranger/lstm/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_raw_time/" \
  -d "$1" -s $STEPS_PER_EPOCH -g "$2" -bi -pe -fi

python3 arranger/lstm/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_time_embedding/" \
  -d "$1" -s $STEPS_PER_EPOCH -g "$2" -bi -pe -fi -te

python3 arranger/lstm/train.py \
  -i "$HOME/data/arranger/$1/preprocessed/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_raw_beat_position/" \
  -d "$1" -s $STEPS_PER_EPOCH -g "$2" -bi -pe -fi -bp


python3 arranger/lstm/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_raw_time/" \
  -d "$1" -g "$2" -pe -fi

python3 arranger/lstm/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_time_embedding/" \
  -d "$1" -g "$2" -pe -fi -te

python3 arranger/lstm/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/default_raw_beat_position/" \
  -d "$1" -g "$2" -pe -fi -bp

python3 arranger/lstm/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_raw_time/" \
  -d "$1" -g "$2" -bi -pe -fi

python3 arranger/lstm/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_time_embedding/" \
  -d "$1" -g "$2" -bi -pe -fi -te

python3 arranger/lstm/infer.py \
  -i "$HOME/data/arranger/$1/json/" \
  -o "$HOME/data/arranger/exp/ablation/$1/lstm/bidirectional_raw_beat_position/" \
  -d "$1" -g "$2" -bi -pe -fi -bp
