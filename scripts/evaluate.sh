#!/bin/bash
set -ex
EXP_DIR="$HOME/data/arranger/exp"
case "$2" in
  common)
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/common/default/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/common/default/oracle/"
    ;;

  zone)
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/zone/default/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/zone/default/oracle/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/zone/permutation/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/zone/permutation/oracle/"
    ;;

  closest)
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/closest/default/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/closest/states/"
    ;;

  baselines)
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/common/default/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/common/default/oracle/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/zone/default/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/zone/default/oracle/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/zone/permutation/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/zone/permutation/oracle/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/closest/default/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/closest/states/"
    ;;

  lstm)
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/lstm/default/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/lstm/default_embedding/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/lstm/default_embedding_onsethint/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/lstm/default_embedding_onsethint_duration/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/lstm/bidirectional/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/lstm/bidirectional_embedding/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/lstm/bidirectional_embedding_onsethint/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/lstm/bidirectional_embedding_onsethint_duration/"
    ;;

  transformer)
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/transformer/default/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/transformer/default_embedding/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/transformer/default_embedding_onsethint/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/transformer/default_embedding_onsethint_duration/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/transformer/lookahead/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/transformer/lookahead_embedding/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/transformer/lookahead_embedding_onsethint/"
    python3 arranger/evaluate.py -i "$EXP_DIR/$1/transformer/lookahead_embedding_onsethint_duration/"
    ;;

  *)
    echo "Skip unrecognized keyword : $2"
    ;;
esac
