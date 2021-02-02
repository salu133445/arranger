#!/bin/bash
set -ex

for DATASET in bach musicnet nes lmd
do
  mkdir -p "$HOME/data/arranger/exp/samples/${DATASET}/"

  # Baselines
  cd "$HOME/data/arranger/exp/$DATASET/common/default/"
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}/common_default.zip" samples/*
  cd -

  cd "$HOME/data/arranger/exp/$DATASET/common/default/oracle/"
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_common_default_oracle.zip" samples/*
  cd -

  cd "$HOME/data/arranger/exp/$DATASET/zone/permutation/"
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_zone_permutation.zip" samples/*
  cd -

  cd "$HOME/data/arranger/exp/$DATASET/zone/permutation/oracle/"
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_zone_permutation_oracle.zip" samples/*
  cd -

  cd "$HOME/data/arranger/exp/$DATASET/closest/default/"
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_closest_default.zip" samples/*
  cd -

  cd "$HOME/data/arranger/exp/$DATASET/closest/states/"
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_closest_states.zip" samples/*
  cd -

  # Models
  cd "$HOME/data/arranger/exp/$DATASET/lstm/default_embedding/"
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_default_embedding.zip" samples/*
  cd -

  cd "$HOME/data/arranger/exp/$DATASET/lstm/default_embedding_onsethint_duration/"
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_default_embedding_onsethint_duration.zip" samples/*
  cd -

  if [ $DATASET != lmd ]
  then
    cd "$HOME/data/arranger/exp/$DATASET/lstm/bidirectional_embedding/"
    zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_bidirectional_embedding.zip" samples/*
    cd -
  fi

  cd "$HOME/data/arranger/exp/$DATASET/lstm/bidirectional_embedding_onsethint_duration/"
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_bidirectional_embedding_onsethint_duration.zip" samples/*
  cd -
done

cd "$HOME/data/arranger/exp/"
zip -rq "$HOME/data/arranger/exp/samples.zip" samples/*.zip
cd -
