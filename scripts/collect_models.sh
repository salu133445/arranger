#!/bin/bash
# set -ex

in_dir=$HOME/data/arranger/exp
out_dir=$HOME/data/arranger/models
for dataset in bach musicnet nes lmd
do
  # LSTMs
  for prefix in default bidirectional
  do
    name=$prefix
    for component in "" embedding onsethint duration
    do
      if [[ -n $component ]]
      then
        name=${name}_$component
      fi
      mkdir -p "$out_dir/$dataset/lstm/$name"
      cp "$in_dir/$dataset/lstm/$name/best_model.hdf5" "$out_dir/$dataset/lstm/$name/best_model.hdf5"
    done
  done

  # Transformers
  for prefix in default lookahead
  do
    name=$prefix
    for component in "" embedding onsethint duration
    do
      if [[ -n $component ]]
      then
        name=${name}_$component
      fi
      mkdir -p "$out_dir/$dataset/transformer/$name"
      cp "$in_dir/$dataset/transformer/$name/best_model.hdf5" "$out_dir/$dataset/transformer/$name/best_model.hdf5"
    done
  done

  # Zone-based
  for name in default permutation
  do
    mkdir -p "$out_dir/$dataset/zone/$name"
    cp "$in_dir/$dataset/zone/$name/optimal_boundaries.txt" "$out_dir/$dataset/zone/$name/optimal_boundaries.txt"
    cp "$in_dir/$dataset/zone/$name/optimal_permutation.txt" "$out_dir/$dataset/zone/$name/optimal_permutation.txt"
  done
done
