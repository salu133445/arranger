#!/bin/bash
set -ex

for DATASET in bach musicnet nes
do
  cd ~/data/arranger/exp/$DATASET/common/default/samples/
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_common_default.zip" *
  cd -

  cd ~/data/arranger/exp/$DATASET/common/default/oracle/samples/
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_common_default_oracle.zip" *
  cd -


  cd ~/data/arranger/exp/$DATASET/zone/default/samples/
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_zone_default.zip" *
  cd -

  cd ~/data/arranger/exp/$DATASET/zone/default/oracle/samples/
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_zone_default_oracle.zip" *
  cd -

  cd ~/data/arranger/exp/$DATASET/zone/permutation/samples/
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_zone_permutation.zip" *
  cd -

  cd ~/data/arranger/exp/$DATASET/zone/permutation/oracle/samples/
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_zone_permutation_oracle.zip" *
  cd -


  cd ~/data/arranger/exp/$DATASET/closest/default/samples/
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_closest_default.zip" *
  cd -

  cd ~/data/arranger/exp/$DATASET/closest/states/samples/
  zip -rq "$HOME/data/arranger/exp/samples/${DATASET}_closest_states.zip" *
  cd -
done

cd "$HOME/data/arranger/exp/"
zip -rq "$HOME/data/arranger/exp/samples.zip" samples/*.zip
cd -
