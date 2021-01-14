# Baseline model - Zone-based algorithm

## Optimal zone boundaries

> This algorithm assumes the tracks are ordered from highest to lowest pitches.

```sh
# Learn the optimal zone boundaries
python3 arranger/zone/learn.py -i data/bach/json/ -o exp/bach/zone/default -d bach

# Inference with optimal zone boundaries
python3 arranger/zone/inference.py -i data/bach/json/ -o exp/bach/zone/default -d bach

# Inference with optimal zone boundaries (oracle)
python3 arranger/zone/inference.py -i data/bach/json/ -o exp/bach/zone/default_oracle -d bach -or
```

## Optimal zone boundaries and permutation

```sh
# Learn the optimal zone boundaries and permutation
python3 arranger/zone/learn.py -i data/bach/json/ -o exp/bach/zone/permutation -p -d bach

# Inference with optimal zone boundaries and permutation
python3 arranger/zone/inference.py -i data/bach/json/ -o exp/bach/zone/permutation -p -d bach

# Inference with optimal zone boundaries and permutation (oracle)
python3 arranger/zone/inference.py -i data/bach/json/ -o exp/bach/zone/permutation_oracle -d bach -or
```
