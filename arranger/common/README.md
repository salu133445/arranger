# Baseline model - Most-common-label algorithm

```sh
# Learn the most common label
python3 arranger/common/learn.py -i data/bach/json/ -o exp/bach/common/default -d bach

# Inference with the most common label
python3 arranger/zone/inference.py -i data/bach/json/ -o exp/bach/common/default -d bach

# Inference with the most common label (oracle)
python3 arranger/common/inference.py -i data/bach/json/ -o exp/bach/common/default_oracle -d bach -or
```
