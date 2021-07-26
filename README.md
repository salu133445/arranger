Arranger
========

Directory structure
-------------------

```text
├─ analysis         Notebooks for analysis
├─ scripts          Scripts for running experiments
├─ models           Pretrained models
└─ arranger         Main Python module
   ├─ config.yaml   Configuration file
   ├─ data          Code for collecting and processing data
   ├─ common        Most-common algorithm
   ├─ zone          Zone-based algorithm
   ├─ closest       Closest-pitch algorithm
   ├─ lstm          LSTM model
   └─ transformer   Transformer model
```

Prerequisites
-------------

You can install the dependencies by running `pipenv install` (recommended) or `python3 setup.py install -e .`. Python>3.6 is required.

Data preparation
----------------

Please follow the instruction in `arranger/data/README.md`.

Models
------

- LSTM model
  - `arranger/lstm/train.py`: Train the LSTM model
  - `arranger/lstm/infer.py`: Infer with the LSTM model
- Transformer model
  - `arranger/transformer/train.py`: Train the Transformer model
  - `arranger/transformer/infer.py`: Infer with the Transformer model

Baseline algorithms
-------------------

- Most-common algorithm
  - `arranger/common/learn.py`: Learn the most common label
  - `arranger/common/infer.py`: Infer with the most-common algorithm
- Zone-based algorithm
  - `arranger/zone/learn.py`: Learn the optimal zone setting
  - `arranger/zone/infer.py`: Infer with the zone-based algorithm
- Closest-pitch algorithm
  - `arranger/closest/infer.py`: Infer with the closest-pitch algorithm
- MLP model
  - `arranger/mlp/train.py`: Train the MLP model
  - `arranger/mlp/infer.py`: Infer with the MLP model

Configuration
-------------

In `arranger/config.yaml`, you can configure the MIDI program numbers used for each track in the sample files generated. You can also configure the color of the generated sample piano roll visualization.
