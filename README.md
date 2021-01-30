Arranger
========

Directory Structure
-------------------

```text
├─ analysis         Notebooks for analysis
├─ scripts          Scripts for running experiments
└─ arranger         Main Python module
   ├─ config.yaml   Configuration file
   ├─ data          Code for collecting and processing data
   ├─ common        Most-common algorithm
   ├─ zone          Zone-based algorithm
   ├─ closest       Closest-pitch algorithm
   ├─ lstm          LSTM model
   ├─ transformer   Transformer model
   └─ cnn           CNN model
```

Prerequisites
-------------

You can install the dependencies by running `pipenv install` (recommended) or `python3 setup.py install -e .`. Python>3.6 is required.

Data preparation
----------------

Please follow the instruction in `arranger/data/README.md`.

Models
------

- `arranger/lstm` : LSTM model
  - `train.py`: Train the LSTM model.
  - `infer.py`: Infer with the LSTM model.
- `arranger/transformer` : Transformer model
  - `train.py`: Train the Transformer model.
  - `infer.py`: Infer with the Transformer model.
- `arranger/cnn` : CNN model
  - `train.py`: Train the CNN model.
  - `infer.py`: Infer with the CNN model.

Baseline Algorithms
-------------------

- `arranger/common` : Most-common algorithm
  - `learn.py`: Learn the most common label.
  - `infer.py`: Infer with the most-common algorithm.
- `arranger/zone` : Zone-based algorithm
  - `learn.py`: Learn the optimal zone setting.
  - `infer.py`: Infer with the zone-based algorithm.
- `arranger/closest` : Closest-pitch algorithm
  - `infer.py`: Infer with the closest-pitch algorithm.

Configuration
-------------

In `arranger/config.yaml`, you can configure the MIDI program numbers used for each track in the sample files generated. You can also configure the color of the generated sample piano roll visualization.
