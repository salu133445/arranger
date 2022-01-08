Arranger
========

Arranger is a project on automatic instrumentation. In a nutshell, we aim to dynamically assign a proper instrument for each note in solo music. Such an automatic instrumentation model could empower a musician to play multiple instruments on a keyboard at the same time. It could also assist a composer in suggesting proper instrumentation for a solo piece.

Our proposed models outperform various baseline models and are able to produce alternative convincing instrumentations for existing arrangements. Check out our [demo](https://salu133445.github.io/arranger/demo)!

Prerequisites
-------------

You can install the dependencies by running `pipenv install` (recommended) or `python3 setup.py install -e .`. Python>3.6 is required.

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

Citing
------

Please cite the following paper if you use the code provided in this repository.

Hao-Wen Dong, Chris Donahue, Taylor Berg-Kirkpatrick and Julian McAuley, "Towards Automatic Instrumentation by Learning to Separate Parts in Symbolic Multitrack Music," _Proceedings of the 22nd International Society for Music Information Retrieval Conference (ISMIR)_, 2021.<br>
[[homepage](https://salu133445.github.io/arranger/)]
[[video](https://youtu.be/-KncOGouAh8)]
[[paper](https://salu133445.github.io/arranger/pdf/arranger-ismir2021-paper.pdf)]
[[slides](https://salu133445.github.io/arranger/pdf/arranger-ismir2021-slides.pdf)]
[[slides (long)](https://salu133445.github.io/arranger/pdf/arranger-research-exam-slides.pdf)]
[[arXiv](https://arxiv.org/abs/2107.05916)]
[[code](https://github.com/salu133445/arranger)]
