# Data Collection & Preprocessing

## Data Collection

### Bach Chorales

```python
# Collect Bach chorales from the music21 corpus
import shutil
import music21.corpus

for path in music21.corpus.getComposer("bach"):
    if path.suffix in (".mxl", ".xml"):
        shutil.copyfile(path, "data/bach/raw/" + path.name)
```

### MusicNet

TBA

```sh
# Download the metadata
wget -O data/musicnet https://homes.cs.washington.edu/~thickstn/media/musicnet_metadata.csv
```

### NES Music Database

```sh
# Download the dataset
wget -O data/nes http://deepyeti.ucsd.edu/cdonahue/nesmdb/nesmdb_midi.tar.gz

# Extract the archive
tar zxf data/nes/nesmdb_midi.tar.gz

# Rename the folder for consistency
mv nesmdb_midi/ raw/
```

### Lakh MIDI Dataset (LMD)

```sh
# Download the dataset
wget -O data/lmd http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz

# Extract the archive
tar zxf data/lmd/lmd_matched.tar.gz

# Rename the folder for consistency
mv lmd_matched/ raw/

# Download the filenames
wget -O data/lmd http://hog.ee.columbia.edu/craffel/lmd/md5_to_paths.json
```

## Data Preprocessing

> The following commands assume Bach chorales. You might want to replace the dataset identifier `bach` with identifiers of other datasets (`musicnet` for MusicNet, `nes` for NES Music Database and `lmd` for Lakh MIDI Dataset).

```sh
# Preprocess the data
python3 arranger/data/collect_bach.py -i data/bach/raw/ -o data/bach/json/ -j 1

# Collect training data
python3 arranger/data/collect.py -i data/bach/json/ -o data/bach/s_500_m_10/ -d bach -s 500 -m 10 -j 1
```
