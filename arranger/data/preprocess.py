"""Collect training data."""
import argparse
import logging
import math
import random
from operator import itemgetter
from pathlib import Path

import joblib
import muspy
import numpy as np
import tqdm

from arranger.utils import load_config, setup_loggers

# Load configuration
CONFIG = load_config()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        required=True,
        help="input data directory",
    )
    parser.add_argument(
        "-o", "--output_dir", type=Path, required=True, help="output directory"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=("bach", "musicnet", "nes", "lmd"),
        help="dataset key",
    )
    parser.add_argument(
        "-s", "--seq_len", type=int, help="maximum training sequence length"
    )
    parser.add_argument(
        "-m",
        "--max_samples",
        type=int,
        help="maximum number of samples per song",
    )
    parser.add_argument(
        "-j", "--n_jobs", type=int, default=1, help="number of workers"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def get_arrays(notes, labels, n_tracks, seq_len):
    """Process data and return as a dictionary of arrays."""
    # Create a dictionary of arrays initialized to zeros
    data = {
        "time": np.zeros((seq_len,), int),
        "pitch": np.zeros((seq_len,), int),
        "duration": np.zeros((seq_len,), int),
        "label": np.zeros((seq_len,), int),
        "onset_hint": np.zeros((n_tracks,), int),
        "pitch_hint": np.zeros((n_tracks,), int),
    }

    # Fill in data
    for i, (note, label) in enumerate(zip(notes, labels)):
        data["time"][i] = note[0]
        data["pitch"][i] = note[1] + 1  # 0 is reserved for 'no pitch'
        data["duration"][i] = note[2]
        data["label"][i] = label + 1  # 0 is reserved for 'no label'

    for i in range(n_tracks):
        nonzero = (data["label"] == i).nonzero()[0]
        if nonzero.size:
            data["onset_hint"][i] = nonzero[0]
            data["pitch_hint"][i] = round(np.mean(data["pitch"][nonzero]))

    return data


def process(filename, dataset, max_samples=None, seq_len=None):
    """Process the data and return as a list of dictionary of arrays."""
    # Load the data
    music = muspy.load(filename)

    # Get track names and number of tracks
    names = list(CONFIG[dataset]["programs"].keys())
    n_tracks = len(names)

    # Collect notes and labels
    notes, labels = [], []
    for track in music.tracks:
        # Skip drum track or empty track
        if track.is_drum or not track.notes:
            continue
        # Get label
        label = names.index(track.name)
        # Collect notes and labels
        for note in track.notes:
            notes.append((note.time, note.pitch, note.duration, note.velocity))
            labels.append(label)

    # Sort the notes and labels (using notes as keys)
    notes, labels = zip(*sorted(zip(notes, labels), key=itemgetter(0)))

    # Set sequence length to number of notes by default
    if seq_len is None:
        arrays = get_arrays(notes, labels, n_tracks, len(notes))
        return [{"arrays": arrays, "filename": filename, "start": 0}]

    # Sample segment indices
    n_candidates = math.ceil(len(notes) / seq_len)
    if max_samples is not None and (n_candidates > max_samples):
        indices = random.sample(range(n_candidates), max_samples)
    else:
        indices = range(n_candidates)

    # Collect arrays
    collected = []
    for idx in indices:
        start = idx * seq_len
        arrays = get_arrays(
            notes[start : start + seq_len],
            labels[start : start + seq_len],
            n_tracks,
            seq_len,
        )
        collected.append(
            {"arrays": arrays, "filename": filename, "start": start}
        )

    return collected


def collect_data(
    filenames, dataset, max_samples=None, seq_len=None, n_jobs=1, quiet=False,
):
    """Collect data from a list of files."""
    assert n_jobs >= 1, "`n_jobs` must be a positive interger."

    data = []

    if n_jobs == 1:
        for filename in tqdm.tqdm(filenames, disable=quiet):
            processed = process(filename, dataset, max_samples, seq_len)
            if processed is not None:
                data.extend(processed)
        return data

    results = joblib.Parallel(n_jobs, verbose=0 if quiet else 5)(
        joblib.delayed(process)(filename, dataset, max_samples, seq_len)
        for filename in filenames
    )
    for processed in results:
        if processed is not None:
            data.extend(processed)
    return data


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    args.output_dir.mkdir(exist_ok=True)
    assert args.n_jobs >= 1, "`n_jobs` must be a positive interger."

    # Set up loggers
    setup_loggers(
        filename=args.output_dir / Path(__file__).with_suffix(".log").name,
        quiet=args.quiet,
    )

    # Set random seed
    random.seed(0)

    # Log command-line arguments
    logging.info("Running with command-line arguments :")
    for arg, value in vars(args).items():
        logging.info(f"- {arg} : {value}")

    features = [
        "time",
        "pitch",
        "duration",
        "label",
        "onset_hint",
        "pitch_hint",
    ]

    # === Training data ===
    logging.info("Processing training data...")

    # Collect training data
    train_data = collect_data(
        list(args.input_dir.glob("train/*.json")),
        args.dataset,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        n_jobs=args.n_jobs,
        quiet=args.quiet,
    )

    # Save arrays
    for name in features:
        np.save(
            args.output_dir / f"{name}_train.npy",
            np.stack([sample["arrays"][name] for sample in train_data]),
        )
    logging.info(f"Successfully saved {len(train_data)} training samples.")

    # === Validation data ===
    logging.info("Processing validation data...")

    # Collect validation data
    val_data = collect_data(
        list(args.input_dir.glob("valid/*.json")),
        args.dataset,
        max_samples=args.max_samples,
        n_jobs=args.n_jobs,
        quiet=args.quiet,
    )

    # Sort collected data by array length (to speed up batch inference)
    val_data.sort(key=lambda x: len(x["arrays"]["time"]))

    # Save arrays
    for name in features:
        np.savez(
            args.output_dir / f"{name}_val.npz",
            *[sample["arrays"][name] for sample in val_data],
        )

    # Save filenames
    with open(args.output_dir / "filenames_val.txt", "w") as f:
        for sample in val_data:
            f.write(f"{sample['filename']}\n")

    # Save start times
    np.savetxt(
        args.output_dir / "starts_val.txt",
        np.stack([sample["start"] for sample in val_data]),
    )
    logging.info(f"Successfully saved {len(val_data)} validation samples.")

    # === Test data ===
    logging.info("Processing test data...")
    test_data = collect_data(
        list(args.input_dir.glob("test/*.json")),
        args.dataset,
        max_samples=args.max_samples,
        n_jobs=args.n_jobs,
        quiet=args.quiet,
    )

    # Sort collected data by array length (to speed up batch inference)
    test_data.sort(key=lambda x: len(x["arrays"]["time"]))

    # Save arrays
    for name in features:
        np.savez(
            args.output_dir / f"{name}_test.npz",
            *[sample["arrays"][name] for sample in test_data],
        )

    # Save filenames
    with open(args.output_dir / "indices_test.txt", "w") as f:
        for sample in test_data:
            f.write(f"{sample['filename']} {sample['start']}\n")

    # Save start times
    np.savetxt(
        args.output_dir / "starts_test.txt",
        np.stack([sample["start"] for sample in test_data]),
    )
    logging.info(f"Successfully saved {len(test_data)} test samples.")


if __name__ == "__main__":
    main()
