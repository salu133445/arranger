"""Preprocess training data."""
import argparse
import logging
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
        "velocity": np.zeros((seq_len,), int),
        "label": np.zeros((seq_len,), int),
        "onset_hint": np.zeros((n_tracks,), int),
        "pitch_hint": np.zeros((n_tracks,), int),
    }

    # Fill in data
    for i, (note, label) in enumerate(zip(notes, labels)):
        data["time"][i] = note[0]
        data["pitch"][i] = note[1] + 1  # 0 is reserved for 'no pitch'
        data["duration"][i] = note[2]
        data["velocity"][i] = note[3]
        data["label"][i] = label + 1  # 0 is reserved for 'no label'

    for i in range(n_tracks):
        nonzero = (data["label"] == i).nonzero()[0]
        if nonzero.size:
            data["onset_hint"][i] = nonzero[0]
            data["pitch_hint"][i] = round(np.mean(data["pitch"][nonzero]))

    return data


def process(filename, dataset):
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
    notes = np.array(notes)
    labels = np.array(labels)

    # Set sequence length to number of notes by default
    arrays = get_arrays(notes, labels, n_tracks, len(notes))

    return arrays


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    args.output_dir.mkdir(exist_ok=True)
    assert args.n_jobs >= 1, "`n_jobs` must be a positive integer."

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
        "velocity",
        "label",
        "onset_hint",
        "pitch_hint",
    ]

    # === Training data ===
    logging.info("Processing training data...")
    ext = "json" if args.dataset != "lmd" else "json.gz"
    for subset in ("train", "valid", "test"):
        filenames = list(args.input_dir.glob(f"{subset}/*.{ext}"))
        if args.n_jobs == 1:
            data = []
            for filename in tqdm.tqdm(filenames, disable=args.quiet, ncols=80):
                processed = process(filename, args.dataset)
                if processed is not None:
                    data.append(
                        {"filename": filename.stem, "arrays": processed}
                    )

        else:
            results = joblib.Parallel(
                args.n_jobs, verbose=0 if args.quiet else 5
            )(
                joblib.delayed(process)(filename, args.dataset)
                for filename in filenames
            )
            data = [
                {"filename": filename.stem, "arrays": result}
                for filename, result in zip(filenames, results)
                if result is not None
            ]

        # Sort collected data by array length (to speed up batch inference)
        if subset in ("valid", "test"):
            data.sort(key=lambda x: len(x["arrays"]["time"]))

        # Save arrays
        for name in features:
            np.savez(
                args.output_dir / f"{name}_{subset}.npz",
                *[sample["arrays"][name] for sample in data],
            )
        logging.info(
            f"Successfully saved {len(data)} samples for subset : {subset}."
        )

        # Save filenames
        with open(args.output_dir / f"filenames_{subset}.txt", "w") as f:
            for sample in data:
                f.write(f"{sample['filename']}\n")


if __name__ == "__main__":
    main()
