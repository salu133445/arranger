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
        "feature": np.zeros((seq_len, 9 + 4 * n_tracks), float),
        "label": np.zeros((seq_len,), int),
    }

    last_pitches = np.zeros(n_tracks, int)
    last_onsets = np.zeros(n_tracks, int)
    last_offsets = np.zeros(n_tracks, int)
    voice_active = np.zeros(n_tracks, bool)

    # Fill in data
    for i, (note, label) in enumerate(zip(notes, labels)):
        # 0 - pitch
        data["feature"][i, 0] = note[1]
        # 1 - duration
        data["feature"][i, 1] = note[2]
        # 2 - isOrnamentation
        data["feature"][i, 2] = note[2] <= 6
        # 7 - metric position
        data["feature"][i, 7] = note[2] % 24

        # pitchProx
        data["feature"][i, 9 : 9 + n_tracks] = (
            np.abs(last_pitches - note[1]) * voice_active
        ) - 1 * (1 - voice_active)
        # interOnsetProx
        data["feature"][i, 9 + n_tracks : 9 + 2 * n_tracks] = (
            np.abs(last_onsets - note[0]) * voice_active
        ) - 1 * (1 - voice_active)
        # offsetOnsetProx
        data["feature"][i, 9 + 2 * n_tracks : 9 + 3 * n_tracks] = (
            np.abs(last_offsets - note[0]) * voice_active
        ) - 1 * (1 - voice_active)
        # voicesOccupied
        data["feature"][i, 9 + 3 * n_tracks : 9 + 4 * n_tracks] = (
            note[0] < last_offsets
        ) * voice_active - 1 * (1 - voice_active)

        # label
        data["label"][i] = label + 1

        # update
        last_pitches[label] = note[1]
        last_onsets[label] = note[0]
        last_offsets[label] = note[0] + note[2]
        voice_active[label] = 1

    # Iterate over the notes to find chords
    collected = []
    last_collected = []
    time = notes[0, 0]
    for i, note in enumerate(notes):
        if note[0] <= time:
            collected.append(i)
            continue
        for idx, note_idx in enumerate(collected):
            # 3 - indexInChord
            data["feature"][note_idx, 3] = idx
            # 6 - chordSize
            data["feature"][note_idx, 6] = len(collected)
        for note_idx, nextnote_idx in zip(collected[:-1], collected[1:]):
            # 4 - pitchDistBelow
            data["feature"][nextnote_idx, 4] = (
                notes[nextnote_idx, 1] - notes[note_idx, 1]
            )
            # 5 - pitchDistAbove
            data["feature"][note_idx, 5] = (
                notes[nextnote_idx, 1] - notes[note_idx, 1]
            )

        for note_idx in last_collected:
            # 8 - numNotesNext
            data["feature"][note_idx, 8] = len(collected)

        last_collected = collected

    return data


def process(filename, dataset, subset):
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

    if subset == "valid" and dataset == "lmd" and len(arrays["feature"]) > 100:
        indices = random.sample(range(len(arrays["feature"])), 100)
        arrays["feature"] = arrays["feature"][indices]
        arrays["label"] = arrays["label"][indices]

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

    features = ["feature", "label"]

    # === Training data ===
    logging.info("Processing training data...")
    ext = "json" if args.dataset != "lmd" else "json.gz"
    for subset in ("train", "valid", "test"):
        filenames = list(args.input_dir.glob(f"{subset}/*.{ext}"))
        if args.n_jobs == 1:
            data = []
            for filename in tqdm.tqdm(filenames, disable=args.quiet, ncols=80):
                processed = process(filename, args.dataset, subset)
                if processed is not None:
                    data.append(
                        {"filename": filename.stem, "arrays": processed}
                    )

        else:
            results = joblib.Parallel(
                args.n_jobs, verbose=0 if args.quiet else 5
            )(
                joblib.delayed(process)(filename, args.dataset, subset)
                for filename in filenames
            )
            data = [
                {"filename": filename.stem, "arrays": result}
                for filename, result in zip(filenames, results)
                if result is not None
            ]

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
