"""Inference with the most-common-label algorithm."""
import argparse
import logging
from operator import itemgetter
from pathlib import Path

import joblib
import muspy
import numpy as np
import tqdm

from arranger.utils import (
    compute_metrics,
    load_config,
    reconstruct_tracks,
    save_comparison,
    save_sample,
    setup_loggers,
)

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
        "-or", "--oracle", action="store_true", help="use oracle per sample"
    )
    parser.add_argument(
        "-j", "--n_jobs", type=int, default=1, help="number of workers"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def process(filename, dataset, oracle, output_dir, save):
    """Process a file."""
    # Load the data
    music = muspy.load(filename)

    # Get track names and number of tracks
    names = list(CONFIG[dataset]["programs"].keys())
    n_tracks = len(names)

    # Collect notes, labels and note counts
    notes, labels = [], []
    counts = np.zeros(n_tracks, int)
    for track in music.tracks:
        # Skip drum track or empty track
        if track.is_drum or not track.notes:
            continue
        # Get label
        label = names.index(track.name)
        # Collect note counts
        counts[label] = len(track.notes)
        # Collect notes and labels
        for note in track.notes:
            notes.append((note.time, note.pitch, note.duration, note.velocity))
            labels.append(label)

    # Sort the notes and labels (using notes as keys)
    notes, labels = zip(*sorted(zip(notes, labels), key=itemgetter(0)))

    # Convert lists to arrays for speed reason
    notes = np.array(notes, int)
    labels = np.array(labels, int)

    if oracle:
        # Find the most common label
        most_common_label = np.argmax(counts)
    else:
        # Load the learnt most common label
        most_common_label = np.loadtxt(output_dir / "most_common_label.txt")

    # Predict the labels using the optimal zone boundaries and permutation
    predictions = np.full_like(labels, most_common_label)

    # Return early if no need to save the sample
    if not save:
        return predictions, labels

    # Shorthands
    sample_dir = output_dir / "samples"
    programs = CONFIG[dataset]["programs"]
    colors = CONFIG["colors"]

    # Reconstruct and save the music using the predicted labels
    music_pred = music.deepcopy()
    music_pred.tracks = reconstruct_tracks(notes, predictions, programs)
    pianoroll_pred = save_sample(
        music_pred, sample_dir, f"{filename.stem}_pred", colors
    )

    # Reconstruct and save the music using the original labels
    music_truth = music.deepcopy()
    music_truth.tracks = reconstruct_tracks(notes, labels, programs)
    pianoroll_truth = save_sample(
        music_truth, sample_dir, f"{filename.stem}_truth", colors
    )

    # Save comparison
    save_comparison(
        pianoroll_truth, pianoroll_pred, sample_dir, f"{filename.stem}_comp"
    )

    # Save the samples with drums
    if CONFIG[dataset]["has_drums"]:
        music_pred.tracks.append(music.tracks[-1])  # append drum track
        pianoroll_pred = save_sample(
            music_pred, sample_dir, f"{filename.stem}_pred_drums", colors
        )
        music_truth.tracks.append(music.tracks[-1])  # append drum track
        pianoroll_truth = save_sample(
            music_truth, sample_dir, f"{filename.stem}_truth_drums", colors
        )
        save_comparison(
            pianoroll_truth,
            pianoroll_pred,
            sample_dir,
            f"{filename.stem}_comp_drums",
        )

    return predictions, labels


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    assert args.n_jobs >= 1, "`n_jobs` must be a positive integer."
    args.output_dir.mkdir(exist_ok=True)
    if args.oracle:
        args.output_dir = args.output_dir / "oracle"
        args.output_dir.mkdir(exist_ok=True)

    # Make sure sample directories exist
    (args.output_dir / "samples").mkdir(exist_ok=True)
    for subdir in ("json", "mid", "png"):
        (args.output_dir / "samples" / subdir).mkdir(exist_ok=True)

    # Set up loggers
    setup_loggers(
        filename=args.output_dir / Path(__file__).with_suffix(".log").name,
        quiet=args.quiet,
    )

    # Log command-line arguments
    logging.info("Running with command-line arguments :")
    for arg, value in vars(args).items():
        logging.info(f"- {arg} : {value}")

    # Load sample filenames
    with open(args.input_dir / "samples.txt") as f:
        sample_filenames = [line.rstrip() for line in f]

    # Collect filenames
    logging.info("Collecting filenames...")
    extension = "json" if args.dataset != "lmd" else "json.gz"
    filenames = list(args.input_dir.glob(f"test/*.{extension}"))
    assert filenames, "No input files found."
    is_samples = (filename.stem in sample_filenames for filename in filenames)

    # Iterate over the test data
    logging.info("Start testing...")
    if args.n_jobs == 1:
        results = [
            process(
                filename, args.dataset, args.oracle, args.output_dir, is_sample
            )
            for filename, is_sample in zip(
                tqdm.tqdm(filenames, disable=args.quiet, ncols=80), is_samples
            )
        ]
    else:
        results = joblib.Parallel(args.n_jobs, verbose=5)(
            joblib.delayed(process)(
                filename, args.dataset, args.oracle, args.output_dir, is_sample
            )
            for filename, is_sample in zip(filenames, is_samples)
        )

    # Compute metrics
    compute_metrics(results, args.output_dir)


if __name__ == "__main__":
    main()
