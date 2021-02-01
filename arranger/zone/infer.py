"""Infer with the zone-based algorithm."""
import argparse
import itertools
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
        "-p", "--permutation", action="store_true", help="consider permutation"
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


def _nonincreasing(seq_len, a, b, seq):
    if seq_len - len(seq) > 1:
        for i in range(a, b):
            yield from _nonincreasing(seq_len, a, i + 1, seq + (i,))
    else:
        for i in range(a, b):
            yield seq + (i,)


def nonincreasing(seq_len, a, b):
    """Yield nonincreasing sequences of a fixed length with values < n."""
    if seq_len > 1:
        for i in range(a, b):
            yield from _nonincreasing(seq_len, a, i + 1, (i,))
    else:
        yield from range(a, b)


def compute_score(counts, boundaries, permutation):
    """Compute the score for the given zone boundaries and permutation."""
    score = 0
    uppers = (128,) + boundaries
    lowers = boundaries + (0,)
    for upper, lower, label in zip(uppers, lowers, permutation):
        score += np.sum(counts[label][lower:upper])
    return score


def find_optimal_zone(counts, permutations, min_pitch=0, max_pitch=127):
    """Find the optimal zone boundaries and permutation."""
    max_score = 0
    optimal_boundaries, optimal_permutation = None, None
    for boundaries in nonincreasing(len(counts) - 1, min_pitch, max_pitch + 1):
        for permutation in permutations:
            score = compute_score(counts, boundaries, permutation)
            if score > max_score:
                max_score = score
                optimal_boundaries = boundaries
                optimal_permutation = permutation
    return optimal_boundaries, optimal_permutation


def predict(notes, boundaries, permutation):
    """Predict the labels for the given zone boundaries and permutation."""
    predictions = np.zeros(len(notes), int)
    uppers = [128] + list(boundaries)
    lowers = list(boundaries) + [0]
    for upper, lower, label in zip(uppers, lowers, permutation):
        in_zone = np.logical_and(notes[:, 1] < upper, notes[:, 1] >= lower)
        predictions[in_zone] = label
    return predictions


def process(filename, dataset, permutation, oracle, output_dir, save):
    """Process a file."""
    # Load the data
    music = muspy.load(filename)

    # Get track names and number of tracks
    names = list(CONFIG[dataset]["programs"].keys())
    n_tracks = len(names)

    # Collect notes, labels and pitch counts
    notes, labels = [], []
    counts = np.zeros((n_tracks, 128), int)
    min_pitch, max_pitch = 0, 127
    for track in music.tracks:
        # Skip drum track or empty track
        if track.is_drum or not track.notes:
            continue
        # Get label
        label = names.index(track.name)
        # Collect pitch counts
        pitches = [note.pitch for note in track]
        counts[label] = np.bincount(pitches, minlength=128)
        # Update minimum and maximum pitches
        min_pitch = min(min_pitch, min(pitches))
        max_pitch = max(max_pitch, max(pitches))
        # Collect notes and labels
        for note in track.notes:
            notes.append((note.time, note.pitch, note.duration, note.velocity))
            labels.append(label)

    # Sort the notes and labels (using notes as keys)
    notes, labels = zip(*sorted(zip(notes, labels), key=itemgetter(0)))

    # Convert lists to arrays for speed reason
    notes = np.array(notes, int)
    labels = np.array(labels, int)

    # Get permutations
    if permutation:
        permutations = tuple(itertools.permutations(range(n_tracks)))
    else:
        permutations = (tuple(range(n_tracks)),)

    if oracle:
        # Compute the scores for different zone settings
        boundaries, permutation = find_optimal_zone(
            counts, permutations, min_pitch, max_pitch
        )
    else:
        # Load the learnt boundaries and permutation
        boundaries = np.loadtxt(output_dir / "optimal_boundaries.txt")
        permutation = np.loadtxt(output_dir / "optimal_permutation.txt")

    # Predict the labels using the optimal zone boundaries and permutation
    predictions = predict(notes, boundaries, permutation)

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
    if args.oracle:
        # NOTE: As computing the oracle takes more time, we only use a subset
        # of the test set (specifically, the sample set).
        filenames = [
            filename
            for filename in args.input_dir.glob(f"test/*.{extension}")
            if filename.stem in sample_filenames
        ]
        is_samples = [True] * len(filenames)
    else:
        filenames = list(args.input_dir.glob(f"test/*.{extension}"))
        is_samples = (
            filename.stem in sample_filenames for filename in filenames
        )
    assert filenames, "No input files found."

    # Iterate over the test data
    logging.info("Start testing...")
    if args.n_jobs == 1:
        results = [
            process(
                filename,
                args.dataset,
                args.permutation,
                args.oracle,
                args.output_dir,
                is_sample,
            )
            for filename, is_sample in zip(
                tqdm.tqdm(filenames, disable=args.quiet, ncols=80), is_samples
            )
        ]
    else:
        results = joblib.Parallel(args.n_jobs, verbose=5)(
            joblib.delayed(process)(
                filename,
                args.dataset,
                args.permutation,
                args.oracle,
                args.output_dir,
                is_sample,
            )
            for filename, is_sample in zip(filenames, is_samples)
        )

    # Compute metrics
    compute_metrics(results, args.output_dir)


if __name__ == "__main__":
    main()
