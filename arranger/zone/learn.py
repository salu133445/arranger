"""Learn the optimal zone setting."""
import argparse
import itertools
import logging
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
        "-p", "--permutation", action="store_true", help="consider permutation"
    )
    parser.add_argument(
        "-j", "--n_jobs", type=int, default=1, help="number of workers"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def _nonincreasing_sequences(seq_len, n, seq):
    if seq_len - len(seq) > 1:
        for i in range(n):
            yield from _nonincreasing_sequences(seq_len, i + 1, seq + (i,))
    else:
        for i in range(n):
            yield seq + (i,)


def nonincreasing_sequences(seq_len, n):
    """Yield nonincreasing sequences of a fixed length with values < n."""
    if seq_len > 1:
        for i in range(n):
            yield from _nonincreasing_sequences(seq_len, i + 1, (i,))
    else:
        yield from range(n)


def compute_score(counts, boundaries, permutation):
    """Compute the score for the given zone boundaries and permutation."""
    score = 0
    uppers = (128,) + boundaries
    lowers = boundaries + (0,)
    for upper, lower, label in zip(uppers, lowers, permutation):
        score += np.sum(counts[label][lower:upper])
    return score


def find_optimal_zone(counts, permutations):
    """Find the optimal zone boundaries and permutation."""
    max_score = 0
    optimal_boundaries, optimal_permutation = None, None
    for boundaries in nonincreasing_sequences(len(counts) - 1, 128):
        for permutation in permutations:
            score = compute_score(counts, boundaries, permutation)
            if score > max_score:
                max_score = score
                optimal_boundaries = boundaries
                optimal_permutation = permutation
    return optimal_boundaries, optimal_permutation


def process(filename, dataset):
    """Process a file."""
    # Load the data
    music = muspy.load(filename)

    # Get track names
    names = list(CONFIG[dataset]["programs"].keys())

    # Get number of tracks
    n_tracks = len(names)

    # Collect pitch counts
    counts = np.zeros((n_tracks, 128), int)
    for track in music.tracks:
        # Skip drum track or empty track
        if track.is_drum or not track.notes:
            continue
        # Get label
        label = names.index(track.name)
        # Collect pitch counts
        pitches = [note.pitch for note in track]
        counts[label] = np.bincount(pitches, minlength=128)

    return counts


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    assert args.n_jobs >= 1, "`n_jobs` must be a positive integer."
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

    # Iterate over the training data
    logging.info("Start learning...")
    extension = "json" if args.dataset != "lmd" else "json.gz"
    filenames = list(args.input_dir.glob(f"train/*.{extension}"))
    assert filenames, "No input files found."
    if args.n_jobs == 1:
        counts = sum(
            process(filename, args.dataset)
            for filename in tqdm.tqdm(filenames, disable=args.quiet, ncols=80)
        )
    else:
        counts = sum(
            joblib.Parallel(args.n_jobs, verbose=5)(
                joblib.delayed(process)(filename, args.dataset)
                for filename in filenames
            )
        )

    # Get permutations
    n_tracks = len(CONFIG[args.dataset]["programs"])
    if args.permutation:
        permutations = tuple(itertools.permutations(range(n_tracks)))
    else:
        permutations = (tuple(range(n_tracks)),)

    # Find the optimal zone boundaries and permutation
    boundaries, permutation = find_optimal_zone(counts, permutations)
    np.savetxt(args.output_dir / "optimal_boundaries.txt", boundaries)
    logging.info(f"Optimal boundaries are : {boundaries}")
    np.savetxt(args.output_dir / "optimal_permutation.txt", permutation)
    logging.info(f"Optimal permutation is : {permutation}")


if __name__ == "__main__":
    main()
