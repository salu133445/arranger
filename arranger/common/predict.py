"""Predict with the most-common-label algorithm."""
import argparse
import logging
from pathlib import Path

import muspy
import numpy as np
import tqdm

from arranger.utils import (
    load_config,
    reconstruct_tracks,
    save_sample_flat,
    setup_loggers,
)

# Load configuration
CONFIG = load_config()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="input filename or directory",
    )
    parser.add_argument(
        "-o", "--output_dir", type=Path, required=True, help="output directory"
    )
    parser.add_argument(
        "-m",
        "--model_filename",
        type=Path,
        required=True,
        help="model filename",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=("bach", "musicnet", "nes", "lmd"),
        help="dataset key",
    )
    parser.add_argument(
        "-a",
        "--audio",
        action="store_true",
        help="whether to write audio",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        default="pred",
        help="suffix to the output filename(s)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def predict(music, model_filename):
    """Predict on a music."""
    # Collect notes, labels and note counts
    notes = []
    for track in music.tracks:
        # Skip drum track or empty track
        if track.is_drum or not track.notes:
            continue
        # Collect notes and labels
        for note in track.notes:
            notes.append((note.time, note.pitch, note.duration, note.velocity))

    # Sort the notes
    notes.sort()

    # Convert lists to arrays for speed reason
    notes = np.array(notes, int)

    # Load the learnt most common label
    most_common_label = np.loadtxt(model_filename)

    # Predict the labels using the optimal zone boundaries and permutation
    predictions = np.full(len(notes), most_common_label)

    return notes, predictions


def process(filename, args):
    """Process a file."""
    # Load the data
    music = muspy.load(filename)

    # Get note and predicted labels
    notes, predictions = predict(music, args.model_filename)

    # Shorthands
    programs = CONFIG[args.dataset]["programs"]
    colors = CONFIG["colors"]

    # Reconstruct and save the music using the predicted labels
    music_pred = music.deepcopy()
    music_pred.tracks = reconstruct_tracks(notes, predictions, programs)
    save_sample_flat(
        music_pred, args.output_dir, f"{filename.stem}_{args.suffix}", colors
    )
    if args.audio:
        muspy.write_audio(
            args.output_dir / f"{filename.stem}_{args.suffix}.wav", music_pred
        )

    # Save the samples with drums
    if CONFIG[args.dataset]["has_drums"]:
        music_pred.tracks.append(music.tracks[-1])  # append drum track
        save_sample_flat(
            music_pred,
            args.output_dir,
            f"{filename.stem}_{args.suffix}_drums",
            colors,
        )
        if args.audio:
            muspy.write_audio(
                args.output_dir / f"{filename.stem}_{args.suffix}_drums.wav",
                music_pred,
            )

    return notes, predictions


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()

    # Check output directory
    if args.output_dir is not None and not args.output_dir.is_dir():
        raise NotADirectoryError("`output_dir` must be an existing directory.")

    # Set up loggers
    setup_loggers(
        filename=args.output_dir / Path(__file__).with_suffix(".log").name,
        quiet=args.quiet,
    )

    # Log command-line arguments
    logging.debug("Running with command-line arguments :")
    for arg, value in vars(args).items():
        logging.debug(f"- {arg} : {value}")

    # Process the file
    if args.input.is_file():
        process(args.input, args)
        return

    # Collect filenames
    logging.info("Collecting filenames...")
    filenames = list(args.input.glob("*.json"))
    assert filenames, "No input files found. Only JSON files are supported."

    # Start inference
    logging.info("Start testing...")
    for filename in tqdm.tqdm(filenames, disable=args.quiet, ncols=80):
        process(filename, args)


if __name__ == "__main__":
    main()
