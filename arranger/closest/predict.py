"""Predict with the closest-pitch algorithm."""
import argparse
import itertools
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
        "-d",
        "--dataset",
        required=True,
        choices=("bach", "musicnet", "nes", "lmd"),
        help="dataset key",
    )
    parser.add_argument(
        "-of",
        "--onsets_filename",
        type=Path,
        required=True,
        help="onsets filename",
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


def _predict_without_states(notes, collected, onsets, last_pitches):
    predictions = []
    for note_idx in collected:
        # Find the labels
        if note_idx in onsets:
            label = onsets.index(note_idx)
        else:
            # Set the score to negative absolute pitch difference
            score = -np.square(last_pitches - notes[note_idx, 1])
            # Set a large negative score if the note appears before the
            # track onset
            for j, onset in enumerate(onsets):
                if onset is None or note_idx < onset:
                    score[j] = -99999
            # Find the label with the highest score
            label = np.argmax(score)
        # Set prediction
        predictions.append(label)
    return predictions


def _predict_with_states(
    notes, collected, onsets, last_pitches, track_states, n_tracks
):
    # Compute the scores
    scores = np.zeros((len(collected), n_tracks), int)
    for idx, note_idx in enumerate(collected):
        # Find the label
        if note_idx in onsets:
            # Set a large score for the correct label
            scores[idx, onsets.index(note_idx)] = 99999
        else:
            # Set the score to negative absolute pitch difference
            scores[idx] = -np.square(last_pitches - notes[note_idx, 1])
            # Substract a large number if the track is active
            scores[idx] -= 99999 * (track_states > 0)
            # Set a large negative score if the note appears before the
            # track onset
            for j, onset in enumerate(onsets):
                if onset is None or note_idx < onset:
                    scores[idx, j] = -9999999

    # Find the optimal labels
    max_score = -99999999
    predictions = None
    for permutation in itertools.permutations(range(n_tracks), len(collected)):
        score = np.sum(scores[range(len(collected)), permutation])
        if score >= max_score:
            max_score = score
            predictions = permutation

    return predictions


def _predict(notes, onsets, n_tracks, states):
    """Predict the labels."""
    # Create the prediction array
    predictions = np.zeros(len(notes), int)

    # Initialize states (recording the time until when each track is active)
    track_states = np.zeros(n_tracks, int) if states else None

    # Initialize last pitches (recording the last active pitch for each track)
    last_pitches = np.zeros(n_tracks, int)
    for i, onset in enumerate(onsets):
        if onset is not None:
            last_pitches[i] = notes[onset][1]

    # Iterate over the notes
    collected = []
    time = notes[0, 0]
    for i, note in enumerate(notes):
        if note[0] <= time:
            collected.append(i)
            continue

        if not states or len(collected) > n_tracks:
            # Predict the labels
            labels = _predict_without_states(
                notes, collected, onsets, last_pitches
            )

            # Set prediction and update last pitches
            for note_idx, label in zip(collected, labels):
                predictions[note_idx] = label
                last_pitches[label] = notes[note_idx, 1]

            # Reset the collected notes
            collected = [i]

            # Set current time
            time = note[0]

            continue

        # Update track states
        track_states[track_states <= time] = 0

        # Predict the labels
        labels = _predict_with_states(
            notes, collected, onsets, last_pitches, track_states, n_tracks
        )

        # Set predictions and update track_states
        for note_idx, label in zip(collected, labels):
            # Set prediction
            predictions[note_idx] = label
            # Update last pitches
            last_pitches[label] = notes[note_idx, 1]
            # Update track state to the end time of the note
            track_states[label] = notes[note_idx, 0] + notes[note_idx, 2]

        # Reset the collected notes
        collected = [i]

        # Set current time
        time = note[0]

    if not states or len(collected) > n_tracks:
        # Predict the labels
        labels = _predict_without_states(
            notes, collected, onsets, last_pitches
        )

        # Set prediction and update last pitches
        for note_idx, label in zip(collected, labels):
            predictions[note_idx] = label

    else:
        # Update track states
        track_states[track_states <= time] = 0

        # Predict the labels
        labels = _predict_with_states(
            notes, collected, onsets, last_pitches, track_states, n_tracks
        )

        # Set predictions and update track_states
        for note_idx, label in zip(collected, labels):
            # Set prediction
            predictions[note_idx] = label

    return predictions


def predict(music, dataset, states, onsets):
    """Predict on a music."""
    # Collect notes and labels
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

    # Get number of tracks
    n_tracks = len(CONFIG[dataset]["programs"])

    # Predict the labels
    predictions = _predict(notes, np.asarray(onsets), n_tracks, states)

    return notes, predictions


def process(filename, args):
    """Process a file."""
    # Load the data
    music = muspy.load(filename)

    # Load onsets
    onsets = np.load(args.onsets_filename)

    # Get note and predicted labels
    notes, predictions = predict(music, args.dataset, args.states, onsets)

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
