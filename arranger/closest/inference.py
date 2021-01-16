"""Inference script for the closest-pitch algorithm."""
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
        "-s", "--states", action="store_true", help="use state array"
    )
    parser.add_argument(
        "-j", "--n_jobs", type=int, default=1, help="number of workers"
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


def predict(notes, onsets, n_tracks, states):
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


def process(filename, states, dataset, output_dir, save):
    """Process a file."""
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

    # Convert lists to arrays for speed reason
    notes = np.array(notes, int)
    labels = np.array(labels, int)

    # Find the onset time for each track
    onsets = []
    for i in range(n_tracks):
        nonzero = (labels == i).nonzero()[0]
        onsets.append(nonzero[0] if nonzero.size else None)

    # Predict the labels
    predictions = predict(notes, onsets, n_tracks, states)

    # Return early if no need to save the sample
    if not save:
        return np.count_nonzero(predictions == labels), len(labels)

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

    return np.count_nonzero(predictions == labels), len(labels)


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    args.output_dir.mkdir(exist_ok=True)
    assert args.n_jobs >= 1, "`n_jobs` must be a positive interger."

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

    # Iterate over the test data
    logging.info("Start testing...")
    filenames = list(args.input_dir.glob("test/*.json"))
    assert filenames, "No input files found."
    is_samples = (filename.stem in sample_filenames for filename in filenames)

    if args.n_jobs == 1:
        filenames = tqdm.tqdm(filenames, disable=args.quiet)
        results = [
            process(
                filename,
                args.states,
                args.dataset,
                args.output_dir,
                is_sample,
            )
            for filename, is_sample in zip(filenames, is_samples)
        ]
    else:
        results = joblib.Parallel(args.n_jobs, verbose=5)(
            joblib.delayed(process)(
                filename,
                args.states,
                args.dataset,
                args.output_dir,
                is_sample,
            )
            for filename, is_sample in zip(filenames, is_samples)
        )

    # Compute accuracy
    correct, total = 0, 0
    for result in results:
        if result is None:
            continue
        correct += result[0]
        total += result[1]
    accuracy = correct / total
    logging.info(f"Test accuracy : {round(accuracy * 100)}% ({accuracy})")


if __name__ == "__main__":
    main()
