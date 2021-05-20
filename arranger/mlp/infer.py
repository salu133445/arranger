"""Infer with the LSTM model."""
import argparse
import logging
import random
from operator import itemgetter
from pathlib import Path

import muspy
import numpy as np
import tensorflow as tf
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
        "-m",
        "--model_filename",
        type=Path,
        help="model filename",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=("bach", "musicnet", "nes", "lmd"),
        help="dataset key",
    )
    # parser.add_argument(
    #     "-sl",
    #     "--seq_len",
    #     type=int,
    #     default=500,
    #     help="sequence length",
    # )
    parser.add_argument(
        "-ml",
        "--max_len",
        type=int,
        default=2000,
        help="maximum sequence length for validation",
    )
    # parser.add_argument(
    #     "-bp",
    #     "--use_beat_postion",
    #     action="store_true",
    #     help="use beat and position rather than time",
    # )
    # parser.add_argument(
    #     "-di",
    #     "--use_duration",
    #     action="store_true",
    #     help="use duration as an input",
    # )
    # parser.add_argument(
    #     "-fi",
    #     "--use_frequency",
    #     action="store_true",
    #     help="use frequency as an input",
    # )
    # parser.add_argument(
    #     "-oh",
    #     "--use_onset_hint",
    #     action="store_true",
    #     help="use onset hint as an input",
    # )
    # parser.add_argument(
    #     "-ph",
    #     "--use_pitch_hint",
    #     action="store_true",
    #     help="use pitch hint as an input",
    # )
    # parser.add_argument(
    #     "-pe",
    #     "--use_pitch_embedding",
    #     action="store_true",
    #     help="use pitch embedding",
    # )
    # parser.add_argument(
    #     "-te",
    #     "--use_time_embedding",
    #     action="store_true",
    #     help="use time embedding",
    # )
    # parser.add_argument(
    #     "-be",
    #     "--use_beat_embedding",
    #     action="store_true",
    #     help="use beat embedding",
    # )
    # parser.add_argument(
    #     "-de",
    #     "--use_duration_embedding",
    #     action="store_true",
    #     help="use duration embedding",
    # )
    # parser.add_argument(
    #     "-mt",
    #     "--max_time",
    #     type=int,
    #     default=4096,
    #     help="maximum time",
    # )
    # parser.add_argument(
    #     "-mb",
    #     "--max_beat",
    #     type=int,
    #     default=4096,
    #     help="maximum number of beats",
    # )
    # parser.add_argument(
    #     "-md",
    #     "--max_duration",
    #     type=int,
    #     default=192,
    #     help="maximum duration",
    # )
    # parser.add_argument(
    #     "-ar",
    #     "--autoregressive",
    #     action="store_true",
    #     help="use autoregressive LSTM",
    # )
    # parser.add_argument(
    #     "-bi",
    #     "--bidirectional",
    #     action="store_true",
    #     help="use bidirectional LSTM",
    # )
    parser.add_argument(
        "-nl",
        "--n_layers",
        type=int,
        default=3,
        help="number of layers",
    )
    parser.add_argument(
        "-nu",
        "--n_units",
        type=int,
        default=128,
        help="number of hidden units per layer",
    )
    parser.add_argument(
        "-or",
        "--oracle",
        action="store_true",
        help="use gold labels for autoregressive model",
    )
    parser.add_argument("-g", "--gpu", type=int, help="GPU device to use")
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


def process(filename, model, save, args):
    """Process a file."""
    # Load the data
    music = muspy.load(filename)

    # Get track names and number of tracks
    names = list(CONFIG[args.dataset]["programs"].keys())
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
    seq_len = min(len(notes), args.max_len)
    notes = np.array(notes[:seq_len])
    labels = np.array(labels[:seq_len])

    inputs = get_arrays(notes, labels, n_tracks, seq_len)

    # Predict the labels
    if args.oracle:
        raw_predictions = model.predict(inputs["feature"])
    else:
        raw_predictions = []

        last_pitches = np.zeros(n_tracks, int)
        last_onsets = np.zeros(n_tracks, int)
        last_offsets = np.zeros(n_tracks, int)
        voice_active = np.zeros(n_tracks, bool)

        # Iterate over time steps
        for i, note in enumerate(notes):

            # Slice the input
            sliced = inputs["feature"][i]

            # pitchProx
            sliced[9 : 9 + n_tracks] = (
                np.abs(last_pitches - note[1]) * voice_active
            ) - 1 * (1 - voice_active)
            # interOnsetProx
            sliced[9 + n_tracks : 9 + 2 * n_tracks] = (
                np.abs(last_onsets - note[0]) * voice_active
            ) - 1 * (1 - voice_active)
            # offsetOnsetProx
            sliced[9 + 2 * n_tracks : 9 + 3 * n_tracks] = (
                np.abs(last_offsets - note[0]) * voice_active
            ) - 1 * (1 - voice_active)
            # voicesOccupied
            sliced[9 + 3 * n_tracks : 9 + 4 * n_tracks] = (
                note[0] < last_offsets
            ) * voice_active - 1 * (1 - voice_active)

            # Predict for a step
            raw_prediction = model.predict(np.expand_dims(sliced, 0))
            raw_predictions.append(raw_prediction)

            # Update previous label
            label = int(raw_prediction[..., 1:].argmax())
            last_pitches[label] = note[1]
            last_onsets[label] = note[0]
            last_offsets[label] = note[0] + note[2]
            voice_active[label] = 1

        raw_predictions = np.concatenate(raw_predictions, 0)
    predictions = np.argmax(raw_predictions[..., 1:], -1).flatten()
    assert len(predictions) == len(labels)

    # Return early if no need to save the sample
    if not save:
        return predictions, labels

    # Shorthands
    sample_dir = args.output_dir / "samples"
    programs = CONFIG[args.dataset]["programs"]
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
    if CONFIG[args.dataset]["has_drums"]:
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
    args.output_dir.mkdir(exist_ok=True)
    if args.oracle:
        args.output_dir = args.output_dir / "oracle"
        args.output_dir.mkdir(exist_ok=True)

    # Make sure sample directories exist
    (args.output_dir / "samples").mkdir(exist_ok=True)
    for subdir in ("json", "mid", "png"):
        (args.output_dir / "samples" / subdir).mkdir(exist_ok=True)

    # Configure TensorFlow
    if args.gpu is not None:
        gpus = tf.config.list_physical_devices("GPU")
        tf.config.set_visible_devices(gpus[args.gpu], "GPU")
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    # Set up loggers
    setup_loggers(
        filename=args.output_dir / Path(__file__).with_suffix(".log").name,
        quiet=args.quiet,
    )
    tf.get_logger().setLevel(logging.INFO)

    # Set random seed
    random.seed(0)

    # Log command-line arguments
    logging.info("Running with command-line arguments :")
    for arg, value in vars(args).items():
        logging.info(f"- {arg} : {value}")

    # === Model ===

    # Build the model
    logging.info("Building model...")

    n_tracks = len(CONFIG[args.dataset]["programs"])
    n_features = 9 + 4 * n_tracks

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(n_features,)))
    for _ in range(args.n_layers):
        model.add(tf.keras.layers.Dense(args.n_units))
    model.add(tf.keras.layers.Dense(n_tracks + 1))

    if not args.quiet:
        model.summary()

    # Load trained weights
    if args.model_filename is None:
        if args.oracle:
            model.load_weights(args.output_dir.parent / "best_model.hdf5")
        else:
            model.load_weights(args.output_dir / "best_model.hdf5")
    else:
        model.load_weights(str(args.model_filename))

    # === Testing ===

    # Load sample filenames
    with open(args.input_dir / "samples.txt") as f:
        sample_filenames = [line.rstrip() for line in f]

    # Collect filenames
    logging.info("Collecting filenames...")
    extension = "json" if args.dataset != "lmd" else "json.gz"
    if args.oracle:
        filenames = list(args.input_dir.glob(f"test/*.{extension}"))
        # is_samples = (
        #     filename.stem in sample_filenames for filename in filenames
        # )
    else:
        # NOTE: As this takes more time, we only use a subset of the test set
        # (specifically, the sample set).
        filenames = [
            filename
            for filename in args.input_dir.glob(f"test/*.{extension}")
            if filename.stem in sample_filenames
        ]
        # is_samples = [True] * len(filenames)
    assert filenames, "No input files found."

    # Iterate over the test data
    logging.info("Start testing...")
    results = [
        process(filename, model, False, args)
        for filename in tqdm.tqdm(filenames, disable=args.quiet, ncols=80)
    ]

    # Compute metrics
    compute_metrics(results, args.output_dir)


if __name__ == "__main__":
    main()
