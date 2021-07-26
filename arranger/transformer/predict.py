"""Predict with the Transformer model."""
import argparse
import logging
from pathlib import Path

import muspy
import numpy as np
import tensorflow as tf
import tqdm

from arranger.transformer.model import TransformerArranger
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
        "-m",
        "--model_filename",
        type=Path,
        required=True,
        help="model filename",
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
        "-of",
        "--onset_hint_filename",
        type=Path,
        help="onset hint filename",
    )
    parser.add_argument(
        "-pf",
        "--pitch_hint_filename",
        type=Path,
        help="pitch hint filename",
    )
    parser.add_argument(
        "-sl",
        "--seq_len",
        type=int,
        default=500,
        help="sequence length",
    )
    parser.add_argument(
        "-ml",
        "--max_len",
        type=int,
        default=2000,
        help="maximum sequence length for validation",
    )
    parser.add_argument(
        "-di",
        "--use_duration",
        action="store_true",
        help="use duration as an input",
    )
    parser.add_argument(
        "-fi",
        "--use_frequency",
        action="store_true",
        help="use frequency as an input",
    )
    parser.add_argument(
        "-oh",
        "--use_onset_hint",
        action="store_true",
        help="use onset hint as an input",
    )
    parser.add_argument(
        "-ph",
        "--use_pitch_hint",
        action="store_true",
        help="use pitch hint as an input",
    )
    parser.add_argument(
        "-pe",
        "--use_pitch_embedding",
        action="store_true",
        help="use pitch embedding",
    )
    parser.add_argument(
        "-te",
        "--use_time_embedding",
        action="store_true",
        help="use time embedding",
    )
    parser.add_argument(
        "-de",
        "--use_duration_embedding",
        action="store_true",
        help="use duration embedding",
    )
    parser.add_argument(
        "-mb",
        "--max_beat",
        type=int,
        default=4096,
        help="maximum number of beats",
    )
    parser.add_argument(
        "-md",
        "--max_duration",
        type=int,
        default=192,
        help="maximum duration",
    )
    parser.add_argument(
        "-lm",
        "--use_lookahead_mask",
        action="store_true",
        help="use lookahead mask",
    )
    parser.add_argument(
        "-ar",
        "--autoregressive",
        action="store_true",
        help="use autoregressive Transformer",
    )
    parser.add_argument(
        "-nl",
        "--n_layers",
        type=int,
        default=3,
        help="number of layers",
    )
    parser.add_argument(
        "-dm",
        "--d_model",
        type=int,
        default=128,
        help="number of hidden units for the attention layer",
    )
    parser.add_argument(
        "-nh",
        "--n_heads",
        type=int,
        default=8,
        help="number of multi-attention heads",
    )
    parser.add_argument(
        "-df",
        "--d_feedforward",
        type=int,
        default=256,
        help="number of hidden units for the feedforward layer",
    )
    parser.add_argument("-g", "--gpu", type=int, help="GPU device to use")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def get_model(args):
    """Return the model."""
    # Create placeholders for the inputs
    inputs = {
        "time": tf.keras.layers.Input((None,), dtype=tf.int32, name="time"),
        "pitch": tf.keras.layers.Input((None,), dtype=tf.int32, name="pitch"),
    }
    n_tracks = len(CONFIG[args.dataset]["programs"])
    if args.use_duration:
        inputs["duration"] = tf.keras.layers.Input(
            (None,), dtype=tf.int32, name="duration"
        )
    if args.use_onset_hint:
        inputs["onset_hint"] = tf.keras.layers.Input(
            (None, n_tracks),
            dtype=tf.int32,
            name="onset_hint",
        )
    if args.use_pitch_hint:
        inputs["pitch_hint"] = tf.keras.layers.Input(
            (n_tracks,), dtype=tf.int32, name="pitch_hint"
        )
    if args.autoregressive:
        inputs["previous_label"] = tf.keras.layers.Input(
            (None, n_tracks), dtype=tf.int32, name="label"
        )

    # Build the model
    arranger = TransformerArranger(
        max_len=args.max_len,
        use_duration=args.use_duration,
        use_frequency=args.use_frequency,
        use_onset_hint=args.use_onset_hint,
        use_pitch_hint=args.use_pitch_hint,
        use_pitch_embedding=args.use_pitch_embedding,
        use_time_embedding=args.use_time_embedding,
        use_duration_embedding=args.use_duration_embedding,
        max_beat=args.max_beat,
        max_duration=args.max_duration,
        use_lookahead_mask=args.use_lookahead_mask,
        autoregressive=args.autoregressive,
        n_tracks=n_tracks,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_feedforward=args.d_feedforward,
    )
    output = arranger(inputs)
    model = tf.keras.Model(inputs, output)

    return model


def get_inputs(notes, use_duration, onset_hint=None, pitch_hint=None):
    """Process data and return as a dictionary of arrays."""
    # Create a dictionary of arrays initialized to zeros
    seq_len = len(notes)
    inputs = {
        "time": np.zeros((seq_len,), int),
        "pitch": np.zeros((seq_len,), int),
    }
    if use_duration:
        inputs["duration"] = np.zeros((seq_len,), int)

    # Fill in data
    for i, note in enumerate(notes):
        inputs["time"][i] = note[0]
        inputs["pitch"][i] = note[1] + 1  # 0 is reserved for 'no pitch'
        if use_duration:
            inputs["duration"][i] = note[2]

    if onset_hint is not None:
        inputs["onset_hint"] = np.asarray(onset_hint)
    if pitch_hint is not None:
        inputs["pitch_hint"] = np.asarray(pitch_hint)

    for key in inputs:
        inputs[key] = np.expand_dims(inputs[key], 0)

    return inputs


def predict(
    music,
    model,
    dataset,
    max_len,
    use_duration,
    use_onset_hint,
    use_pitch_hint,
    autoregressive,
    onset_hint_filename=None,
    pitch_hint_filename=None,
):
    """Predict on a music."""
    # Get track names and number of tracks
    names = list(CONFIG[dataset]["programs"].keys())
    n_tracks = len(names)

    # Collect notes
    notes = []
    for track in music.tracks:
        # Skip drum track or empty track
        if track.is_drum or not track.notes:
            continue
        # Collect notes
        for note in track.notes:
            notes.append((note.time, note.pitch, note.duration, note.velocity))

    # Sort the notes
    notes.sort()
    seq_len = min(len(notes), max_len)
    notes = np.array(notes[:seq_len])

    # Get inputs
    if use_onset_hint and onset_hint_filename is not None:
        onset_hint = np.load(onset_hint_filename)
    else:
        onset_hint = None
    if use_pitch_hint and pitch_hint_filename is not None:
        pitch_hint = np.load(pitch_hint_filename)
    else:
        pitch_hint = None
    inputs = get_inputs(notes, use_duration, onset_hint, pitch_hint)

    # Predict the labels
    if not autoregressive:
        raw_predictions = model.predict(inputs)
    else:
        raw_predictions = []
        previous_label = np.zeros((1, 1, n_tracks))
        # Iterate over time steps
        for i in range(seq_len):
            # Get the sliced input data
            sliced = {"previous_label": previous_label}
            for key, value in inputs.items():
                if key != "pitch_hint":
                    sliced[key] = value[:, i : i + 1]
            if use_pitch_hint:
                sliced["pitch_hint"] = inputs["pitch_hint"]
            # Predict for a step
            raw_prediction = model.predict(sliced)
            raw_predictions.append(raw_prediction)
            # Update previous label
            previous_label.fill(0)
            previous_label[..., int(raw_prediction[..., 1:].argmax())] = 1
        raw_predictions = np.concatenate(raw_predictions, 1)
    predictions = np.argmax(raw_predictions[..., 1:], -1).flatten()

    return notes, predictions


def process(filename, model, args):
    """Process a file."""
    # Load the data
    music = muspy.load(filename)

    # Get note and predicted labels
    notes, predictions = predict(
        music,
        model,
        args.dataset,
        max_len=args.max_len,
        use_duration=args.use_duration,
        use_onset_hint=args.use_onset_hint,
        use_pitch_hint=args.use_pitch_hint,
        autoregressive=args.autoregressive,
        onset_hint_filename=args.onset_hint_filename,
        pitch_hint_filename=args.pitch_hint_filename,
    )

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

    # Log command-line arguments
    logging.debug("Running with command-line arguments :")
    for arg, value in vars(args).items():
        logging.debug(f"- {arg} : {value}")

    # Build the model
    logging.info("Building model...")
    model = get_model(args)

    # Load trained weights
    logging.info("Loading weights...")
    model.load_weights(args.model_filename)

    # Process the file
    if args.input.is_file():
        process(args.input, model, args)
        return

    # Collect filenames
    logging.info("Collecting filenames...")
    filenames = list(args.input.glob("*.json"))
    assert filenames, "No input files found. Only JSON files are supported."

    # Start inference
    logging.info("Start testing...")
    for filename in tqdm.tqdm(filenames, disable=args.quiet, ncols=80):
        process(filename, model, args)


if __name__ == "__main__":
    main()
