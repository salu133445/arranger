"""Training script for the LSTM model."""
import argparse
import logging
import random
from operator import itemgetter
from pathlib import Path

import muspy
import numpy as np
import sklearn.metrics
import tensorflow as tf
import tqdm

from arranger.lstm.model import Arranger
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
        "-a",
        "--augmentation",
        action="store_true",
        help="whether to use data augmentation",
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
        default=96,
        help="maximum duration",
    )
    parser.add_argument(
        "-au",
        "--autoregressive",
        action="store_true",
        help="use autoregressive LSTM",
    )
    parser.add_argument(
        "-bi",
        "--bidirectional",
        action="store_true",
        help="use bidirectional LSTM",
    )
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
        "-bs",
        "--batch_size",
        type=int,
        default=16,
        help="batch size for training",
    )
    parser.add_argument("-g", "--gpu", type=int, help="GPU device to use")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def get_arrays(notes, labels, n_tracks, args):
    """Process data and return as a dictionary of arrays."""
    # Create a dictionary of arrays initialized to zeros
    seq_len = min(len(notes), args.max_len)
    inputs = {
        "time": np.zeros((seq_len,), int),
        "pitch": np.zeros((seq_len,), int),
    }
    if args.use_duration:
        inputs["duration"] = np.zeros((seq_len,), int)

    # Fill in data
    for i, note in enumerate(notes[:seq_len]):
        inputs["time"][i] = note[0]
        inputs["pitch"][i] = note[1] + 1  # 0 is reserved for 'no pitch'
        if args.use_duration:
            inputs["duration"][i] = note[2]

    if args.use_onset_hint:
        inputs["onset_hint"] = np.zeros((seq_len, n_tracks))
    if args.use_pitch_hint:
        inputs["pitch_hint"] = np.zeros((n_tracks,), int)
    if args.use_onset_hint or args.use_pitch_hint:
        for i in range(n_tracks):
            nonzero = (labels == i).nonzero()[0]
            if nonzero.size:
                if args.use_onset_hint:
                    inputs["onset_hint"][: nonzero[0], i] = -1
                    inputs["onset_hint"][nonzero[0] + 1 :, i] = 1
                if args.use_pitch_hint:
                    inputs["pitch_hint"][i] = round(
                        np.mean(inputs["pitch"][nonzero])
                    )
    if args.autoregressive:
        inputs["previous_label"] = np.roll(labels[:seq_len], 1, 0)
        inputs["previous_label"][0] = 0

    for key in inputs:
        inputs[key] = np.expand_dims(inputs[key], 0)

    return inputs


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
    notes = np.array(notes)
    labels = np.array(labels)

    # Get inputs
    inputs = get_arrays(notes, labels, n_tracks, args)

    # Predict the labels
    raw_predictions = model.predict(inputs)
    predictions = np.argmax(raw_predictions[..., 1:], -1).flatten()

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

    # === Data ===

    # Get output shapes nad types
    n_tracks = len(CONFIG[args.dataset]["programs"])
    output_shapes = ({"time": (None,), "pitch": (None,)}, (None,))
    output_types = ({"time": tf.int32, "pitch": tf.int32}, tf.int32)
    if args.use_duration:
        output_shapes[0]["duration"] = (None,)
        output_types[0]["duration"] = tf.int32
    if args.use_onset_hint:
        output_shapes[0]["onset_hint"] = (None, n_tracks)
        output_types[0]["onset_hint"] = tf.int32
    if args.use_pitch_hint:
        output_shapes[0]["pitch_hint"] = (n_tracks,)
        output_types[0]["pitch_hint"] = tf.int32
    if args.autoregressive:
        output_shapes[0]["previous_label"] = (None,)
        output_types[0]["previous_label"] = tf.int32

    # === Model ===

    # Build the model
    logging.info("Building model...")

    # Inputs
    inputs = {
        "time": tf.keras.layers.Input((None,), dtype=tf.int32, name="time"),
        "pitch": tf.keras.layers.Input((None,), dtype=tf.int32, name="pitch"),
    }
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
            (None,), dtype=tf.int32, name="label"
        )
    arranger = Arranger(
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
        autoregressive=args.autoregressive,
        bidirectional=args.bidirectional,
        n_tracks=n_tracks,
        n_layers=args.n_layers,
        n_units=args.n_units,
    )
    output = arranger(inputs)
    model = tf.keras.Model(inputs, output)

    # Load trained weights
    model.load_weights(str(args.model_filename))

    # === Testing ===

    # Load sample filenames
    with open(args.input_dir / "samples.txt") as f:
        sample_filenames = [line.rstrip() for line in f]

    # Iterate over the test data
    logging.info("Start testing...")
    filenames = list(args.input_dir.glob("test/*.json"))
    assert filenames, "No input files found."
    is_samples = (filename.stem in sample_filenames for filename in filenames)
    results = []
    for filename, is_sample in zip(
        tqdm.tqdm(filenames, disable=args.quiet, ncols=80), is_samples
    ):
        results.append(process(filename, model, is_sample, args))

    # Compute accuracy
    correct, total = 0, 0
    accuracies = []
    all_predictions = []
    all_labels = []
    for result in results:
        if result is None:
            continue
        predictions, labels = result
        all_predictions.append(predictions)
        all_labels.append(labels)
        count_correct = np.count_nonzero(predictions == labels)
        correct += count_correct
        total += len(labels)
        accuracies.append(count_correct / total)
    accuracy = correct / total
    logging.info(f"Test accuracy : {round(accuracy * 100)}% ({accuracy})")

    # Save predictions and labels
    np.savez(args.output_dir / "predictions.npz", *all_predictions)
    np.savez(args.output_dir / "labels.npz", *all_labels)

    # Compute unweighted accuracies
    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    logging.info("Unweighted test accuracy :")
    logging.info(f"- Average : {round(mean_acc* 100)}% ({mean_acc})")
    logging.info(f"- Min : {round(min_acc* 100)}% ({min_acc})")
    logging.info(f"- Max : {round(max_acc* 100)}% ({max_acc})")
    logging.info(f"- Standard deviation : {round(std_acc* 100)}% ({std_acc})")
    np.savetxt(args.output_dir / "accuracies.txt", accuracies)

    # Compute F1 score
    concat_predictions = np.concatenate(all_predictions)
    concat_labels = np.concatenate(all_labels)
    f1_score_micro = sklearn.metrics.f1_score(
        concat_predictions, concat_labels, average="micro"
    )
    logging.info(f"F1 score (micro) : {f1_score_micro}")
    f1_score_macro = sklearn.metrics.f1_score(
        concat_predictions, concat_labels, average="macro"
    )
    logging.info(f"F1 score (macro) : {f1_score_macro}")

    # Compute confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(
        concat_predictions, concat_labels, normalize="all"
    )
    with np.printoptions(precision=4, suppress=True):
        logging.info("Confusion_matrix : ")
        logging.info(confusion_matrix)


if __name__ == "__main__":
    main()
