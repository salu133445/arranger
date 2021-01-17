"""Training script for the LSTM model."""
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from arranger.lstm.model import Arranger
from arranger.utils import load_config, load_npz, setup_loggers

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
    parser.add_argument(
        "-e", "--epoch", type=int, default=100, help="maximum number of epochs"
    )
    parser.add_argument("-g", "--gpu", type=int, help="GPU device to use")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def loader(data, labels, training, args):
    """Data loader."""
    for i in random.sample(range(len(labels)), len(labels)):
        # Get start time and end time
        if training:
            if len(data["time"][i]) > args.seq_len:
                start = random.randint(0, len(data["time"][i]) - args.seq_len)
            else:
                start = 0
            end = start + args.seq_len
        else:
            start = 0
            end = min(len(data["time"]), args.max_len)
        # Collect data
        inputs = {
            "time": data["time"][i][start:end],
            "pitch": data["pitch"][i][start:end],
        }
        seq_len = len(inputs["time"])
        if training and args.augmentation:
            # Randomly transpose the music by -5~+6 semitones
            inputs["pitch"] = inputs["pitch"] + random.randint(-5, 6)
            # Handle out-of-range pitch
            inputs["pitch"][inputs["pitch"] > 127] -= 12  # an octave lower
            inputs["pitch"][inputs["pitch"] < 0] += 12  # an octave higher
        if args.use_duration:
            inputs["duration"] = data["duration"][i][start:end]
        if args.use_onset_hint:
            n_tracks = len(data["onset_hint"][i])
            inputs["onset_hint"] = np.zeros((seq_len, n_tracks))
            for idx, onset in enumerate(data["onset_hint"][i]):
                inputs["onset_hint"][:onset, idx] = -1
                inputs["onset_hint"][onset + 1 :, idx] = 1
        if args.use_pitch_hint:
            inputs["pitch_hint"] = data["pitch_hint"][i]
        if args.autoregressive:
            inputs["previous_label"] = np.roll(labels[i][start:end], 1, 0)
            inputs["previous_label"][0] = 0
        # Pad arrays with zeros at the end
        if training and seq_len < args.seq_len:
            for key in inputs:
                if key == "onset_hint":
                    inputs["onset_hint"] = np.pad(
                        inputs["onset_hint"],
                        ((0, args.seq_len - seq_len), (0, 0)),
                    )
                elif key != "pitch_hint":
                    inputs[key] = np.pad(
                        inputs[key], (0, args.seq_len - seq_len)
                    )
            label = np.pad(labels[i][start:end], (0, args.seq_len - seq_len))
        else:
            label = labels[i][start:end]
        yield inputs, label


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    args.output_dir.mkdir(exist_ok=True)

    # Configure TensorFlow
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

    # Load test data
    logging.info("Loading test data...")
    test_data = {
        "time": load_npz(args.input_dir / "time_test.npz"),
        "pitch": load_npz(args.input_dir / "pitch_test.npz"),
    }
    if args.use_duration:
        test_data["duration"] = load_npz(args.input_dir / "duration_test.npz")
    if args.use_onset_hint:
        test_data["onset_hint"] = load_npz(
            args.input_dir / "onset_hint_test.npz"
        )
    if args.use_pitch_hint:
        test_data["pitch_hint"] = load_npz(
            args.input_dir / "pitch_hint_test.npz"
        )
    test_labels = load_npz(args.input_dir / "label_test.npz")
    test_dataset = tf.data.Dataset.from_generator(
        lambda: loader(test_data, test_labels, training=False, args=args),
        output_shapes=output_shapes,
        output_types=output_types,
    )
    test_dataset = test_dataset.batch(1).prefetch(3)

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
    model.load_weights(str(args.output_dir / "best_model.hdf5"))

    # Compile the model
    logging.info("Compiling model...")

    def masked_acc(y_true, y_pred):
        accuracies = tf.equal(
            y_true, 1 + tf.cast(tf.argmax(y_pred[..., 1:], axis=2), tf.float32)
        )
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        accuracies = tf.cast(tf.math.logical_and(mask, accuracies), tf.float32)
        mask = tf.cast(mask, tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=[masked_acc],
    )

    # === Training ===

    # Train the model
    logging.info("Testing model...")
    model.evaluate(test_dataset, batch_size=1, verbose=(1 - args.quiet))

    # TODO: Use predict -> compute metrics & save samples


if __name__ == "__main__":
    main()
