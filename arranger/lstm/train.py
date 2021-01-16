"""Training script for the LSTM model."""
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from arranger.lstm.model import Arranger
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
        "-a",
        "--augmentation",
        action="store_true",
        help="whether to use data augmentation",
    )
    parser.add_argument(
        "-mb",
        "--max_beat",
        type=int,
        default=2048,
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
        "-pli",
        "--use_previous_label",
        action="store_true",
        help="use previous label as an input",
    )
    parser.add_argument(
        "-ohi",
        "--use_onset_hint",
        action="store_true",
        help="use onset hint as an input",
    )
    parser.add_argument(
        "-phi",
        "--use_pitch_hint",
        action="store_true",
        help="use pitch hint as an input",
    )
    parser.add_argument(
        "-bi",
        "--bidirectional",
        action="store_true",
        help="use bidirectional LSTM",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=16,
        help="batch size for training",
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
        "-e", "--epoch", type=int, default=100, help="maximum number of epochs"
    )
    parser.add_argument("-g", "--gpu", type=int, help="GPU device to use")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def load_npy(filename):
    """Load a NPY file into an array."""
    return np.load(filename).astype(np.float32)


def load_npz(filename):
    """Load a NPZ file into a list of arrays."""
    return [arr.astype(np.float32) for arr in np.load(filename).values()]


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

    # Load training data
    logging.info("Loading training data...")
    train_data = {
        "time": load_npy(args.input_dir / "time_train.npy"),
        "pitch": load_npy(args.input_dir / "pitch_train.npy"),
    }
    if args.use_duration:
        train_data["duration"] = load_npy(
            args.input_dir / "duration_train.npy"
        )
    if args.use_onset_hint:
        train_data["onset_hint"] = load_npy(
            args.input_dir / "onset_hint_train.npy"
        )
    if args.use_pitch_hint:
        train_data["pitch_hint"] = load_npy(
            args.input_dir / "pitch_hint_train.npy"
        )
    train_labels = load_npy(args.input_dir / "label_train.npy")

    def loader(data, labels, training):
        """Data loader."""
        for i in random.sample(range(len(labels)), len(labels)):
            inputs = {"time": data["time"][i], "pitch": data["pitch"][i]}
            if training and args.augmentation:
                inputs["pitch"] = inputs["pitch"] + random.randint(-5, 6)
            if args.use_duration:
                inputs["duration"] = data["duration"][i]
            if args.use_previous_label:
                inputs["previous_label"] = data["previous_label"][i]
            if args.use_onset_hint:
                inputs["onset_hint"] = data["onset_hint"][i]
            if args.use_pitch_hint:
                inputs["pitch_hint"] = data["pitch_hint"][i]
            yield inputs, labels[i]

    output_shapes = ({"time": (None,), "pitch": (None,)}, (None,))
    output_types = ({"time": tf.float32, "pitch": tf.float32}, tf.float32)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: loader(train_data, train_labels, training=True),
        output_shapes=output_shapes,
        output_types=output_types,
    )
    train_dataset = train_dataset.batch(args.batch_size).prefetch(3)

    # Load validation data
    logging.info("Loading validation data...")
    val_data = {
        "time": load_npz(args.input_dir / "time_val.npz"),
        "pitch": load_npz(args.input_dir / "pitch_val.npz"),
    }
    if args.use_duration:
        val_data["duration"] = load_npz(args.input_dir / "duration_val.npz")
    if args.use_onset_hint:
        val_data["onset_hint"] = load_npz(
            args.input_dir / "onset_hint_val.npz"
        )
    if args.use_pitch_hint:
        val_data["pitch_hint"] = load_npz(
            args.input_dir / "pitch_hint_val.npz"
        )
    if args.use_previous_label:
        val_data["previous_label"] = load_npz(
            args.input_dir / "previous_label_val.npz"
        )
    val_labels = load_npz(args.input_dir / "label_val.npz")
    val_dataset = tf.data.Dataset.from_generator(
        lambda: loader(val_data, val_labels, training=False),
        output_shapes=output_shapes,
        output_types=output_types,
    )
    val_dataset = val_dataset.batch(1).prefetch(3)

    # === Model ===

    # Build the model
    logging.info("Building model...")

    # Inputs
    n_tracks = len(CONFIG[args.dataset]["programs"])
    inputs = {
        "time": tf.keras.layers.Input((None,), name="time"),
        "pitch": tf.keras.layers.Input((None,), name="pitch"),
    }
    if args.use_duration:
        inputs["duration"] = tf.keras.layers.Input((None,), name="duration")
    if args.use_onset_hint:
        inputs["onset_hint"] = tf.keras.layers.Input(
            (n_tracks,), name="onset_hint"
        )
    if args.use_pitch_hint:
        inputs["pitch_hint"] = tf.keras.layers.Input(
            (n_tracks,), name="pitch_hint"
        )
    if args.use_previous_label:
        inputs["previous_label"] = tf.keras.layers.Input(
            (None,), name="previous_label"
        )
    arranger = Arranger(
        use_duration=args.use_duration,
        use_frequency=args.use_frequency,
        use_previous_label=args.use_previous_label,
        use_onset_hint=args.use_onset_hint,
        use_pitch_hint=args.use_pitch_hint,
        max_beat=args.max_beat,
        max_duration=args.max_duration,
        n_tracks=n_tracks,
        n_layers=args.n_layers,
        n_units=args.n_units,
        bidirectional=args.bidirectional,
    )
    output, _ = arranger(inputs)
    model = tf.keras.Model(inputs, output)

    # Count variables
    n_trainables = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    n_nontrainables = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    logging.info("Model statistics:")
    logging.info(f"- Total parameters : {n_trainables + n_nontrainables}")
    logging.info(f"- Trainable parameters : {n_trainables}")
    logging.info(f"- Nontrainable parameters : {n_nontrainables}")

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

    # =========================================================================
    # Train the model
    logging.info("Training model...")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(args.output_dir / "best_model.hdf5"),
        save_best_only=True,
        save_weights_only=True,
    )
    csv_logger = tf.keras.callbacks.CSVLogger(
        str(args.output_dir / "training.log")
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)
    model.fit(
        train_dataset,
        batch_size=args.batch_size,
        epochs=args.epoch,
        validation_data=val_dataset,
        validation_batch_size=1,
        callbacks=[model_checkpoint, csv_logger, early_stopping],
        verbose=(1 - args.quiet),
    )


if __name__ == "__main__":
    main()
