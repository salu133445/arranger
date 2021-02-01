"""Compute precision and recall."""
import argparse
import logging
from pathlib import Path

import numpy as np
import sklearn.metrics

from arranger.utils import setup_loggers


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
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set up loggers
    setup_loggers(
        filename=args.input_dir / Path(__file__).with_suffix(".log").name,
        quiet=args.quiet,
    )

    # Load ground truth and predictions
    all_predictions = list(
        np.load(args.input_dir / "predictions.npz").values()
    )
    all_labels = list(np.load(args.input_dir / "labels.npz").values())
    concat_predictions = np.concatenate(all_predictions)
    concat_labels = np.concatenate(all_labels)

    # Compute accuracy
    acc = sklearn.metrics.accuracy_score(concat_labels, concat_predictions)
    logging.info(f"Accuracy : {100*acc:.2f}% ({acc})")

    # Compute balanced accuracy
    balanced_acc = sklearn.metrics.balanced_accuracy_score(
        concat_labels, concat_predictions
    )
    logging.info(
        f"Balanced accuracy : {100*balanced_acc:.2f}% ({balanced_acc})"
    )

    # Compute confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(
        concat_labels, concat_predictions, normalize="all"
    )
    with np.printoptions(precision=4, suppress=True):
        logging.info("Confusion matrix : ")
        logging.info(confusion_matrix)
    np.save(args.input_dir / "confusion_matrix.npy", confusion_matrix)

    # Compute soundness and completeness
    n_labels = np.max(concat_labels) + 1
    n_sound, n_complete = 0, 0
    total_sound, total_complete = 0, 0
    for labels, predictions in zip(all_labels, all_predictions):
        for label in range(n_labels):
            # Soundness
            diff_sound = np.diff(labels[predictions == label])
            n_sound += np.count_nonzero(diff_sound == 0)
            total_sound += len(diff_sound)
            # Completeness
            diff_complete = np.diff(predictions[labels == label])
            n_complete += np.count_nonzero(diff_complete == 0)
            total_complete += len(diff_complete)
    soundness = n_sound / total_sound
    completeness = n_complete / total_complete
    logging.info(f"Soundness : {100*soundness:.2f}% ({soundness})")
    logging.info(f"Completeness : {100*completeness:.2f}% ({completeness})")


if __name__ == "__main__":
    main()
