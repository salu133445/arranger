"""Collect audio samples."""
import argparse
import os
import random
import subprocess
import tempfile
import zipfile
from pathlib import Path

import muspy


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        required=True,
        help="input directory",
    )
    parser.add_argument(
        "-o",
        "--output_filename",
        type=Path,
        required=True,
        help="output filename",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set random seed
    random.seed(0)

    # Create a temporary directory
    tmpdir = tempfile.mkdtemp()

    # Create the zip archive
    with zipfile.ZipFile(args.output_filename, "w") as f:
        for dataset in ("bach", "musicnet", "nes", "lmd"):
            # Collect filenames (without suffix and extension)
            names = [
                str(filename.name)[:-10]
                for filename in (
                    args.input_dir / f"{dataset}/common/default/samples/mid"
                ).glob("*_truth.mid")
            ]
            if len(names) > 10:
                names = random.sample(names, 10)

            # Ground truth
            for name in names:
                f.write(
                    args.input_dir
                    / f"{dataset}/common/default/samples/mid/{name}_truth.mid",
                    f"{dataset}/{name}_truth.mid",
                )
                f.write(
                    args.input_dir
                    / f"{dataset}/common/default/samples/mp3/{name}_truth.mp3",
                    f"{dataset}/{name}_truth.mp3",
                )

            # Baselines
            models = ("common", "zone", "closest", "closest")
            settings = ("default", "permutation", "default", "states")
            for model, setting in zip(models, settings):
                sample_dir = (
                    args.input_dir / f"{dataset}/{model}/{setting}/samples"
                )
                if not args.quiet:
                    print(sample_dir)

                # Save MIDI and MP3 files
                for name in names:
                    if dataset == "lmd":
                        f.write(
                            sample_dir / "mid" / f"{name}.json_pred.mid",
                            f"{dataset}/{name}_{model}_{setting}.mid",
                        )
                        f.write(
                            sample_dir / "mp3" / f"{name}.json_pred.mp3",
                            f"{dataset}/{name}_{model}_{setting}.mp3",
                        )
                        f.write(
                            sample_dir / "mid" / f"{name}.json_pred_drums.mid",
                            f"{dataset}/{name}_{model}_{setting}.mid",
                        )
                        f.write(
                            sample_dir / "mp3" / f"{name}.json_pred_drums.mp3",
                            f"{dataset}/{name}_{model}_{setting}.mp3",
                        )
                    else:
                        f.write(
                            sample_dir / "mid" / f"{name}_pred.mid",
                            f"{dataset}/{name}_{model}_{setting}.mid",
                        )
                        f.write(
                            sample_dir / "mp3" / f"{name}_pred.mp3",
                            f"{dataset}/{name}_{model}_{setting}.mp3",
                        )

            # Models
            for model in ("lstm", "transformer"):
                keys = ["embedding", "onsethint", "duration"]
                if model == "lstm":
                    keys1 = ("default", "bidirectional")
                else:
                    keys1 = ("default", "lookahead")
                for key1 in keys1:
                    for i in range(len(keys) + 1):
                        setting = "_".join([key1] + keys[:i])
                        sample_dir = (
                            args.input_dir
                            / f"{dataset}/{model}/{setting}/samples"
                        )
                        if not args.quiet:
                            print(sample_dir)
                        (sample_dir / "mp3").mkdir(exist_ok=True)

                        # Save MIDI and MP3 files
                        for name in names:
                            if dataset == "lmd":
                                f.write(
                                    sample_dir
                                    / "mid"
                                    / f"{name}.json_pred.mid",
                                    f"{dataset}/{name}_{model}_{setting}.mid",
                                )
                                f.write(
                                    sample_dir
                                    / "mp3"
                                    / f"{name}.json_pred.mp3",
                                    f"{dataset}/{name}_{model}_{setting}.mp3",
                                )
                                f.write(
                                    sample_dir
                                    / "mid"
                                    / f"{name}.json_pred_drums.mid",
                                    f"{dataset}/{name}_{model}_{setting}.mid",
                                )
                                f.write(
                                    sample_dir
                                    / "mp3"
                                    / f"{name}.json_pred_drums.mp3",
                                    f"{dataset}/{name}_{model}_{setting}.mp3",
                                )
                            else:
                                f.write(
                                    sample_dir / "mid" / f"{name}_pred.mid",
                                    f"{dataset}/{name}_{model}_{setting}.mid",
                                )
                                f.write(
                                    sample_dir / "mp3" / f"{name}_pred.mp3",
                                    f"{dataset}/{name}_{model}_{setting}.mp3",
                                )

    # Remove the temporary directory
    os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
