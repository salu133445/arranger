"""Collect audio samples."""
import argparse
import random
import zipfile
from pathlib import Path


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
        "-d",
        "--datasets",
        nargs="*",
        choices=("bach", "musicnet", "nes", "lmd"),
        default=["bach", "musicnet", "nes", "lmd"],
        help="dataset(s)",
    )
    parser.add_argument(
        "-k",
        "--keywords",
        nargs="*",
        help="keyword(s)",
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

    # Create the zip archive
    with zipfile.ZipFile(args.output_filename, "w") as f:
        for dataset in args.datasets:
            # Collect filenames (without suffix and extension)
            names = [
                str(filename.name)[:-10]
                for filename in (
                    args.input_dir / f"{dataset}/common/default/samples/mid"
                ).glob("*_truth.mid")
            ]
            if args.keywords:
                names = [
                    name
                    for name in names
                    if any(k in name for k in args.keywords)
                ]
            elif len(names) > 10:
                names = random.sample(names, 5)

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
                if dataset == "lmd":
                    f.write(
                        args.input_dir
                        / f"{dataset}/common/default/samples/mid/{name}_truth_drums.mid",
                        f"{dataset}_drums/{name}_truth_drums.mid",
                    )
                    f.write(
                        args.input_dir
                        / f"{dataset}/common/default/samples/mp3/{name}_truth_drums.mp3",
                        f"{dataset}_drums/{name}_truth_drums.mp3",
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
                    f.write(
                        sample_dir / "mid" / f"{name}_pred.mid",
                        f"{dataset}/{name}_{model}_{setting}.mid",
                    )
                    f.write(
                        sample_dir / "mp3" / f"{name}_pred.mp3",
                        f"{dataset}/{name}_{model}_{setting}.mp3",
                    )
                    if dataset == "lmd":
                        f.write(
                            sample_dir / "mid" / f"{name}_pred_drums.mid",
                            f"{dataset}_drums/{name}_{model}_{setting}.mid",
                        )
                        f.write(
                            sample_dir / "mp3" / f"{name}_pred_drums.mp3",
                            f"{dataset}_drums/{name}_{model}_{setting}_drums.mp3",
                        )

            # Models
            models = ("lstm", "lstm", "lstm", "lstm")
            settings = (
                "default_embedding",
                "default_embedding_onsethint",
                "bidirectional_embedding",
                "bidirectional_embedding_onsethint_duration",
            )
            for model, setting in zip(models, settings):
                sample_dir = (
                    args.input_dir / f"{dataset}/{model}/{setting}/samples"
                )
                if not args.quiet:
                    print(sample_dir)

                # Save MIDI and MP3 files
                for name in names:
                    f.write(
                        sample_dir / "mid" / f"{name}_pred.mid",
                        f"{dataset}/{name}_{model}_{setting}.mid",
                    )
                    f.write(
                        sample_dir / "mp3" / f"{name}_pred.mp3",
                        f"{dataset}/{name}_{model}_{setting}.mp3",
                    )
                    if dataset == "lmd":
                        f.write(
                            sample_dir / "mid" / f"{name}_pred_drums.mid",
                            f"{dataset}_drums/{name}_{model}_{setting}_drums.mid",
                        )
                        f.write(
                            sample_dir / "mp3" / f"{name}_pred_drums.mp3",
                            f"{dataset}_drums/{name}_{model}_{setting}_drums.mp3",
                        )


if __name__ == "__main__":
    main()
