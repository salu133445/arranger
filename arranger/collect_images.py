"""Collect image samples."""
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

    with zipfile.ZipFile(args.output_filename, "w") as f:
        for dataset in args.datasets:
            # Collect filenames (without suffix and extension)
            names = [
                str(filename.name)[:-9]
                for filename in (
                    args.input_dir / f"{dataset}/common/default/samples/png"
                ).glob("*_comp.png")
            ]
            if args.keywords:
                names = [
                    name
                    for name in names
                    if any(k in name for k in args.keywords)
                ]

            # Baselines
            models = ("common", "zone", "closest", "closest")
            settings = ("default", "permutation", "default", "states")
            for model, setting in zip(models, settings):
                sample_dir = (
                    args.input_dir / f"{dataset}/{model}/{setting}/samples"
                )
                if not args.quiet:
                    print(sample_dir)
                for name in names:
                    f.write(
                        sample_dir / "png" / f"{name}_comp.png",
                        f"{dataset}/{name}_{model}_{setting}.png",
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
                for name in names:
                    f.write(
                        sample_dir / "png" / f"{name}_comp.png",
                        f"{dataset}/{name}_{model}_{setting}.png",
                    )


if __name__ == "__main__":
    main()
