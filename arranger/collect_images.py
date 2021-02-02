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
        for dataset in ("bach", "musicnet", "nes", "lmd"):
            # Baselines
            models = ("common", "zone", "closest", "closest")
            settings = ("default", "permutation", "default", "states")
            for model, setting in zip(models, settings):
                sample_dir = (
                    args.input_dir / f"{dataset}/{model}/{setting}/samples"
                )
                if not args.quiet:
                    print(sample_dir)
                for filename in (sample_dir / "png").glob("*_comp.png"):
                    f.write(
                        filename,
                        f"{dataset}/{filename.stem}_{model}_{setting}.png",
                    )

            # Models
            for model in ("lstm", "transformer"):
                keys = ["", "embedding", "onsethint", "duration"]
                if model == "lstm":
                    keys1 = ("default", "bidirectional")
                else:
                    keys1 = ("default", "lookahead")
                for key1 in keys1:
                    keys[0] = key1
                    for i in range(len(keys)):
                        setting = "_".join(keys[:i])
                        sample_dir = (
                            args.input_dir
                            / f"{dataset}/{model}/{setting}/samples"
                        )
                        if not args.quiet:
                            print(sample_dir)
                        for filename in (sample_dir / "png").glob(
                            "*_comp.png"
                        ):
                            f.write(
                                filename,
                                f"{dataset}/{filename.stem}_{model}_{setting}.png",
                            )


if __name__ == "__main__":
    main()
