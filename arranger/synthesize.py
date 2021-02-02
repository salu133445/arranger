"""Synthesize the music."""
import argparse
import subprocess
from pathlib import Path

import muspy


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_filename",
        type=Path,
        required=True,
        help="input filename",
    )
    parser.add_argument(
        "-o",
        "--output_filename",
        type=Path,
        required=True,
        help="ourput filename",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()

    # Load the music
    music = muspy.load(args.input_filename)

    # Synthesize the music
    music.write_audio(args.output_filename.with_suffix(".wav"))

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            args.output_filename.with_suffix(".wav"),
            "-vn",
            "-ar",
            "44100",
            "-ab",
            "192k",
            "-f",
            "mp3",
            args.output_filename,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
