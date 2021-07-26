"""Collect Bach chorales dataset."""
import argparse
import logging
import operator
from pathlib import Path

import music21.converter
import muspy
import tqdm

from arranger.utils import load_config, setup_loggers

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
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def process_music(music):
    """Process a music and return the processed music."""
    # Raise ValueError if no track is found
    if not music.tracks:
        raise ValueError("No track is found.")

    # Adjust resolution to 24 time steps per quarter note
    music.adjust_resolution(24)

    # Collect notes
    notes = {
        "Mixture": [],
        "Drums": [],
    }
    for track in music.tracks:
        if track.is_drum:
            for note in track.notes:
                if note.duration > 0:
                    notes["Drums"].append(note)
        else:
            for note in track.notes:
                if note.duration > 0:
                    notes["Mixture"].append(note)

    # Remove duplicate notes
    for name in notes:
        note_dict = {}
        for note in notes[name]:
            note_dict[(note.time, note.pitch, note.duration)] = note
        notes[name] = list(note_dict.values())
        notes[name].sort(key=operator.attrgetter("time", "pitch", "duration"))

    # Update tracks
    music.tracks = [
        muspy.Track(
            program=CONFIG["mixture_program"],
            is_drum=False,
            name="Mixture",
            notes=notes["Mixture"],
        ),
        muspy.Track(
            program=0, is_drum=True, name="Drums", notes=notes["Drums"]
        ),
    ]

    return music


def process(filename):
    """Process a file and return the processed music."""
    try:
        # Read the file with muspy
        music = muspy.read(filename)
    except Exception:
        # Try music21 if failed
        m21 = music21.converter.parse(filename)
        music = muspy.from_music21_score(m21)
    return process_music(music)


def process_and_save(filename, output_dir):
    """Process a file and save the processed music."""
    # Process file
    music = process(filename)

    # Save the processed music
    music.save(output_dir / Path(filename).with_suffix(".json").name)

    return music


def main():
    """Main function."""
    # Parse command-line options
    args = parse_arguments()

    # Check output directory
    if args.output_dir is not None and not args.output_dir.is_dir():
        raise NotADirectoryError("`output_dir` must be an existing directory.")

    # Set up loggers
    setup_loggers(
        filename=args.output_dir / Path(__file__).with_suffix(".log").name,
        quiet=args.quiet,
    )

    # Process the file
    if args.input.is_file():
        process_and_save(args.input, args.output_dir)
        return

    # Collect filenames
    logging.info("Collecting filenames...")
    filenames = []
    for extension in ("mid", "midi", "mxl", "xml", "abc"):
        filenames.extend(args.input.glob(f"*.{extension}"))
    assert filenames, (
        "No supported input files found. Supported extensions are 'mid', "
        "'midi', 'mxl', 'xml' and 'abc'."
    )

    # Process the collected files
    logging.info("Start preparing data...")
    for filename in tqdm.tqdm(filenames, disable=args.quiet, ncols=80):
        process_and_save(filename, args.output_dir)


if __name__ == "__main__":
    main()
