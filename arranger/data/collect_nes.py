"""Collect game music database."""
import argparse
import logging
import operator
import random
from pathlib import Path

import joblib
import muspy
import tqdm

from arranger.utils import load_config, setup_loggers

# Load configuration
CONFIG = load_config()

NAMES = {
    "p1": "Pulse 1",
    "p2": "Pulse 2",
    "tr": "Triangle",
}


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
        "-s",
        "--samples",
        type=int,
        default=100,
        help="maximum number of samples",
    )
    parser.add_argument(
        "-j", "--n_jobs", type=int, default=1, help="number of workers"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    return parser.parse_args()


def process(filename):
    """Process a file and return the processed music."""
    # Read the file
    music = muspy.read(filename)

    # Return None if no track is found
    if not music.tracks:
        return None

    # Adjust resolution to 100
    # NOTE: MIDI files in NES Music Database do not use metric timing. The
    # temporal resolution is set to 44100Hz and the MIDI resolution is set to
    # 22050. Reducing the MIDI resolution to 100 here is equivalent to
    # downsampling to a temporal resolution of 50 Hz, which is equivalent to a
    # resolution of 30 timesteps per quarter note in a bpm of 100.
    music.adjust_resolution(50)

    # Remove noise track
    # NOTE: Noise tracks in NES Music Database are not standard MIDI tracks.
    # They have pitches from 0 to 15, which corresponds to the some internal
    # parameter that controls the sound.
    music.tracks = [track for track in music.tracks if track.name != "no"]

    # Remove duplicate notes
    for track in music.tracks:
        note_dict = {}
        for note in track.notes:
            note_dict[(note.time, note.pitch, note.duration)] = note
        track.notes = list(note_dict.values())
        track.notes.sort(key=operator.attrgetter("time", "pitch", "duration"))

    # Skip the file if there are less than two active tracks
    if sum((len(track) > 10 for track in music.tracks)) < 2:
        return None

    # Update tracks
    for track in music.tracks:
        # Rename tracks
        track.name = NAMES[track.name]
        # Set programs and drums
        track.program = CONFIG["nes"]["programs"][track.name]
        # Apply a constant velocity to the notes
        # NOTE: Note velocities in NES Music Database are not standard note
        # velocities. For simplicity, we overwrite them with a constant.
        for note in track.notes:
            note.velocity = 64

    return music


def process_and_save(filename, output_dir, split):
    """Process a file and save the processed music."""
    # Process file
    music = process(filename)

    if music is None:
        return

    # Save the processed music
    music.save(output_dir / split / filename.with_suffix(".json").name)

    return filename


def main():
    """Main function."""
    # Parse command-line options
    args = parse_arguments()
    args.output_dir.mkdir(exist_ok=True)
    for subdir in ("train", "valid", "test"):
        (args.output_dir / subdir).mkdir(exist_ok=True)
    assert args.n_jobs >= 1, "`n_jobs` must be a positive integer."

    # Set up loggers
    setup_loggers(
        filename=args.output_dir / Path(__file__).with_suffix(".log").name,
        quiet=args.quiet,
    )

    # Set random seed
    random.seed(0)

    # Collect filenames
    logging.info("Collecting filenames...")
    filenames = list(args.input_dir.rglob("*.mid"))
    splits = (filename.parent.name for filename in filenames)
    assert filenames, "No input files found."

    # Start collecting data
    logging.info("Start collecting data...")
    if args.n_jobs == 1:
        count = 0
        filenames = tqdm.tqdm(filenames, disable=args.quiet, ncols=80)
        for filename, split in zip(filenames, splits):
            if process_and_save(filename, args.output_dir, split):
                count += 1
        logging.info(f"Successfully saved {count} files.")
    else:
        results = joblib.Parallel(args.n_jobs, verbose=0 if args.quiet else 5)(
            joblib.delayed(process_and_save)(filename, args.output_dir, split)
            for filename, split in zip(filenames, splits)
        )
        count = sum((bool(x) for x in results))
        logging.info(f"Successfully saved {count} files.")

    # Sample test files
    sample_filenames = list(args.output_dir.glob("test/*.json"))
    if len(sample_filenames) > args.samples:
        sample_filenames = random.sample(sample_filenames, args.samples)
    with open(args.output_dir / "samples.txt", "w") as f:
        for sample_filename in sample_filenames:
            f.write(f"{sample_filename.stem}\n")
    logging.info(f"Successfully sampled {len(sample_filenames)} test files.")


if __name__ == "__main__":
    main()
