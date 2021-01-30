"""Preprocess string quartets dataset."""
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

SPECIAL_FILES = {
    "1931_dv96_2": ["Violin 1", "Violin 2", "Viola", "Cello"] * 2,
    "1932_dv96_3": ["Violin 1", "Violin 2", "Viola", "Cello", "Cello"],
    "1933_dv96_4": ["Violin 1", "Violin 2"] + ["Viola", "Cello"] * 2,
    "2138_br51n1m2": ["Violin 1", "Violin 2"] * 2 + ["Viola"] + ["Cello"] * 3,
    "2140_br51n1m4": ["Violin 1", "Violin 2", "Viola", "Cello", "Cello"],
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


def get_instrument(name):
    """Return the instrument inferred from the track name."""
    if "viola" in name.lower():
        return "Viola"
    if "cello" in name.lower():
        return "Cello"
    for key in ("1st", "violin 1", "violin1", "violino i"):
        if key in name.lower():
            return "Violin 1"
    for key in ("2nd", "violin 2", "violin2", "violino ii"):
        if key in name.lower():
            return "Violin 2"
    return None


def process(filename):
    """Process a file and return the processed music."""
    # Read the file
    music = muspy.read(filename)

    # Return None if no track is found
    if not music.tracks:
        return None

    # Adjust resolution to 24 time steps per quarter note
    music.adjust_resolution(24)

    # Collect notes
    notes = {"Violin 1": [], "Violin 2": [], "Viola": [], "Cello": []}
    if filename.stem in SPECIAL_FILES:
        instruments = SPECIAL_FILES[str(filename.stem)]
        for i, track in enumerate(music.tracks):
            for note in track.notes:
                if note.duration > 0:
                    notes[instruments[i]].append(note)
    else:
        for track in music.tracks:
            instrument = get_instrument(track.name)
            if instrument is None:
                continue
            for note in track.notes:
                if note.duration > 0:
                    notes[instrument].append(note)

    # Remove duplicate notes
    for name in notes:
        note_dict = {}
        for note in notes[name]:
            note_dict[(note.time, note.pitch, note.duration)] = note
        notes[name] = list(note_dict.values())
        notes[name].sort(key=operator.attrgetter("time", "pitch", "duration"))

    # Skip the file if there are less than two active tracks
    if sum((len(v) > 10 for v in notes.values())) < 2:
        return None

    # Update tracks
    music.tracks = []
    for name in notes:
        track = muspy.Track(
            name=name,
            program=CONFIG["musicnet"]["programs"][name],
            is_drum=False,
            notes=notes[name],
        )
        music.tracks.append(track)

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
    splits = random.choices(
        ("train", "valid", "test"), (8, 1, 1), k=len(filenames)
    )  # Select splits for files randomly using an 8:1:1 train-valid-test ratio
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
