"""Collect Bach chorales dataset."""
import argparse
import logging
import operator
import random
from pathlib import Path

import joblib
import music21.converter
import muspy
import tqdm

from arranger.utils import load_config, setup_loggers

# Set random seed
random.seed(0)

# Load configuration
CONFIG = load_config()

NAMES = {
    "soprano": "Soprano",
    "alto": "Alto",
    "tenor": "Tenor",
    "bass": "Bass",
}
SPECIAL_FILES = {
    "bwv171.6": {
        "Soprano\rOboe 1,2\rViolin1": "Soprano",
        "Alto\rViloin 2": "Alto",
        "Tenor\rViola": "Tenor",
        "Bass\rContinuo": "Bass",
    },
    "bwv41.6": {
        "Soprano Oboe 1 Violin1": "Soprano",
        "Alto Oboe 2 Viloin 2": "Alto",
        "Tenor Viola": "Tenor",
        "Bass Continuo": "Bass",
    },
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
    m21 = music21.converter.parse(filename)
    music = muspy.from_music21_score(m21)

    # Return None if no track is found
    if not music.tracks:
        return None

    # Adjust resolution to 24 time steps per quarter note
    music.adjust_resolution(24)

    # Collect notes
    notes = {"Soprano": [], "Alto": [], "Tenor": [], "Bass": []}
    for track in music.tracks:
        if filename.stem in SPECIAL_FILES:
            instrument = SPECIAL_FILES[str(filename.stem)].get(track.name)
        else:
            instrument = NAMES.get(track.name.lower())
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
            program=CONFIG["bach"]["programs"][name],
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
    filenames = list(args.input_dir.rglob("*.mxl"))
    filenames.extend(args.input_dir.rglob("*.xml"))
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

    # Save split results
    for target in ("train", "valid", "test"):
        with open(args.output_dir / target / "filenames.txt", "w") as f:
            for filename, split in zip(filenames, splits):
                if split == target:
                    f.write(f"{filename.stem}\n")


if __name__ == "__main__":
    main()
