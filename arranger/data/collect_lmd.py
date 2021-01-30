"""Collect pop music dataset."""
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
        "-id",
        "--id_list",
        type=Path,
        help="filename of the cleansed ID list",
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


def get_instrument(program):
    """Return the instrument inferred from the program number."""
    if 0 <= program < 8:
        return "Piano"
    if 24 <= program < 32:
        return "Guitar"
    if 32 <= program < 40:
        return "Bass"
    if 40 <= program < 46 or 48 <= program < 52:
        return "Strings"
    if 56 <= program < 64:
        return "Brass"
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

    # Skip the file if it is longer than 20 minutes
    # (They are probably live-performance MIDI or simply corrupted.)
    if music.get_real_end_time() > 1200:
        return None

    # Collect notes
    notes = {
        "Piano": [],
        "Guitar": [],
        "Bass": [],
        "Strings": [],
        "Brass": [],
        "Drums": [],
    }
    for track in music.tracks:
        if track.is_drum:
            for note in track.notes:
                if note.duration > 0:
                    notes["Drums"].append(note)
        else:
            instrument = get_instrument(track.program)
            if instrument is not None:
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

    # Skip the file if there are less than two active non-drum tracks
    if sum((len(v) > 10 and k != "Drums" for k, v in notes.items())) < 2:
        return None

    # Update tracks
    music.tracks = []
    for name in notes:
        track = muspy.Track(
            name=name,
            program=CONFIG["lmd"]["programs"][name] if name != "Drums" else 0,
            is_drum=(name == "Drums"),
            notes=notes[name],
        )
        track.sort()
        music.tracks.append(track)

    return music


def process_and_save(filename, output_dir, split):
    """Process a file and save the processed music."""
    try:
        # Process file
        music = process(filename)

        # Save the processed music
        if music is None:
            return

        # Save the processed music
        music.save(
            output_dir / split / filename.with_suffix(".json.gz").name,
            compressed=True,
        )

        return filename

    except:  # noqa # pylint: disable=bare-except
        return None


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

    # Load cleansed IDs
    logging.info("Loading IDs...")
    with open(args.id_list) as f:
        file_ids = set(line.split()[0] for line in f)

    # Get filenames
    filenames = [
        filename
        for filename in args.input_dir.rglob("*.mid")
        if filename.stem in file_ids
    ]
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
    sample_filenames = list(args.output_dir.glob("test/*.json.gz"))
    if len(sample_filenames) > args.samples:
        sample_filenames = random.sample(sample_filenames, args.samples)
    with open(args.output_dir / "samples.txt", "w") as f:
        for sample_filename in sample_filenames:
            f.write(f"{sample_filename.stem}\n")
    logging.info(f"Successfully sampled {len(sample_filenames)} test files.")


if __name__ == "__main__":
    main()
