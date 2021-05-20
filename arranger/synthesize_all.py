"""Collect audio samples."""
import argparse
import os
import os.path
import random
import subprocess
import tempfile
from pathlib import Path

import joblib
import muspy
import tqdm


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
        "-q", "--quiet", action="store_true", help="reduce output verbosity"
    )
    parser.add_argument(
        "-j", "--n_jobs", type=int, default=1, help="number of workers"
    )
    return parser.parse_args()


def write_audio(filename):
    """Write the audio."""
    # Create a temporary directory
    tmpdir = tempfile.mkdtemp()

    # Load the music
    music = muspy.load(filename)

    # Synthesize the music
    wav_path = os.path.join(tmpdir, "temp.wav")
    try:
        music.write_audio(wav_path)
    except ValueError:
        # NOTE: A workaround for a MIDI output bug in MusPy
        music.key_signatures = []
        music.write_audio(wav_path)

    # Convert it to mp3
    mp3_path = (
        filename.parent.parent / "mp3" / filename.with_suffix(".mp3").name
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            wav_path,
            "-vn",
            "-ar",
            "44100",
            "-ab",
            "192k",
            "-f",
            "mp3",
            mp3_path,
        ],
        check=True,
    )

    # Remove the temporary directory
    os.remove(wav_path)
    os.rmdir(tmpdir)


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set random seed
    random.seed(0)

    for dataset in ("bach", "musicnet", "nes", "lmd"):
        if dataset != "lmd":
            continue
        # Ground truth
        sample_dir = args.input_dir / f"{dataset}/common/default/samples"
        if not args.quiet:
            print(sample_dir)
        (sample_dir / "mp3").mkdir(exist_ok=True)

        # Save audio files
        filenames = list((sample_dir / "json").glob("*_truth.json"))
        if dataset == "lmd":
            filenames.extend((sample_dir / "json").glob("*_truth_drums.json"))

        if args.n_jobs == 1:
            for filename in tqdm.tqdm(filenames, ncols=80):
                write_audio(filename)
        else:
            joblib.Parallel(
                args.n_jobs,
                prefer="threads",
                verbose=0 if args.quiet else 5,
            )(joblib.delayed(write_audio)(filename) for filename in filenames)

        # Baselines
        models = ("common", "zone", "closest", "closest")
        settings = ("default", "permutation", "default", "states")
        for model, setting in zip(models, settings):
            sample_dir = (
                args.input_dir / f"{dataset}/{model}/{setting}/samples"
            )
            if not args.quiet:
                print(sample_dir)
            (sample_dir / "mp3").mkdir(exist_ok=True)

            # Save audio files
            filenames = list((sample_dir / "json").glob("*_pred.json"))
            if dataset == "lmd":
                filenames.extend(
                    (sample_dir / "json").glob("*_pred_drums.json")
                )
            if args.n_jobs == 1:
                for filename in tqdm.tqdm(filenames, ncols=80):
                    write_audio(filename)
            else:
                joblib.Parallel(
                    args.n_jobs,
                    prefer="threads",
                    verbose=0 if args.quiet else 5,
                )(
                    joblib.delayed(write_audio)(filename)
                    for filename in filenames
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
                        args.input_dir / f"{dataset}/{model}/{setting}/samples"
                    )
                    if not args.quiet:
                        print(sample_dir)
                    (sample_dir / "mp3").mkdir(exist_ok=True)

                    # Save audio files
                    filenames = list((sample_dir / "json").glob("*_pred.json"))
                    if dataset == "lmd":
                        filenames.extend(
                            (sample_dir / "json").glob("*_pred_drums.json")
                        )
                    if args.n_jobs == 1:
                        for filename in tqdm.tqdm(filenames, ncols=80):
                            write_audio(filename)
                    else:
                        joblib.Parallel(
                            args.n_jobs,
                            prefer="threads",
                            verbose=0 if args.quiet else 5,
                        )(
                            joblib.delayed(write_audio)(filename)
                            for filename in filenames
                        )


if __name__ == "__main__":
    main()
