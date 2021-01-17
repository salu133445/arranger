"""Utility functions."""
import logging
from pathlib import Path

import imageio
import muspy
import numpy as np
import yaml


def load_config():
    """Load configuration into a dictionary."""
    with open(Path(__file__).parent / "config.yaml") as f:
        return yaml.safe_load(f)


def setup_loggers(filename, quiet):
    """Set up the loggers."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=filename,
        filemode="w",
    )
    if not quiet:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(console)


def reconstruct_tracks(notes, labels, programs):
    """Reconstruct the tracks from data."""
    # Reconstruct the notes
    note_lists = [[] for _ in range(len(programs))]
    for note, label in zip(notes, labels):
        note_lists[int(label)].append(
            muspy.Note(
                time=int(note[0]),
                pitch=int(note[1]),
                duration=int(note[2]),
                velocity=int(note[3]),
            )
        )

    # Reconstruct the tracks
    tracks = []
    for note_list, (name, program) in zip(note_lists, programs.items()):
        tracks.append(
            muspy.Track(
                program=program, is_drum=False, name=name, notes=note_list
            )
        )

    return tracks


def to_pianoroll(music, colors):
    """Convert a music into a piano roll."""
    multitrack = music.to_pypianoroll()
    stacked = multitrack.stack() > 0
    colormatrix = np.array(colors[: len(music)])
    reshaped = stacked.reshape(len(music), -1)
    recolored = 255 - np.matmul((255 - colormatrix.T), reshaped)
    clipped = np.round(np.clip(recolored, 0, 255)).astype(np.uint8)
    reshaped = clipped.reshape((3, stacked.shape[1], stacked.shape[2]))
    transposed = np.moveaxis(reshaped, 0, -1)
    return np.flip(transposed, axis=1).transpose(1, 0, 2)


def save_sample(music, sample_dir, filename, colors):
    """Save a music sample into different formats."""
    music.save(sample_dir / "json" / f"{filename}.json")
    try:
        music.write(sample_dir / "mid" / f"{filename}.mid")
    except ValueError:
        # NOTE: A workaround for a MIDI output bug in MusPy
        music.key_signatures = []
        music.write(sample_dir / "mid" / f"{filename}.mid")
    pianoroll = to_pianoroll(music, colors)
    imageio.imwrite(sample_dir / "png" / f"{filename}.png", pianoroll)
    return pianoroll


def save_comparison(pianoroll, pianoroll_pred, sample_dir, filename):
    """Save comparisons of piano rolls."""
    binarized = np.tile((pianoroll < 250).any(-1, keepdims=True), (1, 1, 3))
    uncolored = (255 * (1 - binarized)).astype(np.uint8)
    pianoroll_comp = np.concatenate((uncolored, pianoroll, pianoroll_pred), 0)
    imageio.imwrite(
        sample_dir / "png" / f"{filename}.png", pianoroll_comp,
    )
    return pianoroll_comp

def load_npy(filename):
    """Load a NPY file into an array."""
    return np.load(filename).astype(np.int32)


def load_npz(filename):
    """Load a NPZ file into a list of arrays."""
    return [arr.astype(np.int32) for arr in np.load(filename).values()]
