"""Utility functions."""
import logging
from pathlib import Path

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
