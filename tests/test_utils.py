# Core Library
from pathlib import Path

# First party
import clana.utils


def test_load_labels() -> None:
    clana.utils.load_labels(Path("~/.clana/data/labels.csv"), 10)
