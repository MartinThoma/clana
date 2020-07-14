# First party
import clana.utils


def test_load_labels() -> None:
    clana.utils.load_labels("~/.clana/data/labels.csv", 10)
