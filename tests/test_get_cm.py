# Third party
import numpy as np
import numpy.testing
import pkg_resources
from click.testing import CliRunner

# First party
import clana.cli


def test_calculate_cm() -> None:
    labels = ["en", "de"]
    truths = ["de", "de", "en", "de", "en"]
    predictions = ["de", "en", "en", "de", "en"]
    res = clana.get_cm_simple.calculate_cm(labels, truths, predictions)
    numpy.testing.assert_array_equal(res, np.array([[2, 0], [1, 2]]))  # type: ignore[no-untyped-call]


def test_main() -> None:
    path = "examples/wili-labels.csv"
    labels_path = pkg_resources.resource_filename(__name__, path)
    path = "examples/wili-y_test.txt"
    gt_filepath = pkg_resources.resource_filename(__name__, path)
    path = "examples/cld2_results.txt"
    predictions_filepath = pkg_resources.resource_filename(__name__, path)

    runner = CliRunner()
    _ = runner.invoke(
        clana.cli.get_cm_simple,
        [
            "--labels",
            labels_path,
            "--gt",
            gt_filepath,
            "--predictions",
            predictions_filepath,
        ],
    )
