# core modules
import pkg_resources
import unittest

# 3rd party modules
from click.testing import CliRunner
import numpy as np
import numpy.testing

# internal modules
import clana.cli


class GetCMTest(unittest.TestCase):
    def test_calculate_cm(self):
        labels = ["en", "de"]
        truths = ["de", "de", "en", "de", "en"]
        predictions = ["de", "en", "en", "de", "en"]
        res = clana.get_cm_simple.calculate_cm(labels, truths, predictions)
        numpy.testing.assert_array_equal(res, np.array([[2, 0], [1, 2]]))

    def test_main(self):
        path = "../tests/examples/wili-labels.csv"
        labels_path = pkg_resources.resource_filename("clana", path)
        path = "../tests/examples/wili-y_test.txt"
        gt_filepath = pkg_resources.resource_filename("clana", path)
        path = "../tests/examples/cld2_results.txt"
        predictions_filepath = pkg_resources.resource_filename("clana", path)

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
