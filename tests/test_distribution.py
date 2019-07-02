# core modules
import pkg_resources
import unittest

# 3rd party modules
from click.testing import CliRunner

# internal modules
import clana.cli


class DistributionTest(unittest.TestCase):

    def test_cli(self):
        runner = CliRunner()

        path = '../tests/examples/wili-y_train.txt'
        y_train_path = pkg_resources.resource_filename('clana', path)
        result = runner.invoke(clana.cli.distribution,
                               ['--gt', y_train_path])
