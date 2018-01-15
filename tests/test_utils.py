# core modules
import pkg_resources
import unittest

# internal modules
import clana.utils


class UtilsTest(unittest.TestCase):

    def test_load_labels(self):
        clana.utils.load_labels('~/.lidtk/data/labels.csv', 10)
