# core modules
import unittest

# 3rd party modules
import numpy as np
import numpy.testing

# internal modules
import clana.get_cm_simple


class GetCMTest(unittest.TestCase):

    def test_calculate_cm(self):
        labels = ['en', 'de']
        truths = ['de', 'de', 'en', 'de', 'en']
        predictions = ['de', 'en', 'en', 'de', 'en']
        res = clana.get_cm_simple.calculate_cm(labels, truths, predictions)
        numpy.testing.assert_array_equal(res, np.array([[2, 0], [1, 2]]))
