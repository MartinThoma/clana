# core modules
import unittest

# 3rd party modules
import numpy as np

# internal modules
import clana.visualize_cm


class VisualizeTest(unittest.TestCase):

    def test_get_cm_problems1(self):
        cm = np.array([[0, 100], [0, 10]])
        labels = ['0', '1']
        clana.visualize_cm.get_cm_problems(cm, labels)

    def test_get_cm_problems2(self):
        cm = np.array([[12, 100], [0, 0]])
        labels = ['0', '1']
        clana.visualize_cm.get_cm_problems(cm, labels)

    def text_move_1d(self):
        perm = [8, 7, 6, 1, 2]
        from_start = 1
        from_end = 2
        insert_pos = 0
        new_perm = clana.visualize_cm.move_1d(perm,
                                              from_start,
                                              from_end,
                                              insert_pos)
        self.assertEqual(new_perm, [7, 6, 8, 1, 2])

    def test_simulated_annealing(self):
        n = 10
        cm = np.random.randint(low=0, high=100, size=(n, n))
        clana.visualize_cm.simulated_annealing(cm, steps=10)
        clana.visualize_cm.simulated_annealing(cm, steps=10,
                                               deterministic=True)

    def test_create_html_cm(self):
        n = 10
        cm = np.random.randint(low=0, high=100, size=(n, n))
        clana.visualize_cm.create_html_cm(cm, zero_diagonal=True)

    def test_extract_clusters_local(self):
        n = 10
        cm = np.random.randint(low=0, high=100, size=(n, n))
        clana.visualize_cm.extract_clusters(cm,
                                            labels=list(range(n)),
                                            steps=10,
                                            method='local-connectivity')

    def test_extract_clusters_energy(self):
        n = 10
        cm = np.random.randint(low=0, high=100, size=(n, n))
        clana.visualize_cm.extract_clusters(cm,
                                            labels=list(range(n)),
                                            steps=10,
                                            method='energy')

    def test_plot_cm(self):
        n = 25
        cm = np.random.randint(low=0, high=100, size=(n, n))
        clana.visualize_cm.plot_cm(cm, zero_diagonal=True, labels=None)

    def test_plot_cm_big(self):
        n = 5
        cm = np.random.randint(low=0, high=100, size=(n, n))
        clana.visualize_cm.plot_cm(cm, zero_diagonal=True, labels=None)
