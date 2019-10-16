# Third party
import numpy as np
import pkg_resources
from click.testing import CliRunner

# First party
import clana.cli


def test_get_cm_problems1():
    cm = np.array([[0, 100], [0, 10]])
    labels = ["0", "1"]
    clana.visualize_cm.get_cm_problems(cm, labels)


def test_get_cm_problems2():
    cm = np.array([[12, 100], [0, 0]])
    labels = ["0", "1"]
    clana.visualize_cm.get_cm_problems(cm, labels)


def test_move_1d():
    perm = np.array([8, 7, 6, 1, 2])
    from_start = 1
    from_end = 2
    insert_pos = 0
    new_perm = clana.visualize_cm.move_1d(perm, from_start, from_end, insert_pos)
    new_perm = new_perm.tolist()
    assert new_perm == [7, 6, 8, 1, 2]


def test_simulated_annealing():
    n = 10
    cm = np.random.randint(low=0, high=100, size=(n, n))
    clana.visualize_cm.simulated_annealing(cm, steps=10)
    clana.visualize_cm.simulated_annealing(cm, steps=10, deterministic=True)


def test_create_html_cm():
    n = 10
    cm = np.random.randint(low=0, high=100, size=(n, n))
    clana.visualize_cm.create_html_cm(cm, zero_diagonal=True)


def test_plot_cm():
    n = 25
    cm = np.random.randint(low=0, high=100, size=(n, n))
    clana.visualize_cm.plot_cm(cm, zero_diagonal=True, labels=None)


def test_plot_cm_big():
    n = 5
    cm = np.random.randint(low=0, high=100, size=(n, n))
    clana.visualize_cm.plot_cm(cm, zero_diagonal=True, labels=None)


def test_main():
    path = "examples/wili-cld2-cm.json"
    cm_path = pkg_resources.resource_filename(__name__, path)

    path = "examples/perm.json"
    perm_path = pkg_resources.resource_filename(__name__, path)

    runner = CliRunner()
    _ = runner.invoke(
        clana.cli.visualize, ["--cm", cm_path, "--steps", 100, "--perm", perm_path]
    )
