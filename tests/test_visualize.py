# Third party
import numpy as np
import pkg_resources
from click.testing import CliRunner

# First party
import clana.cli


def test_get_cm_problems1() -> None:
    cm = np.array([[0, 100], [0, 10]])
    labels = ["0", "1"]
    clana.visualize_cm.get_cm_problems(cm, labels)


def test_get_cm_problems2() -> None:
    cm = np.array([[12, 100], [0, 0]])
    labels = ["0", "1"]
    clana.visualize_cm.get_cm_problems(cm, labels)


def test_simulated_annealing() -> None:
    n = 10
    cm = np.random.randint(low=0, high=100, size=(n, n))
    clana.visualize_cm.simulated_annealing(cm, steps=10)
    clana.visualize_cm.simulated_annealing(cm, steps=10, deterministic=True)


def test_create_html_cm() -> None:
    n = 10
    cm = np.random.randint(low=0, high=100, size=(n, n))
    clana.visualize_cm.create_html_cm(cm, zero_diagonal=True)


def test_plot_cm() -> None:
    n = 25
    cm = np.random.randint(low=0, high=100, size=(n, n))
    clana.visualize_cm.plot_cm(cm, zero_diagonal=True, labels=None)


def test_plot_cm_big() -> None:
    n = 5
    cm = np.random.randint(low=0, high=100, size=(n, n))
    clana.visualize_cm.plot_cm(cm, zero_diagonal=True, labels=None)


def test_main() -> None:
    path = "examples/wili-cld2-cm.json"
    cm_path = pkg_resources.resource_filename(__name__, path)

    path = "examples/perm.json"
    perm_path = pkg_resources.resource_filename(__name__, path)

    runner = CliRunner()
    _ = runner.invoke(
        clana.cli.visualize, ["--cm", cm_path, "--steps", "100", "--perm", perm_path]
    )
