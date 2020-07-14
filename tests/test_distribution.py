# Third party
import pkg_resources
from click.testing import CliRunner

# First party
import clana.cli


def test_cli() -> None:
    runner = CliRunner()

    path = "examples/wili-y_train.txt"
    y_train_path = pkg_resources.resource_filename(__name__, path)
    _ = runner.invoke(clana.cli.distribution, ["--gt", y_train_path])
