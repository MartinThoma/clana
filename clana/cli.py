"""
clana is a toolkit for classifier analysis.

It specifies some file formats and comes with some tools for typical tasks of
classifier analysis.
"""
# Core Library
import logging.config
import os
import random
from typing import Optional

# Third party
import click
import matplotlib

# First party
import clana
import clana.distribution
import clana.get_cm
import clana.get_cm_simple
import clana.visualize_cm

matplotlib.use("Agg")


config = clana.utils.load_cfg(verbose=True)
logging.config.dictConfig(config["LOGGING"])
logging.getLogger("matplotlib").setLevel("WARN")
random.seed(0)


@click.group()
@click.version_option(version=clana.__version__)
def entry_point() -> None:
    """
    Clana is a toolkit for classifier analysis.

    See https://arxiv.org/abs/1707.09725, Chapter 4.
    """


gt_option = click.option(
    "--gt",
    "gt_filepath",
    required=True,
    type=click.Path(exists=True),
    help="CSV file with delimiter ;",
)
predictions_option = click.option(
    "--predictions",
    "predictions_filepath",
    required=True,
    type=click.Path(exists=True),
    help="CSV file with delimiter ;",
)


@entry_point.group()
def get_cm() -> None:
    """Generate a confusion matrix file."""


@get_cm.command(name="simple")
@click.option(
    "--labels",
    "label_filepath",
    required=True,
    type=click.Path(exists=True),
    help="CSV file with delimiter ;",
)
@predictions_option
@gt_option
@click.option(
    "--clean",
    default=False,
    is_flag=True,
    help="Remove classes that the classifier doesn't know",
)
def get_cm_simple(
    label_filepath: str, predictions_filepath: str, gt_filepath: str, clean: bool
) -> None:
    """
    Generate a confusion matrix.

    The input can be a flat list of predictions and a flat list of ground truth
    elements. Each prediction is on its own line. Additional information can be
    after a semicolon.
    """
    clana.get_cm_simple.main(label_filepath, gt_filepath, predictions_filepath, clean)


@get_cm.command(name="standard")
@predictions_option
@gt_option
@click.option("--n", "n", required=True, type=int, help="Number of classes")
def get_cm_standard(predictions_filepath: str, gt_filepath: str, n: int) -> None:
    """
    Generate a confusion matrix from predictions and ground truth.

    The predictions need to be a list of `identifier;prediction` and the
    ground truth needs to be a list of `identifier;truth` of same length.
    """
    clana.get_cm.main(predictions_filepath, gt_filepath, n)


@entry_point.command(name="distribution")
@gt_option
def distribution(gt_filepath: str) -> None:
    """Get the distribution of classes in a dataset."""
    clana.distribution.main(gt_filepath)


@entry_point.command(name="visualize")
@click.option("--cm", "cm_file", type=click.Path(exists=True), required=True)
@click.option(
    "--perm",
    "perm_file",
    help="json file which defines a permutation to start with.",
    type=click.Path(),
    default=None,
)
@click.option(
    "--steps",
    default=1000,
    show_default=True,
    help="Number of steps to find a good permutation.",
)
@click.option("--labels", "labels_file", default="")
@click.option(
    "--zero_diagonal",
    is_flag=True,
    help=(
        "Good classifiers have the highest elements on the diagonal. "
        "This option sets the diagonal to zero so that errors "
        "can be seen more easily."
    ),
)
@click.option(
    "--output",
    "output_image_path",
    type=click.Path(exists=False),
    help="Where to store the image (either .png or .pdf)",
    default=os.path.abspath(config["visualize"]["save_path"]),
    show_default=True,
)
@click.option(
    "--limit_classes", type=int, help="Limit the number of classes in the output"
)
def visualize(
    cm_file: str,
    perm_file: str,
    steps: int,
    labels_file: str,
    zero_diagonal: bool,
    output_image_path: str,
    limit_classes: Optional[int] = None,
) -> None:
    """Optimize and visualize a confusion matrix."""
    print_file_format_issues(cm_file, labels_file, perm_file)
    clana.visualize_cm.main(
        cm_file,
        perm_file,
        steps,
        labels_file,
        zero_diagonal,
        limit_classes,
        output_image_path,
    )


def print_file_format_issues(cm_file: str, labels_file: str, perm_file: str) -> None:
    """
    Get all potential issues of the file formats.

    Parameters
    ----------
    cm_file : str
    labels_file : str
    perm_file : str
    """
    if not (cm_file.lower().endswith("json") or cm_file.lower().endswith("csv")):
        print(f"[WARNING] A json file is expected for the cm_file, but was {cm_file}")
    if not (perm_file is None or perm_file.lower().endswith("json")):
        print(
            f"[WARNING] A json file is expected fo the perm_file, but was {perm_file}"
        )
    cm = clana.io.read_confusion_matrix(cm_file)
    labels = clana.io.read_labels(labels_file, len(cm))
    special_labels = ["UNK"]
    if len(labels) - len(special_labels) < len(cm):
        print(
            "[WARNING] The shape of the confusion matrix is {cm_shape}, but "
            "only {nb_labels} labels were found: {labels}".format(
                cm_shape=cm.shape, nb_labels=len(labels), labels=labels
            )
        )
        print(
            "Please keep in mind that the first row of the labels file is "
            "the header of the CSV (delimiter: ;)"
        )
