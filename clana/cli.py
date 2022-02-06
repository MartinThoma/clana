"""
clana is a toolkit for classifier analysis.

It specifies some file formats and comes with some tools for typical tasks of
classifier analysis.
"""
# Core Library
import logging.config
import os
import random
from pathlib import Path
from typing import Optional

# Third party
import matplotlib
import typer

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

entry_point = typer.Typer()

# @typer.version_option(version=clana.__version__)
# def entry_point() -> None:
#     """
#     Clana is a toolkit for classifier analysis.

#     See https://arxiv.org/abs/1707.09725, Chapter 4.
#     """


gt_option = typer.Option(..., "--gt", exists=True, help="CSV file with delimiter ;")
predictions_option = typer.Option(
    ...,
    "--predictions",
    exists=True,
    help="CSV file with delimiter ;",
)

get_cm = typer.Typer(help="Generate a confusion matrix file.")

entry_point.add_typer(get_cm, name="get-cm")

clean_option = typer.Option(
    False,
    "--clean",
    is_flag=True,
    help="Remove classes that the classifier doesn't know",
)
labels_option = typer.Option(
    ...,
    "--labels",
    exists=True,
    help="CSV file with delimiter ;",
)


@get_cm.command(name="simple")
def get_cm_simple(
    label_filepath: Path = labels_option,
    predictions_filepath: Path = predictions_option,
    clean: bool = clean_option,
    gt_filepath: Path = gt_option,
) -> None:
    """
    Generate a confusion matrix.

    The input can be a flat list of predictions and a flat list of ground truth
    elements. Each prediction is on its own line. Additional information can be
    after a semicolon.
    """
    clana.get_cm_simple.main(label_filepath, gt_filepath, predictions_filepath, clean)


classes_option = typer.Option(..., "--n", help="Number of classes")


@get_cm.command(name="standard")
def get_cm_standard(
    predictions_filepath: Path = predictions_option,
    n: int = classes_option,
    gt_filepath: Path = gt_option,
) -> None:
    """
    Generate a confusion matrix from predictions and ground truth.

    The predictions need to be a list of `identifier;prediction` and the
    ground truth needs to be a list of `identifier;truth` of same length.
    """
    clana.get_cm.main(predictions_filepath, gt_filepath, n)


@entry_point.command(name="distribution")
def distribution(gt_filepath: Path = gt_option) -> None:
    """Get the distribution of classes in a dataset."""
    clana.distribution.main(gt_filepath)


cm_file_option = typer.Option(..., "--cm", exists=True)
perm_file_option = typer.Option(
    None,
    "--perm",
    help="json file which defines a permutation to start with.",
)
steps_option = typer.Option(
    1000,
    "--steps",
    show_default=True,
    help="Number of steps to find a good permutation.",
)
labels_option = typer.Option("", "--labels")
zero_diagonal_option = typer.Option(
    "--zero_diagonal",
    is_flag=True,
    help=(
        "Good classifiers have the highest elements on the diagonal. "
        "This option sets the diagonal to zero so that errors "
        "can be seen more easily."
    ),
)
output_image_path_option = typer.Option(
    os.path.abspath(config["visualize"]["save_path"]),
    "--output",
    exists=False,
    help="Where to store the image (either .png or .pdf)",
    show_default=True,
)
limit_classes_option = typer.Option(
    None, "--limit_classes", help="Limit the number of classes in the output"
)


@entry_point.command(name="visualize")
def visualize(
    cm_file: Path = cm_file_option,
    perm_file: Path = perm_file_option,
    steps: int = steps_option,
    labels_file: Path = labels_option,
    zero_diagonal: bool = zero_diagonal_option,
    output_image_path: Path = output_image_path_option,
    limit_classes: Optional[int] = limit_classes_option,
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


def print_file_format_issues(cm_file: Path, labels_file: Path, perm_file: Path) -> None:
    """
    Get all potential issues of the file formats.

    Parameters
    ----------
    cm_file : Path
    labels_file : Path
    perm_file : Path
    """
    if not (
        str(cm_file).lower().endswith("json") or str(cm_file).lower().endswith("csv")
    ):
        print(f"[WARNING] A json file is expected for the cm_file, but was {cm_file}")
    if not (perm_file is None or str(perm_file).lower().endswith("json")):
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
