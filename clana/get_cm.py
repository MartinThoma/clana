"""Calculate the confusion matrix (CSV inputs)."""

# Core Library
import csv
import logging
from typing import List, Tuple

# Third party
import numpy as np
import numpy.typing as npt

# First party
import clana.io

logger = logging.getLogger(__name__)


def main(predictions_filepath: str, gt_filepath: str, n: int) -> None:
    """
    Calculate a confusion matrix.

    Parameters
    ----------
    predictions_filepath : str
        CSV file with delimter ; and quoting char "
        The first field is an identifier, the second one is the index of the
        predicted label
    gt_filepath : str
        CSV file with delimter ; and quoting char "
        The first field is an identifier, the second one is the index of the
        ground truth
    n : int
        Number of classes
    """
    # Read CSV files
    with open(predictions_filepath) as fp:
        reader = csv.reader(fp, delimiter=";", quotechar='"')
        predictions = [tuple(row) for row in reader]

    with open(gt_filepath) as fp:
        reader = csv.reader(fp, delimiter=";", quotechar='"')
        truths = [tuple(row) for row in reader]

    cm = calculate_cm(predictions, truths, n)
    path = "cm.json"
    clana.io.write_cm(path, cm)
    logger.info(f"cm was written to '{path}'")


def calculate_cm(
    truths: List[Tuple[str, ...]], predictions: List[Tuple[str, ...]], n: int
) -> npt.NDArray:
    """
    Calculate a confusion matrix.

    Parameters
    ----------
    truths : List[Tuple[str, str]]
    predictions : List[Tuple[str, str]]
    n : int
        Number of classes

    Returns
    -------
    confusion_matrix : numpy array (n x n)
    """
    cm = np.zeros((n, n), dtype=int)

    ident2truth_index = {}
    for identifier, truth_index in truths:
        ident2truth_index[identifier] = int(truth_index)

    if len(predictions) != len(truths):
        msg = f'len(predictions) = {len(predictions)} != {len(truths)} = len(truths)"'
        raise ValueError(msg)

    for ident, pred_index in predictions:
        cm[ident2truth_index[ident]][int(pred_index)] += 1

    return cm
