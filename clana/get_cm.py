#!/usr/bin/env python

"""Calculate the confusion matrix (CSV inputs)."""

# Core Library
import csv
import logging

# Third party
import numpy as np

# First party
import clana.io

logger = logging.getLogger(__name__)


def main(cm_dump_filepath: str, gt_filepath: str, n: int) -> None:
    """
    Calculate a confusion matrix.

    Parameters
    ----------
    cm_dump_filepath : str
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
    cm = calculate_cm(cm_dump_filepath, gt_filepath, n)
    path = "cm.json"
    clana.io.write_cm(path, cm)
    logger.info(f"cm was written to '{path}'")


def calculate_cm(cm_dump_filepath: str, gt_filepath: str, n: int) -> np.ndarray:
    """
    Calculate a confusion matrix.

    Parameters
    ----------
    cm_dump_filepath : str
        CSV file with delimter ; and quoting char "
        The first field is an identifier, the second one is the index of the
        predicted label
    gt_filepath : str
        CSV file with delimter ; and quoting char "
        The first field is an identifier, the second one is the index of the
        ground truth
    n : int
        Number of classes

    Returns
    -------
    confusion_matrix : numpy array (n x n)
    """
    cm = np.zeros((n, n), dtype=int)

    # Read CSV files
    with open(cm_dump_filepath) as fp:
        reader = csv.reader(fp, delimiter=";", quotechar='"')
        predictions = list(reader)

    with open(gt_filepath) as fp:
        reader = csv.reader(fp, delimiter=";", quotechar='"')
        truths = list(reader)

    ident2truth_index = {}
    for identifier, truth_index in truths:
        ident2truth_index[identifier] = int(truth_index)

    if len(predictions) != len(truths):
        msg = 'len(predictions) = {} != {} = len(truths)"'.format(
            len(predictions), len(truths)
        )
        raise ValueError(msg)

    for ident, pred_index in predictions:
        cm[ident2truth_index[ident]][int(pred_index)] += 1

    return cm
