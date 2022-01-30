"""Calculate the confusion matrix (one label per line)."""

# Core Library
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

# Third party
import numpy as np
import numpy.typing as npt
import sklearn.metrics

# First party
import clana.utils

logger = logging.getLogger(__name__)


def main(
    label_filepath: str, gt_filepath: str, predictions_filepath: str, clean: bool
) -> None:
    """
    Get a simple confunsion matrix.

    Parameters
    ----------
    label_filepath : str
        Path to a CSV file with delimiter ;
    gt_filepath : str
        Path to a CSV file with delimiter ;
    predictions : str
        Path to a CSV file with delimiter ;
    clean : bool, optional (default: False)
        Remove classes that the classifier doesn't know
    """
    label_filepath = os.path.abspath(label_filepath)
    labels = clana.utils.load_labels(label_filepath, 0)

    # Read CSV files
    with open(gt_filepath) as fp:
        reader = csv.reader(fp, delimiter=";", quotechar='"')
        truths = [row[0] for row in reader]

    with open(predictions_filepath) as fp:
        reader = csv.reader(fp, delimiter=";", quotechar='"')
        predictions = [row[0] for row in reader]

    cm = calculate_cm(labels, truths, predictions, clean=False)
    # Write JSON file
    cm_filepath = os.path.abspath("cm.json")
    logger.info(f"Write results to '{cm_filepath}'.")
    with open(cm_filepath, "w") as outfile:
        str_ = json.dumps(
            cm.tolist(), indent=2, separators=(",", ": "), ensure_ascii=False
        )
        outfile.write(str_)
    print(cm)


def calculate_cm(
    labels: List[str],
    truths: List[str],
    predictions: List[str],
    replace_unk_preds: bool = False,
    clean: bool = False,
) -> npt.NDArray:
    """
    Calculate a confusion matrix.

    Parameters
    ----------
    labels : List[int]
    truths : List[int]
    predictions : List[int]
    replace_unk_preds : bool, optional (default: True)
        If a prediction is not in the labels in label_filepath, replace it
        with UNK
    clean : bool, optional (default: False)
        Remove classes that the classifier doesn't know

    Returns
    -------
    confusion_matrix : numpy array (n x n)
    """
    # Check data
    if len(predictions) != len(truths):
        msg = f"len(predictions) = {len(predictions)} != {len(truths)} = len(truths)"
        raise ValueError(msg)

    label2i = {}  # map a label to 0, ..., n
    for i, label in enumerate(labels):
        label2i[label] = i

    if clean:
        truths, predictions = clean_truths(truths, predictions)

    if replace_unk_preds:
        predictions = clean_preds(predictions, label2i)

    n = _sanity_check(truths, labels, label2i, predictions)

    # TODO: do no always filter
    filter_data_unk = True
    if filter_data_unk:
        truths2, predictions2 = [], []
        for tru, pred in zip(truths, predictions):
            if pred != "unk":  # TODO: tru != 'UNK'!!!
                truths2.append(tru)
                predictions2.append(pred)
        truths = truths2
        predictions = predictions2

    report = sklearn.metrics.classification_report(truths, predictions, labels=labels)
    print(report)
    print(f"Accuracy: {sklearn.metrics.accuracy_score(truths, predictions) * 100:.2f}%")

    cm = np.zeros((n, n), dtype=int)

    for truth_label, pred_label in zip(truths, predictions):
        cm[label2i[truth_label]][label2i[pred_label]] += 1

    return cm


def clean_truths(
    truths: List[str], predictions: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Remove classes that the classifier doesn't know.

    Parameters
    ----------
    truths : List[int]
    predictions : List[int]

    Returns
    -------
    truths, predictions : List[int], List[int]
    """
    preds = []
    truths_tmp = []
    for tru, pred in zip(truths, predictions):
        if tru in predictions:
            truths_tmp.append(tru)
            preds.append(pred)
    predictions = preds
    truths = truths_tmp
    return truths, predictions


def clean_preds(predictions: List[str], label2i: Dict[str, int]) -> List[str]:
    """
    If a prediction is not in the labels in label_filepath, replace it with UNK.

    Parameters
    ----------
    predictions : List[str]
    label2i : Dict[str, int]
        Maps a label to an index

    Returns
    -------
    predictions : List[str]
    """
    preds = []
    for pred in predictions:
        if pred in label2i:
            preds.append(pred)
        else:
            preds.append("UNK")
    predictions = preds
    return predictions


def _sanity_check(
    truths: List[str],
    labels: List[str],
    label2i: Dict[str, int],
    predictions: List[str],
) -> int:
    for label in truths:
        if label not in label2i:
            logger.error(f"Could not find label '{label}'")
            sys.exit(-1)

    n = len(labels)
    for label in predictions:
        if label not in label2i:
            label2i[label] = len(labels)
            n = len(labels) + 1
            logger.error(
                f"Could not find label '{label}' in labels file => Add class UNK"
            )
    return n
