#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculate the confusion matrix (CSV inputs)."""

# core modules
import csv
import logging

# 3rd party modules
import numpy as np

# internal modules
import clana.io


def main(cm_dump_filepath, gt_filepath, n):
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
    path = 'cm.json'
    clana.io.write_cm(path, cm)
    logging.info('cm was written to \'{}\''.format(path))


def calculate_cm(cm_dump_filepath, gt_filepath, n):
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
    with open(cm_dump_filepath, 'r') as fp:
        reader = csv.reader(fp, delimiter=';', quotechar='"')
        predictions = [row for row in reader]

    with open(gt_filepath, 'r') as fp:
        reader = csv.reader(fp, delimiter=';', quotechar='"')
        truths = [row for row in reader]

    ident2truth_index = {}
    for identifier, truth_index in truths:
        ident2truth_index[identifier] = int(truth_index)

    if len(predictions) != len(truths):
        msg = ('len(predictions) = {} != {} = len(truths)"'
               .format(len(predictions), len(truths)))
        raise ValueError(msg)

    for ident, pred_index in predictions:
        cm[ident2truth_index[ident]][int(pred_index)] += 1

    return cm
