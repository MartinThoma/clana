#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculate the confusion matrix (one label per line)."""

# core modules
import csv
import json
import logging
import os
import sys

# 3rd party modules
import numpy as np
import sklearn.metrics

# internal modules
import clana.utils


def main(label_filepath, gt_filepath, predictions_filepath, clean):
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
    with open(gt_filepath, 'r') as fp:
        reader = csv.reader(fp, delimiter=';', quotechar='"')
        truths = [row[0] for row in reader]

    with open(predictions_filepath, 'r') as fp:
        reader = csv.reader(fp, delimiter=';', quotechar='"')
        predictions = [row[0] for row in reader]

    cm = calculate_cm(labels, truths, predictions, clean=False)
    # Write JSON file
    cm_filepath = os.path.abspath('cm.json')
    logging.info("Write results to '{}'.".format(cm_filepath))
    with open(cm_filepath, 'w') as outfile:
        str_ = json.dumps(cm.tolist(), indent=2,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)
    print(cm)


def calculate_cm(labels,
                 truths,
                 predictions,
                 replace_unk_preds=False,
                 clean=False):
    """
    Calculate a confusion matrix.

    Parameters
    ----------
    labels : list
    truths : list
    predictions : list
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
        msg = ('len(predictions) = {} != {} = len(truths)"'
               .format(len(predictions), len(truths)))
        raise ValueError(msg)

    label2i = {}  # map a label to 0, ..., n
    for i, label in enumerate(labels):
        label2i[label] = i

    if clean:
        logging.debug('@' * 80)
        preds = []
        truths_tmp = []
        for tru, pred in zip(truths, predictions):
            if tru in predictions:
                truths_tmp.append(tru)
                preds.append(pred)
        predictions = preds
        truths = truths_tmp

    if replace_unk_preds:
        preds = []
        for pred in predictions:
            if label in label2i:
                preds.append(label)
            else:
                preds.append('UNK')
        predictions = preds

    # Sanity check
    for label in truths:
        if label not in label2i:
            logging.error('Could not find label \'{}\''.format(label))
            sys.exit(-1)

    n = len(labels)
    for label in predictions:
        if label not in label2i:
            label2i[label] = len(labels)
            n = len(labels) + 1
            logging.error('Could not find label \'{}\' in labels file => '
                          'Add class UNK'
                          .format(label))

    # TODO: do no always filter
    filter_data_unk = True
    if filter_data_unk:
        truths2, predictions2 = [], []
        for tru, pred in zip(truths, predictions):
            if pred != 'unk':  # TODO: tru != 'UNK'!!!
                truths2.append(tru)
                predictions2.append(pred)
        truths = truths2
        predictions = predictions2

    report = sklearn.metrics.classification_report(truths, predictions,
                                                   labels=labels)
    print(report)
    print("Accuracy: {:.2f}%"
          .format(sklearn.metrics.accuracy_score(truths, predictions) * 100))

    cm = np.zeros((n, n), dtype=int)

    for truth_label, pred_label in zip(truths, predictions):
        cm[label2i[truth_label]][label2i[pred_label]] += 1

    return cm
