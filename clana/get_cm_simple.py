#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculate the confusion matrix."""

# core modules
import csv
import json
import logging
import os
import sys

# 3rd party modules
import click
import numpy as np
import sklearn.metrics

# internal modules
import clana.utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


@click.command(name='get-cm-simple', help=__doc__)
@click.option('--labels',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
@click.option('--gt',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
@click.option('--predictions',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
@click.option('--clean',
              default=False,
              is_flag=True,
              help='Remove classes that the classifier doesn\'t know')
def main(labels, gt, predictions, clean):
    """
    Get a simple confunsion matrix.

    Parameters
    ----------
    labels : str
        Path to a CSV file with delimiter ;
    gt : str
        Path to a CSV file with delimiter ;
    predictions : str
        Path to a CSV file with delimiter ;
    clean : bool, optional (default: False)
        Remove classes that the classifier doesn't know
    """
    cm = calculate_cm(labels, gt, predictions, clean=False)
    # Write JSON file
    cm_filepath = os.path.abspath('cm.json')
    logging.info("Write results to '{}'.".format(cm_filepath))
    with open(cm_filepath, 'w') as outfile:
        str_ = json.dumps(cm.tolist(), indent=2,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)
    print(cm)


def calculate_cm(label_filepath,
                 gt_filepath,
                 predictions_filepath,
                 replace_unk_preds=False,
                 clean=False):
    """
    Calculate a confusion matrix.

    Parameters
    ----------
    label_filepath : str
        CSV file with delimter ; and quoting char "
        The first field is an identifier, the second one is the index of the
        ground truth
    gt_filepath : str
        CSV file with delimter ; and quoting char "
        The first field is an identifier, the second one is the index of the
        ground truth
    predictions_filepath : str
        CSV file with delimter ; and quoting char "
        The first field is an identifier, the second one is the index of the
        predicted label
    replace_unk_preds : bool, optional (default: True)
        If a prediction is not in the labels in label_filepath, replace it
        with UNK
    clean : bool, optional (default: False)
        Remove classes that the classifier doesn't know

    Returns
    -------
    confusion_matrix : numpy array (n x n)
    """
    # Read CSV files
    label_filepath = os.path.abspath(label_filepath)
    labels = clana.utils.load_labels(label_filepath, 0)

    with open(gt_filepath, 'r') as fp:
        reader = csv.reader(fp, delimiter=';', quotechar='"')
        truths = [row[0] for row in reader]

    with open(predictions_filepath, 'r') as fp:
        reader = csv.reader(fp, delimiter=';', quotechar='"')
        predictions = [row[0] for row in reader]

    label2i = {}  # map a label to 0, ..., n
    for i, label in enumerate(labels):
        label2i[label] = i

    if clean:
        logging.debug('@' * 120)
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
    assert len(predictions) == len(truths), \
        "len(predictions) = {} != {} = len(truths)".format(len(predictions),
                                                           len(truths))
    for label in truths:
        if label not in label2i:
            logging.error('Could not find label "{}" in file "{}"'
                          .format(label, label_filepath))
            sys.exit(-1)

    n = len(labels)
    for label in predictions:
        if label not in label2i:
            label2i[label] = len(labels)
            n = len(labels) + 1
            logging.error('Could not find label "{}" in file "{}" => '
                          'Add class UNK'
                          .format(label, label_filepath))

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
