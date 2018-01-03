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
import numpy as np
import click

# internal modules
import clana.utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


@click.command(name='get-cm-simple', help=__doc__)
@click.option('--label_filepath',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
@click.option('--gt_filepath',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
@click.option('--predictions_filepath',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
def main(label_filepath, gt_filepath, predictions_filepath):
    cm = calculate_cm(label_filepath,
                      gt_filepath,
                      predictions_filepath)
    # Write JSON file
    cm_filepath = os.path.abspath('cm.json')
    logging.info("Write results to '{}'.".format(cm_filepath))
    with open(cm_filepath, 'w') as outfile:
        str_ = json.dumps(cm.tolist(), indent=2,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)
    print(cm)


def calculate_cm(label_filepath, gt_filepath, predictions_filepath):
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

    Returns
    -------
    confusion_matrix : numpy array (n x n)
    """
    # Read CSV files
    label_filepath = os.path.abspath(label_filepath)
    labels = clana.utils.load_labels(label_filepath, 0)

    with open(predictions_filepath, 'r') as fp:
        reader = csv.reader(fp, delimiter=';', quotechar='"')
        predictions = [row[0] for row in reader]

    with open(gt_filepath, 'r') as fp:
        reader = csv.reader(fp, delimiter=';', quotechar='"')
        truths = [row[0] for row in reader]

    label2i = {}  # map a label to 0, ..., n
    for i, label in enumerate(labels):
        label2i[label] = i

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

    cm = np.zeros((n, n), dtype=int)

    for truth_label, pred_label in zip(truths, predictions):
        cm[label2i[truth_label]][label2i[pred_label]] += 1

    return cm


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--predictions",
                        dest="predictions_filepath",
                        help="",
                        required=True,
                        metavar="FILE")
    parser.add_argument("-t", "--truth",
                        dest="gt_filepath",
                        help="CSV file with delimiter ;",
                        required=True,
                        metavar="FILE")
    parser.add_argument("--labels",
                        dest="label_filepath",
                        help="CSV file with delimiter ;",
                        required=True,
                        metavar="FILE")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.label_filepath,
                      args.gt_filepath,
                      args.predictions_filepath,
                      )
