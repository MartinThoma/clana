#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculate the confusion matrix."""

# core modules
import csv
import json

# 3rd party modules
import numpy as np


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

    assert len(predictions) == len(truths), \
        "len(predictions) = {} != {} = len(truths)".format(len(predictions),
                                                           len(truths))

    for ident, pred_index in predictions:
        cm[ident2truth_index[ident]][int(pred_index)] += 1

    return cm


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--predictions",
                        dest="cm_dump_filepath",
                        help="CSV file with delimiter ;",
                        required=True,
                        metavar="FILE")
    parser.add_argument("-t", "--truth",
                        dest="gt_filepath",
                        help="CSV file with delimiter ;",
                        required=True,
                        metavar="FILE")
    parser.add_argument("-n",
                        dest="n",
                        required=True,
                        type=int,
                        help="Total number of classes")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cm = calculate_cm(args.cm_dump_filepath, args.gt_filepath, args.n)
    # Write JSON file
    with open('cm.json', 'w') as outfile:
        str_ = json.dumps(cm.tolist(), indent=2,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)
    print(cm)
