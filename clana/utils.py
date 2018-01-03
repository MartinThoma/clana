#!/usr/bin/env python

"""Utility functions for clana."""

# core modules
import csv
import os


def load_labels(labels_file, n):
    """
    Load labels from a CSV file.

    Parameters
    ----------
    labels_file : str
    n : int

    Returns
    -------
    labels : list
    """
    assert n >= 0
    if os.path.isfile(labels_file):
        # Read CSV file
        with open(labels_file, 'r') as fp:
            reader = csv.reader(fp, delimiter=';', quotechar='"')
            next(reader, None)  # skip the headers
            labels = [row for row in reader]
            labels = [el[0] for el in labels]  # short by default
    else:
        labels = list(range(n))
    return labels
