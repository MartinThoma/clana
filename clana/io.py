# core modules
import json
import os

# 3rd party modules
import numpy as np

# internal modules
import clana.utils


def read_confusion_matrix(cm_file, make_max=float('inf')):
    """
    Load confusion matrix.

    Parameters
    ----------
    cm_file : str
        Path to a JSON file which contains a confusion matrix (List[List[int]])
    make_max : float, optional (default: +Infinity)
        Crop values at this value.

    Returns
    -------
    cm : np.array
    """
    with open(cm_file) as f:
        cm = json.load(f)
        cm = np.array(cm)

    # Crop values
    n = len(cm)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cm[i][j] = min(cm[i][j], make_max)

    return cm


def read_permutation(perm_file, n):
    """
    Load permutation.

    Parameters
    ----------
    perm_file : str
        Path to a JSON file which contains a permutation of n numbers.
    n : int
        Length of the confusion matrix

    Returns
    -------
    perm : List[int]
        Permutation of the numbers 0, ..., n-1
    """
    if os.path.isfile(perm_file):
        with open(perm_file) as data_file:
            perm = json.load(data_file)
    else:
        perm = list(range(n))
    return perm


def read_labels(labels_file, n):
    """Load labels."""
    labels = clana.utils.load_labels(labels_file, n)
    labels.append('UNK')
    return labels


def write_labels(labels_file, labels):
    with open(labels_file, 'w') as outfile:
        str_ = json.dumps(labels, indent=2,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)


def write_predictions(identifier2prediction, filepath):
    """
    Create a predictions file.

    Parameters
    ----------
    identifier2prediction : Dict[str, str]
        Map an identifier (as used in write_gt) to a prediction.
        The prediction is a single class, not a distribution.
    filepath : str
        Write to this CSV file.
    """
    with open(filepath, 'w') as f:
        for identifier, prediction in identifier2prediction.items():
            f.write('{};{}\n'.format(identifier, prediction))


def write_gt(identifier2label, filepath):
    """
    Write ground truth to a file.

    Parameters
    ----------
    identifier2label : Dict[str, str]
    filepath : str
        Write to this CSV file.
    """
    with open(filepath, 'w') as f:
        for identifier, label in identifier2label.items():
            f.write('{};{}\n'.format(identifier, label))


def write_cm(path, cm):
    """
    Write confusion matrix to path.

    Parameters
    ----------
    path : str
    cm : ndarray
    """
    with open(path, 'w') as outfile:
        str_ = json.dumps(cm.tolist(),
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)
