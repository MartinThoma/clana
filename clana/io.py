# 3rd party modules
# import numpy as np


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
