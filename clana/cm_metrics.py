"""Metrics for confusion matrices."""

# Third party
import numpy as np


def get_accuracy(cm: np.ndarray) -> float:
    """
    Get the accuaracy by the confusion matrix cm.

    Parameters
    ----------
    cm : ndarray

    Returns
    -------
    accuracy : float

    Examples
    --------
    >>> import numpy as np
    >>> cm = np.array([[10, 20], [30, 40]])
    >>> get_accuracy(cm)
    0.5
    >>> cm = np.array([[20, 10], [30, 40]])
    >>> get_accuracy(cm)
    0.6
    """
    return float(sum(cm[i][i] for i in range(len(cm)))) / float(cm.sum())
