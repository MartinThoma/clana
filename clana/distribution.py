"""Get the distribution of classes in a dataset."""

# Core Library
from typing import Dict, List


def main(gt_filepath: str) -> None:
    """
    Get the distribution of classes in a file.

    Parameters
    ----------
    gt_filepath : str
        List of ground truth; one label per line
    """
    # Read text file
    with open(gt_filepath) as fp:
        read_lines = fp.readlines()
        labels_str = [line.rstrip("\n") for line in read_lines]

    distribution = get_distribution(labels_str)
    labels = sorted(distribution.items(), key=lambda n: (-n[1], n[0]))
    label_len = max(len(label[0]) for label in labels)
    count_len = max(len(str(label[1])) for label in labels)
    total_count = sum(label[1] for label in labels)
    for label, count in labels:
        print(
            "{percentage:5.2f}% {label:<{label_len}} "
            "({count:>{count_len}} elements)".format(
                label=label,
                count=count,
                percentage=count / float(total_count) * 100.0,
                label_len=label_len,
                count_len=count_len,
            )
        )


def get_distribution(labels: List[str]) -> Dict[str, int]:
    """
    Get the distribution of the labels.

    Prameters
    ---------
    labels : List[str]
        This list is non-unique.

    Returns
    -------
    distribution : Dict[str, int]
        Maps (label => count)

    Examples
    --------
    >>> dist = get_distribution(['de', 'de', 'en'])
    >>> sorted(dist.items())
    [('de', 2), ('en', 1)]
    """
    distribution: Dict[str, int] = {}
    for label in labels:
        if label not in distribution:
            distribution[label] = 1
        else:
            distribution[label] += 1
    return distribution
