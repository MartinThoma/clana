"""
Optimize confusion matrix.

For more information, see

* http://cs.stackexchange.com/q/70627/2914
* http://datascience.stackexchange.com/q/17079/8820
"""

# Core Library
import json
import logging
from typing import List, Optional, Tuple

# Third party
import numpy as np
import numpy.typing as npt
from jinja2 import Template
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pkg_resources import resource_filename
from sklearn.metrics import silhouette_score

# First party
import clana.clustering
import clana.cm_metrics
import clana.io
import clana.utils
from clana.optimize import (
    calculate_score,
    calculate_weight_matrix,
    simulated_annealing,
)

cfg = clana.utils.load_cfg()
logger = logging.getLogger(__name__)


def main(
    cm_file: str,
    perm_file: str,
    steps: int,
    labels_file: str,
    zero_diagonal: bool,
    limit_classes: Optional[int] = None,
    output: Optional[str] = None,
) -> None:
    """
    Run optimization and generate output.

    Parameters
    ----------
    cm_file : str
    perm_file : str
    steps : int
    labels_file : str
    zero_diagonal : bool
    limit_classes : int, optional (default: no limit)
    output : str
    """
    cm = clana.io.read_confusion_matrix(cm_file)
    perm = clana.io.read_permutation(cm_file, perm_file)
    labels = clana.io.read_labels(labels_file, len(cm))
    n, m = cm.shape
    if n != m:
        raise ValueError(
            f"Confusion matrix is expected to be square, but was {n} x {m}"
        )
    if len(labels) - 1 != n:
        print(
            "Confusion matrix is {n} x {n}, but len(labels)={nb_labels}".format(
                n=n, nb_labels=len(labels)
            )
        )

    cm_orig = cm.copy()

    get_cm_problems(cm, labels)

    weights = calculate_weight_matrix(len(cm))
    print(f"Score: {calculate_score(cm, weights)}")
    result = simulated_annealing(
        cm, perm, score=calculate_score, deterministic=True, steps=steps
    )
    print(f"Score: {calculate_score(result.cm, weights)}")
    print(f"Perm: {list(result.perm)}")
    clana.io.ClanaCfg.store_permutation(cm_file, result.perm, steps)
    labels = [labels[i] for i in result.perm]
    class_indices = list(range(len(labels)))
    class_indices = [class_indices[i] for i in result.perm]
    logger.info(f"Classes: {labels}")
    acc = clana.cm_metrics.get_accuracy(cm_orig)
    print(f"Accuracy: {acc * 100:0.2f}%")
    start = 0
    if limit_classes is None:
        limit_classes = len(cm)
    if output is None:
        output = cfg["visualize"]["save_path"]
    plot_cm(
        result.cm[start:limit_classes, start:limit_classes],
        zero_diagonal=zero_diagonal,
        labels=labels[start:limit_classes],
        output=output,
    )
    create_html_cm(
        result.cm[start:limit_classes, start:limit_classes],
        zero_diagonal=zero_diagonal,
        labels=labels[start:limit_classes],
    )
    if len(cm) < 5:
        print(
            f"You only have {len(cm)} classes. Clustering for less than "
            "5 classes should be done manually."
        )
        return
    grouping = clana.clustering.extract_clusters(result.cm, labels)
    y_pred = [0]
    cluster_i = 0
    for el in grouping:
        if el:
            cluster_i += 1
        y_pred.append(cluster_i)
    logger.info(f"silhouette_score={silhouette_score(cm, y_pred)}")
    # Store grouping as hierarchy
    with open(cfg["visualize"]["hierarchy_path"], "w") as outfile:
        hierarchy = clana.clustering.apply_grouping(class_indices, grouping)
        hierarchy_mixed = clana.clustering._remove_single_element_groups(hierarchy)
        str_ = json.dumps(
            hierarchy_mixed,
            indent=4,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        outfile.write(str_)

    # Print nice
    for group in clana.clustering.apply_grouping(labels, grouping):
        print(f"\t{len(group)}: {list(group)}")


def get_cm_problems(cm: npt.NDArray, labels: List[str]) -> None:
    """
    Find problems of a classifier by analzing its confusion matrix.

    Parameters
    ----------
    cm : ndarray
    labels : List[str]
    """
    n = len(cm)

    # Find classes which are not present in the dataset
    for i in range(n):
        if sum(cm[i]) == 0:
            logger.warning(f"The class '{labels[i]}' was not in the dataset.")

    # Find classes which are never predicted
    cm = cm.transpose()
    never_predicted = []
    for i in range(n):
        if sum(cm[i]) == 0:
            never_predicted.append(labels[i])
    if len(never_predicted) > 0:
        logger.warning(f"The following classes were never predicted: {never_predicted}")


def plot_cm(
    cm: npt.NDArray,
    zero_diagonal: bool = False,
    labels: Optional[List[str]] = None,
    output: str = cfg["visualize"]["save_path"],
) -> None:
    """
    Plot a confusion matrix.

    Parameters
    ----------
    cm : npt.NDArray
    zero_diagonal : bool, optional (default: False)
    labels : Optional[List[str]]
        If this is not given, then numbers are assigned to the classes
    """
    # Third party
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    n = len(cm)
    if zero_diagonal:
        for i in range(n):
            cm[i][i] = 0
    if n > 20:
        size = int(n / 4.0)
    else:
        size = 5
    fig = plt.figure(figsize=(size, size), dpi=80)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if labels is None:
        labels = [str(i) for i in range(len(cm))]
    x = list(range(len(cm)))
    plt.xticks(x, labels, rotation=cfg["visualize"]["xlabels_rotation"])
    y = list(range(len(cm)))
    plt.yticks(y, labels, rotation=cfg["visualize"]["ylabels_rotation"])
    if cfg["visualize"]["norm"] == "LogNorm":
        norm = LogNorm(vmin=max(1, np.min(cm)), vmax=np.max(cm))  # type: ignore
    elif cfg["visualize"]["norm"] is None:
        norm = None
    else:
        raise NotImplementedError(
            f"visualize->norm={cfg['visualize']['norm']} is not implemented. "
            "Try None or LogNorm"
        )
    res = ax.imshow(
        np.array(cm),
        cmap=cfg["visualize"]["colormap"],
        interpolation=cfg["visualize"]["interpolation"],
        norm=norm,
    )
    width, height = cm.shape

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(res, cax=cax)
    plt.tight_layout()

    logger.info(f"Save figure at '{output}'")
    plt.savefig(output)


def create_html_cm(
    cm: npt.NDArray, zero_diagonal: bool = False, labels: Optional[List[str]] = None
) -> None:
    """
    Plot a confusion matrix.

    Parameters
    ----------
    cm : npt.NDArray
    zero_diagonal : bool, optional (default: False)
        If this is set to True, then the diagonal is overwritten with zeroes.
    labels : Optional[List[str]]
        If this is not given, then numbers are assigned to the classes
    """
    if labels is None:
        labels = [str(i) for i in range(len(cm))]

    el_max = 200

    template_path = resource_filename("clana", "templates/base.html")
    with open(template_path) as f:
        base = f.read()

    cm_t = cm.transpose()
    header_cells = []
    for i, label in enumerate(labels):
        precision = cm[i][i] / float(sum(cm_t[i]))
        background_color = "transparent"
        if precision < 0.2:
            background_color = "red"
        elif precision > 0.98:
            background_color = "green"
        header_cells.append(
            {
                "precision": f"{precision:0.2f}",
                "background-color": background_color,
                "label": label,
            }
        )

    body_rows = []
    for i, label, row in zip(range(len(labels)), labels, cm):
        body_row = []
        row_str = [str(el) for el in row]
        support = sum(row)
        recall = cm[i][i] / float(support)
        background_color = "transparent"
        if recall < 0.2:
            background_color = "red"
        elif recall >= 0.98:
            background_color = "green"
        body_row.append(
            {
                "label": label,
                "recall": f"{recall:.2f}",
                "background-color": background_color,
            }
        )
        for _j, pred_label, el in zip(range(len(labels)), labels, row_str):
            background_color = "transparent"
            if el == "0":
                el = ""
            else:
                background_color = get_color_code(float(el), el_max)

            body_row.append(
                {
                    "label": el,
                    "true": label,
                    "pred": pred_label,
                    "background-color": background_color,
                }
            )

        body_rows.append({"row": body_row, "support": support})

    html_template = Template(base)
    html = html_template.render(header_cells=header_cells, body_rows=body_rows)

    with open(cfg["visualize"]["html_save_path"], "w") as f:
        f.write(html)


def get_color(white_to_black: float) -> Tuple[int, int, int]:
    """
    Get grayscale color.

    Parameters
    ----------
    white_to_black : float

    Returns
    -------
    color : Tuple

    Examples
    --------
    >>> get_color(0)
    (255, 255, 255)
    >>> get_color(0.5)
    (128, 128, 128)
    >>> get_color(1)
    (0, 0, 0)
    """
    if not (0 <= white_to_black <= 1):
        raise ValueError(
            f"white_to_black={white_to_black} is not in the interval [0, 1]"
        )

    index = 255 - int(255 * white_to_black)
    r, g, b = index, index, index
    return int(r), int(g), int(b)


def get_color_code(val: float, max_val: float) -> str:
    """
    Get a HTML color code which is between 0 and max_val.

    Parameters
    ----------
    val : number
    max_val : number

    Returns
    -------
    color_code : str

    Examples
    --------
    >>> get_color_code(0, 100)
    '#ffffff'
    >>> get_color_code(100, 100)
    '#000000'
    >>> get_color_code(50, 100)
    '#808080'
    """
    value = min(1.0, float(val) / max_val)
    r, g, b = get_color(value)
    return f"#{r:02x}{g:02x}{b:02x}"
