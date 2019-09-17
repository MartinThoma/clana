# core modules
import csv
import hashlib
import json
import os

# 3rd party modules
import numpy as np
import yaml

# internal modules
import clana.utils


class ClanaCfg(object):
    @classmethod
    def read_clana_cfg(cls, cfg_file):
        """
        Read a .clana config file which contains permutations.

        Parameters
        ----------
        cfg_file : str

        Returns
        -------
        cfg : Dict[str, Any]
        """
        if os.path.isfile(cfg_file):
            with open(cfg_file, "r") as stream:
                cfg = yaml.safe_load(stream)
        else:
            cfg = {"version": clana.__version__, "data": {}}
        return cfg

    @classmethod
    def get_cfg_path_from_cm_path(cls, cm_file):
        return os.path.join(os.path.dirname(os.path.abspath(cm_file)), ".clana")

    @classmethod
    def get_perm(cls, cm_file):
        """
        Get the best permutation found so far for a given cm_file.

        Fallback: list(range(n))

        Parameters
        ----------
        cm_file : str

        Returns
        -------
        perm : List[int]
        """
        cfg_file = cls.get_cfg_path_from_cm_path(cm_file)
        cfg = cls.read_clana_cfg(cfg_file)
        cm_file_base = os.path.basename(cm_file)
        cm = read_confusion_matrix(cm_file)
        n = len(cm)
        perm = list(range(n))
        if cm_file_base in cfg["data"]:
            cm_file_md5 = md5(cm_file)
            if cm_file_md5 in cfg["data"][cm_file_base]:
                print(
                    "Loaded permutation found in {} iterations".format(
                        cfg["data"][cm_file_base][cm_file_md5]["iterations"]
                    )
                )
                perm = cfg["data"][cm_file_base][cm_file_md5]["permutation"]
        return perm

    @classmethod
    def store_permutation(cls, cm_file, permutation, iterations):
        """
        Store a permutation.

        Parameters
        ----------
        cm_file : str
        permutation : List[int]
        iterations : int
        """
        cm_file = os.path.abspath(cm_file)
        cfg_file = cls.get_cfg_path_from_cm_path(cm_file)
        if os.path.isfile(cfg_file):
            cfg = ClanaCfg.read_clana_cfg(cfg_file)
        else:
            cfg = {"version": clana.__version__, "data": {}}

        cm_file_base = os.path.basename(cm_file)
        if cm_file_base not in cfg["data"]:
            cfg["data"][cm_file_base] = {}
        cm_file_md5 = md5(cm_file)
        if cm_file_md5 not in cfg["data"][cm_file_base]:
            cfg["data"][cm_file_base][cm_file_md5] = {
                "permutation": permutation.tolist(),
                "iterations": 0,
            }
        cfg["data"][cm_file_base][cm_file_md5]["permutation"] = permutation.tolist()
        cfg["data"][cm_file_base][cm_file_md5]["iterations"] += iterations

        # Write file
        print(cfg_file)
        with open(cfg_file, "w") as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False, allow_unicode=True)


def read_confusion_matrix(cm_file, make_max=float("inf")):
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
        if cm_file.lower().endswith("csv"):
            cm = []
            with open(cm_file, newline="") as csvfile:
                spamreader = csv.reader(csvfile, delimiter=",", quotechar='"')
                for row in spamreader:
                    cm.append([int(el) for el in row])
        else:
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


def read_permutation(cm_file, perm_file):
    """
    Load permutation.

    Parameters
    ----------
    cm_file : str
    perm_file : str or None
        Path to a JSON file which contains a permutation of n numbers.

    Returns
    -------
    perm : List[int]
        Permutation of the numbers 0, ..., n-1
    """
    assert os.path.isfile(cm_file)
    cm = read_confusion_matrix(cm_file)
    n = len(cm)
    if perm_file is not None and os.path.isfile(perm_file):
        with open(perm_file) as data_file:
            if perm_file.lower().endswith("csv"):
                with open(perm_file) as file:
                    content = file.read()
                perm = [int(el) for el in content.split(",")]
            else:
                perm = json.load(data_file)
    else:
        perm = ClanaCfg.get_perm(cm_file)
    return perm


def read_labels(labels_file, n):
    """
    Load labels.

    Please note that this contains one additional "UNK" label for
    unknown classes.

    Parameters
    ----------
    labels_file : str
    n : int

    Returns
    -------
    labels : List[str]
    """
    labels = clana.utils.load_labels(labels_file, n)
    labels.append("UNK")
    return labels


def write_labels(labels_file, labels):
    with open(labels_file, "w") as outfile:
        str_ = json.dumps(labels, indent=2, separators=(",", ": "), ensure_ascii=False)
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
    with open(filepath, "w") as f:
        for identifier, prediction in identifier2prediction.items():
            f.write("{};{}\n".format(identifier, prediction))


def write_gt(identifier2label, filepath):
    """
    Write ground truth to a file.

    Parameters
    ----------
    identifier2label : Dict[str, str]
    filepath : str
        Write to this CSV file.
    """
    with open(filepath, "w") as f:
        for identifier, label in identifier2label.items():
            f.write("{};{}\n".format(identifier, label))


def write_cm(path, cm):
    """
    Write confusion matrix to path.

    Parameters
    ----------
    path : str
    cm : ndarray
    """
    with open(path, "w") as outfile:
        str_ = json.dumps(cm.tolist(), separators=(",", ": "), ensure_ascii=False)
        outfile.write(str_)


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
