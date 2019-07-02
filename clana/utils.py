#!/usr/bin/env python

"""Utility functions for clana."""

# core modules
import csv
import os
import pkg_resources

# 3rd party modules
import yaml


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


def load_cfg(yaml_filepath=None):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str, optional (default: package config file)

    Returns
    -------
    cfg : dict
    """
    if yaml_filepath is None:
        yaml_filepath = pkg_resources.resource_filename('clana', 'config.yaml')
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.safe_load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg


def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if hasattr(key, 'endswith') and key.endswith("_path"):
            if cfg[key].startswith('~'):
                cfg[key] = os.path.expanduser(cfg[key])
            else:
                cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg
