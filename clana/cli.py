#!/usr/bin/env python

"""
clana is a toolkit for classifier analysis.

It specifies some file formats and comes with some tools for typical tasks of
classifier analysis.
"""
# core modules
import logging.config
import random

# 3rd party modules
import click
import matplotlib
matplotlib.use('Agg')  # noqa

# internal modules
import clana
import clana.visualize_cm
import clana.get_cm
import clana.get_cm_simple
import clana.distribution

config = clana.utils.load_cfg()
logging.config.dictConfig(config['LOGGING'])
logging.getLogger('matplotlib').setLevel('WARN')
random.seed(0)


@click.group()
@click.version_option(version=clana.__version__)
def entry_point():
    pass


@entry_point.command(name='get-cm-simple')
@click.option('--labels', 'label_filepath',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
@click.option('--gt', 'gt_filepath',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
@click.option('--predictions', 'predictions_filepath',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
@click.option('--clean',
              default=False,
              is_flag=True,
              help='Remove classes that the classifier doesn\'t know')
def get_cm_simple(label_filepath, gt_filepath, predictions_filepath, clean):
    clana.get_cm_simple.main(label_filepath,
                             gt_filepath,
                             predictions_filepath,
                             clean)


@entry_point.command(name='get-cm')
@click.option('--predictions', 'cm_dump_filepath',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
@click.option('--gt', 'gt_filepath',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
@click.option('--n', 'n',
              required=True,
              type=int,
              help='Number of classes')
def get_cm(cm_dump_filepath, gt_filepath, n):
    """Generate a confusion matrix from predictions and ground truth."""
    clana.get_cm.main(cm_dump_filepath, gt_filepath, n)


@entry_point.command(name='distribution')
@click.option('--gt', 'gt_filepath',
              required=True,
              type=click.Path(exists=True),
              help='List of labels for the dataset')
def distribution(gt_filepath):
    """Get the distribution of classes in a dataset."""
    clana.distribution.main(gt_filepath)


@entry_point.command(name='visualize')
@click.option('--cm', 'cm_file',
              type=click.Path(exists=True),
              required=True)
@click.option('--perm', 'perm_file',
              help='json file which defines a permutation to start with.',
              type=click.Path(),
              default='')
@click.option('--steps',
              default=1000,
              show_default=True,
              help='Number of steps to find a good permutation.')
@click.option('--labels', 'labels_file',
              default='')
@click.option('--zero_diagonal', is_flag=True)
@click.option('--limit_classes',
              type=int,
              help='Limit the number of classes in the output')
def visualize(cm_file,
              perm_file,
              steps,
              labels_file,
              zero_diagonal,
              limit_classes=None):
    """Optimize and visualize a confusion matrix."""
    clana.visualize_cm.main(cm_file,
                            perm_file,
                            steps,
                            labels_file,
                            zero_diagonal,
                            limit_classes)
