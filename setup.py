"""CLANA is a toolkit for classifier analysis."""

from setuptools import find_packages
from setuptools import setup
import io
import os

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain long description from README and CHANGES
    # Specify encoding to get a unicode type in Python 2 and a str in Python 3
    with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
        README = f.read()
except IOError:
    README = ''

config = {
    'name': 'clana',
    'version': '0.2.0',
    'author': 'Martin Thoma',
    'author_email': 'info@martin-thoma.de',
    'maintainer': 'Martin Thoma',
    'maintainer_email': 'info@martin-thoma.de',
    'packages': find_packages(),
    'scripts': ['bin/clana'],
    'platforms': ['Linux'],
    'url': 'https://github.com/MartinThoma/clana',
    'license': 'MIT',
    'description': __doc__,
    'long_description': README,
    'install_requires': [
        'click>=6.7',
        'matplotlib>=2.1.1',
        'numpy>=1.14.0',
        'PyYAML>=3.12',
        'scikit-learn>=0.19.1',
        'scipy>=1.0.0',
    ],
    'keywords': ['Machine Learning', 'Data Science'],
    'download_url': 'https://github.com/MartinThoma/language-identification',
    'classifiers': ['Development Status :: 2 - Pre-Alpha',
                    'Environment :: Console',
                    'Intended Audience :: Developers',
                    'Intended Audience :: Science/Research',
                    'Intended Audience :: Information Technology',
                    'License :: OSI Approved :: MIT License',
                    'Natural Language :: English',
                    'Programming Language :: Python :: 3.5',
                    'Topic :: Scientific/Engineering :: Information Analysis',
                    'Topic :: Software Development',
                    'Topic :: Utilities'],
    'zip_safe': False,
}

setup(**config)
