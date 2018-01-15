"""CLANA is a toolkit for classifier analysis."""

from setuptools import find_packages
from setuptools import setup
import os
# We need io.open() (Python 3's default open) to specify file encodings
import io

# internal modules
import clana

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
    'version': clana.__version__,
    'author': 'Martin Thoma',
    'author_email': 'info@martin-thoma.de',
    'maintainer': 'Martin Thoma',
    'maintainer_email': 'info@martin-thoma.de',
    'packages': find_packages(),
    'scripts': ['bin/clana'],
    # 'package_data': {'hwrt': ['templates/*', 'misc/*']},
    'platforms': ['Linux'],
    'url': 'https://github.com/MartinThoma/clana',
    'license': 'MIT',
    'description': __doc__,
    'long_description': README,
    'install_requires': [
        'numpy',
        'PyYAML',
        'matplotlib',
        'sklearn',
        'scipy',
        'click',
    ],
    'keywords': ['Machine Learning', 'Data Science'],
    'download_url': 'https://github.com/MartinThoma/language-identification',
    'classifiers': ['Development Status :: 1 - Planning',
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
