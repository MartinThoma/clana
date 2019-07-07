"""CLANA is a toolkit for classifier analysis."""

from setuptools import find_packages
from setuptools import setup
import io
import os

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain long description from README and CHANGES
    # Specify encoding to get a unicode type in Python 2 and a str in Python 3
    readme_path = os.path.join(here, 'docs/README.md')
    with io.open(readme_path, 'r', encoding='utf-8') as f:
        README = f.read()
except IOError:
    README = ''

config = {
    'name': 'clana',
    'version': '0.3.1',
    'author': 'Martin Thoma',
    'author_email': 'info@martin-thoma.de',
    'maintainer': 'Martin Thoma',
    'maintainer_email': 'info@martin-thoma.de',
    'packages': find_packages(),
    'entry_points': {
        'console_scripts': ['clana=clana.cli:entry_point']
    },
    'platforms': ['Linux'],
    'url': 'https://github.com/MartinThoma/clana',
    'license': 'MIT',
    'description': __doc__,
    'long_description': README,
    'long_description_content_type': 'text/markdown',
    'install_requires': [
        'click>=6.7',
        'matplotlib>=2.1.1',
        'numpy>=1.14.0',
        'PyYAML>=5.1.1',
        'scikit-learn>=0.19.1',
        'scipy>=1.0.0',
    ],
    'keywords': ['Machine Learning', 'Data Science'],
    'download_url': 'https://github.com/MartinThoma/language-identification',
    'classifiers': ['Development Status :: 3 - Alpha',
                    'Environment :: Console',
                    'Intended Audience :: Developers',
                    'Intended Audience :: Science/Research',
                    'Intended Audience :: Information Technology',
                    'License :: OSI Approved :: MIT License',
                    'Natural Language :: English',
                    'Programming Language :: Python :: 3.6',
                    'Topic :: Scientific/Engineering :: Information Analysis',
                    'Topic :: Software Development',
                    'Topic :: Utilities'],
    'zip_safe': False,
}

setup(**config)
