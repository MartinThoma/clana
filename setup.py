"""CLANA is a toolkit for classifier analysis."""

from setuptools import find_packages
from setuptools import setup
import io
import os

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain long description from README and CHANGES
    # Specify encoding to get a unicode type in Python 2 and a str in Python 3
    readme_path = os.path.join(here, "README.md")
    with io.open(readme_path, "r", encoding="utf-8") as f:
        README = f.read()
except IOError:
    README = ""

requires_tests = [
    "pytest",
    "pytest-cov",
    "pytest-mccabe",
    "pytest-flake8",
    "pytest-black",
]

install_requires = [
    "click>=6.7",
    "jinja2",
    "matplotlib>=2.1.1",
    "numpy>=1.14.0",
    "PyYAML>=5.1.1",
    "scikit-learn>=0.19.1",
    "scipy>=1.0.0",
]


config = {
    "name": "clana",
    "version": "0.3.9",
    "author": "Martin Thoma",
    "author_email": "info@martin-thoma.de",
    "maintainer": "Martin Thoma",
    "maintainer_email": "info@martin-thoma.de",
    "packages": find_packages(),
    "entry_points": {"console_scripts": ["clana=clana.cli:entry_point"]},
    "platforms": ["Linux"],
    "url": "https://github.com/MartinThoma/clana",
    "license": "MIT",
    "description": __doc__,
    "long_description": README,
    "long_description_content_type": "text/markdown",
    "install_requires": install_requires,
    "tests_require": requires_tests,
    "keywords": ["Machine Learning", "Data Science"],
    "download_url": "https://github.com/MartinThoma/clana",
    "package_data": {"clana": ["clana/config.yaml"]},
    "include_package_data": True,
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development",
        "Topic :: Utilities",
    ],
    "zip_safe": False,
}

setup(**config)
