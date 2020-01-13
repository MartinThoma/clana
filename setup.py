"""CLANA is a toolkit for classifier analysis."""

# Third party
from setuptools import setup

requires_tests = [
    "pytest",
    "pytest-black",
    "pytest-cov",
    "pytest-flake8",
    "pytest-mccabe",
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


setup(
    entry_points={"console_scripts": ["clana=clana.cli:entry_point"]},
    description=__doc__,
    install_requires=install_requires,
    tests_require=requires_tests,
    package_data={"clana": ["clana/config.yaml"]},
)
