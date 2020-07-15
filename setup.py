"""CLANA is a toolkit for classifier analysis."""

# Third party
from setuptools import setup

tests_requires = [
    # coverage is a transitive requirement, introduced by pytest-cov
    # Due to https://github.com/nedbat/coveragepy/issues/716 it is fixed here
    "coverage<5.0.0",
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

all_requires = install_requires + tests_requires


setup(
    install_requires=install_requires,
    extras_require={"tests": tests_requires, "all": all_requires},
)
