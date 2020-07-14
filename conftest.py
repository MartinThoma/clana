"""Configure pytest."""

# Core Library
import logging
from typing import Any, Dict


def pytest_configure(config: Dict[str, Any]) -> None:
    """Flake8 is to verbose. Mute it."""
    logging.getLogger("flake8").setLevel(logging.WARN)
    logging.getLogger("pydocstyle").setLevel(logging.INFO)
