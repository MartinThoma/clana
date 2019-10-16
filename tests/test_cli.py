#!/usr/bin/env python

"""Test the CLI functions."""

# Third party
# 3rd party modules
from click.testing import CliRunner
from pkg_resources import resource_filename

# First party
# internal modules
import clana.cli


def test_visualize():
    runner = CliRunner()
    cm_path = resource_filename(__name__, "examples/cm-2-classes.json")
    commands = ["visualize", "--cm", cm_path]
    result = runner.invoke(clana.cli.entry_point, commands)
    assert result.exit_code == 0
    assert "Accuracy: 76.47%" in result.output, "clana" + " ".join(commands)
