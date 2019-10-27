"""Get the version."""

# Third party
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("clana").version
except pkg_resources.DistributionNotFound:
    __version__ = "not installed"
