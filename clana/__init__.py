import pkg_resources
try:
    __version__ = pkg_resources.get_distribution('clana').version
except:
    __version__ = 'not installed'
