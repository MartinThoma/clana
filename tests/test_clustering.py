# Third party
import numpy as np

# First party
import clana.clustering


def test_extract_clusters_local() -> None:
    n = 10
    cm = np.random.randint(low=0, high=100, size=(n, n))
    clana.clustering.extract_clusters(
        cm, labels=list(range(n)), steps=10, method="local-connectivity"  # type: ignore
    )


def test_extract_clusters_energy() -> None:
    n = 10
    cm = np.random.randint(low=0, high=100, size=(n, n))
    clana.clustering.extract_clusters(
        cm, labels=list(range(n)), steps=10, method="energy"  # type: ignore
    )
