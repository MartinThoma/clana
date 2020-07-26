# Third party
import numpy as np

# First party
import clana.optimize


def test_move_1d() -> None:
    perm = np.array([8, 7, 6, 1, 2])
    from_start = 1
    from_end = 2
    insert_pos = 0
    new_perm = clana.optimize.move_1d(perm, from_start, from_end, insert_pos)
    new_perm = new_perm.tolist()
    assert new_perm == [7, 6, 8, 1, 2]
