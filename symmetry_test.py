import numpy as np
import random

from symmetry import *

def test_d2():
    x = np.matrix([[1, 2, 3], [4, 5, 6]])
    assert(np.array_equal(dihedral(x, 0), [[1, 2, 3], [4, 5, 6]]))
    assert(np.array_equal(dihedral(x, 1), [[4, 5, 6], [1, 2, 3]]))
    assert(np.array_equal(dihedral(x, 2), [[3, 2, 1], [6, 5, 4]]))
    assert(np.array_equal(dihedral(x, 3), [[6, 5, 4], [3, 2, 1]]))
    assert(np.array_equal(dihedral(x, 4), [[1, 2, 3], [4, 5, 6]]))


def test_d4():
    x = np.matrix([[1, 2], [3, 4]])
    assert(np.array_equal(dihedral(x, 0), [[1, 2], [3, 4]]))
    assert(np.array_equal(dihedral(x, 1), [[2, 4], [1, 3]]))
    assert(np.array_equal(dihedral(x, 2), [[4, 3], [2, 1]]))
    assert(np.array_equal(dihedral(x, 3), [[3, 1], [4, 2]]))
    assert(np.array_equal(dihedral(x, 4), [[3, 4], [1, 2]]))
    assert(np.array_equal(dihedral(x, 5), [[4, 2], [3, 1]]))
    assert(np.array_equal(dihedral(x, 6), [[2, 1], [4, 3]]))
    assert(np.array_equal(dihedral(x, 7), [[1, 3], [2, 4]]))


if __name__ == '__main__':
    test_d2()
    test_d4()
