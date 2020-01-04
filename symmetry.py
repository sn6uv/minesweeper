# Basic (dihedral) symmetry library for working with numpy arrays.
#
# Useful for data augmentation.

import random
import numpy as np


_d2 = [
    lambda x: x,
    lambda x: np.flip(x, axis=0),
    lambda x: np.flip(x, axis=1),
    lambda x: np.flip(np.flip(x, axis=0), axis=1),
]

_d4 = [
    lambda x: np.rot90(x, 0),
    lambda x: np.rot90(x, 1),
    lambda x: np.rot90(x, 2),
    lambda x: np.rot90(x, 3),
    lambda x: np.rot90(np.flip(x, axis=0), 0),
    lambda x: np.rot90(np.flip(x, axis=0), 1),
    lambda x: np.rot90(np.flip(x, axis=0), 2),
    lambda x: np.rot90(np.flip(x, axis=0), 3),
]

def dihedral(x, symmetry_index=None):
    '''Returns a random symmetry of an element in D2/D4'''
    if symmetry_index is None:
        symmetry_index = random.randint(0, 8)
    if x.shape[0] == x.shape[1]:
        f = _d4[symmetry_index % len(_d4)]
    else:
        f = _d2[symmetry_index % len(_d2)]
    return f(x)
