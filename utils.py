import numpy as np


def aug(mat, axis, value=1):

    shape = list(mat.shape)
    shape[axis] = 1
    mat = np.concatenate((mat, np.ones(shape)*value), axis=axis)

    return mat