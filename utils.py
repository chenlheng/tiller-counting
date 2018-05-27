import numpy as np
import os
import cv2
import time
from functools import partial
import sys


def aug(mat, axis, value=1):

    shape = list(mat.shape)
    shape[axis] = 1
    mat = np.concatenate((mat, np.ones(shape)*value), axis=axis)

    return mat


def reformArray(dataArray):

    dataList = dataArray.tolist()

    return np.array(dataList)


def read_files(path, data_type):

    raw_files = os.listdir(path)
    files = []
    for file in raw_files:
        if file[-len(data_type):] == data_type:
            files.append(file)

    return files


def init_print():

    start_time = time.time()
    sprint = partial(static_print, start_time=start_time)
    dprint = partial(dynamic_print, start_time=start_time)

    return sprint, dprint


def static_print(message, start_time):

    if not type(message) == str:
        message = str(message)

    sys.stdout.write(' ' * 50 + '\r')
    sys.stdout.flush()
    print(message + '  [ %is ]' % (time.time() - start_time))


def dynamic_print(message, start_time):

    if not type(message) == str:
        message = str(message)

    sys.stdout.write(' ' * 50 + '\r')
    sys.stdout.flush()
    sys.stdout.write(message + '  [ %is ]' % (time.time() - start_time) + '\r')
    sys.stdout.flush()


