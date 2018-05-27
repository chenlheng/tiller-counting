import numpy as np
import os
import cv2


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
            img = cv2.imread(path+file, 0)
            if img.shape[0] == 640 and img.shape[1] == 1600:
                files.append(file)

    return files


