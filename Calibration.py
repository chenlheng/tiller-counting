import numpy as np
import cv2
from numpy import matmul as dot
from numpy.linalg import inv
import utils as ut
from numpy import concatenate as concat


def main(pic, refSize, criteria):

    # Set up constant parameters
    sideLength = 0.02  # ==20mm
    num = refSize[0] * refSize[1]

    # Read in imgs in gray
    img = cv2.imread(path + pic)
    Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Automatically prepare objpoints
    objp = np.ones((refSize[0] * refSize[1], 3), np.float32)*0
    objp[:, :2] = np.mgrid[0:refSize[0], 0:refSize[1]].T.reshape(-1, 2)*sideLength

    # Try to find corners from the original picture
    ret, rawCorners = cv2.findChessboardCorners(Img, refSize, None)
    # shape = (actual_num_column-2, actual_num_row-2)

    if not ret:
        print('Failed')
        return
    # Refine the corner locations
    refinedCorners = cv2.cornerSubPix(Img, rawCorners, (11, 11), (-1, -1), criteria)

    objps = []
    imgps = []
    objps.append(objp)
    imgps.append(refinedCorners)
    imgp = np.reshape(refinedCorners, (num, 2))

    # Compute parameters from the original picture
    ret, inParam, distMat, rotVecs, tranVecs = cv2.calibrateCamera(objps, imgps, Img.shape[::-1], None, None)
    print(inParam)
    rotVec = rotVecs[0]
    tranVec = tranVecs[0]
    rotMat, _ = cv2.Rodrigues(rotVec)


if __name__ == '__main__':

    path = 'G:\Code\\tiller-counting\\0409_camera_calibration/'
    pics = ['undistImage%i.bmp' %i for i in range(1, 31)]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for i in range(len(pics)):
        print(pics[i])
        main(pics[i], (9, 6), criteria)