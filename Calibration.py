import numpy as np
import cv2
from numpy import matmul as dot
from numpy.linalg import inv
import utils as ut
from numpy import concatenate as concat
import random


def computeWorldPoints(refImagePoints, refWorldPoints, imagePoints):

    refWorldPoints = np.array(refWorldPoints)
    refImagePoints = np.array(refImagePoints)
    imagePoints = np.array(imagePoints)

    C = refWorldPoints[:, 0, :]
    B = refWorldPoints[:, 1, :]
    A = refWorldPoints[:, 2, :]
    c = refImagePoints[:, 0, :]
    b = refImagePoints[:, 1, :]
    a = refImagePoints[:, 2, :]
    p = imagePoints[:, :]

    k = (a-p)*(b-c)/(a-c)/(b-p)*2
    P = (A-k*B)/(1-k)

    return P


def getExtremePoints(contour, direction):

    # direction = U/B/L/R
    if direction == 'U':
        return tuple(contour[:][contour[:, 1].argmin()])
    elif direction == 'B':
        return tuple(contour[:][contour[:, 1].argmax()])
    elif direction == 'L':
        return tuple(contour[:][contour[:, 0].argmin()])
    else:
        return tuple(contour[:][contour[:, 0].argmax()])


def process(contours, restriction):

    leftmost, rightmost, topmost, bottommost = restriction
    newContours = []
    for contour in contours:
        temp = []
        for i in range(len(contour)):
            if rightmost>contour[i, 0, 0]>leftmost and bottommost>contour[i, 0, 1]>topmost:
                temp.append(contour[i, 0, :])
        if len(temp)>0:
            newContours.append(np.array(temp))  # [n][m, 2]
    newContours = sorted(newContours, key=lambda newContour: getExtremePoints(newContour, 'L')[0])

    return newContours


def main(path, pic, refSize, criteria, refInds, refLazerInds):

    # Set up constant parameters
    sideLength = 0.02  # ==20mm
    num = refSize[0] * refSize[1]
    n, m = len(refInds), len(refInds[0])

    # Read in imgs in gray
    img = cv2.imread(path + pic)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Automatically prepare objpoints
    objp = np.ones((refSize[0] * refSize[1], 3), np.float32)*0
    objp[:, :2] = np.mgrid[0:refSize[0], 0:refSize[1]].T.reshape(-1, 2)*sideLength

    # Try to find corners from the original picture
    ret, rawCorners = cv2.findChessboardCorners(grayImg, refSize, None)
    # shape = (actual_num_column-2, actual_num_row-2)
    print(ret)
    if not ret:
        print('Failed')
        return
    # Refine the corner locations
    refinedCorners = cv2.cornerSubPix(grayImg, rawCorners, (11, 11), (-1, -1), criteria)
    leftmost = np.min(refinedCorners[:, 0, 0])
    rightmost = np.max(refinedCorners[:, 0, 0])
    topmost = np.min(refinedCorners[:, 0, 1])
    bottommost = np.max(refinedCorners[:, 0, 1])

    objps = []
    imgps = []
    objps.append(objp)
    imgps.append(refinedCorners)
    imgp = np.reshape(refinedCorners, (num, 2))

    # Compute parameters from the original picture
    ret, inParam, distMat, rotVecs, tranVecs = cv2.calibrateCamera(objps, imgps, grayImg.shape[::-1], None, None)
    rotVec = rotVecs[0]
    tranVec = tranVecs[0]
    rotMat, _ = cv2.Rodrigues(rotVec)

    h, w = grayImg.shape  # (1944, 2592)
    _, thr = cv2.threshold(grayImg, 250, 255, cv2.THRESH_BINARY)
    img2, contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = process(contours, (leftmost, rightmost, topmost, bottommost))

    cv2.imwrite(path+'contours_'+pic, img2)
    # cnt = cv2.drawContours(img2, contours, -1, (0, 0, 0), 1)

    refImage = np.reshape(np.array(sorted(np.array(refinedCorners), key=lambda point: point[0, 0])),
                          [6, 9, 2])
    refImagePoints = np.array([[refImage[refInds[i][j][0], refInds[i][j][1]] for j in range(m)] for i in range(n)])
    imagePoints = [
        getExtremePoints(contours[refLazerInds[i][0]], refLazerInds[i][1]) for i in range(n)
        # (2010, 1135),

    ]
    refWorldPoints = (20*np.array(refInds))

    Xp = computeWorldPoints(refImagePoints, refWorldPoints, imagePoints)
    print(Xp)
    aug_Xp = ut.aug(ut.aug(Xp, -1, 0), -1, 1).T  # (4, 2)
    RTmat = np.concatenate((rotMat, tranVec), axis=-1)  # (3, 4)
    Xc = np.matmul(RTmat, aug_Xp)  # (3, 2)

    # x1 = np.array([0, 0, 0, 1])
    # x2 = np.array([0, 20, 0, 1])
    # print(np.matmul(RTmat, x1))
    # print(np.matmul(RTmat, x2))

    # Visualization
    refImage = np.reshape(np.array(refinedCorners), [-1, 2]).tolist()
    for point in refImage:
        point = (int(point[0]), int(point[1]))
        cv2.circle(img, point, 5, (255, 0, 0), -1)
    for contour in contours:
        for point in contour:
            point = (int(point[0]), int(point[1]))
            cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
    for point in imagePoints:
        point = (int(point[0]), int(point[1]))
        cv2.circle(img, point, 5, (0, 255, 0), -1)
    cv2.imwrite(path+'test.bmp', img)

    return Xc


if __name__ == '__main__':

    path = 'G:\Code\\tiller-counting\\0410/'
    pics = ['undistImage%i.bmp' %i for i in (2, 5, 6)]
    Xcs = []

    refInds = [
        [[[0, 1], [0, 2], [0, 3]],
         [[2, 2], [2, 3], [2, 4]],
         [[2, 3], [3, 3], [4, 3]],
         [[3, 3], [3, 4], [3, 5]],
         [[3, 4], [4, 4], [5, 4]],
         [[4, 4], [4, 5], [4, 6]],
         [[5, 5], [4, 5], [3, 5]],
         [[5, 5], [5, 6], [5, 7]]],

        [[[0, 3], [0, 4], [0, 5]],
         [[0, 4], [1, 4], [2, 4]],
         [[1, 4], [1, 5], [1, 6]],
         [[1, 5], [2, 5], [3, 5]],
         [[2, 5], [2, 6], [2, 7]],
         [[4, 6], [4, 7], [4, 8]],
         [[5, 7], [4, 7], [3, 7]],
         [[5, 8], [5, 7], [5, 6]]],
    ]
    refLazerInds = [
        [[0, 'L'], [1, 'R'], [2, 'U'], [2, 'R'], [3, 'U'], [3, 'R'], [4, 'U'], [4, 'R']],
        [[0, 'L'], [0, 'B'], [1, 'L'], [1, 'B'], [2, 'L'], [3, 'R'], [4, 'U'], [7, 'R']]
    ]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for i in range(0, 2):
        print(pics[i])
        Xc = main(path, pics[i], (9, 6), criteria, refInds[i], refLazerInds[i])
        Xcs.append(Xc)

    Xc = np.concatenate(Xcs, axis=-1)
    num = Xc.shape[1]
    Pc = [np.reshape(Xc[:, i], (3, 1)) for i in range(num)]
    print(Xc.shape)
    for k in range(10):
        samples = random.sample(range(num), num)
        print(samples)
        Pc_1, Pc_2, Pc_3 = (Pc[i] for i in samples[:3])
        Xc = np.concatenate((Pc_1, Pc_2, Pc_3), axis=-1)
        error = 0
        for l in samples[3:]:
            test_Xc = Pc[l]
            test_x = test_Xc[0, 0]
            test_y = test_Xc[1, 0]
            test_z = test_Xc[2, 0]
            params = np.matmul(np.ones((1, 3)), np.linalg.inv(Xc)).tolist()
            a, b, c = params[0]
            error += abs((1-a*test_x-b*test_y)/c-test_z)
        print(error/(len(samples)-3))
