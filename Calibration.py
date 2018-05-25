import numpy as np
import cv2
from numpy import matmul as dot
from numpy.linalg import inv
import utils as ut
from numpy import concatenate as concat
import random
from scipy.io import savemat


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


def main(path, pics, refSize, criteria, refInds, refLazerInds):

    # Set up constant parameters
    sideLength = 20  # ==20mm
    num = len(pics)

    # Automatically prepare objpoints
    objp = np.ones((refSize[0] * refSize[1], 3), np.float32)*0
    objp[:, :2] = np.mgrid[0:refSize[0], 0:refSize[1]].T.reshape(-1, 2)*sideLength
    objps = []
    imgps = []
    grayImgs = []
    refinedCorners = []
    Xcs = []

    for pic in pics:
        # Read in imgs in gray
        img = cv2.imread(path + pic)
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (1944, 2592)

        # Try to find corners from the original picture
        ret, rawCorners = cv2.findChessboardCorners(grayImg, refSize, None)
        # shape = (actual_num_column-2, actual_num_row-2)
        print(ret)
        if not ret:
            print('Failed')
            return
        # Refine the corner locations
        refinedCorner = cv2.cornerSubPix(grayImg, rawCorners, (11, 11), (-1, -1), criteria)

        imgps.append(refinedCorner)
        objps.append(objp)
        grayImgs.append(grayImg)
        refinedCorners.append(refinedCorner)

    # Compute parameters from the original picture
    ret, inParam, distMat, rotVecs, tranVecs = cv2.calibrateCamera(objps, imgps, grayImgs[0].shape[::-1], None, None)
    ExParams = []

    for k in range(num):

        pic = pics[k]
        rotVec = rotVecs[k]
        tranVec = tranVecs[k]
        rotMat, _ = cv2.Rodrigues(rotVec)
        refInd = refInds[k]
        refLazerInd = refLazerInds[k]
        grayImg = grayImgs[k]
        refinedCorner = refinedCorners[k]
        ExParam = concat([rotMat, tranVec], axis=-1)  # shape=(3, 4)
        ExParams.append(ExParam)

        n, m = len(refInd), len(refInd[0])
        _, thr = cv2.threshold(grayImg, 250, 255, cv2.THRESH_BINARY)
        img2, contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        leftmost = np.min(refinedCorner[:, 0, 0])
        rightmost = np.max(refinedCorner[:, 0, 0])
        topmost = np.min(refinedCorner[:, 0, 1])
        bottommost = np.max(refinedCorner[:, 0, 1])
        contours = process(contours, (leftmost, rightmost, topmost, bottommost))

        cv2.imwrite(path+'contours_'+pic, img2)
        # cnt = cv2.drawContours(img2, contours, -1, (0, 0, 0), 1)

        refImage = np.reshape(np.array(sorted(np.array(refinedCorner),
                                              key=lambda point: point[0, 0])), [6, 9, 2])
        refImagePoints = np.array([[refImage[refInd[i][j][0], refInd[i][j][1]] for j in range(m)] for i in range(n)])
        imagePoints = [getExtremePoints(contours[refLazerInd[i][0]], refLazerInd[i][1]) for i in range(n)]
        # print(imagePoints)
        refWorldPoints = (sideLength*np.array(refInd))

        Xw = computeWorldPoints(refImagePoints, refWorldPoints, imagePoints)
        testPoint = imagePoints[0]
        # print(Xw)
        aug_Xw = ut.aug(ut.aug(Xw, -1, 0), -1, 1).T  # (4, 2)
        RTmat = np.concatenate((rotMat, tranVec), axis=-1)  # (3, 4)
        Xc = np.matmul(RTmat, aug_Xw)  # (3, 2)
        Xcs.append(Xc)

        # x1 = np.array([0, 0, 0, 1])
        # x2 = np.array([0, 0.02, 0, 1])
        # print(np.matmul(RTmat, x1))
        # print(np.matmul(RTmat, x2))

        # # Visualization
        # refImage = np.reshape(np.array(refinedCorners), [-1, 2]).tolist()
        # for point in refImage:
        #     point = (int(point[0]), int(point[1]))
        #     cv2.circle(img, point, 5, (255, 0, 0), -1)
        # for contour in contours:
        #     for point in contour:
        #         point = (int(point[0]), int(point[1]))
        #         cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
        # for point in imagePoints:
        #     point = (int(point[0]), int(point[1]))
        #     cv2.circle(img, point, 5, (0, 255, 0), -1)
        # cv2.imwrite(path+'test.bmp', img)

    Xc = np.concatenate(Xcs, axis=-1)

    return Xc, inParam, ExParams, testPoint


if __name__ == '__main__':

    path = 'G:\Code\\tiller-counting\\0410/'
    pics = ['undistImage%i.bmp' %i for i in (2, 5)]
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

        [[[0, 3], [0, 4], [0, 5]],
         [[0, 4], [1, 4], [2, 4]],
         [[1, 4], [1, 5], [1, 6]],
         [[1, 5], [2, 5], [3, 5]],
         [[2, 5], [2, 6], [2, 7]],
         [[4, 6], [4, 7], [4, 8]],
         [[5, 7], [4, 7], [3, 7]],
         [[5, 8], [5, 7], [5, 6]]]
    ]
    refLazerInds = [
        [[0, 'L'], [1, 'R'], [2, 'U'], [2, 'R'], [3, 'U'], [3, 'R'], [4, 'U'], [4, 'R']],
        [[0, 'L'], [0, 'B'], [1, 'L'], [1, 'B'], [2, 'L'], [3, 'R'], [4, 'U'], [7, 'R']],
        [[0, 'L'], [0, 'B'], [1, 'L'], [1, 'B'], [2, 'L'], [3, 'R'], [4, 'U'], [7, 'R']]
    ]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    Xc, inParam, ExParams, testPoint = main(path, pics, (9, 6), criteria, refInds, refLazerInds)

    num = Xc.shape[1]
    Pc = [np.reshape(Xc[:, i], (3, 1)) for i in range(num)]
    minError = np.inf
    bestParam = [0, 0, 0]
    for i in range(30):
        samples = random.sample(range(num), num)
        Pc_1, Pc_2, Pc_3 = (Pc[j] for j in samples[:3])
        Xc = np.concatenate((Pc_1, Pc_2, Pc_3), axis=-1)
        params = np.matmul(np.ones((1, 3)), np.linalg.inv(Xc)).tolist()
        a, b, c = params[0]
        error = 0
        for j in samples[3:]:
            test_Xc = Pc[j]
            test_x = test_Xc[0, 0]
            test_y = test_Xc[1, 0]
            test_z = test_Xc[2, 0]
            error += abs((1-a*test_x-b*test_y)/c-test_z)
        error /= (len(samples)-3)
        if error < minError:
            minError = error
            bestParam = [a, b, c]
    print(num)
    print(minError)
    print(bestParam)
    mat = np.zeros((num, 3))

    with open(path+'points.txt', 'w', encoding='utf8') as f:
        for i in range(num):
            test_Xc = Pc[i]
            mat[i, 0] = test_Xc[0, 0]
            mat[i, 1] = test_Xc[1, 0]
            mat[i, 2] = test_Xc[2, 0]
    savemat('points.mat', {'points': mat})
    planeParam = np.reshape(np.array(bestParam), (1, 3))

    projMats = []
    reProjMats = []
    # for i in range(len(ExParams)):
    # for i in range(0, 1):
    #     ExParam = ExParams[i]
    #     projMat = np.matmul(np.concatenate((inParam, planeParam), axis=0),
    #                   ExParam)
    #     print(projMat)
    #     projMats.append(projMat)
    #     print(np.linalg.inv(projMat))
    #     reProjMats.append(np.linalg.inv(projMat))
    # print(np.matmul(projMats[-1], np.array([0, 20, 0, 1])))
