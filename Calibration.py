import numpy as np
import cv2
from numpy import matmul as dot
from numpy.linalg import inv
import utils as ut
from numpy import concatenate as concat


def objp2imgp(augObjp, projMat):

    rawProjAugImgp = dot(augObjp, projMat.T)
    n = len(rawProjAugImgp[:, -1])
    ratio = n / sum(rawProjAugImgp[:, -1])

    return rawProjAugImgp * ratio


def imgp2objp(augImgp, projMat, realZ):

    newProjMat = np.ones((3, 3))
    newProjMat[:, 2] = projMat[:, 2]*realZ + projMat[:, 3]
    newProjMat[:, :2] = projMat[:, :2]

    rawProjObjp = dot(augImgp, inv(newProjMat.T))
    rawProjAugObjp = np.ones((len(rawProjObjp), 4))
    rawProjAugObjp[:, :2] = rawProjObjp[:, :2]
    rawProjAugObjp[:, 2] = rawProjObjp[:, 2] * np.average((newProjMat[:, 2] - projMat[:, 3]) / projMat[:, 2])
    rawProjAugObjp[:, 3] = rawProjObjp[:, 2]

    n = len(rawProjAugObjp[:, -1])
    ratio = n / sum(rawProjAugObjp[:, -1])

    return rawProjAugObjp * ratio


def undistort(pic, refSize, criteria):

    # Set up constant parameters
    sideLength = 0.02  # ==20mm
    realZ = 0  # TODO what if realZ <> 0?
    num = refSize[0] * refSize[1]

    # Read in imgs in gray
    img = cv2.imread(path + pic)
    Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('test', Img)
    # cv2.waitKey(0)
    # print(Img.shape)

    # Automatically prepare objpoints
    objp = np.ones((refSize[0] * refSize[1], 3), np.float32)*realZ
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
    augImgp = ut.aug(imgp, -1)  # shape=(n, 3)
    augObjp = ut.aug(objp, -1)  # shape=(n, 4)

    # Compute parameters from the original picture
    ret, inParam, distMat, rotVecs, tranVecs = cv2.calibrateCamera(objps, imgps, Img.shape[::-1], None, None)
    rotVec = rotVecs[0]
    tranVec = tranVecs[0]
    rotMat, _ = cv2.Rodrigues(rotVec)

    # Compute the undistorted image
    h, w = Img.shape[: 2]
    undistInParam, roi = cv2.getOptimalNewCameraMatrix(inParam, distMat, (w, h), 1, (w, h))
    rawUndistImg = cv2.undistort(Img, inParam, distMat, None, undistInParam)  # shape = (w, h, 3)
    x, y, w, h = roi
    undistImg = rawUndistImg[y:y + h, x:x + w]
    if w == 0 or h == 0:
        print('Failed(roi)')
        return
    # cv2.imwrite('undist_'+pic, undistImg)

    # Find corners from the undistorted picture
    ret, undistImgp = cv2.findChessboardCorners(undistImg, refSize, None)
    if not ret:
        print('Failed(undist)')
        return
    undistImgp = np.reshape(undistImgp, (-1, 2)) + np.array([y, x])
    undistImgp_float64 = ut.reformArray(undistImgp)

    # Compute projection matrices
    projMat = dot(ut.aug(inParam, -1, 0),
                  concat([ut.aug(rotMat, 0, 0), ut.aug(tranVec, 0)], axis=-1))  # shape=(3, 4)
    newProjMat = dot(ut.aug(undistInParam, -1, 0),
                     concat([ut.aug(rotMat, 0, 0), ut.aug(tranVec, 0)], axis=-1))  # shape=(3, 4)
    print((projMat-newProjMat)/projMat)

    # Compute projected corners
    projImgp, _ = cv2.projectPoints(objp, rotVec, tranVec, inParam, distMat)
    projImgp = np.reshape(projImgp, (-1, 2))

    # Experiment Session
    # Test objp2imgp
    projAugImgp = objp2imgp(augObjp, projMat)
    newProjAugImgp = objp2imgp(augObjp, newProjMat)
    error = cv2.norm(np.reshape(augImgp, [-1, 3])[:, :2],
                     np.reshape(projAugImgp, [-1, 3])[:, :2],
                     cv2.NORM_L2) / num
    # print('Objp -> Imgp: %f' % error)

    # Test imgp2objp
    projAugObjp = imgp2objp(augImgp, projMat, realZ)
    error = cv2.norm(np.reshape(augObjp, [-1, 4])[:, :3],
                     np.reshape(projAugObjp, [-1, 4])[:, :3],
                     cv2.NORM_L2) / num
    # print('Imgp -> Objp: %f' % error)

    print('cv2.projectPoints')
    projImgp = ut.reformArray(projImgp)
    imgp = ut.reformArray(imgp)
    error1 = cv2.norm(imgp, projImgp, cv2.NORM_L2) / num
    error2 = cv2.norm(undistImgp, projImgp, cv2.NORM_L2) / num
    print('Error against imgp: %f' % error1)
    print('Error aginast undistImgp: %f' % error2)
    # error_1 = 0.08706215232785179
    # error_2 = 8.430535341754979
    # 可以看出projectPoints函数得到的图像坐标是原图的

    print('projMat')
    error1 = cv2.norm(
        augImgp[:, :2],
        np.reshape(projAugImgp, [-1, 3])[:, :2],
        cv2.NORM_L2) / num
    error2 = cv2.norm(
        undistImgp_float64,
        np.reshape(projAugImgp, [-1, 3])[:, :2],
        cv2.NORM_L2) / num
    print('Error against imgp: %f' % error1)
    print('Error aginast undistImgp: %f' % error2)

    print('newProjMat')
    error1 = cv2.norm(
        augImgp[:, :2],
        np.reshape(newProjAugImgp, [-1, 3])[:, :2],
        cv2.NORM_L2) / num
    error2 = cv2.norm(
        undistImgp_float64,
        np.reshape(newProjAugImgp, [-1, 3])[:, :2],
        cv2.NORM_L2) / num
    print('Error against imgp: %f' % error1)
    print('Error aginast undistImgp: %f' % error2)


if __name__ == '__main__':

    path = 'G:\Code\\tiller-counting\calib_example/'
    # pics = ['2.bmp', '7.bmp', '12.bmp']
    # refSizes = [(3, 3), (3, 3), (6, 6)]
    pics = ['Image%i.tif' %i for i in range(1, 20+1)]
    # refSizes = [(3, 3), (6, 4), (6, 5), (3, 3), (4, 4), (5, 7),
    #             (4, 3), (4, 3), (6, 3), (3, 4), (6, 7), (6, 6)]
    refSizes = [(12, 11), (12, 12), (12, 11), (12, 12), (12, 11),
                (12, 11), (12, 11), (12, 11), (12, 11), (12, 11),
                (12, 11), (12, 11), (12, 12), (12, 12), (12, 12),
                (12, 12), (13, 12), (12, 11), (12, 11), (12, 11), ]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for i in range(len(pics)):
        print(pics[i])
        # undistort(pics[i], refSizes[i], criteria)
        undistort(pics[i], (13, 14), criteria)
        # 1, 2, 4, 7, 13, 14, 15, 16, 17