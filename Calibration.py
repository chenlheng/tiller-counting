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

    # Read in imgs in gray
    img = cv2.imread(path + pic)
    Img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Automatically prepare objpoints
    sideLength = 0.02  # ==20mm
    realZ = 0  # TODO what if realZ <> 0?
    objp = np.ones((refSize[0] * refSize[1], 3), np.float32)*realZ
    objp[:, :2] = np.mgrid[0:refSize[0], 0:refSize[1]].T.reshape(-1, 2)*sideLength

    # Try to find out raw_corners
    ret, rawCorners = cv2.findChessboardCorners(Img, refSize, None)
    # shape = (actual_num_column-2, actual_num_row-2)

    if ret:
        # Refine the cornor locations
        refinedCorners = cv2.cornerSubPix(Img, rawCorners, (11, 11), (-1, -1), criteria)
        num = refSize[0]*refSize[1]

        # img = cv2.drawChessboardCorners(img, refSize, corners2, ret)
        # cv2.imwrite(pattern + 'res_' + pic, img)
        # cv2.waitKey(0)

        objps = []
        imgps = []
        objps.append(objp)
        imgps.append(refinedCorners)
        imgp = np.reshape(refinedCorners, (num, 2))
        ret, inParam, distMat, rotVecs, tranVecs = cv2.calibrateCamera(objps, imgps, Img.shape[::-1], None, None)
        rotVec = rotVecs[0]
        tranVec = tranVecs[0]
        rotMat, _ = cv2.Rodrigues(rotVec)

        projMat = dot(ut.aug(inParam, -1, 0),
            concat([ut.aug(rotMat, 0, 0), ut.aug(tranVec, 0)], axis=-1))  # shape=(3, 4)
        augImgp = ut.aug(imgp, -1)  # shape=(n, 3)
        augObjp = ut.aug(objp, -1)  # shape=(n, 4)

        # Test objp2imgp
        projAugImgp = objp2imgp(augObjp, projMat)
        print(projAugImgp[-1])
        print(augImgp[-1])
        error = sum(projAugImgp - augImgp) / len(projAugImgp[:, -1])
        print('Objp -> Imgp: %f' % error[0])

        # Test imgp2objp
        projAugObjp = imgp2objp(augImgp, projMat, realZ)
        print(projAugObjp[-1])
        print(augObjp[-1])
        error = sum(projAugObjp - augObjp) / len(projAugObjp[:, -1])
        print('Imgp -> Objp: %f' % error[0])

        # Compute undistort images
        h, w = img.shape[: 2]
        undistInParam, roi = cv2.getOptimalNewCameraMatrix(inParam, distMat, (w, h), 1, (w, h))
        rawUndistImg = cv2.undistort(Img, inParam, distMat, None, undistInParam)  # shape = (w, h, 3)
        x, y, w, h = roi
        undistImg = rawUndistImg[y:y + h, x:x + w]
        # cv2.imwrite(path+'res_'+pic, undistImg)

        projImgp, _ = cv2.projectPoints(objp, rotVec, tranVec, inParam, distMat)
        projImgp = np.reshape(projImgp, (-1, 2))
        ret, undistImgp = cv2.findChessboardCorners(undistImg, refSize, None)
        undistImgp = np.reshape(undistImgp, (-1, 2))

        # projImgp 根据标定结果，从角点世界坐标计算得到的图像坐标
        # undistImgp 经过消畸后得到的角点的图像坐标
        error1 = cv2.norm(imgp, projImgp, cv2.NORM_L2) / len(projImgp)
        error2 = cv2.norm(undistImgp, projImgp, cv2.NORM_L2) / len(projImgp)
        # error_1 = 0.08706215232785179
        # error_2 = 8.430535341754979
        # 可以看出projectPoints函数得到的图像坐标是原图的

        print(error1)
        print(error2)

        return ret, undistImg

    else:
        return ret, None


if __name__ == '__main__':

    path = 'G:/Shared/Image/'
    pics = ['2.bmp', '7.bmp', '12.bmp']
    refSizes = [(3, 3), (3, 3), (6, 6)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for i in range(2, len(pics)):
        print(pics[i])
        rst, undistImg = undistort(pics[i], refSizes[i], criteria)