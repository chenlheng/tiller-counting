import cv2
import numpy as np
import os


def estimate(path, img_file, sign):
    img = cv2.imread(path+img_file, 0)
    if sign:
        cv2.imshow('image', img)

    _, thr = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    if sign:
        cv2.imshow('thr', thr)

    img2, contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = cv2.drawContours(img2, contours, -1, (0, 0, 0), 1)
    if sign:
        cv2.imshow('cnt', cnt)

    num = len(contours)
    areaList = []
    totalArea = 0
    for i in range(num):
        areaList.append(cv2.contourArea(contours[i]))
        totalArea += areaList[-1]

    areaList.sort()
    areaList = np.array(areaList)
    mArea = areaList[num//2]
    res = totalArea/mArea

    print('img_file: %s' % img_file)
    print('Total area is %.2f, and mean area is %.2f' % (totalArea, mArea))
    print('Number of contours: %i' % num)
    print('Estimated number of tillers: %i' % int(res))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


sign = False
path = 'raw_data/15/'
img_list = os.listdir(path)

if __name__ == '__main__':
    for img_file in img_list:
        estimate(path, img_file, sign)