import cv2
import numpy as np
import os


def analyze(path, img_file, show):
    img = cv2.imread(path + img_file, 0)
    if show:
        cv2.imshow('image', img)

    _, thr = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    if show:
        cv2.imshow('thr', thr)

    img2, contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = cv2.drawContours(img2, contours, -1, (0, 0, 0), 1)
    if show:
        cv2.imshow('cnt', cnt)

    num = len(contours)
    areaList = []
    totalArea = 0
    for i in range(num):
        areaList.append(cv2.contourArea(contours[i]))
        totalArea += areaList[-1]
    meanArea = totalArea/num
    areaList.sort()
    middleArea = areaList[num//2]
    variance = np.var(areaList)

    # emb: mean_Area, middle_Area, variance, bias
    emb = [meanArea, middleArea, np.sqrt(variance), 1]

    return np.reshape(np.array(emb), [-1, 1])


def update(x, w_in, y, lr, sign):
    est = np.dot(w_in, x)
    w_in = w_in+lr*(y-est)*x.T
    loss = (y-est)**2
    if sign:
        print('[ training ] label: %i, pred: %f, loss: %f' % (y, est[0], loss))
    return w_in


def train(idx_list, n_epoch, lr, emb_list, label_list):

    w_in = np.zeros([1, emb_list[0].shape[0]])

    for epoch in range(n_epoch):
        np.random.shuffle(idx_list)
        if not (epoch+1) % (n_epoch/100):
            sign = True
        else:
            sign = False
        for idx in idx_list:
            w_in = update(emb_list[idx], w_in, label_list[idx], lr*(1.5-epoch/n_epoch), sign)

    return w_in


def test(idx_list, w_in, emb_list, img_list, label_list):

    test_loss = 0
    for idx in idx_list:
        est = np.dot(w_in, emb_list[idx])
        test_loss += (est-label_list[idx])**2
        print('%s: estimated tiller number is %i [ %i ]' % (img_list[idx], int(est[0]+0.5), label_list[idx]))
    print('test_loss: %f' % (np.sqrt(test_loss/len(idx_list))))


def main():

    path = 'G:\Graduation_project\Camera\\2017_12_25/'
    n_epoch = 1000
    seed = 7
    lr = 0.000005

    np.random.seed(seed)
    img_list = os.listdir(path)
    label_list = []
    for img_name in img_list:
        label_list.append(int(img_name.split('_')[0]))
    emb_list = [analyze(path, img_list[idx], False) for idx in range(len(img_list))]

    all_list = list(range(len(img_list)))
    test_list = np.random.randint(0, len(img_list), 4)
    train_list = []
    for idx in range(len(img_list)):
        if idx not in test_list:
            train_list.append(idx)

    w_in = train(train_list, n_epoch, lr, emb_list, label_list)
    test(test_list, w_in, emb_list, img_list, label_list)


if __name__ == '__main__':

    main()
