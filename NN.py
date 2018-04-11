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


def sigmoid(x):

    return 1/(1+np.exp(-x))


def update(x, w_1, b_1, w_2, y, lr, sign):
    est = np.dot(w_2, sigmoid(np.dot(w_1, x)+b_1))
    temp = sigmoid(np.dot(w_1, x)+b_1)
    error = y-est
    w_2 = w_2 + lr*error*temp.T
    w_1 = w_1 + lr*error*np.dot(w_2.T*(temp*(1-temp)), x.T)
    b_1 = b_1+lr*error*(w_2.T*(temp*(1-temp)))
    loss = (y-est)**2
    if sign:
        print('[ training ] label: %i, pred: %f, loss: %f' % (y, est[0], loss))
        print(np.tanh(np.dot(w_1, x)+b_1))
    return w_1, b_1, w_2


def train(idx_list, n_epoch, lr, emb_list, label_list):

    w_1 = np.zeros([3, emb_list[0].shape[0]])
    b_1 = np.ones([3, 1])
    w_2 = np.zeros([1, 3])

    for epoch in range(n_epoch):
        np.random.shuffle(idx_list)
        if not (epoch+1) % (n_epoch/100):
            sign = True
        else:
            sign = False
        for idx in idx_list:
            w_1, b_1, w_2 = update(emb_list[idx], w_1, b_1, w_2, label_list[idx], lr*(1.5-epoch/n_epoch), sign)

    return w_1, b_1, w_2


def test(idx_list, w_1, b_1, w_2, emb_list, img_list, label_list):

    test_loss = 0
    for idx in idx_list:
        est = np.dot(w_2, np.dot(w_1, emb_list[idx])+b_1)
        test_loss += (est-label_list[idx])**2
        print('%s: estimated tiller number is %i [ %i ]' % (img_list[idx], int(est[0]+0.5), label_list[idx]))
    print('test_loss: %f' % (np.sqrt(test_loss/len(idx_list))))


def main():

    path = 'G:\Graduation_project\Camera\\2017_12_25/'
    n_epoch = 1000
    seed = 17
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

    w_1, b_1, w_2 = train(train_list, n_epoch, lr, emb_list, label_list)
    test(test_list, w_1, b_1, w_2, emb_list, img_list, label_list)


if __name__ == '__main__':

    main()
