import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils as ut
import cv2


class Model():

    def __init__(self, args):

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.input_path = args.input_path
        self.lr = args.lr
        self.momentum = args.momentum
        self.n_epoch = args.n_epoch

        if args.act_fn == 'relu':
            self.act_fn = F.relu
        elif args.act_fn == 'sigmoid':
            self.act_fn = F.sigmoid
        else:  # none
            self.act_fn = lambda x: x

        if args.net == 'cnn':
            self.net = CNN_Net(self.act_fn)
        else:  # feature
            self.net = Feature_Net()

        self.criterion = nn.MSELoss()
        if args.optim == 'adagrad':
            self.optim = torch.optim.Adagrad(self.net.parameters(), self.lr)
        elif args.optim == 'adam':
            self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
        else:  # sgd
            self.optim = torch.optim.SGD(self.net.parameters(), self.lr, self.momentum)

        print(self.net)

    def run(self, train_files, test_files):

        for epoch in range(self.n_epoch):

            train_loss = 0
            test_loss = 0
            np.random.shuffle(train_files)
            np.random.shuffle(test_files)

            for train_file in train_files:

                img = cv2.imread(self.input_path+train_file)
                label = [int(train_file.split('_')[0])]
                x = self.net.prepare(img)
                z = torch.FloatTensor(label)
                z = z.view(1, 1)

                self.optim.zero_grad()
                y = self.net(x)
                loss = self.criterion(y, z)
                loss.backward()
                self.optim.step()

                train_loss += loss.item()

            print('[Train] Epoch %i/%i loss: %f' % ((epoch + 1), self.n_epoch, train_loss/len(train_files)))

            with torch.no_grad():

                for test_file in test_files:

                    img = cv2.imread(self.input_path + test_file)
                    label = [int(test_file.split('_')[0])]
                    x = self.net.prepare(img)
                    z = torch.FloatTensor(label)
                    z = z.view(1, 1)

                    y = self.net(x)
                    loss = self.criterion(y, z)

                    test_loss += loss.item()

                print('[Test] Epoch %i/%i loss: %f' % ((epoch + 1), self.n_epoch, test_loss/len(test_files)))
                print(y)
                print(z)
                print()


class Feature_Net(nn.Module):

    def __init__(self):

        super(Feature_Net, self).__init__()

        self.fc = nn.Linear(4, 1)

    def forward(self, x):

        x = self.fc(x)

        return x

    def prepare(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

        img2, contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        num = len(contours)
        areaList = []
        totalArea = 0
        for i in range(num):
            areaList.append(cv2.contourArea(contours[i]))
            totalArea += areaList[-1]
        meanArea = totalArea / num
        areaList.sort()
        middleArea = areaList[num // 2]
        variance = np.var(areaList)

        # emb: mean_Area, middle_Area, variance, bias
        emb = [meanArea, middleArea, np.sqrt(variance), 1]
        x = torch.FloatTensor(emb)
        x = x.view(1, 4)

        return x


class CNN_Net(nn.Module):

    def __init__(self, act_fn):

        super(CNN_Net, self).__init__()
        self.act_fn = act_fn

        self.conv1 = nn.Conv2d(3, 6, 5)  # (636, 1596)
        self.conv2 = nn.Conv2d(6, 16, 10)  # (150, 390)
        self.pool1 = nn.MaxPool2d(4, 4)  # (159, 399)
        self.pool2 = nn.MaxPool2d((5, 13), (5, 13))  # (30, 30)

        self.fc1 = nn.Linear(16 * 30 * 30, 84)
        self.fc2 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):

        x = self.pool1(self.act_fn(self.conv1(x)))
        x = self.pool2(self.act_fn(self.conv2(x)))
        x = x.view(-1, 16 * 30 * 30)
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.act_fn(self.fc3(x))

        return x

    def prepare(self, img):

        height = img.shape[0]
        weight = img.shape[1]
        in_channels = img.shape[2]

        x = torch.FloatTensor(img)
        x = x.view(1, in_channels, height, weight)

        return x

