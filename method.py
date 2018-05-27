import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils as ut
import cv2


# add batch

class Model():

    def __init__(self, args, sprint, dprint):

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.input_path = args.input_path
        self.lr = args.lr
        self.dr = args.dr
        self.momentum = args.momentum
        self.n_epoch = args.n_epoch
        self.thr = args.thr
        self.gpu = args.gpu
        self.sprint = sprint
        self.dprint = dprint

        if args.act_fn == 'relu':
            self.act_fn = F.relu
        elif args.act_fn == 'sigmoid':
            self.act_fn = F.sigmoid
        else:  # none
            self.act_fn = lambda x: x

        if args.net == 'cnn':
            self.net_ = CNN_Net(self.act_fn, self.dr)
        else:  # feature
            self.net_ = Feature_Net(self.thr)
        self.criterion_ = nn.MSELoss()

        if self.gpu > -1:
            self.net = self.net_.cuda(self.gpu)
            self.criterion = self.criterion_.cuda(self.gpu)
        else:
            self.net = self.net_
            self.criterion = self.criterion_

        if args.optim == 'adagrad':
            self.optim = torch.optim.Adagrad(self.net.parameters(), self.lr)
        elif args.optim == 'adam':
            self.optim = torch.optim.Adam(self.net.parameters(), self.lr)
        else:  # sgd
            self.optim = torch.optim.SGD(self.net.parameters(), self.lr, self.momentum)

        print(self.net)

    def train(self, train_files, test_files):

        for epoch in range(self.n_epoch):

            train_loss = 0
            np.random.shuffle(train_files)
            np.random.shuffle(test_files)

            for i, train_file in enumerate(train_files):

                img = cv2.imread(self.input_path+train_file)
                label = [int(train_file.split('-')[0])]
                x_ = self.net.prepare(img)
                z = torch.FloatTensor(label)
                z_ = z.view(1, 1)
                if self.gpu > -1:
                    x = x_.cuda(self.gpu)
                    z = z_.cuda(self.gpu)
                else:
                    x = x_
                    z = z_

                self.optim.zero_grad()
                y = self.net(x)
                loss = self.criterion(y, z)
                loss.backward()
                self.optim.step()

                train_loss += loss.item()

                self.dprint('[Train] file %i/%i loss: %f' % ((i+1), len(train_files), loss.item()))

            self.sprint('[Train] Epoch %i/%i loss: %f' % ((epoch + 1), self.n_epoch, train_loss/len(train_files)))

            self.test(test_files)

    def test(self, test_files):

        test_loss = 0

        with torch.no_grad():
            for i, test_file in enumerate(test_files):
                img = cv2.imread(self.input_path + test_file)
                label = [int(test_file.split('_')[0])]
                x_ = self.net.prepare(img)
                z = torch.FloatTensor(label)
                z_ = z.view(1, 1)
                if self.gpu > -1:
                    x = x_.cuda(self.gpu)
                    z = z_.cuda(self.gpu)
                else:
                    x = x_
                    z = z_

                y = self.net(x)
                loss = self.criterion(y, z)

                test_loss += loss.item()

                self.dprint('[Train] file %i/%i loss: %f' % ((i + 1), len(test_files), loss.item()))

            self.sprint('[Test] loss: %f' % (test_loss / len(test_files)))
            print('Sample Output:')
            print(y)
            print(z)
            print()


class Feature_Net(nn.Module):

    def __init__(self, thr):

        super(Feature_Net, self).__init__()

        self.fc = nn.Linear(4, 1)
        self.thr = thr

    def forward(self, x):

        x = self.fc(x)

        return x

    def prepare(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(img, self.thr, 255, cv2.THRESH_BINARY)

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

    def __init__(self, act_fn, dr):

        super(CNN_Net, self).__init__()
        self.act_fn = act_fn

        self.conv1 = nn.Conv2d(3, 6, 5)  # (1938, 2586)
        self.conv2 = nn.Conv2d(6, 16, 5)  # (864, 1289)
        self.pool1 = nn.MaxPool2d(2, 2)  # (869, 1293)
        self.pool2 = nn.MaxPool2d((864, 1289))
        self.dropout = nn.Dropout2d(dr)
        self.fc1 = nn.Linear(16, 1)

    def forward(self, x):

        x = self.pool1(self.act_fn(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool2(self.act_fn(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 16)
        x = self.act_fn(self.fc1(x))

        return x

    def prepare(self, img):

        height = img.shape[0]  # 1942
        width = img.shape[1]  # 2590
        in_channels = img.shape[2]  # 3

        x = torch.FloatTensor(img)
        x = x.view(1, in_channels, height, width)

        return x

