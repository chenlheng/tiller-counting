import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils as ut
import cv2
import os
from time import gmtime, strftime


class Model:

    def __init__(self, args):

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.input_path = args.input_path
        self.lr = args.lr
        self.l2 = args.l2
        self.dr = args.dr
        self.momentum = args.momentum
        self.train_portion = args.train_portion
        self.data_type = args.data_type
        self.n_epoch = args.n_epoch
        self.batch_size = args.batch_size
        self.thr = args.thr
        self.gpu = args.gpu
        self.mode = args.mode
        self.net_type = args.net
        self.output = Output(args)
        self.sprint = self.output.sprint
        self.dprint = self.output.dprint
        self.best_loss = np.inf

        files = ut.read_files(self.input_path, self.data_type)
        if self.mode == 'debug':
            files = files[:min(100, len(files))]
        np.random.shuffle(files)
        self.input_path = self.input_path
        self.n_file = len(files)
        self.n_train = int(self.train_portion * self.n_file)
        self.n_test = self.n_file - self.n_train
        self.train_files = files[:self.n_train]
        self.test_files = files[self.n_train:]
        self.train_data = Data(args.input_path, self.train_files, args.batch_size, args.thr)
        self.test_data = Data(args.input_path, self.test_files, args.batch_size, args.thr)
        print('Training set: %i photos\nTest set: %i photos' % (self.n_train, self.n_test))

        if args.act_fn == 'relu':
            self.act_fn = F.relu
        elif args.act_fn == 'sigmoid':
            self.act_fn = F.sigmoid
        else:  # none
            self.act_fn = lambda x: x

        if self.net_type == 'cnn':
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
            self.optim = torch.optim.Adagrad(self.net.parameters(), lr=self.lr, weight_decay=self.l2)
        elif args.optim == 'adam':
            self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.l2)
        else:  # sgd
            self.optim = torch.optim.SGD(self.net.parameters(), lr=self.lr,
                                         momentum=self.momentum, weight_decay=self.l2)

        print(self.net)

    def train(self):

        self.net.train()
        self.run(self.train_data)

    def test(self):

        self.net.eval()
        with torch.no_grad():
            self.run(self.test_data, train=False)

    def run(self, data, train=True):

        if train:
            state = 'Train'
            n_epoch = self.n_epoch
        else:
            state = 'Test'
            n_epoch = 1

        for epoch in range(n_epoch):

            epoch_loss = 0
            data.init_data()
            flag, no, x_, z_ = data.next_batch(self.net_type)
            self.optim.zero_grad()

            while flag:

                if self.gpu > -1:
                    x = x_.cuda(self.gpu)
                    z = z_.cuda(self.gpu)
                else:
                    x = x_
                    z = z_

                y = self.net(x, train)
                loss = self.criterion(y, z)

                if train:
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()

                epoch_loss += np.sqrt(loss.item())
                message = '[%s] file %i/%i loss: %f' % (state, no, data.get_file_num(), np.sqrt(loss.item()))
                self.dprint(message)
                # self.output.write(message+'\n')

                flag, no, x_, z_ = data.next_batch(self.net_type)

            message = '[%s] Epoch %i/%i loss: %f' % (state, (epoch + 1), n_epoch, epoch_loss/no*self.batch_size)
            self.sprint(message)
            self.output.write(message+'\n')

            if train:  # Test
                self.test()
                self.net.train()
            else:
                if self.best_loss > epoch_loss:
                    self.output.save(self.net)
                print('Sample output:')
                print(z)
                print(y)


class Feature_Net(nn.Module):

    def __init__(self, thr):

        super(Feature_Net, self).__init__()

        self.fc = nn.Linear(4, 1)
        self.thr = thr

    def forward(self, x, _):

        x = self.fc(x)

        return x


class CNN_Net(nn.Module):

    def __init__(self, act_fn, dr):

        super(CNN_Net, self).__init__()
        self.act_fn = act_fn

        # Add layers, change pools, remove fc's & add dropout
        self.conv1 = nn.Conv2d(3, 6, 11)  # (1932, 2580)
        self.conv2 = nn.Conv2d(6, 16, 11)  # (956, 1280)
        self.pool1 = nn.MaxPool2d(2, 2)  # (966, 1290)
        self.pool2 = nn.MaxPool2d((956, 1280), (956, 1280))
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout2d(dr)
        self.fc1 = nn.Linear(16, 48)
        self.fc2 = nn.Linear(48, 1)

    def forward(self, x, flag):

        x = self.pool1(self.act_fn(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool2(self.act_fn(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(-1, 16)
        x = self.fc2(F.relu(self.fc1(x)))

        return x


class Data():
    
    def __init__(self, input_path, files, batch_size, thr):

        self.input_path = input_path
        self.files = files
        self.n_file = len(self.files)
        self.batch_size = batch_size
        self.thr = thr
        self.no = 0

    def init_data(self):

        np.random.shuffle(self.files)
        self.no = 0

    def get_file_num(self):

        return self.n_file // self.batch_size * self.batch_size

    def next_batch(self, net):

        if self.no + self.batch_size >= self.n_file:
            return False, self.no, None, None

        data_batch = []
        label_batch = []

        for no in range(self.no, self.no + self.batch_size):

            img = cv2.imread(self.input_path + self.files[no])
            label = [int(self.files[no].split('-')[0])]

            if net == 'cnn':
                x = self.prepare_cnn(img)
            else:
                x = self.prepare_feature(img, self.thr, self.files[no])
            data_batch.append(x)

            z = torch.FloatTensor(label)
            z = z.view(1, 1)
            label_batch.append(z)

        self.no += self.batch_size

        data_batch = torch.cat(data_batch, dim=0)
        label_batch = torch.cat(label_batch, dim=0)

        return True, self.no, data_batch, label_batch

    def prepare_cnn(self, img):

        height = img.shape[0]  # 1942
        width = img.shape[1]  # 2590
        in_channels = img.shape[2]  # 3

        x = torch.FloatTensor(img)
        x = x.view(1, in_channels, height, width)

        return x

    def prepare_feature(self, img, thr, file_name):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)

        img2, raw_contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = []
        for contour in raw_contours:
            if 10 <= contour.shape[0] <= 100:
                contours.append(contour)

        num = len(contours)
        if num == 0:
            x = torch.Tensor([0, 0, 0, 1])
            print('Warning: %s has no contours' % file_name)
        else:
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


class Output():

    def __init__(self, args):

        self.sprint, self.dprint = ut.init_print()

        output_path, note = args.output_path, args.note
        folder = '%s/' % note
        self.output_path = output_path+folder
        self.output_file = self.output_path + 'results.txt'

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        with open(self.output_file, 'w') as _:
            pass

        self.f = open(self.output_file, 'a', 1)
        self.write(strftime("%Y-%m-%d %H:%M:%S\n", gmtime()))
        self.write(str(args) + '\n')

    def write(self, message):

        self.f.write(message)

    def save(self, model):

        torch.save(model, self.output_path+'model')

