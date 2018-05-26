import utils as ut
from method import Model
import argparse
import os
import numpy as np
import cv2


def parser_args():

    parser = argparse.ArgumentParser()

    # Parse args
    parser.add_argument('-input_path', type=str, default='/Users/lhchen/Downloads/raw_data/all/')
    parser.add_argument('-output_path', type=str, default='')
    parser.add_argument('-data_type', type=str, default='bmp')
    parser.add_argument('-net', type=str, default='cnn')
    parser.add_argument('-optim', type=str, default='adagrad')

    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-n_epoch', type=int, default=100)
    parser.add_argument('-act_fn', type=str, default='relu')

    parser.add_argument('-seed', type=int, default=1)

    args = parser.parse_args()

    # Check args
    if args.output_path == '':
        args.output_path = args.input_path
    assert args.data_type in ['bmp', 'jpeg', 'jpg', 'png']
    assert args.act_fn in ['relu', 'sigmoid', 'none']
    assert args.net in ['cnn', 'feature']
    assert args.optim in ['adagrad', 'adam', 'sgd']
    assert os.path.isdir(args.input_path) and os.path.isdir(args.output_path)

    return args


if __name__ == '__main__':

    args = parser_args()

    model = Model(args)

    raw_files = ut.read_files(args.input_path, args.data_type)
    np.random.shuffle(raw_files)
    n_file = len(raw_files)
    train_files = raw_files[:int(0.66*n_file)]
    test_files = raw_files[int(0.66*n_file):]

    # for file in raw_files:
    #     img = cv2.imread(args.input_path+file)
    #     print(img.shape)

    model.run(train_files, test_files)

