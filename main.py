import utils as ut
from method import Model
import argparse
import os
import numpy as np
import cv2
import torch


def parser_args():

    parser = argparse.ArgumentParser()

    # Parse args
    parser.add_argument('-input_path', type=str, default='/home/lhchen/nas/tiller_counting/data/0527/')
    parser.add_argument('-output_path', type=str, default='/home/lhchen/nas/tiller_counting/data/res/')
    parser.add_argument('-data_type', type=str, default='jpg')
    parser.add_argument('-net', type=str, default='cnn')
    parser.add_argument('-optim', type=str, default='adagrad')

    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-l2', type=float, default=0.1)
    parser.add_argument('-dr', type=float, default=0.5)
    parser.add_argument('-x', type=float, default=0.0)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-n_epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=10)
    parser.add_argument('-thr', type=int, default=190)
    parser.add_argument('-act_fn', type=str, default='relu')

    parser.add_argument('-train_portion', type=float, default=0.8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-gpu', type=int, default=-1)
    parser.add_argument('-note', type=str, default='')
    parser.add_argument('-mode', type=str, default='train')

    args = parser.parse_args()

    # Check args
    if args.output_path == '':
        args.output_path = args.input_path
    assert args.data_type in ['bmp', 'jpeg', 'jpg', 'png']
    assert args.act_fn in ['relu', 'sigmoid', 'none']
    assert args.net in ['cnn', 'feature']
    assert args.optim in ['adagrad', 'adam', 'sgd']
    assert os.path.isdir(args.input_path) and os.path.isdir(args.output_path)
    if args.gpu >= 0:
        assert torch.cuda.device_count() > args.gpu
    if args.note == '':
        args.note = 'test'
    assert args.mode in ['train', 'resume', 'debug']

    return args


if __name__ == '__main__':

    args = parser_args()

    model = Model(args)

    # for file in raw_files:
    #     img = cv2.imread(args.input_path+file)
    #     print(img.shape)

    model.train()

