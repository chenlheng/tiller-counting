import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


def parser_args():

    parser = argparse.ArgumentParser()

    # Parse args
    parser.add_argument('-path', type=str, default='/home/lhchen/nas/tiller_counting/data/res/')
    parser.add_argument('-upper', type=int, default=-1)
    parser.add_argument('-lower', type=int, default=-1)
    parser.add_argument('-all', action='store_true', default=False)


    args = parser.parse_args()

    assert args.upper > 0 and args.lower > 0

    return args


if __name__ == '__main__':

    args = parser_args()
    no = args.lower

    while no <= args.upper:

        folder = '%i/' % no

        if os.path.exists(args.path+folder):

            epochs = []
            train_loss = []
            test_loss = []

            with open(args.path+folder+'results.txt', 'r') as f:

                for line in f:
                    if '[Train]' in line:
                        data = line.strip().split(' ')
                        epochs.append(int(data[2].split('/')[0]))
                        train_loss.append(float(data[-1]))

                    elif '[Test]' in line:
                        data = line.strip().split(' ')
                        test_loss.append(float(data[-1]))

            n_epoch = len(epochs)
            if not (n_epoch == len(train_loss) and n_epoch == len(test_loss)):
                print('%i is not complete' % no)
            else:
                epochs = np.array(epochs)
                train_loss = np.array(train_loss)
                test_loss = np.array(test_loss)

                plt.plot(epochs, train_loss, color='g', label='train(%.2f)' % min(train_loss))
                plt.plot(epochs, test_loss, color='r', label='test(%.2f)' % min(test_loss))
                plt.axhline(min(test_loss))
                plt.legend()
                plt.savefig(args.path + '%i.png' % no)

            if not args.all:
                plt.cla()
            no += 1
