import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import csv


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

    with open(args.path + 'res.csv', 'w', newline='') as f:
        pass
    csv_file = open(args.path + 'res.csv', 'a', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['file', 'epoch', 'train_loss', 'test_loss'])

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

                for i in range(len(epochs)):
                    csv_writer.writerow([no, epochs[i], train_loss[i], test_loss[i]])

                epochs = np.array(epochs)
                train_loss = np.array(train_loss)
                test_loss = np.array(test_loss)

                plt.plot(epochs, train_loss, color='g', label='%i-train(%.2f)' % (no, min(train_loss)))
                plt.plot(epochs, test_loss, color='r', label='%i-test(%.2f)' % (no, min(test_loss)))
                plt.axhline(min(test_loss))
                plt.legend()
                plt.savefig(args.path + '%i.png' % no)

            if not args.all:
                plt.cla()
            no += 1
