import argparse
import os

from srcOld.main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distogram Predictor')

    # General
    parser.add_argument('--seed', default=123558, type=int, metavar='N', help='seed number')
    parser.add_argument('--basefolder', default=os.path.basename(__file__).split(".")[0], type=str, metavar='N', help='Basefolder where results are saved')
    parser.add_argument('--feature-type', default='1D', type=str, metavar='N', help='Input feature types')
    parser.add_argument('--mode', default='standard', type=str, metavar='N', help='Mode to run in (debug,fast,paper)')
    # data
    # parser.add_argument('--dataset-train', default='e:/training30_80_320.lmdb', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-train', default='e:/small_test_lmdb.lmdb', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-train', default='./data/training_100.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-test', default='e:/testing_80_320.lmdb', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-train', default='./data/pnet/train.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-train', default='./data/pnet/synthetic.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-train', default='./data/testing_small.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-test', default='./data/testing_small.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    parser.add_argument('--dataset-train', default='./data/testing.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    parser.add_argument('--dataset-test', default='./data/testing.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-test', default='./data/cameo.csv', type=str, metavar='N', help='Name of dataset to run, currently implemented: "bcell","synthetic"')
    parser.add_argument('--nsamples', default=10000, type=int, metavar='N', help='Number of datasamples, only used with synthetic now')
    parser.add_argument('--seq-len', default=512, type=int, metavar='N', help='Length each sequence will be extended/cropped to')
    parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='batch size used in dataloader')
    # Learning

    parser.add_argument('--nlayers', default=20, type=int, metavar='N', help='Number of residual layers in network')
    parser.add_argument('--SL-lr', default=1e-4, type=float, metavar='N', help='Learning Rate')
    # parser.add_argument('--SL-network', default='unet', type=str, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--max-iter', default=10000, type=int, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--report-iter', default=10, type=int, metavar='N', help='select the neural network to train (resnet)')
    # parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='select the neural network to train (resnet)')
    # parser.add_argument('--load-state', default='./results/checkpoints/790000_checkpoint.tar', type=str, metavar='N', help='select the neural network to train (resnet)')
    # parser.add_argument('--load-state', default='', type=str, metavar='N', help='select the neural network to train (resnet)')


    args = parser.parse_args()
    losses = main(args)
