import argparse
import os
from src.main import main

parser = argparse.ArgumentParser(description='A-Optimal Active Learning')

# General
parser.add_argument('--seed', default=123558, type=int, metavar='N', help='seed number')
parser.add_argument('--basefolder', default=os.path.basename(__file__).split(".")[0], type=str, metavar='N', help='Basefolder where results are saved')
parser.add_argument('--mode', default='epitope', type=str, metavar='N', help='Mode to run in (debug,fast,paper)')
# Data
parser.add_argument('--dataset-train', default='./data/pnet/train.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: "bcell","synthetic"')
# parser.add_argument('--dataset-test', default='./data/cameo.csv', type=str, metavar='N', help='Name of dataset to run, currently implemented: "bcell","synthetic"')
parser.add_argument('--nsamples', default=10000, type=int, metavar='N', help='Number of datasamples, only used with synthetic now')
parser.add_argument('--seq-len', default=512, type=int, metavar='N', help='Length each sequence will be extended/cropped to')
parser.add_argument('--batch-size', default=50, type=int, metavar='N', help='batch size used in dataloader')
# Learning

parser.add_argument('--nlayers', default=10, type=int, metavar='N', help='Number of residual layers in network')
parser.add_argument('--SL-lr', default=5e-5, type=float, metavar='N', help='Learning Rate')
# parser.add_argument('--SL-network', default='unet', type=str, metavar='N', help='select the neural network to train (resnet)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='select the neural network to train (resnet)')
# parser.add_argument('--load-state', default='./results/checkpoints/790000_checkpoint.tar', type=str, metavar='N', help='select the neural network to train (resnet)')
parser.add_argument('--load-state', default='', type=str, metavar='N', help='select the neural network to train (resnet)')


args = parser.parse_args()
losses = main(args)
