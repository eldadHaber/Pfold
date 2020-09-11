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
    # parser.add_argument('--dataset-test', default='e:/testing_80_320.lmdb', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    parser.add_argument('--dataset-train', default='./data/testing_small.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    parser.add_argument('--dataset-test', default='./data/testing_small.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    parser.add_argument('--seq-len', default=512, type=int, metavar='N', help='Length each sequence will be extended/cropped to')
    parser.add_argument('--batch-size', default=1, type=int, metavar='N', help='batch size used in dataloader')
    parser.add_argument('--network', default='transformer', type=str, metavar='N', help='network to use')

    # Learning

    parser.add_argument('--SL-lr', default=1e-4, type=float, metavar='N', help='Learning Rate')
    # parser.add_argument('--SL-network', default='unet', type=str, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--max-iter', default=500, type=int, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--report-iter', default=1, type=int, metavar='N', help='select the neural network to train (resnet)')
    # parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='select the neural network to train (resnet)')
    # parser.add_argument('--load-state', default='./results/checkpoints/790000_checkpoint.tar', type=str, metavar='N', help='select the neural network to train (resnet)')
    # parser.add_argument('--load-state', default='', type=str, metavar='N', help='select the neural network to train (resnet)')


    args = parser.parse_args()
    if args.network.lower() == 'transformer':
        args.network_args = {
        'chan_in': 42,  # the size of vocabulary
        'emsize': 256,  # embedding dimension
        'nhid': 512, # nhid = 1024  # the dimension of the feedforward network model in nn.TransformerEncoder
        'nlayers': 2,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        'nhead': 8,  # the number of heads in the multiheadattention models
        'dropout': 1e-6,  # 0.2 # the dropout value
        'chan_out': 9,  # the output channels
        'stencil': 5}
    elif args.network.lower() == 'vnet':
        args.network_args = {
        'arch': [[42, 64, 1, 5], [64, 64, 5, 5], [64, 128, 1, 5], [128, 128, 15, 5], [128, 256, 1, 5]],
        'chan_out': 3
        }
    else:
        raise UserWarning("network: {:} not recognised for arg.network_args".format(args.network))


    losses = main(args)
