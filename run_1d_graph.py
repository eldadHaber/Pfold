import argparse
import os

from src.main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distogram Predictor')

    # General
    parser.add_argument('--seed', default=123558, type=int, metavar='N', help='seed number')
    parser.add_argument('--basefolder', default=os.path.basename(__file__).split(".")[0], type=str, metavar='N', help='Basefolder where results are saved')
    parser.add_argument('--mode', default='standard', type=str, metavar='N', help='Mode to run in (debug,fast,paper)')
    # data
    parser.add_argument('--dataset-train', default='f:/final_dataset_1d_train/', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    parser.add_argument('--dataset-test', default='f:/final_dataset_1d_validate/', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')

    # Input features
    parser.add_argument('--seq-flip-prop', default=0.5, type=float, metavar='N', help='Input feature types')
    parser.add_argument('--feature-dim', default=1, type=int, metavar='N', help='Input feature types')
    parser.add_argument('--inpainting', default=False, type=bool, metavar='N', help='Input feature types')
    parser.add_argument('--i-seq', default=True, type=bool, metavar='N', help='Input feature types')
    parser.add_argument('--i-pssm', default=True, type=bool, metavar='N', help='Input feature types')
    parser.add_argument('--i-entropy', default=True, type=bool, metavar='N', help='Input feature types')
    parser.add_argument('--i-cov_all', default=False, type=bool, metavar='N', help='Input feature types')
    parser.add_argument('--i-cov', default=True, type=bool, metavar='N', help='Input feature types')
    parser.add_argument('--i-contact', default=True, type=bool, metavar='N', help='Input feature types')

    # Learning
    parser.add_argument('--network', default='graph', type=str, metavar='N', help='network to use')
    parser.add_argument('--batch-size', default=10, type=int, metavar='N', help='batch size used in dataloader')
    parser.add_argument('--SL-lr', default=1e-3, type=float, metavar='N', help='Learning Rate')
    parser.add_argument('--max-iter', default=200000, type=int, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--report-iter', default=1000, type=int, metavar='N', help='select the neural network to train (resnet)')

    args = parser.parse_args()
    if args.network.lower() == 'transformer':
        args.network_args = {
        'emsize': 128,  # embedding dimension
        'nhid': 256, # nhid = 1024  # the dimension of the feedforward network model in nn.TransformerEncoder
        'nlayers': 2,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        'nhead': 8,  # the number of heads in the multiheadattention models
        'dropout': 1e-2,  # 0.2 # the dropout value
        'chan_out': 3,  # the output channels, need 3 for each atom type
        'stencil': 5}
    elif args.network.lower() == 'vnet':

        args.network_args = {
        'nblocks': 4,
        'nlayers_pr_block': 5,
        'channels': 512,
        'chan_out': 3,
        'stencil_size': 3,
        }
    elif args.network.lower() == 'graph':
        args.network_args = {
        'nblocks': 4,
        'nlayers_pr_block': 5,
        'channels': 1024,
        'chan_out': 3,
        'stencil_size': 3,
        }
    else:
        raise UserWarning("network: {:} not recognised for arg.network_args".format(args.network))


    losses = main(args)



