import argparse
import os
from typing import Dict, Any

from supervised.config import config
from supervised.main import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distogram Predictor')

    # General
    parser.add_argument('--seed', default=123557, type=int, metavar='N', help='seed number')
    parser.add_argument('--basefolder', default=os.path.basename(__file__).split(".")[0], type=str, metavar='N', help='Basefolder where results are saved')
    parser.add_argument('--mode', default='standard', type=str, metavar='N', help='Mode to run in (debug,fast,paper)')
    parser.add_argument('--viz', default=True, type=bool, metavar='N', help='select the neural network to train (resnet)')
    # data
    # parser.add_argument('--dataset-train', default='./data/train_npz/', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-test', default='./data/test_FM/', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    parser.add_argument('--dataset-train', default='./data/casp11_testing/', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    parser.add_argument('--dataset-test', default='./data/casp11_testing/', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')

    parser.add_argument('--use-loss-coord', default=True, type=bool, metavar='N', help='Input feature types')
    parser.add_argument('--use-loss-reg', default=True, type=bool, metavar='N', help='Input feature types')

    # Input features
    parser.add_argument('--feature-dim', default=1, type=int, metavar='N', help='Input feature types')

    # Learning
    parser.add_argument('--network', default='vnet', type=str, metavar='N', help='network to use')
    parser.add_argument('--batch-size', default=20, type=int, metavar='N', help='batch size used in dataloader')
    parser.add_argument('--SL-lr', default=1e-3, type=float, metavar='N', help='Learning Rate')
    parser.add_argument('--max-iter', default=80000, type=int, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--report-iter', default=2, type=int, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--checkpoint', default=10000, type=int, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--exp_dist_loss', default=-1, type=float, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--load-nn-dists', default='./data/nn-distances.npz', type=str, metavar='N', help='Input feature types')
    parser.add_argument('--load-from-previous', default='', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--load-from-previous', default='C:/Users/Tue/PycharmProjects/Pfold/results/run_1d_vnet/2020-11-19_10_05_23/checkpoint.pt', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')

    args = parser.parse_args()

    args.data_args = {
        'i_seq': True,
        'i_pssm': True,
        'i_entropy': True,
        'i_cov': False,
        'i_contact': False,
        'o_rCa': True,
        'o_rCb': True,
        'o_rN': False,
        'o_dist': True,
        'log_units': -10,
        'flip_protein': 0.5,
        'AA_list': 'ACDEFGHIKLMNPQRSTVWY-'
    }



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
        'nblocks': 2,
        'nlayers_pr_block': 2,
        'channels': 80,
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

    if args.network.lower() == 'vnet':
        pad_modulo = 2*args.network_args['nblocks']
    else:
        pad_modulo = 1
    args.data_args['pad_modulo'] = pad_modulo
    #Now we want to save all the arguments in args to a global config variable that can be imported anywhere.
    config.update(vars(parser.parse_args()))
    config['network_args'] = args.network_args
    config['data_args'] = args.data_args

    losses = main()


