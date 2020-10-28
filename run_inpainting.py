import argparse
import os

from srcOld.main import main




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distogram Predictor')

    # General
    parser.add_argument('--seed', default=123558, type=int, metavar='N', help='seed number')
    parser.add_argument('--basefolder', default=os.path.basename(__file__).split(".")[0], type=str, metavar='N', help='Basefolder where results are saved')
    parser.add_argument('--feature-dim', default=1, type=int, metavar='N', help='Input feature types')
    parser.add_argument('--mode', default='standard', type=str, metavar='N', help='Mode to run in (debug,fast,paper)')
    # data
    # parser.add_argument('--dataset-train', default='e:/training30_80_320.lmdb', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-train', default='e:/training_100_80_320.lmdb', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-test', default='e:/testing_80_320.lmdb', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-train', default='e:/testing_small_80_320.lmdb', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-test', default='e:/testing_small_80_320.lmdb', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-train', default='./data/testing.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-test', default='./data/testing.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    parser.add_argument('--dataset-train', default='./data/clean_pnet_train/', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-train', default='./data/clean_pnet_test/', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    parser.add_argument('--dataset-test', default='./data/clean_pnet_test/', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-train', default='./data/lmdb/training_80_320.lmdb', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-test', default='./data/lmdb/testing_80_320.lmdb', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-train', default='./data/testing_small.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    # parser.add_argument('--dataset-test', default='./data/testing_small.pnet', type=str, metavar='N', help='Name of dataset to run, currently implemented: ')
    parser.add_argument('--batch-size', default=10, type=int, metavar='N', help='batch size used in dataloader')
    parser.add_argument('--network', default='vnet', type=str, metavar='N', help='network to use')

    # Learning

    parser.add_argument('--sigma', default=-1, type=float, metavar='N', help='exponential_rate')
    parser.add_argument('--SL-lr', default=1e-3, type=float, metavar='N', help='Learning Rate')
    # parser.add_argument('--SL-network', default='unet', type=str, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--max-iter', default=2000, type=int, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--report-iter', default=50, type=int, metavar='N', help='select the neural network to train (resnet)')
    parser.add_argument('--draw-seq-from-msa', default=False, type=bool, help='Draws the sequence from the pssm matrix, using it for data augmentation')
    # parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='select the neural network to train (resnet)')
    # parser.add_argument('--load-state', default='./results/checkpoints/790000_checkpoint.tar', type=str, metavar='N', help='select the neural network to train (resnet)')
    # parser.add_argument('--load-state', default='', type=str, metavar='N', help='select the neural network to train (resnet)')


    # import numpy as np
    # file_out = './data/ID_103.npz'
    # file_in = './data/ID_103.npz'
    # x = np.arange(10)
    # y = np.ones((3,3))
    # # np.savez(file_out,x=x,y=y)
    # data = np.load(file_in,allow_pickle=True)
    # data.files

    args = parser.parse_args()
    if args.network.lower() == 'transformer':
        args.network_args = {
        'chan_in': 25,  # the number of channels in (21 for one-hot, 22 for one-hot + entropy, 41 for one-hot + pssm, 42 for one-hot + pssm + entropy)
        'emsize': 128,  # embedding dimension
        'nhid': 256, # nhid = 1024  # the dimension of the feedforward network model in nn.TransformerEncoder
        'nlayers': 2,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        'nhead': 8,  # the number of heads in the multiheadattention models
        'dropout': 1e-2,  # 0.2 # the dropout value
        'chan_out': 3,  # the output channels, need 3 for each atom type
        'stencil': 5}
    elif args.network.lower() == 'vnet':

        args.network_args = {
        'chan_in': 25,
        'nblocks': 4,
        'nlayers_pr_block': 5,
        'channels': 256,
        'chan_out': 3
        }
    elif args.network.lower() == 'graph':
        args.network_args = {
        'chan_in': 25,
        'nblocks': 4,
        'nlayers_pr_block': 5,
        'channels': 256,
        'chan_out': 3
        }
    else:
        raise UserWarning("network: {:} not recognised for arg.network_args".format(args.network))


    losses = main(args)



