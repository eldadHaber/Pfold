import numpy as np
# import scipy
# import scipy.spatial
import string
# import tensorflow as tf

import random
import torch

import numpy as np
import torch.nn as nn


# Define some functions

def move_tuple_to(args,device,non_blocking=True):
    new_args = ()
    for arg in args:
        new_args += (arg.to(device,non_blocking=non_blocking),)
    return new_args


def fix_seed(seed, include_cuda=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if you are using GPU
    if include_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def determine_network_param(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)




# read A3M and convert letters into
# integers in the 0..20 range
def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename, "r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa


# 1-hot MSA to PSSM
def msa2pssm(msa1hot, w):
    beff = tf.reduce_sum(w)
    f_i = tf.reduce_sum(w[:, None, None] * msa1hot, axis=0) / beff + 1e-9
    h_i = tf.reduce_sum(-f_i * tf.math.log(f_i), axis=1)
    return tf.concat([f_i, h_i[:, None]], axis=1)


# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    with tf.name_scope('reweight'):
        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff  # id_min = len_protein * cutoff
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])  # inner product of onehot
        id_mask = id_mtx > id_min  # mask
        w = 1.0 / tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32), -1)  # Not really sure what this does?
    return w


# shrunk covariance inversion
def fast_dca(msa1hot, weights, penalty=4.5):
    nr = tf.shape(msa1hot)[0]
    nc = tf.shape(msa1hot)[1]
    ns = tf.shape(msa1hot)[2]

    with tf.name_scope('covariance'):
        x = tf.reshape(msa1hot, (nr, nc * ns))
        num_points = tf.reduce_sum(weights) - tf.sqrt(tf.reduce_mean(weights))
        mean = tf.reduce_sum(x * weights[:, None], axis=0, keepdims=True) / num_points
        x = (x - mean) * tf.sqrt(weights[:, None])
        cov = tf.matmul(tf.transpose(x), x) / num_points

    with tf.name_scope('inv_convariance'):
        cov_reg = cov + tf.eye(nc * ns) * penalty / tf.sqrt(tf.reduce_sum(weights))
        inv_cov = tf.linalg.inv(cov_reg)

        x1 = tf.reshape(inv_cov, (nc, ns, nc, ns))
        x2 = tf.transpose(x1, [0, 2, 1, 3])
        features = tf.reshape(x2, (nc, nc, ns * ns))

        x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:, :-1, :, :-1]), (1, 3))) * (1 - tf.eye(nc))
        apc = tf.reduce_sum(x3, 0, keepdims=True) * tf.reduce_sum(x3, 1, keepdims=True) / tf.reduce_sum(x3)
        contacts = (x3 - apc) * (1 - tf.eye(nc))

    return tf.concat([features, contacts[:, :, None]], axis=2)

