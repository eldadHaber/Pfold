import numpy as np
from numpy.linalg import norm
import random

def draw_from_probability_matrix(P):
    '''
    Given a probability matrix P, we generate a vector where each element in the vector is drawn randomly according to the probabilities in the probability matrix.
    P: ndarray of shape (l,n), where l length of the protein (in amino acids), and n is the number of possible amino acids in the one hot encoding.
    NOTE it is assumed that each row in the probability matrix P sums to 1.
    '''

    u = np.random.rand(P.shape[0])
    idxs = (P.cumsum(1) < u[:,None]).sum(1)
    return idxs


