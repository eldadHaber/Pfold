import hnswlib
import numpy as np
import sys
import nmslib
import time
import math
from scipy.sparse import csr_matrix

# print(sys.version)
# print("NMSLIB version:", nmslib.__version__)


def ANN_hnsw(x, k=10, euclidian_metric=False, union=True, eff=None,cutoff=False):
    '''
    Calculates the approximate nearest neighbours using the Hierarchical Navigable Small World Graph for fast ANN search. see: https://github.com/nmslib/hnswlib
    :param x: 2D numpy array with the first dimension being different data points, and the second the features of each point.
    :param k: Number of neighbours to compute
    :param euclidian_metric: Determines whether to use cosine angle or euclidian metric. Possible options are: 'l2' (euclidean) or 'cosine'
    :param union: The adjacency matrix will be made symmetrical, this determines whether to include the connections that only go one way or remove them. If union is True, then they are included.
    :param eff: determines how accurate the ANNs are built, see https://github.com/nmslib/hnswlib for details.
    :param cutoff: Includes a cutoff distance, such that any connection which is smaller than the cutoff is removed. If True, the cutoff is automatically calculated, if False, no cutoff is used, if a number, it is used as the cutoff threshold. Note that the cutoff has a safety built in that makes sure each data point has at least one neighbour to minimize the risk of getting a disjointed graph.
    :return: Symmetric adjacency matrix, mean distance of all connections (including the self connections)
    '''
    nsamples = len(x)
    dim = len(x[0])
    # Generating sample data
    data = x
    data_labels = np.arange(nsamples)
    if eff is None:
        eff = nsamples

    # Declaring index
    if euclidian_metric:
        p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip
    else:
        p = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip

    # Initing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=nsamples, ef_construction=eff, M=16)

    # Element insertion (can be called several times):
    p.add_items(data, data_labels)

    # Controlling the recall by setting ef:
    p.set_ef(eff)  # ef should always be > k

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = p.knn_query(data, k=k)

    if cutoff:
        if type(cutoff) is bool: # Automatically determine the threshold
            dd_mean = np.mean(distances)
            dd_var = np.var(distances)
            dd_std = np.sqrt(dd_var)
            threshold = dd_mean+dd_std
        else:
            threshold = cutoff
        useable = distances < threshold
        useable[:,0] = True
    else:
        useable = distances == distances
    w = 1/np.sum(useable,axis=1)
    return w



def ANN_sparse(x_index, x_query, k=10, eff=None,cutoff=False):
    # Set index parameters
    # These are the most important onese
    M = 16
    nsamples = x_index.shape[0]
    if eff is None:
        eff = nsamples


    num_threads = 16
    index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': eff, 'post': 0}
    # Number of neighbors
    # Intitialize the library, specify the space, the type of the vector and add data points
    index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
    index.addDataPointBatch(x_index)
    # Create an index
    start = time.time()
    index.createIndex(index_time_params)
    end = time.time()
    # print('Index-time parameters', index_time_params)
    # print('Indexing time = %f' % (end - start))
    # Setting query-time parameters

    query_time_params = {'efSearch': eff}
    # print('Setting query-time parameters', query_time_params)
    index.setQueryTimeParams(query_time_params)

    #Querying
    # if x_query
    nbrs = index.knnQueryBatch(x_query, k=k, num_threads=num_threads)

    try:
        out = np.asarray(nbrs)
    # labels = out[:,0,:]
        distances = out[:,1,:]
    except:
        lengths = [len(i[0]) for i in nbrs]
        length = np.min(lengths)
        distances = np.asarray([i[1][0:length] for i in nbrs])
    # end = time.time()
    # print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
    #       (end - start, float(end - start) / query_qty, num_threads * float(end - start) / query_qty))

    if cutoff:
        if type(cutoff) is bool: # Automatically determine the threshold
            dd_mean = np.mean(distances)
            dd_var = np.var(distances)
            dd_std = np.sqrt(dd_var)
            threshold = dd_mean+dd_std
        else:
            threshold = cutoff
        useable = distances < threshold
        useable[:,0] = True
    else:
        useable = distances == distances
    w = 1/np.sum(useable,axis=1)

    return np.float32(w)