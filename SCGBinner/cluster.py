import hnswlib
import leidenalg
import numpy as np
import pandas as pd
import functools
import time
import os
import scipy.sparse as sp
import logging

from igraph import Graph
from sklearn.preprocessing import normalize
from sklearn.cluster._kmeans import euclidean_distances, stable_cumsum, KMeans, check_random_state, row_norms, MiniBatchKMeans

from utils import get_length, calculateN50, save_result
from scripts.gen_bins_from_tsv import gen_bins as gen_bins_from_tsv
from typing import List, Optional, Union
from multiprocessing.sharedctypes import RawArray
from ctypes import c_double
from ctypes import c_float, c_int32

logger = logging.getLogger('SCGBinner')

logger.setLevel(logging.INFO)

# logging
formatter = logging.Formatter('%(asctime)s - %(message)s')

console_hdr = logging.StreamHandler()
console_hdr.setFormatter(formatter)

logger.addHandler(console_hdr)


def fit_hnsw_index(logger, features, ef: int = 100, M: int = 16,
                   space: str = 'l2', save_index_file: bool = False) -> hnswlib.Index:
    """
    Fit an HNSW index with the given features using the HNSWlib library; Convenience function to create HNSW graph.

    :param logger: The logger object for logging messages.
    :param features: A list of lists containing the embeddings.
    :param ef: The ef parameter to tune the HNSW algorithm (default: 100).
    :param M: The M parameter to tune the HNSW algorithm (default: 16).
    :param space: The space in which the index operates (default: 'l2').
    :param save_index_file: The path to save the HNSW index file (optional).

    :return: The HNSW index created using the given features.

    This function fits an HNSW index to the provided features, allowing efficient similarity search in high-dimensional spaces.
    """

    time_start = time.time()
    num_elements = len(features)
    labels_index = np.arange(num_elements)
    EMBEDDING_SIZE = len(features[0])

    # Declaring index
    # possible space options are l2, cosine or ip
    p = hnswlib.Index(space=space, dim=EMBEDDING_SIZE)

    # Initing index - the maximum number of elements should be known
    p.init_index(max_elements=num_elements, ef_construction=ef, M=M)

    # Element insertion
    int_labels = p.add_items(features, labels_index)

    # Controlling the recall by setting ef
    # ef should always be > k
    p.set_ef(ef)

    # If you want to save the graph to a file
    if save_index_file:
        p.save_index(save_index_file)
    time_end = time.time()
    logger.info('Time cost:\t' +str(time_end - time_start) + "s")
    return p



def gen_seed_idx(seedURL: str, contig_id_list: List[str]) -> List[int]:
    """
    Generate a list of indices corresponding to seed contig IDs from a given URL.

    :param seedURL: The URL or path to the file containing seed contig names.
    :param contig_id_list: List of all contig IDs to match with the seed contig names.
    :return: List[int]
    """
    seed_list = []
    with open(seedURL) as f:
        for line in f:
            if line.rstrip('\n') in contig_id_list:
                seed_list.append(line.rstrip('\n'))
    name_map = dict(zip(contig_id_list, range(len(contig_id_list))))
    seed_idx = [name_map[seed_name] for seed_name in seed_list]
    return seed_idx


# change from sklearn.cluster.kmeans
def partial_seed_init(X, n_clusters: int, random_state, seed_idx, n_local_trials=None) -> np.ndarray:
    """
    Partial initialization of KMeans centers with seeds from seed_idx.

    Parameters:
    :param X: Features.
    :param n_clusters: The number of clusters.
    :param random_state: Determines random number generation for centroid initialization. Use an int for reproducibility.
    :param seed_idx: Indices of seed points for initialization.
    :param n_local_trials: The number of local seeding trials. Default is None.

    Returns:
    :return centers (ndarray): The initialized cluster centers.

    This function initializes a KMeans clustering by partially seeding the centers with provided seeds.
    It is a modification of the KMeans initialization algorithm.
    """
    random_state = check_random_state(random_state)
    x_squared_norms = row_norms(X, squared=True)

    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly

    center_id = seed_idx[0]

    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)

    for c, center_id in enumerate(seed_idx[1:], 1):
        if sp.issparse(X):
            centers[c] = X[center_id].toarray()
        else:
            centers[c] = X[center_id]
        closest_dist_sq = np.minimum(closest_dist_sq,
                                     euclidean_distances(
                                         centers[c, np.newaxis], X, Y_norm_squared=x_squared_norms,
                                         squared=True))
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(len(seed_idx), n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers


def run_leiden(output_file: str, namelist: List[str],
               length_weight: List[float], max_edges: int,
               bandwidth: float = 0.1, lmode: str = 'l2', initial_list: Optional[List[Union[int, None]]] = None,
               is_membership_fixed: Optional[bool] = None, resolution_parameter: float = 1.0,
               partgraph_ratio: int = 50):
    """
    Run Leiden community detection algorithm and save the results.

    :param output_file: The path to the output file.
    :param namelist: A list of contig names.
    :param length_weight: List of length weights.
    :param max_edges: Maximum number of edges.
    :param norm_embeddings: Array of normalized embeddings.
    :param bandwidth: Bandwidth parameter (default: 0.1).
    :param lmode: Distance mode ('l1' or 'l2', default: 'l2').
    :param initial_list: Initial membership list (default: None).
    :param is_membership_fixed: Whether membership is fixed (default: None).
    :param resolution_parameter: Resolution parameter (default: 1.0).
    :param partgraph_ratio: Partition graph ratio (default: 50).

    :return: None
    """
    #
    norm_embeddings = _emb_array
    ann_neighbor_indices = _ann_idx_array
    ann_distances = _ann_dist_array

    n_nodes = len(norm_embeddings)
    chunk_size = 50000  #
    edges = []
    weights = []

    #
    dist_cutoff = np.percentile(ann_distances[:, 1:max_edges+1], partgraph_ratio)

    #
    for chunk_start in range(0, n_nodes, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_nodes)
        chunk_slice = slice(chunk_start, chunk_end)

        #
        neighbors = ann_neighbor_indices[chunk_slice, 1:max_edges + 1]
        dists = ann_distances[chunk_slice, 1:max_edges + 1]

        #
        mask = dists <= dist_cutoff
        local_rows, cols = np.where(mask)  #

        #
        global_sources = local_rows + chunk_start

        #
        targets = neighbors[local_rows, cols]
        filtered_dists = dists[local_rows, cols]

        #
        if lmode == 'l1':
            w = np.exp(-np.sqrt(filtered_dists) / bandwidth)
        else:
            w = np.exp(-filtered_dists / bandwidth)

        #
        valid = global_sources < targets
        edges.extend(zip(global_sources[valid], targets[valid]))
        weights.extend(w[valid])

    g = Graph(n_nodes, edges)

    res = leidenalg.RBERVertexPartition(g,
                                        weights=weights, initial_membership = initial_list,
                                        resolution_parameter = resolution_parameter,node_sizes=length_weight)

    optimiser = leidenalg.Optimiser()
    optimiser.optimise_partition(res, is_membership_fixed=is_membership_fixed,n_iterations=-1)

    part = list(res)


    contig_labels_dict ={}
    # dict of communities
    numnode = 0
    rang = []
    for ci in range(len(part)):
        rang.append(ci)
        numnode = numnode+len(part[ci])
        for id in part[ci]:
            contig_labels_dict[namelist[id]] = 'group'+str(ci)

    logger.info(output_file)
    f = open(output_file, 'w')
    for contigIdx in range(len(contig_labels_dict)):
        f.write(namelist[contigIdx] + "\t" + str(contig_labels_dict[namelist[contigIdx]]) + "\n")
    f.close()


def cluster(logger, args, prefix=None):
    """
    Cluster contigs and save the results.

    :param args: Command-line arguments and settings.
    :param prefix: A prefix for the clustering mode (optional).
    :return: None
    """
    logger.info("Start clustering.")

    emb_file = args.emb_file
    output_path = args.output_path
    seed_file = output_path + '/fasta_quarter.SEED'
    contig_file = args.contig_file
    contig_len = args.contig_len

    output_path = output_path + '/cluster_res/'

    embHeader = pd.read_csv(emb_file, sep='\t', nrows=1)
    embMat = pd.read_csv(emb_file, sep='\t', usecols=range(1, embHeader.shape[1])).values
    namelist = pd.read_csv(emb_file, sep='\t', usecols=range(1)).values[:, 0]

    lengths = get_length(contig_file)
    length_weight = []
    for seq_id in namelist:
        length_weight.append(lengths[seq_id])

    N50 = calculateN50(length_weight)
    logger.info('N50:\t' + str(N50))

    length_weight = []

    for seq_id in namelist:
        length_weight.append(lengths[seq_id])

    valid_idx = np.array(length_weight) >= contig_len
    embMat = embMat[valid_idx]
    namelist = namelist[valid_idx]
    length_weight = list(np.array(length_weight)[np.array(length_weight) >= contig_len])

    if args.not_l2normaize:
        norm_embeddings = embMat
    else:
        norm_embeddings = normalize(embMat)

    # #### method1
    os.makedirs(output_path, exist_ok=True)

    import multiprocessing
    num_workers = args.num_threads

    ##### #########  hnswlib_method
    parameter_list = [1, 5,10,30,50,70, 90, 110]
    bandwidth_list = [0.05, 0.1,0.15, 0.2,0.3]
    partgraph_ratio_list =[50,100,80]
    max_edges_list = [100]
    for max_edges in max_edges_list:
        p = fit_hnsw_index(logger, norm_embeddings, ef=max_edges * 10)
        seed_bacar_marker_idx = gen_seed_idx(seed_file, contig_id_list=namelist)
        initial_list = list(np.arange(len(namelist)))
        is_membership_fixed = [i in seed_bacar_marker_idx for i in initial_list]

        time_start = time.time()
        ann_neighbor_indices, ann_distances = p.knn_query(norm_embeddings, max_edges+1)

        #
        shared_emb = RawArray(c_float, norm_embeddings.size)
        shared_ann_idx = RawArray(c_int32, ann_neighbor_indices.size)
        shared_ann_dist = RawArray(c_float, ann_distances.size)

        #
        np.copyto(np.frombuffer(shared_emb, dtype=c_float).reshape(norm_embeddings.shape), norm_embeddings)
        np.copyto(np.frombuffer(shared_ann_idx, dtype=c_int32).reshape(ann_neighbor_indices.shape),
                  ann_neighbor_indices)
        np.copyto(np.frombuffer(shared_ann_dist, dtype=c_float).reshape(ann_distances.shape), ann_distances)

        #ann_distances is l2 distance's square
        time_end = time.time()
        logger.info('knn query time cost:\t' +str(time_end - time_start) + "s")

        with multiprocessing.Pool(num_workers, initializer=init_worker,
                                  initargs=(shared_emb, norm_embeddings.shape, c_float,
                                            shared_ann_idx, ann_neighbor_indices.shape, c_int32,
                                            shared_ann_dist, ann_distances.shape, c_float)) as pool:
            for partgraph_ratio in partgraph_ratio_list:
                for bandwidth in bandwidth_list:
                    for para in parameter_list:
                        output_file = output_path + 'Leiden_bandwidth_' + str(
                            bandwidth) + '_res_maxedges' + str(max_edges) + 'respara_'+str(para)+'_partgraph_ratio_'+str(partgraph_ratio)+'.tsv'

                        if not (os.path.exists(output_file)):
                            pool.apply_async(
                                run_leiden,
                                (output_file, namelist, length_weight, max_edges, bandwidth, 'l2', initial_list,
                                 is_membership_fixed, para, partgraph_ratio)
                            )
            pool.close()
            pool.join()
        logger.info('multiprocess Done')


def init_worker(shared_emb, emb_shape, emb_type,
                shared_ann_idx, ann_idx_shape, ann_idx_type,
                shared_ann_dist, ann_dist_shape, ann_dist_type):
    global _emb_array, _emb_shape
    global _ann_idx_array, _ann_idx_shape
    global _ann_dist_array, _ann_dist_shape

    _emb_shape = emb_shape
    _ann_idx_shape = ann_idx_shape
    _ann_dist_shape = ann_dist_shape

    #
    if emb_type == c_float:
        _emb_array = np.frombuffer(shared_emb, dtype=np.float32).reshape(emb_shape)
    _ann_idx_array = np.frombuffer(shared_ann_idx, dtype=np.int32).reshape(ann_idx_shape)
    _ann_dist_array = np.frombuffer(shared_ann_dist, dtype=np.float32).reshape(ann_dist_shape)


