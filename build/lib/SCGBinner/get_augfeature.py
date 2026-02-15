import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed

def extract_aligned_matrix(file_path, namelist):
    df = pd.read_csv(file_path, sep='\t' if file_path.endswith('.tsv') else ',', index_col=0)
    df.index = df.index.str.split('_aug').str[0]
    return df.reindex(namelist).values


def normalize_matrix(mat, method, sqrt=False):
    if sqrt:
        mat = np.sqrt(mat)
    mat += 1e-5
    if method == 'mean':
        mat = mat / mat.mean(axis=0, keepdims=True)
    elif method == 'minmax':
        mat = (mat - mat.min(axis=0, keepdims=True)) / (mat.max(axis=0, keepdims=True) - mat.min(axis=0, keepdims=True))
    elif method == 'standard':
        mat = (mat - mat.mean(axis=0, keepdims=True)) / mat.std(axis=0, keepdims=True)
    else:
        mat = mat / mat.max(axis=0, keepdims=True)
    return mat

def get_kmer_coverage(data_path: str, n_views: int = 2, kmer_model_path: str = 'empty',
                      device = torch.device('cpu'), nokmer: bool = False, cov_meannormalize: bool = False,
                      cov_minmaxnormalize: bool = False, cov_standardization: bool = False, addvars: bool = False,
                      vars_sqrt: bool = False, kmer_l2_normalize: bool = False, kmerMetric_notl2normalize: bool = False):
    """
    Get features

    :param data_path: The path to the data directory.
    :param n_views: The number of views (default: 2).
    :param kmer_model_path: The path to the k-mer model (default: 'empty').
    :param device: The device for computation (default: 'cpu').
    :param nokmer: Flag to exclude k-mer data (default: False).
    :param cov_meannormalize: Flag to mean normalize coverage data (default: False).
    :param cov_minmaxnormalize: Flag to min-max normalize coverage data (default: False).
    :param cov_standardization: Flag to standardize coverage data (default: False).
    :param addvars: Flag to include additional variables (default: False).
    :param vars_sqrt: Flag to take the square root of variables (default: False).
    :param kmer_l2_normalize: Flag to L2 normalize k-mer data (default: False).
    :param kmerMetric_notl2normalize: Flag to not L2 normalize k-mer metric data (default: False).

    :return: A list of preprocessed data and a list of contig names.
    """
    norm_method = 'none'
    if cov_meannormalize:
        norm_method = 'mean'
    elif cov_minmaxnormalize:
        norm_method = 'minmax'
    elif cov_standardization:
        norm_method = 'standard'

    namelist = pd.read_csv(data_path + 'aug0_datacoverage_mean.tsv', sep='\t', usecols=range(1)).values[:, 0]

    def process_view(view):
        cov_file = f"{data_path}aug{view}_datacoverage_mean.tsv"
        covMat = extract_aligned_matrix(cov_file, namelist)

        varsMat = None
        if addvars:
            vars_file = f"{data_path}aug{view}_datacoverage_var.tsv"
            varsMat = extract_aligned_matrix(vars_file, namelist)

        compositMat = None
        if not nokmer:
            kmer_file = f"{data_path}aug{view}/kmer_4_f0.csv"
            compositMat = extract_aligned_matrix(kmer_file, namelist)

        return covMat, varsMat, compositMat

    results = [process_view(v) for v in range(n_views)]

    covMats = normalize_matrix(np.vstack([r[0] for r in results]), norm_method)

    if addvars:
        varsMats = normalize_matrix(np.vstack([r[1] for r in results]), norm_method, sqrt=vars_sqrt)

    if not nokmer:
        compositMats = np.vstack([r[2] for r in results])
        compositMats = compositMats + 1
        compositMats = compositMats / compositMats.sum(axis=1, keepdims=True)
        if kmer_l2_normalize:
            compositMats = normalize(compositMats)

    X_ts = covMats
    if not nokmer:
        X_ts = np.hstack((X_ts, compositMats))
    if addvars:
        X_ts = np.hstack((varsMats, X_ts))

    return list(torch.split(torch.from_numpy(X_ts).float(), len(namelist))), namelist



def get_ContrastiveLearningDataset(data_path: str, n_views: int = 2, kmer_model_path: str = 'empty',
                                   device=torch.device('cpu'), nokmer: bool = False, cov_meannormalize: bool = False,
                                   cov_minmaxnormalize: bool = False, cov_standardization: bool = False, addvars: bool = False,
                                   vars_sqrt: bool = False, kmer_l2_normalize: bool = False, kmerMetric_notl2normalize: bool = False):
    """
    Get a Contrastive Learning dataset based on input parameters.

    :param data_path: The path to the data.
    :param n_views: The number of views for data (default: 2).
    :param kmer_model_path: The path to the k-mer model (default: 'empty').
    :param device: The device to use for computations (default: 'cpu').
    :param nokmer: Whether to use k-mer features (default: False).
    :param cov_meannormalize: Whether to mean normalize coverage (default: False).
    :param cov_minmaxnormalize: Whether to min-max normalize coverage (default: False).
    :param cov_standardization: Whether to standardize coverage (default: False).
    :param addvars: Whether to add additional variables (default: False).
    :param vars_sqrt: Whether to take the square root of additional variables (default: False).
    :param kmer_l2_normalize: Whether to L2 normalize k-mer features (default: False).
    :param kmerMetric_notl2normalize: Whether not to L2 normalize k-mer features (default: False).

    :return: A tuple containing the dataset and a list of names.
    """
    if not data_path.endswith('/'):
        data_path = data_path + '/'
    dataset, namelist = get_kmer_coverage(data_path, n_views, kmer_model_path, device, nokmer, cov_meannormalize, cov_minmaxnormalize, cov_standardization,addvars,vars_sqrt,kmer_l2_normalize,kmerMetric_notl2normalize)
    return dataset, namelist
