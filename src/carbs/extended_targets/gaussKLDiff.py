"""
Implementations for finding Kullback-Leibler distance between gaussian distributions

Algorithm from publication Granström, Karl, and Umut Orguner. "On the reduction of Gaussian inverse 
Wishart mixtures." Information Fusion (FUSION), 2012 15th International 
Conference on. IEEE, 2012.
"""

import numpy as np

def gauss_KL_diff(means:list, covs:list) -> np.ndarray:
    """
    Implementation for more than two components.

    Parameters
    ----------
    means : list
        dx1 numpy arrays of each mean
    covs : list
        dxd numpy arrays of each covariance

    Returns
    -------
    ndarray
        2d numpy array distance matrix

    """
    num_comp = len(means)

    if num_comp < 2:
        raise RuntimeError("KL Distance not defined for less than two distributions")

    KL_diff_matrix = np.zeros((num_comp,num_comp))

    for i in range(num_comp):
        for j in range(i+1, num_comp):
            mean1 = means[i]
            mean2 = means[j]
            cov1 = covs[i]
            cov2 = covs[j]
            val = single_gauss_KL_diff(mean1, mean2, cov1, cov2)
            KL_diff_matrix[i,j] = val
            KL_diff_matrix[j,i] = val
    return KL_diff_matrix

def single_gauss_KL_diff(mean1:np.ndarray, mean2:np.ndarray, cov1:np.ndarray, cov2:np.ndarray) -> float:
    """
    Implementation for two components.

    Parameters
    ----------
    mean1 : ndarray
        dx1 numpy arrays of mean 1
    mean2 : ndarray
        dx1 numpy arrays of mean 2
    cov1 : ndarray
        dxd numpy arrays of covariance 1
    cov2 : ndarray
        dxd numpy arrays of covariance 2

    Returns
    -------
    float
        KL gaussian distance between the two distribution
    """
    d = mean1.shape[0]
    mean_dif = mean1 - mean2
    inv_cov1 = np.linalg.inv(cov1)
    inv_cov2 = np.linalg.inv(cov2)
    dist = 0.5 * (np.trace(cov2 @ inv_cov1 + cov1 @ inv_cov2) + mean_dif.transpose() @ inv_cov2 @ mean_dif + mean_dif.transpose() @ inv_cov1 @ mean_dif) - d
    return dist

