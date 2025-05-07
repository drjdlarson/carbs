"""
Implementations for finding Kullback-Leibler distance between Gamma distributions

Algorithm from publication Granström, Karl, and Umut Orguner. "Estimation and Maintenance 
of Measurement Rates for Multiple Extended Target Tracking." Information Fusion (FUSION), 
2012 15th International Conference on. IEEE, 2012.
"""

import numpy as np
import scipy.special

def gamma_KL_diff(alpha:list, beta:list) -> np.ndarray:
    """
    Implementation for more than two components.

    Parameters
    ----------
    alpha : list
        float shapes
    beta : list
        float rates

    Returns
    -------
    ndarray
        2d numpy array distance matrix

    """
    num_comp = len(alpha)

    if num_comp < 2:
        raise RuntimeError("KL Distance not defined for less than two distributions")

    KL_diff_matrix = np.zeros((num_comp,num_comp))

    for i in range(num_comp):
        for j in range(i+1, num_comp):
            a1 = alpha[i]
            a2 = alpha[j]
            b1 = beta[i]
            b2 = beta[j]
            val = single_gamma_KL_diff(a1, a2, b1, b2)
            KL_diff_matrix[i,j] = val
            KL_diff_matrix[j,i] = val
    return KL_diff_matrix

def single_gamma_KL_diff(a1:float, a2:float, b1:float, b2:float) -> float:
    """
    Implementation for two components.

    Parameters
    ----------
    a1 : float
        shape of distribution 1
    a2 : float
        shape of distribution 2
    b1 : float
        rate distribution 1
    b1 : float
        rate of distribution 2

    Returns
    -------
    float
        KL gamma distance between the two distribution
    """
    t1 = (a1 - a2) * (scipy.special.digamma(a1) - scipy.special.digamma(a2) + np.log(b2/b1))
    t2 = (b2 - b1) * (a1/b1 - a2/b2)
    
    return t1 + t2

    
    

