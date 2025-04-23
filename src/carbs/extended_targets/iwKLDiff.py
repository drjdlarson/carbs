"""
Implementations for finding Kullback-Leibler distance between 
inverse-wishart distributions

Algorithm from publication Granström, Karl, and Umut Orguner. 
"On the reduction of Gaussian inverse  Wishart mixtures." 
Information Fusion (FUSION), 2012 15th International 
Conference on. IEEE, 2012.
"""

import numpy as np
import scipy
import scipy.special

def iw_KL_diff(dof:list, scale:list) -> np.ndarray:
    """
    Implementation for more than two components.

    Parameters
    ----------
    dof : list
        float degrees of freedom
    scale : list
        dxd numpy arrays of each scale matrix

    Returns
    -------
    ndarray
        2d numpy array distance matrix

    """
    num_comp = len(dof)

    if num_comp < 2:
        raise RuntimeError("KL Distance not defined for less than two distributions")

    KL_diff_matrix = np.zeros((num_comp,num_comp))

    for i in range(num_comp):
        for j in range(i+1, num_comp):
            dof1 = dof[i]
            dof2 = dof[j]
            scale1 = scale[i]
            scale2 = scale[j]
            val = single_iw_KL_diff(dof1, dof2, scale1, scale2)
            KL_diff_matrix[i,j] = val
            KL_diff_matrix[j,i] = val
    return KL_diff_matrix

def single_iw_KL_diff(dof1:float, dof2:float, scale1:np.ndarray, 
                      scale2:np.ndarray) -> float:
    """
    Implementation for two components.

    Parameters
    ----------
    dof1 : float
        degree of fredom of distribution 1
    dof2 : float
        degree of fredom of distribution 2
    scale1 : ndarray
        dxd numpy arrays scale matrix of distribution 1
    scale1 : ndarray
        dxd numpy arrays scale matrix of distribution 2

    Returns
    -------
    float
        KL inverse-wishart distance between the two distribution
    """
    d = scale1.shape[0]
    d_terms = np.array(range(1,d+1),dtype='float')
    d_terms *= 0.5
    v1_D = np.repeat(((dof1 - float(d))/2),d) - d_terms
    v2_D = np.repeat(((dof2 - float(d))/2),d) - d_terms

    temp = (float(dof1 - d - 1) * np.linalg.inv(scale1) - float(dof2 - d - 1) *\
             np.linalg.inv(scale2)) @ (scale2 - scale1)
    t1 = 0.5 * np.trace(temp)

    t2 = 0.5 * (dof2 - dof1) * (np.linalg.slogdet(scale1)[1] - \
                                np.sum(scipy.special.digamma(v1_D)) - \
                                    np.linalg.slogdet(scale2)[1] + \
                                        np.sum(scipy.special.digamma(v2_D)))
    
    return t1 + t2

    
    

