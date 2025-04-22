import numpy as np
from carbs.extended_targets.GGIW_Serums_Models import GGIW 

def ggiw_merge(w:list, means:list, covs:list, alphas:list, betas:list, dofs:list, scale:list, labels:list = None) -> GGIW:
    """
    Implementation to merge a group of GGIW components

    Parameters
    ----------
    w : list
        float weight of GGIW group components
    means : list
        dx1 numpy array of means of Gaussian component of GGIW group
    covs : list
        dxd numpy array of covariances of Gaussian component of GGIW group
    alphas : list
        float shape parameters of Gamma component of GGIW group
    betas : list
        float rate parameters of Gamma component of GGIW group
    dofs : list
        float degrees of freedom of IW component of GGIW group
    scale : list
        dxd numpy array scale matrix of IW component of GGIW group
    labels : list, optional
        lable of GGIW group (default to None)
    """

    num_comp = len(w)
    d = means[0].shape[0]

    # Merge of Gaussian components
    w_merged = sum(w)
    mean_merged = np.array(means).reshape((num_comp, d)).transpose() @ np.array(w).reshape((num_comp,1)) / w_merged

    cov_merged = np.zeros((d,d))
    for ii in range(num_comp):
        e = means[ii] - mean_merged
        cov_merged += w[ii] * (covs[ii] + e @ e.transpose())
    cov_merged = cov_merged/w_merged
    
    # Merge Gamma components

    # Merge IW compoenents


    return 1
