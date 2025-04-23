"""
Implementations for merging GGIW distributions

Algorithm from publication Granström, Karl, and Umut Orguner. 
"On the reduction of Gaussian inverse Wishart mixtures." 
Information Fusion (FUSION), 2012 15th International 
Conference on. IEEE, 2012.
"""

import numpy as np
import scipy.special
import scipy.optimize
from carbs.extended_targets.GGIW_Serums_Models import GGIW 

def _dof_cost_function(nu:float, w_merged:float, d:int, scalar:float)->float:
    """
    Implementation of Eq. (9d) from reference
    """
    d_terms = np.array(range(1,d+1),dtype='float')
    d_terms *= 0.5
    
    t1 = w_merged * d * np.log(nu - d -1)

    v1_D = np.repeat(((nu - float(d))/2),d) - d_terms
    t2 = w_merged * np.sum(scipy.special.digamma(v1_D))

    return t1 - t2 + scalar

def ggiw_merge(w:list, means:list, covs:list, alphas:list, betas:list, dofs:list,
                scale:list, labels:list = None, opt_nu:bool = True) -> GGIW:
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
    shape_d = scale[0].shape[0]

    # Merge of Gaussian components
    w_merged = sum(w)
    mean_merged = sum(n*m for n,m in zip(w, means))/ w_merged
    cov_merged = np.zeros((d,d))
    e_lst = [x-mean_merged for x in means]
    cov_merged = sum([n*(c + e@e.transpose()) for n,c,e in zip(w,covs,e_lst)])/w_merged
    
    # Merge IW components
    exp_dof = sum(a*b for a,b in zip(w, dofs))
    inv_scale = [np.linalg.inv(x) for x in scale] # Pre-compute matrix inversion

    if opt_nu:
        # Precompute terms for Eq. (9d) that are not dependent on input params
        t3 = w_merged * shape_d * np.log(w_merged)

        temp = np.zeros((shape_d,shape_d))
        t5 = 0
        d_terms = np.array(range(1,shape_d+1),dtype='float')
        d_terms *= 0.5
        for ii in range(num_comp):
            temp += w[ii] * (dofs[ii] - shape_d - 1) * inv_scale[ii]
            t5 += w[ii] * \
                sum(scipy.special.digamma(np.repeat(((dofs[ii] - float(shape_d))/2),shape_d) - d_terms))

        t4 = w_merged * np.linalg.slogdet(temp)[1]

        t6 = sum(n * np.linalg.slogdet(sc)[1] for n,sc in zip(w, scale))
        scalar = t3 - t4 + t5 - t6

        solve_res = scipy.optimize.root_scalar(_dof_cost_function, x0=exp_dof, \
                                           args=(w_merged, shape_d, scalar), \
                                           maxiter=1000, method='bisect',\
                                           bracket=[shape_d+2.0, 10000.0])

        if solve_res.converged:
            dof_merged = solve_res.root
        else:
            dof_merged = exp_dof
    else:
        dof_merged = exp_dof

    temp = sum(n * (a - shape_d - 1) * inv_sc for n,a,inv_sc in zip(w, dofs, inv_scale))
    scale_merged = w_merged * (dof_merged - shape_d - 1) * np.linalg.inv(temp)
    
    # Merge Gamma compoenents

    return 1
