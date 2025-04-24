"""
Implementations for merging GGIW distributions

Algorithm from publication:

[1] Granström, Karl, and Umut Orguner. 
"On the reduction of Gaussian inverse Wishart mixtures." 
Information Fusion (FUSION), 2012 15th International 
Conference on. IEEE, 2012. 

and

[2]  Granström, Karl, and Umut Orguner.
"Estimation and Maintenance of Measurement Rates for Multiple
Extended Target Tracking" Information Fusion (FUSION), 
2012 15th International Conference on. IEEE, 2012.
"""

import numpy as np
import scipy.special
import scipy.optimize
from carbs.extended_targets.GGIW_Serums_Models import GGIW 

def _dof_cost_function(nu:float, w_merged:float, d:int, scalar:float)->float:
    """
    Implementation of part of Eq. (9d) from reference [1]
    """
    d_terms = np.array(range(1,d+1),dtype='float')
    d_terms *= 0.5
    
    t1 = w_merged * d * np.log(nu - d -1)

    v1_D = np.repeat(((nu - float(d))/2),d) - d_terms
    t2 = w_merged * np.sum(scipy.special.digamma(v1_D))

    return t1 - t2 + scalar

def _gamma_cost_function(alpha:float, scalar:float)->float:
    """
    Implementation of part of Eq. (25) from reference [1]
    """
    return np.log(alpha) - scipy.special.digamma(alpha) + scalar


def ggiw_merge(w:list, means:list, covs:list, alphas:list, betas:list, IWdof:list,
                IWshape:list, labels:list = None, opt_nu:bool = True, opt_alpha:bool = True) -> tuple:
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
    IWdof : list
        float degrees of freedom of IW component of GGIW group
    shape : list
        dxd numpy array shape matrix of IW component of GGIW group
    labels : list, optional
        lable of GGIW group (default to None)
    opt_nu : bool, optional
        Set to true to use optimization for merged IW DOF calc. Else IW DOF is the weighted sum (default to True)
    opt_alpha : bool, optional
        Set to true to use optimization for merged Gamma shape calc. Else Gamma shape is the weighted sum (default to True)

    Returns
    -------
    tuple
        intensity, GGIW merged, label(optional)
    """

    num_comp = len(w)
    d = means[0].shape[0]
    shape_d = IWshape[0].shape[0]

    # Merge of Gaussian components
    w_merged = sum(w)
    mean_merged = sum(n*m for n,m in zip(w, means))/ w_merged
    cov_merged = np.zeros((d,d))
    e_lst = [x-mean_merged for x in means]
    cov_merged = sum([n*(c + e@e.transpose()) for n,c,e in zip(w,covs,e_lst)])/w_merged
    
    # Merge IW components
    exp_dof = sum(a*b for a,b in zip(w, IWdof))/w_merged
    inv_scale = [np.linalg.inv(x) for x in IWshape] # Pre-compute matrix inversion

    if opt_nu:
        # Precompute terms for Eq. (9d) from [1] that are not dependent on input params
        t3 = w_merged * shape_d * np.log(w_merged)

        temp = np.zeros((shape_d,shape_d))
        t5 = 0
        d_terms = np.array(range(1,shape_d+1),dtype='float')
        d_terms *= 0.5
        for ii in range(num_comp):
            temp += w[ii] * (IWdof[ii] - shape_d - 1) * inv_scale[ii]
            t5 += w[ii] * \
                sum(scipy.special.digamma(np.repeat(((IWdof[ii] - float(shape_d))/2),shape_d) - d_terms))

        t4 = w_merged * np.linalg.slogdet(temp)[1]

        t6 = sum(n * np.linalg.slogdet(sc)[1] for n,sc in zip(w, IWshape))
        scalar = t3 - t4 + t5 - t6

        solve_res = scipy.optimize.root_scalar(_dof_cost_function, x0=exp_dof, \
                                           args=(w_merged, shape_d, scalar), \
                                           maxiter=1000, method='brenth',\
                                           bracket=[shape_d+2.0, 10000.0])

        if solve_res.converged:
            
            dof_merged = solve_res.root
        else:
            dof_merged = exp_dof
    else:
        dof_merged = exp_dof
    dof_merged = float(round(max(dof_merged, 2 * shape_d + 3)))  # Numerical hack to ensure IW is well defined

    temp = sum(n * (a - shape_d - 1) * inv_sc for n,a,inv_sc in zip(w, IWdof, inv_scale))
    scale_merged = w_merged * (dof_merged - shape_d - 1) * np.linalg.inv(temp)
    
    # Merge Gamma compoenents
    exp_alpha = sum(a*b for a,b in zip(w, alphas))/w_merged
    denom = sum([n * a/b for n,a,b in zip(w, alphas, betas)])/w_merged
    if opt_alpha:
        # Precompute terms for Eq. (25) from [2] that are not dependent on input params
        t3 = (1/w_merged) * sum([n * (scipy.special.digamma(a) - np.log(b)) for n,a,b in zip(w, alphas, betas)])

        t4 = np.log(denom)

        scalar = t3-t4

        solve_res = scipy.optimize.root_scalar(_gamma_cost_function, x0=exp_alpha, args=(scalar), \
                                           maxiter=1000)
        
        if solve_res.converged:
            alpha_merged = solve_res.root
        else:
            alpha_merged = exp_alpha
    else:
        alpha_merged = exp_alpha

    beta_merged = alpha_merged / denom

    # Label inplementation port from MATLAB. Retain label of componenet with highest weight
    # Only return a merged label if a label list was given
    if labels is not None:
        max_w_ind = np.argmax(w)
        return w_merged, GGIW(mean=mean_merged, covariance=cov_merged, alpha=alpha_merged, \
                          beta=beta_merged, IWdof=dof_merged, IWshape=scale_merged), labels[max_w_ind]
    else:
        return w_merged, GGIW(mean=mean_merged, covariance=cov_merged, alpha=alpha_merged, \
                          beta=beta_merged, IWdof=dof_merged, IWshape=scale_merged)
