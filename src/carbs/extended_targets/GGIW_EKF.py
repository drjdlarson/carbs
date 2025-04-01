import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.integrate as s_integrate
from copy import deepcopy

import gncpy.dynamics.basic as gdyn
import gncpy.math as gmath
import gncpy.filters._filters as cpp_bindings


import scipy.stats as stats
import scipy.special as special
from gncpy.filters import ExtendedKalmanFilter
from carbs.extended_targets.GGIW_Serums_Models import GGIW





""" WHEN WE MOVE GGIW AND GGIW MIXTURE CLASSES TO SERUMS, WE MUST CHANGE THE ABOVE IMPORTS AND ENSURE ITS THE SAME NAMES AND EVERYTHING. """






class GGIW_ExtendedKalmanFilter(ExtendedKalmanFilter):
    """Implementation of a continuous-discrete time Extended Kalman Filter for extended targets modeled as a 
    Gamma Gaussian Inverse Wishart distribution. 
    
    The gamma distribution describes the rate of detections of the target.
    The gaussian distribution describes the kinematics / states of the target. 
    The Inverse Wishart distribution describes the extent or spatial size of the target. 


    Reference paper: 
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7527981 

    
    One thing I changed is that the predict / correct steps take in and return GGIW objects rather than only the states. 
    With this edit, lines in carbs like: 
    'for ii, (m, P) in enumerate(zip(probDensity.means, probDensity.covariances)):
            self.filter.cov = P
            n_mean = self.filter.predict(timestep, m, **filt_args)
            covariances[ii] = self.filter.cov.copy()
            means[ii] = n_mean
        return smodels.GaussianMixture(
            means=means, covariances=covariances, weights=weights
        )'
    will not require doing the 'self.filter.cov = P', instead it's a GGIW object in and GGIW object out. 
    This is beneficial since we're adding additional properties to the distribution, and then we can use one instance 
    of the filter while using a mixture of numerous instances of the GGIW objects instead of changing it for each GGIW instance in the loop.

    Potential Future Work: 
    - integrate M matrix for rotating the extent model based on kinematic state 
        - This assumes X = M*X*M^T, where X is the IW matrix, and M is the transformation matrix as a function of kinematic state (eq 10 in reference paper)
        - New class for M dynamics? 
        - Include as input, similar to setting kinematic motion model (ie a double integrator can be easily plugged in)
    - look into self.__model attribute. 
        - I think it's for C++ coding, and could throw some stuff off for the GGIW? 
        - I commented out the self._init_model() beacuse I'm convinced that it is for C++ capabilities
    """
    def __init__(self, forgetting_factor = 1.0, tau = 1.0, cont_cov=True, dyn_obj=None, ode_lst=None, **kwargs):
        super().__init__(cont_cov, dyn_obj, ode_lst, **kwargs)

        # The setup steps should all be the same for the kinematics, 
        # so I'm simply editing the predict and correct functions as described above. 

        self.cont_cov = cont_cov
        self.integrator_type = "dopri5"
        self.integrator_params = {}

        self._ode_lst = None

        if dyn_obj is not None or ode_lst is not None:
            self.set_state_model(dyn_obj=dyn_obj, ode_lst=ode_lst)

        self._integrator = None

        self.__model = None
        self.__predParams = None
        self.__corrParams = None
        
        self.forgetting_factor = forgetting_factor   # for prediction step of gamma distribution, essentially affects how fast alpha and beta change over time
        self.tau = tau                               # for IW dof prediction

    def predict(
        self,
        timestep,
        GGIW_obj,
        dyn_fun_params=None,
        cur_input=None, 
        control_fun_params=None,
    ):
        
        cur_alpha = GGIW_obj.alpha
        cur_beta = GGIW_obj.beta
        cur_state = GGIW_obj.mean
        cur_cov = GGIW_obj.covariance
        cur_IWdof = GGIW_obj.IWdof
        cur_IWshape = GGIW_obj.IWshape


        # self._init_model()

        if self.__model is not None:
            if control_fun_params is None:
                control_fun_params = ()
            (
                self.__predParams.stateTransParams,
                self.__predParams.controlParams,
            ) = self._dyn_obj.args_to_params(dyn_fun_params, control_fun_params)[:2]
            next_state = self.__model.predict(
                timestep, cur_state, cur_input, self.__predParams
            ).reshape((-1, 1))

        else:
            if dyn_fun_params is None:
                dyn_fun_params = ()
            next_state, state_mat, dt = self._predict_next_state(
                timestep, cur_state, dyn_fun_params
            )

            if self.cont_cov:
                if dt is None:
                    raise RuntimeError(
                        "dt can not be None when using a continuous covariance model"
                    )

                def ode(t, x, n_states, F, proc_noise):
                    P = x.reshape((n_states, n_states))
                    P_dot = F @ P + P @ F.T + proc_noise
                    return P_dot.ravel()

                integrator = s_integrate.ode(ode)
                integrator.set_integrator(
                    self.integrator_type, **self.integrator_params
                )
                integrator.set_initial_value(cur_cov.flatten(), timestep)
                integrator.set_f_params(cur_state.size, state_mat, self.proc_noise)
                tmp = integrator.integrate(timestep + dt)
                if not integrator.successful():
                    msg = "Failed to integrate covariance at {}".format(timestep)
                    raise RuntimeError(msg)
                next_cov = tmp.reshape(cur_cov.shape)
            else:
                next_cov = state_mat @ cur_cov @ state_mat.T + self.proc_noise

        # next_cov = state_mat @ cur_cov @ state_mat.T + self.proc_noise

        # All predict steps above are the same for traditional EKFs and are only for the kinematics
        # Now for the additions: 

        next_alpha = cur_alpha / self.forgetting_factor
        next_beta = cur_beta / self.forgetting_factor

        next_IWdof = 2 * GGIW_obj.d + 2 + np.exp(-dt / self.tau) * (cur_IWdof - 2 * GGIW_obj.d - 2)
        next_IWshape = (next_IWdof - 2 * GGIW_obj.d - 2)/(cur_IWdof - 2 * GGIW_obj.d - 2) * cur_IWshape

        next_dist = GGIW(alpha=next_alpha, beta=next_beta, mean=next_state, covariance=next_cov, IWdof=next_IWdof, IWshape=next_IWshape)

        return next_dist
        
    def correct(
            self, 
            timestep, 
            meas, 
            GGIW_obj, 
            meas_fun_args=()
        ):
        
        """ Correction step.
        
        IMPORTANT: 
        meas : Nm x W numpy array, because np.size(meas, axis=0) is passed into the _est_meas function. Here, W is the number of detections from single extended target. """ 

        cur_alpha = GGIW_obj.alpha
        cur_beta = GGIW_obj.beta
        cur_state = GGIW_obj.mean
        cur_cov = GGIW_obj.covariance
        cur_IWdof = GGIW_obj.IWdof
        cur_IWshape = GGIW_obj.IWshape

        num_meas = len(meas) 
        
        meas_d = meas[0].shape[0]
        meas_arr = np.array(meas).reshape(num_meas, meas_d)

        W = np.size(meas_arr,axis=1)

        est_meas, meas_mat = self._est_meas(
            timestep, cur_state, np.size(meas_arr, axis=0), meas_fun_args
        )

        mean_meas = np.mean(meas_arr, axis=1)
        mean_meas = mean_meas.reshape((np.size(meas_arr,axis=0)),1)

        diff_Z = meas_arr - mean_meas
        Z = diff_Z @ diff_Z.T            # Essentially the scatter

        cur_IWshape = 0.5*(cur_IWshape+cur_IWshape.T)

        X_hat = cur_IWshape * (cur_IWdof - 2 * GGIW_obj.d - 2)**(-1)
        X_hat = (X_hat + X_hat.T)*0.5

        epsilon = mean_meas - meas_mat @ cur_state

        N = epsilon @ epsilon.T

        cur_cov = 0.5 * (cur_cov + cur_cov.T)

        S = meas_mat @ cur_cov @ meas_mat.T + X_hat / W + self.meas_noise
        S = (S + S.T) * 0.5 

        Vs = la.cholesky(S)
        det_S = la.det(Vs)
        inv_sqrt_S = la.inv(Vs)
        iS = inv_sqrt_S * inv_sqrt_S.T 

        K = cur_cov @ meas_mat.T @ iS 
        
        X_sqrt = sla.sqrtm(X_hat)
        S_sqrt_inv = sla.sqrtm(iS)

        N_hat = X_sqrt @ S_sqrt_inv @ N @ S_sqrt_inv.T @ X_sqrt.T

        next_alpha = cur_alpha + W
        next_beta = cur_beta + 1
        next_state = cur_state + K @ epsilon 
        next_cov = cur_cov - K @ meas_mat @ cur_cov
        next_IWdof = cur_IWdof + W
        next_IWshape = cur_IWshape + N_hat + Z 

        next_dist = GGIW(alpha=next_alpha, beta=next_beta, mean=next_state, covariance=next_cov, IWdof=next_IWdof, IWshape=next_IWshape)

        gam = next_alpha / next_beta

        
        # Compute each term
        term1  = (cur_IWdof - GGIW_obj.d - 1)/2 * np.log(np.linalg.det(cur_IWshape))
        term2  = - (next_IWdof - GGIW_obj.d - 1)/2 * np.log(np.linalg.det(next_IWshape))
        term3  = special.gammaln((next_IWdof - GGIW_obj.d - 1)/2)
        term4  = - special.gammaln((cur_IWdof - GGIW_obj.d - 1)/2)
        term5  = 0.5 * np.log(np.linalg.det(X_hat))
        term6  = -0.5 * np.log(det_S)
        term7  = special.gammaln(next_alpha)
        term8  = -special.gammaln(cur_alpha)
        term9  = cur_alpha * np.log(cur_beta)
        term10 = - next_alpha * np.log(next_beta)
        term11 = - ((W * np.log(np.pi) + np.log(W)) * GGIW_obj.d / 2)

        # Sum them up
        meas_fit_prob = (
            term1 + term2 + term3 + term4 + term5 + term6 + 
            term7 + term8 + term9 + term10 + term11
        )

        return (next_dist, meas_fit_prob) 
    
    def _calc_meas_fit(self): #, meas, GGIW_pred, GGIW_upd, X_hat, inov_cov):
        
        # W = np.size(meas, axis=1) 

        # d = GGIW_pred.IWshape.ndim

        # pred_alpha = GGIW_pred.alpha
        # pred_beta = GGIW_pred.beta
        # pred_state = GGIW_pred.mean
        # pred_cov = GGIW_pred.covariance
        # pred_IWdof = GGIW_pred.IWdof
        # pred_IWshape = GGIW_pred.IWshape 

        # upd_alpha = GGIW_upd.alpha
        # upd_beta = GGIW_upd.beta
        # upd_state = GGIW_upd.mean
        # upd_cov = GGIW_upd.covariance
        # upd_IWdof = GGIW_upd.IWdof
        # upd_IWshape = GGIW_upd.IWshape 

        L = 1

        # L = (np.pi**W * W) ** (-d/2) * la.det(pred_IWshape)**((pred_IWdof-d-1)/2) / \
        #         la.det(upd_IWshape)**((upd_IWdof-d-1)/2) * special.multigammaln((upd_IWdof-d-1)/2,d) / \
        #         special.multigammaln((pred_IWdof-d-1)/2,d) * la.det(X_hat)**0.5 / la.det(inov_cov)**0.5 * \
        #         special.gamma(upd_alpha)*pred_beta**(pred_alpha) / (special.gamma(pred_alpha)*upd_beta**(upd_alpha))

        return L

