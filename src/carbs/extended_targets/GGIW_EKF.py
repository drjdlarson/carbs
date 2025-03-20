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
        cur_input=None,
        dyn_fun_params=None,
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

        next_IWdof = 2 * GGIW_obj.d + 2 + np.exp(-timestep / self.tau) * (cur_IWdof - 2 * GGIW_obj.d - 2)
        next_IWshape = cur_IWshape

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

        W = np.size(meas, axis=1)

        # self._init_model()

        # if self.__model is not None:
        #     # self.__corrParams.measParams = self._measObj.args_to_params(meas_fun_args)
        #     out = self.__model.correct(timestep, meas)
        #     return out[0].reshape((-1, 1)), out[1]

        # else:
        est_meas, meas_mat = self._est_meas(
            timestep, cur_state, np.size(meas, axis=0), meas_fun_args
        )

        mean_meas = np.mean(meas, axis=1)
        mean_meas = mean_meas.reshape((np.size(meas,axis=0)),1)

        Z = 0
        for kk in range(0,W,1):
            diff_Z = meas[:,kk] - mean_meas
            Z += diff_Z @ diff_Z.T            # Essentiall the scatter

        X_hat = cur_IWshape * (cur_IWdof - 2 * GGIW_obj.d - 2)**(-1)

        epsilon = mean_meas - meas_mat @ cur_state

        N = epsilon @ epsilon.T

        ### get the Kalman gain (updated for GGIW) ###
        # cov_meas_T = cur_cov @ meas_mat.T
        # inov_cov = meas_mat @ cov_meas_T
        inov_cov = meas_mat @ cur_cov @ meas_mat.T + X_hat / W 

        # estimate the measurement noise online if applicable
        if self._est_meas_noise_fnc is not None:
            self.meas_noise = self._est_meas_noise_fnc(est_meas, inov_cov)

        inov_cov += self.meas_noise   # I'm keeping meas_noise just to see its effects (might make trajectories more smooth?) 

        inov_cov = (inov_cov + inov_cov.T) * 0.5            # To support numerical stability / positive definiteness 

        if self.use_cholesky_inverse:
            sqrt_inv_inov_cov = la.inv(la.cholesky(inov_cov))
            inv_inov_cov = sqrt_inv_inov_cov.T @ sqrt_inv_inov_cov
        else:
            inv_inov_cov = la.inv(inov_cov)

        kalman_gain = cur_cov @ meas_mat.T @ inv_inov_cov    # Kalman gain finally

        X_hat_power = sla.sqrtm(X_hat)

        inov_cov_power = -sla.sqrtm(inov_cov)

        N_hat = X_hat_power @ inov_cov_power @ N @ X_hat_power.T @ inov_cov_power.T 

        next_alpha = cur_alpha + W
        next_beta = cur_beta + 1
        next_state = cur_state + kalman_gain @ epsilon 
        next_cov = cur_cov - kalman_gain @ meas_mat @ cur_cov
        next_IWdof = cur_IWdof + W
        next_IWshape = cur_IWshape + N_hat + Z 

        next_dist = GGIW(alpha=next_alpha, beta=next_beta, mean=next_state, covariance=next_cov, IWdof=next_IWdof, IWshape=next_IWshape)

        # # update the state with measurement
        # inov = meas - est_meas
        # next_state = cur_state + kalman_gain @ inov

        # # update the covariance
        # n_states = cur_state.shape[0]
        # cur_cov = (np.eye(n_states) - kalman_gain @ meas_mat) @ cur_cov

        # calculate the measurement fit probability assuming Gaussian 
        meas_fit_prob = self._calc_meas_fit(meas,GGIW_obj,next_dist,X_hat,inov_cov) # meas, est_meas, inov_cov) 

        return (next_dist, meas_fit_prob)

    def _calc_meas_fit(self, meas, GGIW_pred, GGIW_upd, X_hat, inov_cov):
        
        W = np.size(meas, axis=1) 

        d = GGIW_pred.IWshape.ndim

        pred_alpha = GGIW_pred.alpha
        pred_beta = GGIW_pred.beta
        pred_state = GGIW_pred.mean
        pred_cov = GGIW_pred.covariance
        pred_IWdof = GGIW_pred.IWdof
        pred_IWshape = GGIW_pred.IWshape 

        upd_alpha = GGIW_upd.alpha
        upd_beta = GGIW_upd.beta
        upd_state = GGIW_upd.mean
        upd_cov = GGIW_upd.covariance
        upd_IWdof = GGIW_upd.IWdof
        upd_IWshape = GGIW_upd.IWshape 

        L = 1

        # L = (np.pi**W * W) ** (-d/2) * la.det(pred_IWshape)**((pred_IWdof-d-1)/2) / \
        #         la.det(upd_IWshape)**((upd_IWdof-d-1)/2) * special.multigammaln((upd_IWdof-d-1)/2,d) / \
        #         special.multigammaln((pred_IWdof-d-1)/2,d) * la.det(X_hat)**0.5 / la.det(inov_cov)**0.5 * \
        #         special.gamma(upd_alpha)*pred_beta**(pred_alpha) / (special.gamma(pred_alpha)*upd_beta**(upd_alpha))

        return L

