"""Implements RFS guidance algorithms.

This module contains the classes and data structures
for RFS guidance related algorithms.
"""
import io
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
from warnings import warn
from scipy.optimize import linear_sum_assignment

import gncpy.control as gcontrol
import gncpy.plotting as gplot
import serums.models as smodels
from serums.distances import calculate_ospa
from serums.enums import SingleObjectDistance


def gaussian_density_cost(state_dist, goal_dist, safety_factor, y_ref):
    r"""Implements a GM density based cost function.

    Notes
    -----
    Implements the following cost function based on the difference between
    Gaussian mixtures, with additional terms to improve convergence when
    far from the targets.

    .. math::
        J &= \sum_{k=1}^{T} 10 N_g\sigma_{g,max}^2 \left( \sum_{j=1}^{N_g}
                \sum_{i=1}^{N_g} w_{g,k}^{(j)} w_{g,k}^{(i)}
                \mathcal{N}( \mathbf{m}^{(j)}_{g,k}; \mathbf{m}^{(i)}_{g,k},
                P^{(j)}_{g, k} + P^{(i)}_{g, k} ) \right. \\
            &- \left. 20 \sigma_{d, max}^2 N_d \sum_{j=1}^{N_d} \sum_{i=1}^{N_g}
                w_{d,k}^{(j)} w_{g,k}^{(i)} \mathcal{N}(
                \mathbf{m}^{(j)}_{d, k}; \mathbf{m}^{(i)}_{g, k},
                P^{(j)}_{d, k} + P^{(i)}_{g, k} ) \right) \\
            &+ \sum_{j=1}^{N_d} \sum_{i=1}^{N_d} w_{d,k}^{(j)}
                w_{d,k}^{(i)} \mathcal{N}( \mathbf{m}^{(j)}_{d,k};
                \mathbf{m}^{(i)}_{d,k}, P^{(j)}_{d, k} + P^{(i)}_{d, k} ) \\
            &+ \alpha \sum_{j=1}^{N_d} \sum_{i=1}^{N_g} w_{d,k}^{(j)}
                w_{g,k}^{(i)} \ln{\mathcal{N}( \mathbf{m}^{(j)}_{d,k};
                \mathbf{m}^{(i)}_{g,k}, P^{(j)}_{d, k} + P^{(i)}_{g, k} )}

    Parameters
    ----------
    state_dist : :class:`serums.models.GaussianMixture`
        Initial state distribution.
    goal_dist : :class:`serums.models.GaussianMixture`
        Desired state distribution.
    safety_factor : float
        Overbounding tuning factor for extra convergence term.
    y_ref : float
        Reference point to use on the sigmoid function, must be in the range
        (0, 1).

    Returns
    -------
    float
        cost.
    """
    all_goals = np.array([m.ravel().tolist() for m in goal_dist.means])
    all_states = np.array([m.ravel().tolist() for m in state_dist.means])
    target_center = np.mean(all_goals, axis=0).reshape((-1, 1))
    num_targets = all_goals.shape[0]
    num_objects = all_states.shape[0]
    state_dim = all_states.shape[1]

    # find radius of influence and shift
    diff = all_goals - target_center.T
    max_dist = np.sqrt(np.max(np.sum(diff * diff, axis=1)))
    radius_of_influence = safety_factor * max_dist
    shift = radius_of_influence + np.log(1 / y_ref - 1)

    # get actiavation term
    diff = all_states - target_center.T
    max_dist = np.sqrt(np.max(np.sum(diff * diff, axis=1)))
    activator = 1 / (1 + np.exp(-(max_dist - shift)))

    # get maximum variance
    max_var_obj = max(map(lambda x: float(np.max(np.diag(x))), state_dist.covariances))
    max_var_target = max(
        map(lambda x: float(np.max(np.diag(x))), goal_dist.covariances)
    )

    # Loop for all double summation terms
    sum_obj_obj = 0
    sum_obj_target = 0
    quad = 0
    for out_w, out_dist in state_dist:
        # create temporary gaussian object for calculations
        temp_gauss = smodels.Gaussian(mean=out_dist.mean)

        # object to object cost
        for in_w, in_dist in state_dist:
            temp_gauss.covariance = out_dist.covariance + in_dist.covariance
            sum_obj_obj += in_w * out_w * temp_gauss.pdf(in_dist.mean)

        # object to target and quadratic
        for tar_w, tar_dist in goal_dist:
            # object to target
            temp_gauss.covariance = out_dist.covariance + tar_dist.covariance
            sum_obj_target += tar_w * out_w * temp_gauss.pdf(tar_dist.mean)

            # quadratic
            diff = out_dist.mean - tar_dist.mean
            log_term = (
                np.log(
                    (2 * np.pi) ** (-0.5 * state_dim)
                    / np.sqrt(la.det(temp_gauss.covariance))
                )
                - 0.5 * diff.T @ la.inv(temp_gauss.covariance) @ diff
            )
            quad += out_w * tar_w * log_term.item()

    sum_target_target = 0
    for out_w, out_dist in goal_dist:
        temp_gauss = smodels.Gaussian(mean=out_dist.mean)
        for in_w, in_dist in goal_dist:
            temp_gauss.covariance = out_dist.covariance + in_dist.covariance
            sum_target_target += out_w * in_w * temp_gauss.pdf(in_dist.mean)

    return (
        10
        * num_objects
        * max_var_obj
        * (sum_obj_obj - 2 * max_var_target * num_targets * sum_obj_target)
        + sum_target_target
        + activator * quad
    )


class ELQR:
    """Implements the ELQR algorithm for swarms.

    Notes
    -----
    This follows :cite:`Thomas2021_RecedingHorizonExtendedLinearQuadraticRegulatorforRFSBasedSwarms`.

    Attributes
    ----------
    max_iters : int
        Maximum number of iterations to optimize.
    tol : float
        Relative tolerance for convergence.
    """

    def __init__(self, max_iters=1e3, tol=1e-4):
        """Initialize an object.

        Parameters
        ----------
        max_iters : int, optional
            Maximum number of iterations to optimize. The default is 1e3.
        tol : float, optional
            Relative tolerance for convergence. The default is 1e-4.
        """
        super().__init__()

        self.max_iters = int(max_iters)
        self.tol = tol
        self.end_dist = smodels.GaussianMixture()
        self.non_quad_weight = 1

        self._singleELQR = gcontrol.ELQR()
        self._elqr_lst = []
        self._start_covs = []
        self._start_weights = []
        self._time_vec = np.array([])
        self._cur_ind = None
        self._non_quad_fun = None

        self.__safety_factor = 1
        self.__y_ref = 0.5

    def get_state_dist(self, tt):
        """Calculates the current state distribution.

        Parameters
        ----------
        tt : float
            Current timestep.

        Returns
        -------
        :class:`serums.models.GaussianMixture`
            State distribution.
        """
        kk = int(np.argmin(np.abs(tt - self._time_vec)))
        means = [
            params["traj"][kk, :].copy().reshape((-1, 1)) for params in self._elqr_lst
        ]
        return smodels.GaussianMixture(
            means=means,
            covariances=[c.copy() for c in self._start_covs],
            weights=self._start_weights.copy(),
        )

    def non_quad_fun_factory(self):
        r"""Factory for creating the non-quadratic cost function.

        This should generate a function of the same form needed by the single
        agent controller, that is `time, state, control input, end state,
        is initial flag, is final flag, *args`. For this class, it implements
        a GM density based cost function plus an optional additional cost function
        of the same form. It returns a float for the non-quadratic part of
        the cost.

        Returns
        -------
        callable
            function to calculate cost.
        """

        def non_quadratic_fun(
            tt, state, ctrl_input, end_state, is_initial, is_final, *args
        ):
            state_dist = self.get_state_dist(tt)
            state_dist.remove_components(
                [self._cur_ind,]  # noqa
            )
            state_dist.add_components(
                state.reshape((-1, 1)),
                self._start_covs[self._cur_ind],
                self._start_weights[self._cur_ind],
            )
            cost = gaussian_density_cost(
                state_dist, self.end_dist, self.__safety_factor, self.__y_ref
            )
            if self._non_quad_fun is not None:
                cost += self.non_quad_weight * self._non_quad_fun(
                    tt, state, ctrl_input, end_state, is_initial, is_final, *args
                )
            return cost

        return non_quadratic_fun

    def set_cost_model(
        self,
        safety_factor=None,
        y_ref=None,
        non_quad_fun=None,
        quad_modifier=None,
        non_quad_weight=None,
    ):
        """Set the parameters for the cost non-quadratic function.

        This sets the parameters for the built-in non-quadratic cost function
        and allows specifying an additional non-quadratic function (see
        :meth:`gncpy.control.ELQR`) for the agents. Any additional non-quadratic
        terms must be set here instead of within the single control model directly
        due to the built-in multi-agent non-quadratic cost.

        Parameters
        ----------
        safety_factor : float, optional
            Additional scaling factor on the radius of influence. The default
            is None.
        y_ref : float, optional
            Reference point on the sigmoid function to use when determining
            th radius of influence. Should be between 0 and 1 (non-inclusive).
            The default is None.
        non_quad_fun : callable, optional
            Optional additional non-quadratic cost. Must have the same form as
            that of the single agent controller. See :meth:`gncpy.control.ELQR`.
            The default is None.
        quad_modifier : callable, optional
            Function to modify the quadratization process, see
            :meth:`gncpy.control.ELQR`. The default is None.
        non_quad_weight : float
            Additional scaling to be applied to the additional non_quadratic
            cost (the non built-in term). The default is None
        """
        if safety_factor is not None:
            self.__safety_factor = safety_factor

        if y_ref is not None:
            self.__y_ref = y_ref

        if non_quad_fun is not None:
            self._non_quad_fun = non_quad_fun

        if quad_modifier is not None:
            self._singleELQR.set_cost_model(
                quad_modifier=quad_modifier, skip_validity_check=True
            )

        if non_quad_weight is not None:
            self.non_quad_weight = non_quad_weight

    def set_control_model(self, singleELQR):
        """Sets the single agent control model used.

        Parameters
        ----------
        singleELQR : :class:`gncpy.control.ELQR`
            Single agent controller for generating trajectories.
        quad_modifier : callable, optional
            Modifing function for the quadratization. See
            :meth:`gncpy.control.ELQR.set_cost_model`. The default is None.
        """
        self._singleELQR = deepcopy(singleELQR)

    def find_end_state(self, tt, cur_ind, cur_state):
        """Finds the ending state for the given current state.

        Parameters
        ----------
        tt : float
            Current timestep
        cur_ind : int
            Current index into the state distribution
        cur_state : N x 1 numpy array
            Current state.

        Returns
        -------
        N x 1 numpy array
            Best ending state given the current state.
        """
        all_ends = np.vstack([m.ravel() for m in self.end_dist.means])
        diff = all_ends - cur_state.T
        min_ind = int(np.argmin(np.sum(diff * diff, axis=1)))

        return self.end_dist.means[min_ind]

    def init_elqr_lst(self, tt, start_dist):
        """Initialize the list of single agent ELQR controllers.

        Parameters
        ----------
        tt : float
            current time.
        start_dist : :class:`serums.models.GaussianMixture`
            Starting gaussian mixture.

        Returns
        -------
        num_timesteps : int
            total number of timesteps.
        """
        self._elqr_lst = []
        for ind, (w, dist) in enumerate(start_dist):
            p = {}
            p["elqr"] = deepcopy(self._singleELQR)
            end_state = np.zeros((dist.location.size, 1))
            p["old_cost"], num_timesteps, p["traj"], self._time_vec = p["elqr"].reset(
                tt, dist.location, end_state
            )
            self._elqr_lst.append(p)

        # reset with proper end state
        for ind, ((w, dist), p) in enumerate(zip(start_dist, self._elqr_lst)):
            end_state = self.find_end_state(tt, ind, dist.location)
            p["elqr"].reset(tt, dist.location, end_state)

        return num_timesteps

    def targets_to_wayareas(self, end_states):
        """Converts target locations to wayareas with automatic scaling.

        Performs a Principal Component Analysis (PCA) on the ending state
        locations to create a Gaussian Mixture.

        Parameters
        ----------
        end_states : Nt x N numpy array
            All possible ending states, one per row.

        Returns
        -------
        :class:`serums.models.GaussianMixture`
            Ending state distribution.
        """

        def find_principal_components(data):
            num_samps = data.shape[0]
            num_feats = data.shape[1]

            mean = np.sum(data, 0) / num_samps
            covars = np.zeros((num_feats, num_feats))
            for ii in range(0, num_feats):
                for jj in range(0, num_feats):
                    acc = 0
                    for samp in range(0, num_samps):
                        acc += (data[samp, ii] - mean[ii]) * (data[samp, jj] - mean[jj])
                    covars[ii, jj] = acc / num_samps
            (w, comps) = la.eig(covars)
            inds = np.argsort(w)[::-1]
            return comps[:, inds].T

        def find_largest_proj_dist(new_dirs, old_dirs):
            vals = np.zeros(new_dirs.shape[0])
            for ii in range(0, new_dirs.shape[1]):
                for jj in range(0, old_dirs.shape[1]):
                    proj = np.abs(new_dirs[:, [ii]].T @ old_dirs[:, [jj]])
                    if proj > vals[ii]:
                        vals[ii] = proj
            return vals

        thresh = 1e-2

        wayareas = smodels.GaussianMixture()
        all_ends = np.vstack([s.ravel() for s in end_states])
        aug_end_states = np.vstack((all_ends, np.mean(all_ends, axis=0)))

        directions = np.zeros(
            (aug_end_states.shape[1], aug_end_states.shape[0], aug_end_states.shape[0])
        )
        for ii, s_pt in enumerate(aug_end_states):
            for jj, e_pt in enumerate(aug_end_states):
                directions[:, ii, jj] = e_pt - s_pt

        weight = 1 / len(end_states)
        for wp_ind, center in enumerate(all_ends):
            sample_data = np.delete(aug_end_states, wp_ind, axis=0)
            sample_dirs = np.delete(directions[:, wp_ind, :].squeeze(), wp_ind, axis=1)
            comps = find_principal_components(sample_data)
            vals = find_largest_proj_dist(comps, sample_dirs)
            vals[vals <= thresh] = thresh

            cov = comps @ np.diag(vals) @ la.inv(comps)
            wayareas.add_components(center.reshape((-1, 1)), cov, weight)

        return wayareas

    def gen_final_traj(
        self,
        num_timesteps,
        start,
        elqr,
        state_args,
        ctrl_args,
        cost_args,
        inv_state_args,
        inv_ctrl_args,
    ):
        """Generates the final trajectory state and control trajectories.

        Parameters
        ----------
        num_timesteps : int
            total number of timesteps.
        start : N x 1 numpy array
            Initial state.
        elqr : :class:`gncpy.control.ELQR`
            Single agent controller for the given starting state.
        state_args : tuple
            Additional arguments for the state matrix.
        ctrl_args : tuple
            Additional arguments for the input matrix.
        cost_args : tuple
            Additional arguments for the cost function.
        inv_state_args : tuple
            Additional arguments for the inverse state transition matrix.
        inv_ctrl_args : tuple
            Additional arguments for the inverse input matrix.

        Returns
        -------
        state_traj : Nh+1 x N numpy array
            state trajectory.
        ctrl_signal : Nh x Nu numpy array
            control signal.
        cost : float
            cost of the trajectory.
        """
        ctrl_signal = np.nan * np.ones((num_timesteps, elqr.u_nom.size))
        state_traj = np.nan * np.ones((num_timesteps + 1, start.size))
        cost = 0
        state_traj[0, :] = start.flatten()
        for kk, tt in enumerate(self._time_vec[:-1]):
            ctrl_signal[kk, :] = (
                elqr.feedback_gain[kk] @ state_traj[kk, :].reshape((-1, 1))
                + elqr.feedthrough_gain[kk]
            ).ravel()
            if elqr.control_constraints is not None:
                ctrl_signal[kk, :] = elqr.control_constraints(
                    tt, ctrl_signal[kk, :].reshape((-1, 1))
                ).ravel()
            cost += elqr.cost_function(
                tt,
                state_traj[kk, :].reshape((-1, 1)),
                ctrl_signal[kk, :].reshape((-1, 1)),
                cost_args,
                is_initial=(kk == 0),
                is_final=False,
            )
            state_traj[kk + 1, :] = elqr.prop_state(
                tt,
                state_traj[kk, :].reshape((-1, 1)),
                ctrl_signal[kk, :].reshape((-1, 1)),
                state_args,
                ctrl_args,
                True,
                inv_state_args,
                inv_ctrl_args,
            ).ravel()

        cost += elqr.cost_function(
            self._time_vec[-1],
            state_traj[num_timesteps, :].reshape((-1, 1)),
            ctrl_signal[num_timesteps - 1, :].reshape((-1, 1)),
            cost_args,
            is_initial=False,
            is_final=True,
        )

        return state_traj, ctrl_signal, cost

    def draw_init_states(
        self, fig, states, plt_inds, marker, zorder, cmap=None, color=None
    ):
        kwargs = dict(marker=marker, zorder=zorder,)
        if color is not None:
            kwargs["color"] = color

        for c_ind, (w, dist) in enumerate(states):
            s = dist.location
            if cmap is not None:
                kwargs["color"] = cmap(c_ind)
            fig.axes[0].scatter(s[plt_inds[0], 0], s[plt_inds[1], 0], **kwargs)

    def save_animation(self, fig, fig_h, fig_w, frame_list):
        with io.BytesIO() as buff:
            fig.savefig(buff, format="raw")
            buff.seek(0)
            img = np.frombuffer(buff.getvalue(), dtype=np.uint8).reshape(
                (fig_h, fig_w, -1)
            )
        frame_list.append(Image.fromarray(img))

    def init_plot(
        self,
        show_animation,
        save_animation,
        cmap,
        start_dist,
        fig,
        plt_opts,
        ttl,
        plt_inds,
    ):
        frame_list = []
        fig_h = None
        fig_w = None
        if show_animation:
            if cmap is None:
                cmap = gplot.get_cmap(len(start_dist))

            if fig is None:
                fig = plt.figure()
                fig.add_subplot(1, 1, 1)
                fig.axes[0].set_aspect("equal", adjustable="box")

                if plt_opts is None:
                    plt_opts = gplot.init_plotting_opts(f_hndl=fig)

                if ttl is None:
                    ttl = "Multi-Agent ELQR"

                gplot.set_title_label(fig, 0, plt_opts, ttl=ttl)

                # draw start
                self.draw_init_states(fig, start_dist, plt_inds, "o", 1000, cmap=cmap)

            self.draw_init_states(fig, self.end_dist, plt_inds, "x", 1000, color="r")

            fig.tight_layout()
            plt.pause(0.1)

            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            fig_w, fig_h = fig.canvas.get_width_height()

            # save first frame of animation
            if save_animation:
                self.save_animation(fig, fig_h, fig_w, frame_list)

        return fig, fig_h, fig_w, frame_list, cmap

    def reset(self, start_dist):
        self._start_covs = [c.copy() for c in start_dist.covariances]
        self._start_weights = [w for w in start_dist.weights]

    def output_helper(
        self,
        c_ind,
        num_timesteps,
        start,
        params,
        state_args,
        ctrl_args,
        cost_args,
        inv_state_args,
        inv_ctrl_args,
        costs,
        state_trajs,
        ctrl_signals,
        show_animation,
        fig,
        plt_inds,
        cmap,
    ):
        self._cur_ind = c_ind
        params["elqr"].set_cost_model(
            non_quadratic_fun=self.non_quad_fun_factory(), skip_validity_check=True
        )
        st, cs, c = self.gen_final_traj(
            num_timesteps,
            start,
            params["elqr"],
            state_args,
            ctrl_args,
            cost_args,
            inv_state_args,
            inv_ctrl_args,
        )
        costs.append(c)
        state_trajs.append(st)
        ctrl_signals.append(cs)

        if show_animation:
            fig.axes[0].plot(
                st[:, plt_inds[0]],
                st[:, plt_inds[1]],
                linestyle="-",
                color=cmap(c_ind),
            )
            plt.pause(0.01)

    def create_outputs(
        self,
        start_dist,
        num_timesteps,
        state_args,
        ctrl_args,
        cost_args,
        inv_state_args,
        inv_ctrl_args,
        show_animation,
        fig,
        plt_inds,
        cmap,
    ):
        costs = []
        state_trajs = []
        ctrl_signals = []
        for c_ind, ((w, dist), params) in enumerate(zip(start_dist, self._elqr_lst)):
            self.output_helper(
                c_ind,
                num_timesteps,
                dist.location,
                params,
                state_args,
                ctrl_args,
                cost_args,
                inv_state_args,
                inv_ctrl_args,
                costs,
                state_trajs,
                ctrl_signals,
                show_animation,
                fig,
                plt_inds,
                cmap,
            )

        return state_trajs, costs, ctrl_signals

    def plan(
        self,
        tt,
        start_dist,
        end_dist,
        state_args=None,
        ctrl_args=None,
        cost_args=None,
        inv_state_args=None,
        inv_ctrl_args=None,
        provide_details=False,
        disp=True,
        show_animation=False,
        save_animation=False,
        plt_opts=None,
        ttl=None,
        fig=None,
        cmap=None,
        plt_inds=None,
        update_end=True,
        update_end_iters=1,
    ):
        """Main planning function.

        Parameters
        ----------
        tt : float
            Starting timestep for the plan.
        start_dist : :class:`serums.models.GaussianMixture`
            Starting state distribution.
        end_dist : :class:`serums.models.GaussianMixture`
            Ending state distribution.
        state_args : tuple, optional
            Additional arguments for getting the state transition matrix. The
            default is None.
        ctrl_args : tuple, optional
            Additional arguements for getting the input matrix. The default is
            None.
        cost_args : tuple, optional
            Additional arguments for the cost function. The default is None.
        inv_state_args : tuple, optional
            Additional arguments to get the inverse state matrix. The default
            is None.
        inv_ctrl_args : tuple, optional
            Additional arguments to get the inverse input matrix. The default
            is None.
        provide_details : bool, optional
            Falg for if optional outputs should be output. The default is False.
        disp : bool, optional
            Falg for if additional text should be printed. The default is True.
        show_animation : bool, optional
            Flag for if an animation is generated. The default is False.
        save_animation : bool, optional
            Flag for saving the animation. Only applies if the animation is
            shown. The default is False.
        plt_opts : dict, optional
            Additional plotting options. See
            :func:`gncpy.plotting.init_plotting_opts`. The default is None.
        ttl : string, optional
            Title of the generated plot. The default is None.
        fig : matplotlib figure, optional
            Handle to the figure. If supplied only the end states are added.
            The default is None.
        cmap : matplotlib colormap, optional
            Color map for the different agents. See :func:`gncpy.plotting.get_cmap`.
            The default is None.
        plt_inds : list, optional
            Indices in the state vector to plot. The default is None.

        Returns
        -------
        state_trajs : list
            Each element is an Nh+1xN numpy array.
        costs : list, optional
            Each element is a float for the cost of that trajectory
        ctrl_signals : list, optional
            Each element is an NhxNu numpy array
        fig : matplotlib figure, optional
            Handle to the generated figure
        frame_list : list, optional
            Each element is a PIL image if the animation is being saved.
        """
        if state_args is None:
            state_args = ()
        if ctrl_args is None:
            ctrl_args = ()
        if cost_args is None:
            cost_args = ()
        if inv_state_args is None:
            inv_state_args = ()
        if inv_ctrl_args is None:
            inv_ctrl_args = ()
        if plt_inds is None:
            plt_inds = [0, 1]

        self.end_dist = end_dist

        num_timesteps = self.init_elqr_lst(tt, start_dist)
        old_cost = float("inf")
        self.reset(start_dist)

        fig, fig_h, fig_w, frame_list, cmap = self.init_plot(
            show_animation,
            save_animation,
            cmap,
            start_dist,
            fig,
            plt_opts,
            ttl,
            plt_inds,
        )

        if disp:
            print("Starting ELQR optimization loop...")

        for itr in range(self.max_iters):
            # forward pass for each gaussian, step by step
            for kk in range(num_timesteps):
                for ind, params in enumerate(self._elqr_lst):
                    self._cur_ind = ind
                    params["elqr"].set_cost_model(
                        non_quadratic_fun=self.non_quad_fun_factory(),
                        skip_validity_check=True,
                    )
                    params["traj"][kk + 1, :] = params["elqr"].forward_pass_step(
                        itr,
                        kk,
                        self._time_vec,
                        params["traj"],
                        state_args,
                        ctrl_args,
                        cost_args,
                        inv_state_args,
                        inv_ctrl_args,
                    )

            # quadratize final cost for each gaussian
            for c_ind, params in enumerate(self._elqr_lst):
                self._cur_ind = c_ind
                params["elqr"].set_cost_model(
                    non_quadratic_fun=self.non_quad_fun_factory(),
                    skip_validity_check=True,
                )
                # if update_end and itr != 0 and itr % update_end_iters == 0:
                #     params["elqr"].end_state = self.find_end_state(
                #         self._time_vec[-1], c_ind, params["traj"][-1, :].reshape((-1, 1)),
                #     )
                params["traj"] = params["elqr"].quadratize_final_cost(
                    itr, num_timesteps, params["traj"], self._time_vec, cost_args
                )

            # backward pass for each gaussian
            for kk in range(num_timesteps - 1, -1, -1):
                for c_ind, params in enumerate(self._elqr_lst):
                    self._cur_ind = c_ind
                    params["elqr"].set_cost_model(
                        non_quadratic_fun=self.non_quad_fun_factory(),
                        skip_validity_check=True,
                    )
                    params["traj"][kk, :] = params["elqr"].backward_pass_step(
                        itr,
                        kk,
                        self._time_vec,
                        params["traj"],
                        state_args,
                        ctrl_args,
                        cost_args,
                        inv_state_args,
                        inv_ctrl_args,
                    )

            # get true cost
            cost = 0
            x = np.nan * np.ones(
                (len(self._elqr_lst), self._elqr_lst[0]["traj"].shape[1])
            )
            for kk, tt in enumerate(self._time_vec[:-1]):
                for ind, params in enumerate(self._elqr_lst):
                    x[ind] = params["traj"][kk].copy()
                    u = (
                        params["elqr"].feedback_gain[kk] @ x[ind].reshape((-1, 1))
                        + params["elqr"].feedthrough_gain[kk]
                    )
                    if params["elqr"].control_constraints is not None:
                        u = params["elqr"].control_constraints(tt, u)
                    self._cur_ind = ind
                    params["elqr"].set_cost_model(
                        non_quadratic_fun=self.non_quad_fun_factory(),
                        skip_validity_check=True,
                    )
                    cost += params["elqr"].cost_function(
                        tt,
                        x[ind].reshape((-1, 1)),
                        u,
                        cost_args,
                        is_initial=(kk == 0),
                        is_final=False,
                    )
                    x[ind] = (
                        params["elqr"]
                        .prop_state(
                            tt,
                            x[ind].reshape((-1, 1)),
                            u,
                            state_args,
                            ctrl_args,
                            True,
                            inv_state_args,
                            inv_ctrl_args,
                        )
                        .ravel()
                    )

            for ind, params in enumerate(self._elqr_lst):
                if update_end and itr != 0 and itr % update_end_iters == 0:
                    params["elqr"].end_state = self.find_end_state(
                        self._time_vec[-1], ind, x[ind].reshape((-1, 1))
                    )
                u = (
                    params["elqr"].feedback_gain[-1] @ x[ind].reshape((-1, 1))
                    + params["elqr"].feedthrough_gain[-1]
                )
                if params["elqr"].control_constraints is not None:
                    u = params["elqr"].control_constraints(tt, u)

                self._cur_ind = ind
                params["elqr"].set_cost_model(
                    non_quadratic_fun=self.non_quad_fun_factory(),
                    skip_validity_check=True,
                )
                cost += params["elqr"].cost_function(
                    self._time_vec[-1],
                    x[ind].reshape((-1, 1)),
                    u,
                    cost_args,
                    is_initial=False,
                    is_final=True,
                )

            # for ind, params in enumerate(self._elqr_lst):
            #     self._cur_ind = ind
            #     params["elqr"].set_cost_model(
            #         non_quadratic_fun=self.non_quad_fun_factory(),
            #         skip_validity_check=True,
            #     )
            #     x = params["traj"][0, :].copy().reshape((-1, 1))
            #     for kk, tt in enumerate(self._time_vec[:-1]):
            #         u = (
            #             params["elqr"].feedback_gain[kk] @ x
            #             + params["elqr"].feedthrough_gain[kk]
            #         )
            #         if params["elqr"].control_constraints is not None:
            #             u = params["elqr"].control_constraints(tt, u)
            #         cost += params["elqr"].cost_function(
            #             tt, x, u, cost_args, is_initial=(kk == 0), is_final=False,
            #         )
            #         x = params["elqr"].prop_state(
            #             tt,
            #             x,
            #             u,
            #             state_args,
            #             ctrl_args,
            #             True,
            #             inv_state_args,
            #             inv_ctrl_args,
            #         )
            #     params["elqr"].end_state = self.find_end_state(
            #         self._time_vec[-1], ind, params["traj"][-1, :].reshape((-1, 1))
            #     )
            #     cost += params["elqr"].cost_function(
            #         self._time_vec[-1],
            #         x,
            #         u,
            #         cost_args,
            #         is_initial=False,
            #         is_final=True,
            #     )

            if disp:
                print("\tIteration: {:3d} Cost: {:10.4f}".format(itr, cost))

            if show_animation:
                for c_ind, params in enumerate(self._elqr_lst):
                    img = params["elqr"].draw_traj(
                        fig,
                        plt_inds,
                        fig_h,
                        fig_w,
                        c_ind == (len(self._elqr_lst) - 1),
                        num_timesteps,
                        self._time_vec,
                        state_args,
                        ctrl_args,
                        inv_state_args,
                        inv_ctrl_args,
                        color=cmap(c_ind),
                        alpha=0.2,
                        zorder=-10,
                    )

                if save_animation:
                    frame_list.append(img)

            # check for convergence
            if np.abs((old_cost - cost) / cost) < self.tol:
                break
            old_cost = cost

        # generate control and state trajectories for all agents
        state_trajs, costs, ctrl_signals = self.create_outputs(
            start_dist,
            num_timesteps,
            state_args,
            ctrl_args,
            cost_args,
            inv_state_args,
            inv_ctrl_args,
            show_animation,
            fig,
            plt_inds,
            cmap,
        )

        if show_animation and save_animation:
            plt.pause(0.01)
            self.save_animation(fig, fig_h, fig_w, frame_list)

        details = (costs, ctrl_signals, fig, frame_list)
        return (state_trajs, *details) if provide_details else state_trajs


class ELQROSPA(ELQR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.end_dist = np.array([])

        self.loiter_dist = 1

        self.__ospa_inds = []
        self.__ospa_cutoff = 1e3
        self.__ospa_core = SingleObjectDistance.EUCLIDEAN

    def get_state_dist(self, tt):
        """Calculates the current state distribution.

        Parameters
        ----------
        tt : float
            Current timestep.

        Returns
        -------
        Na x N numpy array
            State distribution, one row per agent
        """
        kk = int(np.argmin(np.abs(tt - self._time_vec)))
        return np.vstack([params["traj"][kk, :].flatten() for params in self._elqr_lst])

    def non_quad_fun_factory(self):
        """Factory for creating the non-quadratic cost function.

        This should generate a function of the same form needed by the single
        agent controller, that is `time, state, control input, end state,
        is initial flag, is final flag, *args`. For this class, it implements
        an OSPA based cost function plus an optional additional cost function
        of the same form. It returns a float for the non-quadratic part of
        the cost.

        Returns
        -------
        callable
            function to calculate cost.
        """

        def non_quadratic_fun(
            tt, state, ctrl_input, end_state, is_initial, is_final, *args
        ):
            if self.__ospa_inds is None:
                inds = [i for i in range(state.size)]
            else:
                inds = self.__ospa_inds
            start_dist = self.get_state_dist(tt)
            start_dist[self._cur_ind, :] = state.flatten()
            start_dist = start_dist[:, inds].T.reshape(
                (len(inds), 1, start_dist.shape[0])
            )

            end_dist = self.end_dist[:, inds].T.reshape(
                (len(inds), 1, self.end_dist.shape[0])
            )

            cost = calculate_ospa(
                start_dist,
                end_dist,
                self.__ospa_cutoff,
                1,
                core_method=self.__ospa_core,
            )[0].item()
            if self._non_quad_fun is not None:
                cost += self.non_quad_weight * self._non_quad_fun(
                    tt, state, ctrl_input, end_state, is_initial, is_final, *args
                )
            return cost

        return non_quadratic_fun

    def set_cost_model(
        self,
        ospa_inds=None,
        ospa_cutoff=None,
        ospa_method=None,
        non_quad_fun=None,
        quad_modifier=None,
        non_quad_weight=None,
    ):
        """Set the parameters for the cost non-quadratic function.

        This sets the parameters for the built-in non-quadratic cost function
        and allows specifying an additional non-quadratic function (see
        :meth:`gncpy.control.ELQR`) for the agents. Any additional non-quadratic
        terms must be set here instead of within the single control model directly
        due to the built-in multi-agent non-quadratic cost.

        Parameters
        ----------
        non_quad_fun : callable, optional
            Optional additional non-quadratic cost. Must have the same form as
            that of the single agent controller. See :meth:`gncpy.control.ELQR`.
            The default is None.
        quad_modifier : callable, optional
            Function to modify the quadratization process, see
            :meth:`gncpy.control.ELQR`. The default is None.
        non_quad_weight : float
            Additional scaling to be applied to the additional non_quadratic
            cost (the non built-in term). The default is None
        """
        if ospa_inds is not None:
            self.__ospa_inds = ospa_inds

        if ospa_cutoff is not None:
            self.__ospa_cutoff = ospa_cutoff

        if ospa_method is not None:
            self.__ospa_core = ospa_method

        if non_quad_fun is not None:
            self._non_quad_fun = non_quad_fun

        if quad_modifier is not None:
            self._singleELQR.set_cost_model(
                quad_modifier=quad_modifier, skip_validity_check=True
            )

        if non_quad_weight is not None:
            self.non_quad_weight = non_quad_weight

    def find_end_state(self, tt, cur_ind, cur_state):
        """Finds the ending state for the given current state.

        Parameters
        ----------
        tt : float
            Current timestep
        cur_ind : int
            Current index into the state distribution
        cur_state : N x 1 numpy array
            Current state.

        Returns
        -------
        N x 1 numpy array
            Best ending state given the current state.
        """
        if self.__ospa_inds is None:
            inds = [i for i in range(cur_state.size)]
        else:
            inds = self.__ospa_inds
        dist = self.get_state_dist(tt)
        dist[cur_ind] = cur_state.ravel()
        dist = dist[:, inds]

        end_dist = self.end_dist[:, inds]
        if end_dist.shape[0] < dist.shape[0]:
            if end_dist.shape[0] > 1:
                center = np.mean(end_dist, axis=0).reshape((1, -1))
            else:
                # direction from end to current state
                direc = cur_state.ravel()[inds] - end_dist[0]
                mag = np.linalg.norm(direc)
                if mag > 1e-16:
                    direc /= mag
                else:
                    # current agent is at the only target, use all agents center instead
                    a_cent = np.mean(dist, axis=0)
                    direc = a_cent - end_dist[0]
                    mag = np.linalg.norm(direc)
                    if mag > 1e-16:
                        direc /= mag
                    else:
                        # all agents center is at the only target...just pick x direction
                        direc = np.zeros(len(inds))
                        direc[0] = 1
                direc *= self.loiter_dist
                center = (end_dist[0] + direc).reshape((1, -1))

            # add enough "fake" targets so there is a 1-to-1 matching
            n_miss = dist.shape[0] - end_dist.shape[0]
            end_dist = np.vstack((end_dist, center * np.ones((n_miss, len(inds)))))

        distances, a_exists, t_exists = calculate_ospa(
            dist.T.reshape((len(inds), 1, -1)),
            end_dist.T.reshape((len(inds), 1, -1)),
            self.__ospa_cutoff,
            1,
            core_method=self.__ospa_core,
            use_empty=False,
        )[6:]
        cont_sub = distances[
            0 : np.sum(a_exists).astype(int), 0 : np.sum(t_exists).astype(int), 0
        ]
        tar_inds = linear_sum_assignment(cont_sub)[1]

        min_ind = tar_inds[cur_ind]
        if min_ind >= self.end_dist.shape[0]:
            out = np.zeros((self.end_dist.shape[1], 1))
            out[inds] = end_dist[min_ind].reshape(out[inds].shape)
        else:
            out = self.end_dist[min_ind, :].reshape((-1, 1))
        return out

    def init_elqr_lst(self, tt, start_dist):
        """Initialize the list of single agent ELQR controllers.

        Parameters
        ----------
        tt : float
            current time.
        start_dist : Na x N numpy array
            Starting states, one per row.

        Returns
        -------
        num_timesteps : int
            total number of timesteps.
        """
        self._elqr_lst = []
        # initialize with fake end state
        for ind, s in enumerate(start_dist):
            p = {}
            p["elqr"] = deepcopy(self._singleELQR)
            end_state = np.zeros((s.size, 1))
            p["old_cost"], num_timesteps, p["traj"], self._time_vec = p["elqr"].reset(
                tt, s.reshape((-1, 1)), end_state
            )
            self._elqr_lst.append(p)

        # reset with proper end state
        for ind, (s, p) in enumerate(zip(start_dist, self._elqr_lst)):
            end_state = self.find_end_state(tt, ind, s)
            p["elqr"].reset(tt, s.reshape((-1, 1)), end_state)

        return num_timesteps

    def targets_to_wayareas(self, end_states):
        warn("targets_to_wayareas not used by ELQROSPA")
        return None

    def draw_init_states(
        self, fig, states, plt_inds, marker, zorder, color=None, cmap=None
    ):
        kwargs = dict(marker=marker, zorder=zorder,)
        if color is not None:
            kwargs["color"] = color

        for c_ind, s in enumerate(states):
            if cmap is not None:
                kwargs["color"] = cmap(c_ind)
            fig.axes[0].scatter(s[plt_inds[0]], s[plt_inds[1]], **kwargs)

    def reset(self, start_dist):
        pass

    def create_outputs(
        self,
        start_dist,
        num_timesteps,
        state_args,
        ctrl_args,
        cost_args,
        inv_state_args,
        inv_ctrl_args,
        show_animation,
        fig,
        plt_inds,
        cmap,
    ):
        costs = []
        state_trajs = []
        ctrl_signals = []
        for c_ind, (s, params) in enumerate(zip(start_dist, self._elqr_lst)):
            self.output_helper(
                c_ind,
                num_timesteps,
                s.reshape((-1, 1)),
                params,
                state_args,
                ctrl_args,
                cost_args,
                inv_state_args,
                inv_ctrl_args,
                costs,
                state_trajs,
                ctrl_signals,
                show_animation,
                fig,
                plt_inds,
                cmap,
            )

        return state_trajs, costs, ctrl_signals

    def plan(self, tt, start_dist, end_dist, **kwargs):
        return super().plan(tt, start_dist, end_dist, **kwargs)
