def modify_quadratize():
    import numpy as np

    import serums.models as smodels
    import gncpy.control as gcontrol
    from gncpy.dynamics.basic import IRobotCreate
    from carbs.guidance import ELQR

    d2r = np.pi / 180

    tt = 0  # starting time when calculating control
    dt = 0.01
    time_horizon = 90 * dt  # run for 90 timesteps

    # define starting and ending state for control calculation
    end_states = [
        np.array([1.5, 2, 90 * d2r]).reshape((3, 1)),
        np.array([0, 2.5, 90 * d2r]).reshape((3, 1)),
        np.array([-1.5, 2, 90 * d2r]).reshape((3, 1)),
    ]
    start_dist = smodels.GaussianMixture()
    start_dist.add_components(
        np.array([-1, -0.25, 90 * d2r]).reshape((-1, 1)),
        np.diag([0.25, 0.25, 1 * d2r]),
        0.5,
    )
    start_dist.add_components(
        np.array([1, -0.25, 90 * d2r]).reshape((-1, 1)),
        np.diag([0.25, 0.25, 1 * d2r]),
        0.5,
    )

    # define nominal control input
    u_nom = np.zeros((2, 1))

    # define dynamics
    # IRobot Create has a dt that can be set here or it can be set by the control
    # algorithm
    dynObj = IRobotCreate(wheel_separation=0.258, radius=0.335 / 2.0)

    # define Q and R weights for using standard cost function
    Q = np.diag([1500, 1000, 1000])
    R = 0.5 * np.eye(u_nom.size)

    # define modifications for quadratizing the cost function
    def quad_modifier(itr, tt, P, Q, R, q, r):
        rot_cost = 0.4
        # only modify if within the first 4 iterations
        if itr < 4:
            Q[-1, -1] = rot_cost
            q[-1] = -rot_cost * np.pi / 2

        return P, Q, R, q, r

    # create control obect
    singleELQR = gcontrol.ELQR(time_horizon=time_horizon)
    singleELQR.set_state_model(u_nom, dynObj=dynObj)
    singleELQR.dt = dt  # set here or within the dynamic object
    singleELQR.set_cost_model(Q=Q, R=R)

    # create guidance object
    safety_factor = 4
    y_ref = 0.9

    elqr = ELQR()
    elqr.set_control_model(singleELQR)
    elqr.set_cost_model(
        quad_modifier=quad_modifier,
        safety_factor=safety_factor,
        y_ref=y_ref,
    )

    # convert end locations into a distribution or "wayareas"
    end_dist = elqr.targets_to_wayareas(end_states)

    # calculate guidance
    (
        state_trajectories,
        costs,
        control_signals,
        fig,
        frame_list,
    ) = elqr.plan(
        tt,
        start_dist,
        end_dist,
        provide_details=True,
        show_animation=True,
        save_animation=True,
        plt_inds=[0, 1],
        ttl="Multi-Agent GM Based ELQR",
    )

    return frame_list


def elqr_ospa():
    import numpy as np

    import serums.models as smodels
    import gncpy.control as gcontrol
    from gncpy.dynamics.basic import IRobotCreate
    from carbs.guidance import ELQROSPA
    from serums.distances import SingleObjectDistance

    d2r = np.pi / 180

    tt = 0  # starting time when calculating control
    dt = 0.01
    time_horizon = 90 * dt  # run for 90 timesteps

    # define starting and ending state for control calculation
    end_states = np.array([[1.5, 2, 90 * d2r], [0, 2.5, 90 * d2r], [-1.5, 2, 90 * d2r]])
    start_states = np.array([[-1, -0.25, 90 * d2r], [1, -0.25, 90 * d2r]])

    # define nominal control input
    u_nom = np.zeros((2, 1))

    # define dynamics
    # IRobot Create has a dt that can be set here or it can be set by the control
    # algorithm
    dynObj = IRobotCreate(wheel_separation=0.258, radius=0.335 / 2.0)

    # define Q and R weights for using standard cost function
    Q = np.diag([1500, 1000, 1000])
    R = 0.5 * np.eye(u_nom.size)

    # define modifications for quadratizing the cost function
    def quad_modifier(itr, tt, P, Q, R, q, r):
        rot_cost = 0.4
        # only modify if within the first 4 iterations
        if itr < 4:
            Q[-1, -1] = rot_cost
            q[-1] = -rot_cost * np.pi / 2

        return P, Q, R, q, r

    # create control obect
    singleELQR = gcontrol.ELQR(time_horizon=time_horizon)
    singleELQR.set_state_model(u_nom, dynObj=dynObj)
    singleELQR.dt = dt  # set here or within the dynamic object
    singleELQR.set_cost_model(Q=Q, R=R)

    # create guidance object
    pos_inds = [0, 1]
    cutoff = 10

    elqr = ELQROSPA()
    elqr.set_control_model(singleELQR)
    elqr.set_cost_model(
        quad_modifier=quad_modifier,
        ospa_inds=pos_inds,
        ospa_cutoff=cutoff,
        ospa_method=SingleObjectDistance.EUCLIDEAN,
    )

    # calculate guidance
    (
        state_trajectories,
        costs,
        control_signals,
        fig,
        frame_list,
    ) = elqr.plan(
        tt,
        start_states,
        end_states,
        provide_details=True,
        show_animation=True,
        save_animation=True,
        plt_inds=pos_inds,
        ttl="Multi-Agent OSPA Based ELQR",
    )

    return frame_list


def elqr_ospa_obstacles():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle

    import gncpy.control as gcontrol
    import gncpy.plotting as gplot
    from gncpy.dynamics.basic import IRobotCreate
    from carbs.guidance import ELQROSPA
    from serums.distances import SingleObjectDistance

    d2r = np.pi / 180

    tt = 0  # starting time when calculating control
    dt = 0.01
    time_horizon = 150 * dt  # run for 120 timesteps

    # define starting and ending state for control calculation
    end_states = np.array(
        [[1.5, 2, 90 * d2r], [0, 0.75, 90 * d2r], [-1.5, 2, 90 * d2r]]
    )
    start_states = np.array([[-1, -0.15, 90 * d2r], [1, -0.15, 90 * d2r]])

    # define nominal control input
    u_nom = np.zeros((2, 1))

    # define dynamics
    # IRobot Create has a dt that can be set here or it can be set by the control
    # algorithm
    dynObj = IRobotCreate(wheel_separation=0.258, radius=0.335 / 2.0)

    # define Q and R weights for using standard cost function
    Q = np.diag([1500, 1000, 1000])
    R = 0.4 * np.eye(u_nom.size)

    # define modifications for quadratizing the cost function
    def quad_modifier(itr, tt, P, Q, R, q, r):
        rot_cost = 0.4
        # only modify if within the first 2 iterations
        if itr < 2:
            Q[-1, -1] = rot_cost
            q[-1] = -rot_cost * np.pi / 2

        return P, Q, R, q, r

    # create control obect
    singleELQR = gcontrol.ELQR(time_horizon=time_horizon)
    singleELQR.set_state_model(u_nom, dynObj=dynObj)
    singleELQR.dt = dt  # set here or within the dynamic object
    singleELQR.set_cost_model(Q=Q, R=R)  # can also set the quad modifier here

    # define some circular obstacles with center pos and radius (x, y, radius)
    obstacles = np.array(
        [
            [0, -1.35, 0.2],
            [1.0, -0.5, 0.2],
            [-0.95, -0.5, 0.2],
            [-0.2, 0.3, 0.2],
            [0.8, 0.7, 0.2],
            [1.1, 2.0, 0.2],
            [-1.2, 0.8, 0.2],
            [-1.1, 2.1, 0.2],
            [-0.1, 1.6, 0.2],
            [-1.1, -1.9, 0.2],
            [(10 + np.sqrt(2)) / 10, (-15 - np.sqrt(2)) / 10, 0.2],
        ]
    )

    # define enviornment bounds for the robot
    bottom_left = np.array([-2, -3])
    top_right = np.array([2, 3])

    # define additional non-quadratic cost
    obs_factor = 1
    scale_factor = 1
    cost_args = (obstacles, obs_factor, scale_factor, bottom_left, top_right)

    def non_quadratic_cost(
        tt,
        state,
        ctrl_input,
        end_state,
        is_initial,
        is_final,
        _obstacles,
        _obs_factor,
        _scale_factor,
        _bottom_left,
        _top_right,
    ):
        cost = 0
        # cost for obstacles
        for obs in _obstacles:
            diff = state.ravel()[0:2] - obs[0:2]
            dist = np.sqrt(np.sum(diff * diff))
            # signed distance is negative if the robot is within the obstacle
            signed_dist = (dist - dynObj.radius) - obs[2]
            if signed_dist > 0:
                continue
            cost += _obs_factor * np.exp(-_scale_factor * signed_dist).item()

        # add cost for going out of bounds
        for ii, b in enumerate(_bottom_left):
            dist = (state[ii] - b) - dynObj.radius
            cost += _obs_factor * np.exp(-_scale_factor * dist).item()

        for ii, b in enumerate(_top_right):
            dist = (b - state[ii]) - dynObj.radius
            cost += _obs_factor * np.exp(-_scale_factor * dist).item()

        return cost

    # create guidance object
    pos_inds = [0, 1]
    cutoff = 10

    elqr = ELQROSPA()
    elqr.set_control_model(singleELQR)
    elqr.set_cost_model(
        quad_modifier=quad_modifier,
        ospa_inds=pos_inds,
        ospa_cutoff=cutoff,
        ospa_method=SingleObjectDistance.EUCLIDEAN,
        non_quad_fun=non_quadratic_cost,
        non_quad_weight=10,
    )

    # create figure with obstacles to plot animation on
    cmap = gplot.get_cmap(start_states.shape[0])
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    fig.axes[0].set_aspect("equal", adjustable="box")
    fig.axes[0].set_xlim((bottom_left[0], top_right[0]))
    fig.axes[0].set_ylim((bottom_left[1], top_right[1]))
    for c_ind, start_state in enumerate(start_states):
        fig.axes[0].scatter(
            start_state[pos_inds[0]],
            start_state[pos_inds[1]],
            marker="o",
            color=cmap(c_ind),
            zorder=1000,
        )
    for obs in obstacles:
        c = Circle(obs[:2], radius=obs[2], color="k", zorder=1000)
        fig.axes[0].add_patch(c)
    plt_opts = gplot.init_plotting_opts(f_hndl=fig)
    gplot.set_title_label(
        fig, 0, plt_opts, ttl="Multi-Agent OSPA Based ELQR w/ Obstacles"
    )

    # calculate guidance
    (
        state_trajectories,
        costs,
        control_signals,
        fig,
        frame_list,
    ) = elqr.plan(
        tt,
        start_states,
        end_states,
        cost_args=cost_args,
        provide_details=True,
        show_animation=True,
        save_animation=True,
        plt_inds=pos_inds,
        cmap=cmap,
        fig=fig,
    )

    return frame_list


def run():
    import os

    print("Generating ELQR examples")
    fps = 10
    duration = int(1 / fps * 1e3)

    fout = os.path.join(os.path.dirname(__file__), "elqr_modify_quadratize.gif")
    if not os.path.isfile(fout):
        frame_list = modify_quadratize()

        frame_list[0].save(
            fout,
            save_all=True,
            append_images=frame_list[1:],
            duration=duration,  # convert s to ms
            loop=0,
        )

    fout = os.path.join(os.path.dirname(__file__), "elqr_ospa_cost.gif")
    if not os.path.isfile(fout):
        frame_list = elqr_ospa()

        frame_list[0].save(
            fout,
            save_all=True,
            append_images=frame_list[1:],
            duration=duration,  # convert s to ms
            loop=0,
        )

    fout = os.path.join(os.path.dirname(__file__), "elqr_ospa_cost_obstacles.gif")
    if not os.path.isfile(fout):
        frame_list = elqr_ospa_obstacles()

        frame_list[0].save(
            fout,
            save_all=True,
            append_images=frame_list[1:],
            duration=duration,  # convert s to ms
            loop=0,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.close("all")

    run()

    plt.show()
