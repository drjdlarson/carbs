import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def modify_quadratize():
    import numpy as np

    import serums.models as smodels
    import gncpy.control as gcontrol
    from gncpy.dynamics.basic import IRobotCreate
    from caser.guidance import ELQR

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
    singleELQR.set_cost_model(Q=Q, R=R, quad_modifier=quad_modifier)

    # create guidance object
    safety_factor = 4
    y_ref = 0.9

    elqr = ELQR()
    elqr.set_control_model(singleELQR)

    # convert end locations into a distribution or "wayareas"
    end_dist = elqr.targets_to_wayareas(end_states)
    cost_args = (end_dist, safety_factor, y_ref)  # set args for non-quad cost

    # calculate guidance
    (state_trajectories, costs, control_signals, fig, frame_list,) = elqr.plan(
        tt,
        start_dist,
        end_dist,
        cost_args=cost_args,
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
    from caser.guidance import ELQROSPA
    from serums.distances import SingleObjectDistance

    d2r = np.pi / 180

    tt = 0  # starting time when calculating control
    dt = 0.01
    time_horizon = 90 * dt  # run for 90 timesteps

    # define starting and ending state for control calculation
    end_states = np.array(
        [[1.5, 2, 90 * d2r], [0, 2.5, 90 * d2r], [-1.5, 2, 90 * d2r]]
    )
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
    singleELQR.set_cost_model(Q=Q, R=R, quad_modifier=quad_modifier)

    # create guidance object
    pos_inds = [0, 1]
    cutoff = 10

    elqr = ELQROSPA()
    elqr.set_control_model(singleELQR)

    # set args for non-quad cost
    cost_args = (end_states, SingleObjectDistance.EUCLIDEAN, pos_inds, cutoff)

    # calculate guidance
    (state_trajectories, costs, control_signals, fig, frame_list,) = elqr.plan(
        tt,
        start_states,
        end_states,
        cost_args=cost_args,
        provide_details=True,
        show_animation=True,
        save_animation=True,
        plt_inds=pos_inds,
        ttl="Multi-Agent OSPA Based ELQR",
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


if __name__ == "__main__":
    plt.close("all")

    run()

    plt.show()
