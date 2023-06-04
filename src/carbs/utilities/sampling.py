"""Implements general sampling methods."""
import numpy as np


def gibbs(in_costs, num_iters, rng=None):
    """Implements a Gibbs sampler.

    Notes
    -----
    Computationally efficient form of the Metropolis Hastings Markov Chain
    Monte Carlo algorithm, useful when direct sampling of the target
    multivariate distribution is difficult. Generates samples by selecting
    from a histogram constructed from the distribution and randomly sampling
    from the most likely bins in the histogram. This is based on the sampler in
    :cite:`Vo2017_AnEfficientImplementationoftheGeneralizedLabeledMultiBernoulliFilter`.

    Parameters
    ----------
    in_costs : N x M numpy array
        Cost matrix.
    num_iters : int
        Number of iterations to run.
    rng : numpy random generator, optional
        Random number generator to be used when sampling. The default is None
        which implies :code:`default_rng()`

    Returns
    -------
    assignments : M (max size) x N numpy array
        The unique entries from the sampling.
    costs : M x 1 numpy array
        Cost of each assignment.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determine size of cost matrix
    cost_size = np.shape(in_costs)

    # Initialize assignment and cost matrices
    assignments = np.zeros((num_iters, cost_size[0]))
    costs = np.zeros((num_iters, 1))

    # Initialize current solution
    cur_soln = np.arange(cost_size[0], 2 * cost_size[0])

    assignments[0] = cur_soln
    rows = np.arange(0, cost_size[0]).astype(int)
    costs[0] = np.sum(in_costs[rows, cur_soln])

    # Loop over all possible assignments and determine costs
    mask = np.ones(len(cur_soln), dtype=bool)
    for sol in range(1, num_iters):
        for var in range(0, cost_size[0]):
            if var > 0:
                mask[var - 1] = True
            mask[var] = False
            temp_samp = np.exp(-in_costs[var])
            temp_samp[cur_soln[mask]] = 0

            hist_in_array = np.zeros(temp_samp[temp_samp > 0].size + 1)
            csum = np.cumsum(temp_samp[temp_samp > 0].ravel())
            hist_in_array[1:] = csum / csum[-1]

            cur_soln[var] = np.digitize(rng.uniform(size=(1,1)), hist_in_array) - 1
            if np.any(temp_samp > 0):
                cur_soln[var] = np.nonzero(temp_samp > 0)[0][cur_soln[var]]

        mask[-1] = True
        assignments[sol] = cur_soln
        costs[sol] = np.sum(in_costs[rows, cur_soln])

    [assignments, I] = np.unique(assignments, return_index=True, axis=0)

    return assignments, costs[I]

# TODO: Multisensor Gibbs potentially
def mm_gibbs(in_costs, num_iters, rng=None):
    """Implements a minimally-Markovian Gibbs sampler.

        Notes
        -----
        Computationally efficient form of the Metropolis Hastings Markov Chain
        Monte Carlo algorithm, useful when direct sampling of the target
        multivariate distribution is difficult. Generates samples by selecting
        from a histogram constructed from the distribution and randomly sampling
        from the most likely bins in the histogram. This is based on the sampler in
        :cite:`Vo2017_AnEfficientImplementationoftheGeneralizedLabeledMultiBernoulliFilter`.

        Parameters
        ----------
        in_costs : S x N x M numpy array
            Cost matrix.
        num_iters : int
            Number of iterations to run.
        rng : numpy random generator, optional
            Random number generator to be used when sampling. The default is None
            which implies :code:`default_rng()`

        Returns
        -------
        assignments : M (max size) x N x S numpy array
            The unique entries from the sampling.
        costs : M x S numpy array
            Cost of each assignment for each sensor.
        """

    if rng is None:
        rng = np.random.default_rng()

    # Determine size of cost matrix
    cost_size = np.shape(in_costs)

    # Initialize assignment and cost matrices
    #assignments is iterations x sensors x solution
    assignments = np.zeros((num_iters, cost_size[0], cost_size[1]))
    costs = np.zeros((num_iters, 1))
    P_n = []
    Q_n = []
    #TODO: calculation for P_n and Q_n
    for var in range(0, cost_size[1]):
        prod_gamma_n = 1
        prod_v_n = 1
        for s in range(0, cost_size[0]):
            gamma_n = np.sum(in_costs[s, var][in_costs[s, var] != -1]) #third index needs to represent -1 vs non neg
            prod_gamma_n = prod_gamma_n*gamma_n
            prod_v_n = prod_v_n * np.sum(in_costs[s, var][in_costs[s, var] == -1]) # index where in_costs are -1
        P_n.append(prod_gamma_n/(prod_v_n + prod_gamma_n))
        Q_n.append(1 - P_n[var])

    # Initialize current solution
    costsum = 0
    rows = np.arange(0, cost_size[1]).astype(int)
    cur_soln = np.zeros((cost_size[0], len(np.arange(cost_size[1], 2*cost_size[1]))),dtype=int)
    for s in range(0, cost_size[0]):
        cur_soln[s, :] = np.arange(cost_size[1], 2 * cost_size[1])
        costsum += np.sum(in_costs[s, rows, np.arange(cost_size[1], 2 * cost_size[1]).astype(int)])

    assignments[0, :, :] = cur_soln
    costs[0] = costsum

    # Loop over all possible assignments and determine costs
    mask = np.ones(np.shape(cur_soln), dtype=bool)
    for sol in range(1, num_iters):
        costsum = 0
        for var in range(0, cost_size[1]):
            #TODO: write first categorical
            # sample markovian stationary distribution,
            # if greater than some probability as calculated above by pn,
            # then perform mm-gibbs, otherwise all values are set to -1.
            #
            bins = np.array([0.0, Q_n[var], 1.0])
            i_n = np.digitize(rng.uniform(size=(1, 1)), bins) - 1
            "i_n = Categorical ('+', '-', ), [Pn(Lambda(1:num_sensors)), Qn(Lambda(1:num_sensors)]"
            if i_n > 0: # i_n = "+"

                for s in range(0, cost_size[0]):
                    if var > 0:
                        mask[s, var - 1] = True
                    mask[s, var] = False
                temp_samp = np.exp(-in_costs[:, var, :])
                temp_samp[cur_soln[mask]] = 0
                for s in range(0, cost_size[0]):
                    hist_in_array = np.zeros(temp_samp[s][temp_samp[s] > 0].size + 1)
                    csum = np.cumsum(temp_samp[s][temp_samp[s] > 0].ravel())
                    hist_in_array[1:] = csum / csum[-1]
                    cur_soln[s, var] = np.digitize(rng.uniform(size=(1, 1)), hist_in_array) - 1
                    if np.any(temp_samp[s] > 0):
                        cur_soln[s, var] = np.nonzero(temp_samp > 0)[0][cur_soln[s, var]]

            else: # i_n = "-"
                cur_soln[:, var] = -1 * np.ones(1, cost_size[0])

        assignments[sol, :, :] = cur_soln
        costs[sol] = np.sum(in_costs[:, rows, cur_soln]) #some variant of this


    [assignments, I] = np.unique(assignments, return_index=True, axis=0)

    return assignments, costs[I]

