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
