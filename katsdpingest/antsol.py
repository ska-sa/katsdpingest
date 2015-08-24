"""Antenna-based gain solvers."""

import logging

import numpy as np

logger = logging.getLogger("kat.katsdpingest.antsol")


def stefcal(vis, num_ants, antA, antB, weights=1.0, num_iters=30, ref_ant=-1,
            init_gain=None, conv_thresh=0.0001):
    """Solve for antenna gains using StefCal (array dot product version).

    The observed visibilities are provided in a NumPy array of any shape and
    dimension, as long as the last dimension represents baselines. The gains
    are then solved in parallel for the rest of the dimensions. For example,
    if the *vis* array has shape (T, F, B) containing *T* dumps / timestamps,
    *F* frequency channels and *B* baselines, the resulting gain array will be
    of shape (T, F, num_ants), where *num_ants* is the number of antennas.

    In order to get a proper solution it is important to include the conjugate
    visibilities as well by reversing antenna pairs, e.g. by forming

    full_vis = np.concatenate((vis, vis.conj()), axis=-1)
    full_antA = np.r_[antA, antB]
    full_antB = np.r_[antB, antA]

    Parameters
    ----------
    vis : array of complex, shape (M, ..., N)
        Complex cross-correlations between antennas A and B, assuming *N*
        baselines or antenna pairs on the last dimension
    num_ants : int
        Number of antennas
    antA, antB : array of int, shape (N,)
        Antenna indices associated with visibilities
    weights : float or array of float, shape (M, ..., N), optional
        Visibility weights (positive real numbers)
    num_iters : int, optional
        Number of iterations
    ref_ant : int, optional
        Reference antenna for which phase will be forced to 0.0. Alternatively,
        if *ref_ant* is -1, the median gain phase will be 0.
    init_gain : array of complex, shape(num_ants,) or None, optional
        Initial gain vector (all equal to 1.0 by default)
    conv_thresh : float, optional
        Convergence threshold (max relative l_2 norm of change in gain vector)

    Returns
    -------
    gains : array of complex, shape (M, ..., num_ants)
        Complex gains per antenna

    Notes
    -----
    The model visibilities are assumed to be 1, implying a point source model.

    The algorithm is iterative but should converge in a small number of
    iterations (10 to 30).

    References
    ----------
    .. [1] Salvini, Wijnholds, "Fast gain calibration in radio astronomy using
       alternating direction implicit methods: Analysis and applications," 2014,
       preprint at `<http://arxiv.org/abs/1410.2101>`_

    """
    # Each row of this array contains the indices of baselines with the same antA
    baselines_per_antA = np.array([(antA == m).nonzero()[0] for m in range(num_ants)])
    # Each row of this array contains the corresponding antB indices with the same antA
    antB_per_antA = antB[baselines_per_antA]
    weighted_vis = weights * vis
    weighted_vis = weighted_vis[..., baselines_per_antA]
    # Initial estimate of gain vector
    gain_shape = tuple(list(vis.shape[:-1]) + [num_ants])
    g_curr = np.ones(gain_shape, dtype=np.complex) if init_gain is None else init_gain
    logger.debug("StefCal solving for %s gains from vis with shape %s" %
                 ('x'.join(str(gs) for gs in gain_shape), vis.shape))
    for n in range(num_iters):
        # Basis vector (collection) represents gain_B* times model (assumed 1)
        g_basis = g_curr[..., antB_per_antA]
        # Do scalar least-squares fit of basis vector to vis vector for whole collection in parallel
        g_new = (g_basis * weighted_vis).sum(axis=-1) / (g_basis.conj() * g_basis).sum(axis=-1)
        # Get gains for reference antenna (or first one, if none given)
        g_ref = g_new[..., max(ref_ant, 0)][..., np.newaxis].copy()
        # Force reference gain to have zero phase
        g_new *= np.abs(g_ref) / g_ref
        logger.debug("Iteration %d: mean absolute gain change = %f" %
                     (n + 1, 0.5 * np.abs(g_new - g_curr).mean()))
        # Salvini & Wijnholds tweak gains every *even* iteration but their counter starts at 1
        if n % 2:
            # Check for convergence of relative l_2 norm of change in gain vector
            error = np.linalg.norm(g_new - g_curr, axis=-1) / np.linalg.norm(g_new, axis=-1)
            if np.max(error) < conv_thresh:
                g_curr = g_new
                break
            else:
                # Avoid getting stuck bouncing between two gain vectors by going halfway in between
                g_curr = 0.5 * (g_new + g_curr)
        else:
            # Normal update every *odd* iteration
            g_curr = g_new
    if ref_ant < 0:
        middle_angle = np.arctan2(np.median(g_curr.imag, axis=-1),
                                  np.median(g_curr.real, axis=-1))
        g_curr *= np.exp(-1j * middle_angle)[..., np.newaxis]
    return g_curr


# Quick test of StefCal by running this as a script
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    M = 100 # Number of dumps
    N = 7 # Number of antennas / inputs
    K = 10000 # Number of integrations per dump
    noise_power = 10.0 # Compared to signal power of 1.0
    ref_ant = 3

    # We want lower triangle, but use upper triangle indices instead because
    # of 'C' ordering of arrays
    vech = np.triu_indices(N, 1)
    antA, antB = np.meshgrid(np.arange(N), np.arange(N))
    antA, antB = antA[vech], antB[vech]
    # Include both triangles when solving (i.e. visibility + its conjugate)
    antA, antB = np.r_[antA, antB], np.r_[antB, antA]

    # Generate random gain vector with magnitudes around 1 and random phase
    abs_g = np.abs(1 + 0.1 * np.random.randn(N))
    g = abs_g * np.exp(2j * np.pi * np.random.rand(N))
    # Fix reference gain to have magnitude 1 and phase 0
    g_norm = g / g[ref_ant]

    # Assume simple point source model + diagonal noise contribution
    ggH = np.outer(g, g.conj())
    Rn = noise_power * np.eye(N)
    R = ggH + Rn

    vis = np.zeros((M, len(antA)), dtype=np.complex)
    for m in range(M):
        # Generate random sample covariance matrix V from true covariance R
        L = np.linalg.cholesky(R)
        X = np.random.randn(R.shape[0], K) + 1.0j * np.random.randn(R.shape[0], K)
        V = 0.5 * np.dot(X, X.conj().transpose())
        V = np.dot(L, np.dot(V, L.conj().transpose())) / K
        # Extract cross-correlations from covariance matrix and stack them into vector
        vis[m] = V[(antA, antB)]

    print 'Testing StefCal:\n----------------'
    g_estm = stefcal(vis, N, antA, antB, num_iters=100, ref_ant=ref_ant)
    compare = '\n'.join([("%+5.3f%+5.3fj -> %+5.3f%+5.3fj" %
                          (gt.real, gt.imag, ge.real, ge.imag))
                         for gt, ge in np.c_[g_norm, g_estm.mean(axis=0)]])
    print '\nOriginal gain -> Mean estimated gain vector:\n' + compare
    print 'StefCal mean absolute gain error = %f' % \
          (np.abs(g_estm.mean(axis=0) - g_norm).mean(),)
