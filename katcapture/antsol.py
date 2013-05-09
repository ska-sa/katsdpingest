"""Antenna-based gain solvers."""

import logging

import numpy as np

logger = logging.getLogger("kat.antsol")


def stefcal(vis, num_ants, antA, antB, weights=1.0, num_iters=10, ref_ant=-1, init_gain=None):
    """Solve for antenna gains using StefCal (array dot product version).

    Parameters
    ----------
    vis : array of complex, shape (M, ..., N)
        Complex cross-correlations between antennas A and B
    num_ants : int
        Number of antennas
    antA, antB : array of int, shape (N,)
        Antenna indices associated with visibilities
    weights : float or array of float, shape (M, ..., N), optional
        Visibility weights (positive real numbers)
    num_iters : int, optional
        Number of iterations
    ref_ant : int, optional
        Reference antenna whose gain will be forced to be 1.0. Alternatively,
        if *ref_ant* is -1, the average gain magnitude will be 1 and the median
        gain phase will be 0.
    init_gain : array of complex, shape(num_ants,) or None, optional
        Initial gain vector (all equal to 1.0 by default)

    Returns
    -------
    gains : array of complex, shape (M, ..., num_ants)
        Complex gains per antenna

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
    for n in range(num_iters):
        # Basis vector (collection) represents gain_B* times model (assumed 1)
        g_basis = g_curr[..., antB_per_antA]
        # Do scalar least-squares fit of basis vector to vis vector for whole collection in parallel
        g_new = (g_basis * weighted_vis).sum(axis=-1) / (g_basis.conj() * g_basis).sum(axis=-1)
        # Normalise g_new to match g_curr so that taking their average and
        # difference make sense (without copy the elements of g_new are mangled up)
        g_new /= (g_new[..., ref_ant][..., np.newaxis].copy() if ref_ant >= 0 else
                  g_new[..., 0][..., np.newaxis].copy())
        logger.info("Iteration %d: mean absolute gain change = %f" % (n + 1, 0.5 * np.abs(g_new - g_curr).mean()))
        # Avoid getting stuck during iteration
        g_curr = 0.5 * (g_new + g_curr)
    if ref_ant < 0:
        avg_amp = np.mean(np.abs(g_curr), axis=-1)
        middle_angle = np.arctan2(np.median(g_curr.imag, axis=-1),
                                  np.median(g_curr.real, axis=-1))
        g_curr /= (avg_amp * np.exp(1j * middle_angle))[..., np.newaxis]
    return g_curr
