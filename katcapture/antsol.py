"""Antenna-based gain solvers."""

import logging

import numpy as np

logger = logging.getLogger("antsol")


def stefcal(vis, num_ants, antA, antB, weights=1.0, num_iters=10, ref_ant=-1, init_gain=None):
    """Solve for antenna gains using StefCal.

    Parameters
    ----------
    vis : array of complex, shape (N,)
        Complex cross-correlations between antennas A and B
    num_ants : int
        Number of antennas
    antA, antB : array of int, shape (N,)
        Antenna indices associated with visibilities
    weights : float or array of float, shape (N,), optional
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
    gains : array of complex, shape (num_ants,)
        Complex gains, one per antenna

    """
    # Initialise design matrix for solver
    g_prev = np.zeros((len(vis), num_ants), dtype=np.complex)
    rows = np.arange(len(vis))
    weighted_vis = weights * vis
    weights = np.atleast_2d(weights).T
    # Initial estimate of gain vector
    g_curr = np.ones(num_ants, dtype=np.complex) if init_gain is None else init_gain
    for n in range(num_iters):
        # Insert current gain into design matrix as gain B
        g_prev[rows, antA] = g_curr[antB].conj()
        # Solve for gain A
        g_new = np.linalg.lstsq(weights * g_prev, weighted_vis)[0]
        # Normalise g_new to match g_curr so that taking their average and difference make sense
        g_new /= (g_new[ref_ant] if ref_ant >= 0 else g_new[0])
        logger.debug("Iteration %d: mean absolute gain change = %f" % (n + 1, 0.5 * np.abs(g_new - g_curr).mean()))
        # Avoid getting stuck
        g_curr = 0.5 * (g_new + g_curr)
    if ref_ant < 0:
        g_curr /= (np.mean(np.abs(g_curr)) * np.exp(1j * np.arctan2(np.median(g_curr.imag), np.median(g_curr.real))))
    return g_curr
