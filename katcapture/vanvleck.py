"""Van Vleck (quantisation) correction."""

import numpy as np
import scipy.stats
import scipy.optimize
import scikits.fitting

def quant_norm_pmf(levels, mean=0.0, var=1.0):
    """Probability mass function of quantised normal variable."""
    edges = np.r_[-np.inf, levels[:-1] + np.diff(levels) / 2., np.inf]
    return np.diff(scipy.stats.norm.cdf(edges, loc=mean, scale=np.sqrt(var)))

def avg_squared_quant_norm_meanstd(levels, N, mean=0.0, var=1.0):
    """Mean and standard deviation of averaged squared quantised normal variable."""
    levels = np.asarray(levels)
    pmf = quant_norm_pmf(levels, mean, var)
    squared_levels = levels ** 2
    squared_quant_mean = np.dot(squared_levels, pmf)
    squared_quant_var = np.dot((squared_levels - squared_quant_mean) ** 2, pmf)
    return squared_quant_mean, np.sqrt(squared_quant_var / N)

def create_correction(N):
    """Produce Van Vleck correction functions for dumps containing N accumulations."""
    # 15-level "4-bit" ROACH requantiser
    levels = np.arange(-7., 8.)

    # Sweep across range of input power values (this is x = sigma ** 2, the quantity we want know)
    x_grid = np.logspace(-2, 6, 1600) ** 2
    # Obtain corresponding output power values (this is the measurement y)
    y_mean, y_std = np.array([avg_squared_quant_norm_meanstd(levels, N, var=x) for x in x_grid]).T
    # Avoid standard deviations that are too small (they will get cubed in log_joint_pdf_dx2)
    valid_std = y_std ** 3 > 0
    x_grid, y_mean, y_std = x_grid[valid_std], y_mean[valid_std], y_std[valid_std]

    # Fit input-output relationship (both in mean and uncertainty)
    y_mean_spline = scikits.fitting.Spline1DFit(s=0)
    y_mean_spline.fit(x_grid, y_mean)
    y_std_spline = scikits.fitting.Spline1DFit(s=0)
    y_std_spline.fit(x_grid, y_std)

    def log_joint_pdf(x, y, prior='uniform'):
        """Log pdf height of unnormalised joint distribution p(x, y)."""
        f, g = y_mean_spline(x), np.abs(y_std_spline(x))
        a, b = 1 / g, f / g
        log_prior = - np.log(x) if prior == 'jeffreys' else 0.0
        return -0.5 * (a * y - b) ** 2 + np.log(a) + log_prior

    def log_joint_pdf_dx(x, y, prior='uniform'):
        """First derivative of log pdf height of unnormalised joint distribution p(x, y)."""
        f, g = y_mean_spline(x), np.abs(y_std_spline(x))
        df_dx, dg_dx = y_mean_spline._interp(x, nu=1), y_std_spline._interp(x, nu=1)
        a, b = 1 / g, f / g
        da_dx, db_dx = -dg_dx / (g ** 2), (df_dx * g - f * dg_dx) / (g ** 2)
        dprior_dx = - 1 / x if prior == 'jeffreys' else 0.0
        return - (a * y - b) * (da_dx * y - db_dx) + da_dx / a + dprior_dx

    def log_joint_pdf_dx2(x, y, prior='uniform'):
        """Second derivative of log pdf height of unnormalised joint distribution p(x, y)."""
        f, g = y_mean_spline(x), np.abs(y_std_spline(x))
        df_dx, dg_dx = y_mean_spline._interp(x, nu=1), y_std_spline._interp(x, nu=1)
        d2f_dx2, d2g_dx2 = y_mean_spline._interp(x, nu=2), y_std_spline._interp(x, nu=2)
        a, b = 1 / g, f / g
        da_dx, db_dx = -dg_dx / (g ** 2), (df_dx * g - f * dg_dx) / (g ** 2)
        # Simplify second derivatives, mainly to reduce the g ** 4 factor to limit numerical underflow
        d2a_dx2 = - d2g_dx2 / (g ** 2) + 2 * (dg_dx ** 2) / (g ** 3)
        d2b_dx2 = d2f_dx2 / g - 2 * df_dx * dg_dx / (g ** 2) - f * d2g_dx2 / (g ** 2) + 2 * f * (dg_dx ** 2) / (g ** 3)
        d2prior_dx2 = -1 / (x ** 2) if prior == 'jeffreys' else 0.0
        return - (da_dx * y - db_dx) ** 2 - (a * y - b) * (d2a_dx2 * y - d2b_dx2) + \
               (d2a_dx2 * a - da_dx ** 2) / (a ** 2) + d2prior_dx2

    # Now work backwards from measurement y to estimate of input x
    # Use a Bezier curve to get a smooth S shape passing through 0 and 1 which concentrates more points at grid edges
    y_max = np.abs(levels).max()
    y_grid_len = 500
    y_grid_control = np.array([[0., 2 / 3., 1 / 3., 1.], [0.0, 0.0, 1.0, 1.0]])
    y_grid_control[0] *= y_grid_len - 1
    y_grid_control[1] *= y_max ** 2
    # Bernstein basis polynomials of Bezier curve
    bern = lambda x: np.array([(1 - x) ** 3, 3 * x * (1 - x) ** 2, 3 * (1 - x) * x ** 2, x ** 3])
    def ygrid_bezier(x):
        """Bezier curve y-coordinate (y_grid) as a function of x-coordinate (index)."""
        if np.iterable(x):
            t = np.array([scipy.optimize.brentq(lambda t: np.dot(y_grid_control[0], bern(t)) - xx, 0, 1) for xx in x])
        else:
            t = scipy.optimize.brentq(lambda t: np.dot(y_grid_control[0], bern(t)) - x, 0, 1)
        return np.dot(y_grid_control[1], bern(t))
    # Create grid of output power values, used to set up a spline for the inverse mapping from y to x
    y_grid = ygrid_bezier(np.arange(1, y_grid_len - 2))

    # Choice of prior should not matter much
    prior = 'uniform'
    # Joint pdf between input and output, P(x, y), used to obtain initial guess for correction function
    joint = np.array([log_joint_pdf(x, y_grid, prior) for x in x_grid])
    # Use Laplace's method to get stats of posterior pdf P(x | y), by finding peak in P(x, y) as function of x
    x_mean_guess = x_grid[np.argmax(joint, axis=0)]
    x_mean = np.array([scipy.optimize.brentq(lambda x: log_joint_pdf_dx(x, y, prior), 0.9 * x0, 1.1 * x0)
                       for x0, y in zip(x_mean_guess, y_grid)])
    x_std = np.array([1 / np.sqrt(-log_joint_pdf_dx2(x, y, prior)) for x, y in zip(x_mean, y_grid)])

    # Actual Van Vleck correction function (both mean and uncertainty)
    x_mean_spline = scikits.fitting.Spline1DFit(s=0)
    x_mean_spline.fit(y_grid, x_mean)
    x_std_spline = scikits.fitting.Spline1DFit(s=0)
    x_std_spline.fit(y_grid, x_std)

    return x_mean_spline, x_std_spline
