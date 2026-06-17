import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import zeta


def zipf_cdf(x, s, xmin=1, sup="continuous"):
    """
    CDF of the discrete Zipf distribution.

    Parameters
    ----------
    x : scalar or array-like
        Evaluation point(s).
    s : float
        Zipf exponent (must satisfy s > 1).
    xmin : int, optional
        Minimum support value.

    Returns
    -------
    F : scalar or ndarray
        P(X <= x).
    """
    x = np.asarray(x)

    if s <= 1:
        raise ValueError("s must be > 1.")
    if xmin < 1 or int(xmin) != xmin:
        raise ValueError("xmin must be a positive integer.")

    F = np.zeros_like(x, dtype=float)

    mask = x >= xmin
    if sup == "continuous":
        F[mask] = 1 - (x[mask] / xmin) ** (-s + 1)
    elif sup == "discrete":
        n = np.floor(x[mask]).astype(int)
        normalization = zeta(s, xmin)  # Hurwitz zeta
        F[mask] = 1.0 - zeta(s, n) / normalization
    else:
        raise ValueError("sup must be either 'discrete' or 'continuous'")

    return F


def zipf_pmf(x, s, xmin=1):
    """
    CDF of the discrete Zipf distribution.

    Parameters
    ----------
    x : scalar or array-like
        Evaluation point(s).
    s : float
        Zipf exponent (must satisfy s > 1).
    xmin : int, optional
        Minimum support value.

    Returns
    -------
    F : scalar or ndarray
        P(X <= x).
    """
    x = np.asarray(x)

    if s <= 1:
        raise ValueError("s must be > 1.")
    if xmin < 1 or int(xmin) != xmin:
        raise ValueError("xmin must be a positive integer.")

    return x ** (-s) / zeta(s, xmin)


def zipf_logpmf(x, s):
    """
    Log PMF of the Zipf distribution.

    P(X=x) = (x)^(-s) / zeta(s),
    x = 1,2,...

    Parameters
    ----------
    x : array-like
    s : float (>1)
    """
    return -s * np.log(x) - np.log(zeta(s))


def zipf_negloglik(s, counts, freq):
    """
    Negative grouped log-likelihood.
    """
    if s <= 1:
        return np.inf

    log_p = zipf_logpmf(counts, s)

    return -np.sum(freq * log_p)


def ks_statistic(p_emp, p_model):
    F_emp = np.cumsum(p_emp)
    F_mod = np.cumsum(p_model)

    return np.max(np.abs(F_emp - F_mod))


def zipf_negks(s, counts, p_emp):
    """
    Negative grouped log-likelihood.
    """
    if s <= 1:
        return np.inf

    log_p = zipf_logpmf(counts, s)

    return -ks_statistic(p_emp, np.exp(log_p))


def zipf_negks_cdf(s, counts, cdf):
    """
    Negative grouped log-likelihood.
    """
    if s <= 1:
        return np.inf

    log_p = zipf_logpmf(counts, s)

    return -np.max(np.abs(cdf, np.exp(log_p).cumsum()))


def fit_zipf_freq(counts, freq):
    """
    MLE fit for the Zipf distribution.
    """
    counts = np.asarray(counts, dtype=float)
    freq = np.asarray(freq, dtype=float)

    return minimize_scalar(
        zipf_negloglik, bounds=(1.0001, 10.0), method="bounded", args=(counts, freq)
    )


def fit_zipf_prop(counts, prop):
    """
    MLE fit for the Zipf distribution.
    """
    counts = np.asarray(counts, dtype=float)
    prop = np.asarray(prop, dtype=float)

    return minimize_scalar(
        zipf_negks, bounds=(1.0001, 10.0), method="bounded", args=(counts, prop)
    )


def fit_zipf_cdf(counts, cdf):
    """
    MLE fit for the Zipf distribution.
    """
    counts = np.asarray(counts, dtype=float)
    cdf = np.asarray(cdf, dtype=float)

    return minimize_scalar(
        zipf_negks_cdf, bounds=(1.0001, 10.0), method="bounded", args=(counts, cdf)
    )


def fit_powerlaw_mle(data, xmin=None):
    """
    Fit a continuous power-law distribution using MLE.

    Parameters
    ----------
    data : array-like
        Observations x_i.
    xmin : float, optional
        Lower cutoff. If None, xmin=min(data).

    Returns
    -------
    alpha : float
        Estimated power-law exponent.
    xmin : float
        Lower cutoff used.
    loglik : float
        Maximized log-likelihood.
    n : int
        Number of observations used.
    """
    data = np.asarray(data, dtype=float)

    if np.any(data <= 0):
        raise ValueError("All observations must be positive.")

    if xmin is None:
        xmin = np.min(data)

    # Only use observations in the tail
    tail = data[data >= xmin]

    n = len(tail)

    if n == 0:
        raise ValueError("No observations satisfy x >= xmin.")

    denominator = np.sum(np.log(tail / xmin))
    # denominator = np.sum(np.log(tail / (xmin - 1 / 2)))

    if denominator <= 0:
        raise ValueError("MLE is undefined for these data.")

    alpha = 1.0 + n / denominator

    loglik = n * np.log(alpha - 1) - n * np.log(xmin) - alpha * denominator

    return alpha, xmin, loglik, n
