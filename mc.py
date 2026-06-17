import numpy as np
from zipf import zipf_cdf, fit_powerlaw_mle
from sampler import sample_zipf


def renyi_divergence(p_emp, p_model, alpha: float = 2.0):
    eps = 1e-15

    p = np.maximum(p_emp, eps)
    q = np.maximum(p_model, eps)

    return 1 / (alpha - 1) * np.log(np.sum(p**alpha * q ** (1 - alpha)))


def step(smp, xmin, renyi_alpha: float = 2.0):
    freqs, support = np.histogram(smp, bins=20)
    smp_pmf = freqs / freqs.sum()

    # Fit observed data
    s_hat = fit_powerlaw_mle(smp, xmin)[0]
    cdf_hat = zipf_cdf(support, s_hat, xmin)[1:]
    cdf_hat = cdf_hat / cdf_hat[-1]
    pmf_hat = [cdf_hat[0], *np.diff(cdf_hat)]

    obs_ks = np.max(np.abs(smp_pmf.cumsum() - cdf_hat))
    obs_renyi = renyi_divergence(smp_pmf, pmf_hat, renyi_alpha)

    return s_hat, obs_ks, obs_renyi


def mc_single(smp, xmin=1, renyi_alpha: float = 2.0):
    N = len(smp)

    s_hat, obs_ks, obs_renyi = step(smp, xmin, renyi_alpha)

    smp_syn = sample_zipf(n=N, s=s_hat, xmin=xmin)
    _, ks, renyi = step(smp_syn, xmin, renyi_alpha)

    return {
        "s_hat": s_hat,
        "KS": {"observed": obs_ks, "greater": int(ks >= obs_ks)},
        "Renyi": {"observed": obs_renyi, "greater": int(renyi >= obs_renyi)},
    }
