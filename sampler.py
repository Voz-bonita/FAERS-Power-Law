import numpy as np
from zipf import zipf_logpmf, zipf_pmf


def sample_lognormal(n, mu, sigma, xmin):
    samples = []
    while len(samples) < n:
        x = np.exp(np.random.normal(mu, sigma))
        if x >= xmin:
            samples.append(x)

    return np.array(samples)


def sample_exponential(n, lamb, xmin):
    u = np.random.uniform(size=n)
    return xmin - np.log(u) / lamb


def sample_zipf(n, s, xmin):
    # support = np.arange(xmin, xmin + 1e10)
    # p = zipf_pmf(support, s, xmin)
    # p = p / p.sum()
    # return np.random.choice(support, size=n, replace=True, p=p)
    u = np.random.rand(n)
    return xmin * (1 - u) ** (-1 / (s - 1))


def sample_to_freq(sample):
    counts, freqs = np.unique(sample, return_counts=True)
    return counts, freqs
