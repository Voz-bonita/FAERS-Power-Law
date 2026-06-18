import pandas as pd
import numpy as np
from zipf import fit_powerlaw_mle
from sampler import sample_zipf
from tqdm import tqdm
from multiprocessing import Pool


def jitter(smp, xmin, eps=3e-1):
    eps = 3e-1
    n = len(smp)
    noise = np.random.uniform(-eps, eps, size=n)
    noise[smp + noise < xmin] = np.abs(noise[smp + noise < xmin])
    return smp + noise


def get_estimate(s):
    XMIN = 1
    N = 10_000
    R = 1_000
    estimates = {"s": [], "continuous": [], "discrete": [], "jitter": []}
    for _ in range(R):
        smp = sample_zipf(N, s, XMIN)
        if (smp > 1e19).sum() > 1:
            raise ValueError("Integer Overflow")
        smp = smp[smp <= 1e19]
        smp_discrete = np.round(smp).astype(int)
        smp_jitter = jitter(smp_discrete, XMIN)

        s_hat_continuous, _, _, _ = fit_powerlaw_mle(smp, XMIN)
        s_hat_discrete, _, _, _ = fit_powerlaw_mle(smp_discrete, XMIN)
        s_hat_jitter, _, _, _ = fit_powerlaw_mle(smp_jitter, XMIN)

        estimates["continuous"].append(s_hat_continuous)
        estimates["discrete"].append(s_hat_discrete)
        estimates["jitter"].append(s_hat_jitter)
        estimates["s"].append(s)

    return estimates


if __name__ == "__main__":
    N_s = 1_000
    s_smp = 1.5 + np.random.exponential(0.2, size=N_s)
    pool = Pool(6)
    out = []
    for o in tqdm(pool.imap_unordered(get_estimate, s_smp), total=N_s):
        out.append(o)

    pd.concat([pd.DataFrame(o) for o in out]).to_parquet("data/jitter_data.parquet")
