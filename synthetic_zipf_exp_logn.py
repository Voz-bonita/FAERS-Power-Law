import pandas as pd
import numpy as np
from sampler import sample_exponential, sample_lognormal, sample_zipf
from visualize import format_plot
from mc import mc_single
import matplotlib.pyplot as plt
from scipy.stats import ecdf
from tqdm import tqdm


def plot_cdfs():
    XMIN = 15
    S_SIZE = int(1e3)

    smp_exp = sample_exponential(S_SIZE, 0.125, XMIN)
    smp_exp_cdf = ecdf(smp_exp).cdf
    smp_logn = sample_lognormal(S_SIZE, 0.3, 2, XMIN)
    smp_logn_cdf = ecdf(smp_logn).cdf
    smp_zipf = sample_zipf(S_SIZE, 2.5, XMIN)
    smp_zipf_cdf = ecdf(smp_zipf).cdf

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    ax.scatter(
        smp_exp_cdf.quantiles,
        1 - smp_exp_cdf.probabilities,
        color="red",
        label=f"Exp(0.125; {XMIN})",
        s=2,
    )
    ax.scatter(
        smp_logn_cdf.quantiles,
        1 - smp_logn_cdf.probabilities,
        color="white",
        label=f"Zipf(2.5; {XMIN})",
        s=2,
    )
    ax.scatter(
        smp_zipf_cdf.quantiles,
        1 - smp_zipf_cdf.probabilities,
        color="orange",
        label=f"Lognormal(0.3, 2; {XMIN})",
        s=2,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$P(X \geq x)$")
    format_plot(ax, "white")
    plt.legend()
    plt.savefig("assets/synthetic_cdf.png", transparent=True)
    plt.close()


def mc_p_value():
    R_inner = 1
    R_outer = 100
    sample_sizes = np.array(
        [np.arange(10**i, 10 ** (i + 1), 10**i) for i in range(1, 4)]
    ).flatten()

    zipf_p_value_data = {"N": [], "KS": [], "Renyi": []}
    exponential_p_value_data = {"N": [], "KS": [], "Renyi": []}
    lognormal_p_value_data = {"N": [], "KS": [], "Renyi": []}

    def update_p_value_data(smp, p_value_data, xmin):
        temp = {"KS": [], "Renyi": []}
        for _ in range(R_inner):
            mc_result = mc_single(smp, xmin=xmin, renyi_alpha=0.5)
            temp["KS"].append(mc_result["KS"]["greater"])
            temp["Renyi"].append(mc_result["Renyi"]["greater"])

        p_value_data["N"].append(len(smp))
        p_value_data["KS"].append(np.mean(temp["KS"]))
        p_value_data["Renyi"].append(np.mean(temp["Renyi"]))

    XMIN = 15
    for size in tqdm(sample_sizes):
        for _ in range(R_outer):
            sample_zf = sample_zipf(n=size, s=2.5, xmin=XMIN)
            update_p_value_data(sample_zf, zipf_p_value_data, XMIN)

            sample_exp = sample_exponential(n=size, lamb=0.125, xmin=XMIN)
            update_p_value_data(sample_exp, exponential_p_value_data, XMIN)

            sample_logn = sample_lognormal(n=size, mu=0.3, sigma=2, xmin=XMIN)
            update_p_value_data(sample_logn, lognormal_p_value_data, XMIN)

    df_zipf = pd.DataFrame(zipf_p_value_data).groupby("N")[["KS", "Renyi"]].mean()
    df_exp = pd.DataFrame(exponential_p_value_data).groupby("N")[["KS", "Renyi"]].mean()
    df_logn = pd.DataFrame(lognormal_p_value_data).groupby("N")[["KS", "Renyi"]].mean()

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    ax.plot(
        df_exp.index, df_exp.KS, color="red", label=f"KS Exp(0.125; {XMIN})", marker="o"
    )
    ax.plot(
        df_exp.index,
        df_exp.Renyi,
        color="grey",
        label=f"Rényi Exp(0.125; {XMIN})",
        marker="o",
    )
    ax.plot(
        df_zipf.index, df_zipf.KS, color="white", label=f"Zipf(2.5; {XMIN})", marker="o"
    )
    ax.plot(
        df_logn.index,
        df_logn.KS,
        color="orange",
        label=f"Lognormal(0.3, 2; {XMIN})",
        marker="o",
    )
    ax.set_xscale("log")
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"P-valor Médio")
    format_plot(ax, "white")
    plt.legend()
    plt.savefig("assets/synthetic_p_value_ks_renyi.png", transparent=True)
    plt.show()


def main():
    np.random.seed(0)
    plot_cdfs()
    mc_p_value()


if __name__ == "__main__":
    main()
