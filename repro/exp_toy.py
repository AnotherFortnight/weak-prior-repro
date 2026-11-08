"""Figure 2 (posterior consistency, 2-D) and a numerical check of Theorem 3.2 (high-dimensional GMM prior).

Part 1: two well-separated 2-D Gaussian-mixture priors, y_i = A x* + eps_i (eps ~ N(0, I)), exact GMM posterior
        (Prop. A.2 with N i.i.d. observations = one observation of the mean with noise sigma^2/N).
Part 2: GMM prior whose means are 2000 real CelebA-HQ images (32x32 RGB, n = 3072), random inpainting with m observed
        entries. Two priors with different weights (ratio bound C = 10) -> posterior component weights; we report
        P(J != j*) and the total-variation distance between the two posteriors' component weights as m grows.
Run: python -s repro/exp_toy.py   (CPU, about 1 min)
"""
import io
import os

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "toy")

PRIORS = {
    "A": ([0.5, 0.5], [[-1.0, 2.6], [1.7, 1.2]], [[[0.08, 0.0], [0.0, 0.5]], [[0.5, 0.18], [0.18, 0.12]]]),
    "B": ([0.6, 0.4], [[-2.3, -1.7], [-1.0, -1.3]], [[[0.5, 0.05], [0.05, 0.08]], [[0.25, 0.08], [0.08, 0.08]]]),
}
COL = {"A": "#f4a582", "B": "#66c2a5"}


def gmm_posterior(w, mu, cov, A, ybar, s2):
    """Prop. A.2: posterior of x under GMM prior and ybar = A x + N(0, s2 I)."""
    w, mu, cov = np.asarray(w), np.asarray(mu), np.asarray(cov)
    logw, means, covs = [], [], []
    for wj, mj, Sj in zip(w, mu, cov):
        Sy = s2 * np.eye(len(ybar)) + A @ Sj @ A.T
        r = ybar - A @ mj
        logw.append(np.log(wj) - 0.5 * r @ np.linalg.solve(Sy, r) - 0.5 * np.linalg.slogdet(Sy)[1])
        C = np.linalg.inv(np.linalg.inv(Sj) + A.T @ A / s2)
        covs.append(C)
        means.append(C @ (np.linalg.solve(Sj, mj) + A.T @ ybar / s2))
    logw = np.array(logw)
    wt = np.exp(logw - logw.max())
    return wt / wt.sum(), np.array(means), np.array(covs)


def gmm_density(grid, w, mu, cov):
    d = 0
    for wj, mj, Sj in zip(w, mu, cov):
        r = grid - mj
        q = np.einsum("...i,ij,...j->...", r, np.linalg.inv(Sj), r)
        d = d + wj * np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(Sj)))
    return d


def part1(rng):
    x_star = np.array([0.6, -0.4])
    A = np.array([[1.0, 0.3], [-0.4, 0.9]])
    ys = x_star @ A.T + rng.standard_normal((1000, 2))
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    gx, gy = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-3.5, 4.5, 300))
    grid = np.stack([gx, gy], -1)
    for name, (w, mu, cov) in PRIORS.items():
        comp = rng.choice(len(w), 600, p=w)
        pts = np.array([rng.multivariate_normal(mu[c], cov[c]) for c in comp])
        ax[0].scatter(*pts.T, s=4, alpha=0.5, color=COL[name], label=f"Prior {name}")
        for k, N in enumerate([10, 50]):
            wt, m, C = gmm_posterior(w, mu, cov, A, ys[:N].mean(0), 1.0 / N)
            dens = gmm_density(grid, wt, m, C)
            ax[k + 1].contour(gx, gy, dens, levels=dens.max() * np.array([0.05, 0.3, 0.6, 0.9]), colors=COL[name])
    errs = {n: [] for n in PRIORS}
    Ns = np.unique(np.r_[np.logspace(0, 3, 25).astype(int), 10, 50])
    for N in Ns:
        for name, (w, mu, cov) in PRIORS.items():
            e = []
            for rep in range(100):
                y = x_star @ A.T + rng.standard_normal((N, 2))
                wt, m, _ = gmm_posterior(w, mu, cov, A, y.mean(0), 1.0 / N)
                e.append(np.linalg.norm(wt @ m - x_star))
            errs[name].append(np.mean(e))
    for k, title in enumerate(["Prior", "Posterior (N = 10)", "Posterior (N = 50)"]):
        ax[k].plot(*x_star, "k*", ms=10, label="True x*")
        ax[k].set(title=title, xlim=(-4, 4), ylim=(-3.5, 4.5), xlabel="Dim 1", ylabel="Dim 2")
    ax[0].legend(fontsize=7, loc="upper left")
    for name in PRIORS:
        ax[3].loglog(Ns, errs[name], "o-", ms=3, color=COL[name], label=f"Prior {name}")
    ax[3].loglog(Ns, 1.5 / np.sqrt(Ns), "k--", lw=1, label="1/sqrt(N)")
    ax[3].set(title="||E[x | y_1:N] - x*|| (100 reps)", xlabel="N")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_posterior_concentration.png"), dpi=150)
    plt.close(fig)
    return {n: (errs[n][list(Ns).index(10)], errs[n][list(Ns).index(50)], errs[n][-1]) for n in PRIORS}


def part2(rng, tau=0.1, sigma=0.05, C=10.0, trials=300):
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(ROOT, "data/celebahq_hf/data/validation-00000-of-00001.parquet"))
    mus = np.stack([np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB").resize((32, 32), Image.BOX),
                               dtype=np.float64).ravel() / 127.5 - 1 for r in t.column("image").to_pylist()])
    M, n = mus.shape
    wA = np.full(M, 1 / M)
    wB = np.exp(rng.uniform(-np.log(C) / 2, np.log(C) / 2, M))
    wB /= wB.sum()
    rows = []
    for m in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        miss = {"A": [], "B": []}
        tv, gaps, agree = [], [], []
        for _ in range(trials):
            j = rng.integers(M)
            om = rng.choice(n, m, replace=False)
            y = mus[j, om] + tau * rng.standard_normal(m) + sigma * rng.standard_normal(m)
            s = ((y - mus[:, om]) ** 2).sum(1) / (2 * (sigma ** 2 + tau ** 2))      # score (4), A = P_Omega
            post = {}
            for name, w in (("A", wA), ("B", wB)):
                lw = np.log(w) - s
                p = np.exp(lw - lw.max())
                post[name] = p / p.sum()
                miss[name].append(1 - post[name][j])
            tv.append(0.5 * np.abs(post["A"] - post["B"]).sum())
            ss = np.sort(s)
            gaps.append((ss[1] - ss[0]) / m)
            agree.append(post["A"].argmax() == post["B"].argmax() == j)
        rows.append([m, np.mean(miss["A"]), np.mean(miss["B"]), np.mean(tv), np.median(gaps), np.mean(agree)])
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(rows[:, 0], rows[:, 1], "o-", color=COL["A"], label="P(J != j*), prior A (uniform w)")
    ax.semilogy(rows[:, 0], rows[:, 2], "s-", color=COL["B"], label=f"P(J != j*), prior B (w ratio <= {C:g})")
    ax.semilogy(rows[:, 0], rows[:, 3], "k--", label="TV(post A, post B) over components")
    ax.set_xscale("log")
    ax.set(xlabel="observed entries m (random inpainting, n = 3072)", ylabel="probability / distance",
           title=f"Thm 3.2 check: GMM prior on {M} CelebA-HQ images")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "thm32_check.png"), dpi=150)
    plt.close(fig)
    np.savetxt(os.path.join(OUT, "thm32_check.csv"), rows, delimiter=",", fmt="%.6g",
               header="m,P_miss_A,P_miss_B,TV_A_B,median_gap_per_dim,both_argmax_correct", comments="")
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    e = part1(rng)
    print("Part 1: mean ||E[x|y]-x*|| at N=10 / 50 / 1000:", {k: tuple(round(v, 3) for v in t) for k, t in e.items()})
    assert all(e[n][2] < e[n][0] for n in e), "posterior error should shrink with N"
    rows = part2(rng)
    print("Part 2: m, P_miss_A, P_miss_B, TV(A,B), median gap/dim, both argmax == j*")
    for r in rows:
        print("  " + "  ".join(f"{v:.4g}" for v in r))
    assert rows[-1, 3] < 1e-3 and rows[-1, 5] > 0.99, "high-dimensional posteriors should agree"
    print("toy experiments done")


if __name__ == "__main__":
    main()
