"""Table 2 / Table 14: spatial autocorrelation of unconditional k-step DDIM samples (App. C.8).

C_x(r) = 1/C sum_c [ |P_r|^-1 sum_{(p,q) in P_r} (x_cp - m_c)(x_cq - m_c) ] / [ |Omega|^-1 sum_p (x_cp - m_c)^2 ],
P_r = ordered in-image pixel pairs with round(||p - q||) = r (no wrap-around), averaged over 100 samples.
Also reports both DDIM timestep spacings, because the paper does not state which one it used.
Run: python -s repro/exp_autocorr.py
"""
import csv
import itertools
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "autocorr")
LAGS = [0, 1, 2, 4, 8, 16, 32]
MAXLAG = 32
PAPER = {  # Table 2
    ("celeba", 3): [1.0000, 0.9558, 0.9357, 0.8866, 0.7767, 0.5595, 0.2666],
    ("celeba", 20): [1.0000, 0.9814, 0.9594, 0.9100, 0.8108, 0.6261, 0.3481],
    ("bedroom", 3): [1.0000, 0.9645, 0.9370, 0.8786, 0.7637, 0.5632, 0.3052],
    ("bedroom", 20): [1.0000, 0.9615, 0.9260, 0.8573, 0.7437, 0.5618, 0.3207],
}


def autocorr_profile(x, maxlag=MAXLAG):
    """x: (B,C,H,W) float tensor -> (B, maxlag+1) profile, exact pooled definition of App. C.8."""
    B, C, H, W = x.shape
    x = x.double() - x.double().mean((2, 3), keepdim=True)
    f = torch.fft.rfft2(x, s=(2 * H, 2 * W))
    s = torch.fft.irfft2(f.abs() ** 2, s=(2 * H, 2 * W))          # S(d) = sum_p x_p x_{p+d}
    d0 = torch.arange(2 * H, device=x.device)
    d0 = torch.minimum(d0, 2 * H - d0)
    d1 = torch.arange(2 * W, device=x.device)
    d1 = torch.minimum(d1, 2 * W - d1)
    npairs = (H - d0).clamp(min=0)[:, None] * (W - d1).clamp(min=0)[None, :]
    rbin = torch.round(torch.sqrt(d0[:, None].double() ** 2 + d1[None, :].double() ** 2)).long()
    keep = rbin <= maxlag
    idx = rbin[keep]
    num = torch.zeros(B, C, maxlag + 1, dtype=torch.float64, device=x.device)
    num.index_add_(2, idx, s[..., keep])
    den = torch.zeros(maxlag + 1, dtype=torch.float64, device=x.device).index_add_(0, idx, npairs[keep].double())
    var = s[..., 0, 0] / (H * W)
    return ((num / den) / var[..., None]).mean(1)


def sample(domain, k, spacing, n=100, bs=10):
    model = wp.unet(domain)
    xs = []
    with torch.no_grad():
        for i in range(0, n, bs):
            z = torch.stack([torch.randn(3, 256, 256, generator=torch.Generator().manual_seed(s))
                             for s in range(i, i + bs)]).to(wp.DEV)
            xs.append(wp.ddim(model, z, k, spacing).clamp(-1, 1))
    return torch.cat(xs)


def load_dir(d, n=100):
    ims = [np.asarray(Image.open(os.path.join(d, f"{i:03d}.png")), dtype=np.float32) / 127.5 - 1 for i in range(n)]
    return torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).to(wp.DEV)


def selftest():
    # white noise: lag>=1 ~ 0;  constant-gradient-free smooth field: monotone decay
    g = torch.Generator().manual_seed(0)
    w = autocorr_profile(torch.randn(4, 3, 64, 64, generator=g).to(wp.DEV))
    assert abs(w[:, 0].mean() - 1) < 1e-9 and w[:, 1:].abs().max() < 0.05, w
    sm = torch.nn.functional.avg_pool2d(torch.randn(4, 3, 96, 96, generator=g), 9, 1).to(wp.DEV)
    p = autocorr_profile(sm, 12).mean(0)
    assert torch.all(p[1:9].diff() < 0), p


def main():
    selftest()
    os.makedirs(OUT, exist_ok=True)
    profiles, rows = {}, []
    for domain, k, spacing in itertools.product(["celeba", "bedroom"], [3, 20], ["leading", "trailing"]):
        x = sample(domain, k, spacing)
        prof = autocorr_profile(x).mean(0).cpu().numpy()
        profiles[(domain, k, spacing)] = prof
        ref = np.array(PAPER[(domain, k)])
        rmse = float(np.sqrt(np.mean((prof[LAGS] - ref) ** 2)))
        rows.append([domain, k, spacing] + [f"{prof[l]:.4f}" for l in LAGS] + [f"{rmse:.4f}"])
        print(domain, k, spacing, " ".join(f"{prof[l]:.4f}" for l in LAGS), f"rmse_vs_paper={rmse:.4f}", flush=True)
        grid = ((x[:8].permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
        Image.fromarray(np.concatenate(list(grid), axis=1)).save(os.path.join(OUT, f"samples_{domain}_{k}_{spacing}.png"))
    for name in ["celeba", "bedroom", "church"]:
        prof = autocorr_profile(load_dir(os.path.join(ROOT, "data", "test", name))).mean(0).cpu().numpy()
        profiles[(name, "real", "-")] = prof
        rows.append([name, "real test images", "-"] + [f"{prof[l]:.4f}" for l in LAGS] + [""])
        print(name, "real", " ".join(f"{prof[l]:.4f}" for l in LAGS), flush=True)
    with open(os.path.join(OUT, "table2.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["model", "ddim_steps", "spacing"] + [f"lag_{l}" for l in LAGS] + ["rmse_vs_paper"]]
                                + rows)
    with open(os.path.join(OUT, "profiles_full.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["model", "steps", "spacing"] + [f"lag_{l}" for l in range(MAXLAG + 1)]]
                                + [list(k) + [f"{v:.5f}" for v in p] for k, p in profiles.items()])
    # Table 14 (Pearson over lags {1,2,4,8,16,32}) and the main-text number (lags 1..32), per spacing
    for spacing in ["leading", "trailing"]:
        keys = [(d, k, spacing) for d in ["celeba", "bedroom"] for k in [3, 20]]
        mat = [[np.corrcoef(profiles[a][LAGS[1:]], profiles[b][LAGS[1:]])[0, 1] for b in keys] for a in keys]
        with open(os.path.join(OUT, f"table14_{spacing}.csv"), "w", newline="") as f:
            csv.writer(f).writerows([[""] + [f"{d} {k}" for d, k, _ in keys]]
                                    + [[f"{a[0]} {a[1]}"] + [f"{v:.4f}" for v in r] for a, r in zip(keys, mat)])
        r = np.corrcoef(profiles[("bedroom", 3, spacing)][1:], profiles[("celeba", 20, spacing)][1:])[0, 1]
        print(f"[{spacing}] Pearson(Bedroom 3, CelebA 20) over lags 1..32 = {r:.4f} (paper 0.9987)")
        print(f"[{spacing}] Table 14 min off-diagonal = {np.min(mat):.4f} (paper 0.9965)")


if __name__ == "__main__":
    main()
