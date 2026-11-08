"""Table 1: per-dimension score gap for 70 % random inpainting (Sec. 3.2, "Justifying the identifiable assumption").

For each of the 100 test images: mask 70 % of the pixels, add N(0, 0.01^2) noise to the rest; apply the same mask to
every image of a candidate pool (which contains the target) and compute the MSE on the observed entries ([-1, 1]
scale). Gap = second-smallest minus smallest MSE. The paper does not state the pool size, so several are reported.
Pools: CelebA-HQ val (2000) + train shard 0 (4667); LSUN val (300) + HF test split (bedroom 15157, church 6312).
Run: python -s repro/exp_scoregap.py   (CPU, a few minutes)
"""
import csv
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import center_crop_arr, lsun_images  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "scoregap")
PAPER = {"celeba": (0.22, 0.06, 0.11), "church": (0.28, 0.06, 0.15), "bedroom": (0.27, 0.07, 0.09)}
SIZES = [100, 300, 1000, 3000, 10000]


def parquet_images(path, crop):
    import pyarrow.parquet as pq
    for rec in pq.read_table(path, columns=["image"]).column("image").to_pylist():
        im = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
        yield center_crop_arr(im) if crop else np.asarray(im)


def pool(name):
    """uint8 (N,256,256,3): the 100 test images first, then all other pool images."""
    test = [np.asarray(Image.open(os.path.join(DATA, "test", name, f"{i:03d}.png"))) for i in range(100)]
    with open(os.path.join(DATA, "test", name, "sources.csv")) as f:
        used = {r["source_key"] for r in csv.DictReader(f)}
    if name == "celeba":
        val = parquet_images(os.path.join(DATA, "celebahq_hf/data/validation-00000-of-00001.parquet"), False)
        others = [a for i, a in enumerate(val) if f"val_{i:04d}" not in used]
        others += list(parquet_images(os.path.join(DATA, "celebahq_hf/data/train-00000-of-00006.parquet"), False))
    else:
        lmdb = {"bedroom": "lsun_bedroom_val/bedroom_val_lmdb", "church": "lsun_church_val/church_outdoor_val_lmdb"}
        others = [a for k, a in lsun_images(os.path.join(DATA, lmdb[name])) if k not in used]
        hf = {"bedroom": "lsun_bedroom_hf/data/test-00000-of-00001-7c2280a6897e2462.parquet",
              "church": "lsun_church_hf/data/test-00000-of-00001-b3f98c62a94e5d25.parquet"}
        others += list(parquet_images(os.path.join(DATA, hf[name]), True))
    return np.stack(test + others)


def masked_mse(P, rng, sigma=0.01, chunk=1000):
    """S[t, j] = mean over observed entries of target t of (y_t - c_j)^2, for the 100 targets vs all candidates."""
    D = P[0].size
    x = P[:100].reshape(100, -1).astype(np.float32) / 127.5 - 1
    keep = np.zeros((100, 256 * 256), np.float32)
    for t in range(100):
        keep[t, rng.permutation(256 * 256)[: 256 * 256 - int(0.7 * 256 * 256)]] = 1
    m = np.repeat(keep, 3, axis=1)                                  # HWC layout: mask shared by RGB
    y = (x + sigma * rng.standard_normal(x.shape, dtype=np.float32)) * m
    nobs = m.sum(1)
    S = np.empty((100, len(P)), np.float32)
    for s in range(0, len(P), chunk):
        c = P[s:s + chunk].reshape(-1, D).astype(np.float32) / 127.5 - 1
        S[:, s:s + chunk] = ((y ** 2).sum(1)[:, None] - 2 * y @ c.T + m @ (c ** 2).T) / nobs[:, None]
    return S


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for name in ["celeba", "church", "bedroom"]:
        P = pool(name)
        S = masked_mse(P, rng)
        assert np.all(S[np.arange(100), np.arange(100)] < 2e-4), "target itself should score ~sigma^2"
        perm = rng.permutation(np.arange(100, len(P)))               # random non-test candidates
        for size in [s for s in SIZES if s < len(P)] + [len(P)]:
            gaps, hit = [], []
            for t in range(100):
                if size == len(P):
                    cand = np.arange(len(P))
                else:                                               # target + other test images + random extras
                    cand = np.r_[np.arange(100), perm[: max(size - 100, 0)]][:size]
                    cand = cand if t in cand else np.r_[t, cand[:-1]]
                s = np.sort(S[t, cand])
                gaps.append(s[1] - s[0])
                hit.append(S[t, cand].argmin() == np.flatnonzero(cand == t)[0])
            g = np.array(gaps)
            rows.append([name, size, f"{g.mean():.3f}", f"{g.std():.3f}", f"{g.min():.3f}", f"{np.mean(hit):.2f}"])
            print(f"{name:8s} pool {size:6d}: gap mean {g.mean():.3f} (std {g.std():.3f}) min {g.min():.3f} "
                  f"| argmin==target {np.mean(hit):.2f} | paper {PAPER[name]}", flush=True)
    with open(os.path.join(OUT, "table1.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["dataset", "pool_size", "gap_mean", "gap_std", "gap_min", "argmin_is_target"]] + rows)


if __name__ == "__main__":
    main()
