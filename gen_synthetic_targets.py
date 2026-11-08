"""Generate 1/f^beta Gaussian random field target images and verify their statistics.

Run: python gen_synthetic_targets.py   (numpy + matplotlib only)
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------- parameters ----------------
SIZE = 256
CHANNELS = 3
BETAS = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]
N_IMAGES = 20
SEED_BASE = 0
CLIP_SIGMA = 3.0          # common scale: z-scored value * (1/CLIP_SIGMA), then clip to [-1, 1]
MAX_LAG = 32
REPORT_LAGS = [0, 1, 2, 4, 8, 16, 32]
FIT_RANGE = (4, 64)       # radial frequency band (cycles/image) for slope fit
N_GRID_ROWS = 4
OUT = "synthetic_targets"
MED_DIR = "data/montgomery"   # NLM Montgomery CXR set: CXR_png/, leftMask/, rightMask/
N_MED = 20

PAPER = {  # Table 2 prior-sample autocorrelation at REPORT_LAGS
    "CelebA 3-step": [1.0000, 0.9558, 0.9357, 0.8866, 0.7767, 0.5595, 0.2666],
    "CelebA 20-step": [1.0000, 0.9814, 0.9594, 0.9100, 0.8108, 0.6261, 0.3481],
    "Bedroom 3-step": [1.0000, 0.9645, 0.9370, 0.8786, 0.7637, 0.5632, 0.3052],
    "Bedroom 20-step": [1.0000, 0.9615, 0.9260, 0.8573, 0.7437, 0.5618, 0.3207],
}

# ---------------- shared grids ----------------
k = np.fft.fftfreq(SIZE) * SIZE                      # integer frequencies, cycles/image
KR = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)      # radial frequency
KR_BIN = np.rint(KR).astype(int)
d = np.minimum(np.arange(SIZE), SIZE - np.arange(SIZE))   # circular lag distance
LAG_BIN = np.rint(np.sqrt(d[:, None] ** 2 + d[None, :] ** 2)).astype(int)


def radial_mean(a, bins, nmax):
    return np.bincount(bins.ravel(), a.ravel(), minlength=nmax + 1)[: nmax + 1] / \
        np.bincount(bins.ravel(), minlength=nmax + 1)[: nmax + 1]


def make_field(beta, rng):
    amp = np.zeros_like(KR)
    amp[KR > 0] = KR[KR > 0] ** (-beta / 2)          # DC = 0
    phase = rng.uniform(0, 2 * np.pi, KR.shape)
    return np.fft.ifft2(amp * np.exp(1j * phase)).real


def make_image(beta, bi, i):
    chans = []
    for c in range(CHANNELS):
        rng = np.random.default_rng(SEED_BASE + 100000 * bi + 10 * i + c)
        x = make_field(beta, rng)
        chans.append((x - x.mean()) / x.std())
    z = np.stack(chans) / CLIP_SIGMA
    clipped = np.abs(z) > 1
    return np.clip(z, -1, 1).astype(np.float32), clipped.mean()


def autocorr(ch):
    x = ch - ch.mean()
    ac = np.fft.ifft2(np.abs(np.fft.fft2(x)) ** 2).real
    return radial_mean(ac / ac[0, 0], LAG_BIN, MAX_LAG)


PAD_LAG_BIN = None


def autocorr_padded(ch):
    """Non-periodic autocorrelation: zero-pad to 2N, divide by overlap count, lag 0 = 1."""
    global PAD_LAG_BIN
    n0, n1 = ch.shape
    x = ch - ch.mean()
    ac = np.fft.ifft2(np.abs(np.fft.fft2(x, s=(2 * n0, 2 * n1))) ** 2).real
    d0 = np.minimum(np.arange(2 * n0), 2 * n0 - np.arange(2 * n0))
    d1 = np.minimum(np.arange(2 * n1), 2 * n1 - np.arange(2 * n1))
    overlap = np.clip(n0 - d0, 1, None)[:, None] * np.clip(n1 - d1, 1, None)[None, :]
    ac /= overlap
    if PAD_LAG_BIN is None:
        PAD_LAG_BIN = np.rint(np.sqrt(d0[:, None] ** 2 + d1[None, :] ** 2)).astype(int)
    return radial_mean(ac / ac[0, 0], PAD_LAG_BIN, MAX_LAG)


def load_medical():
    """Square crop around the union of the manual lung masks, block-mean downsample to SIZE."""
    names = sorted(os.listdir(os.path.join(MED_DIR, "CXR_png")))[:N_MED]
    imgs = []
    for nm in names:
        x = plt.imread(os.path.join(MED_DIR, "CXR_png", nm)).astype(np.float64)
        if x.ndim == 3:
            x = x[..., 0]
        m = sum(plt.imread(os.path.join(MED_DIR, k, nm)) > 0 for k in ("leftMask", "rightMask"))
        ys, xs = np.nonzero(m)
        side = max(ys.max() - ys.min(), xs.max() - xs.min()) // SIZE * SIZE   # multiple of SIZE
        side = min(side, min(x.shape) // SIZE * SIZE)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        y0 = int(np.clip(cy - side // 2, 0, x.shape[0] - side))
        x0 = int(np.clip(cx - side // 2, 0, x.shape[1] - side))
        crop = x[y0:y0 + side, x0:x0 + side]
        f = side // SIZE
        small = crop.reshape(SIZE, f, SIZE, f).mean(axis=(1, 3))        # area (anti-aliased) resize
        g = small * 2 - 1                                               # [0,1] pixel range -> [-1,1]
        imgs.append(np.repeat(g[None], CHANNELS, 0).astype(np.float32))
    return names, imgs


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    nfreq = SIZE // 2
    spectra, acs, gacs, clip_frac, grid = {}, {}, {}, {}, {}
    for bi, beta in enumerate(BETAS):
        d_out = os.path.join(OUT, "images", f"beta_{beta:.1f}")
        g_out = os.path.join(OUT, "images_gray", f"beta_{beta:.1f}")
        os.makedirs(d_out, exist_ok=True)
        os.makedirs(g_out, exist_ok=True)
        ps, ac, gac, cf, per_img_ac1 = 0.0, 0.0, 0.0, [], []
        for i in range(N_IMAGES):
            img, frac = make_image(beta, bi, i)
            cf.append(frac)
            np.save(os.path.join(d_out, f"{i:03d}.npy"), img)
            u8 = np.rint((img.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            plt.imsave(os.path.join(d_out, f"{i:03d}.png"), u8)
            img_ac = 0.0
            for ch in img.astype(np.float64):
                ps += radial_mean(np.abs(np.fft.fft2(ch - ch.mean())) ** 2, KR_BIN, nfreq)
                img_ac += autocorr(ch)
            ac += img_ac
            per_img_ac1.append(img_ac[1] / CHANNELS)
            # grayscale twin: channel 0 copied to all 3 channels, same normalization
            gray = np.repeat(img[:1], CHANNELS, 0)
            np.save(os.path.join(g_out, f"{i:03d}.npy"), gray)
            plt.imsave(os.path.join(g_out, f"{i:03d}.png"),
                       np.rint((gray.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8))
            gac += autocorr_padded(gray[0].astype(np.float64))
            if i < N_GRID_ROWS:
                grid[(i, bi)] = u8
        n = N_IMAGES * CHANNELS
        spectra[beta], acs[beta], clip_frac[beta] = ps / n, ac / n, float(np.mean(cf))
        gacs[beta] = gac / N_IMAGES
        grid[("ac1", bi)] = per_img_ac1
        print(f"beta={beta:.1f}  lag1={acs[beta][1]:.4f}  clipped={100 * clip_frac[beta]:.3f}%")

    # ---- power spectrum + slope fit ----
    f = np.arange(nfreq + 1)
    band = (f >= FIT_RANGE[0]) & (f <= FIT_RANGE[1])
    slopes = {b: np.polyfit(np.log(f[band]), np.log(spectra[b][band]), 1)[0] for b in BETAS}
    write_csv(os.path.join(OUT, "slope_fit.csv"), ["beta", "fitted_slope", "expected_slope", "clipped_pixel_fraction"],
              [[b, f"{slopes[b]:.4f}", -b, f"{clip_frac[b]:.6f}"] for b in BETAS])
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    for bi, b in enumerate(BETAS):
        ax.loglog(f[1:], spectra[b][1:], color=cmap(bi / (len(BETAS) - 1)),
                  label=f"β={b:.1f} (fit {slopes[b]:.2f})")
    ax.axvspan(*FIT_RANGE, color="grey", alpha=0.12, label="fit band")
    ax.set(xlabel="radial frequency (cycles/image)", ylabel="power", title="Radially averaged power spectrum")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "power_spectrum.png"), dpi=150)
    plt.close(fig)

    # ---- autocorrelation table ----
    write_csv(os.path.join(OUT, "autocorr.csv"), ["beta"] + [f"lag_{l}" for l in REPORT_LAGS],
              [[b] + [f"{acs[b][l]:.4f}" for l in REPORT_LAGS] for b in BETAS])
    write_csv(os.path.join(OUT, "autocorr_full.csv"), ["beta"] + [f"lag_{l}" for l in range(MAX_LAG + 1)],
              [[b] + [f"{v:.4f}" for v in acs[b]] for b in BETAS])

    # ---- comparison with paper ----
    refs = {**PAPER, "paper mean": list(np.mean(list(PAPER.values()), axis=0))}
    rows, mean_rmse = [], {}
    for b in BETAS:
        ours = acs[b][REPORT_LAGS]
        row = [b]
        for name, ref in refs.items():
            ref = np.asarray(ref)
            row += [f"{np.sqrt(np.mean((ours - ref) ** 2)):.4f}", f"{np.corrcoef(ours, ref)[0, 1]:.4f}"]
        mean_rmse[b] = np.mean([np.sqrt(np.mean((ours - np.asarray(r)) ** 2)) for r in PAPER.values()])
        rows.append(row + [f"{mean_rmse[b]:.4f}"])
    header = ["beta"] + [f"{n} {m}" for n in refs for m in ("RMSE", "Pearson")] + ["avg RMSE over 4 refs"]
    write_csv(os.path.join(OUT, "paper_distance.csv"), header, rows)
    best = min(mean_rmse, key=mean_rmse.get)
    print(f"closest beta to paper priors (avg RMSE): {best}  ({mean_rmse[best]:.4f})")

    # ---- medical images (non-periodic -> padded autocorrelation) vs grayscale synthetic ----
    med = None
    if os.path.isdir(os.path.join(MED_DIR, "CXR_png")):
        names, med_imgs = load_medical()
        m_out = os.path.join(OUT, "images_medical")
        os.makedirs(m_out, exist_ok=True)
        med_ac = []
        for i, (nm, img) in enumerate(zip(names, med_imgs)):
            np.save(os.path.join(m_out, f"{i:03d}.npy"), img)
            plt.imsave(os.path.join(m_out, f"{i:03d}.png"),
                       np.rint((img.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8))
            med_ac.append(autocorr_padded(img[0].astype(np.float64)))
        med = np.mean(med_ac, axis=0)
        write_csv(os.path.join(m_out, "sources.csv"), ["index", "file"], list(enumerate(names)))
        write_csv(os.path.join(OUT, "autocorr_gray_padded.csv"), ["curve"] + [f"lag_{l}" for l in REPORT_LAGS],
                  [[f"beta {b}"] + [f"{gacs[b][l]:.4f}" for l in REPORT_LAGS] for b in BETAS]
                  + [["medical"] + [f"{med[l]:.4f}" for l in REPORT_LAGS]])
        mrows = []
        for b in BETAS:
            r_pts = np.sqrt(np.mean((med[REPORT_LAGS] - gacs[b][REPORT_LAGS]) ** 2))
            r_all = np.sqrt(np.mean((med - gacs[b]) ** 2))
            mrows.append([f"beta {b}", f"{r_pts:.4f}", f"{r_all:.4f}"])
        for name, ref in PAPER.items():
            r_pts = np.sqrt(np.mean((med[REPORT_LAGS] - np.asarray(ref)) ** 2))
            mrows.append([f"paper {name}", f"{r_pts:.4f}", ""])
        write_csv(os.path.join(OUT, "medical_distance.csv"), ["reference", "RMSE at report lags", "RMSE lags 0-32"], mrows)
        med_beta = BETAS[int(np.argmin([float(r[1]) for r in mrows[:len(BETAS)]]))]
        print(f"medical lag1={med[1]:.4f} lag32={med[32]:.4f}  closest grayscale beta: {med_beta}")
        grid["med"] = [(np.rint((im.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8), a[1])
                       for im, a in zip(med_imgs, med_ac)]

    fig, ax = plt.subplots(figsize=(7, 5))
    lags = np.arange(MAX_LAG + 1)
    for bi, b in enumerate(BETAS):
        ax.plot(lags, acs[b], color=cmap(bi / (len(BETAS) - 1)), label=f"β={b:.1f}")
    for name, ref, c in zip(PAPER, PAPER.values(), ["C3", "C1", "C0", "C4"]):
        ax.plot(REPORT_LAGS, ref, "--o", color=c, ms=3, label=f"paper {name}")
    if med is not None:
        ax.plot(lags, med, "k-", lw=2.5, label="Montgomery CXR (padded)")
    ax.set(xlabel="lag (pixels)", ylabel="autocorrelation", title="Spatial autocorrelation vs. paper Table 2")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "autocorr_vs_paper.png"), dpi=150)
    plt.close(fig)

    # ---- image grid ----
    rows = N_GRID_ROWS + (med is not None)
    fig, axes = plt.subplots(rows, len(BETAS), figsize=(1.6 * len(BETAS), 1.8 * rows))
    if med is not None:  # last row: medical samples (not tied to the column's β)
        for j, (u8, a1) in enumerate(grid["med"][:len(BETAS)]):
            a = axes[-1, j]
            a.imshow(u8)
            a.set_xticks([]), a.set_yticks([])
            a.set_xlabel(f"CXR ρ1={a1:.3f}", fontsize=7)
        axes[-1, 0].set_ylabel("medical", fontsize=9)
    for bi, b in enumerate(BETAS):
        for i in range(N_GRID_ROWS):
            a = axes[i, bi]
            a.imshow(grid[(i, bi)])
            a.set_xticks([]), a.set_yticks([])
            a.set_xlabel(f"ρ1={grid[('ac1', bi)][i]:.3f}", fontsize=7)
            if i == 0:
                a.set_title(f"β={b:.1f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "image_grid.png"), dpi=150)
    plt.close(fig)

    # ---- self-checks ----
    assert all(abs(slopes[b] + b) < 0.25 for b in BETAS), slopes
    assert abs(acs[0.0][1]) < 0.02 and acs[4.0][1] > 0.99
    assert np.all(np.diff([acs[b][1] for b in BETAS]) <= 0), "lag-1 should fall as beta drops"
    print("self-checks passed")


if __name__ == "__main__":
    main()
