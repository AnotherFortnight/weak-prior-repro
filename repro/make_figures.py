"""Qualitative figures from saved PNGs (Figures 1, 3, 5-10, 14, 15 analogues).

Run: python -s repro/make_figures.py   -> results/figures/*.png
"""
import os

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
OUT = os.path.join(ROOT, "results", "figures")


def img(path):
    return np.asarray(Image.open(path).convert("RGB")) if os.path.exists(path) else np.full((256, 256, 3), 255, np.uint8)


def grid(rows, col_titles, row_titles, path, size=128):
    th, tw = 18, 180
    H, W = len(rows) * size + th, len(col_titles) * size + tw
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for j, t in enumerate(col_titles):
        d.text((tw + j * size + 4, 3), t, fill="black")
    for i, (rt, row) in enumerate(zip(row_titles, rows)):
        d.text((4, th + i * size + size // 2 - 6), rt, fill="black")
        for j, a in enumerate(row):
            canvas.paste(Image.fromarray(a).resize((size, size), Image.BICUBIC), (tw + j * size, th + i * size))
    canvas.save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    cols = ["measurement", "DPS (in)", "Ours CelebA", "Ours Bedroom", "Ours Church", "reference"]
    for task in ["inpaint", "sr4", "gauss"]:
        rows, names = [], []
        for tgt in ["celeba", "bedroom", "church"]:
            for i in range(2):
                r = [img(os.path.join(RUNS, f"ino_{task}_{tgt}_from-{tgt}_sigma0.01", f"{i:03d}_measurement.png")),
                     img(os.path.join(RUNS, f"dps_{task}_{tgt}_from-{tgt}_sigma0.01", f"{i:03d}.png"))]
                r += [img(os.path.join(RUNS, f"ino_{task}_{tgt}_from-{p}_sigma0.01", f"{i:03d}.png"))
                      for p in ["celeba", "bedroom", "church"]]
                r.append(img(os.path.join(ROOT, "data", "test", tgt, f"{i:03d}.png")))
                rows.append(r)
                names.append(f"{tgt} #{i}")
        grid(rows, cols, names, os.path.join(OUT, f"recon_{task}.png"))
    # Table 4, nonlinear deblurring (in-domain only): DMPlug vs ours
    rows, names = [], []
    for tgt in ["celeba", "bedroom", "church"]:
        for i in range(2):
            rows.append([img(os.path.join(RUNS, f"{m}_nonlinear_{tgt}_from-{tgt}_sigma0.01", f"{i:03d}{s}.png"))
                         for m, s in [("ino", "_measurement"), ("dmplug", ""), ("ino", "")]]
                        + [img(os.path.join(ROOT, "data", "test", tgt, f"{i:03d}.png"))])
            names.append(f"{tgt} #{i}")
    grid(rows, ["measurement", "DMPlug", "Ours", "reference"], names, os.path.join(OUT, "recon_nonlinear.png"))
    # Figure 1 analogue: optimisation trajectories (iteration 1, 5, 10, 100, 1000 | output | reference | measurement)
    rows, names = [], []
    for task, tgt, priors in [("inpaint", "celeba", ["celeba", "bedroom"]), ("sr4", "bedroom", ["bedroom", "celeba"])]:
        for p in priors:
            f = os.path.join(RUNS, f"ino_{task}_{tgt}_from-{p}_sigma0.01", "000_strip.png")
            if os.path.exists(f):
                a = img(f)
                rows.append([a[:, k * 256:(k + 1) * 256] for k in range(a.shape[1] // 256)])
                names.append(f"{tgt} {task}, {p} prior")
    if rows:
        grid(rows, ["it 1", "it 5", "it 10", "it 100", "it 1000", "HTK output", "reference", "measurement"],
             names, os.path.join(OUT, "fig1_trajectories.png"))
    # failure modes (Figures 3, 14, 15)
    rows, names = [], []
    for task in ["box0.3", "box0.6", "sr16"]:
        for i in range(2):
            r = [img(os.path.join(RUNS, f"ino_{task}_church_from-church_sigma0.01", f"{i:03d}_measurement.png")),
                 img(os.path.join(RUNS, f"dps_{task}_church_from-church_sigma0.01", f"{i:03d}.png")),
                 img(os.path.join(RUNS, f"ino_{task}_church_from-church_sigma0.01", f"{i:03d}.png")),
                 img(os.path.join(RUNS, f"ino_{task}_church_from-celeba_sigma0.01", f"{i:03d}.png")),
                 img(os.path.join(ROOT, "data", "test", "church", f"{i:03d}.png"))]
            rows.append(r)
            names.append(f"{task} #{i}")
    grid(rows, ["measurement", "DPS", "Ours (In)", "Ours (Out)", "reference"], names,
         os.path.join(OUT, "failure_modes.png"))
    autocorr_plot()
    print("figures written to", OUT)


def autocorr_plot():
    """Table 2 as a figure: reproduced profiles (leading spacing) vs the paper's representative lags."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp_autocorr import LAGS, PAPER
    f = os.path.join(ROOT, "results", "autocorr", "profiles_full.csv")
    if not os.path.exists(f):
        return
    rows = {(r["model"], r["steps"], r["spacing"]): np.array([float(r[f"lag_{l}"]) for l in range(33)])
            for r in csv.DictReader(open(f))}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (dom, k), c in zip(PAPER, ["C3", "C1", "C0", "C9"]):
        ax.plot(range(33), rows[(dom, str(k), "leading")], "-", color=c, label=f"{dom} {k}-step (repro)")
        ax.plot(LAGS, PAPER[(dom, k)], "o", color=c, mfc="none", label=f"{dom} {k}-step (paper)")
    for dom, ls in [("celeba", ":"), ("bedroom", "--")]:
        ax.plot(range(33), rows[(dom, "real", "-")], ls, color="gray", label=f"{dom} real test images")
    ax.set(xlabel="lag r (pixels)", ylabel="C(r)", title="Spatial autocorrelation of 3/20-step DDIM samples")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "table2_autocorr.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
