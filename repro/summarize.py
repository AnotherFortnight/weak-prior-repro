"""Collect results/runs/* into paper-style tables: reproduced mean +- s.e. [paper value].

Run: python -s repro/summarize.py   -> results/summary.md, results/fig4_box_lpips.png
"""
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
LP = "lpips_vgg"          # LPIPS backbone used in the main tables (both are stored per image)
T = ["celeba", "bedroom", "church"]

# ---- paper numbers: (PSNR, LPIPS, SSIM) ----
P3 = {  # Tables 3 and 6, keyed by (task, method, prior) -> per target
    ("inpaint", "dps", "in"): [(31.98, .14, .88), (27.97, .23, .82), (24.15, .25, .73)],
    ("inpaint", "ino", "celeba"): [(33.78, .15, .91), (27.78, .27, .83), (23.56, .36, .72)],
    ("inpaint", "ino", "bedroom"): [(32.76, .17, .90), (28.88, .20, .87), (24.22, .28, .75)],
    ("inpaint", "ino", "church"): [(32.62, .16, .90), (28.66, .21, .86), (24.93, .23, .77)],
    ("sr4", "dps", "in"): [(26.82, .22, .74), (22.95, .39, .64), (20.28, .36, .52)],
    ("sr4", "ino", "celeba"): [(31.27, .18, .86), (25.88, .35, .74), (22.68, .41, .63)],
    ("sr4", "ino", "bedroom"): [(30.34, .22, .84), (26.59, .25, .78), (22.86, .33, .65)],
    ("sr4", "ino", "church"): [(30.10, .22, .84), (26.30, .27, .77), (23.09, .27, .67)],
    ("gauss", "dps", "in"): [(27.55, .19, .76), (23.94, .31, .67), (20.66, .30, .55)],
    ("gauss", "ino", "celeba"): [(29.98, .21, .82), (25.02, .41, .69), (22.21, .45, .58)],
    ("gauss", "ino", "bedroom"): [(27.97, .27, .77), (25.28, .30, .72), (22.09, .38, .59)],
    ("gauss", "ino", "church"): [(27.66, .29, .75), (24.76, .35, .69), (22.43, .30, .62)],
    ("nonlinear", "dps", "in"): [(24.88, .26, .71), (22.51, .38, .63), (20.31, .39, .54)],
    ("nonlinear", "ino", "celeba"): [(25.21, .27, .74), (22.34, .47, .63), (20.52, .51, .52)],
    ("nonlinear", "ino", "bedroom"): [(24.59, .39, .69), (22.67, .39, .64), (20.79, .47, .54)],
    ("nonlinear", "ino", "church"): [(24.73, .39, .69), (22.59, .42, .63), (20.88, .39, .56)],
}
PDMP = {  # Tables 4 and 7: DMPlug, per target (celeba, bedroom, church)
    "inpaint": [(32.778, .189, .892), (28.271, .255, .836), (24.957, .246, .771)],
    "gauss": [(29.52, .23, .81), (25.08, .33, .71), (22.24, .32, .61)],
    "sr4": [(31.122, .198, .857), (26.309, .280, .767), (23.183, .280, .667)],
    "nonlinear": [(24.85, .28, .73), (22.21, .41, .63), (20.48, .42, .53)],
}
P5 = {("gauss", "church"): (22.034, 22.394, 22.048, 22.404), ("gauss", "bedroom"): (24.825, 25.595, 24.854, 25.596),
      ("inpaint", "church"): (24.724, 24.843, 24.743, 24.844),
      ("inpaint", "bedroom"): (28.669, 29.156, 28.559, 29.162)}
P12 = {0.3: [(25.00, .22, .79), (25.28, .23, .81), (22.42, .32, .77)],
       0.4: [(22.28, .24, .76), (22.31, .25, .78), (19.20, .35, .73)],
       0.5: [(19.38, .27, .71), (19.09, .30, .72), (16.81, .39, .67)],
       0.6: [(17.72, .30, .65), (17.27, .34, .66), (15.35, .43, .61)]}
P16 = {0.01: [(31.98, .1447, .8827, None), (33.78, .1476, .9147, 974.09), (32.62, .1626, .8985, 973.42)],
       0.05: [(30.12, .1788, .8403, None), (33.39, .1457, .9096, 940.61), (32.08, .1676, .8889, 921.33)],
       0.2: [(26.93, .2281, .7602, None), (30.41, .1981, .8478, 358.95), (28.38, .2862, .7877, 351.41)],
       0.5: [(24.45, .2692, .7004, None), (26.30, .2596, .7335, 168.64), (23.86, .4366, .6183, 155.89)],
       1.0: [(22.42, .3095, .6518, None), (20.52, .3771, .4915, 119.05), (16.88, .6171, .2962, 108.28)]}

ZETA_NL = "_zeta0.3"   # DPS on nonlinear deblurring: zeta' = 1 (DPS paper) diverges on 0-29 % of images, see WORKLOG
P15 = {  # Table 15: DPS on nonlinear deblurring, (source model, target) -> (PSNR, LPIPS, SSIM)
    ("celeba", "celeba"): (24.88, .264, .709), ("church", "celeba"): (22.38, .433, .631),
    ("bedroom", "celeba"): (23.57, .411, .660), ("church", "church"): (20.31, .385, .539),
    ("celeba", "church"): (20.06, .508, .490), ("bedroom", "church"): (19.84, .469, .512),
    ("bedroom", "bedroom"): (22.51, .375, .633), ("celeba", "bedroom"): (21.98, .477, .588),
    ("church", "bedroom"): (22.04, .422, .614)}
P13 = {  # Table 13, ImageNet: (task, method, model) -> (PSNR, LPIPS, SSIM)
    ("inpaint", "ino", "dit"): (28.44, .394, .727), ("inpaint", "ino", "sd21"): (26.82, .374, .745),
    ("inpaint", "dps", "sd21"): (28.21, .292, .797), ("inpaint", "dsg", "sd21"): (23.33, .503, .597),
    ("gauss", "ino", "dit"): (26.48, .458, .678), ("gauss", "ino", "sd21"): (25.40, .443, .685),
    ("gauss", "dps", "sd21"): (25.49, .388, .728), ("gauss", "dsg", "sd21"): (22.09, .510, .539)}
P17 = {  # Table 17, CelebA-HQ inpainting: (NFE, seconds on L40S, PSNR, LPIPS, SSIM)
    "INO, 333 iterations": (999, 56.34, 31.86, .2039, .8721), "DPS, single run": (1000, 40.44, 32.27, .1449, .8851),
    "INO, 1000 iterations": (3000, 171.16, 34.13, .1485, .9174), "DPS, best of 3": (3000, 120.40, 32.36, .1442, .8855),
    "DPS, best of 5": (5000, 200.04, 32.40, .1436, .8864)}


def htk_at(logs, h, K=5):
    """Mean PSNR of HoldoutTopK (latest of the K lowest holdout losses) restricted to the first h iterations."""
    out = []
    for f in logs:
        z = np.load(f)
        ho, ps = z["holdout"][:h, 0], z["psnr"][:h, 0]
        out.append(ps[np.argsort(ho, kind="stable")[:K].max()])
    return np.mean(out)


def cell(method, task, target, prior=None, sigma=0.01, suffix=""):
    d = os.path.join(RUNS, f"{method}_{task}_{target}_from-{prior or target}_sigma{sigma:g}{suffix}")
    res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "[0-9][0-9][0-9].json")))]
    logs = sorted(glob.glob(os.path.join(d, "[0-9][0-9][0-9]_log.npz")))
    return res, logs


def vals(res, rule, metric):
    v = np.array([r[rule][metric] for r in res if rule in r], dtype=float)
    return v[np.isfinite(v)]          # a diverged DPS sample (NaN) is counted in n but not in the mean


def fmt(res, rule, metric, ref=None, nd=2):
    v = vals(res, rule, metric)
    if len(v) == 0:
        s = "—"
    else:
        s = f"{v.mean():.{nd}f}±{v.std(ddof=1) / np.sqrt(len(v)):.{nd}f}" if len(v) > 1 else f"{v[0]:.{nd}f}"
    return s + (f" [{ref:.{nd}f}]" if ref is not None else "")


def triple(res, rule, ref):
    ref = ref or (None, None, None)
    return [fmt(res, rule, "psnr", ref[0]), fmt(res, rule, LP, ref[1]), fmt(res, rule, "ssim", ref[2])]


def table3(lines, tasks):
    hdr = "| Task | Method | Model | " + " | ".join(f"{t} PSNR | LPIPS | SSIM" for t in T) + " | n |"
    lines += [hdr, "|" + "---|" * (3 + 3 * len(T) + 1)]
    for task in tasks:
        for method, prior in [("dps", "in"), ("ino", "celeba"), ("ino", "bedroom"), ("ino", "church")]:
            row, ns = [task, "DPS" if method == "dps" else "Ours", prior], []
            for j, tgt in enumerate(T):
                res, _ = cell(method, task, tgt, None if prior == "in" else prior, suffix=ZETA_NL if (method, task) == ("dps", "nonlinear") else "")
                row += triple(res, "dps" if method == "dps" else "htk5", P3.get((task, method, prior), [None] * 3)[j])
                ns.append(len(res))
            lines.append("| " + " | ".join(row + ["/".join(map(str, ns))]) + " |")


def main():
    L = ["# Reproduction summary", "",
         "Cells: reproduced mean ± s.e. over the first n test images, [paper value]. "
         f"LPIPS = {LP}. Ours = 3-step DDIM INO with AdamSphere + HoldoutTopK (K = 5).", ""]
    L += ["## Table 3 (inpainting, 4x SR) and Table 6 (Gaussian and nonlinear deblurring)", ""]
    table3(L, ["inpaint", "sr4", "gauss", "nonlinear"])

    L += ["", "## Tables 4 / 7: Ours vs DMPlug (in-domain, same z0)", "",
          "| Task | Target | Ours PSNR | LPIPS | SSIM | DMPlug PSNR | LPIPS | SSIM | n |", "|" + "---|" * 9]
    for task in ["inpaint", "sr4", "gauss", "nonlinear"]:
        for j, tgt in enumerate(T):
            ours, _ = cell("ino", task, tgt)
            dmp, _ = cell("dmplug", task, tgt)
            if not dmp:
                continue
            L.append("| " + " | ".join([task, tgt] + triple(ours, "htk5", P3[(task, "ino", tgt)][j])
                                       + triple(dmp, "wmv", PDMP[task][j]) + [f"{len(ours)}/{len(dmp)}"]) + " |")

    L += ["", "## Table 5: ablation, PSNR (A) Adam+variance (B) Adam+HoldoutTopK (C) AdamSphere+variance "
          "(D) AdamSphere+HoldoutTopK", "",
          "Variance = ES-WMV (W=10, P=100) evaluated on the same trajectories as HoldoutTopK.", "",
          "| Task | Dataset | (A) | (B) | (C) | (D) | n |", "|---|---|---|---|---|---|---|"]
    for (task, tgt), ref in P5.items():
        adam, _ = cell("ino", task, tgt, suffix="_adam")
        sph, _ = cell("ino", task, tgt)
        if not adam:
            continue
        L.append("| " + " | ".join([task, tgt, fmt(adam, "wmv", "psnr", ref[0], 3), fmt(adam, "htk5", "psnr", ref[1], 3),
                                    fmt(sph, "wmv", "psnr", ref[2], 3), fmt(sph, "htk5", "psnr", ref[3], 3),
                                    f"{len(adam)}/{len(sph)}"]) + " |")

    L += ["", "## Table 12: box inpainting on Church (failure mode); Ours(In) = Church prior, Ours(Out) = CelebA prior",
          "", "| Mask | Method | PSNR | LPIPS | SSIM | n |", "|---|---|---|---|---|---|"]
    box = {}
    for s, refs in P12.items():
        for (name, method, prior), ref in zip([("DPS", "dps", None), ("Ours (In)", "ino", None),
                                               ("Ours (Out)", "ino", "celeba")], refs):
            res, _ = cell(method, f"box{s}", "church", prior)
            if res:
                box[(s, name)] = vals(res, "dps" if method == "dps" else "htk5", LP)
                L.append("| " + " | ".join([f"{s}x{s}", name] + triple(res, "dps" if method == "dps" else "htk5", ref)
                                           + [str(len(res))]) + " |")
    if box:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for name, c in [("Ours (In)", "C0"), ("Ours (Out)", "C3")]:
            xs = [s for s in P12 if (s, name) in box and (s, "DPS") in box]
            n = [min(len(box[(s, name)]), len(box[(s, "DPS")])) for s in xs]
            d = [np.mean(box[(s, name)][:k] - box[(s, "DPS")][:k]) for s, k in zip(xs, n)]
            ax.plot(xs, d, "o-", color=c, label=f"{name} − DPS (repro)")
            ax.plot(list(P12), [P12[s][1 if name == 'Ours (In)' else 2][1] - P12[s][0][1] for s in P12], "x--",
                    color=c, alpha=0.6, label=f"{name} − DPS (paper)")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set(xlabel="box side (fraction of image)", ylabel=f"Δ LPIPS ({LP})", title="Figure 4: box inpainting, Church")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "results", "fig4_box_lpips.png"), dpi=150)
        plt.close(fig)

    L += ["", "## 16x super-resolution on Church (Figure 3/15 failure mode, no paper numbers)", "",
          "| Method | PSNR | LPIPS | SSIM | n |", "|---|---|---|---|---|"]
    for name, method, prior in [("DPS", "dps", None), ("Ours (In)", "ino", None), ("Ours (Out)", "ino", "celeba")]:
        res, _ = cell(method, "sr16", "church", prior)
        if res:
            L.append("| " + " | ".join([name] + triple(res, "dps" if method == "dps" else "htk5", None)
                                       + [str(len(res))]) + " |")

    L += ["", "## Table 16: noise level, CelebA inpainting; INO out-of-domain = Church prior", "",
          "| σ | Method | PSNR | LPIPS | SSIM | selected iter | n |", "|---|---|---|---|---|---|---|"]
    for sg, refs in P16.items():
        for (name, method, prior), ref in zip([("DPS", "dps", None), ("INO in", "ino", None),
                                               ("INO out", "ino", "church")], refs):
            res, _ = cell(method, "inpaint", "celeba", prior, sigma=sg)
            if res:
                rule = "dps" if method == "dps" else "htk5"
                it = "—" if method == "dps" else fmt(res, rule, "iter", ref[3], 0)
                L.append("| " + " | ".join([f"{sg:g}", name] + triple(res, rule, ref[:3]) + [it, str(len(res))]) + " |")

    L += ["", "## HoldoutTopK K-sensitivity and other selection rules (mean PSNR over all INO AdamSphere cells, "
          "σ = 0.01)", ""]
    rules = ["htk1", "htk3", "htk5", "htk10", "htk20", "htk50", "wmv", "last", "oracle", "iter1"]
    L += ["| cell | n | " + " | ".join(rules) + " | sel. iter (htk5) |", "|" + "---|" * (len(rules) + 3)]
    for d in sorted(glob.glob(os.path.join(RUNS, "ino_*_sigma0.01"))):
        res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "[0-9][0-9][0-9].json")))]
        if res:
            L.append("| " + " | ".join([os.path.basename(d), str(len(res))] + [fmt(res, r, "psnr") for r in rules]
                                       + [fmt(res, "htk5", "iter", None, 0)]) + " |")

    L += ["", "## CelebA test-set check: original CelebA test faces (178x218, upsampled to 256) instead of CelebA-HQ 256", "",
          "| Task | Method | Model | CelebA-HQ PSNR (Table 3 above) | orig. CelebA PSNR [paper CelebA] | LPIPS | SSIM | n |",
          "|" + "---|" * 8]
    for task in ["inpaint", "sr4", "gauss"]:
        for method, prior in [("dps", "in"), ("ino", "celeba"), ("ino", "church")]:
            res, _ = cell(method, task, "celeba178", "celeba" if prior == "in" else prior)
            if res:
                rule, hq = ("dps" if method == "dps" else "htk5"), cell(method, task, "celeba", None if prior == "in" else prior)[0]
                L.append("| " + " | ".join([task, "DPS" if method == "dps" else "Ours", prior, fmt(hq, rule, "psnr")]
                                           + triple(res, rule, P3[(task, method, prior)][0]) + [str(len(res))]) + " |")

    L += ["", "## Table 15: DPS with matched and mismatched 1000-step priors, nonlinear deblurring", "",
          "Step size zeta' = 0.3 (chosen on the first 20 images of Church and Bedroom from {1, 0.3, 0.1}). With the DPS "
          "paper's zeta' = 1, some images diverge to stripe artefacts: last two columns (mean over finite PSNRs).", "",
          "| Target | Source model | PSNR | LPIPS | SSIM | finite / n | PSNR, zeta' = 1 | images < 12 dB or NaN, zeta' = 1 |",
          "|---|---|---|---|---|---|---|---|"]
    for tgt in T:
        for src in [tgt] + [x for x in T if x != tgt]:
            res, _ = cell("dps", "nonlinear", tgt, src, suffix=ZETA_NL)
            r1, _ = cell("dps", "nonlinear", tgt, src)
            p1 = np.array([r["dps"]["psnr"] for r in r1], dtype=float)
            extra = [f"{np.nanmean(p1):.2f}", f"{int((~(p1 >= 12)).sum())}/{len(p1)}"] if len(p1) else ["", ""]
            if res:
                nf = len(vals(res, "dps", "psnr"))
                L.append("| " + " | ".join([tgt, src] + triple(res, "dps", P15[(src, tgt)]) + [f"{nf}/{len(res)}"] + extra)
                         + " |")

    L += ["", "## Table 13: few-step latent priors on ImageNet 512 (repro/latent.py)", "",
          "Ours = 3-step DDIM in latent space + VAE decoder, INO 500 iterations, HoldoutTopK K = 5. DPS / DSG: SD 2.1, "
          "500 Euler-ancestral steps (third-party implementation cited by the paper).", "",
          "| Task | Method | Model | PSNR | LPIPS | SSIM | n |", "|---|---|---|---|---|---|---|"]
    for (task, method, model), ref in P13.items():
        res, _ = cell(method, task, "imagenet", model)
        if res:
            L.append("| " + " | ".join([task, "Ours" if method == "ino" else method.upper(), model]
                                       + triple(res, "htk5" if method == "ino" else method, ref) + [str(len(res))]) + " |")

    L += ["", "## Table 17: matched compute, CelebA-HQ inpainting", "",
          "DPS best of k: of k independent DPS runs on the same measurement, the one with the smallest residual "
          "||A(x) - y||, which needs no ground truth; 'oracle' picks the highest PSNR instead. The paper does not say "
          "how it picks. Seconds: ours on GH200 (batched), [paper on L40S]. INO at 333 iterations: PSNR only.", "",
          "| Method | NFE | seconds / image | PSNR | LPIPS | SSIM | oracle PSNR | n |", "|---|---|---|---|---|---|---|---|"]
    ino, logs = cell("ino", "inpaint", "celeba")
    reps = [cell("dps", "inpaint", "celeba", suffix=f"_rep{r}")[0] for r in range(5)]
    if ino and all(len(r) == len(ino) for r in reps):
        sec_ino, sec_dps = np.mean([r["seconds"] for r in ino]), np.mean([r["seconds"] for r in reps[0]])
        for name, (nfe, psec, pp, pl, pss) in P17.items():
            if name.startswith("INO"):
                h = int(name.split()[1])
                tri = ([f"{htk_at(logs, h):.2f} [{pp:.2f}]", f"— [{pl:.2f}]", f"— [{pss:.2f}]"] if h < 1000
                       else triple(ino, "htk5", (pp, pl, pss)))
                row, sec, orc = tri, sec_ino * h / 1000, ""
            else:
                k = 1 if "single" in name else int(name.split()[-1])
                pick = [min(rs, key=lambda r: r["dps"]["resid"]) for rs in zip(*reps[:k])]
                orc = f"{np.mean([max(r['dps']['psnr'] for r in rs) for rs in zip(*reps[:k])]):.2f}"
                row, sec = triple(pick, "dps", (pp, pl, pss)), sec_dps * k
            L.append("| " + " | ".join([name, str(nfe), f"{sec:.0f} [{psec:.0f}]"] + row + [orc, str(len(ino))]) + " |")

    L += ["", "## Compute (fp16 + torch.compile; B = images per GPU pass, peak MiB is per pass)", "",
          "| cell type | device | B | mean seconds / image | peak MiB | INO PSNR@333 it (htk5 on first 333) | @1000 |",
          "|---|---|---|---|---|---|---|"]
    for method, task, tgt in [("ino", "inpaint", "celeba"), ("dps", "inpaint", "celeba"), ("dmplug", "inpaint", "celeba")]:
        res, logs = cell(method, task, tgt)
        if not res:
            continue
        cut = [f"{htk_at(logs, 333):.2f}", f"{htk_at(logs, 1000):.2f}"] if logs and method == "ino" else ""
        L.append("| " + " | ".join([f"{method} {task} {tgt}", "/".join(sorted({r.get("device", "RTX A1000") for r in res})),
                                    "/".join(sorted({str(r.get("batch", 1)) for r in res}, key=int)),
                                    f"{np.mean([r['seconds'] for r in res]):.0f}",
                                    f"{np.mean([r['peak_mib'] for r in res]):.0f}"] + (cut or ["", ""])) + " |")
    open(os.path.join(ROOT, "results", "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
