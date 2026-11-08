"""Run one experiment cell (method x task x target x prior) on test images [start, n); resumable.

  python -s repro/run.py ino    --task inpaint --target celeba --prior bedroom --n 5   # ours, mismatched prior
  python -s repro/run.py ino    --task inpaint --target church --opt adam --n 5        # ablation: plain Adam
  python -s repro/run.py dps    --task sr4     --target church --n 5                  # 1000-step DPS, in-domain
  python -s repro/run.py dmplug --task gauss   --target bedroom --n 5                 # Adam lr 1e-2 + ES-WMV
  python -s repro/run.py ino    --task sr4     --target church --n 100 --batch 16     # large GPU: 16 images per pass

Writes results/runs/<cell>/NNN.json (metrics of every selection rule), NNN_log.npz (per-iteration fit / holdout
loss and PSNR) and NNN.png (main output: htk5 for ours, wmv for DMPlug). Batched images are optimised
independently (per-image seeds, losses, Adam moments and selection rules), so --batch only changes speed.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["inpaint", "sr4", "gauss", "sr16", "box0.3", "box0.4", "box0.5", "box0.6", "nonlinear"]  # append only: the index sets the seeds
MAIN_RULE = {"ino": "htk5", "dmplug": "wmv", "dps": "dps"}


def load(target, i):
    a = np.asarray(Image.open(os.path.join(ROOT, "data", "test", target, f"{i:03d}.png")), dtype=np.float32)
    return torch.from_numpy(a / 127.5 - 1).permute(2, 0, 1)[None].to(wp.DEV)


def save_png(x, path):
    a = ((x[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    Image.fromarray(a).save(path)


def cell_name(a):
    prior = a.prior or a.target
    s = f"{a.method}_{a.task}_{a.target}_from-{prior}_sigma{a.sigma:g}"
    if a.method == "ino" and a.opt != "adamsphere":
        s += f"_{a.opt}"
    if a.method == "dps" and a.zeta != 1.0:
        s += f"_zeta{a.zeta:g}"
    return s + (f"_{a.tag}" if a.tag else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=list(MAIN_RULE))
    p.add_argument("--task", required=True, choices=TASKS)
    # celeba178 = original CelebA test faces (test-set check); it has no model of its own, so pass --prior
    p.add_argument("--target", required=True, choices=list(wp.MODEL_ID) + ["celeba178"])
    p.add_argument("--prior", choices=list(wp.MODEL_ID), help="source domain of the diffusion model (default: target)")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--opt", default="adamsphere", choices=["adamsphere", "adam"])
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--zeta", type=float, default=1.0)
    p.add_argument("--tag", default="")
    p.add_argument("--batch", type=int, default=1, help="images per GPU pass (about 4.3 GB each); DMPlug runs 1")
    a = p.parse_args()

    prior = a.prior or a.target
    lr, hfrac = wp.TASK_HP[a.task]
    opt = a.opt
    if a.method == "dmplug":                     # DMPlug defaults (Wang et al. 2024, App. D.2)
        opt, lr, hfrac = "adam", 0.01, 0.0
    out = os.path.join(ROOT, "results", "runs", cell_name(a))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(a), "prior": prior, "opt": opt, "lr": lr, "holdout": hfrac}, f, indent=1)

    torch.backends.cudnn.benchmark = True
    model = wp.unet(prior)
    # shortcut: DMPlug stays at batch 1 - batched, its loop runs until every image has stopped, which changes its
    # 'last' / 'oracle' rules and logs; batch it (freeze stopped images) only if DMPlug becomes the bottleneck
    bs = 1 if a.method == "dmplug" else a.batch
    todo = [i for i in range(a.start, a.n) if not os.path.exists(os.path.join(out, f"{i:03d}.json"))]
    for c in range(0, len(todo), bs):
        idx = todo[c:c + bs]
        seeds = [1000 * TASKS.index(a.task) + i for i in idx]   # same mask / noise / z0 for every method and prior
        gt = torch.cat([load(a.target, i) for i in idx])
        task = wp.make_task(a.task, gt, seeds, a.sigma)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if a.method == "dps":
            outs = {"dps": (wp.dps(model, task, gt.shape, seeds, a.zeta), [999] * len(idx))}
        else:
            outs, log = wp.ino(model, task, gt, seeds, iters=a.iters, lr=lr, opt=opt, holdout=hfrac,
                               stop_on_wmv=a.method == "dmplug")
        torch.cuda.synchronize()
        info = {"seconds": (time.time() - t0) / len(idx),          # per image (batch time / batch size)
                "peak_mib": torch.cuda.max_memory_allocated() / 2 ** 20,   # whole batch
                "batch": len(idx), "device": torch.cuda.get_device_name()}
        mets = {rule: wp.metrics(x, gt) for rule, (x, _) in outs.items()}
        meas = task.y if task.y.shape == gt.shape else \
            torch.nn.functional.interpolate(task.y, size=gt.shape[-2:], mode="nearest")
        for b, i in enumerate(idx):
            if a.method != "dps":
                np.savez_compressed(os.path.join(out, f"{i:03d}_log.npz"),
                                    **{k: np.asarray(v, dtype=np.float32)[:, b:b + 1] for k, v in log.items() if v})
            res = dict(info)
            for rule, (x, its) in outs.items():
                res[rule] = {k: v[b] for k, v in mets[rule].items()} | {"iter": int(its[b])}
            main_x = outs[MAIN_RULE[a.method]][0][b:b + 1]
            save_png(main_x, os.path.join(out, f"{i:03d}.png"))
            if i < 3:
                save_png(meas[b:b + 1], os.path.join(out, f"{i:03d}_measurement.png"))
                strip = [outs[k][0][b:b + 1] for k in ("iter1", "iter5", "iter10", "iter100", "iter1000") if k in outs]
                save_png(torch.cat(strip + [main_x, gt[b:b + 1], meas[b:b + 1]], dim=3),
                         os.path.join(out, f"{i:03d}_strip.png"))      # Figure-1 style: trajectory | output | ref | y
            with open(os.path.join(out, f"{i:03d}.json"), "w") as f:
                json.dump(res, f, indent=1)
            m = res[MAIN_RULE[a.method]]
            print(f"{cell_name(a)} img {i}: PSNR {m['psnr']:.2f} SSIM {m['ssim']:.3f} LPIPS(alex/vgg) "
                  f"{m['lpips_alex']:.3f}/{m['lpips_vgg']:.3f} iter {m['iter']} | {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
