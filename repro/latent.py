"""Table 13 / Sec. 4.4: few-step latent diffusion priors (Stable Diffusion 2.1, DiT) on ImageNet 512, vs latent DPS / DSG.

  python -s repro/latent.py setup                                   # once, with internet + transformers: models, null prompt
  python -s repro/latent.py ino --model sd21 --task inpaint --n 100 --batch 4
  python -s repro/latent.py ino --model dit  --task gauss   --n 100 --batch 4
  python -s repro/latent.py dps --task inpaint --n 100 --batch 4    # SD 2.1, 500 Euler-ancestral steps
  python -s repro/latent.py dsg --task gauss   --n 100 --batch 4

Paper (App. C.7, C.12): 500 optimisation steps; lr 0.2 (SD 2.1) / 0.1 (DiT); null prompt / class 1000; DPS and DSG with
500 steps and the default configs of github.com/tongdaxu/diffusers-Diffusion-Posterior-Sampling (third-party code, read
for this): Euler-ancestral sampler, no classifier-free guidance, DPS scale 4.8, DSG guidance rate 0.02.
Not stated, our choices: 3 DDIM steps ([666, 333, 0], as in the pixel experiments); 100 random ImageNet val images,
512 px; Table 18 operators and sigma = 0.01 (as for the pixel tasks); HoldoutTopK K = 5 with the Table 19 holdout share.
Output: results/runs/<method>_<task>_<target>_from-<model>_sigma0.01/, same format as run.py.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp  # noqa: E402
from run import ROOT, TASKS, save_png  # noqa: E402

SD, DIT = "Manojb/stable-diffusion-2-1-base", "facebook/DiT-XL-2-512"   # App. C.12 / the official 512 DiT
NULL = os.path.join(ROOT, "data", "latent", "sd21_null_prompt.pt")     # text embedding of "" (needs transformers once)
LR = {"sd21": 0.2, "dit": 0.1}
SCALE = {"dps": 4.8, "dsg": 0.02}
DEV = wp.DEV
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True   # fp32 weights, TF32 math (as the repo)


def setup():
    from huggingface_hub import snapshot_download
    snapshot_download(SD, allow_patterns=["model_index.json", "scheduler/*", "unet/config.json", "vae/config.json",
                                          "unet/diffusion_pytorch_model.safetensors",
                                          "vae/diffusion_pytorch_model.safetensors"])
    snapshot_download(DIT, allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*"])
    from diffusers import StableDiffusionPipeline
    p = StableDiffusionPipeline.from_pretrained(SD, safety_checker=None, requires_safety_checker=False)
    with torch.no_grad():
        emb = p.encode_prompt("", "cpu", 1, False)[0]
    os.makedirs(os.path.dirname(NULL), exist_ok=True)
    torch.save(emb, NULL)
    print("null prompt", tuple(emb.shape), "->", NULL)


class Prior:
    """eps(x_t, t) and decode(z) of a latent diffusion model; weights frozen, fp32."""

    def __init__(self, name):
        from diffusers import AutoencoderKL, DDPMScheduler
        repo = SD if name == "sd21" else DIT
        self.name = name
        self.vae = AutoencoderKL.from_pretrained(repo, subfolder="vae").to(DEV).eval().requires_grad_(False)
        self.abar = DDPMScheduler.from_pretrained(repo, subfolder="scheduler").alphas_cumprod.to(DEV)
        if name == "sd21":
            from diffusers import UNet2DConditionModel
            self.net = UNet2DConditionModel.from_pretrained(repo, subfolder="unet")
            self.emb = torch.load(NULL, map_location=DEV)                    # (1, 77, 1024)
        else:
            from diffusers import DiTTransformer2DModel
            self.net = DiTTransformer2DModel.from_pretrained(repo, subfolder="transformer")
        self.net = self.net.to(DEV).eval().requires_grad_(False)

    def eps(self, x, t):
        tt = torch.full((x.shape[0],), int(t), device=DEV)
        if self.name == "sd21":
            return self.net(x, tt, encoder_hidden_states=self.emb.expand(x.shape[0], -1, -1)).sample
        y = torch.full((x.shape[0],), 1000, device=DEV)                       # class 1000 = the null class
        return self.net(x, timestep=tt, class_labels=y).sample[:, :4]        # learned sigma: first 4 channels = eps

    def decode(self, z):
        return self.vae.decode(z / self.vae.config.scaling_factor).sample


def generator(m, k=3):
    """G(z): deterministic k-step DDIM in latent space, then the VAE decoder (no latent clipping)."""
    ts = wp.ddim_timesteps(k)

    def gen(z):
        x = z
        for i, t in enumerate(ts):
            a = m.abar[t]
            ap = m.abar[ts[i + 1]] if i + 1 < k else torch.ones_like(a)
            e = m.eps(x, t)
            x = ap.sqrt() * (x - (1 - a).sqrt() * e) / a.sqrt() + (1 - ap).sqrt() * e
        return m.decode(x)
    return gen


def posterior_sample(m, task, seeds, algo, steps=500):
    """Latent DPS / DSG with the Euler-ancestral sampler, as in the tongdaxu repo (one guidance step per sampler step).

    DPS: z <- z_next - scale * grad_z ||A(D(z0_hat)) - y||.  DSG (Yang et al. 2024): replace the sampler noise
    d_sample by the mix d_sample + g (d* - d_sample), d* = -r grad / ||grad||, rescaled to r = sqrt(n) sigma_up.
    """
    from diffusers import EulerAncestralDiscreteScheduler
    s = EulerAncestralDiscreteScheduler.from_pretrained(SD, subfolder="scheduler")
    s.set_timesteps(steps, device=DEV)
    gens = [torch.Generator(device=DEV).manual_seed(int(x) + 104729) for x in seeds]
    shape = (4, 64, 64)
    z = torch.stack([torch.randn(shape, generator=g, device=DEV) for g in gens]) * s.init_noise_sigma
    sig = s.sigmas.to(DEV)
    for i, t in enumerate(s.timesteps):
        sg, sn = sig[i], sig[i + 1]
        z.requires_grad_(True)
        z0 = z - sg * m.eps(z / (sg ** 2 + 1).sqrt(), t)
        nrm = ((task.A(m.decode(z0)) - task.y) * task.obs).flatten(1).norm(dim=1)
        grad, = torch.autograd.grad(nrm.sum(), z)                  # per image: images do not interact
        with torch.no_grad():
            up = (sn ** 2 * (sg ** 2 - sn ** 2) / sg ** 2).sqrt()
            mean = z + (z - z0) / sg * ((sn ** 2 - up ** 2).sqrt() - sg)
            noise = up * torch.stack([torch.randn(shape, generator=g, device=DEV) for g in gens])
            if algo == "dps":
                z = mean + noise - SCALE["dps"] * grad
            else:
                r = math.sqrt(math.prod(shape)) * up
                d_star = -r * grad / (grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8)
                mix = noise + SCALE["dsg"] * (d_star - noise)
                z = mean + mix / (mix.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-8) * r
        z = z.detach()
    with torch.no_grad():
        return m.decode(z).clamp(-1, 1)


def load(target, i):
    a = np.asarray(Image.open(os.path.join(ROOT, "data", "test", target, f"{i:03d}.png")).convert("RGB"), np.float32)
    return torch.from_numpy(a / 127.5 - 1).permute(2, 0, 1)[None].to(DEV)


def main():
    if sys.argv[1:] == ["setup"]:
        return setup()
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=["ino", "dps", "dsg"])
    p.add_argument("--model", default="sd21", choices=list(LR), help="prior of ours (DPS / DSG always use SD 2.1)")
    p.add_argument("--task", required=True, choices=["inpaint", "gauss"])
    p.add_argument("--target", default="imagenet")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, help="diagnostic: override the learning rate (name it with --tag)")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    model = a.model if a.method == "ino" else "sd21"
    lr, hfrac = a.lr or LR[model], wp.TASK_HP[a.task][1]
    name = f"{a.method}_{a.task}_{a.target}_from-{model}_sigma0.01" + (f"_{a.tag}" if a.tag else "")
    out = os.path.join(ROOT, "results", "runs", name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(a), "model": model, "lr": lr, "holdout": hfrac,
                   "scale": SCALE.get(a.method), "steps": a.iters if a.method == "ino" else 500}, f, indent=1)
    m = Prior(model)
    gen = generator(m)
    todo = [i for i in range(a.n) if not os.path.exists(os.path.join(out, f"{i:03d}.json"))]
    for c in range(0, len(todo), a.batch):
        idx = todo[c:c + a.batch]
        seeds = [1000 * TASKS.index(a.task) + i for i in idx]
        gt = torch.cat([load(a.target, i) for i in idx])
        task = wp.make_task(a.task, gt, seeds)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if a.method == "ino":
            outs, log = wp.ino(None, task, gt, seeds, iters=a.iters, lr=lr, holdout=hfrac, gen=gen,
                               zshape=(4, 64, 64), snaps=(0, 4, 9, 99, a.iters - 1))
            rule = "htk5"
        else:
            outs, log, rule = {a.method: (posterior_sample(m, task, seeds, a.method), [499] * len(idx))}, {}, a.method
        torch.cuda.synchronize()
        info = {"seconds": (time.time() - t0) / len(idx), "peak_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                "batch": len(idx), "device": torch.cuda.get_device_name()}
        mets = {r: wp.metrics(x, gt) for r, (x, _) in outs.items()}
        for b, i in enumerate(idx):
            if log:
                np.savez_compressed(os.path.join(out, f"{i:03d}_log.npz"),
                                    **{k: np.asarray(v, dtype=np.float32)[:, b:b + 1] for k, v in log.items() if v})
            res = dict(info)
            for r, (x, its) in outs.items():
                res[r] = {k: v[b] for k, v in mets[r].items()} | {"iter": int(its[b])}
            save_png(outs[rule][0][b:b + 1], os.path.join(out, f"{i:03d}.png"))
            if i < 3:
                save_png(task.y[b:b + 1], os.path.join(out, f"{i:03d}_measurement.png"))
            with open(os.path.join(out, f"{i:03d}.json"), "w") as f:
                json.dump(res, f, indent=1)
            mm = res[rule]
            print(f"{name} img {i}: PSNR {mm['psnr']:.2f} SSIM {mm['ssim']:.3f} LPIPS(alex/vgg) "
                  f"{mm['lpips_alex']:.3f}/{mm['lpips_vgg']:.3f} iter {mm['iter']} | {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
