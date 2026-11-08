"""Core of the reproduction, written from the paper text only (arXiv 2601.22443v2, Sec. 2-4, App. B, C.12).

* pretrained google/ddpm-*-256 UNets (App. C.12) and a k-step deterministic DDIM generator G(z)
* forward operators of Table 18 (+ box inpainting / 16x SR of Sec. 4.3), Gaussian noise sigma = 0.01
* ino(): initial-noise optimisation argmin_z ||A(G(z)) - y||^2 with ADAMSPHERE or Adam, and the
  HOLDOUTTOPK / ES-WMV (DMPlug) selection rules
* dps(): diffusion posterior sampling, 1000-step DDPM (Chung et al., ICLR 2023, Alg. 1 + App. D)
"""
import math
import os
import sys
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

DEV = "cuda"
MODEL_ID = {"celeba": "google/ddpm-celebahq-256", "bedroom": "google/ddpm-bedroom-256",
            "church": "google/ddpm-church-256"}
# scheduler_config.json of all three checkpoints: linear betas 1e-4..0.02, T = 1000, fixed_small variance
BETAS = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
ABAR = torch.cumprod(1 - BETAS, 0)
# Table 18/19: task -> (learning rate, holdout fraction of the observed entries)
TASK_HP = {"inpaint": (0.02, 0.10), "gauss": (0.02, 0.20), "sr4": (0.01, 0.05), "sr16": (0.01, 0.05),
           "box0.3": (0.02, 0.10), "box0.4": (0.02, 0.10), "box0.5": (0.02, 0.10), "box0.6": (0.02, 0.10),
           "nonlinear": (0.01, 0.10)}
BKSE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bkse")

_UNETS = {}
_BLUR = []


def blur_model():
    """BKSE KernelWizard F(x, k) with GOPRO_wVAE.pth (Tran et al. 2021; Table 18). x in [0, 1], k: (B, 512, 2, 2)."""
    if not _BLUR:
        sys.path.insert(0, BKSE_DIR)        # git clone https://github.com/VinAIResearch/blur-kernel-space-exploring
        from models.kernel_encoding.kernel_wizard import KernelWizard
        opt = {"input_nc": 3, "nf": 64, "front_RBs": 10, "back_RBs": 20, "N_frames": 1, "kernel_dim": 512,
               "use_vae": True, "Adapter": {"norm": "none", "use_dropout": False},  # options/generate_blur/default.yml
               "KernelExtractor": {"norm": "none", "use_sharp": True, "n_blocks": 4, "padding_type": "reflect",
                                   "use_dropout": False}}
        m = KernelWizard(opt)
        m.load_state_dict(torch.load(os.path.join(BKSE_DIR, "experiments", "pretrained", "GOPRO_wVAE.pth"),
                                     map_location="cpu", weights_only=True))
        _BLUR.append(m.to(DEV).eval().requires_grad_(False))
        # full fp32 for these convs: with TF32 a GH200 gives a different y on each call (max diff 1.3e-3, vs 1.4e-6)
        torch.backends.cudnn.allow_tf32 = False
    return _BLUR[0]


def unet(domain, half=True, compile=True):
    """fp16 weights + torch.compile: 2x faster than fp32 on the RTX A1000; eps rel. error ~4e-4 (same as TF32)."""
    key = (domain, half, compile)
    if key not in _UNETS:
        from diffusers import UNet2DModel
        m = UNet2DModel.from_pretrained(MODEL_ID[domain]).to(DEV).eval().requires_grad_(False)
        m = m.half() if half else m
        _UNETS[key] = torch.compile(m) if compile else m
    return _UNETS[key]


def eps_pred(model, x, t):
    tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
    return model(x.to(next(model.parameters()).dtype), tt).sample.float()


# ---------------------------------------------------------------- k-step DDIM generator
def ddim_timesteps(k, spacing="leading"):
    """'leading' = diffusers DDIMScheduler default ([666, 333, 0] for k=3); 'trailing' = [999, 666, 333]."""
    if spacing == "leading":
        return [i * (1000 // k) for i in reversed(range(k))]
    return [round(1000 - i * 1000 / k) - 1 for i in range(k)]


def ddim(model, z, k=3, spacing="leading", clip=True, ckpt=False):
    """Deterministic DDIM (eta = 0) from x_T = z; returns the clean image of the last step."""
    ts = ddim_timesteps(k, spacing)
    x = z
    for i, t in enumerate(ts):
        a = ABAR[t].item()
        a_prev = ABAR[ts[i + 1]].item() if i + 1 < k else 1.0
        if ckpt:
            eps = torch.utils.checkpoint.checkpoint(eps_pred, model, x, t, use_reentrant=False)
        else:
            eps = eps_pred(model, x, t)
        x0 = (x - math.sqrt(1 - a) * eps) / math.sqrt(a)
        if clip:
            x0 = x0.clamp(-1, 1)
        x = math.sqrt(a_prev) * x0 + math.sqrt(1 - a_prev) * eps
    return x


# ---------------------------------------------------------------- forward operators
def gaussian_kernel(size=61, std=3.0):
    ax = torch.arange(size, dtype=torch.float64) - size // 2
    g = torch.exp(-ax ** 2 / (2 * std ** 2))
    k = torch.outer(g, g)
    return (k / k.sum()).float()


def make_task(name, gt, seeds, sigma=0.01):
    """Build y = A(x) + noise for a batch gt (B,3,H,W) in [-1,1]; one RNG per image so batching is irrelevant.

    Returns A (callable), y, obs (B,1,h,w) {0,1} mask of observed measurement entries, gens (per-image RNGs).
    """
    B, C, H, W = gt.shape
    gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
    if name == "inpaint":                       # 70 % of pixels removed, mask shared by RGB (exact count)
        obs = torch.zeros(B, 1, H * W)
        for b, g in enumerate(gens):
            obs[b, 0, torch.randperm(H * W, generator=g)[: H * W - int(0.7 * H * W)]] = 1
        obs = obs.view(B, 1, H, W).to(DEV)
        A = lambda x: x * obs
    elif name.startswith("box"):                # centred square of side frac * H removed
        s = round(float(name[3:]) * H)
        obs = torch.ones(B, 1, H, W, device=DEV)
        obs[:, :, (H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s] = 0
        A = lambda x: x * obs
    elif name.startswith("sr"):                 # average over non-overlapping f x f blocks (Sec. 3.2)
        f = int(name[2:])
        obs = torch.ones(B, 1, H // f, W // f, device=DEV)
        A = lambda x: F.avg_pool2d(x, f)
    elif name == "gauss":                       # 61 x 61 Gaussian kernel, std 3, reflect padding
        k = gaussian_kernel().to(DEV)[None, None].repeat(C, 1, 1, 1)
        obs = torch.ones(B, 1, H, W, device=DEV)
        A = lambda x: F.conv2d(F.pad(x, (30, 30, 30, 30), mode="reflect"), k, groups=C)
    elif name == "nonlinear":                   # BKSE blur, one kernel code k ~ N(0, 1.2^2 I) per image, as in BKSE's
        ks = torch.stack([1.2 * torch.randn(512, 2, 2, generator=g) for g in gens]).to(DEV)  # generate_blur.py
        obs = torch.ones(B, 1, H, W, device=DEV)
        A = lambda x: blur_model().adaptKernel((x + 1) / 2, ks) * 2 - 1
    else:
        raise ValueError(name)
    with torch.no_grad():
        clean = A(gt)
        noise = torch.stack([torch.randn(clean.shape[1:], generator=g) for g in gens]).to(DEV)
        y = (clean + sigma * noise) * obs
    return SimpleNamespace(name=name, A=A, y=y, obs=obs, gens=gens, sigma=sigma)


def split_holdout(obs, frac, gens):
    """Randomly move `frac` of each image's observed entries into a holdout set (App. B)."""
    ho = torch.zeros_like(obs)
    for b, g in enumerate(gens):
        idx = obs[b].flatten().nonzero().squeeze(1).cpu()
        pick = idx[torch.randperm(len(idx), generator=g)[: round(frac * len(idx))]]
        ho[b].view(-1)[pick.to(obs.device)] = 1
    return obs - ho, ho


# ---------------------------------------------------------------- metrics
def psnr(x, gt):
    mse = (((x.clamp(-1, 1) - gt) / 2) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(1 / mse)).tolist()


def ssim(x, gt):
    from skimage.metrics import structural_similarity
    a = ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().double().numpy()
    b = ((gt + 1) / 2).permute(0, 2, 3, 1).cpu().double().numpy()
    return [structural_similarity(p, q, channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                  use_sample_covariance=False) for p, q in zip(a, b)]


_LPIPS = {}


def lpips(x, gt, net):
    if net not in _LPIPS:
        import lpips as lp
        _LPIPS[net] = lp.LPIPS(net=net, verbose=False).to(DEV).eval()
    with torch.no_grad():
        return _LPIPS[net](x.clamp(-1, 1), gt).flatten().tolist()


def metrics(x, gt):
    return {"psnr": psnr(x, gt), "ssim": ssim(x, gt), "lpips_alex": lpips(x, gt, "alex"),
            "lpips_vgg": lpips(x, gt, "vgg")}


# ---------------------------------------------------------------- initial-noise optimisation
def _tangent(u, z, r2):
    return u - (u * z).flatten(1).sum(1).view(-1, 1, 1, 1) / r2 * z


def ino(model, task, gt, seeds, iters=1000, lr=0.02, opt="adamsphere", holdout=0.1, ks=(1, 3, 5, 10, 20, 50),
        wmv=(10, 100), k_steps=3, spacing="leading", clip=True, ckpt=False, snaps=(0, 4, 9, 99, 999),
        stop_on_wmv=False):
    """argmin_z ||A(G(z)) - y||^2 on the fit entries (Eq. 2).

    opt: 'adamsphere' (App. B: tangent gradient -> Adam moments -> tangent direction -> normalisation retraction,
    radius r = ||z0||) or 'adam'. holdout > 0 enables HOLDOUTTOPK (fit on obs minus holdout); holdout = 0 fits all
    observed entries (variance-based ES baselines). Every rule is evaluated on the same trajectory:
      htk{K}: latest iterate among the K lowest holdout losses; wmv: ES-WMV (window W, patience P) as in DMPlug;
      last: final iterate; oracle: best PSNR w.r.t. the ground truth (analysis only, never used for selection);
      iter{t}: the iterate after t-1 updates (iter1 = unconditional k-step sample), as in Figure 1.
    stop_on_wmv: terminate once ES-WMV has stopped for every image (DMPlug behaviour; 'last' is then that iterate).
    """
    B = gt.shape[0]
    z = torch.stack([torch.randn(gt.shape[1:], generator=torch.Generator().manual_seed(int(s) + 7919))
                     for s in seeds]).to(DEV)
    r2 = (z.flatten(1).norm(dim=1) ** 2).view(B, 1, 1, 1)
    fit, ho = split_holdout(task.obs, holdout, task.gens) if holdout > 0 else (task.obs, None)
    m, v = torch.zeros_like(z), torch.zeros_like(z)
    b1, b2, eps = 0.9, 0.999, 1e-8
    kmax = max(ks)
    top = [[] for _ in range(B)]                       # per image: list of (holdout loss, iteration, image)
    W, P = wmv
    win, var_min = deque(maxlen=W), [math.inf] * B
    wmv_best, wmv_it, wmv_last, wmv_stop = [None] * B, [0] * B, [0] * B, [False] * B
    orc_best, orc_it, orc_psnr = gt.clone(), [0] * B, [-math.inf] * B
    log = {"fit": [], "holdout": [], "psnr": []}
    snap = {}
    nfit = fit.flatten(1).sum(1) * gt.shape[1]
    for it in range(iters):
        z.requires_grad_(True)
        x = ddim(model, z, k_steps, spacing, clip, ckpt)
        res2 = (task.A(x) - task.y) ** 2
        lfit = (res2 * fit).flatten(1).sum(1)          # sum-reduced: scale-free for Adam, safe for fp16 grads
        grad, = torch.autograd.grad(lfit.sum(), z)
        with torch.no_grad():
            x = x.detach().clamp(-1, 1)
            p = psnr(x, gt)
            if it in snaps:
                snap[f"iter{it + 1}"] = (x.clone(), [it] * B)
            log["fit"].append((lfit / nfit).tolist())
            log["psnr"].append(p)
            if ho is not None:
                lho = ((res2 * ho).flatten(1).sum(1) / (ho.flatten(1).sum(1) * gt.shape[1])).tolist()
                log["holdout"].append(lho)
                for b in range(B):
                    if len(top[b]) < kmax or lho[b] < top[b][-1][0]:
                        top[b].append((lho[b], it, x[b].half()))
                        top[b].sort(key=lambda e: e[0])
                        del top[b][kmax:]
            win.append(x)
            if len(win) == W:
                stack = torch.stack(list(win))         # W,B,C,H,W
                var = ((stack - stack.mean(0)) ** 2).flatten(2).sum(2).mean(0).tolist()
                for b in range(B):
                    if wmv_stop[b]:
                        continue
                    if var[b] < var_min[b]:
                        var_min[b], wmv_best[b], wmv_it[b], wmv_last[b] = var[b], x[b].clone(), it, it
                    elif it - wmv_last[b] >= P:
                        wmv_stop[b] = True
            for b in range(B):
                if p[b] > orc_psnr[b]:
                    orc_psnr[b], orc_best[b], orc_it[b] = p[b], x[b], it
            g = _tangent(grad, z, r2) if opt == "adamsphere" else grad
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            d = (m / (1 - b1 ** (it + 1))) / ((v / (1 - b2 ** (it + 1))).sqrt() + eps)
            if opt == "adamsphere":
                d = _tangent(d, z, r2)
                z = z - lr * d
                z = z * (r2.sqrt() / z.flatten(1).norm(dim=1).view(B, 1, 1, 1))
            else:
                z = z - lr * d
        z = z.detach()
        if stop_on_wmv and all(wmv_stop):
            break
    out = {**snap, "last": (x, [it] * B), "oracle": (orc_best, orc_it),
           "wmv": (torch.stack([wb if wb is not None else x[b] for b, wb in enumerate(wmv_best)]), wmv_it)}
    if ho is not None:
        for K in ks:
            sel = [max(t[:K], key=lambda e: e[1]) for t in top]
            out[f"htk{K}"] = (torch.stack([s[2].float() for s in sel]), [s[1] for s in sel])
    return out, log


# ---------------------------------------------------------------- DPS baseline
def dps(model, task, gt_shape, seeds, zeta=1.0, loss_scale=1024.0):
    """DPS (Gaussian), 1000 ancestral DDPM steps with fixed_small variance.

    x_{i-1} = x'_{i-1} - zeta * grad_{x_i} ||y - A(x0_hat)||  (zeta_i = zeta'/||y - A(x0_hat)|| applied to the squared
    norm, up to the conventional factor 2), x0_hat clipped to [-1, 1]; zeta' = 1 for every task (DPS App. D, FFHQ).
    """
    B = gt_shape[0]
    gens = [torch.Generator(device=DEV).manual_seed(int(s) + 104729) for s in seeds]
    x = torch.stack([torch.randn(gt_shape[1:], generator=g, device=DEV) for g in gens])
    for t in reversed(range(1000)):
        a, beta = ABAR[t].item(), BETAS[t].item()
        a_prev = ABAR[t - 1].item() if t > 0 else 1.0
        x.requires_grad_(True)
        eps = eps_pred(model, x, t)
        x0 = ((x - math.sqrt(1 - a) * eps) / math.sqrt(a)).clamp(-1, 1)
        nrm = ((task.A(x0) - task.y) * task.obs).flatten(1).norm(dim=1)
        grad, = torch.autograd.grad(nrm.sum() * loss_scale, x)
        with torch.no_grad():
            grad = torch.nan_to_num(grad / loss_scale)
            mean = (math.sqrt(a_prev) * beta / (1 - a)) * x0 + (math.sqrt(1 - beta) * (1 - a_prev) / (1 - a)) * x
            if t > 0:
                noise = torch.stack([torch.randn(gt_shape[1:], generator=g, device=DEV) for g in gens])
                mean = mean + math.sqrt(beta * (1 - a_prev) / (1 - a)) * noise
            x = (mean - zeta * grad).detach()
    return x0.detach()
