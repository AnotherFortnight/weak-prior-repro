"""Quick checks of wp.py (about 1 min): operators, holdout split, AdamSphere geometry, fp16 gradient accuracy.

Run: python -s repro/selftest.py
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp  # noqa: E402


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x, t):
        return SimpleNamespace(sample=self.c(x))


def main():
    torch.manual_seed(0)
    gt = torch.rand(2, 3, 256, 256, device=wp.DEV) * 2 - 1
    t = wp.make_task("inpaint", gt, [1, 2])
    assert torch.all(t.obs.flatten(1).sum(1) == 256 * 256 - int(0.7 * 256 * 256))
    assert not torch.equal(t.obs[0], t.obs[1])
    assert (t.y * (1 - t.obs)).abs().max() == 0
    assert abs(float((t.y - gt * t.obs).std()) / 0.01 / t.obs.mean().sqrt() - 1) < 0.05   # noise level
    fit, ho = wp.split_holdout(t.obs, 0.1, t.gens)
    assert torch.all(fit * ho == 0) and torch.equal(fit + ho, t.obs)
    assert abs(float(ho.sum() / t.obs.sum()) - 0.1) < 1e-3
    assert wp.make_task("sr4", gt, [1, 2]).y.shape == (2, 3, 64, 64)
    assert wp.make_task("sr16", gt, [1, 2]).y.shape == (2, 3, 16, 16)
    tb = wp.make_task("box0.5", gt, [1, 2])
    assert tb.obs[0, 0].sum() == 256 * 256 - 128 * 128 and tb.obs[0, 0, 128, 128] == 0 and tb.obs[0, 0, 0, 0] == 1
    tg = wp.make_task("gauss", gt, [1, 2])
    assert abs(float(wp.gaussian_kernel().sum()) - 1) < 1e-6 and tg.y.shape == gt.shape
    const = torch.full_like(gt, 0.3)
    assert torch.allclose(tg.A(const), const, atol=1e-5)                # blur preserves constants
    # same seed -> same measurement (fair comparison across methods)
    assert torch.equal(wp.make_task("inpaint", gt, [1, 2]).y, t.y)
    tn = wp.make_task("nonlinear", gt, [1, 2])                           # BKSE blur: fixed per seed, differentiable
    assert torch.allclose(wp.make_task("nonlinear", gt, [1, 2]).y, tn.y, atol=1e-4)    # GPU convs: not bit-exact
    assert tn.y.shape == gt.shape
    x = gt.clone().requires_grad_(True)
    tn.A(x).square().sum().backward()
    assert x.grad.abs().sum() > 0
    # AdamSphere geometry
    z, u = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    r2 = (z.flatten(1).norm(dim=1) ** 2).view(2, 1, 1, 1)
    assert (wp._tangent(u, z, r2) * z).flatten(1).sum(1).abs().max() < 1e-4
    # INO end to end with a dummy generator: all selection rules returned, selected iterations valid
    out, log = wp.ino(Dummy().to(wp.DEV), t, gt, [1, 2], iters=30, lr=0.02, wmv=(5, 10), ks=(1, 5))
    assert {"last", "oracle", "wmv", "htk1", "htk5", "iter1", "iter10"} <= set(out), out.keys()
    assert all(0 <= i < 30 for _, (x, its) in out.items() for i in its) and len(log["holdout"]) == 30
    assert wp.psnr(gt, gt)[0] == float("inf") and abs(wp.psnr(gt * 0 + 1, gt * 0 - 1)[0]) < 1e-6
    # fp16 + torch.compile gradient vs fp32 eager gradient of the INO loss
    z = torch.randn(1, 3, 256, 256, device=wp.DEV)
    ti = wp.make_task("inpaint", gt[:1], [3])
    grads = []
    for half, comp in [(False, False), (True, True)]:
        zz = z.clone().requires_grad_(True)
        x = wp.ddim(wp.unet("celeba", half, comp), zz)
        g, = torch.autograd.grad(((ti.A(x) - ti.y) ** 2).sum(), zz)
        grads.append(g.flatten())
    cos = float(torch.nn.functional.cosine_similarity(grads[0], grads[1], dim=0))
    rel = float((grads[0] - grads[1]).norm() / grads[0].norm())
    print(f"fp16+compile vs fp32 INO gradient: cosine {cos:.5f}, rel. error {rel:.4f}")
    assert cos > 0.99 and torch.isfinite(grads[1]).all()
    print("selftest passed")


if __name__ == "__main__":
    main()
