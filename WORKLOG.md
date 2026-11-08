# WORKLOG

## 2026-11-06 — Synthetic 1/f^β target images
- **What changed:**
  - Added `gen_synthetic_targets.py`. It generates 9 β × 20 Gaussian random-field images (3×256×256) as .npy and .png.
  - The script also verifies the power-spectrum slopes, computes the radial autocorrelation (lags 0–32), and compares it with the paper's Table 2.
  - The script, figures, CSVs and README are written to `synthetic_targets/`.
- **Why:** to test whether the weak diffusion prior fails as the target's local correlation decreases. This step covers target generation only; no diffusion model is run.
- **Files touched:** `gen_synthetic_targets.py` (new), `synthetic_targets/**` (new, ~170 MB), `WORKLOG.md` (new). `main.py` is untouched.
- **Verification:** `python3 gen_synthetic_targets.py` prints "self-checks passed".
  - The asserts check that |slope + β| < 0.25, that lag-1 is ≈0 at β=0 and > 0.99 at β=4, and that lag-1 decreases monotonically with β.
  - Measured: fitted slopes are within 0.02 of -β, and at most 0.27% of pixels are clipped.
- **Result:** the paper's prior autocorrelation lies between β=2.5 (lowest RMSE, 0.058) and β=3.0 (highest Pearson, 0.997).
- **Risks/TODO:**
  - Autocorrelation is computed on the clipped images; clipping is ≤0.27% of pixels.
  - Seeds are deterministic but depend on the β index, so changing the order of the β list changes the images.
  - Possible next step: add β=2.75 to bracket the paper's curve more tightly.

## 2026-11-06 — Medical (chest X-ray) targets
- **What changed:**
  - With the user's approval, downloaded 20 NLM Montgomery CXRs, their lung masks and the readme to `data/montgomery/` (~90 MB).
  - `gen_synthetic_targets.py` now:
    - crops the lung-mask bounding box, block-mean downsamples to 256 and maps to [-1, 1] in 3 identical channels;
    - adds a zero-padded, overlap-corrected autocorrelation;
    - saves grayscale synthetic twins in `images_gray/`;
    - writes `medical_distance.csv` and `autocorr_gray_padded.csv`;
    - adds the CXR curve to `autocorr_vs_paper.png` and a CXR row to `image_grid.png`.
  - Added a medical section to the README.
- **Why:** to place real medical targets on the β axis.
- **Files touched:** `gen_synthetic_targets.py`, `synthetic_targets/README.md`, `synthetic_targets/{images_gray,images_medical}/` (new), new CSVs, both figures, `data/montgomery/` (new).
- **Verification:** `python3 gen_synthetic_targets.py` prints "self-checks passed".
  - Padded grayscale curves match the circular RGB curves within 0.015.
  - The grid PNG has ~1.18M distinct colours, so the blotchy preview was a viewer artifact.
- **Result:** CXR lag-1 = 0.992, lag-32 = 0.593. Equivalent β ≈ 3.5 (RMSE 0.007), which is smoother than the paper's priors (β ≈ 2.5–3).
- **Risks/TODO:**
  - Dataset terms forbid sharing outside the research group; do not publish `data/` or `images_medical/`.
  - The equivalent β depends on the downsampling factor (~12×).
  - There is no self-check assert for the padded autocorrelation yet.
  - `synthetic_targets/` is now ~340 MB.

## 2026-11-06 — Paper reproduction: environment, data, core code, Table 2
- **Scope:** reproduce arXiv 2601.22443v2 ("Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance", ICML 2026) from the paper and its appendix only. The authors' GitHub repo and project page are **not** consulted. Baseline details come from the cited papers (DPS arXiv 2209.14687, DMPlug arXiv 2405.16749).
- **What changed:**
  - New conda env `weakprior` (Python 3.10, torch 2.12.0+cu130, diffusers 0.41, lpips, scikit-image, lmdb, pyarrow). The user asked to leave existing envs untouched; a temporary `.venv` created earlier this session was deleted. Only torch 2.12/cu130 runs CUDA on this WSL machine; older torch builds and `nvidia-smi` segfault.
  - Data in `data/`:
    - LSUN bedroom/church_outdoor official val LMDBs (300 images each).
    - CelebA-HQ 256 validation split (2000 images) plus train shard 0 (`korexyz/celeba-hq-256x256`).
    - Models `google/ddpm-{celebahq,bedroom,church}-256` in the HF cache.
  - `repro/prepare_data.py` writes `data/test/{celeba,bedroom,church}/000-099.png`: 100 random images (seed 0), with ADM-style crop for LSUN.
  - `repro/wp.py` holds the core: 3-step DDIM generator, operators (Table 18), metrics, INO with AdamSphere / HoldoutTopK / ES-WMV, and DPS.
  - `repro/selftest.py`, `repro/run.py` (resumable experiment cells), `repro/exp_autocorr.py` (Table 2/14).
- **Why:** the hardware is a laptop RTX A1000 with 8 GB. fp16 weights + `torch.compile` make it about 2× faster than fp32. Measured costs: one INO iteration ≈ 0.37 s (1000 iterations ≈ 6 min/image); DPS ≈ 2.5 min/image.
- **Key finding:** the paper does not state the DDIM timestep spacing. Table 2 settles it: the "leading" spacing [666, 333, 0] (diffusers default) matches the paper's 3-step autocorrelations (RMSE 0.035 / 0.015); "trailing" does not (0.099 / 0.084). All INO runs use leading spacing with clip_sample = True.
- **Verification:**
  - `python -s repro/selftest.py` → "selftest passed"; fp16+compile vs fp32 INO gradient cosine = 0.99998.
  - `python -s repro/exp_autocorr.py` → `results/autocorr/`. Pearson(Bedroom-3, CelebA-20) = 0.9946 (paper 0.9987).
- **Risks/TODOs:**
  - CelebA long-lag autocorrelation is higher than the paper's (lag 32: 0.32 vs 0.27).
  - HoldoutTopK K is not given in the paper; K = 5 is fixed a priori and other K values are reported as sensitivity.
  - Nonlinear deblurring (BKSE), latent diffusion, DAPS/DiffPIR/DRP and InverseBench are out of scope for now.

## 2026-11-06 — Reproduction: Figure 2, Table 1, validation, experiment queue started
- **What changed:**
  - `repro/exp_toy.py` covers Figure 2 (2-D GMM posterior concentration) plus a numerical check of Theorem 3.2: a GMM prior over 2000 CelebA-HQ images, random inpainting, two priors whose weights differ by ≤10×.
  - `repro/exp_scoregap.py` covers Table 1 at several pool sizes. The extra pool images come from the HF LSUN test splits (`data/lsun_{bedroom,church}_hf`, ~350 MB).
  - `repro/summarize.py` builds the paper-style tables; `repro/make_figures.py` builds the qualitative grids; `repro/queue.sh` runs the experiments breadth-first (n = 2, then n = 5); `repro/README.md` records all implementation choices.
  - `wp.ino`: added trajectory snapshots (Figure 1) and DMPlug-style termination at the ES-WMV stop.
  - Updated `repro/selftest.py`.
- **Results:**
  - **Fig. 2:** both posteriors concentrate on x* (error ∝ 1/√N).
  - **Thm 3.2 check:** TV between the two priors' posteriors < 1e-12 for m ≥ 32 observed entries.
  - **Table 1:** with the 100 targets as the pool, gaps are CelebA 0.224 (0.062), Church 0.299 (0.064), Bedroom 0.260 (0.074), vs paper 0.22 (0.06), 0.28 (0.06), 0.27 (0.07). Larger pools (6.6k–15k images) shrink the gaps to 0.16–0.22, but the true image is always the arg-min.
  - **Validation (CelebA inpainting, image 0):**
    - INO in-domain: PSNR 32.56; HTK picked iteration 918; ES-WMV stopped at 472 with 31.87 dB.
    - DPS: 31.92 dB (paper mean 31.98).
    - Runtime: INO ≈ 400 s/image, DPS ≈ 155 s/image.
- **Verification:** `python -s repro/selftest.py` passed. Outputs in `results/toy/`, `results/scoregap/table1.csv`, `logs/*.log`.
- **Queue:** started 2026-11-06 ~17:50 as `setsid nohup bash repro/queue.sh > logs/queue.log`. The PID is in `logs/queue.pid`; stop with `touch logs/STOP`. Estimated ≈14 h for the n = 2 round and ≈36 h in total.
- **Risks/TODOs:**
  - Two images per cell is very noisy; draw conclusions only after n = 5.
  - The LPIPS backbone is unknown. VGG matches the paper's level better, AlexNet its DPS-vs-ours ordering; both are stored.

## 2026-11-07 — Batched run.py and cell list, to move the experiments to Arrhenius (naiss-small)
- **What changed:**
  - `repro/run.py` gained `--batch B`, which runs B images per GPU pass. Per-image seeds, losses, Adam moments and selection rules are unchanged, so the batch size only changes speed.
  - DMPlug always runs with batch 1. In a batch, its loop would continue until every image has stopped, which would change its `last`/`oracle` rules.
  - Each JSON now also records `batch` and `device`. With batching, `seconds` is per image and `peak_mib` is for the whole batch.
  - Added `repro/cluster/cells.txt`: the 76 cells of `queue.sh` (49 INO, 18 DPS, 9 DMPlug), one per line.
- **Why:** the user asked to run the experiments on their NAISS Small allocation. That is the Arrhenius project `naiss2026-4-1816`, which has a `-gpu` account. One image peaks at 4.2 GB, so a large GPU can process many images per pass.
- **Files touched:**
  - `repro/run.py` (the old version is saved in the scratchpad as `run.py.before_batch`)
  - `repro/cluster/cells.txt`
- **Verification:**
  - A CPU test with a dummy generator: batch 2 reproduces batch 1 exactly for all 12 INO rules, DMPlug and DPS, over 3 images (including a short last chunk).
  - `py_compile` passes.
  - The local queue continues with the default batch 1.
- **Risks/TODOs:**
  - Arrhenius needs a password + TOTP login. Pending: the GPU type/architecture, the GPU budget and quotas, the env, and a test job.
  - Project group quota incident (see the Ossification WORKLOG, 2026-10-29): keep files out of the 128 GB project group quota unless it is checked first.

## 2026-11-07 — Arrhenius setup (GH200) and test jobs
- **What changed:**
  - Added `repro/cluster/`:
    - `requirements.txt`: the same pinned versions as the local env.
    - `env.sh`, which every job sources.
    - `setup.sbatch`: one-off; builds the venv, caches the models and LPIPS weights, runs the selftest.
    - `cells.sbatch`: one array task = one line of `cells.txt`, with N images and B images per pass.
  - Working directory on the cluster: `~/weakprior_repro` (code, `data/test`, results, logs).
  - The project disk (`/nobackup/proj/disk/naiss2026-4-1816/personal/wenxuanl/weakprior_repro`) holds only `env.tar` (5.2 GB, one file) and the model caches. Each job unpacks the venv to node-local `/scratch/local` and deletes it on exit.
- **Problems hit and fixes:**
  1. Setup job 4951661 failed in 4 s. Jobs inherit the x86 login environment, and the module tree is x86-only. Fix: build the venv from the native aarch64 Python `/software/sse2/el9_gh200/easybuild/prefix/software/Python/3.13.5-GCCcore-14.3.0/bin/python3` (RPATH-linked, no module needed).
  2. Setup job 4951691 failed with `Disk quota exceeded`. The project disk's limit is the **number of files**, not space: soft 200k, hard 300k. It was already at 282,972 files (252,599 in `pose_mixed/`, not this project), with a grace period of about 8 weeks. Space is fine (142 GB of 1.95 TB).
     - Removed the half-built env (4 GB, mine) at once. The file count is back to its earlier value.
     - Moved code and data to home (30 GB, 500k-file quota, 22.5k files used).
- **Verification:**
  - Setup job 4951814 completed in 2:41 on a GH200 (driver 580.196.07, 97.9 GB).
  - The selftest passed: fp16+compile vs fp32 INO gradient cosine 0.99998.
  - Model snapshots are the same commits as local (cd5c944 / d6ccd88 / 5a1118b). All 12 pinned packages are identical (`logs/pip_freeze_arrhenius.txt` on the cluster).
- **Test jobs:** 4951914 (cells 1 and 2: DPS and INO, CelebA inpainting, 16 images, batch 16) and 4951915 (cell 68: DMPlug, 2 images).
- **Risks/TODOs:**
  - The cluster uses Python 3.13; local uses 3.10.
  - Batched GH200 numerics differ slightly from the A1000 at batch 1, so compare the overlapping images.
  - Tell the user that the project disk is over its soft file quota.

## 2026-11-07 — Full run submitted on Arrhenius (100 images × 76 cells); local queue stopped
- **Test jobs (all COMPLETED):**
  - CelebA inpainting, in-domain, 16 images: ours 29.20 ± 0.72 dB (LPIPS-VGG 0.241), DPS 29.90 ± 0.65 dB (0.188). The paper reports 33.78 vs 31.98, so this is preliminary.
  - Same seeds, A1000 batch 1 vs GH200 batch 16: ours 32.60 / 28.30 vs 32.56 / 28.29 dB; DPS 31.92 / 29.39 vs 31.93 / 29.24 dB. Every gap is ≤ 0.15 dB.
  - Billing is 1 per GPU-hour.
  - Peak memory at batch 16: INO 34.8 GB, DPS 12.6 GB, so batch 25 fits.
- **User decision:** 100 images per cell (paper scale), about 50 GPU-h.
- **Submitted:**
  - Array 4952229: cells 1–67, `cells.sbatch 100 25`, 3 h.
  - Array 4952230: cells 68–76 (DMPlug), `100 1`, 5 h.
  - All 76 tasks started at once.
- **Local queue:**
  - It had been hung since 2026-11-06 19:17 in `ino inpaint church --prior celeba`. The main process spun at 100% CPU for 15.6 h with no image finished. The cause was not determined: likely a torch.compile hang or a WSL GPU wait.
  - I killed it (PID 7568 and its compile worker 7722); the queue exited via `logs/STOP`.
  - The laptop GPU is healthy afterwards (7.0 / 8 GB free).
  - The 18 finished laptop images were moved to `results/runs_a1000` as the hardware cross-check. `results/runs` will hold the cluster results.
- **Files touched:**
  - `repro/README.md` (Arrhenius section; scale paragraph)
  - `repro/cluster/*`
  - `results/runs_a1000/` (moved)
- **Risks/TODOs:**
  - Copy the results back, then run `summarize.py` and `make_figures.py`.
  - DMPlug runs one image at a time (~15 of the 50 GPU-h). Batching it with frozen stopped images would save ~12 GPU-h if this is re-run.
  - The previous status estimate ("round 1 done Saturday morning") was wrong, because the queue was hung.

## 2026-11-07 — Full run finished (76 cells × 100 images); results copied back and summarised
- **What:**
  - Arrays 4952229 and 4952230: all 76 tasks COMPLETED, 100 images in every cell, 46.4 GPU-h (sum of task wall-clock). The longest task was DMPlug at 2:28.
  - Copied `results/runs/` (923 MB, 76 cells) and the cell logs (`logs/arrhenius/`) back to the laptop.
  - Ran `summarize.py` → `results/summary.md` and `make_figures.py` → `results/figures/`.
- **Changes:**
  - `summarize.py`: the compute table now reads the device and batch from the per-image JSON. It used to be hard-coded "RTX A1000".
  - `make_figures.py`: the row-label margin was widened from 110 to 180 px; labels were cut off.
- **Hardware check:** the 18 laptop images (`results/runs_a1000`, A1000, B = 1) vs the same images on the GH200 (B = 16 or 25) differ in PSNR by:
  - INO: at most 0.04 dB.
  - DPS: at most 0.33 dB (a 1000-step sampler).
  - Mean |Δ| is 0.065 dB, well below the per-cell standard error (0.2–0.4 dB).
- **Results vs the paper (PSNR, in-domain):**
  - **LSUN:** close to the paper.
    - Bedroom / church inpainting: 28.65 / 23.90 vs 28.88 / 24.93.
    - Bedroom / church Gaussian deblurring: 25.16 / 21.48 vs 25.28 / 22.43.
  - **CelebA:** 2.7–4.6 dB below the paper, for every prior, and DMPlug shows the same gap.
    - Inpainting: 29.18 vs 33.78.
    - 4× SR: 28.37 vs 31.27.
    - Gaussian deblurring: 27.30 vs 29.98.
  - The oracle (best iterate on the whole trajectory) for CelebA inpainting is 29.31. So the gap is not caused by the stopping rule.
- **Diagnosis so far:**
  - Our CelebA-HQ images are lossless PNGs, so JPEG artifacts are ruled out.
  - The paper writes "CelebA-HQ (Liu et al., 2015)" and its tables say "CelebA". DMPlug used "100 images from CelebA [Liu et al.] resized to 256 × 256".
  - The paper's CelebA numbers match DMPlug's own CelebA numbers: DMPlug inpainting 34.03 / SR 31.25; DPS inpainting 32.24.
  - Hypothesis: the paper's faces are original CelebA (178×218, upsampled), not CelebA-HQ. Those images have 4.7× less high-frequency energy (mean squared Laplacian 0.0044 vs 0.0204).
- **Check submitted:**
  - New target `celeba178`: 100 random images (seed 0) from the first CelebA test shard of `flwrlabs/celeba`, same ADM-style crop and resize. Built by `prepare_data.py celeba178`.
  - Cells 77–83 in `cluster/cells.txt` (DPS and INO for 3 tasks, plus INO with the church prior on inpainting). Array 4958937, about 3 GPU-h.
- **Files touched:**
  - `repro/summarize.py`, `repro/make_figures.py`, `repro/prepare_data.py`, `repro/run.py` (`--target celeba178`)
  - `repro/cluster/cells.txt`
  - `data/test/celeba178/`, `data/celeba_orig_hf/` (public CelebA)
- **Risks/TODOs:**
  - The crop/resize of the paper's faces is unknown. Ours is short side to 256 (bicubic), then a centre crop.
  - DMPlug is not part of the check (2 h per cell at batch 1).

## 2026-11-07 — CelebA test-set check finished (array 4958937, cells 77–83)
- **What:**
  - 7 cells × 100 images on original CelebA test faces (`celeba178`). All COMPLETED; 2.65 GPU-h.
  - Copied back to `results/runs/*celeba178*` and `logs/arrhenius/`. `summary.md` has the new "CelebA test-set check" table.
- **Results** (PSNR, CelebA-HQ → original CelebA [paper]):
  - DPS inpainting 29.46 → 31.95 [31.98]: now matches the paper. SR 27.50 → 27.60 [26.82]. Deblurring 26.91 → 26.92 [27.55].
  - INO, CelebA prior: inpainting 29.18 → 31.23 [33.78]. SR 28.37 → 28.75 [31.27]. Deblurring 27.30 → 27.07 [29.98].
  - INO, church prior, inpainting: 28.31 → 31.47 [32.62].
  - Selected iteration for INO inpainting: 610 → 950 (in-domain) and 937 (church prior) [974 / 973].
- **Interpretation:**
  - The paper's faces are most likely original CelebA (DPS inpainting and the selected iteration now match).
  - A residual INO gap of 2.5–2.9 dB remains on faces with the CelebA prior.
  - On original CelebA the INO PSNR is still rising at iteration 1000: +0.46 dB over iterations 750–1000 for inpainting, vs +0.07 on CelebA-HQ and +0.09 on bedroom. So in 1000 iterations our optimizer gets less far than the paper's on smooth faces. Candidate: an unstated AdamSphere or step-size detail. Not tested.
- **Files touched:** `repro/README.md` (Results section with the claim-by-claim verdict and the CelebA gap), `WORKLOG.md`.
- **Verification:**
  - Every number in the README verdict table was recomputed from `results/runs` with summarize.py's helpers.
  - Out-of-domain loss vs the paper: |Δ| ≤ 0.3 dB in 15 of 18 cells, ≤ 0.39 dB in all 18.
- **Compute total:** about 49.4 GPU-h on `naiss2026-4-1816-gpu`: 46.4 (main) + 2.65 (check) + about 0.3 (setup/tests).
- **Left on the cluster** (not deleted; the user decides):
  - `~/weakprior_repro` (code, data/test, results about 1 GB, logs).
  - `/nobackup/proj/disk/naiss2026-4-1816/personal/wenxuanl/weakprior_repro/` (`env.tar` 5.2 GB + model caches).
- **Risks/TODOs:**
  - The residual CelebA gap is unexplained.
  - DMPlug was not run on `celeba178`.
  - The SR / deblurring check with out-of-domain priors on `celeba178` was not run.

## 2026-11-07 — Nonlinear deblurring added (main-text Table 4); scope set to "main text only"
- **User decision:** "正文做完了就行了": reproduce the main text only. Appendix-only items (Tables 6 nonlinear rows, 8–11, 13, 15, 17) are left out by choice.
- **Main-text gap:** only the nonlinear-deblurring rows of Table 4 (ours vs DMPlug, in-domain, 3 targets).
- **What changed:**
  - `wp.py`: new task `nonlinear`, y = F(x, k) + n.
    - F is BKSE `KernelWizard.adaptKernel` (Tran et al. 2021; GOPRO_wVAE.pth as named in the paper's Table 18), applied to [0, 1] images, with no clamping.
    - One kernel code per image, k ~ N(0, 1.2² I) of shape 512×2×2, drawn from the image's seed. The 1.2 is taken from BKSE's own `generate_blur.py`; DPS Eq. 58 leaves σ_k open.
    - lr 0.01 and holdout 10 % (Table 19).
  - `run.py`: `nonlinear` appended to `TASKS`, so existing seeds are unchanged.
  - `selftest.py`: nonlinear check (same seed → same y within 1e-4; gradient flows).
  - `summarize.py`: paper values for Table 4 nonlinear (DMPlug) and the Table 6 nonlinear rows (ours); the Table 4/7 section now includes `nonlinear`.
  - `cluster/cells.txt`: cells 84–89 (INO in-domain and DMPlug × 3 targets).
  - README: implementation row, setup line, and "Not reproduced" rewritten for the main-text scope.
- **Third-party code:** BKSE cloned to `data/bkse` (commit f2fe73e, BSD-style VinAI license). It was read before use; the needed modules are plain PyTorch layers. Weights: GOPRO_wVAE.pth (461 MB, sha256 f97facc7…), from the Google Drive link in BKSE's README.
- **Verification:**
  - CPU check (`scratchpad/check_nonlinear.py`): deterministic per seed, gradient flows.
  - The blur looks like real motion blur; PSNR(blurred, gt) is 28.4 / 23.1 on two sample images.
  - First GPU selftest (job 4970127) failed: two GPU forward passes were not bit-identical. Relaxed to `allclose(atol=1e-4)`, which is far below the noise σ = 0.01. Resubmitted as selftest 4970246, then arrays 4970247 (INO, B = 25) and 4970248 (DMPlug, B = 1).
- **Risks/TODOs:**
  - σ_k = 1.2 and "no clamping" are our choices.
  - Nonlinear runs use the CelebA-HQ test set, like the other main tables.
- **Update (same day), TF32 fix:**
  - The resubmitted selftest 4970246 failed again: `allclose(atol=1e-4)` was not met.
  - Measured on a GH200 (srun): two evaluations of the BKSE blur differ by up to 1.3e-3 with cuDNN TF32 on, and 1.4e-6 with it off. On the laptop A1000 they are identical.
  - Fix in `wp.blur_model()`: `torch.backends.cudnn.allow_tf32 = False`, so the forward operator is full fp32 and gives the same y on every call. It only affects fp32 convs (BKSE, LPIPS); the UNet runs in fp16.
  - The selftest then passed on a GH200 (INO gradient cosine 0.99998).
  - Submitted arrays 4970649 (cells 84–86, INO, B = 25) and 4970650 (cells 87–89, DMPlug, B = 1, 5 h).

## 2026-11-07 — Nonlinear deblurring finished; main text complete
- **What:** arrays 4970649 (cells 84–86, INO, B = 25) and 4970650 (cells 87–89, DMPlug, B = 1) all COMPLETED, 100/100 images each. Results copied to `results/runs/*nonlinear*`, logs to `logs/arrhenius/cell_4970649_*`, `cell_4970650_*`.
- **Results (PSNR, ours / DMPlug [paper]):** CelebA 26.92 / 26.45 [25.21 / 24.85], Bedroom 26.03 / 24.74 [22.67 / 22.21], Church 22.87 / 21.90 [20.88 / 20.48].
  - Ours ≥ DMPlug 3/3, paired +0.47 ± 0.07 / +1.28 ± 0.12 / +0.97 ± 0.08 dB; ours better on 95 / 93 / 97 % of images. LPIPS and SSIM go the same way.
  - Both methods are 1.4–3.4 dB above the paper: our blur is probably milder than theirs (σ_k and clamping are not stated).
  - HoldoutTopK picks iteration 997–999; ES-WMV stops at a median of 500–750. ES-WMV on the INO trajectories equals DMPlug to ≤ 0.02 dB, so the gain is the stopping rule.
- **Files:** `results/summary.md` (regenerated), `repro/README.md` (Tables 4/7 verdict now 11/12, new nonlinear absolute-PSNR row, stopping-rule bullet, compute ≈ 58 GPU-h, scope line reworded to "every main-text experiment has been run").
- **Verification:** `python -s repro/summarize.py`; the paired differences, %-better and selected iterations were computed from the per-image JSONs with a one-off script (not in `summarize.py`); the paired means equal the differences of the table means since both cells use the same 100 images.
- **Compute:** 1.8 GPU-h (INO) + 6.8 GPU-h (DMPlug) = 8.6 GPU-h; total ≈ 58 GPU-h.
- **Risks/TODOs:** nonlinear runs only on the CelebA-HQ test set. Cluster leftovers: `~/weakprior_repro` (≈ 1 GB), `env.tar` + caches on the project disk, `logs/check_nl_gpu.py`.

## 2026-11-08 — Prepared the project for sharing on GitHub (not committed yet)
- **Why:** the user wants to share code and results with the other group members, via GitHub.
- **What changed:**
  - `git init -b main` in the project root. Nothing is committed: commit and push wait for the user's go.
  - `.gitignore` (new). Excludes `data/`, the per-image reconstruction PNGs in `results/runs/*/` (960 MB), `.idea/`, `__pycache__/`, the PyCharm template `main.py`, and the regenerable synthetic images.
    - Also excludes the chest X-rays: `synthetic_targets/images_medical/`, plus `synthetic_targets/image_grid.png`, whose last row shows 9 of them.
  - `README.md` (new, root): what is where, env setup (`repro/cluster/requirements.txt` plus pyarrow, lmdb, matplotlib for the local-only scripts), and what is not in the repository.
  - `repro/README.md`: the local python path `P=...` replaced by `python -s`.
- **Share set:** 16,037 files, 137 MB, largest file about 1 MB. A grep for data/, CXR and per-image PNG paths finds none of them. No tokens or passwords in the shared text files.
- **Verification:** copied exactly the share set to a scratch directory and ran `python -s repro/summarize.py` there. The resulting `results/summary.md` is byte-identical to ours.
- **Risks/TODOs:**
  - This machine has no GitHub login: no `gh`, no GitHub SSH key. Installing the gh release binary was blocked by the auto-mode permission check. The user can run `sudo apt install gh` (Ubuntu 24.04: gh 2.45.0) and then `gh auth login`.
  - `repro/make_figures.py` needs the excluded PNGs. Plan: a zip as a GitHub Release asset, on request.
  - The repository should be private: CelebA and LSUN images are for non-commercial research use only.

## 2026-11-08 — Root README is now the project report (experiments, results, figures)
- **Why:** the user asked for a clear list of which experiments were run, with results shown, for the group members. On GitHub the root `README.md` is the repository's front page.
- **What changed:**
  - `README.md` (root) rewritten:
    - paper claim and short answer
    - method in 5 bullets
    - table of the 9 experiment groups with paper table, what was run and number of settings (36 + 15 + 4 + 15 + 12 + 7 = 89)
    - verdict table and CelebA gap, moved here from `repro/README.md`
    - 9 embedded figures with one-line captions; "Not reproduced"; repository map; setup.
  - `repro/README.md`: the Results / CelebA gap / Not reproduced sections replaced by a pointer to the root README. The `cells.txt` row now mentions 84–89.
  - `repro/make_figures.py`: new `results/figures/recon_nonlinear.png` (measurement, DMPlug, ours, reference; 3 sets × 2 images).
- **Verification:**
  - `make_figures.py` rerun: the 6 existing figures are byte-identical (md5), and the new one was checked by eye.
  - Every relative link in both READMEs exists and is not git-ignored.
  - Captions were written after viewing each figure; the box-LPIPS numbers were checked against Table 12 in `summary.md`.
  - Share set: 16,038 files.
- **Risks/TODOs:**
  - The README is in English, like the other docs.
  - Not committed or pushed yet; waiting for the user's go.
  - The GitHub CLI is logged in as AnotherFortnight (gh installed by the user).

## 2026-11-08 — Published on GitHub (public)
- **User decision:** push, and make everything public, including the per-image PNGs ("算了直接全部公开吧"). This was decided after the user was told that CelebA's terms forbid redistributing its images and that the Church images carry Shutterstock watermarks. Commit email: the existing global git email.
- **What:**
  - Repository https://github.com/AnotherFortnight/weak-prior-repro (public, branch `main`).
  - `.gitignore` now also excludes `logs/paper_figs/` (page renders of the paper) and the PNGs of `results/runs_a1000/`.
  - Per-image PNGs go to GitHub release `v1.0` as `reconstructions_png.zip` instead of into git: 9,488 files, 959 MiB, stored without compression, made with Python `zipfile` because `zip` is not installed. The README links to the release.
  - Pushed with gh as the git credential helper for this push only; the global git config is unchanged.
- **Not published:** `data/`, the chest X-ray images (`images_medical/`, `image_grid.png`), the regenerable synthetic images.
- **Verification:**
  - `zipfile.testzip` passed with 9,488 entries.
  - `git ls-remote origin main` equals the local HEAD 57a1be8, and the repo reports PUBLIC.
  - Release `v1.0` is published. Its asset is 1,005,090,000 bytes, with the same SHA-256 as the local zip (a18ed269…).
- **Risks:**
  - CelebA / LSUN terms: the user's choice.
  - No LICENSE file, so default copyright applies.
  - The CXR statistics (curves and CSVs, not images) are public.
