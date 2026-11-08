"""Build the 256x256 test sets: 100 random images per domain (seed 0).

CelebA-HQ: korexyz/celeba-hq-256x256 validation split (2000 images, already 256x256).
LSUN Bedroom / Church outdoor: official LSUN val LMDBs (300 images each), ADM-style center crop + resize.
celeba178 (test-set check only): original CelebA aligned test faces (178x218, first of 3 test shards of
  flwrlabs/celeba), same crop + resize, i.e. upsampled x1.44. The paper cites Liu et al. 2015 for its faces; DMPlug did too.
Run: python -s repro/prepare_data.py [domain ...]     (no argument: all domains)
"""
import csv
import io
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
N_TEST, SEED, SIZE = 100, 0, 256


def center_crop_arr(img, size=SIZE):
    """Guided-diffusion style: box-halve while large, bicubic to short side = size, center crop."""
    while min(img.size) >= 2 * size:
        img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
    s = size / min(img.size)
    img = img.resize(tuple(round(x * s) for x in img.size), resample=Image.BICUBIC)
    a = np.asarray(img)
    y0, x0 = (a.shape[0] - size) // 2, (a.shape[1] - size) // 2
    return a[y0:y0 + size, x0:x0 + size]


def celeba_images():
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(DATA, "celebahq_hf/data/validation-00000-of-00001.parquet"))
    for i, rec in enumerate(t.column("image").to_pylist()):
        yield f"val_{i:04d}", np.asarray(Image.open(io.BytesIO(rec["bytes"])).convert("RGB"))


def celeba178_images():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("flwrlabs/celeba", "img_align+identity+attr/test-00000-of-00003.parquet", repo_type="dataset",
                        local_dir=os.path.join(DATA, "celeba_orig_hf"))
    for i, rec in enumerate(pq.read_table(f, columns=["image"]).column("image").to_pylist()):
        yield rec.get("path") or f"test0_{i:04d}", center_crop_arr(Image.open(io.BytesIO(rec["bytes"])).convert("RGB"))


def lsun_images(path):
    import lmdb
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        for k, v in txn.cursor():
            yield k.decode(), center_crop_arr(Image.open(io.BytesIO(v)).convert("RGB"))


def main():
    sources = {
        "celeba": celeba_images,
        "bedroom": lambda: lsun_images(os.path.join(DATA, "lsun_bedroom_val/bedroom_val_lmdb")),
        "church": lambda: lsun_images(os.path.join(DATA, "lsun_church_val/church_outdoor_val_lmdb")),
        "celeba178": celeba178_images,
    }
    for name, gen in sources.items():
        if sys.argv[1:] and name not in sys.argv[1:]:
            continue
        items = list(gen())
        assert all(a.shape == (SIZE, SIZE, 3) for _, a in items), name
        pick = np.random.default_rng(SEED).permutation(len(items))[:N_TEST]
        out = os.path.join(DATA, "test", name)
        os.makedirs(out, exist_ok=True)
        rows = []
        for j, idx in enumerate(pick):
            key, arr = items[idx]
            Image.fromarray(arr).save(os.path.join(out, f"{j:03d}.png"))
            rows.append([j, key])
        with open(os.path.join(out, "sources.csv"), "w", newline="") as f:
            csv.writer(f).writerows([["index", "source_key"]] + rows)
        print(f"{name}: {len(items)} available -> {N_TEST} saved to {out}")


if __name__ == "__main__":
    main()
