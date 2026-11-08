#!/bin/bash
set -e
cd "/home/wenxuanli/projects/Deep Learning Course"
P=/home/wenxuanli/miniconda3/envs/weakprior/bin/python
curl -sSL -o data/lsun_bedroom_val/bedroom_val_lmdb.zip http://dl.yf.io/lsun/scenes/bedroom_val_lmdb.zip
curl -sSL -o data/lsun_church_val/church_outdoor_val_lmdb.zip http://dl.yf.io/lsun/scenes/church_outdoor_val_lmdb.zip
echo "lsun done"
cd /tmp && PYTHONNOUSERSITE=1 $P -I -c "
from huggingface_hub import snapshot_download, hf_hub_download
for r in ['google/ddpm-celebahq-256','google/ddpm-bedroom-256','google/ddpm-church-256']:
    print(snapshot_download(r, allow_patterns=['*.json','*.bin']))
print(hf_hub_download('korexyz/celeba-hq-256x256', 'data/train-00000-of-00006.parquet', repo_type='dataset', local_dir='/home/wenxuanli/projects/Deep Learning Course/data/celebahq_hf'))
print(hf_hub_download('korexyz/celeba-hq-256x256', 'README.md', repo_type='dataset', local_dir='/home/wenxuanli/projects/Deep Learning Course/data/celebahq_hf'))
"
echo "ALL DONE"
