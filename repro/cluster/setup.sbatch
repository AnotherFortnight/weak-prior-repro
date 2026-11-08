#!/bin/bash
#SBATCH -A naiss2026-4-1816-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 00:45:00
#SBATCH -J wp-setup
#SBATCH -o logs/setup_%j.log
# One-off, on a GH200 node (aarch64): build the venv on local scratch, cache the 3 DDPMs and the LPIPS backbones,
# run the selftest, then store the venv as $BIG/env.tar.   Run from ~/weakprior_repro:  sbatch repro/cluster/setup.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source repro/cluster/env.sh
"$PYBASE" -m venv "$LOCAL/env"
"$LOCAL/env/bin/pip" install -q -r repro/cluster/requirements.txt
$PY -c "
import lpips
from huggingface_hub import snapshot_download
for m in ['google/ddpm-celebahq-256', 'google/ddpm-bedroom-256', 'google/ddpm-church-256']:
    print(m, snapshot_download(m))
lpips.LPIPS(net='vgg', verbose=False); lpips.LPIPS(net='alex', verbose=False)
"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
which gcc && gcc --version | head -1
"$LOCAL/env/bin/pip" freeze > logs/pip_freeze_arrhenius.txt
HF_HUB_OFFLINE=1 $PY repro/selftest.py
tar -C "$LOCAL" -cf "$BIG/env.tar.tmp" env && mv "$BIG/env.tar.tmp" "$BIG/env.tar" && ls -la "$BIG/env.tar"
