#!/bin/bash
#SBATCH -A naiss2026-4-1816-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J wp-latent-setup
#SBATCH -o logs/latent_setup_%j.log
# One-off (Table 13): cache SD 2.1 base and DiT-XL-2-512, save the SD null-prompt embedding to data/latent/, then a
# short timing test of every latent method on 2 example ImageNet images.   Run from ~/weakprior_repro.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source repro/cluster/env.sh
"$LOCAL/env/bin/python" -m pip install -q transformers   # this job only (text encoder for the null prompt); env.tar is unchanged
$PY -c "import transformers; print('transformers', transformers.__version__)"
$PY repro/latent.py setup
for c in "ino --model sd21 --task inpaint" "ino --model dit --task gauss" "dps --task inpaint" "dsg --task gauss"; do
  HF_HUB_OFFLINE=1 $PY repro/latent.py $c --target imagenet_ex --n 2 --batch 2 --iters 20 2>&1 | grep -v -i warn
done
nvidia-smi --query-gpu=name,memory.used --format=csv
