# Sourced by every Arrhenius job, run from ~/weakprior_repro (code, data, results, logs: $HOME allows 500k files).
# The project disk has space but its 300k-file quota is nearly full, so it only holds the venv as ONE tar file and
# the model caches; each job unpacks the ~20k-file venv to node-local scratch and deletes it on exit.
# The x86 login environment leaks into jobs and its modules are x86-only, hence the native aarch64 (GH200)
# EasyBuild Python, which is RPATH-linked and needs no module.
: "${SLURM_JOB_ID:?source this inside a Slurm job}"
PYBASE=/software/sse2/el9_gh200/easybuild/prefix/software/Python/3.13.5-GCCcore-14.3.0/bin/python3
BIG=/nobackup/proj/disk/naiss2026-4-1816/personal/wenxuanl/weakprior_repro
LOCAL=/scratch/local/wp_$SLURM_JOB_ID
unset LD_LIBRARY_PATH
export HF_HOME="$BIG/cache/hf" TORCH_HOME="$BIG/cache/torch" PIP_NO_CACHE_DIR=1 PYTHONNOUSERSITE=1
mkdir -p "$LOCAL" && trap 'rm -rf "$LOCAL"' EXIT
[ -f "$BIG/env.tar" ] && tar -C "$LOCAL" -xf "$BIG/env.tar"
PY="$LOCAL/env/bin/python -s"
