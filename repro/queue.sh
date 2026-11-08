#!/bin/bash
# Sequential experiment queue, breadth-first: every experiment at n=2 images, then all extended to n=5.
# Every run.py call is resumable (finished images are skipped), so the queue can be stopped and restarted.
#   start:  nohup bash repro/queue.sh > logs/queue.log 2>&1 &
#   stop:   touch logs/STOP   (finishes the current image cell first; or kill the PID in logs/queue.pid)
cd "$(dirname "$0")/.." || exit 1
echo $$ > logs/queue.pid
rm -f logs/STOP
PY="/home/wenxuanli/miniconda3/envs/weakprior/bin/python -s"
FILTER='warn|deprecat|accelerate|low_cpu|pip install|^```|^\.$|safetensors|unsafe|max_autotune'

run() {
  [ -e logs/STOP ] && { echo "STOP file found, exiting"; exit 0; }
  $PY repro/run.py "$@" 2>&1 | grep -v -iE "$FILTER"
}

main_table() {  # Table 3 / 6: DPS (in-domain) + ours with each of the three priors
  local task=$1 n=$2
  for tgt in celeba bedroom church; do
    run dps --task "$task" --target "$tgt" --n "$n"
    for prior in celeba bedroom church; do run ino --task "$task" --target "$tgt" --prior "$prior" --n "$n"; done
  done
}

failure() {  # Table 12 + 16x SR on Church: DPS, ours in-domain (Church), ours out-of-domain (CelebA)
  local n=$1
  for task in box0.3 box0.4 box0.5 box0.6 sr16; do
    run dps --task $task --target church --n "$n"
    run ino --task $task --target church --n "$n"
    run ino --task $task --target church --prior celeba --n "$n"
  done
}

noise() {  # Table 16: CelebA inpainting, out-of-domain prior = Church
  local n=$1
  for s in 0.05 0.2 0.5 1.0; do
    run dps --task inpaint --target celeba --sigma $s --n "$n"
    run ino --task inpaint --target celeba --sigma $s --n "$n"
    run ino --task inpaint --target celeba --prior church --sigma $s --n "$n"
  done
}

ablation() {  # Table 5 (plain Adam trajectories) and Tables 4/7 (DMPlug)
  local n=$1
  for task in inpaint gauss; do for tgt in church bedroom; do
    run ino --task $task --target $tgt --opt adam --n "$n"
  done; done
  for task in inpaint sr4 gauss; do for tgt in celeba bedroom church; do
    run dmplug --task $task --target $tgt --n "$n"
  done; done
}

for n in 2 5; do
  main_table inpaint $n
  main_table sr4 $n
  main_table gauss $n
  failure $n
  noise $n
  ablation $n
  $PY repro/summarize.py > /dev/null 2>&1
done
echo "queue finished"
