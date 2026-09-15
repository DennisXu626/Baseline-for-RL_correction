#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?usage: run_stage1.sh gate|smoke|5m OUTPUT [GPU]}"
OUT="${2:?missing output}"
GPU="${3:-1}"
RUNTIME="${RUNTIME_ROOT:-/ssd/sy/kailang/clean3/curriculum_20260915}"
V12="${V12_ROOT:-/ssd/sy/kailang/pour17/direct58d}"
PYTHON="${PYTHON_BIN:-/ssd/sy/kailang/env/rl-correction-pour/bin/python}"
case "$OUT" in /ssd/sy/kailang/*) ;; *) echo "output outside kailang" >&2; exit 2;; esac
test ! -e "$OUT" && test ! -e "$OUT.log"
cd "$RUNTIME" && source /ssd/sy/kailang/tmp/cuda_env.sh
KIT="--/renderer/multiGpu/enabled=false --/renderer/activeGpu=$GPU"
common=(--runtime_root "$RUNTIME" --v12_root "$V12" --output_root "$OUT" --device cuda:0 --headless "--kit_args=$KIT")
case "$MODE" in
 gate) command=(tasks/h2s2r_clean3/check_ours_stage1_contract.py "${common[@]}");;
 smoke) command=(tasks/h2s2r_clean3/train_ours_stage1.py "${common[@]}" --num_envs 512 --seed 42 --smoke_epochs 3 --anneal_ema 0.7 --anneal_sustain 10);;
 5m) command=(tasks/h2s2r_clean3/train_ours_stage1.py "${common[@]}" --num_envs 512 --seed 42 --max_agent_steps 5000000 --anneal_ema 0.7 --anneal_sustain 10);;
 *) echo "invalid mode" >&2; exit 2;; esac
nohup env CUDA_VISIBLE_DEVICES="$GPU" VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json RL_ISAAC_NO_GUARD=1 SHARPA_WANDB=0 "$PYTHON" -B -u "${command[@]}" >"$OUT.log" 2>&1 &
echo "$!"
