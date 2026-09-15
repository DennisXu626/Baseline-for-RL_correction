#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:?usage: record_stage1.sh CKPT OUT ROW [GPU]}"; OUT="${2:?}"; ROW="${3:?}"; GPU="${4:-1}"
RUNTIME="${RUNTIME_ROOT:-/ssd/sy/kailang/clean3/curriculum_20260915}"
V12="${V12_ROOT:-/ssd/sy/kailang/pour17/direct58d}"
PYTHON="${PYTHON_BIN:-/ssd/sy/kailang/env/rl-correction-pour/bin/python}"
case "$OUT" in /ssd/sy/kailang/*) ;; *) exit 2;; esac
test -f "$CKPT" && test ! -e "$OUT" && test ! -e "$OUT.log"
cd "$RUNTIME" && source /ssd/sy/kailang/tmp/cuda_env.sh
KIT="--/renderer/multiGpu/enabled=false --/renderer/activeGpu=$GPU"
nohup env CUDA_VISIBLE_DEVICES="$GPU" VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json RL_ISAAC_NO_GUARD=1 "$PYTHON" -B -u tasks/h2s2r_clean3/record_ours_stage1.py --runtime_root "$RUNTIME" --v12_root "$V12" --checkpoint "$CKPT" --output_root "$OUT" --release_row "$ROW" --controls 300 --device cuda:0 --headless "--kit_args=$KIT" >"$OUT.log" 2>&1 &
echo "$!"
