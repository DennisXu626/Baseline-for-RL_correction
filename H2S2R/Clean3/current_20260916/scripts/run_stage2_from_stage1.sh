#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?usage: run_stage2_from_stage1.sh smoke|formal CHECKPOINT OUTPUT [GPU] [SHA256] [RELEASE_ROW]}"
CHECKPOINT="${2:?missing checkpoint}"
OUT="${3:?missing output}"
GPU="${4:-0}"
RUNTIME="${RUNTIME_ROOT:-/ssd/sy/kailang/clean3/curriculum_20260915}"
V12="${V12_ROOT:-/ssd/sy/kailang/pour17/direct58d}"
BASE_RL_REBUILD="${BASE_RL_REBUILD:-$V12/rl_rebuild}"
PYTHON="${PYTHON_BIN:-/ssd/sy/kailang/env/rl-correction-pour/bin/python}"
ALLOWED_ROOT="${ALLOWED_ROOT:-/ssd/sy/kailang}"
ROBOT_USD="${ROBOT_USD:-}"
CUDA_ENV="${CUDA_ENV-/ssd/sy/kailang/tmp/cuda_env.sh}"
# Defaults select the recommended final row-35 checkpoint.  The early row-50
# best remains selectable by passing its hash and row explicitly.
EXPECTED_SHA="${5:-34fc3873612f17d9041fe75986b87ce1ddac5d0fd6968d95fcd82476c3a5f9e5}"
RELEASE_ROW="${6:-35}"
case "$CHECKPOINT" in "$ALLOWED_ROOT"/*) ;; *) echo "checkpoint outside allowed root" >&2; exit 2;; esac
case "$OUT" in "$ALLOWED_ROOT"/*) ;; *) echo "output outside allowed root" >&2; exit 2;; esac
test -f "$CHECKPOINT" && test ! -e "$OUT" && test ! -e "$OUT.log"
test -f "$BASE_RL_REBUILD/baselines/h2s2r/contract.py"
test -f "$BASE_RL_REBUILD/baselines/h2s2r/pour17/observation.py"
test -f "$BASE_RL_REBUILD/baselines/h2s2r/pour17/bimanual.py"
test -f "$BASE_RL_REBUILD/correction/env/dexmate_env_cfg.py"
printf '%s  %s\n' "$EXPECTED_SHA" "$CHECKPOINT" | sha256sum -c -
cd "$RUNTIME"
if test -n "$CUDA_ENV" && test -f "$CUDA_ENV"; then source "$CUDA_ENV"; fi
KIT="--/renderer/multiGpu/enabled=false --/renderer/activeGpu=$GPU"
common=(tasks/h2s2r_clean3/train_ours_stage2_from_stage1.py --runtime_root "$RUNTIME" --v12_root "$V12" --output_root "$OUT" --checkpoint "$CHECKPOINT" --checkpoint_sha256 "$EXPECTED_SHA" --allowed_root "$ALLOWED_ROOT" --release_row "$RELEASE_ROW" --seed 42 --device cuda:0 --headless "--kit_args=$KIT")
if test -n "$ROBOT_USD"; then common+=(--robot_usd "$ROBOT_USD"); fi
case "$MODE" in
 smoke) command=("${common[@]}" --num_envs 512 --smoke_epochs 3);;
 formal) command=("${common[@]}" --num_envs 4096 --max_agent_steps 150000000);;
 *) echo "invalid mode" >&2; exit 2;;
esac
nohup env CUDA_VISIBLE_DEVICES="$GPU" VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json RL_ISAAC_NO_GUARD=1 SHARPA_WANDB=0 H2S2R_BASE_RL_REBUILD="$BASE_RL_REBUILD" "$PYTHON" -B -u "${command[@]}" >"$OUT.log" 2>&1 &
echo "$!"
