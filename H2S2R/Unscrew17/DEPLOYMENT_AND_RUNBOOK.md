# Deployment and runbook

All paths below assume the A800 project allocation under `/ssd/sy/kailang`.
Do not write project files elsewhere on that host. Commands are examples for an
authorized teammate; choose a new run name and never overwrite an existing run.

## 1. Required inputs

Verified runtime archive:

```text
/ssd/sy/kailang/unscrew17_clip17_runtime_inputs_20260916.tar.gz
sha256 abf2cd67434491c10c49ef868e58ddd63d3e0d9e0b4b6c00c1a67f4429f8b338
size   50806059 bytes
```

Original A6000 handoff source:

```text
/home/msc-auto/clip17_latest_handoff_20260914.zip
sha256 4ab65d3683eb352aeeb2c7c17ddbf6854ff8471ddc8436f9d05545756b964aa0
```

Local ground USD required to avoid network/S3 loading:

```text
/ssd/sy/kailang/pour17/bundle/pour17/world/default_environment.usd
sha256 8a21c317d638d33a4e6c20c958a7ed8f4d8c5195efd7e20f1b85178b4a15638a
```

Runtime:

```text
Python:   /ssd/sy/kailang/env/rl-correction-pour/bin/python
IsaacLab: /ssd/sy/kailang/env/IsaacLab-5.1
GPU:      select after a fresh measured resource check
```

## 2. Build an isolated checkout

Set concrete paths in the shell without reusing `HOME` or `CODEX_HOME`:

```bash
TASK_ROOT=/ssd/sy/kailang/unscrew17
INPUT_ROOT=$TASK_ROOT/runtime_inputs_20260916
CODE_ROOT=$TASK_ROOT/direct58d_asymcritic_NEXT
PACKAGE_ROOT=/path/to/Baseline-for-RL_correction/H2S2R/Unscrew17

mkdir -p "$INPUT_ROOT" "$CODE_ROOT" "$CODE_ROOT/assets"
tar -xzf /ssd/sy/kailang/unscrew17_clip17_runtime_inputs_20260916.tar.gz -C "$INPUT_ROOT"
cp -a "$INPUT_ROOT/repo/." "$CODE_ROOT/"
cp -a "$PACKAGE_ROOT/overlay/." "$CODE_ROOT/"
cp /ssd/sy/kailang/pour17/bundle/pour17/world/default_environment.usd "$CODE_ROOT/assets/default_environment.usd"
```

Verify the three external inputs before continuing:

```bash
sha256sum /ssd/sy/kailang/unscrew17_clip17_runtime_inputs_20260916.tar.gz
sha256sum "$CODE_ROOT/assets/default_environment.usd"
sha256sum -c "$PACKAGE_ROOT/FILES.sha256"
```

The last command must be run from `H2S2R/Unscrew17`; it verifies the GitHub
package, not the extracted private assets.

## 3. Hard preflight gates

### Filesystem

```bash
df -h / /tmp /ssd
df -i / /tmp /ssd
```

Do not launch while `/` or `/tmp` has zero free space. The previous formal run
failed during scene construction for exactly this reason. The current frozen
Clip17 base explicitly places an IsaacLab log under `/tmp`; relocating it is a
separate operational patch and was not silently included in this method
snapshot. If project policy requires every generated file under
`/ssd/sy/kailang`, obtain approval for that one-line log-root relocation before
launch.

### GPU and RAM

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
free -h
```

Existing PIDs alone are not a rejection. Select a card only after measured
VRAM/RAM headroom is sufficient, record the physical mapping, and monitor this
task. Do not stop other users' jobs or remove their locks.

### Static source

```bash
cd "$CODE_ROOT"
/ssd/sy/kailang/env/rl-correction-pour/bin/python -m py_compile \
  tasks/h2s2r_unscrew17/cfg.py \
  tasks/h2s2r_unscrew17/env.py \
  tasks/h2s2r_unscrew17/train_direct58d_lstm.py \
  rl_rebuild/baselines/h2s2r/official_ppo_adapter.py
```

## 4. Runtime environment

For each launch, set these only in the new child process:

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
RL_ISAAC_NO_GUARD=1
PYTHONPATH=$CODE_ROOT
```

`RL_ISAAC_NO_GUARD=1` is not a global setting and does not authorize unsafe GPU
sharing. It only bypasses the legacy exclusive-GPU guard after resource
admission.

## 5. Validation sequence

Run the 1-env preflight first:

```bash
cd "$CODE_ROOT"
env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  RL_ISAAC_NO_GUARD=1 PYTHONPATH="$CODE_ROOT" \
  /ssd/sy/kailang/env/rl-correction-pour/bin/python -B -u \
  tasks/h2s2r_unscrew17/train_direct58d_lstm.py \
  --output_root "$TASK_ROOT/gates/preflight_NEXT/run" \
  --name UNSCREW17_PREFLIGHT_NEXT --num_envs 1 --seed 42 \
  --preflight --max_epochs 1 --device cuda:GPU_INDEX --headless
```

Then run the 1024-env, 3-epoch smoke:

```bash
cd "$CODE_ROOT"
env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  RL_ISAAC_NO_GUARD=1 PYTHONPATH="$CODE_ROOT" \
  /ssd/sy/kailang/env/rl-correction-pour/bin/python -B -u \
  tasks/h2s2r_unscrew17/train_direct58d_lstm.py \
  --output_root "$TASK_ROOT/gates/ppo_smoke_1024_NEXT/run" \
  --name UNSCREW17_PPO_SMOKE_1024_NEXT --num_envs 1024 --seed 42 \
  --smoke_test --device cuda:GPU_INDEX --headless
```

The 1024 setting is smoke-only. It is not the formal training setup.

Required pass evidence:

- log reaches `MAX EPOCHS NUM!` after 3/3 epochs;
- actor normalizer is `(342,)` and critic normalizer `(513,)`;
- epoch checkpoints exist;
- no fatal traceback or model/gradient non-finite error;
- reward and FPS are finite.

## 6. Formal 5M gate

Only after both validation stages pass and resources are rechecked:

```bash
cd "$CODE_ROOT"
env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  RL_ISAAC_NO_GUARD=1 PYTHONPATH="$CODE_ROOT" \
  /ssd/sy/kailang/env/rl-correction-pour/bin/python -B -u \
  tasks/h2s2r_unscrew17/train_direct58d_lstm.py \
  --output_root "$TASK_ROOT/gates/formal_5m_4096env_NEXT/run" \
  --name UNSCREW17_DIRECT58D_FORMAL_5M_4096ENV_NEXT \
  --num_envs 4096 --seed 42 --max_agent_steps 5000000 \
  --milestone_steps 5000000 --device cuda:GPU_INDEX --headless
```

The entry defaults to 4096 environments, but formal commands still pass it
explicitly so the run manifest and shell history are unambiguous. With horizon
16, each rollout contains 65,536 agent steps. The first rollout crossing 5M is
5,046,272 steps.

Do not resume the stopped 2026-09-16 run: it never reached PPO and has no valid
formal checkpoint.

## 7. Monitoring without busy polling

Check at launch, then at intervals appropriate to scene construction. Once the
process is healthy but clearly long-running, hand monitoring back to the user
instead of repeatedly polling.

```bash
pgrep -af '[t]rain_direct58d_lstm.py.*UNSCREW17_DIRECT58D_FORMAL'
grep -E 'stage=|fps step|milestone|saving checkpoint|MAX EPOCHS|fatal:|Traceback|NaN|Inf' \
  "$TASK_ROOT/gates/formal_5m_4096env_NEXT/train.log" | tail -40
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
```
