# Agent Environment Record

Last updated: 2026-05-30 00:00 UTC

This file records the current progress and machine-specific environment for the
Codex session running on the `jzy` server. `AGENTS.md` remains the project-level
instruction file. This file is a local progress and environment note requested
by the user.

## Repository Sync

- Repository: `git@github.com:zackaryjing/car_pole.git`
- Local path: `/root/projects/reinforce_learning/car_pole`
- Branch: `main`
- Latest synced commit: `b168015 Update handoff context for DQN server validation`
- Pull result on 2026-05-29: fast-forwarded from `117e223` to `b168015`
- Working tree after pull and test: clean before creating this file

Recent remote progress now present locally:

- Added RL evaluation interfaces and replay support.
- Added episode trajectory save/load and best-record utilities.
- Added random, replay, and centerline heuristic policies.
- Added sensor observation as default training observation.
- Added sensor DQN v1 pipeline:
  - `rl_racing.rl.replay_buffer.ReplayBuffer`
  - `rl_racing.rl.networks.MLPQNetwork`
  - `rl_racing.rl.dqn.train_dqn`
  - `rl_racing.rl.dqn.load_policy`
  - CLI entrypoint `rl-racing-train-dqn`
- Expanded tests for DQN, replay buffer, observations, renderer, and episode utilities.

## Current Machine

- Hostname: `jzy`
- OS context: headless Linux server accessed through Codex
- Project directory: `/root/projects/reinforce_learning/car_pole`
- Training conda environment for this machine: `dmcad`
- Python executable: `/root/miniconda3/envs/dmcad/bin/python`
- Python version: `Python 3.10.20`

Available conda environments:

```text
base       /root/miniconda3
deepcad    /root/miniconda3/envs/deepcad
deepcad38  /root/miniconda3/envs/deepcad38
dmcad      /root/miniconda3/envs/dmcad
maniskill  /root/miniconda3/envs/maniskill
openpi     /root/miniconda3/envs/openpi
text2cad   /root/miniconda3/envs/text2cad
```

## Installed Packages In Current dmcad Env

Verified packages:

```text
numpy       1.26.4
pygame      2.6.1
pytest      9.0.3
torch       2.1.2
torchvision 0.16.2
torchaudio  2.1.2
```

PyTorch CUDA verification in a GPU-accessible context:

```text
torch 2.1.2
torch CUDA 11.8
cuda True 8
NVIDIA GeForce RTX 4090
```

On 2026-05-29, `pygame` and `pytest` were installed into `dmcad`, and this
project was installed editable with:

```bash
/root/miniconda3/envs/dmcad/bin/python -m pip install pygame pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
/root/miniconda3/envs/dmcad/bin/python -m pip install -e .
```

GPU inventory:

```text
8 x NVIDIA GeForce RTX 4090
Driver Version: 525.116.04
Memory per GPU: 24564 MiB
```

Note: normal sandboxed commands may not always access NVML/GPU devices. Use a
GPU-accessible or escalated context when validating CUDA.

## Current Verification

Command:

```bash
/root/miniconda3/envs/dmcad/bin/python -m pytest
```

Result on this server before DQNv2:

```text
32 passed in 19.78s
```

Result after adding DQNv2 vectorized training:

```text
35 passed in 13.29s
```

Additional non-training smoke checks completed in `dmcad`:

- `RacingEnv` sensor reset/step: observation shape `(66,)`, dtype `float32`.
- DQN CLI help: imports and argument parser work.
- Very short CUDA DQN smoke with `total_steps=64`, `learning_starts=16`,
  `batch_size=8`, output under `/tmp/rl_racing_smoke/dmcad_cuda_smoke`.
- Smoke generated `metrics.csv`, `checkpoints/final.pt`, step checkpoints, and
  trajectory `.npz/.json` files.
- Loaded the smoke checkpoint with `load_policy(..., device="cuda")` and ran one
  short episode successfully through the policy path.

This was only a runtime smoke check, not formal training.

## DQNv2 Vectorized Training Status

DQNv2 adds a synchronous subprocess vector environment for sensor-observation
DQN training:

- `rl_racing.rl.vector_env.SubprocVectorEnv`
- batched transition insertion with `ReplayBuffer.add_batch`
- CLI arguments `--num-envs` and `--gradient-steps`
- training fallback to the original DQN path when `--num-envs 1`

CUDA smoke check on physical GPU 7 completed successfully:

```bash
CUDA_VISIBLE_DEVICES=7 /root/miniconda3/envs/dmcad/bin/python -m rl_racing.rl.train_dqn \
  --device cuda \
  --num-envs 8 \
  --total-steps 512 \
  --learning-starts 64 \
  --batch-size 64 \
  --replay-size 4096 \
  --train-frequency 8 \
  --gradient-steps 2 \
  --target-update-interval 128 \
  --eval-interval 256 \
  --eval-episodes 2 \
  --checkpoint-interval 256 \
  --output-root /tmp/rl_racing_smoke \
  --run-name dqn_v2_cuda7_smoke
```

Result:

```text
run_dir=/tmp/rl_racing_smoke/dqn_v2_cuda7_smoke
final_step=512
best_successes=0
```

Suggested one-to-two-hour GPU 7 run using CPU parallel rollout workers:

```bash
CUDA_VISIBLE_DEVICES=7 /root/miniconda3/envs/dmcad/bin/python -m rl_racing.rl.train_dqn \
  --device cuda \
  --num-envs 64 \
  --total-steps 1000000 \
  --learning-starts 20000 \
  --batch-size 2048 \
  --replay-size 1000000 \
  --train-frequency 64 \
  --gradient-steps 4 \
  --target-update-interval 10000 \
  --eval-interval 50000 \
  --eval-episodes 5 \
  --checkpoint-interval 50000 \
  --hidden-dim 512 \
  --epsilon-decay-steps 700000 \
  --run-name dqn_v2_sensor_gpu7_1m_env64_seed0
```

If GPU utilization remains low and CPU workers are keeping up, increase
`--gradient-steps` to `8` or `--batch-size` to `4096`. If environment workers
become the bottleneck, increase `--num-envs` toward `96` or `128`.

## Useful Commands On This Server

Run full tests:

```bash
/root/miniconda3/envs/dmcad/bin/python -m pytest
```

Run manual play if a display/X forwarding is available, though local display is
preferred for real input latency:

```bash
/root/miniconda3/envs/dmcad/bin/python -m rl_racing.play --view follow --seed 0 --render-fps 60 --sim-speed 1.5
```

Short DQN smoke on GPU:

```bash
/root/miniconda3/envs/dmcad/bin/python -m rl_racing.rl.train_dqn \
  --device cuda \
  --total-steps 2000 \
  --learning-starts 100 \
  --batch-size 64 \
  --eval-interval 1000 \
  --checkpoint-interval 1000 \
  --run-name smoke_cuda
```

Suggested longer single-GPU DQN run:

```bash
/root/miniconda3/envs/dmcad/bin/python -m rl_racing.rl.train_dqn \
  --device cuda \
  --total-steps 200000 \
  --batch-size 256 \
  --run-name dqn_sensor_seed0
```

## Development Notes

- `AGENTS.md` currently describes another Linux workstation using
  `/home/jing/miniconda3/envs/system_dev/bin/python`. That is project-level
  guidance for that machine. For this `jzy` server session, use `dmcad` as
  recorded in this file unless the user asks to switch conda environments.
- The current server is suitable for PyTorch training and batch validation.
- Manual pygame feel testing is still better on a local display machine than
  through X11 forwarding because forwarding latency can dominate perceived
  controls.
- Continue to keep simulation, rendering, observations, and reward logic
  decoupled.
- Add focused tests for any new core behavior.
- Commit and push at meaningful checkpoints.
