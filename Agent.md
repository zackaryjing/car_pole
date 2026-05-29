# Agent Environment Record

Last updated: 2026-05-29 17:46 UTC

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
- Active conda environment: `base`
- Python executable: `/root/miniconda3/bin/python`
- Python version: `Python 3.13.11`

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

## Installed Packages In Current Base Env

Verified packages:

```text
numpy       2.4.2
pygame      2.6.1
pytest      9.0.3
torch       2.7.1+cu118
torchvision 0.22.1+cu118
torchaudio  2.7.1+cu118
```

PyTorch CUDA verification in a GPU-accessible context:

```text
torch 2.7.1+cu118
torchvision 0.22.1+cu118
torchaudio 2.7.1+cu118
cuda True 8
NVIDIA GeForce RTX 4090
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
python -m pytest
```

Result on this server:

```text
32 passed, 1 warning in 13.59s
```

The warning is from pygame importing deprecated `pkg_resources`; it does not
affect the tests.

## Useful Commands On This Server

Run full tests:

```bash
python -m pytest
```

Run manual play if a display/X forwarding is available, though local display is
preferred for real input latency:

```bash
python -m rl_racing.play --view follow --seed 0 --render-fps 60 --sim-speed 1.5
```

Short DQN smoke on GPU:

```bash
rl-racing-train-dqn \
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
rl-racing-train-dqn \
  --device cuda \
  --total-steps 200000 \
  --batch-size 256 \
  --run-name dqn_sensor_seed0
```

## Development Notes

- `AGENTS.md` currently describes another Linux workstation using
  `/home/jing/miniconda3/envs/system_dev/bin/python`. That is project-level
  guidance for that machine. For this `jzy` server session, use the environment
  recorded in this file unless the user asks to switch conda environments.
- The current server is suitable for PyTorch training and batch validation.
- Manual pygame feel testing is still better on a local display machine than
  through X11 forwarding because forwarding latency can dominate perceived
  controls.
- Continue to keep simulation, rendering, observations, and reward logic
  decoupled.
- Add focused tests for any new core behavior.
- Commit and push at meaningful checkpoints.

