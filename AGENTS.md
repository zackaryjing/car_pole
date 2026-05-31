# AGENTS.md

Project-level instructions for Codex agents working in this repository.

## Project Purpose

This project is a from-scratch reinforcement learning playground built around a
2D pygame racing environment. The current phase includes game/environment
development plus an initial sensor-observation DQN pipeline. Keep manual play
responsive, deterministic, and testable while expanding RL training code.

Core design constraints:

- Keep simulation, reward calculation, observations, and rendering decoupled.
- Training code must use `RacingEnv.reset()` and `RacingEnv.step()` directly and
  must not depend on a visible pygame window.
- UI/manual-play code should expose testable timing and responsiveness paths so
  agents can verify frame pacing and event-loop behavior without manual play.
- New core behavior needs focused tests.

## Python Environment

On this Linux workstation, use the `agent_lab` conda environment:

```bash
/home/jing/miniconda3/envs/agent_lab/bin/python
```

Current verified version:

```text
Python 3.11.15
```

Use this interpreter explicitly for tests and Python commands in Codex sessions:

```bash
/home/jing/miniconda3/envs/agent_lab/bin/python -m pytest
```

Do not assume `python` is available on PATH.

## Cross-Machine Notes

This repository is developed across multiple platforms and Codex instances.
When adding a new machine-specific setup, document it in `docs/setup.md` and, if
it affects future agents, summarize the rule here.

## Useful Commands

Run the test suite:

```bash
/home/jing/miniconda3/envs/agent_lab/bin/python -m pytest
```

This local workstation has CPU-only PyTorch installed. It is suitable for game
development, DQN inference, and lightweight smoke tests. Full training
validation should happen on a machine with a suitable GPU.

Manual play on a machine with a display:

```bash
/home/jing/miniconda3/envs/agent_lab/bin/python -m rl_racing.play --view follow --seed 0
```

Headless tests:

```bash
SDL_VIDEODRIVER=dummy /home/jing/miniconda3/envs/agent_lab/bin/python -m pytest
```
