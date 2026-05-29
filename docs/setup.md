# Setup

本项目会在多台机器、多个 Codex 实例中开发。每台机器的推荐 Python 环境
需要记录在本文档中；Codex 项目级工作约定记录在仓库根目录的
`AGENTS.md`，这是 Codex CLI 当前默认会在启动项目时读取的项目指令文件。

## Environment Matrix

| Platform / machine | Environment | Python | Interpreter | Status |
| --- | --- | --- | --- | --- |
| 当前 Linux 工作站 | conda `system_dev` | 3.9.24 | `/home/jing/miniconda3/envs/system_dev/bin/python` | 当前 Codex 默认使用 |
| 服务器历史记录 | conda `base` | 3.13 | `/root/miniconda3/bin/python` | 历史记录，见下方 |
| 本地开发机 | conda `rl-racing` | 3.11 建议 | `python` after `conda activate rl-racing` | 建议配置 |

当前 Linux 工作站请显式使用：

```bash
/home/jing/miniconda3/envs/system_dev/bin/python
```

验证：

```bash
/home/jing/miniconda3/envs/system_dev/bin/python --version
/home/jing/miniconda3/envs/system_dev/bin/python -m pytest
```

当前已验证：

```text
Python 3.9.24
30 passed, 1 skipped
```

当前 Linux 工作站没有安装 PyTorch，也没有 NVIDIA GPU。DQN smoke tests 在
这里会 skip；训练正确性需要在安装了 PyTorch 的服务器或本地 GPU 机器上验证。

不要假设 `python` 命令在 Codex shell 中可用；当前 shell 中应使用上述绝对
路径。

## Historical Base Environment

以下是早期服务器 base 环境记录，保留用于迁移和排障。

当前服务器环境：

- Python: `/root/miniconda3/bin/python`
- Python version: 3.13
- GPU: 8 x NVIDIA GeForce RTX 4090
- Driver: 525.116.04
- Driver CUDA capability reported by `nvidia-smi`: 12.0

基础依赖：

```bash
python -m pip install numpy pygame pytest
```

CUDA 版 PyTorch：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  python -m pip install \
  torch==2.7.1+cu118 \
  torchvision==0.22.1+cu118 \
  torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

说明：

- 当前机器 driver 是 525.116.04，只报告 CUDA 12.0 capability。
- 不要安装 `cu128`、`cu130` 等 wheel；这些会因为 driver 太旧导致 `torch.cuda.is_available()` 为 `False`。
- 当前验证可用组合是 `torch 2.7.1+cu118`、`torchvision 0.22.1+cu118`、`torchaudio 2.7.1+cu118`。
- 下载时显式 unset proxy。这个服务器上默认代理对大 wheel 下载较慢，绕过代理后速度明显更稳定。

## Verification

验证 pygame：

```bash
python -c "import pygame; print(pygame.version.ver)"
```

验证 PyTorch 和 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

当前已验证结果：

```text
torch 2.7.1+cu118
torchvision 0.22.1+cu118
torchaudio 2.7.1+cu118
cuda True 8
NVIDIA GeForce RTX 4090
```

如果 `torch.cuda.is_available()` 为 `False`：

- 先确认 `nvidia-smi -L` 能列出 GPU。
- 再确认完整 `nvidia-smi` 能读取 driver 状态。
- 如果当前 shell 在受限 sandbox 中，换普通 shell 或提权上下文验证。当前 Codex 普通 sandbox 会出现 NVML/GPU 访问受限，但提权上下文可以正常看到 8 张 GPU。

## Manual Play Over SSH

有图形窗口时：

```bash
ssh -X user@server
python -m rl_racing.play --view follow --seed 0
```

本地需要 X server：

- Linux desktop: 通常自带。
- macOS: 安装并启动 XQuartz。
- Windows: VcXsrv、MobaXterm 或类似工具。

无窗口测试或训练：

```bash
SDL_VIDEODRIVER=dummy python -m pytest
```
