# Setup

本项目当前使用 base conda 环境开发和运行。

## Base Environment

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
