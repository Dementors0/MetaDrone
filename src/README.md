# Vision-based Agile Flight Training Code

## Overview

This repository contains the training code for our research on **Learning Vision-based Agile Flight via Differentiable Physics**.

## Environment Setup
### Python Environment

The code is tested with the following environment:

- **PyTorch**: 2.2.2
- **Python**: 3.11
- **CUDA**: 11.8

The code should be compatible with other PyTorch and CUDA versions.

### Build CUDA Ops

To build the CUDA operations, run the following command:



```bash
# 修复 dynamics_kernel.cu
sed -i.bak 's/\.type()/.scalar_type()/g' src/dynamics_kernel.cu

# 修复 quadsim_kernel.cu
sed -i.bak 's/\.type()/.scalar_type()/g' src/quadsim_kernel.cu

export CUDA_HOME=/usr/local/cuda
pip install -e src
```

## Training

To start the training process, use the following command:

```bash
# For multi-agemt
python main_cuda.py $(cat configs/multi_agent.args)
# For single-agemt
python main_cuda.py $(cat configs/single_agent.args)
```

## Evaluation

To evaluate the trained model in multi-agent settings, use the following command to launch the simulator:
```bash
cd <path to multi agent code supplementary>
./LinuxNoEditor/Blocks.sh -ResX=896 -ResY=504 -windowed -WinX=512 -WinY=304 -settings=$PWD/settings.json
```

Then, run the following command to evaluate the trained model:
```bash
python eval.py --resume /home/robot/validation_code/swarm_v1/swarm/swarm.pth --target_speed 2.5
```
