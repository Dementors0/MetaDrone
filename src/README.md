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
# 推荐：使用 conda 环境
conda activate DiffPhysDrone

# 修复 dynamics_kernel.cu
sed -i.bak 's/\.type()/.scalar_type()/g' src/dynamics_kernel.cu

# 修复 quadsim_kernel.cu
sed -i.bak 's/\.type()/.scalar_type()/g' src/quadsim_kernel.cu

export CUDA_HOME=/usr/local/cuda

# 清理历史安装残留（可选）
rm -rf /home/robot/.conda/envs/DiffPhysDrone/lib/python3.11/site-packages/~uadsim-cuda*

# 关键：避免 editable + build isolation 导致的 "ModuleNotFoundError: torch"
PIP_NO_BUILD_ISOLATION=1 pip install ./src --no-build-isolation --no-use-pep517

# 验证导入（先导入 torch，再导入 quadsim_cuda）
python -c "import torch; import quadsim_cuda; print(torch.__version__)"
```

If you still see `ImportError: libc10.so: cannot open shared object file`, run:

```bash
export TORCH_LIB=$(python -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH
python -c "import torch; import quadsim_cuda; print(torch.__version__)"
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
