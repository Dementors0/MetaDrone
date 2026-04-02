#7.3
#修改地圖起始位置和終點
#可調整地圖大小和障礙物密度

import argparse
import atexit
import math
from collections import defaultdict
from random import normalvariate
import os
import datetime
import json
import sys
import multiprocessing as mp
import numpy as np
import matplotlib

matplotlib.use('Agg', force=True)

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("[SDP] flash=False, mem_efficient=False, math=True (for higher-order gradients)")

try:
    from env_multi import Env
except ModuleNotFoundError:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from env_multi import Env
try:
    from WorkNet_transformer import WorkNet
    from LossGenNet_transformer import LossGenNet
except ModuleNotFoundError:
    from WorkNet import WorkNet
    from LossGenNet import LossGenNet

########### 0. 工具类：动态归一化 ##########
class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-5):
        super().__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('count', torch.tensor(1e-4))
        self.epsilon = epsilon

    def forward(self, x, update=True):
        x = sanitize_tensor(x, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        if update:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
                batch_count = x.shape[0]

                delta = batch_mean - self.mean
                tot_count = self.count + batch_count

                new_mean = self.mean + delta * batch_count / tot_count
                m_a = self.var * self.count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
                new_var = M2 / tot_count

                self.mean.copy_(new_mean)
                self.var.copy_(new_var)
                self.count.copy_(tot_count)
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

class LossNormalizer:
    """Tracks running std of each loss component for scale normalization.
    Ensures all loss components contribute equally regardless of raw magnitude.
    Division is differentiable; statistics are detached."""
    def __init__(self, n_losses, momentum=0.01):
        self.running_std = [1.0] * n_losses
        self.momentum = momentum

    def normalize(self, *losses):
        normalized = []
        for i, loss in enumerate(losses):
            with torch.no_grad():
                batch_std = loss.detach().std()
                if not torch.isfinite(batch_std):
                    batch_std = torch.tensor(1.0, device=loss.device)
                batch_std = max(batch_std.item(), 1e-6)
                self.running_std[i] = (1 - self.momentum) * self.running_std[i] + self.momentum * batch_std
            normalized.append(loss / self.running_std[i])
        return normalized


def safe_normalize(x, dim=-1, eps=1e-6):
    return F.normalize(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dim=dim, eps=eps)


def sanitize_tensor(x, nan=0.0, posinf=1e3, neginf=-1e3):
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


@torch.no_grad()
def sanitize_module_(module, clamp_value=10.0):
    for p in module.parameters():
        p.data = sanitize_tensor(p.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value).clamp(-clamp_value, clamp_value)

########### 1. 参数配置 ##########
parser = argparse.ArgumentParser()

# ============================================================================
# 基础训练参数
# ============================================================================
parser.add_argument('--batch_size', type=int, default=8,
                    help='每次迭代的样本数')
parser.add_argument('--num_iters', type=int, default=5000000,
                    help='总训练迭代次数')
parser.add_argument('--exp_name', type=str, default="default",
                    help='实验名称标签，用于区分不同实验的保存目录')

# ============================================================================
# 双层优化策略参数 (Bilevel Optimization)
# ============================================================================
# 训练循环: 每 cycle_len = lgn_steps + worker_steps 步为一周期
# 前 lgn_steps 步训练 LGN (元学习器)，后 worker_steps 步训练 Worker (策略网络)
parser.add_argument('--lgn_steps', type=int, default=1,
                    help='每周期内 LGN 训练步数')
parser.add_argument('--worker_steps', type=int, default=5,
                    help='每周期内 Worker 训练步数')
parser.add_argument('--inner_lr', type=float, default=3e-3,
                    help='LGN phase 内循环可微梯度下降的学习率 (用于 unrolled bilevel)')
parser.add_argument('--inner_steps', type=int, default=1,
                    help='LGN phase 内循环可微 SGD 步数')

# ============================================================================
# 学习率
# ============================================================================
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Worker 网络学习率 (AdamW)')
parser.add_argument('--lgn_lr', type=float, default=2e-4,
                    help='LGN 网络学习率 (AdamW)')

# ============================================================================
# 环境物理参数
# ============================================================================
parser.add_argument('--grad_decay', type=float, default=0.4,
                    help='环境梯度衰减系数，传入 Env 构造函数')
parser.add_argument('--speed_mtp', type=float, default=1.0,
                    help='速度乘数，调节无人机最大速度')
parser.add_argument('--scene_scale', type=float, default=0.5,
                    help='场景缩放比例，调节障碍物区域大小和生成范围')
parser.add_argument('--obstacle_count_scale', type=float, default=0.5,
                    help='障碍物数量缩放比例 (球体/体素/圆柱)')
parser.add_argument('--soft_speed_limit_softness', type=float, default=0.05,
                    help='物理软限速平滑度，越小越接近硬截断')
parser.add_argument('--max_speed_ceiling', type=float, default=10.0,
                    help='env.max_speed 上限值 (m/s)')
parser.add_argument('--hard_vpred_clip', type=float, default=20.0,
                    help='env.run 中 v_pred 硬截断阈值')
parser.add_argument('--hard_speed_clip', type=float, default=30.0,
                    help='env.run 中速度张量硬截断阈值')
parser.add_argument('--start_goal_plane_y_abs', type=float, default=25,
                    help='起点/终点 Y 坐标绝对值，起点在 +Y，终点在 -Y')
parser.add_argument('--fov_x_half_tan', type=float, default=0.53,
                    help='深度相机水平视场角的 tan(fov_x/2)')
parser.add_argument('--cam_angle', type=int, default=10,
                    help='相机俯仰角度 (度)')

# ============================================================================
# Rollout 参数
# ============================================================================
parser.add_argument('--timesteps', type=int, default=300,
                    help='Worker phase 每 episode 最大步数')
parser.add_argument('--lgn_timesteps', type=int, default=300,
                    help='LGN phase 每 episode 步数，较小值可减少二阶梯度显存占用')
parser.add_argument('--detach_interval', type=int, default=12,
                    help='每隔 N 步截断时序记忆梯度以限制计算图深度 (<=0 禁用)')
parser.add_argument('--goal_radius', type=float, default=0.5,
                    help='所有无人机进入此半径内判定为到达终点，提前结束 episode (m)')
parser.add_argument('--maze_update_interval', type=int, default=50,
                    help='每 N 次迭代重新生成迷宫，中间仅重置无人机位置')
parser.add_argument('--height_floor', type=float, default=0.0,
                    help='高度下界(地面)，低于该值会受惩罚')
parser.add_argument('--height_ceiling', type=float, default=None,
                    help='高度上界(天花板)；默认使用环境 map_z_max')
parser.add_argument('--height_bound_sharpness', type=float, default=20.0,
                    help='地面/天花板约束软惩罚斜率，越大越接近硬约束')
parser.add_argument('--height_smooth_weight', type=float, default=0.2,
                    help='高度变化平滑惩罚权重，抑制相邻时刻 z 的剧烈变化')

# ============================================================================
# Transformer 序列长度
# ============================================================================
parser.add_argument('--worker_max_seq_len', type=int, default=32,
                    help='Worker Transformer 最大序列长度')
parser.add_argument('--lgn_max_seq_len', type=int, default=32,
                    help='LGN Transformer 最大序列长度')

# ============================================================================
# 环境模式 Flag
# ============================================================================
parser.add_argument('--single', default=False, action='store_true',
                    help='单无人机模式')
parser.add_argument('--gate', default=False, action='store_true',
                    help='启用门型障碍物')
parser.add_argument('--ground_voxels', default=False, action='store_true',
                    help='启用地面体素障碍物')
parser.add_argument('--scaffold', default=False, action='store_true',
                    help='启用脚手架障碍物')
parser.add_argument('--random_rotation', default=False, action='store_true',
                    help='启用随机旋转')
parser.add_argument('--no_odom', default=False, action='store_true',
                    help='不使用里程计速度作为状态输入 (state_dim 从 10 变为 7)')

# ============================================================================
# LGN 网络参数
# ============================================================================
parser.add_argument('--lgn_output_temperature', type=float, default=1.0,
                    help='LGN softmax 输出温度，越低权重分布越尖锐')
parser.add_argument('--lgn_weight_floor', type=float, default=0.01,
                    help='兼容参数(当前未启用下限约束)')

# ============================================================================
# Proxy Loss (Worker 训练目标) 参数
# ============================================================================
# 避障损失: loss = softplus((safe_margin - dist_obj) * 12) + ...
parser.add_argument('--avoid_safe_margin', type=float, default=0.35,
                    help='避障损失开始生效的距离阈值 (m)，进入此范围开始惩罚')

# 自适应速度目标: v_target = v_max * speed_factor_obs * speed_factor_goal
# speed_factor_obs: 近障碍物时减速 (sigmoid)
# speed_factor_goal: 近终点时减速 (线性)
parser.add_argument('--speed_goal_slow_dist', type=float, default=2.5,
                    help='距终点小于此距离时开始线性减速 (m)')
parser.add_argument('--speed_near_obs_floor', type=float, default=0.05,
                    help='近障碍物时速度因子下限，越小制动越强')

# ============================================================================
# Meta Loss (LGN 训练目标) 碰撞惩罚参数
# ============================================================================
# meta_coll = soft_weight * softplus(-dist) + hard_weight * depth^2 + event_weight * sigmoid(...)
parser.add_argument('--meta_coll_soft_weight', type=float, default=5.0,
                    help='软碰撞项权重，靠近墙壁即升高，提供连续梯度')
parser.add_argument('--meta_coll_hard_weight', type=float, default=40.0,
                    help='硬碰撞项权重，穿墙深度平方惩罚')
parser.add_argument('--meta_coll_event_weight', type=float, default=80.0,
                    help='碰撞事件项权重，episode 级碰撞惩罚')
parser.add_argument('--meta_coll_event_temp', type=float, default=80.0,
                    help='碰撞事件 sigmoid 温度，越大越接近阶跃')
parser.add_argument('--meta_coll_event_threshold', type=float, default=0.01,
                    help='碰撞事件触发阈值，穿入深度超过此值判定为碰撞 (m)')

# ============================================================================
# 全局规划引导 (Guidance) Meta Loss 参数
# ============================================================================
# A* 规划器生成参考路径，计算方向/速度/横向偏差等损失
parser.add_argument('--meta_guidance_weight', type=float, default=0.5,
                    help='全局引导损失总权重')
parser.add_argument('--guide_sample_count', type=int, default=10,
                    help='每 episode 采样计算引导损失的关键点数量')
parser.add_argument('--guide_sample_strategy', type=str, default='random',
                    choices=['random', 'uniform', 'adaptive', 'critical'],
                    help='采样策略: random/uniform/adaptive(危险+曲率)/critical(起终点+危险)')
parser.add_argument('--guide_max_accel', type=float, default=5.0,
                    help='梯形速度剖面最大加速度 (m/s^2)')
parser.add_argument('--guide_max_decel', type=float, default=6.0,
                    help='梯形速度剖面最大减速度 (m/s^2)')
parser.add_argument('--guide_dir_weight', type=float, default=0.5,
                    help='方向对齐损失权重')
parser.add_argument('--guide_speed_weight', type=float, default=0.3,
                    help='超速惩罚权重')
parser.add_argument('--guide_lateral_weight', type=float, default=0.3,
                    help='横向偏差惩罚权重 (到规划路径的几何距离)')
parser.add_argument('--guide_speed_diff_weight', type=float, default=0.2,
                    help='速度偏差惩罚权重 (超速+欠速)')
parser.add_argument('--guide_escape_weight', type=float, default=1.0,
                    help='已碰撞点的逃脱惩罚权重')
parser.add_argument('--guide_recovery_speed_weight', type=float, default=0.15,
                    help='规划无效点的速度抑制惩罚权重')
parser.add_argument('--guide_collision_threshold', type=float, default=-0.05,
                    help='判定为碰撞的穿入深度阈值 (m)')
parser.add_argument('--guide_accel_weight', type=float, default=0.1,
                    help='加速度不匹配惩罚权重 (减速需求)')

# ============================================================================
# A* 规划器参数
# ============================================================================
parser.add_argument('--planner_resolution', type=float, default=0.3,
                    help='占用栅格地图分辨率 (m)')
parser.add_argument('--planner_margin', type=float, default=0.15,
                    help='A* 规划器障碍物膨胀边距 (m)')
parser.add_argument('--planner_parallel', dest='planner_parallel', action='store_true',
                    help='启用多进程并行规划')
parser.add_argument('--no_planner_parallel', dest='planner_parallel', action='store_false',
                    help='禁用多进程并行规划')
parser.set_defaults(planner_parallel=True)
parser.add_argument('--planner_workers', type=int, default=0,
                    help='规划器进程数 (<=0 自动)')
parser.add_argument('--planner_pool_maxtasks', type=int, default=256,
                    help='进程池 maxtasksperchild，防止长时间运行内存泄漏')

# ============================================================================
# 梯度爆炸保护参数
# ============================================================================
# 检测到梯度范数超阈值或非有限时跳过 optimizer.step()
# 连续爆炸超过 skip_window 次后重置优化器状态
parser.add_argument('--grad_explosion_threshold', type=float, default=100.0,
                    help='梯度范数阈值，超过则跳过本次更新')
parser.add_argument('--grad_explosion_skip_window', type=int, default=5,
                    help='连续爆炸次数阈值，超过后重置优化器状态')
parser.add_argument('--enable_grad_protection', action='store_true', default=True,
                    help='启用梯度爆炸保护')

# ============================================================================
# Stuck Loss (卡住惩罚) 参数
# ============================================================================
# 检测并惩罚两种卡住状态:
# 1. loss_stuck: 局部窗口内位移过小 → softplus((threshold - displacement) * 10)
# 2. loss_collision_duration: 连续碰撞累计步数 → streak * in_collision
parser.add_argument('--stuck_loss_weight', type=float, default=2.0,
                    help='局部位移惩罚权重')
parser.add_argument('--stuck_window', type=int, default=15,
                    help='卡住检测时间窗口 (步数)')
parser.add_argument('--stuck_displacement_threshold', type=float, default=0.3,
                    help='窗口内最小期望位移 (m)，低于此值触发惩罚')
parser.add_argument('--collision_duration_weight', type=float, default=10.0,
                    help='碰撞持续时间惩罚权重，连续碰撞越久惩罚越重')
parser.add_argument('--meta_smooth_jerk_weight', type=float, default=0.001,
                    help='元损失中动作一阶差分(jerk)平滑惩罚权重，抑制姿态抖动')
parser.add_argument('--meta_smooth_snap_weight', type=float, default=0.0002,
                    help='元损失中归一化动作二阶差分(snap)平滑惩罚权重，抑制高频震荡')
parser.add_argument('--meta_smooth_v_pred_weight', type=float, default=0.1,
                    help='元损失中速度预测误差权重，促进网络学习准确的速度预测')

args = parser.parse_args()

# Planner parallel runtime config (used by guidance reference computation)
PLANNER_PARALLEL_ENABLE = bool(args.planner_parallel)
PLANNER_NUM_WORKERS = int(args.planner_workers)
PLANNER_POOL_MAXTASKS = max(1, int(args.planner_pool_maxtasks))
_PLANNER_POOL = None
_PLANNER_POOL_SIZE = 0

########## 2. 目录与日志初始化 ##########
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
script_name = os.path.splitext(os.path.basename(__file__))[0]
save_dir_name = f"{script_name}_{args.exp_name}_{current_time}"
save_dir = os.path.join("..", "checkpoints", save_dir_name)
video_dir = os.path.join(save_dir, 'videos')

os.makedirs(save_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
print(f"Training artifacts will be saved to: {save_dir}")

with open(os.path.join(save_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))

device = torch.device('cuda')

########## 3. 环境初始化 ##########
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          scene_scale=args.scene_scale,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle,
          obstacle_count_scale=args.obstacle_count_scale,
          speed_limit_softness=args.soft_speed_limit_softness,
          max_speed_ceiling=args.max_speed_ceiling,
          hard_vpred_clip=args.hard_vpred_clip,
          hard_speed_clip=args.hard_speed_clip,
          start_goal_plane_y_abs=args.start_goal_plane_y_abs)

state_dim = 7 if args.no_odom else 10

if args.no_odom:
    try:
        worknet = WorkNet(7, 6, max_seq_len=args.worker_max_seq_len)
    except TypeError:
        worknet = WorkNet(7, 6)
else:
    try:
        worknet = WorkNet(7 + 3, 6, max_seq_len=args.worker_max_seq_len)
    except TypeError:
        worknet = WorkNet(7 + 3, 6)
worknet = worknet.to(device)

try:
    lgn = LossGenNet(
        state_dim=state_dim,
        max_seq_len=args.lgn_max_seq_len,
        output_temperature=args.lgn_output_temperature,
        weight_floor=args.lgn_weight_floor,
    ).to(device)
except TypeError:
    lgn = LossGenNet(state_dim=state_dim).to(device)
state_normalizer = RunningMeanStd(shape=(state_dim,)).to(device)

########## 4. 优化器配置 ##########
optim_worker = AdamW(worknet.parameters(), args.lr)
optim_lgn = AdamW(lgn.parameters(), args.lgn_lr)
sched = CosineAnnealingLR(optim_worker, args.num_iters, args.lr * 0.01)

loss_normalizer = LossNormalizer(4)  # normalize 4 loss components to equal scale

########## 6. 辅助函数 ##########
scaler_q = defaultdict(list)

def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        scaler_q[k].append(float(v))

def is_save_iter(i):
    return (i + 1) % 10000 == 0 if i >= 2000 else (i + 1) % 500 == 0


def is_save_trajectory_iter(i):
    return i == 0 or (i + 1) % 500 == 0


def rotation_matrix_to_rpy_deg(R):
    """Convert rotation matrix to roll-pitch-yaw in degrees (ZYX convention)."""
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]
    r10 = R[..., 1, 0]
    r00 = R[..., 0, 0]

    pitch = torch.asin(torch.clamp(-r20, -1.0, 1.0))
    roll = torch.atan2(r21, r22)
    yaw = torch.atan2(r10, r00)
    return torch.rad2deg(torch.stack([roll, pitch, yaw], dim=-1))

def get_grad_stats(module):
    total_sq = 0.0
    max_abs = 0.0
    nonfinite_cnt = 0
    grad_elem_cnt = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        finite_mask = torch.isfinite(g)
        nonfinite_cnt += int((~finite_mask).sum().item())
        if finite_mask.any():
            g_finite = g[finite_mask]
            total_sq += float((g_finite * g_finite).sum().item())
            max_abs = max(max_abs, float(g_finite.abs().max().item()))
        grad_elem_cnt += g.numel()
    global_norm = math.sqrt(total_sq)
    return global_norm, max_abs, nonfinite_cnt, grad_elem_cnt

def get_grad_norm_from_grads(grads):
    total_sq = 0.0
    nonfinite_cnt = 0
    grad_elem_cnt = 0
    for g in grads:
        if g is None:
            continue
        g = g.detach()
        finite_mask = torch.isfinite(g)
        nonfinite_cnt += int((~finite_mask).sum().item())
        if finite_mask.any():
            g_finite = g[finite_mask]
            total_sq += float((g_finite * g_finite).sum().item())
        grad_elem_cnt += g.numel()
    return math.sqrt(total_sq), nonfinite_cnt, grad_elem_cnt

def get_loss_to_worker_grad_norm(loss, params):
    if not loss.requires_grad:
        return 0.0, 0, 0
    grads = torch.autograd.grad(
        loss, params, allow_unused=True, retain_graph=True, create_graph=False,
    )
    return get_grad_norm_from_grads(grads)


def merge_intervals(intervals, min_gap=1e-4):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + min_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


########## 6.1 全局规划引导元损失辅助函数 ##########

import heapq
from typing import List, Tuple, Optional, Dict

class GlobalPlanner:
    """
    3D A* 全局路径规划器

    构建占用栅格地图并使用 A* 算法规划从起点到终点的最优路径，
    然后从路径中提取参考方向、速度和加速度。
    """

    def __init__(self, resolution: float = 0.3, margin: float = 0.15,
                 z_min: float = 0.0, z_max: float = 2.5, device='cuda'):
        """
        Args:
            resolution: 栅格分辨率 (米)
            margin: 安全边距 (米)，障碍物膨胀量
            z_min, z_max: Z轴范围
            device: 计算设备
        """
        self.resolution = resolution
        self.margin = margin
        self.z_min = z_min
        self.z_max = z_max
        self.device = device

        # 缓存的占用栅格地图
        self.occupancy_grid = None
        self.grid_origin = None  # [x_min, y_min, z_min]
        self.grid_shape = None   # [nx, ny, nz]

        # 缓存的规划路径 (每个 batch 元素一条路径)
        self.cached_paths = {}   # batch_idx -> path tensor [N, 3]
        self.plan_stats = {'success': 0, 'total': 0}

        # 3D 邻居偏移 (26-连通)
        self._neighbors_26 = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cost = math.sqrt(dx**2 + dy**2 + dz**2)
                    self._neighbors_26.append((dx, dy, dz, cost))

    def build_occupancy_grid(self, env, batch_idx: int = 0):
        """
        从环境障碍物构建 3D 占用栅格地图

        Args:
            env: 环境对象，包含 voxels, balls, cyl, cyl_h 等障碍物
            batch_idx: batch 索引
        """
        # 确定地图边界
        x_min, x_max = -15.0, 15.0
        y_min, y_max = -25.0, 25.0

        if hasattr(env, 'p_target') and env.p_target is not None:
            target = env.p_target[batch_idx].detach().cpu()
            y_min = min(y_min, float(target[1]) - 5.0)
            y_max = max(y_max, float(target[1]) + 5.0)

        # 栅格尺寸
        nx = int(math.ceil((x_max - x_min) / self.resolution))
        ny = int(math.ceil((y_max - y_min) / self.resolution))
        nz = int(math.ceil((self.z_max - self.z_min) / self.resolution))

        self.grid_origin = np.array([x_min, y_min, self.z_min])
        self.grid_shape = (nx, ny, nz)

        # 初始化为空闲
        self.occupancy_grid = np.zeros((nx, ny, nz), dtype=np.uint8)

        # 填充障碍物
        total_margin = self.margin + 0.15  # 额外的安全边距

        # 1. 体素盒体障碍物
        if hasattr(env, 'voxels') and env.voxels.numel() > 0:
            voxels = env.voxels[batch_idx].detach().cpu().numpy()
            for vox in voxels:
                cx, cy, cz, hx, hy, hz = vox[:6]
                if hx > 15 or hy > 15 or hz > 15:  # 跳过占位大盒体
                    continue
                # 膨胀障碍物
                x0 = max(0, int((cx - hx - total_margin - x_min) / self.resolution))
                x1 = min(nx, int((cx + hx + total_margin - x_min) / self.resolution) + 1)
                y0 = max(0, int((cy - hy - total_margin - y_min) / self.resolution))
                y1 = min(ny, int((cy + hy + total_margin - y_min) / self.resolution) + 1)
                z0 = max(0, int((cz - hz - total_margin - self.z_min) / self.resolution))
                z1 = min(nz, int((cz + hz + total_margin - self.z_min) / self.resolution) + 1)
                self.occupancy_grid[x0:x1, y0:y1, z0:z1] = 1

        # 2. 球形障碍物
        if hasattr(env, 'balls') and env.balls.numel() > 0:
            balls = env.balls[batch_idx].detach().cpu().numpy()
            for ball in balls:
                bx, by, bz, br = ball[:4]
                r_inflated = br + total_margin
                # 球的包围盒
                x0 = max(0, int((bx - r_inflated - x_min) / self.resolution))
                x1 = min(nx, int((bx + r_inflated - x_min) / self.resolution) + 1)
                y0 = max(0, int((by - r_inflated - y_min) / self.resolution))
                y1 = min(ny, int((by + r_inflated - y_min) / self.resolution) + 1)
                z0 = max(0, int((bz - r_inflated - self.z_min) / self.resolution))
                z1 = min(nz, int((bz + r_inflated - self.z_min) / self.resolution) + 1)
                # 精确球形检测
                for ix in range(x0, x1):
                    for iy in range(y0, y1):
                        for iz in range(z0, z1):
                            px = x_min + (ix + 0.5) * self.resolution
                            py = y_min + (iy + 0.5) * self.resolution
                            pz = self.z_min + (iz + 0.5) * self.resolution
                            if (px - bx)**2 + (py - by)**2 + (pz - bz)**2 < r_inflated**2:
                                self.occupancy_grid[ix, iy, iz] = 1

        # 3. 竖直圆柱障碍物 (沿Z轴)
        if hasattr(env, 'cyl') and env.cyl.numel() > 0:
            cyl = env.cyl[batch_idx].detach().cpu().numpy()
            for c in cyl:
                cx, cy, cr = c[:3]
                r_inflated = cr + total_margin
                x0 = max(0, int((cx - r_inflated - x_min) / self.resolution))
                x1 = min(nx, int((cx + r_inflated - x_min) / self.resolution) + 1)
                y0 = max(0, int((cy - r_inflated - y_min) / self.resolution))
                y1 = min(ny, int((cy + r_inflated - y_min) / self.resolution) + 1)
                for ix in range(x0, x1):
                    for iy in range(y0, y1):
                        px = x_min + (ix + 0.5) * self.resolution
                        py = y_min + (iy + 0.5) * self.resolution
                        if (px - cx)**2 + (py - cy)**2 < r_inflated**2:
                            self.occupancy_grid[ix, iy, :] = 1

        # 4. 水平圆柱障碍物 (沿Y轴)
        if hasattr(env, 'cyl_h') and env.cyl_h.numel() > 0:
            cyl_h = env.cyl_h[batch_idx].detach().cpu().numpy()
            for c in cyl_h:
                cx, cz, cr = c[:3]
                r_inflated = cr + total_margin
                x0 = max(0, int((cx - r_inflated - x_min) / self.resolution))
                x1 = min(nx, int((cx + r_inflated - x_min) / self.resolution) + 1)
                z0 = max(0, int((cz - r_inflated - self.z_min) / self.resolution))
                z1 = min(nz, int((cz + r_inflated - self.z_min) / self.resolution) + 1)
                for ix in range(x0, x1):
                    for iz in range(z0, z1):
                        px = x_min + (ix + 0.5) * self.resolution
                        pz = self.z_min + (iz + 0.5) * self.resolution
                        if (px - cx)**2 + (pz - cz)**2 < r_inflated**2:
                            self.occupancy_grid[ix, :, iz] = 1

        return self.occupancy_grid

    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """世界坐标转栅格索引"""
        idx = ((pos - self.grid_origin) / self.resolution).astype(int)
        return tuple(np.clip(idx, 0, np.array(self.grid_shape) - 1))

    def grid_to_world(self, idx: Tuple[int, int, int]) -> np.ndarray:
        """栅格索引转世界坐标（单元格中心）"""
        return self.grid_origin + (np.array(idx) + 0.5) * self.resolution

    def is_valid(self, idx: Tuple[int, int, int]) -> bool:
        """检查栅格索引是否有效且空闲"""
        nx, ny, nz = self.grid_shape
        if not (0 <= idx[0] < nx and 0 <= idx[1] < ny and 0 <= idx[2] < nz):
            return False
        return self.occupancy_grid[idx[0], idx[1], idx[2]] == 0

    def heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """A* 启发式函数（欧几里得距离）"""
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) * self.resolution

    def plan_astar(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                   max_iterations: int = 50000) -> Optional[List[np.ndarray]]:
        """
        3D A* 路径规划

        Args:
            start_pos: 起点世界坐标 [3]
            goal_pos: 终点世界坐标 [3]
            max_iterations: 最大迭代次数

        Returns:
            path: 路径点列表 [N, 3] 或 None（规划失败）
        """
        if self.occupancy_grid is None:
            return None

        start_idx = self.world_to_grid(start_pos)
        goal_idx = self.world_to_grid(goal_pos)

        # 如果起点在障碍物内，尝试找到最近的空闲点
        if not self.is_valid(start_idx):
            start_idx = self._find_nearest_free(start_idx)
            if start_idx is None:
                return None

        # 如果终点在障碍物内，尝试找到最近的空闲点
        if not self.is_valid(goal_idx):
            goal_idx = self._find_nearest_free(goal_idx)
            if goal_idx is None:
                return None

        # A* 搜索
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start_idx, goal_idx), 0, start_idx))

        came_from = {}
        g_score = {start_idx: 0}

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current_g, current = heapq.heappop(open_set)

            # 到达目标
            if current == goal_idx:
                # 重建路径
                path = [self.grid_to_world(current)]
                while current in came_from:
                    current = came_from[current]
                    path.append(self.grid_to_world(current))
                path.reverse()
                return path

            # 跳过过期节点
            if current_g > g_score.get(current, float('inf')):
                continue

            # 扩展邻居
            for dx, dy, dz, move_cost in self._neighbors_26:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                if not self.is_valid(neighbor):
                    continue

                tentative_g = g_score[current] + move_cost * self.resolution

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal_idx)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # 规划失败
        return None

    def _find_nearest_free(self, idx: Tuple[int, int, int], search_radius: int = 10) -> Optional[Tuple[int, int, int]]:
        """寻找最近的空闲栅格"""
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if abs(dx) == r or abs(dy) == r or abs(dz) == r:
                            neighbor = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                            if self.is_valid(neighbor):
                                return neighbor
        return None

    def smooth_path(self, path: List[np.ndarray], window_size: int = 5) -> np.ndarray:
        """
        路径平滑处理（移动平均）

        Args:
            path: 原始路径点列表
            window_size: 平滑窗口大小

        Returns:
            smoothed_path: 平滑后的路径 [N, 3]
        """
        if len(path) < 3:
            return np.array(path)

        path_array = np.array(path)
        smoothed = np.copy(path_array)

        half_w = window_size // 2
        for i in range(half_w, len(path) - half_w):
            smoothed[i] = path_array[max(0, i-half_w):min(len(path), i+half_w+1)].mean(axis=0)

        # 保持起点和终点不变
        smoothed[0] = path_array[0]
        smoothed[-1] = path_array[-1]

        return smoothed

    def extract_reference_from_path(self, current_pos: np.ndarray, path: np.ndarray,
                                    max_speed: float = 5.0,
                                    max_accel: float = 5.0,
                                    max_decel: float = 6.0,
                                    lookahead_dist: float = 1.0) -> Dict:
        """
        从规划路径中提取当前位置的参考方向、速度和加速度

        使用梯形速度剖面和动力学约束计算参考值：
        - 速度剖面考虑：最大速度、曲率限制、终点减速
        - 加速度基于速度剖面的变化率计算
        - 横向偏差为到路径的真正几何距离

        Args:
            current_pos: 当前位置 [3]
            path: 规划路径 [N, 3]
            max_speed: 最大速度 (m/s)
            max_accel: 最大加速度 (m/s^2)
            max_decel: 最大减速度 (m/s^2)，正值
            lookahead_dist: 前瞻距离 (m)

        Returns:
            dict: 包含参考方向、速度、加速度、曲率、横向偏差等
        """
        if path is None or len(path) < 2:
            return {
                'direction': np.array([0.0, 1.0, 0.0]),
                'speed': max_speed * 0.5,
                'acceleration': np.array([0.0, 0.0, 0.0]),
                'curvature': 0.0,
                'path_progress': 0.0,
                'dist_to_goal': 10.0,
                'lateral_error': 0.0,  # 横向偏差
                'valid': False
            }

        # ========== 1. 找到路径上最近点并计算横向偏差 ==========
        distances = np.linalg.norm(path - current_pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_point = path[nearest_idx]

        # 横向偏差：当前位置到路径最近点的距离
        lateral_error = distances[nearest_idx]

        # 更精确的横向偏差：投影到路径线段上
        if 0 < nearest_idx < len(path) - 1:
            # 检查是否应该投影到前一段或后一段
            for seg_start, seg_end in [(nearest_idx - 1, nearest_idx), (nearest_idx, nearest_idx + 1)]:
                if seg_end >= len(path):
                    continue
                p0 = path[seg_start]
                p1 = path[seg_end]
                seg_vec = p1 - p0
                seg_len = np.linalg.norm(seg_vec)
                if seg_len > 1e-6:
                    seg_dir = seg_vec / seg_len
                    t = np.clip(np.dot(current_pos - p0, seg_dir), 0, seg_len)
                    proj_point = p0 + t * seg_dir
                    proj_dist = np.linalg.norm(current_pos - proj_point)
                    if proj_dist < lateral_error:
                        lateral_error = proj_dist
                        nearest_point = proj_point

        # ========== 2. 计算路径进度和剩余路径长度 ==========
        path_progress = nearest_idx / max(len(path) - 1, 1)

        # 计算从当前点到终点的路径长度
        remaining_path_length = 0.0
        for i in range(nearest_idx, len(path) - 1):
            remaining_path_length += np.linalg.norm(path[i+1] - path[i])
        dist_to_goal = remaining_path_length

        # ========== 3. 计算前瞻点和参考方向 ==========
        lookahead_idx = nearest_idx
        accumulated_dist = 0.0
        for i in range(nearest_idx, len(path) - 1):
            segment_dist = np.linalg.norm(path[i+1] - path[i])
            accumulated_dist += segment_dist
            if accumulated_dist >= lookahead_dist:
                lookahead_idx = i + 1
                break
        else:
            lookahead_idx = len(path) - 1

        lookahead_point = path[lookahead_idx]
        direction_vec = lookahead_point - current_pos
        direction_norm = np.linalg.norm(direction_vec)
        if direction_norm > 1e-6:
            direction = direction_vec / direction_norm
        else:
            if nearest_idx < len(path) - 1:
                direction = path[nearest_idx + 1] - path[nearest_idx]
                direction = direction / (np.linalg.norm(direction) + 1e-6)
            else:
                direction = np.array([0.0, 1.0, 0.0])

        # ========== 4. 计算局部曲率（用于速度限制）==========
        curvature = 0.0
        if 1 <= nearest_idx < len(path) - 1:
            v1 = path[nearest_idx] - path[nearest_idx - 1]
            v2 = path[nearest_idx + 1] - path[nearest_idx]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm > 1e-6 and v2_norm > 1e-6:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle_change = math.acos(cos_angle)
                curvature = angle_change / (self.resolution + 1e-6)

        # ========== 5. 计算梯形速度剖面 ==========
        # 5.1 曲率速度限制：v_max_curve = sqrt(a_lateral_max / curvature)
        a_lateral_max = 4.0  # 最大侧向加速度 (m/s^2)
        if curvature > 0.1:
            v_curve_limit = min(max_speed, math.sqrt(a_lateral_max / (curvature + 1e-6)))
        else:
            v_curve_limit = max_speed

        # 5.2 终点减速限制：v^2 = 2 * a_decel * d
        # 在终点速度应为 0，所以 v_max_decel = sqrt(2 * max_decel * dist_to_goal)
        v_decel_limit = math.sqrt(2.0 * max_decel * max(dist_to_goal, 0.01))

        # 5.3 前方曲率预瞰（提前减速）
        v_lookahead_limit = max_speed
        lookahead_curvature_dist = 2.0  # 向前看 2m
        accumulated = 0.0
        max_future_curvature = 0.0
        for i in range(nearest_idx, min(nearest_idx + 20, len(path) - 1)):
            seg_len = np.linalg.norm(path[i+1] - path[i])
            accumulated += seg_len
            if accumulated > lookahead_curvature_dist:
                break
            if 1 <= i < len(path) - 1:
                v1 = path[i] - path[i-1]
                v2 = path[i+1] - path[i]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_a = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
                    future_curv = math.acos(cos_a) / (self.resolution + 1e-6)
                    max_future_curvature = max(max_future_curvature, future_curv)

        if max_future_curvature > 0.1:
            v_future = math.sqrt(a_lateral_max / (max_future_curvature + 1e-6))
            # 需要在 lookahead_curvature_dist 内减速到 v_future
            v_lookahead_limit = math.sqrt(v_future**2 + 2.0 * max_decel * lookahead_curvature_dist)

        # 5.4 综合速度限制
        ref_speed = min(max_speed, v_curve_limit, v_decel_limit, v_lookahead_limit)
        ref_speed = max(ref_speed, 0.1)  # 最小速度

        # ========== 6. 计算参考加速度（基于速度剖面变化）==========
        # 切向加速度：基于当前位置的速度限制变化
        tangent_accel = 0.0

        # 如果当前速度限制低于最大速度，说明需要减速
        if ref_speed < max_speed * 0.9:
            # 估算需要的减速度
            if dist_to_goal > 0.1:
                # v_final^2 - v_current^2 = 2 * a * d
                # 假设需要在 dist_to_goal 内减到 0
                tangent_accel = -(ref_speed ** 2) / (2 * max(dist_to_goal, 0.1))
                tangent_accel = max(tangent_accel, -max_decel)
            else:
                tangent_accel = -max_decel

        # 如果在弯道，额外减速
        if curvature > 0.5:
            curve_decel = -curvature * 1.5  # 曲率越大，减速越多
            tangent_accel = min(tangent_accel, curve_decel)

        # 限制加速度范围
        tangent_accel = np.clip(tangent_accel, -max_decel, max_accel)

        # 加速度向量 = 切向加速度 * 方向
        acceleration = direction * tangent_accel

        return {
            'direction': direction,
            'speed': ref_speed,
            'acceleration': acceleration,
            'curvature': curvature,
            'path_progress': path_progress,
            'dist_to_goal': dist_to_goal,
            'lateral_error': lateral_error,  # 新增：横向偏差
            'valid': True
        }

    def plan_and_cache(self, env, start_positions: torch.Tensor,
                       goal_positions: torch.Tensor, batch_indices: List[int] = None):
        """
        为多个 batch 元素规划并缓存路径

        Args:
            env: 环境对象
            start_positions: 起点位置 [B, 3]
            goal_positions: 目标位置 [B, 3]
            batch_indices: 要规划的 batch 索引列表，None 表示全部
        """
        B = start_positions.shape[0]
        if batch_indices is None:
            batch_indices = list(range(B))

        success_count = 0

        for b in batch_indices:
            # 为该 batch 构建占用栅格
            self.build_occupancy_grid(env, batch_idx=b)

            start = start_positions[b].detach().cpu().numpy()
            goal = goal_positions[b].detach().cpu().numpy()

            # A* 规划
            path = self.plan_astar(start, goal)

            if path is not None:
                # 路径平滑
                path = self.smooth_path(path, window_size=5)
                self.cached_paths[b] = path
                success_count += 1
            else:
                # 规划失败：标记为 None，由恢复项接管，避免强制拟合不可靠参考
                self.cached_paths[b] = None

        self.plan_stats = {
            'success': int(success_count),
            'total': int(len(batch_indices)),
        }

    def clear_cache(self):
        """清除缓存的路径"""
        self.cached_paths.clear()
        self.occupancy_grid = None


def _compute_planner_worker_count():
    if PLANNER_NUM_WORKERS > 0:
        return PLANNER_NUM_WORKERS
    cpu_total = os.cpu_count() or 8
    # 在高核机器上预留少量核给训练主进程与系统线程
    return max(1, min(28, cpu_total - 3))


def _shutdown_planner_pool():
    global _PLANNER_POOL, _PLANNER_POOL_SIZE
    if _PLANNER_POOL is not None:
        _PLANNER_POOL.close()
        _PLANNER_POOL.join()
        _PLANNER_POOL = None
        _PLANNER_POOL_SIZE = 0


def _get_planner_pool():
    global _PLANNER_POOL, _PLANNER_POOL_SIZE
    if not PLANNER_PARALLEL_ENABLE:
        return None

    target_size = _compute_planner_worker_count()
    if _PLANNER_POOL is not None and _PLANNER_POOL_SIZE == target_size:
        return _PLANNER_POOL

    if _PLANNER_POOL is not None:
        _shutdown_planner_pool()

    # 本脚本顶层直接启动训练，spawn 会重入模块；Linux 下优先 fork。
    start_method = 'fork' if sys.platform.startswith('linux') else 'spawn'
    ctx = mp.get_context(start_method)
    _PLANNER_POOL = ctx.Pool(processes=target_size, maxtasksperchild=PLANNER_POOL_MAXTASKS)
    _PLANNER_POOL_SIZE = target_size
    return _PLANNER_POOL


atexit.register(_shutdown_planner_pool)


def _plan_sample_points_worker(payload):
    """Per-sample planning worker: build occupancy once and solve all sampled points."""
    voxels_np = payload['voxels']
    balls_np = payload['balls']
    cyl_np = payload['cyl']
    cyl_h_np = payload['cyl_h']
    sampled_positions = payload['sampled_positions']  # [S, 3]
    sampled_dist = payload['sampled_dist']            # [S]
    goal_np = payload['goal']                         # [3]
    resolution = float(payload['resolution'])
    margin = float(payload['margin'])
    z_min = float(payload['z_min'])
    z_max = float(payload['z_max'])
    max_speed = float(payload['max_speed'])
    max_accel = float(payload['max_accel'])
    max_decel = float(payload['max_decel'])
    lookahead_dist = float(payload['lookahead_dist'])
    invalid_dist_threshold = float(payload['invalid_dist_threshold'])

    planner = GlobalPlanner(resolution=resolution, margin=margin, z_min=z_min, z_max=z_max, device='cpu')

    class _EnvShim:
        pass

    env_shim = _EnvShim()
    env_shim.voxels = torch.from_numpy(voxels_np).unsqueeze(0)
    env_shim.balls = torch.from_numpy(balls_np).unsqueeze(0)
    env_shim.cyl = torch.from_numpy(cyl_np).unsqueeze(0)
    env_shim.cyl_h = torch.from_numpy(cyl_h_np).unsqueeze(0)
    env_shim.p_target = torch.from_numpy(goal_np).reshape(1, 3)

    planner.build_occupancy_grid(env_shim, batch_idx=0)

    S = sampled_positions.shape[0]
    ref_direction = np.zeros((S, 3), dtype=np.float32)
    ref_speed = np.zeros((S,), dtype=np.float32)
    ref_acceleration = np.zeros((S, 3), dtype=np.float32)
    lateral_error = np.zeros((S,), dtype=np.float32)
    valid_mask = np.zeros((S,), dtype=np.bool_)

    plan_total = 0
    plan_success = 0
    curv_sum = 0.0
    prog_sum = 0.0
    lat_sum = 0.0
    lat_max = 0.0
    metric_count = 0

    for s in range(S):
        pos_np = sampled_positions[s]
        dist = float(sampled_dist[s])

        if dist < invalid_dist_threshold:
            continue

        plan_total += 1
        path = planner.plan_astar(pos_np, goal_np)
        if path is None:
            continue
        path = planner.smooth_path(path, window_size=5)

        ref_info = planner.extract_reference_from_path(
            pos_np,
            path,
            max_speed=max_speed,
            max_accel=max_accel,
            max_decel=max_decel,
            lookahead_dist=lookahead_dist,
        )

        ref_direction[s] = np.asarray(ref_info['direction'], dtype=np.float32)
        ref_speed[s] = float(ref_info['speed'])
        ref_acceleration[s] = np.asarray(ref_info['acceleration'], dtype=np.float32)
        lateral_error[s] = float(ref_info.get('lateral_error', 0.0))
        valid_mask[s] = bool(ref_info['valid'])
        if valid_mask[s]:
            plan_success += 1

        curv_sum += float(ref_info.get('curvature', 0.0))
        prog_sum += float(ref_info.get('path_progress', 0.0))
        lat = float(ref_info.get('lateral_error', 0.0))
        lat_sum += lat
        lat_max = max(lat_max, lat)
        metric_count += 1

    return {
        'ref_direction': ref_direction,
        'ref_speed': ref_speed,
        'ref_acceleration': ref_acceleration,
        'lateral_error': lateral_error,
        'valid_mask': valid_mask,
        'plan_total': int(plan_total),
        'plan_success': int(plan_success),
        'curv_sum': float(curv_sum),
        'prog_sum': float(prog_sum),
        'lat_sum': float(lat_sum),
        'lat_max': float(lat_max),
        'metric_count': int(metric_count),
    }


# 全局规划器实例（在迷宫更新时重新规划）
global_planner = GlobalPlanner(resolution=0.3, margin=0.15)


def compute_guidance_reference_from_planner(env, p, v, p_target, dist_obj, planner: GlobalPlanner,
                                             max_speed=5.0, max_accel=5.0, max_decel=6.0,
                                             lookahead_dist=1.0, invalid_dist_threshold=-0.05):
    """
    使用全局规划器生成参考方向、速度、加速度和横向偏差

    基于 A* 规划路径和梯形速度剖面计算动力学可行的参考值：
    - 方向：指向前瞻点
    - 速度：考虑曲率、终点减速、动力学约束
    - 加速度：基于速度剖面变化率
    - 横向偏差：到规划路径的真正几何距离

    Args:
        p: 位置 [S, B, 3] 或 [B, 3]
        v: 速度 [S, B, 3] 或 [B, 3]
        p_target: 目标位置 [B, 3]
        dist_obj: 到障碍物的距离 [S, B] 或 [B]
        planner: 全局规划器实例
        max_speed: 最大速度 (m/s)
        max_accel: 最大加速度 (m/s^2)
        max_decel: 最大减速度 (m/s^2)
        lookahead_dist: 前瞻距离 (m)

    Returns:
        ref_direction: 参考方向 [S, B, 3]
        ref_speed: 参考速度 [S, B]
        ref_acceleration: 参考加速度 [S, B, 3]
        lateral_error: 横向偏差 [S, B]，到规划路径的距离
        valid_mask: 有效性掩码 [S, B]
        planner_info: 规划器信息字典
    """
    squeeze_output = False
    if p.dim() == 2:
        p = p.unsqueeze(0)
        v = v.unsqueeze(0)
        dist_obj = dist_obj.unsqueeze(0)
        squeeze_output = True

    S, B, _ = p.shape
    device = p.device

    ref_direction = torch.zeros(S, B, 3, device=device)
    ref_speed = torch.zeros(S, B, device=device)
    ref_acceleration = torch.zeros(S, B, 3, device=device)
    lateral_error = torch.zeros(S, B, device=device)
    valid_mask = torch.zeros(S, B, dtype=torch.bool, device=device)

    curv_sum = 0.0
    prog_sum = 0.0
    lat_sum = 0.0
    lat_max = 0.0
    metric_count = 0
    sample_plan_total = 0
    sample_plan_success = 0

    used_parallel = False
    pool = _get_planner_pool()
    if pool is not None and B > 1:
        try:
            payloads = []
            for b in range(B):
                vox_np = env.voxels[b].detach().cpu().numpy() if hasattr(env, 'voxels') else np.zeros((0, 6), dtype=np.float32)
                balls_np = env.balls[b].detach().cpu().numpy() if hasattr(env, 'balls') else np.zeros((0, 4), dtype=np.float32)
                cyl_np = env.cyl[b].detach().cpu().numpy() if hasattr(env, 'cyl') else np.zeros((0, 3), dtype=np.float32)
                cyl_h_np = env.cyl_h[b].detach().cpu().numpy() if hasattr(env, 'cyl_h') else np.zeros((0, 3), dtype=np.float32)

                payloads.append({
                    'voxels': np.asarray(vox_np, dtype=np.float32),
                    'balls': np.asarray(balls_np, dtype=np.float32),
                    'cyl': np.asarray(cyl_np, dtype=np.float32),
                    'cyl_h': np.asarray(cyl_h_np, dtype=np.float32),
                    'sampled_positions': np.asarray(p[:, b].detach().cpu().numpy(), dtype=np.float32),
                    'sampled_dist': np.asarray(dist_obj[:, b].detach().cpu().numpy(), dtype=np.float32),
                    'goal': np.asarray(p_target[b].detach().cpu().numpy(), dtype=np.float32),
                    'resolution': planner.resolution,
                    'margin': planner.margin,
                    'z_min': planner.z_min,
                    'z_max': planner.z_max,
                    'max_speed': max_speed,
                    'max_accel': max_accel,
                    'max_decel': max_decel,
                    'lookahead_dist': lookahead_dist,
                    'invalid_dist_threshold': invalid_dist_threshold,
                })

            results = pool.map(_plan_sample_points_worker, payloads)
            used_parallel = True

            for b, out in enumerate(results):
                ref_direction[:, b] = torch.from_numpy(out['ref_direction']).to(device=device, dtype=p.dtype)
                ref_speed[:, b] = torch.from_numpy(out['ref_speed']).to(device=device, dtype=p.dtype)
                ref_acceleration[:, b] = torch.from_numpy(out['ref_acceleration']).to(device=device, dtype=p.dtype)
                lateral_error[:, b] = torch.from_numpy(out['lateral_error']).to(device=device, dtype=p.dtype)
                valid_mask[:, b] = torch.from_numpy(out['valid_mask']).to(device=device)

                sample_plan_total += int(out['plan_total'])
                sample_plan_success += int(out['plan_success'])
                curv_sum += float(out['curv_sum'])
                prog_sum += float(out['prog_sum'])
                lat_sum += float(out['lat_sum'])
                lat_max = max(lat_max, float(out['lat_max']))
                metric_count += int(out['metric_count'])
        except Exception:
            used_parallel = False

    if not used_parallel:
        # 串行回退路径：每个 batch 仅构建一次占用栅格，随后复用
        occupancy_cache = {}
        for b in range(B):
            planner.build_occupancy_grid(env, batch_idx=b)
            occupancy_cache[b] = (
                planner.occupancy_grid,
                planner.grid_origin.copy(),
                planner.grid_shape,
            )

        for b in range(B):
            planner.occupancy_grid, planner.grid_origin, planner.grid_shape = occupancy_cache[b]
            goal_np = p_target[b].detach().cpu().numpy()

            for s in range(S):
                pos_np = p[s, b].detach().cpu().numpy()
                dist = dist_obj[s, b].item()

                if dist < invalid_dist_threshold:
                    continue

                sample_plan_total += 1
                path = planner.plan_astar(pos_np, goal_np)
                if path is None:
                    continue
                path = planner.smooth_path(path, window_size=5)

                ref_info = planner.extract_reference_from_path(
                    pos_np, path,
                    max_speed=max_speed,
                    max_accel=max_accel,
                    max_decel=max_decel,
                    lookahead_dist=lookahead_dist,
                )

                ref_direction[s, b] = torch.tensor(ref_info['direction'], device=device, dtype=p.dtype)
                ref_speed[s, b] = ref_info['speed']
                ref_acceleration[s, b] = torch.tensor(ref_info['acceleration'], device=device, dtype=p.dtype)
                lateral_error[s, b] = ref_info.get('lateral_error', 0.0)
                valid_mask[s, b] = bool(ref_info['valid'])
                if bool(ref_info['valid']):
                    sample_plan_success += 1

                curv_sum += float(ref_info.get('curvature', 0.0))
                prog_sum += float(ref_info.get('path_progress', 0.0))
                lat = float(ref_info.get('lateral_error', 0.0))
                lat_sum += lat
                lat_max = max(lat_max, lat)
                metric_count += 1

    plan_total = max(1, int(sample_plan_total))
    plan_success = int(sample_plan_success)
    planner_info = {
        'avg_curvature': (curv_sum / metric_count) if metric_count > 0 else 0.0,
        'avg_path_progress': (prog_sum / metric_count) if metric_count > 0 else 0.0,
        'avg_lateral_error': (lat_sum / metric_count) if metric_count > 0 else 0.0,
        'max_lateral_error': lat_max if metric_count > 0 else 0.0,
        'planner_success_ratio': float(plan_success) / float(plan_total),
        'reference_valid_ratio': float(valid_mask.float().mean().item()),
        'sample_plan_total': int(sample_plan_total),
        'sample_plan_success': int(sample_plan_success),
    }

    if squeeze_output:
        ref_direction = ref_direction.squeeze(0)
        ref_speed = ref_speed.squeeze(0)
        ref_acceleration = ref_acceleration.squeeze(0)
        lateral_error = lateral_error.squeeze(0)
        valid_mask = valid_mask.squeeze(0)

    return ref_direction, ref_speed, ref_acceleration, lateral_error, valid_mask, planner_info


def compute_escape_penalty(v, vec_to_pt, dist_obj, collision_mask):
    """
    对已碰撞点，计算逃逸惩罚而非规划引导
    鼓励速度方向与逃逸方向（远离障碍物内部）一致

    Args:
        v: 速度 [S, B, 3]
        vec_to_pt: 指向最近障碍物表面的向量 [S, B, 3]
        dist_obj: 到障碍物的距离 [S, B]
        collision_mask: 碰撞掩码 [S, B]

    Returns:
        escape_loss: 逃逸方向一致性损失 [S, B]
        depth_penalty: 碰撞深度惩罚 [S, B]
    """
    # 逃逸方向：vec_to_pt 指向障碍物表面最近点
    # 当在障碍物内时，应该朝 vec_to_pt 的方向移动以逃出
    escape_dir = safe_normalize(vec_to_pt, dim=-1)  # [S, B, 3]
    v_dir = safe_normalize(v, dim=-1)  # [S, B, 3]

    # 逃逸方向一致性损失：1 - cos(v, escape_dir)
    # 当速度方向与逃逸方向一致时，损失为0
    escape_alignment = 1.0 - (v_dir * escape_dir).sum(dim=-1)  # [S, B]

    # 只对碰撞点计算
    escape_loss = escape_alignment * collision_mask.float()  # [S, B]

    # 额外惩罚：碰撞深度越大，惩罚越重
    depth_penalty = F.relu(-dist_obj).pow(2) * collision_mask.float()  # [S, B]

    return escape_loss, depth_penalty


def sample_guidance_points(p_history, v_history, dist_obj, sample_count, strategy='random'):
    """
    智能采样轨迹上的关键点

    Args:
        p_history: 位置历史 [T, B, 3]
        v_history: 速度历史 [T, B, 3]
        dist_obj: 到障碍物的距离 [T, B]
        sample_count: 采样点数
        strategy: 采样策略 ('random', 'uniform', 'adaptive', 'critical')

    Returns:
        indices: 采样点索引 tensor [S, B]（每个 batch 独立采样）
    """
    T, B = p_history.shape[:2]
    device = p_history.device

    # 确保采样数不超过总时间步数
    sample_count = min(sample_count, T)

    if strategy == 'random':
        # 随机采样时间步（不放回），每个 batch 独立采样并按时间排序
        if sample_count >= T:
            base = torch.arange(T, device=device, dtype=torch.long)
            indices = base[:, None].expand(T, B).clone()
        else:
            cols = []
            for b in range(B):
                idx_b = torch.randperm(T, device=device)[:sample_count].sort().values
                cols.append(idx_b)
            indices = torch.stack(cols, dim=1)

    elif strategy == 'uniform':
        # 均匀采样（时间步一致，但按 batch 维度展开）
        base = torch.linspace(0, T - 1, sample_count, device=device).long()
        indices = base[:, None].expand(sample_count, B).clone()

    elif strategy == 'adaptive':
        # 自适应采样：优先采样危险点和变化大的点（每个 batch 独立）
        with torch.no_grad():
            # 危险度：dist_obj 越小越危险
            danger_score = F.softplus(-dist_obj * 5.0)  # [T, B]

            # 速度变化度（曲率指标）
            v_diff = (v_history[1:] - v_history[:-1]).norm(dim=-1)  # [T-1, B]
            v_diff = F.pad(v_diff, (0, 0, 0, 1), value=0.0)  # [T, B]

            # 综合分数
            importance = danger_score + v_diff  # [T, B]

            # 每个 batch 选择最重要的点
            k = min(sample_count, T)
            cols = []
            for b in range(B):
                _, top_indices = importance[:, b].topk(k)
                cols.append(top_indices.sort().values)
            indices = torch.stack(cols, dim=1)

    elif strategy == 'critical':
        # 只采样关键时刻：轨迹的起点、终点、最危险点（每个 batch 独立）
        with torch.no_grad():
            danger = F.softplus(-dist_obj * 5.0)  # [T, B]
            cols = []
            for b in range(B):
                critical_indices = {0, T - 1}
                remaining_count = sample_count - len(critical_indices)
                if remaining_count > 0 and T > 2:
                    danger_mid = danger[1:-1, b]
                    k = min(remaining_count, int(danger_mid.numel()))
                    if k > 0:
                        _, top_danger = danger_mid.topk(k)
                        for idx in (top_danger + 1).tolist():
                            critical_indices.add(int(idx))

                idx_b = torch.tensor(sorted(critical_indices), device=device, dtype=torch.long)
                if idx_b.numel() < sample_count:
                    pad = idx_b[-1].repeat(sample_count - idx_b.numel())
                    idx_b = torch.cat([idx_b, pad], dim=0)
                elif idx_b.numel() > sample_count:
                    idx_b = idx_b[:sample_count]
                cols.append(idx_b)

            indices = torch.stack(cols, dim=1)

    else:
        # 默认均匀采样
        base = torch.linspace(0, T - 1, sample_count, device=device).long()
        indices = base[:, None].expand(sample_count, B).clone()

    return indices


def compute_global_guidance_meta_loss(env, p_history, v_history, p_target, vec_to_pt, dist_obj,
                                       a_history=None,
                                       sample_count=10, strategy='random',
                                       max_speed=5.0, max_accel=5.0, max_decel=6.0,
                                       dir_weight=0.5, speed_weight=0.3, lateral_weight=0.3,
                                       escape_weight=1.0, collision_threshold=-0.05,
                                       accel_weight=0.2, speed_diff_weight=0.2,
                                       recovery_speed_weight=0.15):
    """
    全局规划器引导的元损失，使用 A* 算法规划的全局路径作为参考

    基于梯形速度剖面和动力学约束计算损失：
    - 方向一致性：速度方向与规划方向的夹角
    - 速度偏差：实际速度与规划参考速度的差异（双向惩罚）
    - 横向偏差：到规划路径的真正几何距离
    - 加速度偏差：实际加速度与规划参考加速度的差异

    Args:
        p_history: 位置历史 [T, B, 3]
        v_history: 速度历史 [T, B, 3]
        p_target: 目标位置 [B, 3]
        vec_to_pt: 指向最近障碍物的向量 [T, B, 3]
        dist_obj: 到障碍物的距离 [T, B]
        a_history: 可选的环境真实加速度历史 [T, B, 3]，提供时优先用于加速度监督
        sample_count: 采样点数
        strategy: 采样策略
        max_speed: 最大速度 (m/s)
        max_accel: 最大加速度 (m/s^2)
        max_decel: 最大减速度 (m/s^2)
        dir_weight: 方向一致性损失权重
        speed_weight: 速度偏差惩罚权重
        lateral_weight: 横向偏差惩罚权重（到路径的几何距离）
        escape_weight: 逃逸惩罚权重
        collision_threshold: 碰撞判定阈值
        accel_weight: 加速度偏差权重
        speed_diff_weight: 速度差（超速/低速）惩罚权重

    Returns:
        guidance_loss: 标量损失
        loss_components: 各分项损失的字典
    """
    T, B, _ = p_history.shape

    # 1. 采样关键点
    sample_indices = sample_guidance_points(
        p_history, v_history, dist_obj, sample_count, strategy
    )
    S = sample_indices.shape[0]

    # 2. 提取采样点的状态
    b_idx = torch.arange(B, device=p_history.device).unsqueeze(0).expand(S, B)
    p_sampled = p_history[sample_indices, b_idx]      # [S, B, 3]
    v_sampled = v_history[sample_indices, b_idx]      # [S, B, 3]
    vec_sampled = vec_to_pt[sample_indices, b_idx]    # [S, B, 3]
    dist_sampled = dist_obj[sample_indices, b_idx]    # [S, B]
    a_sampled = None
    if a_history is not None:
        a_sampled = a_history[sample_indices, b_idx]  # [S, B, 3]

    # 3. 使用全局规划器计算参考（A* 规划 + 梯形速度剖面）
    ref_dir, ref_speed, ref_accel, lateral_error, valid_mask, planner_info = compute_guidance_reference_from_planner(
        env, p_sampled, v_sampled, p_target, dist_sampled, global_planner,
        max_speed=max_speed, max_accel=max_accel, max_decel=max_decel,
        lookahead_dist=1.0,
        invalid_dist_threshold=collision_threshold,
    )

    # 4. 计算各项损失
    v_dir = safe_normalize(v_sampled, dim=-1)  # [S, B, 3]
    v_speed = v_sampled.norm(dim=-1)  # [S, B]

    # 4.1 方向一致性损失：1 - cos(v_dir, ref_dir)
    loss_dir_align = 1.0 - (v_dir * ref_dir).sum(dim=-1)  # [S, B]

    # 4.2 速度偏差：双向惩罚（超速和低速都惩罚）
    # 超速惩罚更重，低速惩罚较轻
    loss_overspeed = F.relu(v_speed - ref_speed)  # [S, B]
    loss_underspeed = F.relu(ref_speed - v_speed) * 0.3  # 低速惩罚较轻
    loss_speed_diff = loss_overspeed + loss_underspeed  # [S, B]

    # 4.3 横向偏差：到规划路径的真正几何距离
    # 使用平滑的 L1 损失，对小偏差不过度惩罚
    loss_lateral = F.smooth_l1_loss(lateral_error, torch.zeros_like(lateral_error), reduction='none')  # [S, B]

    # 4.4 加速度偏差惩罚
    if S > 1:
        # 实际加速度：优先使用环境提供的 a_history，缺失时回退到速度差分估算
        if a_sampled is not None:
            v_diff = a_sampled
        else:
            v_diff = torch.zeros_like(v_sampled)
            for i in range(S - 1):
                dt_approx = (sample_indices[i + 1] - sample_indices[i]).to(v_sampled.dtype) / 15.0  # [B]
                step_acc = (v_sampled[i + 1] - v_sampled[i]) / (dt_approx[:, None] + 1e-6)
                valid_dt = dt_approx > 0
                if valid_dt.any():
                    v_diff[i, valid_dt] = step_acc[valid_dt]
            v_diff[-1] = v_diff[-2] if S > 1 else torch.zeros_like(v_diff[-1])

        # 加速度偏差：实际加速度与参考加速度的差异
        # ref_accel 是基于速度剖面计算的参考加速度（通常是减速）
        accel_error = (v_diff - ref_accel).norm(dim=-1)  # [S, B]

        # 只惩罚明显的加速度偏差（阈值 0.5 m/s^2）
        loss_accel_mismatch = F.relu(accel_error - 0.5)  # [S, B]

        # 额外检查：需要减速时是否真的在减速
        ref_accel_mag = ref_accel.norm(dim=-1)  # [S, B]
        ref_accel_dir = safe_normalize(ref_accel, dim=-1)
        actual_accel_along_ref = (v_diff * ref_accel_dir).sum(dim=-1)  # [S, B]
        # 如果规划器要求减速（ref_accel 有显著幅度且为负）但实际在加速
        need_decel = ref_accel_mag > 0.5
        not_deceling = actual_accel_along_ref < -0.5  # 实际加速度与减速方向相反
        loss_decel_violation = need_decel.float() * not_deceling.float() * ref_accel_mag
        loss_accel_mismatch = loss_accel_mismatch + loss_decel_violation
    else:
        loss_accel_mismatch = torch.zeros_like(loss_dir_align)

    # 5. 对不可规划点/碰撞点的恢复处理
    collision_mask = dist_sampled < collision_threshold  # [S, B]
    invalid_mask = (~valid_mask) & (~collision_mask)
    recovery_mask = collision_mask | invalid_mask
    valid_guidance_mask = valid_mask & (~collision_mask)

    loss_escape, loss_depth = compute_escape_penalty(
        v_sampled, vec_sampled, dist_sampled, recovery_mask
    )
    loss_recovery_speed = v_speed * invalid_mask.float()

    # 6. 组合损失
    # 有效点（非碰撞）用规划引导
    loss_speed_profile = speed_weight * loss_overspeed + speed_diff_weight * loss_underspeed
    guidance_for_valid = (
        dir_weight * loss_dir_align +
        loss_speed_profile +
        lateral_weight * loss_lateral +
        accel_weight * loss_accel_mismatch
    )  # [S, B]

    # 碰撞点和不可规划点用恢复惩罚
    guidance_for_recovery = (
        escape_weight * (loss_escape + loss_depth)
        + recovery_speed_weight * loss_recovery_speed
    )  # [S, B]

    # 根据点状态选择损失
    guidance_loss_per_point = torch.where(
        valid_guidance_mask,
        guidance_for_valid,
        guidance_for_recovery,
    )  # [S, B]

    # 总损失
    guidance_loss = guidance_loss_per_point.mean()

    # 返回各分项用于日志
    loss_components = {
        'dir_align': loss_dir_align.mean(),
        'speed_diff': loss_speed_diff.mean(),
        'overspeed': loss_overspeed.mean(),
        'underspeed': loss_underspeed.mean(),
        'lateral_error': loss_lateral.mean(),
        'accel_mismatch': loss_accel_mismatch.mean(),
        'escape': loss_escape.mean(),
        'depth': loss_depth.mean(),
        'recovery_speed': loss_recovery_speed.mean(),
        'valid_ratio': valid_mask.float().mean(),
        'invalid_ratio': invalid_mask.float().mean(),
        'collision_ratio': collision_mask.float().mean(),
        'sample_count': S,
        'avg_curvature': planner_info.get('avg_curvature', 0.0),
        'avg_path_progress': planner_info.get('avg_path_progress', 0.0),
        'avg_lateral_error': planner_info.get('avg_lateral_error', 0.0),
        'max_lateral_error': planner_info.get('max_lateral_error', 0.0),
        'planner_success_ratio': planner_info.get('planner_success_ratio', 0.0),
        'avg_ref_speed': ref_speed.mean().item(),
    }

    return guidance_loss, loss_components



def draw_sphere(ax, cx, cy, cz, r, color='royalblue', alpha=0.18, res=12):
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.1, edgecolor='k', antialiased=True, shade=True)


def draw_cylinder_z(ax, cx, cy, r, z0, z1, color='teal', alpha=0.14, res_theta=18, res_h=2):
    theta = np.linspace(0.0, 2.0 * np.pi, res_theta)
    z = np.linspace(z0, z1, res_h)
    th_grid, z_grid = np.meshgrid(theta, z)
    x = cx + r * np.cos(th_grid)
    y = cy + r * np.sin(th_grid)
    ax.plot_surface(x, y, z_grid, color=color, alpha=alpha, linewidth=0.1, edgecolor='k', antialiased=True, shade=True)


def draw_cylinder_y(ax, cx, zc, r, y0, y1, color='darkorange', alpha=0.16, res_theta=18, res_h=2):
    theta = np.linspace(0.0, 2.0 * np.pi, res_theta)
    y = np.linspace(y0, y1, res_h)
    th_grid, y_grid = np.meshgrid(theta, y)
    x = cx + r * np.cos(th_grid)
    z = zc + r * np.sin(th_grid)
    ax.plot_surface(x, y_grid, z, color=color, alpha=alpha, linewidth=0.1, edgecolor='k', antialiased=True, shade=True)


def _plotly_add_cuboid(fig, cx, cy, cz, hx, hy, hz, color='lightgray', opacity=0.65):
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                            color=color, opacity=opacity, flatshading=True,
                            hoverinfo='skip', showscale=False))


def _plotly_add_sphere(fig, cx, cy, cz, r, color='royalblue', opacity=0.75, res=16):
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    c = np.zeros_like(x)
    fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=c,
                             colorscale=[[0, color], [1, color]], showscale=False,
                             opacity=opacity, hoverinfo='skip'))


def _plotly_add_cylinder_z(fig, cx, cy, r, z0, z1, color='teal', opacity=0.72, res_theta=22):
    th = np.linspace(0.0, 2.0 * np.pi, res_theta)
    z = np.array([z0, z1])
    th_grid, z_grid = np.meshgrid(th, z)
    x = cx + r * np.cos(th_grid)
    y = cy + r * np.sin(th_grid)
    c = np.zeros_like(x)
    fig.add_trace(go.Surface(x=x, y=y, z=z_grid, surfacecolor=c,
                             colorscale=[[0, color], [1, color]], showscale=False,
                             opacity=opacity, hoverinfo='skip'))


def _plotly_add_cylinder_y(fig, cx, zc, r, y0, y1, color='darkorange', opacity=0.72, res_theta=22):
    th = np.linspace(0.0, 2.0 * np.pi, res_theta)
    y = np.array([y0, y1])
    th_grid, y_grid = np.meshgrid(th, y)
    x = cx + r * np.cos(th_grid)
    z = zc + r * np.sin(th_grid)
    c = np.zeros_like(x)
    fig.add_trace(go.Surface(x=x, y=y_grid, z=z, surfacecolor=c,
                             colorscale=[[0, color], [1, color]], showscale=False,
                             opacity=opacity, hoverinfo='skip'))


def _is_ceiling_or_side_boundary_voxel(box_xyz_half, env):
    """Hide enclosure voxels (ceiling + 4 side walls) while keeping floor visible."""
    cx, cy, cz, hx, hy, hz = [float(v) for v in box_xyz_half]
    if not all(hasattr(env, key) for key in ('map_x_max', 'map_y_min', 'map_y_max', 'map_z_max', 'boundary_half')):
        return False

    map_x_max = float(env.map_x_max)
    map_y_min = float(env.map_y_min)
    map_y_max = float(env.map_y_max)
    map_z_max = float(env.map_z_max)
    boundary_half = float(env.boundary_half)
    map_y_span = max(1e-6, map_y_max - map_y_min)

    tol = max(0.08, boundary_half * 1.6)

    side_x = (
        (abs(cx - 0.0) <= tol or abs(cx - map_x_max) <= tol)
        and abs(hx - boundary_half) <= tol
        and hy >= 0.45 * map_y_span
    )
    side_y = (
        (abs(cy - map_y_min) <= tol or abs(cy - map_y_max) <= tol)
        and abs(hy - boundary_half) <= tol
        and hx >= 0.45 * map_x_max
    )
    ceiling = (
        abs(cz - map_z_max) <= tol
        and abs(hz - boundary_half) <= tol
        and hx >= 0.45 * map_x_max
        and hy >= 0.45 * map_y_span
    )
    return bool(side_x or side_y or ceiling)


def save_interactive_3d_html(html_path, env, p_cpu, v_cpu, R_cpu=None, idx=0, axis_len=0.3, axis_step=5, astar_path=None, astar_paths_sampled=None):
    """保存交互式3D轨迹HTML，带有无人机姿态坐标系和A*规划路径

    Args:
        R_cpu: 姿态矩阵 [T, 3, 3]，如果提供则绘制坐标系
        axis_len: 坐标轴长度(米)
        axis_step: 每隔多少个时间步绘制一次坐标系
        astar_path: A*规划路径 [N, 3] numpy数组，从起点到终点的路径
        astar_paths_sampled: 采样点的A*路径列表 [(sample_idx, path), ...]，每个path是[N, 3] numpy数组
    """
    if go is None:
        return False

    traj_xyz = p_cpu.numpy()
    speed_cpu = v_cpu.norm(dim=-1).numpy()
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=traj_xyz[:, 0], y=traj_xyz[:, 1], z=traj_xyz[:, 2],
        mode='lines+markers',
        marker=dict(
            size=3,
            color=speed_cpu,
            colorscale='Turbo',
            cmin=0.0,
            cmax=15.0,
            colorbar=dict(title='Speed (m/s)')
        ),
        line=dict(color='limegreen', width=5),
        name='Trajectory'
    ))

    # 绘制从起点到终点的 A* 规划路径 (橙色虚线)
    if astar_path is not None and len(astar_path) > 1:
        fig.add_trace(go.Scatter3d(
            x=astar_path[:, 0], y=astar_path[:, 1], z=astar_path[:, 2],
            mode='lines+markers',
            marker=dict(size=2, color='gold', symbol='diamond'),
            line=dict(color='orange', width=4, dash='dash'),
            name='A* Path (Start→Goal)'
        ))

    # 绘制所有采样点的 A* 路径 (淡紫色细线，按时间步渐变)
    if astar_paths_sampled is not None and len(astar_paths_sampled) > 0:
        # 使用颜色渐变表示时间顺序
        num_paths = len(astar_paths_sampled)
        for path_idx, (sample_t, path) in enumerate(astar_paths_sampled):
            if path is None or len(path) < 2:
                continue
            # 颜色从浅蓝到深紫渐变
            color_ratio = path_idx / max(num_paths - 1, 1)
            r = int(100 + 100 * color_ratio)
            g = int(150 * (1 - color_ratio))
            b = int(200 + 55 * color_ratio)
            color = f'rgb({r},{g},{b})'

            fig.add_trace(go.Scatter3d(
                x=path[:, 0], y=path[:, 1], z=path[:, 2],
                mode='lines',
                line=dict(color=color, width=2),
                opacity=0.6,
                showlegend=(path_idx == 0),
                name='A* Sampled Paths' if path_idx == 0 else None,
                legendgroup='astar_sampled',
                hovertemplate=f't={sample_t}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>'
            ))

    # 绘制无人机姿态坐标系 (X-红, Y-绿, Z-蓝)
    if R_cpu is not None:
        R_np = R_cpu.numpy()  # [T, 3, 3]
        T = len(traj_xyz)
        # 每隔 axis_step 个点绘制一次坐标系
        for t in range(0, T, axis_step):
            pos = traj_xyz[t]
            R = R_np[t]  # [3, 3], 列向量为机体坐标系的X,Y,Z轴
            # X轴 (红色) - 机头方向
            x_axis = R[:, 0] * axis_len
            fig.add_trace(go.Scatter3d(
                x=[pos[0], pos[0] + x_axis[0]],
                y=[pos[1], pos[1] + x_axis[1]],
                z=[pos[2], pos[2] + x_axis[2]],
                mode='lines',
                line=dict(color='red', width=4),
                showlegend=(t == 0),
                name='X-axis (Forward)' if t == 0 else None,
                legendgroup='x_axis'
            ))
            # Y轴 (绿色) - 左侧方向
            y_axis = R[:, 1] * axis_len
            fig.add_trace(go.Scatter3d(
                x=[pos[0], pos[0] + y_axis[0]],
                y=[pos[1], pos[1] + y_axis[1]],
                z=[pos[2], pos[2] + y_axis[2]],
                mode='lines',
                line=dict(color='green', width=4),
                showlegend=(t == 0),
                name='Y-axis (Left)' if t == 0 else None,
                legendgroup='y_axis'
            ))
            # Z轴 (蓝色) - 上方向 (推力方向)
            z_axis = R[:, 2] * axis_len
            fig.add_trace(go.Scatter3d(
                x=[pos[0], pos[0] + z_axis[0]],
                y=[pos[1], pos[1] + z_axis[1]],
                z=[pos[2], pos[2] + z_axis[2]],
                mode='lines',
                line=dict(color='blue', width=4),
                showlegend=(t == 0),
                name='Z-axis (Up/Thrust)' if t == 0 else None,
                legendgroup='z_axis'
            ))

    x_vals = [traj_xyz[:, 0]]
    y_vals = [traj_xyz[:, 1]]
    z_vals = [traj_xyz[:, 2]]

    if hasattr(env, 'voxels') and env.voxels.numel() > 0:
        vox = env.voxels[idx].detach().cpu().numpy()
        vox = vox[(vox[:, 3:6] < 20).all(axis=1)]
        for box in vox[:180]:
            if _is_ceiling_or_side_boundary_voxel(box, env):
                continue
            cx, cy, cz, hx, hy, hz = box.tolist()
            _plotly_add_cuboid(fig, cx, cy, cz, hx, hy, hz, color='lightgray', opacity=0.7)
            x_vals.extend([[cx - hx], [cx + hx]])
            y_vals.extend([[cy - hy], [cy + hy]])
            z_vals.extend([[cz - hz], [cz + hz]])

    if hasattr(env, 'balls') and env.balls.numel() > 0:
        balls = env.balls[idx].detach().cpu().numpy()
        for bx, by, bz, br in balls[:80]:
            _plotly_add_sphere(fig, float(bx), float(by), float(bz), float(br), color='royalblue', opacity=0.78, res=14)
            x_vals.extend([[bx - br], [bx + br]])
            y_vals.extend([[by - br], [by + br]])
            z_vals.extend([[bz - br], [bz + br]])

    z0_vis, z1_vis = -0.2, 2.2
    if hasattr(env, 'cyl') and env.cyl.numel() > 0:
        cyl = env.cyl[idx].detach().cpu().numpy()
        for cx, cy, cr in cyl[:100]:
            _plotly_add_cylinder_z(fig, float(cx), float(cy), float(cr), z0_vis, z1_vis, color='teal', opacity=0.76)
            x_vals.extend([[cx - cr], [cx + cr]])
            y_vals.extend([[cy - cr], [cy + cr]])

    y0_vis, y1_vis = -9.5, 9.5
    if hasattr(env, 'cyl_h') and env.cyl_h.numel() > 0:
        cyl_h = env.cyl_h[idx].detach().cpu().numpy()
        for cx, cz, cr in cyl_h[:100]:
            _plotly_add_cylinder_y(fig, float(cx), float(cz), float(cr), y0_vis, y1_vis, color='darkorange', opacity=0.76)
            x_vals.extend([[cx - cr], [cx + cr]])
            z_vals.extend([[cz - cr], [cz + cr]])

    fig.add_trace(go.Scatter3d(x=[traj_xyz[0, 0]], y=[traj_xyz[0, 1]], z=[traj_xyz[0, 2]], mode='markers',
                               marker=dict(size=6, color='green', symbol='circle'), name='Start'))
    fig.add_trace(go.Scatter3d(x=[traj_xyz[-1, 0]], y=[traj_xyz[-1, 1]], z=[traj_xyz[-1, 2]], mode='markers',
                               marker=dict(size=6, color='black', symbol='x'), name='End'))
    if hasattr(env, 'p_target'):
        tgt = env.p_target[idx].detach().cpu().numpy()
        fig.add_trace(go.Scatter3d(x=[tgt[0]], y=[tgt[1]], z=[tgt[2]], mode='markers',
                                   marker=dict(size=8, color='red', symbol='diamond'), name='Goal'))
        x_vals.append([tgt[0]])
        y_vals.append([tgt[1]])
        z_vals.append([tgt[2]])

    # 将 A* 路径坐标也纳入范围计算
    if astar_path is not None and len(astar_path) > 0:
        x_vals.append(astar_path[:, 0])
        y_vals.append(astar_path[:, 1])
        z_vals.append(astar_path[:, 2])

    # 将采样点 A* 路径坐标也纳入范围计算
    if astar_paths_sampled is not None:
        for _, path in astar_paths_sampled:
            if path is not None and len(path) > 0:
                x_vals.append(path[:, 0])
                y_vals.append(path[:, 1])
                z_vals.append(path[:, 2])

    x_all = np.concatenate([np.asarray(v) for v in x_vals])
    y_all = np.concatenate([np.asarray(v) for v in y_vals])
    z_all = np.concatenate([np.asarray(v) for v in z_vals])

    fig.update_layout(
        title='Interactive 3D Trajectory & Obstacles',
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            xaxis=dict(range=[float(x_all.min()) - 0.5, float(x_all.max()) + 0.5]),
            yaxis=dict(range=[float(y_all.min()) - 0.5, float(y_all.max()) + 0.5]),
            zaxis=dict(range=[float(z_all.min()) - 0.2, float(z_all.max()) + 0.2]),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6, y=1.4, z=1.2))
        ),
        template='plotly_white',
        showlegend=True,
        margin=dict(l=5, r=5, t=40, b=5)
    )
    fig.write_html(html_path, include_plotlyjs='cdn')
    return True



def compute_overlap_loss_per_step(p_history, sigma=0.5, time_window=10):
    """
    Step-wise 重叠损失计算
    返回: [Batch, Time] (注意: 调用处需要permute)
    """
    p_history = p_history.permute(1, 0, 2) # [B, T, 3]
    n_batch, n_points, n_dims = p_history.shape

    if n_points < time_window + 1:
        return torch.zeros((n_batch, n_points), device=p_history.device)

    # 计算距离矩阵
    dist_matrix = torch.cdist(p_history, p_history, p=2)
    overlap_energy = torch.exp(- (dist_matrix ** 2) / (2 * sigma ** 2))

    indices = torch.arange(n_points, device=p_history.device)
    time_diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    mask = (time_diff > time_window).float()

    # 计算每个时间步的能量总和
    energy_sum = (overlap_energy * mask.unsqueeze(0)).sum(dim=2)
    mask_sum = mask.sum(dim=1).unsqueeze(0) + 1e-6

    # 返回 [Batch, Time]
    loss_per_step = energy_sum / mask_sum
    return loss_per_step


def compute_stuck_loss(p_history, collision_depth, stuck_window=15, displacement_threshold=0.3):
    """
    计算卡住惩罚损失

    检测两种卡住状态：
    1. 局部窗口内位移过小 (stuck due to obstacles or getting lost)
    2. 持续碰撞状态 (sustained contact with obstacles)

    Args:
        p_history: [T, B, 3] 位置历史
        collision_depth: [T, B] 碰撞深度 (正值表示穿入)
        stuck_window: 检测窗口大小 (步数)
        displacement_threshold: 最小期望位移 (m)

    Returns:
        loss_stuck: [T, B] 卡住惩罚
        loss_collision_duration: [T, B] 碰撞持续时间惩罚
        stuck_ratio: 标量，卡住比例（用于监控）
    """
    T, B, _ = p_history.shape
    device = p_history.device

    # 1. 局部位移惩罚：检测低速移动/原地踏步
    loss_stuck = torch.zeros((T, B), device=device)
    if T > stuck_window:
        for t in range(stuck_window, T):
            # 计算过去 stuck_window 步的累计位移
            window_start = t - stuck_window
            displacement = (p_history[t] - p_history[window_start]).norm(dim=-1)  # [B]
            # 惩罚位移小于阈值的情况
            # 使用平滑惩罚：softplus((threshold - displacement) * scale)
            loss_stuck[t] = F.softplus((displacement_threshold - displacement) * 10.0)

    # 2. 碰撞持续时间惩罚：检测持续接触
    # 使用滑动窗口累计碰撞事件
    in_collision = (collision_depth > 0).float()  # [T, B]
    loss_collision_duration = torch.zeros_like(in_collision)

    # 累计连续碰撞步数
    collision_streak = torch.zeros((B,), device=device)
    for t in range(T):
        collision_streak = collision_streak * in_collision[t] + in_collision[t]
        # 随着连续碰撞步数增加，惩罚递增
        loss_collision_duration[t] = collision_streak * in_collision[t]

    # 计算卡住比例（监控用）
    with torch.no_grad():
        stuck_mask = loss_stuck > 0.5
        stuck_ratio = stuck_mask.float().mean()

    return loss_stuck, loss_collision_duration, stuck_ratio


def unrolled_meta_rollout(env, worknet, fast_params, state_normalizer, args, B, device):
    """
    Validation rollout with virtually-updated worker params (via functional_call).
    Computes and returns meta_loss (position + collision + height) plus components.
    Control effort is tracked for monitoring but is not included in optimization target.
    LGN is NOT needed here; this meta loss is task-performance based.
    Reuses the same maze layout for consistent LGN signal.
    """
    # 保持同一张迷宫布局, 仅重置无人机状态用于验证rollout
    env.reset_drone_only()

    p_list, v_list, a_list, vec_list = [], [], [], []
    act_buf = [env.act.detach()] * 2
    v_preds_val = []  # [新增] 收集速度预测值用于计算预测误差
    h_val = None

    for t in range(args.lgn_timesteps):
        ctl_dt = 1.0 / 15.0
        depth, flow = env.render(ctl_dt)
        depth = sanitize_tensor(depth, nan=24.0, posinf=24.0, neginf=0.3)

        p_list.append(env.p)
        v_list.append(env.v)
        a_list.append(env.a)
        vec_list.append(env.find_vec_to_nearest_pt())

        target_v_raw = env.p_target - env.p.detach()
        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        max_speed = torch.as_tensor(env.max_speed, device=target_v_norm.device,
                                    dtype=target_v_norm.dtype)
        target_v = (target_v_raw / (target_v_norm + 1e-6)) * torch.minimum(target_v_norm, max_speed)

        R = env.R
        state_list = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.margin[:, None]]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state_list.insert(0, local_v)

        raw_state = sanitize_tensor(torch.cat(state_list, -1), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        state_t = state_normalizer(raw_state, update=False)  # 不更新统计量
        state_t = sanitize_tensor(state_t, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        x_pooled = F.max_pool2d((3 / depth.clamp(0.3, 24) - 0.6)[:, None], 4, 4)
        x_pooled = sanitize_tensor(x_pooled, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        # Worker forward with virtually-updated params
        act_out, _, h_val = functional_call(worknet, fast_params, (x_pooled, state_t, h_val))
        act_out = sanitize_tensor(act_out, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        a_pred, v_pred, *_ = (R @ act_out.reshape(B, 3, -1)).unbind(-1)
        v_preds_val.append(v_pred)  # [新增] 收集速度预测值
        real_act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        real_act = sanitize_tensor(real_act, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        act_buf.append(real_act)

        env.run(real_act, ctl_dt, target_v_raw)

        # Early termination when all drones reach their goals
        with torch.no_grad():
            _dist_to_goal_val = torch.norm(env.p - env.p_target, 2, -1)
            if t >= 10 and (_dist_to_goal_val < args.goal_radius).all():
                break

        # 周期性截断以限制显存
        if args.detach_interval > 0 and (t + 1) % args.detach_interval == 0:
            if h_val is not None:
                h_val = h_val.detach()

    # --- 计算 Meta Loss ---
    p_val = torch.stack(p_list)
    a_val = torch.stack(a_list)
    act_val = torch.stack(act_buf)
    vec_val = torch.stack(vec_list)
    if vec_val.dim() == 4:
        vec_val = vec_val.mean(1)

    dist_val = sanitize_tensor(vec_val.norm(2, -1) - env.margin, nan=0.0, posinf=10.0, neginf=-10.0)

    # [新增] 计算卡住损失和碰撞持续时间
    collision_depth_val = F.relu(-dist_val)
    loss_stuck_val, loss_collision_duration_val, stuck_ratio = compute_stuck_loss(
        p_val, collision_depth_val,
        stuck_window=args.stuck_window,
        displacement_threshold=args.stuck_displacement_threshold
    )

    m_pos  = torch.norm(p_val[-1] - env.p_target, 2, -1).mean()
    collision_depth_val = F.relu(-dist_val)
    m_coll_soft = F.softplus(-dist_val * 32.0).clamp(max=100.0).mean()
    m_coll_hard = collision_depth_val.pow(2).mean()
    m_coll_peak = collision_depth_val.max(dim=0).values
    m_coll_event = torch.sigmoid((m_coll_peak - args.meta_coll_event_threshold) * args.meta_coll_event_temp).mean()
    m_coll = (args.meta_coll_soft_weight * m_coll_soft
              + args.meta_coll_hard_weight * m_coll_hard
              + args.meta_coll_event_weight * m_coll_event)
    m_ctrl = act_val.norm(2, -1).sum()
    # 与 main_cuda.py 一致的平滑损失构造：jerk + snap
    m_jerk = act_val.diff(1, 0).mul(15.0).pow(2).sum(-1).mean()
    m_snap = (F.normalize(act_val - env.g_std, dim=-1)
              .diff(1, 0).diff(1, 0).mul(15.0 ** 2).pow(2).sum(-1).mean())
    # Meta rollout 的高度惩罚：仅约束不越界 + 抑制高度剧烈变化
    z_val = p_val[:, :, 2]
    z_floor = float(args.height_floor)
    z_ceiling = float(args.height_ceiling) if args.height_ceiling is not None else float(getattr(env, 'map_z_max', 5.0))
    z_sharpness = float(args.height_bound_sharpness)
    m_height_bound = (
        F.softplus((z_val - z_ceiling) * z_sharpness)
        + F.softplus((z_floor - z_val) * z_sharpness)
    )
    z_delta_val = z_val[1:] - z_val[:-1]
    m_height_smooth = F.smooth_l1_loss(z_delta_val, torch.zeros_like(z_delta_val), reduction='none')
    m_height_smooth = torch.cat([torch.zeros_like(z_val[:1]), m_height_smooth], dim=0)
    m_height = (m_height_bound + float(args.height_smooth_weight) * m_height_smooth).mean()
    
    # [新增] 速度预测误差损失
    v_preds_val_tensor = torch.stack(v_preds_val)  # [T, B, 3]
    v_val = torch.stack(v_list)  # [T, B, 3]
    m_v_pred = F.mse_loss(v_preds_val_tensor, v_val.detach())
    
    # [新增] 卡住惩罚和碰撞持续时间
    m_stuck = loss_stuck_val.mean()
    m_collision_duration = loss_collision_duration_val.mean()
    m_guidance, _ = compute_global_guidance_meta_loss(
        env, p_val, v_val, env.p_target, vec_val, dist_val,
        a_history=a_val,
        sample_count=args.guide_sample_count,
        strategy=args.guide_sample_strategy,
        max_speed=float(env.max_speed),
        max_accel=args.guide_max_accel,
        max_decel=args.guide_max_decel,
        dir_weight=args.guide_dir_weight,
        speed_weight=args.guide_speed_weight,
        lateral_weight=args.guide_lateral_weight,
        escape_weight=args.guide_escape_weight,
        collision_threshold=args.guide_collision_threshold,
        accel_weight=args.guide_accel_weight,
        speed_diff_weight=args.guide_speed_diff_weight,
        recovery_speed_weight=args.guide_recovery_speed_weight,
    )

    meta_val = sanitize_tensor(
        m_pos
        + m_coll
        + m_height * 2.0
        + args.meta_guidance_weight * m_guidance
        + args.meta_smooth_jerk_weight * m_jerk
        + args.meta_smooth_snap_weight * m_snap
        + args.meta_smooth_v_pred_weight * m_v_pred
        + args.stuck_loss_weight * m_stuck
        + args.collision_duration_weight * m_collision_duration,
        nan=1e3, posinf=1e3, neginf=1e3
    )
    return meta_val, m_pos, m_coll, m_ctrl

########## 7. 训练主循环 ##########

# 使用命令行参数重新初始化全局规划器
global_planner = GlobalPlanner(
    resolution=args.planner_resolution,
    margin=args.planner_margin,
    device=device
)
print(f"[GlobalPlanner] Initialized with resolution={args.planner_resolution}m, margin={args.planner_margin}m")

pbar = tqdm(range(args.num_iters), ncols=120)
B = args.batch_size
cycle_len = args.lgn_steps + args.worker_steps
maze_update_counter = 0

# [新增] 梯度爆炸保护状态
grad_explosion_count = 0
worker_explosion_consecutive = 0
lgn_explosion_consecutive = 0

state_normalizer.train()

for i in pbar:
    cycle_pos = i % cycle_len
    train_lgn_phase = cycle_pos < args.lgn_steps
    phase_str = f"LGN ({cycle_pos+1}/{args.lgn_steps})" if train_lgn_phase else f"Work ({cycle_pos-args.lgn_steps+1}/{args.worker_steps})"

    if maze_update_counter % args.maze_update_interval == 0:
        env.reset()          # full reset: new maze + new drones
    else:
        env.reset_drone_only()  # keep maze, reset drones only
    maze_update_counter += 1
    worknet.reset()

    p_history, v_history, a_history, target_v_history, vec_to_pt_history = [], [], [], [], []
    rpy_history = []
    R_history = []  # 记录姿态矩阵用于可视化
    real_act_history = []
    depth_history = []
    act_buffer = [env.act.detach()] * 2
    trajectory_lgn_weights = []
    v_preds = []  # [新增] 收集每个时间步的速度预测值用于计算预测误差

    h = None
    lgn_hx = None
    do_save_viz = is_save_trajectory_iter(i)
    rollout_steps = args.lgn_timesteps if train_lgn_phase else args.timesteps

    ###### A. Rollout ######
    for t in range(rollout_steps):
        ctl_dt = 1.0 / 15.0
        depth, flow = env.render(ctl_dt)
        depth = sanitize_tensor(depth, nan=24.0, posinf=24.0, neginf=0.3)

        if do_save_viz:
            depth_history.append(depth[0].detach().cpu().clone())

        p_history.append(env.p)
        v_history.append(env.v)
        a_history.append(env.a)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())
        rpy_history.append(rotation_matrix_to_rpy_deg(env.R))
        R_history.append(env.R.detach().clone())  # 保存姿态矩阵

        target_v_raw_curr = env.p_target - env.p.detach()
        target_v_norm = torch.norm(target_v_raw_curr, 2, -1, keepdim=True)
        max_speed = torch.as_tensor(env.max_speed, device=target_v_norm.device, dtype=target_v_norm.dtype)
        target_v = (target_v_raw_curr / (target_v_norm + 1e-6)) * torch.minimum(target_v_norm, max_speed)
        target_v_history.append(target_v)

        R = env.R
        state_list = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.margin[:, None]]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom: state_list.insert(0, local_v)
        
        raw_state_tensor = sanitize_tensor(torch.cat(state_list, -1), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        state_tensor = state_normalizer(raw_state_tensor, update=True)
        state_tensor = sanitize_tensor(state_tensor, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        x_pooled = F.max_pool2d((3 / depth.clamp(0.3, 24) - 0.6)[:, None], 4, 4)
        x_pooled = sanitize_tensor(x_pooled, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        # LGN Forward (直接使用可正可负权重，不做下限约束)
        current_weights, lgn_hx = lgn(x_pooled, state_tensor, lgn_hx)
        current_weights = sanitize_tensor(current_weights, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)

        # [诊断] 在第一个时间步检查 current_weights 的梯度链
        if t == 0 and i % 100 == 0:
            _cw_req = current_weights.requires_grad
            _cw_gfn = type(current_weights.grad_fn).__name__ if current_weights.grad_fn else "None"
            _lgn_hx_req = lgn_hx.requires_grad if lgn_hx is not None else "N/A"
            print(f"[DIAG iter={i} t=0] current_weights.requires_grad={_cw_req}, grad_fn={_cw_gfn}, lgn_hx.requires_grad={_lgn_hx_req}")

        trajectory_lgn_weights.append(current_weights)

        # Worker Forward
        act, _, h = worknet(x_pooled, state_tensor, h)
        act = sanitize_tensor(act, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)  # [新增] 收集速度预测值
        real_act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        real_act = sanitize_tensor(real_act, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        real_act_history.append(real_act)
        act_buffer.append(real_act)

        env.run(real_act, ctl_dt, target_v_raw_curr)

        # Early termination when all drones reach their goals
        with torch.no_grad():
            _dist_to_goal = torch.norm(env.p - env.p_target, 2, -1)
            if t >= 10 and (_dist_to_goal < args.goal_radius).all():
                break

        if args.detach_interval > 0 and (t + 1) % args.detach_interval == 0:
            if h is not None:
                h = h.detach()
            # LGN phase 时保留 lgn_hx 梯度，Worker phase 时截断
            if lgn_hx is not None and not train_lgn_phase:
                lgn_hx = lgn_hx.detach()

    ###### B. Loss Calculation (Step-wise) ######
    p_history = torch.stack(p_history)     # [T, B, 3]
    v_history = torch.stack(v_history)     # [T, B, 3]
    a_history = torch.stack(a_history)     # [T, B, 3]
    act_buffer = torch.stack(act_buffer)   # [T+2, B, 3]
    weights_seq = torch.stack(trajectory_lgn_weights) # [T, B, 4]

    # [诊断] 检查 weights_seq 的梯度属性
    if i % 100 == 0:
        _ws_req = weights_seq.requires_grad
        _ws_gfn = type(weights_seq.grad_fn).__name__ if weights_seq.grad_fn else "None"
        _ws_leaf = weights_seq.is_leaf
        print(f"[DIAG iter={i}] weights_seq: requires_grad={_ws_req}, is_leaf={_ws_leaf}, grad_fn={_ws_gfn}")

    rpy_history = torch.stack(rpy_history) # [T, B, 3]
    R_history = torch.stack(R_history)     # [T, B, 3, 3]
    real_act_history = torch.stack(real_act_history) # [T, B, 3]

    vec_to_pt = torch.stack(vec_to_pt_history)
    if vec_to_pt.dim() == 4: vec_to_pt = vec_to_pt.mean(1)
    
    # 1. 计算各项 Raw Loss (保留 [T, B] 维度用于 Step-wise 加权)

    # 碰撞距离 (先计算, 后续速度目标依赖它)
    dist_obj = vec_to_pt.norm(2, -1) - env.margin  # [T, B]

    # 自适应速度目标: 近障碍物/近终点时自动减速
    speed_actual = v_history.norm(2, -1)  # [T, B]
    dist_to_goal = (env.p_target - p_history).norm(2, -1)  # [T, B]
    v_max = float(env.max_speed)
    speed_factor_obs = torch.sigmoid((dist_obj - 0.8) * 5.0)   # ~0 near wall, ~1 far
    speed_factor_goal = torch.clamp(dist_to_goal / args.speed_goal_slow_dist, 0.0, 1.0)
    v_target_adaptive = v_max * (args.speed_near_obs_floor + (1.0 - args.speed_near_obs_floor) * speed_factor_obs) * speed_factor_goal
    loss_speed_seq = F.smooth_l1_loss(speed_actual, v_target_adaptive.detach(), reduction='none')

    target_dir = safe_normalize(env.p_target - p_history, dim=-1)
    v_dir = safe_normalize(v_history, dim=-1)
    loss_direction_seq = (1.0 - (v_dir * target_dir).sum(-1))

    # 多尺度避障 + 前瞻碰撞预测
    vec_to_pt_dir = safe_normalize(vec_to_pt, dim=-1)
    approach_speed = (v_history * vec_to_pt_dir).sum(-1)  # 正值=正在靠近障碍物
    dist_future_02 = dist_obj - F.relu(approach_speed) * 0.2  # 0.2s前瞻
    dist_future_04 = dist_obj - F.relu(approach_speed) * 0.4  # 0.4s前瞻
    collision_depth = F.relu(-dist_obj)
    safe_margin = args.avoid_safe_margin
    loss_avoidance_seq = (
        F.softplus((safe_margin - dist_obj) * 12.0) +
        0.5 * F.softplus(-dist_obj * 32.0) +
        0.3 * F.softplus((safe_margin - dist_future_02) * 10.0) +
        0.2 * F.softplus((safe_margin - dist_future_04) * 10.0) +
        collision_depth.pow(2)
    )

    # 注意: compute_overlap_loss_per_step 返回 [B, T], 需要 permute 成 [T, B]
    loss_exploration_seq = compute_overlap_loss_per_step(p_history, sigma=1.0, time_window=50).permute(1, 0)

    # [新增] Stuck Loss: 卡住惩罚
    loss_stuck_seq, loss_collision_duration_seq, stuck_ratio = compute_stuck_loss(
        p_history, collision_depth,
        stuck_window=args.stuck_window,
        displacement_threshold=args.stuck_displacement_threshold
    )

    actual_T = p_history.shape[0]

    # 高度约束损失 (固定权重, 不经LGN控制): 不越界 + 平滑高度变化
    z_pos = p_history[:, :, 2]  # [T, B]
    z_floor = float(args.height_floor)
    z_ceiling = float(args.height_ceiling) if args.height_ceiling is not None else float(getattr(env, 'map_z_max', 5.0))
    z_sharpness = float(args.height_bound_sharpness)
    loss_height_bound_seq = (
        F.softplus((z_pos - z_ceiling) * z_sharpness)
        + F.softplus((z_floor - z_pos) * z_sharpness)
    )
    z_delta = z_pos[1:] - z_pos[:-1]
    loss_height_smooth_seq = F.smooth_l1_loss(z_delta, torch.zeros_like(z_delta), reduction='none')
    loss_height_smooth_seq = torch.cat([torch.zeros_like(z_pos[:1]), loss_height_smooth_seq], dim=0)
    loss_height_seq = loss_height_bound_seq + float(args.height_smooth_weight) * loss_height_smooth_seq

    # 归一化各损失项到相同尺度 (可微除法, 统计量不反传)
    loss_speed_n, loss_dir_n, loss_avoid_n, loss_expl_n = \
        loss_normalizer.normalize(
            loss_speed_seq, loss_direction_seq, loss_avoidance_seq, loss_exploration_seq
        )

    # 纯学习模式: 仅使用 LGN 动态权重，不叠加先验或规则门控
    # 这里直接使用原始权重，不做和为1归一化。
    effective_weights_seq = weights_seq

    # [诊断] 检查loss项和effective_weights_seq的梯度属性
    if i % 100 == 0:
        _wsn_req = effective_weights_seq.requires_grad
        _wsn_gfn = type(effective_weights_seq.grad_fn).__name__ if effective_weights_seq.grad_fn else "None"
        _ls_req = loss_speed_n.requires_grad
        _ld_req = loss_dir_n.requires_grad
        _la_req = loss_avoid_n.requires_grad
        _le_req = loss_expl_n.requires_grad
        print(f"[DIAG iter={i}] effective_weights_seq: requires_grad={_wsn_req}, grad_fn={_wsn_gfn}")
        print(f"[DIAG iter={i}] loss_n requires_grad: speed={_ls_req}, dir={_ld_req}, avoid={_la_req}, expl={_le_req}")

    # 2. Step-wise 加权 (Broadcasting: [T, B] * [T, B])
    weighted_loss_map = (
        effective_weights_seq[:, :, 0] * loss_speed_n +
        effective_weights_seq[:, :, 1] * loss_dir_n +
        effective_weights_seq[:, :, 2] * loss_avoid_n +
        effective_weights_seq[:, :, 3] * loss_expl_n
    )

    # [新增] 固定权重的卡住惩罚 (不经过LGN，保证基本安全性)
    loss_stuck_total = args.stuck_loss_weight * loss_stuck_seq.mean()
    loss_collision_duration_total = args.collision_duration_weight * loss_collision_duration_seq.mean()

    # 3. 最终 Proxy Loss (含固定权重的高度约束、卡住惩罚)
    proxy_loss = (
        weighted_loss_map.mean() +
        2.0 * loss_height_seq.mean() +
        loss_stuck_total +
        loss_collision_duration_total
    )

    # [诊断] 检查proxy_loss计算链中的梯度属性
    if i % 100 == 0:
        _wlm_req = weighted_loss_map.requires_grad
        _wlm_gfn = type(weighted_loss_map.grad_fn).__name__ if weighted_loss_map.grad_fn else "None"
        _ew_req = effective_weights_seq.requires_grad
        _ew_gfn = type(effective_weights_seq.grad_fn).__name__ if effective_weights_seq.grad_fn else "None"
        _pl_req = proxy_loss.requires_grad
        _pl_gfn = type(proxy_loss.grad_fn).__name__ if proxy_loss.grad_fn else "None"
        print(f"[DIAG iter={i}] effective_weights: requires_grad={_ew_req}, grad_fn={_ew_gfn}")
        print(f"[DIAG iter={i}] weighted_loss_map: requires_grad={_wlm_req}, grad_fn={_wlm_gfn}")
        print(f"[DIAG iter={i}] proxy_loss: requires_grad={_pl_req}, grad_fn={_pl_gfn}")

    # --- Meta Loss Components ---
    loss_meta_pos = torch.norm(p_history[-1] - env.p_target, 2, -1).mean()
    loss_meta_coll_soft = F.softplus(-dist_obj * 32.0).clamp(max=100.0).mean()
    loss_meta_coll_hard = collision_depth.pow(2).mean()
    loss_meta_coll_peak = collision_depth.max(dim=0).values
    loss_meta_coll_event = torch.sigmoid(
        (loss_meta_coll_peak - args.meta_coll_event_threshold) * args.meta_coll_event_temp
    ).mean()
    loss_meta_coll_event_rate = (loss_meta_coll_peak > 0).float().mean()
    loss_meta_coll = (
        args.meta_coll_soft_weight * loss_meta_coll_soft
        + args.meta_coll_hard_weight * loss_meta_coll_hard
        + args.meta_coll_event_weight * loss_meta_coll_event
    )
    loss_meta_ctrl = act_buffer.norm(2, -1).sum()
    # 与 main_cuda.py 一致的平滑损失构造：jerk + snap
    loss_meta_jerk = act_buffer.diff(1, 0).mul(15.0).pow(2).sum(-1).mean()
    loss_meta_snap = (F.normalize(act_buffer - env.g_std, dim=-1)
                      .diff(1, 0).diff(1, 0).mul(15.0 ** 2).pow(2).sum(-1).mean())
    loss_meta_height = loss_height_seq.mean()
    
    # [新增] 速度预测误差损失：仿照 main_cuda.py 的构造逻辑
    # v_preds: [T, B, 3], v_history: [T, B, 3]
    v_preds_tensor = torch.stack(v_preds)  # [T, B, 3]
    loss_meta_v_pred = F.mse_loss(v_preds_tensor, v_history.detach())

    # --- 全局规划引导损失 ---
    # planner guidance 仅在 LGN phase 计算，worker phase 跳过以降低规划开销
    if train_lgn_phase:
        loss_meta_guidance, guidance_components = compute_global_guidance_meta_loss(
            env, p_history, v_history, env.p_target, vec_to_pt, dist_obj,
            a_history=a_history,
            sample_count=args.guide_sample_count,
            strategy=args.guide_sample_strategy,
            max_speed=float(env.max_speed),
            max_accel=args.guide_max_accel,
            max_decel=args.guide_max_decel,
            dir_weight=args.guide_dir_weight,
            speed_weight=args.guide_speed_weight,
            lateral_weight=args.guide_lateral_weight,
            escape_weight=args.guide_escape_weight,
            collision_threshold=args.guide_collision_threshold,
            accel_weight=args.guide_accel_weight,
            speed_diff_weight=args.guide_speed_diff_weight,
            recovery_speed_weight=args.guide_recovery_speed_weight,
        )
    else:
        zero = torch.tensor(0.0, device=p_history.device)
        loss_meta_guidance = zero
        guidance_components = {
            'dir_align': zero,
            'speed_diff': zero,
            'overspeed': zero,
            'underspeed': zero,
            'lateral_error': zero,
            'accel_mismatch': zero,
            'escape': zero,
            'depth': zero,
            'recovery_speed': zero,
            'valid_ratio': zero,
            'invalid_ratio': zero,
            'collision_ratio': zero,
            'sample_count': 0.0,
            'avg_curvature': 0.0,
            'avg_path_progress': 0.0,
            'avg_lateral_error': 0.0,
            'max_lateral_error': 0.0,
            'planner_success_ratio': 0.0,
            'avg_ref_speed': 0.0,
        }

    # 训练目标仅使用位置/碰撞/高度/引导/卡住; 控制项仅用于日志监控
    # [新增] 在 meta loss 中也加入卡住惩罚
    loss_meta_stuck = loss_stuck_seq.mean()
    loss_meta_collision_duration = loss_collision_duration_seq.mean()

    meta_loss = (
        loss_meta_pos +
        loss_meta_coll +
        loss_meta_height * 2.0 +
        args.meta_guidance_weight * loss_meta_guidance +
        args.meta_smooth_jerk_weight * loss_meta_jerk +
        args.meta_smooth_snap_weight * loss_meta_snap +
        args.meta_smooth_v_pred_weight * loss_meta_v_pred +
        args.stuck_loss_weight * loss_meta_stuck +
        args.collision_duration_weight * loss_meta_collision_duration
    )
    proxy_loss = sanitize_tensor(proxy_loss, nan=1e3, posinf=1e3, neginf=1e3)
    meta_loss = sanitize_tensor(meta_loss, nan=1e3, posinf=1e3, neginf=1e3)

    ###### C. Optimization ######
    optim_worker.zero_grad()
    optim_lgn.zero_grad()
    lgn_update_loss = 0.0
    worker_grad_norm = 0.0
    worker_grad_max = 0.0
    worker_grad_nonfinite = 0.0
    worker_grad_elems = 0.0
    worker_clip_pre = 0.0
    proxy_grad_speed = 0.0
    proxy_grad_dir = 0.0
    proxy_grad_avoid = 0.0
    proxy_grad_expl = 0.0
    proxy_grad_speed_nonfinite = 0.0
    proxy_grad_dir_nonfinite = 0.0
    proxy_grad_avoid_nonfinite = 0.0
    proxy_grad_expl_nonfinite = 0.0
    proxy_grad_speed_elems = 0.0
    proxy_grad_dir_elems = 0.0
    proxy_grad_avoid_elems = 0.0
    proxy_grad_expl_elems = 0.0
    lgn_grad_norm = 0.0
    lgn_grad_max = 0.0
    lgn_grad_nonfinite = 0.0
    lgn_grad_elems = 0.0
    lgn_clip_pre = 0.0

    rollout_is_finite = bool(
        torch.isfinite(proxy_loss).all()
        and torch.isfinite(meta_loss).all()
        and torch.isfinite(weights_seq).all()
        and torch.isfinite(p_history).all()
        and torch.isfinite(v_history).all()
    )

    if not rollout_is_finite:
        pbar.set_description(f"[{phase_str}] non-finite rollout skipped")
        continue

    worker_params = tuple(worknet.parameters())
    proxy_grad_speed, proxy_grad_speed_nonfinite, proxy_grad_speed_elems = \
        get_loss_to_worker_grad_norm(loss_speed_seq.mean(), worker_params)
    proxy_grad_dir, proxy_grad_dir_nonfinite, proxy_grad_dir_elems = \
        get_loss_to_worker_grad_norm(loss_direction_seq.mean(), worker_params)
    proxy_grad_avoid, proxy_grad_avoid_nonfinite, proxy_grad_avoid_elems = \
        get_loss_to_worker_grad_norm(loss_avoidance_seq.mean(), worker_params)
    proxy_grad_expl, proxy_grad_expl_nonfinite, proxy_grad_expl_elems = \
        get_loss_to_worker_grad_norm(loss_exploration_seq.mean(), worker_params)

    if train_lgn_phase:
        # ===== Unrolled Bilevel: 可微内循环 (梯度加权版) =====
        # 核心改动：不再对 proxy_loss 统一求梯度，
        # 而是分别对四个 proxy 分项求梯度，再用 LGN 权重直接加权

        fast_params = dict(worknet.named_parameters())

        # 获取用于代理损失加权的实际权重（按全序列平均）
        # effective_weights_seq: [T, B, 4]，取平均得到 [4]
        lgn_weights_for_grad = effective_weights_seq.mean(dim=[0, 1])  # [4]

        # [诊断] 检查 lgn_weights_for_grad 的梯度属性
        if i % 100 == 0:
            print(f"[DIAG iter={i}] lgn_weights_for_grad: requires_grad={lgn_weights_for_grad.requires_grad}, "
                  f"grad_fn={type(lgn_weights_for_grad.grad_fn).__name__ if lgn_weights_for_grad.grad_fn else 'None'}, "
                  f"values={lgn_weights_for_grad.detach().cpu().tolist()}")

        # 预先计算四个 loss 分项的标量（用于分别求梯度）
        # loss_speed_n, loss_dir_n, loss_avoid_n, loss_expl_n: [T, B]
        loss_speed_scalar = loss_speed_n.mean()
        loss_dir_scalar = loss_dir_n.mean()
        loss_avoid_scalar = loss_avoid_n.mean()
        loss_expl_scalar = loss_expl_n.mean()

        for _inner in range(args.inner_steps):
            # 分别对四个 loss 分项求梯度
            g_speed = torch.autograd.grad(
                loss_speed_scalar, tuple(fast_params.values()),
                create_graph=True, allow_unused=True, retain_graph=True
            )
            g_dir = torch.autograd.grad(
                loss_dir_scalar, tuple(fast_params.values()),
                create_graph=True, allow_unused=True, retain_graph=True
            )
            g_avoid = torch.autograd.grad(
                loss_avoid_scalar, tuple(fast_params.values()),
                create_graph=True, allow_unused=True, retain_graph=True
            )
            g_expl = torch.autograd.grad(
                loss_expl_scalar, tuple(fast_params.values()),
                create_graph=True, allow_unused=True, retain_graph=True
            )

            # 用 LGN 权重直接加权四组梯度，得到 combined_grads
            # combined_grad[j] = w0 * g_speed[j] + w1 * g_dir[j] + w2 * g_avoid[j] + w3 * g_expl[j]
            w0, w1, w2, w3 = lgn_weights_for_grad[0], lgn_weights_for_grad[1], lgn_weights_for_grad[2], lgn_weights_for_grad[3]
            combined_grads = []
            for gs, gd, ga, ge in zip(g_speed, g_dir, g_avoid, g_expl):
                if gs is None and gd is None and ga is None and ge is None:
                    combined_grads.append(None)
                else:
                    cg = (
                        (w0 * gs if gs is not None else 0.0) +
                        (w1 * gd if gd is not None else 0.0) +
                        (w2 * ga if ga is not None else 0.0) +
                        (w3 * ge if ge is not None else 0.0)
                    )
                    combined_grads.append(cg)

            # [诊断] 在第一个 inner step 检查 combined_grads -> lgn
            if _inner == 0 and i % 100 == 0:
                _cg0 = next((g for g in combined_grads if g is not None), None)
                if _cg0 is not None:
                    try:
                        _lgn_params = list(lgn.parameters())
                        _cg_to_lgn = torch.autograd.grad(
                            _cg0.sum(), _lgn_params,
                            allow_unused=True, retain_graph=True, create_graph=False
                        )
                        _cg_none = sum(x is None for x in _cg_to_lgn)
                        _cg_nonzero = sum((x is not None and x.abs().sum().item() > 1e-10) for x in _cg_to_lgn)
                        print(f"[DIAG iter={i}] combined_grads[0] -> lgn: None={_cg_none}/{len(_cg_to_lgn)}, NonZero={_cg_nonzero}/{len(_cg_to_lgn)}")
                    except Exception as _e:
                        print(f"[DIAG iter={i}] combined_grads -> lgn check failed: {_e}")
                else:
                    print(f"[DIAG iter={i}] combined_grads all None")

            # 用 combined_grads 更新 fast_params（不做 sanitize/clamp，保留高阶梯度链）
            fast_params = {
                name: (p - args.inner_lr * cg if cg is not None else p)
                for (name, p), cg in zip(fast_params.items(), combined_grads)
            }

        # Step 2: 用虚拟更新后的 worker 做验证 rollout → meta_loss
        meta_loss_unrolled, meta_pos_ur, meta_coll_ur, meta_ctrl_ur = \
            unrolled_meta_rollout(env, worknet, fast_params, state_normalizer, args, B, device)
        if not torch.isfinite(meta_loss_unrolled):
            pbar.set_description(f"[{phase_str}] non-finite unroll skipped")
            continue

        # ====== [诊断] 梯度链路检查 (在backward之前) ======
        if i % 100 == 0:
            # 检查各节点的 requires_grad
            _diag_weights_req = weights_seq.requires_grad
            _diag_proxy_req = proxy_loss.requires_grad
            _diag_meta_req = meta_loss_unrolled.requires_grad

            # 检查 weights_seq 的 grad_fn（如果有）
            _diag_weights_grad_fn = type(weights_seq.grad_fn).__name__ if weights_seq.grad_fn else "None"
            _diag_proxy_grad_fn = type(proxy_loss.grad_fn).__name__ if proxy_loss.grad_fn else "None"
            _diag_meta_grad_fn = type(meta_loss_unrolled.grad_fn).__name__ if meta_loss_unrolled.grad_fn else "None"

            # 使用 torch.autograd.grad 显式测试梯度是否能流到 lgn.parameters()
            lgn_param_list = list(lgn.parameters())
            try:
                _test_grads = torch.autograd.grad(
                    meta_loss_unrolled, lgn_param_list,
                    allow_unused=True, retain_graph=True, create_graph=False
                )
                _grad_none_count = sum(1 for g in _test_grads if g is None)
                _grad_nonzero_count = sum(1 for g in _test_grads if g is not None and g.abs().sum() > 1e-10)
                _grad_total = len(_test_grads)
            except Exception as _e:
                _grad_none_count = -1
                _grad_nonzero_count = -1
                _grad_total = len(lgn_param_list)
                print(f"[DIAG] autograd.grad failed: {_e}")

            # 检查 proxy_loss -> lgn 的梯度
            try:
                _proxy_grads = torch.autograd.grad(
                    proxy_loss, lgn_param_list,
                    allow_unused=True, retain_graph=True, create_graph=False
                )
                _proxy_grad_none = sum(1 for g in _proxy_grads if g is None)
                _proxy_grad_nonzero = sum(1 for g in _proxy_grads if g is not None and g.abs().sum() > 1e-10)
            except Exception as _e:
                _proxy_grad_none = -1
                _proxy_grad_nonzero = -1
                print(f"[DIAG] proxy->lgn grad failed: {_e}")

            print(f"[DIAG iter={i}] weights_seq.requires_grad={_diag_weights_req}, grad_fn={_diag_weights_grad_fn}")
            print(f"[DIAG iter={i}] proxy_loss.requires_grad={_diag_proxy_req}, grad_fn={_diag_proxy_grad_fn}")
            print(f"[DIAG iter={i}] meta_loss_unrolled.requires_grad={_diag_meta_req}, grad_fn={_diag_meta_grad_fn}")
            print(f"[DIAG iter={i}] meta_loss -> lgn: None={_grad_none_count}/{_grad_total}, NonZero={_grad_nonzero_count}/{_grad_total}")
            print(f"[DIAG iter={i}] proxy_loss -> lgn: None={_proxy_grad_none}/{_grad_total}, NonZero={_proxy_grad_nonzero}/{_grad_total}")

            # [关键检查] meta_loss_unrolled -> fast_params
            # 如果这里也是全 None，说明问题在 unrolled_meta_rollout / env.run
            # 如果这里不是 None，说明 meta -> fast_params 通了，但 fast_params -> lgn 的高阶链断了
            try:
                _fast_param_list = [p for _, p in fast_params.items()]
                _meta_to_fast = torch.autograd.grad(
                    meta_loss_unrolled, _fast_param_list,
                    allow_unused=True, retain_graph=True, create_graph=False
                )
                _fast_none_cnt = sum(g is None for g in _meta_to_fast)
                _fast_nonzero_cnt = sum((g is not None and g.abs().sum().item() > 1e-10) for g in _meta_to_fast)
                _fast_total = len(_meta_to_fast)
            except Exception as _e:
                _fast_none_cnt = -1
                _fast_nonzero_cnt = -1
                _fast_total = len(fast_params)
                print(f"[DIAG] meta->fast_params grad failed: {_e}")
            print(f"[DIAG iter={i}] meta_loss -> fast_params: None={_fast_none_cnt}/{_fast_total}, NonZero={_fast_nonzero_cnt}/{_fast_total}")

            # [最终定位] fast_params -> lgn 直接检查
            # 如果这里全 None，说明 fast_params 根本不依赖 LGN（inner update 高阶链断了）
            # 如果这里有 NonZero，说明依赖存在但被其他地方切断
            try:
                _fast_param_list = [p for _, p in fast_params.items()]
                # 用所有 fast_params 的和作为中间输出
                _fast_sum = sum([fp.sum() for fp in _fast_param_list])
                _fast_to_lgn = torch.autograd.grad(
                    _fast_sum, lgn_param_list,
                    allow_unused=True, retain_graph=True, create_graph=False
                )
                _fast_lgn_none = sum(g is None for g in _fast_to_lgn)
                _fast_lgn_nonzero = sum((g is not None and g.abs().sum().item() > 1e-10) for g in _fast_to_lgn)
            except Exception as _e:
                _fast_lgn_none = -1
                _fast_lgn_nonzero = -1
                print(f"[DIAG] fast_params->lgn grad failed: {_e}")
            print(f"[DIAG iter={i}] fast_params -> lgn: None={_fast_lgn_none}/{_grad_total}, NonZero={_fast_lgn_nonzero}/{_grad_total}")

            # 检查 trajectory_lgn_weights 的第一个元素
            if len(trajectory_lgn_weights) > 0:
                _first_w = trajectory_lgn_weights[0]
                print(f"[DIAG iter={i}] first_lgn_weight.requires_grad={_first_w.requires_grad}, grad_fn={type(_first_w.grad_fn).__name__ if _first_w.grad_fn else 'None'}")

        # Step 3: 反向传播贯穿整条链路
        #   meta_loss → fast_params → ∇proxy_loss → LGN weights → LGN params
        # 纯学习模式: LGN 仅由 unrolled meta loss 驱动
        lgn_total = meta_loss_unrolled
        lgn_total.backward()
        lgn_grad_norm, lgn_grad_max, lgn_grad_nonfinite, lgn_grad_elems = get_grad_stats(lgn)

        # [新增] LGN 梯度爆炸保护
        lgn_grad_is_bad = (
            not math.isfinite(lgn_grad_norm) or
            lgn_grad_norm > args.grad_explosion_threshold or
            lgn_grad_nonfinite > 0
        )

        if args.enable_grad_protection and lgn_grad_is_bad:
            lgn_explosion_consecutive += 1
            grad_explosion_count += 1
            pbar.set_description(f"[{phase_str}] LGN grad explosion #{lgn_explosion_consecutive} skipped (norm={lgn_grad_norm:.2e})")

            # 连续爆炸过多，重置优化器状态
            if lgn_explosion_consecutive >= args.grad_explosion_skip_window:
                optim_lgn.state.clear()
                lgn_explosion_consecutive = 0
                print(f"[WARN] LGN optimizer state reset at iter {i}")
        else:
            lgn_explosion_consecutive = 0
            lgn_clip_pre = float(nn.utils.clip_grad_norm_(lgn.parameters(), 1.0).item())
            optim_lgn.step()
            sanitize_module_(lgn, clamp_value=5.0)

        lgn_update_loss = meta_loss_unrolled.detach()

        # [新增] 诊断 LGN 梯度链路
        with torch.no_grad():
            lgn_params_with_grad = sum(1 for p in lgn.parameters() if p.grad is not None and p.grad.abs().sum() > 1e-10)
            lgn_total_params = sum(1 for _ in lgn.parameters())
    else:
        proxy_loss.backward()
        worker_grad_norm, worker_grad_max, worker_grad_nonfinite, worker_grad_elems = get_grad_stats(worknet)

        # [新增] Worker 梯度爆炸保护
        worker_grad_is_bad = (
            not math.isfinite(worker_grad_norm) or
            worker_grad_norm > args.grad_explosion_threshold or
            worker_grad_nonfinite > 0
        )

        if args.enable_grad_protection and worker_grad_is_bad:
            worker_explosion_consecutive += 1
            grad_explosion_count += 1
            pbar.set_description(f"[{phase_str}] Worker grad explosion #{worker_explosion_consecutive} skipped (norm={worker_grad_norm:.2e})")

            # 连续爆炸过多，重置优化器状态
            if worker_explosion_consecutive >= args.grad_explosion_skip_window:
                optim_worker.state.clear()
                worker_explosion_consecutive = 0
                print(f"[WARN] Worker optimizer state reset at iter {i}")
        else:
            worker_explosion_consecutive = 0
            worker_clip_pre = float(nn.utils.clip_grad_norm_(worknet.parameters(), 5.0).item())
            optim_worker.step()
            sanitize_module_(worknet, clamp_value=10.0)
            sched.step()

        # LGN 诊断变量初始化（仅在 worker phase）
        lgn_params_with_grad = 0
        lgn_total_params = sum(1 for _ in lgn.parameters())

    ###### D. Logging & Saving (Enhanced) ######
    if train_lgn_phase:
        pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Unroll: {lgn_update_loss:.3f}")
    else:
        pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Loss: {meta_loss:.3f}")
    
    with torch.no_grad():
        success = torch.all(dist_obj > 0, 0)
        # 计算平均权重 (用于 Scalar 显示)
        avg_weights = effective_weights_seq.mean(dim=[0, 1]).cpu()  # 实际用于加权损失的权重
        # 仅用于监控分布形态；不参与训练加权
        _w_prob = torch.softmax(effective_weights_seq.detach(), dim=-1)
        weight_entropy = (-_w_prob * torch.log(_w_prob.clamp_min(1e-8))).sum(dim=-1).mean()
        v_norm = v_history.norm(dim=-1)
        avg_speed = v_norm.mean()
        min_speed_threshold = float(env.max_speed) * 0.7
        act_cmd_mean = real_act_history.mean(dim=(0, 1))
        act_cmd_abs_mean = real_act_history.abs().mean(dim=(0, 1))
        act_cmd_norm_mean = real_act_history.norm(dim=-1).mean()
        
        log_data = {
            # ============================== 主要Loss ==============================
            # Worker策略网络的总代理损失。当收敛时应逐渐下降。
            # 现象：初期波动较大，训练中期应该趋势向下，如果持续增加则需要调整学习率或参数
            'Loss/1_Proxy_Total': proxy_loss,
            # LGN元学习器在反向unroll中的总优化目标。应该能够自动学习合适的权重
            # 现象：通常小于proxy_loss，训练中期应该相对稳定，大的波动表示元学习不稳定
            'Loss/2_Meta_Total': meta_loss,

            # ============================== 权重监控（LGN输出） ==============================
            # [权重/0_速度] 实际用于损失加权的速度项权重（非归一化）
            'Weights/0_Speed': avg_weights[0],
            # [权重/1_方向] 实际用于损失加权的方向项权重（非归一化）
            'Weights/1_Direction': avg_weights[1],
            # [权重/2_避障] 实际用于损失加权的避障项权重（非归一化）
            'Weights/2_Avoidance': avg_weights[2],
            # [权重/3_探索] 实际用于损失加权的探索项权重（非归一化）
            'Weights/3_Exploration': avg_weights[3],

            # ============================== 权重分布统计 ==============================
            # 整个Episode中的有效权重最小值（权重可正可负）
            'Weight_Stats/Raw_Min': effective_weights_seq.min(),
            # 整个Episode中的权重序列的最大值。用于检测权重上界
            'Weight_Stats/Raw_Max': effective_weights_seq.max(),
            # 整个Episode中的权重序列的平均值。用于监控总体权重水平
            'Weight_Stats/Raw_Mean': effective_weights_seq.mean(),
            # 监控用熵：对有效权重做softmax后计算，不参与训练
            'Weight_Stats/Entropy': weight_entropy,

            # ============================== Proxy Loss 分项 ==============================
            # [代理损失/0_速度] 鼓励无人机达到自适应速度目标的损失
            # 现象：初期可能较大，应逐渐减小。持续增加表示无人机无法跟上速度目标
            'Proxy_Comp/0_Speed': loss_speed_seq.mean(),
            # [代理损失/1_方向] 鼓励无人机方向与参考路径对齐的损失
            # 现象：应该相对较小(通常<0.5)。太大表示规划引导效果差或参考路径偏离实际
            'Proxy_Comp/1_Direction': loss_direction_seq.mean(),
            # [代理损失/2_避障] 碰撞惩罚项(安全余度内的软惩罚)
            # 现象：应该较小且趋势向下。如果>1.0说明无人机频繁接近障碍物
            'Proxy_Comp/2_Avoidance': loss_avoidance_seq.mean(),
            # [代理损失/2_1_穿墙深度] 实际穿入障碍的深度(米)。诊断用，衡量碰撞严重程度
            # 现象：应该接近0。若>0.1米说明有明显穿墙现象，需要提升避障能力
            'Proxy_Comp/2_1_Collision_Depth': collision_depth.mean(),
            # [代理损失/3_探索] 鼓励多样化轨迹的散度惩罚(R^3中的轨迹方差)
            # 现象：通常较小。初期可能较大，训练后减小，说明策略收敛
            'Proxy_Comp/3_Exploration': loss_exploration_seq.mean(),
            # [代理损失/4_高度] 惩罚与参考路径的高度偏差
            # 现象：应该较小。如果>0.5说明无人机高度控制不当
            'Proxy_Comp/4_Height': loss_height_seq.mean(),

            # ============================== 卡住(Stuck)损失 ==============================
            # [卡住损失/5] 检测局部移动过慢的惩罚(看不到约15步的最小位移)
            # 现象：应该接近0。如果>0.1说明无人机经常陷入局部卡住状态
            'Proxy_Comp/5_Stuck': loss_stuck_seq.mean(),
            # [碰撞时间损失/6] 连续碰撞时长的惩罚(与碰撞持续帧数成正比)
            # 现象：应该接近0。若>0.2说明无人机碰撞后难以恢复
            'Proxy_Comp/6_Collision_Duration': loss_collision_duration_seq.mean(),
            # [卡住总损失/10] 整个Episode的卡住位移惩罚总和
            # 现象：应该较小。若持续>1.0说明需要调整stuck_displacement_threshold或提升导航能力
            'Proxy_Comp/10_Stuck_Total': loss_stuck_total,
            # [碰撞时间总损失/11] 整个Episode的碰撞时长惩罚总和
            # 现象：应该较小。若持续>1.0说明避障算法需要改进
            'Proxy_Comp/11_Collision_Duration_Total': loss_collision_duration_total,

            # ============================== 卡住状态诊断 ==============================
            # [卡住比例] Episode中陷入卡住状态的步数占比 (0-1)
            # 现象：应该 < 0.1。若 > 0.3 说明导航效率很低，无人机经常卡住
            'Stuck/Ratio': stuck_ratio,
            # [碰撞连续时长_平均] 单次碰撞持续的平均步数
            # 现象：应该 < 5步。若 > 10步说明碰撞后无法及时脱离
            'Stuck/Collision_Streak_Mean': loss_collision_duration_seq.mean(),
            # [碰撞连续时长_最大] 单次碰撞持续的最长步数
            # 现象：应该 < 30步。若 > 100步说明有严重的碰撞困境
            'Stuck/Collision_Streak_Max': loss_collision_duration_seq.max(),

            # === [增强] Meta Loss 分项 ===
            'Meta_Comp/1_Position': loss_meta_pos,
            'Meta_Comp/2_Collision': loss_meta_coll,
            'Meta_Comp/2_1_Collision_Soft': loss_meta_coll_soft,#软碰撞项（靠墙即升高，连续梯度）
            'Meta_Comp/2_2_Collision_Hard': loss_meta_coll_hard,# 穿墙深度平方项。对”已经进墙”的样本施加强惩罚，且穿得越深罚越重
            'Meta_Comp/2_3_Collision_Event': loss_meta_coll_event,# 可微事件惩罚（用于训练）
            'Meta_Comp/2_4_Collision_Event_Rate': loss_meta_coll_event_rate,# 真实事件率（监控用）
            'Meta_Comp/3_Control': loss_meta_ctrl,
            'Meta_Comp/4_Height': loss_meta_height,
            'Meta_Comp/8_Smooth_Jerk': loss_meta_jerk,
            'Meta_Comp/9_Smooth_Snap': loss_meta_snap,
            'Meta_Comp/10_Smooth_V_Pred': loss_meta_v_pred,  # [新增] 速度预测误差
            # === [新增] Meta Loss 中的卡住损失 ===
            'Meta_Comp/6_Stuck': loss_meta_stuck,
            'Meta_Comp/7_Collision_Duration': loss_meta_collision_duration,

            # ============================== 全局规划引导损失 ==============================
            # [元损失/5_引导] 整个全局规划引导的加权总损失
            # 现象：应该相对较小但稳定。若剧烈波动说明A*规划质量不稳定
            'Meta_Comp/5_Guidance': loss_meta_guidance,

            # ====== 引导损失详细分项 ======
            # 轨迹方向与参考路径的对齐误差。衡量是否沿着规划的路径行进
            # 现象：应该 < 0.5。若 > 1.0说明无人机的方向与规划路径偏离太大
            'Guidance/Dir_Align': guidance_components['dir_align'],
            # 超速惩罚。无人机速度超过规划给定的参考速度的惩罚
            # 现象：应该较小。若较大说明无人机尝试以高于规划建议的速度行进
            'Guidance/Overspeed': guidance_components['overspeed'],
            # 欠速惩罚。无人机速度远低于规划给定的参考速度的惩罚
            # 现象：通常较小。若和超速都很大说明速度控制不稳定
            'Guidance/Underspeed': guidance_components.get('underspeed', 0.0),
            # 速度差异总体惩罚(超速+欠速)
            # 现象：应该 < 0.5。表示无人机能够相对准确地跟踪参考速度
            'Guidance/Speed_Diff': guidance_components.get('speed_diff', 0.0),
            # 横向偏差。轨迹到参考路径的几何距离(m)
            # 现象：应该 < 1.0m。若>2.0m说明路径跟踪精度差，可能需要调整LGN权重
            'Guidance/Lateral_Error': guidance_components['lateral_error'],
            # 加速度不匹配。实际加速度与规划建议的加速度差异
            # 现象：应该较小。若较大说明无人机加速度控制与规划预期不符
            'Guidance/Accel_Mismatch': guidance_components.get('accel_mismatch', 0.0),
            # 逃脱惩罚。已碰撞位置重新脱离的难度
            # 现象：应该较小。若>0.5说明无人机碰撞后难以逃脱(需要提升碰撞反应)
            'Guidance/Escape': guidance_components['escape'],
            # 深度项。与规划路径的三维距离(或高度相关)
            # 现象：应该较小。表示三维轨迹与规划路径吻合度好
            'Guidance/Depth': guidance_components['depth'],
            # 恢复速度惩罚。在规划失效或阻挡区域恢复导航时的速度抑制
            # 现象：通常很小。若>0.1说明经常遇到规划失效的情况
            'Guidance/Recovery_Speed': guidance_components.get('recovery_speed', 0.0),
            # 有效规划点比例 (A*规划成功的点数占比)
            # 现象：应该接近1.0。若 < 0.9说明环境过于复杂或A*参数不当
            'Guidance/Valid_Ratio': guidance_components['valid_ratio'],
            # 无效规划点比例 (A*规划失败/blocked的点数占比)
            # 现象：应该接近0。若 > 0.1说明存在无法规划通过的点，需要检查参数
            'Guidance/Invalid_Ratio': guidance_components.get('invalid_ratio', 0.0),
            # 采样点中碰撞的比例 (用于A*路径的碰撞检测)
            # 现象：应该接近0。若 > 0.1说明参考路径本身与障碍物碰撞，需要检查障碍物膨胀边距
            'Guidance/Collision_Ratio': guidance_components['collision_ratio'],
            # 本Episode中从轨迹采样的关键引导点总数
            # 现象：应该等于guide_sample_count参数。表示正常的采样策略工作
            'Guidance/Sample_Count': guidance_components['sample_count'],
            # 规划路径的曲率平均值(弯度)。用于判断任务难度
            # 现象：直线任务<0.5，弯曲任务>1.0。表示任务复杂性
            'Guidance/Avg_Curvature': guidance_components.get('avg_curvature', 0.0),
            # 规划路径的平均进展比例(已覆盖距离/总距离)
            # 现象：应该逐步增加到1.0。表示无人机沿着规划路径前进
            'Guidance/Avg_Path_Progress': guidance_components.get('avg_path_progress', 0.0),
            # 规划路径上的平均参考速度
            # 现象：3-8 m/s之间合理。表示A*规划计算的梯形速度剖面
            'Guidance/Avg_Ref_Speed': guidance_components.get('avg_ref_speed', 0.0),
            # 横向偏差的平均值
            # 现象：应该 < 'Guidance/Lateral_Error'的一半。表示偏差分布相对均匀
            'Guidance/Avg_Lateral_Error': guidance_components.get('avg_lateral_error', 0.0),
            # 横向偏差的最大值(最坏情况)
            # 现象：应该接近'Guidance/Lateral_Error'。若远小于说明偏差集中在某些点
            'Guidance/Max_Lateral_Error': guidance_components.get('max_lateral_error', 0.0),
            # A*规划成功率(从起点到终点规划成功的比例)
            # 现象：应该 > 0.95。若 < 0.8说明规划器参数需要调整
            'Guidance/Planner_Success_Ratio': guidance_components.get('planner_success_ratio', 0.0),

            # ============================== 性能指标 ==============================
            # 到达目标的成功率(batch维度平均)
            # 现象：初期接近0，随训练应逐渐增加到 > 0.8。好的模型应 > 0.95
            'Metrics/Success_Rate': success.float().mean(),
            # Episode中整体平均速度
            # 现象：应该 3-8 m/s。初期可能较低(0.5-2.0)，训练后应加快
            'Metrics/Avg_Speed': avg_speed,
            # 是否平均速度低于最小值阈值的标记(0或1)
            # 现象：应该 < 0.2。若持续 > 0.5说明无人机移动过慢，需要提升速度目标权重
            'Metrics/Speed_Below_Threshold': (avg_speed < min_speed_threshold).float(),
            # Episode中的最小瞬间速度
            # 现象：应该 > 0.1 m/s。若接近0说明无人机在卡住或悬停
            'Metrics/Min_Speed': v_norm.min(),
            # Episode中的最大瞬间速度
            # 现象：应该 < 15 m/s。若 > 20 m/s说明速度限制失效或参数设置过大
            'Metrics/Max_Speed': v_norm.max(),
            # Episode持续的实际步数(在goal_radius内提前结束前)
            # 现象：应该 < timesteps参数。通常 100-280步。远小于说明到达快(效率高)
            'Metrics/Episode_Length': actual_T,
            # 自适应速度目标值(综合考虑障碍物距离和目标距离)
            # 现象：应该 3-12 m/s。表示动态调整的速度参考
            'Metrics/Adaptive_Speed_Target': v_target_adaptive.mean(),

            # ============================== 控制信号监控 ==============================
            # 加速度命令的平均模长 (与速度类似，单位m/s^2)
            # 现象：应该 1-5 m/s^2。表示平均控制强度
            'Control/Accel_Cmd_Norm_Mean': act_cmd_norm_mean,
            # X轴加速度命令的平均值(可正可负，表示前后方向)
            # 现象：应该接近0。若持续 > 1.0或 < -1.0说明前后控制不平衡
            'Control/Accel_Cmd_X_Mean': act_cmd_mean[0],
            # Y轴加速度命令的平均值(可正可负，表示左右方向)
            # 现象：应该接近0。表示左右平衡
            'Control/Accel_Cmd_Y_Mean': act_cmd_mean[1],
            # Z轴加速度命令的平均值(可正可负，表示上下方向)
            # 现象：应该接近0。若持续 > 0.5说明无人机倾向向上移动
            'Control/Accel_Cmd_Z_Mean': act_cmd_mean[2],
            # X轴加速度命令的绝对值平均(表示强度，不考虑方向)
            # 现象：应该 < 5 m/s^2。表示前后方向的平均控制强度
            'Control/Accel_Cmd_X_AbsMean': act_cmd_abs_mean[0],
            # Y轴加速度命令的绝对值平均
            # 现象：应该 < 5 m/s^2
            'Control/Accel_Cmd_Y_AbsMean': act_cmd_abs_mean[1],
            # Z轴加速度命令的绝对值平均
            # 现象：应该 < 3 m/s^2。Z方向控制通常小于水平方向
            'Control/Accel_Cmd_Z_AbsMean': act_cmd_abs_mean[2],

            # ============================== 状态归一化统计 ==============================
            # 运行平均的状态向量平均值。实时更新的归一化器均值
            # 现象：应该接近0(全局平均的偏差应为0)。非0表示状态分布偏移
            'Norm/State_Mean': state_normalizer.mean[0],
            # 运行平均的状态向量方差。实时更新的归一化器方差
            # 现象：应该接近1.0。远离1.0表示状态分布方差不均
            'Norm/State_Var': state_normalizer.var[0],
            # 运行平均的更新次数计数器
            # 现象：应该与迭代数相关。可用来检查归一化器是否在更新
            'Norm/Update_Count': state_normalizer.count,

            # [备用] 与上述重复，保留以兼容旧版脚本
            'Stats/Norm_Mean': state_normalizer.mean[0],
            'Stats/Norm_Var': state_normalizer.var[0],
            'Stats/Norm_Count': state_normalizer.count,

            # ============================== 梯度监控(梯度爆炸诊断) ==============================
            # Worker网络的全局梯度范数(所有参数的梯度的L2范数)
            # 现象：应该 < 100。若 > 1000表示梯度爆炸。通常 1-50是正常范围
            'Grad/Worker_Global_Norm': worker_grad_norm,
            # Worker网络中单个梯度元素的最大绝对值
            # 现象：应该 < 10。若 > 100表示存在爆炸的梯度
            'Grad/Worker_Max_Abs': worker_grad_max,
            # Worker网络中非有限梯度(NaN/Inf)的参数数量
            # 现象：应该 = 0。若 > 0表示梯度爆炸或数值不稳定
            'Grad/Worker_NonFinite_Count': worker_grad_nonfinite,
            # Worker网络的总梯度元素数量
            # 现象：应该固定(取决于模型大小)。可用来归一化梯度统计
            'Grad/Worker_GradElem_Count': worker_grad_elems,
            # Worker梯度裁剪前的范数(如果启用了梯度裁剪)
            # 现象：应该 >= Worker_Global_Norm。差异大表示梯度裁剪反复触发
            'Grad/Worker_Clip_PreNorm': worker_clip_pre,

            # LGN网络的全局梯度范数(二阶可微unroll中的梯度)
            # 现象：应该 < 100。LGN梯度通常大于Worker(因为是Hessian-vec计算)
            'Grad/LGN_Global_Norm': lgn_grad_norm,
            # LGN网络中单个梯度元素的最大绝对值
            # 现象：应该 < 10。通常大于Worker的对应值
            'Grad/LGN_Max_Abs': lgn_grad_max,
            # LGN网络中非有限梯度的参数数量
            # 现象：应该 = 0。若 > 0表示二阶梯度数值不稳定，需要减小inner_steps或inner_lr
            'Grad/LGN_NonFinite_Count': lgn_grad_nonfinite,
            # LGN网络的总梯度元素数量
            # 现象：应该固定，通常小于Worker(因为LGN设计更小)
            'Grad/LGN_GradElem_Count': lgn_grad_elems,
            # LGN梯度裁剪前的范数
            # 现象：应该 >= LGN_Global_Norm
            'Grad/LGN_Clip_PreNorm': lgn_clip_pre,

            # ============================== 梯度爆炸保护监控 ==============================
            # 自训练开始以来检测到梯度爆炸的总次数
            # 现象：应该接近0或增速很慢。若快速增加说明梯度不稳定
            'Grad_Protection/Explosion_Count_Total': grad_explosion_count,
            # Worker网络连续爆炸的次数(重置前)
            # 现象：应该 < skip_window。若等于skip_window说明刚重置了优化器状态
            'Grad_Protection/Worker_Consecutive_Explosions': worker_explosion_consecutive,
            # LGN网络连续爆炸的次数(重置前)
            # 现象：应该 < skip_window
            'Grad_Protection/LGN_Consecutive_Explosions': lgn_explosion_consecutive,

            # ============================== LGN梯度链路诊断 ==============================
            # LGN参数中有梯度流动的参数数量
            # 现象：应该等于或接近Total_Params。若远小于说明存在断梯问题
            'LGN_Diag/Params_With_Grad': lgn_params_with_grad,
            # LGN的总参数数量
            # 现象：固定值(取决于LGN架构)。通常 数千到 数十万
            'LGN_Diag/Total_Params': lgn_total_params,
            # 有梯度的参数比例 (0-1)
            # 现象：应该接近1.0。若 < 0.95说明部分参数未参与梯度更新(可能被冻结或有bug)
            'LGN_Diag/Grad_Coverage_Ratio': lgn_params_with_grad / max(lgn_total_params, 1),
            # 权重序列(weights_seq)是否requires_grad的标记(Unroll中是否启用)
            # 现象：LGN phase期间应该=1.0。Worker phase应该=0.0。不符合表示Unroll配置有问题
            'LGN_Diag/Weights_Seq_Requires_Grad': float(weights_seq.requires_grad),

            # ============================== 四个权重对Worker梯度的单独贡献 ==============================
            # 速度损失对Worker梯度的贡献范数
            # 现象：应该 > 0。表示速度目标正常诱导梯度
            'Grad_ProxyWorker/0_Speed_Norm': proxy_grad_speed,
            # 方向损失对Worker梯度的贡献范数
            # 现象：应该 > 0。表示方向约束正常诱导梯度
            'Grad_ProxyWorker/1_Direction_Norm': proxy_grad_dir,
            # 避障损失对Worker梯度的贡献范数
            # 现象：应该 > 0。通常这项很大(避障很重要)
            'Grad_ProxyWorker/2_Avoidance_Norm': proxy_grad_avoid,
            # 探索损失对Worker梯度的贡献范数
            # 现象：通常较小。在某些训练阶段可能增大以鼓励多样性
            'Grad_ProxyWorker/3_Exploration_Norm': proxy_grad_expl,

            # 每项损失对应梯度中的非有限元素数(NaN/Inf)
            # 现象：应该 = 0
            'Grad_ProxyWorker/0_Speed_NonFinite': proxy_grad_speed_nonfinite,
            'Grad_ProxyWorker/1_Direction_NonFinite': proxy_grad_dir_nonfinite,
            'Grad_ProxyWorker/2_Avoidance_NonFinite': proxy_grad_avoid_nonfinite,
            'Grad_ProxyWorker/3_Exploration_NonFinite': proxy_grad_expl_nonfinite,

            # 每项损失对应梯度的元素数量(用于归一化比较)
            # 现象：应该相同(因为Worker输出相同)
            'Grad_ProxyWorker/0_Speed_GradElem': proxy_grad_speed_elems,
            'Grad_ProxyWorker/1_Direction_GradElem': proxy_grad_dir_elems,
            'Grad_ProxyWorker/2_Avoidance_GradElem': proxy_grad_avoid_elems,
            'Grad_ProxyWorker/3_Exploration_GradElem': proxy_grad_expl_elems
        }

        if train_lgn_phase:
            log_data['Loss/3_LGN_Unrolled_Meta'] = lgn_update_loss
            log_data['Meta_Unrolled/1_Position'] = meta_pos_ur
            log_data['Meta_Unrolled/2_Collision'] = meta_coll_ur
            log_data['Meta_Unrolled/3_Control'] = meta_ctrl_ur

        smooth_dict(log_data)

        # ============================== 每25次迭代记录一次 ==============================
        if (i + 1) % 25 == 0:
            # 将收集的所有标量的平均值写入tensorboard
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()
            # [训练模式] 1.0=LGN阶段(元学习), 0.0=Worker阶段(策略学习)
            # 用于在tensorboard中直观看训练的两个阶段交替
            writer.add_scalar('Status/Train_Mode', 1.0 if train_lgn_phase else 0.0, i + 1)
            # [迷宫年龄] 当前迷宫的年龄(从0开始，每maze_update_interval次更新后重置)
            # 值范围[0, maze_update_interval-1]，用来监控环境是否在定期更新
            writer.add_scalar('Status/Maze_Age', (maze_update_counter - 1) % args.maze_update_interval, i + 1)

        if is_save_iter(i):
            torch.save(worknet.state_dict(), os.path.join(save_dir, f'worker_ckpt_{i:06d}.pth'))
            torch.save(lgn.state_dict(), os.path.join(save_dir, f'lgn_ckpt_{i:06d}.pth'))
            torch.save(state_normalizer.state_dict(), os.path.join(save_dir, f'norm_ckpt_{i:06d}.pth'))

        if is_save_trajectory_iter(i):
            idx = 0
            
            # ============================== 1. 位置时序图 (X,Y,Z vs T) ==============================
            # 显示无人机在episode中的三维位置随时间的变化
            # 用于观察：轨迹是否平滑，是否快速到达目标，是否存在振荡
            fig_p, ax = plt.subplots()
            p_cpu = p_history[:, idx].cpu()
            ax.plot(p_cpu[:, 0], label='x'); ax.plot(p_cpu[:, 1], label='y'); ax.plot(p_cpu[:, 2], label='z')
            ax.legend(); ax.set_title(f"Iter {i} Pos (Time Series)")
            writer.add_figure('Trajectory/Position_Series', fig_p, i + 1)
            plt.close(fig_p)

            # ============================== 2. 三维轨迹 + 障碍物可视化 ==============================
            # 立体显示无人机轨迹(按速度着色)、所有障碍物(不同类型不同颜色)、起点(绿)、终点(黑)、目标(红)
            # 用于观察：是否成功避障，轨迹与障碍物的距离，是否有不必要的绕路
            fig_map = plt.figure(figsize=(8, 6))
            ax = fig_map.add_subplot(111, projection='3d')

            # 速度着色三维轨迹
            v_cpu = v_history[:, idx].cpu()
            speed_cpu = v_cpu.norm(dim=-1).numpy()
            traj_xyz = p_cpu.numpy()
            cmap = plt.get_cmap('coolwarm')
            norm = Normalize(vmin=0.0, vmax=15.0)
            sc = ax.scatter(traj_xyz[:, 0], traj_xyz[:, 1], traj_xyz[:, 2],
                            c=speed_cpu, cmap=cmap, norm=norm, s=9, alpha=0.95, label='Trajectory')
            ax.plot(traj_xyz[:, 0], traj_xyz[:, 1], traj_xyz[:, 2], color='steelblue', linewidth=1.0, alpha=0.55)
            fig_map.colorbar(sc, ax=ax, pad=0.08, shrink=0.75, label='Speed (m/s)')

            # 三维障碍物盒体显示
            x_all = [p_cpu[:, 0]]
            y_all = [p_cpu[:, 1]]
            z_all = [p_cpu[:, 2]]
            if hasattr(env, 'voxels') and env.voxels.numel() > 0:
                vox = env.voxels[0].detach().cpu()
                # 过滤掉用于roof占位的超大盒体，避免坐标轴被极端值拉爆
                valid_vox = vox[(vox[:, 3:6] < 20).all(dim=1)]
                for box in valid_vox.numpy():
                    cx, cy, cz, hx, hy, hz = box.tolist()
                    ax.bar3d(cx - hx, cy - hy, cz - hz, 2 * hx, 2 * hy, 2 * hz,
                             color='lightgray', alpha=0.65, edgecolor='dimgray', linewidth=0.25, shade=True)
                if valid_vox.numel() > 0:
                    c = valid_vox[:, :3]
                    h = valid_vox[:, 3:]
                    x_all.extend([c[:, 0] - h[:, 0], c[:, 0] + h[:, 0]])
                    y_all.extend([c[:, 1] - h[:, 1], c[:, 1] + h[:, 1]])
                    z_all.extend([c[:, 2] - h[:, 2], c[:, 2] + h[:, 2]])

            # balls: 球形障碍物 (cx, cy, cz, r)
            if hasattr(env, 'balls') and env.balls.numel() > 0:
                balls = env.balls[0].detach().cpu().numpy()
                for bx, by, bz, br in balls:
                    draw_sphere(ax, float(bx), float(by), float(bz), float(br), color='royalblue', alpha=0.72, res=14)
                    x_all.extend([torch.tensor([bx - br]), torch.tensor([bx + br])])
                    y_all.extend([torch.tensor([by - br]), torch.tensor([by + br])])
                    z_all.extend([torch.tensor([bz - br]), torch.tensor([bz + br])])

            # cyl: 竖直圆柱障碍物 (cx, cy, r), 沿 z 方向
            z0_vis, z1_vis = -0.2, 2.2
            if len(z_all) > 0:
                z_stack = torch.cat(z_all)
                z0_vis = min(z0_vis, float(z_stack.min().item()) - 0.1)
                z1_vis = max(z1_vis, float(z_stack.max().item()) + 0.1)
            if hasattr(env, 'cyl') and env.cyl.numel() > 0:
                cyl = env.cyl[0].detach().cpu().numpy()
                for cx, cy, cr in cyl:
                    draw_cylinder_z(ax, float(cx), float(cy), float(cr), z0_vis, z1_vis, color='teal', alpha=0.72)
                    x_all.extend([torch.tensor([cx - cr]), torch.tensor([cx + cr])])
                    y_all.extend([torch.tensor([cy - cr]), torch.tensor([cy + cr])])

            # cyl_h: 水平圆柱障碍物 (cx, cz, r), 沿 y 方向
            y0_vis, y1_vis = -9.5, 9.5
            if len(y_all) > 0:
                y_stack = torch.cat(y_all)
                y0_vis = min(y0_vis, float(y_stack.min().item()) - 0.1)
                y1_vis = max(y1_vis, float(y_stack.max().item()) + 0.1)
            if hasattr(env, 'cyl_h') and env.cyl_h.numel() > 0:
                cyl_h = env.cyl_h[0].detach().cpu().numpy()
                for cx, cz, cr in cyl_h:
                    draw_cylinder_y(ax, float(cx), float(cz), float(cr), y0_vis, y1_vis, color='darkorange', alpha=0.74)
                    x_all.extend([torch.tensor([cx - cr]), torch.tensor([cx + cr])])
                    z_all.extend([torch.tensor([cz - cr]), torch.tensor([cz + cr])])

            # 起点、终点、目标点
            ax.scatter([traj_xyz[0, 0]], [traj_xyz[0, 1]], [traj_xyz[0, 2]], c='green', s=45, marker='o', label='Start')
            ax.scatter([traj_xyz[-1, 0]], [traj_xyz[-1, 1]], [traj_xyz[-1, 2]], c='black', s=45, marker='x', label='End')
            if hasattr(env, 'p_target'):
                target_xyz = env.p_target[idx].detach().cpu()
                ax.scatter([float(target_xyz[0])], [float(target_xyz[1])], [float(target_xyz[2])],
                           c='red', s=70, marker='*', label='Goal')
                x_all.append(target_xyz[0:1])
                y_all.append(target_xyz[1:2])
                z_all.append(target_xyz[2:3])

            x_cat = torch.cat(x_all)
            y_cat = torch.cat(y_all)
            z_cat = torch.cat(z_all)
            ax.set_xlim(float(x_cat.min().item()) - 0.5, float(x_cat.max().item()) + 0.5)
            ax.set_ylim(float(y_cat.min().item()) - 0.5, float(y_cat.max().item()) + 0.5)
            ax.set_zlim(float(z_cat.min().item()) - 0.2, float(z_cat.max().item()) + 0.2)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.view_init(elev=28, azim=38)
            ax.legend(loc='upper left')
            ax.set_title(f"Iter {i} 3D Trajectory & Obstacles")
            plt.close(fig_map)

            interactive_html = os.path.join(video_dir, f'trajectory3d_iter_{i+1:06d}.html')
            R_cpu = R_history[:, idx].cpu()  # [T, 3, 3] 姿态矩阵

            # 计算A*规划路径用于可视化
            astar_path_viz = None
            astar_paths_sampled = []
            try:
                # 使用轨迹起点和目标点进行A*规划
                goal_pos = env.p_target[idx].detach().cpu().numpy()

                # 构建占用栅格
                global_planner.build_occupancy_grid(env, batch_idx=idx)

                # 起点到终点的路径
                start_pos = p_history[0, idx].detach().cpu().numpy()
                astar_path_raw = global_planner.plan_astar(start_pos, goal_pos)
                if astar_path_raw is not None:
                    astar_path_viz = global_planner.smooth_path(astar_path_raw, window_size=5)

                # 采样所有轨迹点的A*路径 (每隔一定步数采样)
                T_total = p_history.shape[0]
                sample_interval = max(1, T_total // 20)  # 最多约20条采样路径
                for t in range(0, T_total, sample_interval):
                    sample_pos = p_history[t, idx].detach().cpu().numpy()
                    path_raw = global_planner.plan_astar(sample_pos, goal_pos)
                    if path_raw is not None:
                        path_smooth = global_planner.smooth_path(path_raw, window_size=5)
                        astar_paths_sampled.append((t, path_smooth))

            except Exception as e:
                print(f"[Warning] A* path visualization failed: {e}")
                astar_path_viz = None
                astar_paths_sampled = []

            if save_interactive_3d_html(interactive_html, env, p_cpu, v_cpu, R_cpu=R_cpu, idx=idx,
                                        astar_path=astar_path_viz, astar_paths_sampled=astar_paths_sampled):
                writer.add_text('Trajectory/Interactive3D_HTML', interactive_html, i + 1)
            
            # ============================== 3. 速度时序图 (Vx,Vy,Vz,Speed) ==============================
            # 显示无人机三个轴方向的速度分量以及合成速度随时间的变化
            # 用于观察：速度是否平稳，是否达到目标速度，是否存在制动现象(接近目标时减速)
            fig_v, ax = plt.subplots()
            v_cpu = v_history[:, idx].cpu()
            ax.plot(v_cpu[:, 0], label='vx'); ax.plot(v_cpu[:, 1], label='vy'); ax.plot(v_cpu[:, 2], label='vz')
            ax.plot(v_cpu.norm(dim=-1), label='speed', linestyle='--')
            ax.legend(); ax.set_title(f"Iter {i} Velocity (Time Series)")
            writer.add_figure('Trajectory/Velocity_Series', fig_v, i + 1)
            plt.close(fig_v)

            # ============================== 3.1 姿态时序图 (Roll/Pitch/Yaw, 度) ==============================
            # 显示无人机姿态角(欧拉角)随时间的变化
            # 用于观察：无人机倾斜程度，是否有过度倾斜，转向动作是否平稳
            fig_rpy, ax = plt.subplots()
            rpy_cpu = rpy_history[:, idx].cpu()
            ax.plot(rpy_cpu[:, 0], label='roll(deg)')
            ax.plot(rpy_cpu[:, 1], label='pitch(deg)')
            ax.plot(rpy_cpu[:, 2], label='yaw(deg)')
            ax.legend(); ax.set_title(f"Iter {i} Attitude RPY (Time Series)")
            writer.add_figure('Trajectory/Attitude_RPY_Series', fig_rpy, i + 1)
            plt.close(fig_rpy)

            # ============================== 3.2 控制加速度时序图 (ax,ay,az) ==============================
            # 显示WorkNet网络输出的加速度命令随时间的变化(已映射到真实物理量)
            # 用于观察：控制输入是否饱和，是否平滑，是否存在过大的突跳或振荡
            fig_act, ax = plt.subplots()
            act_cpu = real_act_history[:, idx].cpu()
            ax.plot(act_cpu[:, 0], label='ax_cmd')
            ax.plot(act_cpu[:, 1], label='ay_cmd')
            ax.plot(act_cpu[:, 2], label='az_cmd')
            ax.plot(act_cpu.norm(dim=-1), label='|a_cmd|', linestyle='--')
            ax.legend(); ax.set_title(f"Iter {i} Control Accel Cmd (Time Series)")
            writer.add_figure('Trajectory/Control_Accel_Cmd_Series', fig_act, i + 1)
            plt.close(fig_act)

            # 4. [新增] 权重时序变化图 - 验证 Step-wise 效果
            fig_w, ax = plt.subplots()
            w_cpu = effective_weights_seq[:, idx, :].cpu() # [T, 4] 实际用于加权损失的权重
            labels = ['Speed', 'Dir', 'Avoid', 'Expl']
            for wi in range(4):
                ax.plot(w_cpu[:, wi], label=labels[wi])
            ax.legend(); ax.set_title(f"Iter {i} Weights Profile (Per Step, Effective)")
            writer.add_figure('Debug/Weights_StepWise', fig_w, i + 1)
            plt.close(fig_w)

            # ============================== 4.1 权重精确值记录 ==============================
            # 与轨迹同步记录的权重快照，用于分析权重的动态变化规律
            # 与每25次迭代记录一次的Weights/*不同，这里是在每300步(保存轨迹时)记录一次，便于与轨迹直接对应
            writer.add_scalar('Weights_Snapshot/0_Speed', avg_weights[0], i + 1)
            writer.add_scalar('Weights_Snapshot/1_Direction', avg_weights[1], i + 1)
            writer.add_scalar('Weights_Snapshot/2_Avoidance', avg_weights[2], i + 1)
            writer.add_scalar('Weights_Snapshot/3_Exploration', avg_weights[3], i + 1)
            writer.add_scalar('Weights_Snapshot/Entropy', weight_entropy, i + 1)
            # 权重分布统计(与轨迹同步)。用于精确对应时刻的权重分布
            writer.add_scalar('Weights_Snapshot/Raw_Min', weights_seq.min(), i + 1)
            writer.add_scalar('Weights_Snapshot/Raw_Max', weights_seq.max(), i + 1)
            writer.add_scalar('Weights_Snapshot/Raw_Mean', weights_seq.mean(), i + 1)

            # ============================== 5. 深度图视频序列 ==============================
            # 深度相机的原始输入序列转换为彩色视频，便于可视化学习过程中的感知能力
            # 用于观察：深度估计是否准确，是否能正确检测障碍物距离
            if len(depth_history) > 0:
                depth_stack = torch.stack(depth_history).float()  # [T, H, W], meters
                # 使用逆深度增强近处障碍可见性，并做分位数拉伸避免整段几乎同值导致全黑
                inv_depth = 3.0 / depth_stack.clamp(0.3, 24.0) - 0.6  # 与网络输入一致的尺度
                p2 = torch.quantile(inv_depth, 0.02)
                p98 = torch.quantile(inv_depth, 0.98)
                inv_norm = ((inv_depth - p2) / (p98 - p2 + 1e-6)).clamp(0.0, 1.0)  # [T, H, W]

                # 转为 RGB 彩色帧，避免灰度写视频时编码器/播放器显示发黑
                cmap_np = plt.get_cmap('magma')
                inv_np = inv_norm.cpu().numpy()  # [T, H, W], [0,1]
                frames = []
                for _k in range(inv_np.shape[0]):
                    rgb = (cmap_np(inv_np[_k])[..., :3] * 255.0).astype('uint8')  # [H, W, 3]
                    frames.append(rgb)

                # fallback 给 TensorBoard 帧日志使用灰度uint8
                depth_uint8 = (inv_norm * 255).to(torch.uint8)
                mp4_path = os.path.join(video_dir, f'depth_iter_{i+1:06d}.mp4')
                gif_path = os.path.join(video_dir, f'depth_iter_{i+1:06d}.gif')
                if imageio is not None:
                    try:
                        imageio.mimsave(mp4_path, frames, fps=15, macro_block_size=None)
                    except Exception:
                        imageio.mimsave(gif_path, frames, format='GIF', fps=15)
                else:
                    # imageio 不可用时退化为逐帧图像记录到 TensorBoard
                    for _fi, _frame in enumerate(depth_uint8):
                        writer.add_image(f'Video/Depth_Frame/{_fi:03d}', _frame.unsqueeze(0), i + 1)

print(f"Training Finished. Artifacts in: {save_dir}")
