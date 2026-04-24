#8.1
#制作四种独立地图,修复势场01


import argparse
import atexit
import math
from collections import defaultdict, deque
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
    from potential_map_utils import (
        PLANNER_DRONE_RADIUS as POTENTIAL_MAP_DRONE_RADIUS,
        PotentialMapCache,
        query_potential_guidance,
    )
except (ModuleNotFoundError, ImportError):
    PotentialMapCache = None
    query_potential_guidance = None
    POTENTIAL_MAP_DRONE_RADIUS = 0.13
from env import probe_update_state_vec_common_upstream
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

def safe_normalize(x, dim=-1, eps=1e-6):
    return F.normalize(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dim=dim, eps=eps)


def safe_l2_norm(x, dim=-1, keepdim=False, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim) + eps)


def sanitize_tensor(x, nan=0.0, posinf=1e3, neginf=-1e3):
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def extract_depth_geometry_features(depth, near_threshold=1.5):
    """
    从原始深度图提取显式几何/风险统计特征。

    输入:
        depth: [B, H, W]
    输出:
        geom_feat: [B, 19]
    """
    d = depth.clamp(0.3, 24.0)
    B, H, W = d.shape

    w1 = max(1, W // 3)
    w2 = max(w1 + 1, (2 * W) // 3)
    w2 = min(w2, W - 1) if W > 2 else w2

    h1 = max(1, H // 3)
    h2 = max(h1 + 1, (2 * H) // 3)
    h2 = min(h2, H - 1) if H > 2 else h2

    left = d[:, :, :w1]
    center = d[:, :, w1:w2]
    right = d[:, :, w2:]

    upper = d[:, :h1, :]
    middle = d[:, h1:h2, :]
    lower = d[:, h2:, :]

    def _mean_min_ratio(region):
        flat = region.reshape(B, -1)
        mean_v = flat.mean(dim=-1)
        min_v = flat.min(dim=-1).values
        near_ratio = (flat < near_threshold).float().mean(dim=-1)
        return mean_v, min_v, near_ratio

    def _mean_ratio(region):
        flat = region.reshape(B, -1)
        mean_v = flat.mean(dim=-1)
        near_ratio = (flat < near_threshold).float().mean(dim=-1)
        return mean_v, near_ratio

    l_mean, l_min, l_ratio = _mean_min_ratio(left)
    c_mean, c_min, c_ratio = _mean_min_ratio(center)
    r_mean, r_min, r_ratio = _mean_min_ratio(right)

    u_mean, u_ratio = _mean_ratio(upper)
    m_mean, m_ratio = _mean_ratio(middle)
    lo_mean, lo_ratio = _mean_ratio(lower)

    flat_all = d.reshape(B, -1)
    g_mean = flat_all.mean(dim=-1)
    g_std = flat_all.std(dim=-1, unbiased=False)
    lr_diff = l_mean - r_mean
    center_vs_side = c_mean - 0.5 * (l_mean + r_mean)

    geom_feat = torch.stack([
        l_mean, l_min, l_ratio,
        c_mean, c_min, c_ratio,
        r_mean, r_min, r_ratio,
        u_mean, u_ratio,
        m_mean, m_ratio,
        lo_mean, lo_ratio,
        g_mean, g_std, lr_diff, center_vs_side,
    ], dim=-1)
    return sanitize_tensor(geom_feat, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)


def extract_progress_features(p_history_list, v_history_list, dist_obj_history_list, p_target, window=8):
    """
    构造近期进展/卡住摘要特征。

    输出维度: [B, 8]
    - progress_to_goal
    - disp_k
    - speed_mean_k
    - collision_depth_mean_k
    - stuck_score
    - progress_efficiency
    - tortuosity
    - heading_align_improvement
    """
    if len(p_history_list) == 0:
        B = p_target.shape[0]
        return torch.zeros((B, 8), device=p_target.device, dtype=p_target.dtype)

    p_now = p_history_list[-1]
    device = p_now.device
    dtype = p_now.dtype

    k = max(1, min(int(window), len(p_history_list)))
    p_prev = p_history_list[-k]

    dist_now = safe_l2_norm(p_target - p_now, dim=-1)
    dist_prev = safe_l2_norm(p_target - p_prev, dim=-1)
    progress_to_goal = dist_prev - dist_now

    disp_k = safe_l2_norm(p_now - p_prev, dim=-1)

    v_tail = torch.stack(v_history_list[-k:], dim=0)
    speed_mean_k = safe_l2_norm(v_tail, dim=-1).mean(dim=0)

    if len(dist_obj_history_list) > 0:
        dist_tail = torch.stack(dist_obj_history_list[-k:], dim=0)
        depth_tail = F.relu(-dist_tail)
        # dist_tail 可能是 [k, B] 或 [k, sub_div, B]，统一压缩到 [B]
        while depth_tail.dim() > 1:
            depth_tail = depth_tail.mean(dim=0)
        collision_depth_mean_k = depth_tail
    else:
        collision_depth_mean_k = torch.zeros_like(progress_to_goal)

    stuck_score = F.softplus((0.3 - disp_k) * 10.0)
    # 效率比: 跑了多少净位移是否真正转化为接近目标
    progress_efficiency = progress_to_goal / (disp_k + 1e-6)

    # 曲折度: 窗口路径长度 / 窗口净位移，越大表示越绕/打转
    p_tail = torch.stack(p_history_list[-k:], dim=0)  # [k, B, 3]
    if k > 1:
        path_len_k = safe_l2_norm(p_tail[1:] - p_tail[:-1], dim=-1).sum(dim=0)
    else:
        path_len_k = torch.zeros_like(disp_k)
    tortuosity = path_len_k / (disp_k + 1e-6)

    v_now = v_history_list[-1]
    target_dir_now = safe_normalize(p_target - p_now, dim=-1)
    v_dir_now = safe_normalize(v_now, dim=-1)
    heading_align_now = (v_dir_now * target_dir_now).sum(dim=-1)

    if len(v_history_list) >= k:
        v_prev = v_history_list[-k]
        target_dir_prev = safe_normalize(p_target - p_prev, dim=-1)
        v_dir_prev = safe_normalize(v_prev, dim=-1)
        heading_align_prev = (v_dir_prev * target_dir_prev).sum(dim=-1)
        heading_align_improvement = heading_align_now - heading_align_prev
    else:
        heading_align_improvement = torch.zeros_like(heading_align_now)

    progress_feat = torch.stack([
        progress_to_goal,
        disp_k,
        speed_mean_k,
        collision_depth_mean_k,
        stuck_score,
        progress_efficiency,
        tortuosity,
        heading_align_improvement,
    ], dim=-1)

    progress_feat = sanitize_tensor(progress_feat, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
    return progress_feat.to(device=device, dtype=dtype)


@torch.no_grad()
def sanitize_module_(module, clamp_value=10.0):
    for p in module.parameters():
        p.data = sanitize_tensor(p.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value).clamp(-clamp_value, clamp_value)

########### 1. 参数配置 ##########
parser = argparse.ArgumentParser()
parser.add_argument('--resume_worker', default="", help='Path to pretrained worker model')
parser.add_argument('--resume_lgn', default="", help='Path to pretrained lgn model')
parser.add_argument('--resume_norm', default="", help='Path to pretrained normalization stats')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_iters', type=int, default=5000000)

# [优化策略参数]
parser.add_argument('--lgn_steps', type=int, default=1)
parser.add_argument('--worker_steps', type=int, default=5)

# 基础物理参数
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--scene_scale', type=float, default=0.5,#調節環境大小
                    help='Global scene size scale for obstacle field extent and spawn area')
parser.add_argument('--obstacle_count_scale', type=float, default=0.3,#調節障礙物數量
                    help='Global multiplier for obstacle counts (balls/voxels/cylinders)')
parser.add_argument('--easy_density_scale', type=float, default=2.0,
                    help='Density multiplier for easy-region obstacle generation')
parser.add_argument('--hard_density_scale', type=float, default=1.0,
                    help='Density multiplier for hard-region obstacle generation')
parser.add_argument('--soft_speed_limit_softness', type=float, default=0.05,#物理軟限速的平滑度（越小越接近硬截斷，越大越平滑）。
                    help='Softness for physical speed cap in env._apply_speed_limit (smaller = harder cap)')
parser.add_argument('--max_speed_ceiling', type=float, default=15.0,#env.max_speed 的上限值（軟限速的速度天花板）。
                    help='Upper bound of env max_speed used by soft speed limiter')
parser.add_argument('--hard_vpred_clip', type=float, default=20.0,#env.run 中 v_pred 的硬截斷閾值（原本固定 20）。
                    help='Hard clip magnitude for v_pred in env.run')
parser.add_argument('--hard_speed_clip', type=float, default=30.0,#env.run 中 v_free/self.v 的硬截斷閾值（原本固定 30）。
                    help='Hard clip magnitude for velocity tensors (v_free/self.v) in env.run')
parser.add_argument('--start_goal_plane_y_abs', type=float, default=25,#調節起點和終點的位置
                    help='Start/goal planes are set to +Y and -Y using this absolute value')
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--lgn_timesteps', type=int, default=80,
                    help='Rollout steps used in LGN phase; smaller value reduces 2nd-order gradient memory')
parser.add_argument('--exploration_time_window', type=int, default=150,
                    help='Look-back gap for exploration overlap loss; effective window is auto-clipped to keep valid long-range pairs')
parser.add_argument('--detach_interval', type=int, default=12,
                    help='Detach temporal memory every N steps to limit graph depth (<=0 disables)')
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--goal_radius', type=float, default=0.5,
                    help='Episode terminates when all drones are within this radius of their goal')
parser.add_argument('--maze_update_interval', type=int, default=50,
                    help='Regenerate maze every N iterations; drone-only reset in between for stable LGN signal')

# Transformer memory参数
parser.add_argument('--worker_max_seq_len', type=int, default=32)
parser.add_argument('--lgn_max_seq_len', type=int, default=32)

# 环境Flag
parser.add_argument('--single', default=True, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
# [开关1] U 型局部最优陷阱地图开关（默认: 关闭）
# - 默认行为：不传任何参数时 include_u_local_optimum=False。
# - 显式开启：--include_u_local_optimum
# - 关闭陷阱：--no_include_u_local_optimum（地图为 hard/easy/easy 三块随机重排，且不再包含 U 区）
# - 说明：这两个参数写在同一行时，以最后一个为准（argparse store_true/store_false 同目标变量）。
parser.add_argument('--include_u_local_optimum', dest='include_u_local_optimum', action='store_true',
                    help='Include U-shaped local-optimum trap region in three-zone map')
parser.add_argument('--no_include_u_local_optimum', dest='include_u_local_optimum', action='store_false',
                    help='Disable U-shaped trap region and use shuffled hard/easy/easy region order')
parser.set_defaults(include_u_local_optimum=False)
# [开关1.1] 两分区紧凑地图开关（默认: 开启）
# - 默认行为：不传任何参数时 compact_two_zone_map=True，使用紧凑两分区布局。
# - 显式开启：--compact_two_zone_map（仅保留 easy/hard 两种地图，Y 向尺寸缩小，起终点平面随之调整）
# - 显式关闭：--no_compact_two_zone_map
# - 说明：与 include_u_local_optimum 共存时，开启紧凑两分区会优先使用两分区布局（不再包含 U 区）。
parser.add_argument('--compact_two_zone_map', dest='compact_two_zone_map', action='store_true',
                    help='Use compact two-zone map (easy+hard only), with smaller map and adjusted start/goal planes')
parser.add_argument('--no_compact_two_zone_map', dest='compact_two_zone_map', action='store_false',
                    help='Use default map layout (current behavior)')
parser.set_defaults(compact_two_zone_map=True)
parser.add_argument('--unified_four_maps', dest='unified_four_maps', action='store_true',
                    help='Use unified single-type map generation cycling through easy/hard/u-min/hairpin')
parser.add_argument('--no_unified_four_maps', dest='unified_four_maps', action='store_false',
                    help='Disable unified four-map generation and use legacy multi-region layout')
parser.set_defaults(unified_four_maps=True)
parser.add_argument('--map_type', type=str, default='cycle',
                    choices=['cycle', 'easy', 'hard', 'u-min', 'u_min', 'hairpin'],
                    help='Force a single unified map type; cycle rotates through all four types')
# [统一四地图调度参数]
# 仅在 unified_four_maps=True 且 map_type=cycle 时生效。
# 参数分两类（四种地图都可独立设置）：
# 1) 生成开关参数（是否生成该类型）：
#    unified_map_<type>_enable
#    - True: 该类型会被纳入训练地图池（会生成）
#    - False: 该类型不生成
# 2) 训练使用数量参数（该类型每轮“打包”用多少张）：
#    unified_map_<type>_count
#    - 含义：该类型连续使用多少次 reset（可理解为连续多少张同类型地图）
#    - 例如 easy_count=3：会连续用 3 张 easy，再切到下一个随机类型块
# 类型块切换顺序由 CUDA 随机打乱（实现见 env_multi.py）。
# 如果 map_type 被强制为单一类型（easy/hard/u-min/hairpin），以下 enable/count 不参与调度。

# easy 组
parser.add_argument('--unified_map_easy_enable', dest='unified_map_easy_enable', action='store_true',
                    help='[生成开关] 是否生成 easy 地图类型')
parser.add_argument('--no_unified_map_easy_enable', dest='unified_map_easy_enable', action='store_false',
                    help='[生成开关] 不生成 easy 地图类型')
parser.add_argument('--unified_map_easy_count', type=int, default=1,
                    help='[训练数量] easy 类型每轮连续使用多少张（多少次 reset）')

# hard 组
parser.add_argument('--unified_map_hard_enable', dest='unified_map_hard_enable', action='store_true',
                    help='[生成开关] 是否生成 hard 地图类型')
parser.add_argument('--no_unified_map_hard_enable', dest='unified_map_hard_enable', action='store_false',
                    help='[生成开关] 不生成 hard 地图类型')
parser.add_argument('--unified_map_hard_count', type=int, default=1,
                    help='[训练数量] hard 类型每轮连续使用多少张（多少次 reset）')

# u-min 组
parser.add_argument('--unified_map_u_min_enable', dest='unified_map_u_min_enable', action='store_true',
                    help='[生成开关] 是否生成 u-min 地图类型')
parser.add_argument('--no_unified_map_u_min_enable', dest='unified_map_u_min_enable', action='store_false',
                    help='[生成开关] 不生成 u-min 地图类型')
parser.add_argument('--unified_map_u_min_count', type=int, default=1,
                    help='[训练数量] u-min 类型每轮连续使用多少张（多少次 reset）')

# hairpin 组
parser.add_argument('--unified_map_hairpin_enable', dest='unified_map_hairpin_enable', action='store_true',
                    help='[生成开关] 是否生成 hairpin 地图类型')
parser.add_argument('--no_unified_map_hairpin_enable', dest='unified_map_hairpin_enable', action='store_false',
                    help='[生成开关] 不生成 hairpin 地图类型')
parser.add_argument('--unified_map_hairpin_count', type=int, default=1,
                    help='[训练数量] hairpin 类型每轮连续使用多少张（多少次 reset）')

parser.set_defaults(
    unified_map_easy_enable=True,
    unified_map_hard_enable=True,
    unified_map_u_min_enable=True,
    unified_map_hairpin_enable=True,
)
# [开关2] 墙壁物理反馈开关（默认: 关闭）
# - 默认行为：不传任何参数时 wall_physical_feedback=False，采用自由运动结果（当前代码行为）。
# - 开启反馈：--wall_physical_feedback（启用软接触反馈，修正穿墙/贴墙时的位置与速度）
# - 显式关闭：--no_wall_physical_feedback
# - 典型组合：
#   1) 保持当前基线：不加这两个开关（或显式 --include_u_local_optimum --no_wall_physical_feedback）
#   2) 仅去掉 U 陷阱：--no_include_u_local_optimum
#   3) 仅加墙体反馈：--wall_physical_feedback
#   4) 同时去陷阱+加反馈：--no_include_u_local_optimum --wall_physical_feedback
parser.add_argument('--wall_physical_feedback', dest='wall_physical_feedback', action='store_true',
                    help='Enable wall-contact physical feedback correction in env dynamics')
parser.add_argument('--no_wall_physical_feedback', dest='wall_physical_feedback', action='store_false',
                    help='Disable wall-contact physical feedback and use free-motion result')
parser.set_defaults(wall_physical_feedback=False)

# 学习率
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--lgn_lr', type=float, default=2e-4)
parser.add_argument('--inner_lr', type=float, default=5e-4,
                    help='Inner loop LR for differentiable worker update in LGN phase')
parser.add_argument('--inner_steps', type=int, default=1,
                    help='Number of differentiable inner SGD steps (unrolled bilevel)')
parser.add_argument('--inner_grad_clip', type=float, default=10.0,
                    help='Global-norm cap for differentiable inner gradients to stabilize second-order chain')
parser.add_argument('--exp_name', type=str, default="default", help="Extra tag for experiment")

# 避障/碰撞超参
parser.add_argument('--avoid_safe_margin', type=float, default=0.35,
                    help='Proxy avoidance rises smoothly inside this clearance to walls')
parser.add_argument('--lgn_output_temperature', type=float, default=1.0,
                    help='Compatibility arg (currently not used): only non-speed LGN weights are constrained non-negative')
parser.add_argument('--lgn_weight_floor', type=float, default=0.01,
                    help='Compatibility arg (unused): no extra floor is applied beyond non-speed non-negative constraints')
parser.add_argument('--lgn_weight_ceiling', type=float, default=100.0,
                    help='Compatibility arg (unused): no ceiling constraint is applied to LGN weights')
parser.add_argument('--speed_goal_slow_dist', type=float, default=2.5,
                    help='Distance-to-goal (m) where speed target starts linearly reducing to prevent straight-line rushing')
parser.add_argument('--meta_coll_soft_weight', type=float, default=5.0,
                    help='Soft collision term weight in meta loss')
parser.add_argument('--meta_coll_hard_weight', type=float, default=40.0,
                    help='Hard penetration-depth penalty weight in meta loss')
parser.add_argument('--meta_coll_event_weight', type=float, default=80.0,
                    help='Episode-level collision event penalty weight in meta loss')
parser.add_argument('--meta_coll_event_temp', type=float, default=80.0,
                    help='Sharpness for differentiable episode collision event penalty (sigmoid temperature)')
parser.add_argument('--meta_coll_event_threshold', type=float, default=0.01,
                    help='Penetration-depth threshold (m) where differentiable collision-event penalty turns on')
parser.add_argument('--speed_near_obs_floor', type=float, default=0.05,
                    help='Minimum speed factor near obstacles in adaptive speed target (lower = stronger braking)')
parser.add_argument('--stuck_loss_weight', type=float, default=2.0,
                    help='Weight for local displacement-based stuck penalty')
parser.add_argument('--stuck_window', type=int, default=15,
                    help='Window size for stuck detection (steps)')
parser.add_argument('--stuck_displacement_threshold', type=float, default=0.3,
                    help='Minimum displacement in window before stuck penalty activates (m)')
parser.add_argument('--collision_duration_weight', type=float, default=10.0,
                    help='Weight for collision-duration diagnostic penalty')
parser.add_argument('--meta_smooth_jerk_weight', type=float, default=0.001,
                    help='Meta loss weight for first-order action difference (jerk) smoothing')
parser.add_argument('--meta_smooth_snap_weight', type=float, default=0.0002,
                    help='Meta loss weight for second-order normalized action difference (snap) smoothing')
parser.add_argument('--meta_smooth_v_pred_weight', type=float, default=0.1,
                    help='Meta loss weight for velocity prediction error')

# 全局规划引导元损失参数
parser.add_argument('--meta_guidance_weight', type=float, default=0.5,
                    help='Weight for global guidance meta loss (path-guiding dense supervision)')
parser.add_argument('--guide_sample_count', type=int, default=10,
                    help='Number of keypoints to sample for guidance loss computation')
parser.add_argument('--guide_sample_strategy', type=str, default='random',
                    choices=['random', 'uniform', 'adaptive', 'critical'],
                    help='Sampling strategy: random/uniform/adaptive(danger+curvature)/critical(start+end+danger)')
parser.add_argument('--guide_max_accel', type=float, default=5.0,
                    help='Max acceleration for trapezoidal velocity profile (m/s^2)')
parser.add_argument('--guide_max_decel', type=float, default=6.0,
                    help='Max deceleration for trapezoidal velocity profile (m/s^2)')
parser.add_argument('--guide_dir_weight', type=float, default=0.5,
                    help='Weight for direction alignment loss within guidance loss')
parser.add_argument('--guide_speed_weight', type=float, default=0.3,
                    help='Weight for overspeed penalty within guidance loss')
parser.add_argument('--guide_lateral_weight', type=float, default=0.3,
                    help='Weight for lateral error penalty (geometric distance to planned path)')
parser.add_argument('--guide_speed_diff_weight', type=float, default=0.2,
                    help='Weight for speed difference penalty (overspeed + underspeed)')
parser.add_argument('--guide_escape_weight', type=float, default=1.0,
                    help='Weight for escape penalty on collided points')
parser.add_argument('--guide_recovery_speed_weight', type=float, default=0.15,
                    help='Extra speed damping weight on planner-invalid sampled points')
parser.add_argument('--guide_collision_threshold', type=float, default=-0.05,
                    help='Penetration threshold below which point is considered collided')
parser.add_argument('--guide_accel_weight', type=float, default=0.1,
                    help='Weight for acceleration mismatch penalty (deceleration requirement)')
parser.add_argument('--planner_resolution', type=float, default=0.15,
                    help='Resolution of the occupancy grid for A* planning (meters)')
parser.add_argument('--planner_margin', type=float, default=0.07,
                    help='Safety margin for obstacle inflation in planner (meters)')
parser.add_argument('--planner_parallel', dest='planner_parallel', action='store_true',
                    help='Enable sample-level parallel global planning with multiprocessing pool')
parser.add_argument('--no_planner_parallel', dest='planner_parallel', action='store_false',
                    help='Disable sample-level parallel global planning')
parser.set_defaults(planner_parallel=True)
parser.add_argument('--planner_workers', type=int, default=0,
                    help='Number of planner worker processes (<=0 means auto)')
parser.add_argument('--planner_pool_maxtasks', type=int, default=256,
                    help='maxtasksperchild for planner process pool to avoid long-run memory growth')
# [引导后端切换开关]
# - astar: 使用在线 A* 规划引导（原有方案）
# - dijkstra_potential: 使用离线缓存 Dijkstra 势场引导（新方案）
# 使用方法：
# - 默认不写时为 astar（在线规划）。
# - 切到势场模式：--guidance_backend dijkstra_potential
#   需要配合预计算地图目录，例如：
#   --precomputed_map_dir ../precomputed_maps_all4_custom --num_precomputed_maps 100
# - 切回 A* 模式：--guidance_backend astar
# 说明：这是统一开关，优先于旧的 use_precomputed_potential_maps/use_astar_guidance 组合语义。
parser.add_argument('--guidance_backend', type=str, default='astar',
                    choices=['astar', 'dijkstra_potential'],
                    help='Switch guidance backend between online A* and cached Dijkstra potential field')
parser.add_argument('--use_precomputed_potential_maps', default=False, action='store_true',
                    help='Use precomputed Dijkstra potential-map guidance instead of online A* planning')
parser.add_argument('--precomputed_map_dir', type=str, default='../precomputed_maps_all4_custom',
                    help='Directory containing precomputed potential cache files (*.pt)')
parser.add_argument('--num_precomputed_maps', type=int, default=0,
                    help='Max number of precomputed maps to load from precomputed_map_dir (<=0 means load all)')
# [势场查询参数]
# - trilinear: 三线性插值，点落在栅格内部时按8个角点加权，训练更平滑（推荐）
# - nearest: 最近邻查询，调试方便但梯度更离散
parser.add_argument('--potential_interpolation', type=str, default='trilinear',
                    choices=['nearest', 'trilinear'],
                    help='Interpolation mode for querying potential/vector field at continuous positions')
# [势场下降约束参数]
# ReLU(phi[t+1]-phi[t]+delta) 中的 delta。
# 取负值(如 -0.01)表示“允许极小上升噪声，但整体应下降”。
parser.add_argument('--potential_delta_margin', type=float, default=-0.01,
                    help='Delta margin in potential decrease loss: ReLU(phi[t+1]-phi[t]+delta)')
parser.add_argument('--use_astar_guidance', default=False, action='store_true',
                    help='Force legacy online A* guidance even when precomputed maps are enabled')
parser.add_argument('--diag_interval', type=int, default=100,
                    help='Print detailed DIAG logs every N iterations (<=0 disables)')
parser.add_argument('--diag_second_order', default=True, action='store_true',
                    help='Enable heavy second-order diagnostic probes (can be noisy and slow)')
parser.add_argument('--terminal_log_interval', type=int, default=500,
                    help='Update terminal progress/log text every N iterations')

args = parser.parse_args()


def _write_density_sync_file(parsed_args):
    sync_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mmgj_density_defaults.json")
    payload = {
        "easy_density_scale": float(parsed_args.easy_density_scale),
        "hard_density_scale": float(parsed_args.hard_density_scale),
        "source": "mmgj_runtime_args",
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        with open(sync_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[density-sync] warning: failed to write {sync_path}: {exc}")


_write_density_sync_file(args)

# 统一 guidance 开关生效：根据 guidance_backend 映射为旧布尔参数，保持后续逻辑兼容。
if args.guidance_backend == 'dijkstra_potential':
    args.use_precomputed_potential_maps = True
    args.use_astar_guidance = False
else:
    args.use_precomputed_potential_maps = False
    args.use_astar_guidance = True

# Planner parallel runtime config (used by guidance reference computation)
PLANNER_PARALLEL_ENABLE = bool(args.planner_parallel)
PLANNER_NUM_WORKERS = int(args.planner_workers)
PLANNER_POOL_MAXTASKS = max(1, int(args.planner_pool_maxtasks))
_PLANNER_POOL = None
_PLANNER_POOL_SIZE = 0
POTENTIAL_MAP_CACHE = None

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
          easy_density_scale=args.easy_density_scale,
          hard_density_scale=args.hard_density_scale,
          speed_limit_softness=args.soft_speed_limit_softness,
          max_speed_ceiling=args.max_speed_ceiling,
          hard_vpred_clip=args.hard_vpred_clip,
          hard_speed_clip=args.hard_speed_clip,
          start_goal_plane_y_abs=args.start_goal_plane_y_abs,
          include_u_local_optimum=args.include_u_local_optimum,
          compact_two_zone_map=args.compact_two_zone_map,
          unified_four_maps=args.unified_four_maps,
          forced_map_type=("" if args.map_type == "cycle" else args.map_type),
          unified_map_easy_enable=args.unified_map_easy_enable,
          unified_map_hard_enable=args.unified_map_hard_enable,
          unified_map_u_min_enable=args.unified_map_u_min_enable,
          unified_map_hairpin_enable=args.unified_map_hairpin_enable,
          unified_map_easy_count=args.unified_map_easy_count,
          unified_map_hard_count=args.unified_map_hard_count,
          unified_map_u_min_count=args.unified_map_u_min_count,
          unified_map_hairpin_count=args.unified_map_hairpin_count,
          wall_physical_feedback=args.wall_physical_feedback)


def _align_env_goal_planes_to_precomputed_map(map_data, env_obj, map_idx_hint=None, tol=1e-3):
    start_y_from_map = None
    goal_y_from_map = None

    if "spawn_start_y" in map_data:
        start_y_from_map = float(map_data["spawn_start_y"])
    elif "spawn_start_bounds" in map_data:
        sb = map_data["spawn_start_bounds"]
        if isinstance(sb, torch.Tensor):
            sb = sb.detach().cpu().numpy()
        sb = np.asarray(sb, dtype=np.float32).reshape(-1)
        if sb.size >= 2:
            start_y_from_map = 0.5 * float(sb[0] + sb[1])

    if "spawn_goal_y" in map_data:
        goal_y_from_map = float(map_data["spawn_goal_y"])
    elif "goal_world" in map_data:
        gw = map_data["goal_world"]
        if isinstance(gw, torch.Tensor):
            gw = gw.detach().cpu().numpy()
        gw = np.asarray(gw, dtype=np.float32).reshape(-1)
        if gw.size >= 2:
            goal_y_from_map = float(gw[1])
    elif "spawn_goal_bounds" in map_data:
        gb = map_data["spawn_goal_bounds"]
        if isinstance(gb, torch.Tensor):
            gb = gb.detach().cpu().numpy()
        gb = np.asarray(gb, dtype=np.float32).reshape(-1)
        if gb.size >= 2:
            goal_y_from_map = 0.5 * float(gb[0] + gb[1])

    if start_y_from_map is None and goal_y_from_map is None:
        return

    old_start = float(getattr(env_obj, "spawn_start_y", -11.5))
    old_goal = float(getattr(env_obj, "spawn_goal_y", 11.5))
    new_start = old_start if start_y_from_map is None else float(start_y_from_map)
    new_goal = old_goal if goal_y_from_map is None else float(goal_y_from_map)

    env_obj.spawn_start_y = new_start
    env_obj.spawn_goal_y = new_goal

    if abs(new_start - old_start) > float(tol) or abs(new_goal - old_goal) > float(tol):
        idx_msg = "unknown" if map_idx_hint is None else str(int(map_idx_hint))
        print(
            f"[PotentialMap] Align start/goal planes to map_idx={idx_msg}: "
            f"start_y {old_start:.3f}->{new_start:.3f}, goal_y {old_goal:.3f}->{new_goal:.3f}"
        )

if args.use_precomputed_potential_maps:
    if PotentialMapCache is None or query_potential_guidance is None:
        raise RuntimeError("potential_map_utils.py is required for --use_precomputed_potential_maps")
    POTENTIAL_MAP_CACHE = PotentialMapCache(
        map_dir=args.precomputed_map_dir,
        num_maps=args.num_precomputed_maps,
    )
    if len(POTENTIAL_MAP_CACHE) <= 0:
        raise RuntimeError(
            f"No precomputed maps found in {args.precomputed_map_dir}. "
            f"Please run precompute_potential_maps.py first."
        )
    first_map = POTENTIAL_MAP_CACHE.get_map(0)
    _align_env_goal_planes_to_precomputed_map(first_map, env, map_idx_hint=0)
    env.reset_from_precomputed_map(first_map)
    env.current_map_idx = 0
    print(
        f"[PotentialMap] Enabled. loaded={len(POTENTIAL_MAP_CACHE)} "
        f"from {args.precomputed_map_dir}, current_map_idx=0"
    )

_upstream_probe = probe_update_state_vec_common_upstream(device)
env.update_state_vec_in_meta_path = bool(_upstream_probe["is_common_upstream"])
print(
    f"[Phase1 Probe] update_state_vec common-upstream="
    f"{env.update_state_vec_in_meta_path} (delta={_upstream_probe['delta']:.6g})"
)

state_dim = 7 if args.no_odom else 10
geom_dim = 19
progress_dim = 8

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
        geom_dim=geom_dim,
        progress_dim=progress_dim,
        max_seq_len=args.lgn_max_seq_len,
        output_temperature=args.lgn_output_temperature,
        weight_floor=args.lgn_weight_floor,
    ).to(device)
except TypeError:
    lgn = LossGenNet(state_dim=state_dim).to(device)
state_normalizer = RunningMeanStd(shape=(state_dim,)).to(device)
geom_normalizer = RunningMeanStd(shape=(geom_dim,)).to(device)
progress_normalizer = RunningMeanStd(shape=(progress_dim,)).to(device)

########## 4. 加载预训练模型 ##########
# def load_checkpoint(model, path, name):
#     if path and os.path.isfile(path):
#         print(f"Loading {name} from {path}")
#         model.load_state_dict(torch.load(path, map_location=device), strict=False)
#     elif path:
#         print(f"Warning: {name} path provided but file not found: {path}")

# load_checkpoint(worknet, args.resume_worker, "Worker")
# load_checkpoint(lgn, args.resume_lgn, "LGN")
# if args.resume_norm:
#     load_checkpoint(state_normalizer, args.resume_norm, "Norm Stats")
# elif args.resume_worker:
#     norm_path = args.resume_worker.replace('worker_', 'norm_')
#     load_checkpoint(state_normalizer, norm_path, "Auto-inferred Norm Stats")

########## 5. 优化器配置 ##########
optim_worker = AdamW(worknet.parameters(), args.lr)
optim_lgn = AdamW(lgn.parameters(), args.lgn_lr)
sched = CosineAnnealingLR(optim_worker, args.num_iters, args.lr * 0.01)

########## 6. 辅助函数 ##########
scaler_q = defaultdict(list)

def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        scaler_q[k].append(float(v))

def is_save_iter(i):
    return (i + 1) % 1000 == 0 if i >= 2000 else (i + 1) % 500 == 0


def is_save_trajectory_iter(i):
    if i < 2000:
        return i == 0 or (i + 1) % 100 == 0
    return (i + 1) % 500 == 0


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


def scale_scalar_objective(x, target_mag=10.0, eps=1e-6):
    """Rescale scalar objective by detached magnitude to avoid gradient blow-up."""
    if x is None:
        return x
    denom = x.detach().abs().clamp_min(eps)
    return x * (float(target_mag) / denom)


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


def _diag_should_log(iter_idx):
    return args.diag_interval > 0 and (iter_idx % args.diag_interval == 0)


def _diag_grad_meta(x):
    if x is None:
        return "None"
    gfn = type(x.grad_fn).__name__ if getattr(x, 'grad_fn', None) is not None else "None"
    return f"requires_grad={x.requires_grad}, is_leaf={x.is_leaf}, grad_fn={gfn}"


def _diag_tensor_finite(tag, x, iter_idx):
    if x is None:
        print(f"[DIAG iter={iter_idx}] {tag}: None")
        return
    with torch.no_grad():
        xd = x.detach()
        finite_mask = torch.isfinite(xd)
        finite_cnt = int(finite_mask.sum().item())
        total_cnt = int(xd.numel())
        nonfinite_cnt = total_cnt - finite_cnt
        if finite_cnt > 0:
            vals = xd[finite_mask]
            vmin = float(vals.min().item())
            vmax = float(vals.max().item())
        else:
            vmin = float('nan')
            vmax = float('nan')
    print(
        f"[DIAG iter={iter_idx}] {tag}: finite={finite_cnt}/{total_cnt}, "
        f"nonfinite={nonfinite_cnt}/{total_cnt}, min={vmin:.6g}, max={vmax:.6g}"
    )


def _diag_grad_tuple_to_params(tag, grad_tuple, params, iter_idx, retain_graph=True):
    params = list(params)
    total_params = len(params)
    if total_params == 0:
        print(f"[DIAG iter={iter_idx}] {tag}: None=0/0, NonZero=0/0, Norm=0.000000, NonFinite=0/0")
        return

    if grad_tuple is None:
        print(f"[DIAG iter={iter_idx}] {tag}: None={total_params}/{total_params}, NonZero=0/{total_params}, Norm=0.000000, NonFinite=0/0")
        return

    grads = [g for g in grad_tuple if g is not None]
    if len(grads) == 0:
        print(f"[DIAG iter={iter_idx}] {tag}: None={total_params}/{total_params}, NonZero=0/{total_params}, Norm=0.000000, NonFinite=0/0")
        return

    probe = None
    for g in grads:
        s = g.sum()
        probe = s if probe is None else (probe + s)

    try:
        mapped = torch.autograd.grad(
            probe,
            params,
            allow_unused=True,
            retain_graph=retain_graph,
            create_graph=False,
        )
    except Exception as e:
        print(f"[DIAG iter={iter_idx}] {tag}: grad-check failed: {e}")
        return

    none_cnt = sum(g is None for g in mapped)
    nonzero_cnt = 0
    total_sq = 0.0
    nonfinite = 0
    total_elems = 0
    for g in mapped:
        if g is None:
            continue
        gd = g.detach()
        finite_mask = torch.isfinite(gd)
        nonfinite += int((~finite_mask).sum().item())
        total_elems += gd.numel()
        if finite_mask.any():
            vals = gd[finite_mask]
            total_sq += float((vals * vals).sum().item())
            if float(vals.abs().sum().item()) > 1e-12:
                nonzero_cnt += 1

    print(
        f"[DIAG iter={iter_idx}] {tag}: None={none_cnt}/{total_params}, "
        f"NonZero={nonzero_cnt}/{total_params}, Norm={math.sqrt(total_sq):.6f}, "
        f"NonFinite={nonfinite}/{total_elems}"
    )


def _diag_output_to_params(tag, output, params, iter_idx, retain_graph=True):
    params = list(params)
    total_params = len(params)
    if total_params == 0:
        print(f"[DIAG iter={iter_idx}] {tag}: norm=0.000000, NonFinite=0/0")
        return
    try:
        grads = torch.autograd.grad(
            output,
            params,
            allow_unused=True,
            retain_graph=retain_graph,
            create_graph=False,
        )
    except Exception as e:
        print(f"[DIAG iter={iter_idx}] {tag}: grad-check failed: {e}")
        return

    norm, nonfinite, grad_elems = get_grad_norm_from_grads(grads)
    print(f"[DIAG iter={iter_idx}] {tag}: norm={norm:.6f}, NonFinite={nonfinite}/{grad_elems}")


def _diag_output_to_params_count(tag, output, params, iter_idx, retain_graph=True):
    params = list(params)
    total_params = len(params)
    if total_params == 0:
        print(f"[DIAG iter={iter_idx}] {tag}: None=0/0, NonZero=0/0")
        return
    try:
        grads = torch.autograd.grad(
            output,
            params,
            allow_unused=True,
            retain_graph=retain_graph,
            create_graph=False,
        )
    except Exception as e:
        print(f"[DIAG iter={iter_idx}] {tag}: grad-check failed: {e}")
        return

    none_cnt = sum(g is None for g in grads)
    nonzero_cnt = 0
    for g in grads:
        if g is None:
            continue
        if float(g.detach().abs().sum().item()) > 1e-12:
            nonzero_cnt += 1
    print(f"[DIAG iter={iter_idx}] {tag}: None={none_cnt}/{total_params}, NonZero={nonzero_cnt}/{total_params}")


def _grad_or_none_tuple(loss, params, create_graph=True, retain_graph=True):
    params = tuple(params)
    if not getattr(loss, "requires_grad", False):
        return tuple(None for _ in params)
    return torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        allow_unused=True,
        retain_graph=retain_graph,
    )


########## 6.1 全局规划引导元损失辅助函数 ##########

import heapq
from typing import List, Tuple, Optional, Dict

class GlobalPlanner:
    """
    3D A* 全局路径规划器

    构建占用栅格地图并使用 A* 算法规划从起点到终点的最优路径，
    然后从路径中提取参考方向、速度和加速度。
    """

    def __init__(
        self,
        resolution: float = 0.15,
        margin: float = 0.07,
        z_min: float = 0.0,
        z_max: float = 5.0,
        drone_radius: float = POTENTIAL_MAP_DRONE_RADIUS,
        device='cuda',
    ):
        """
        Args:
            resolution: 栅格分辨率 (米)
            margin: 安全边距 (米)，障碍物膨胀量
            z_min, z_max: Z轴范围
            drone_radius: 规划器使用的无人机半径 (米)
            device: 计算设备
        """
        self.resolution = resolution
        self.margin = margin
        self.z_min = z_min
        self.z_max = z_max
        self.drone_radius = max(0.0, float(drone_radius))
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
        # 以环境地图边界为准，和预构建地图口径保持一致。
        if all(hasattr(env, k) for k in ('map_x_max', 'map_y_min', 'map_y_max')):
            x_min = -0.5
            x_max = float(getattr(env, 'map_x_max')) + 0.5
            y_min = float(getattr(env, 'map_y_min')) - 1.0
            y_max = float(getattr(env, 'map_y_max')) + 1.0
        else:
            x_min, x_max = -15.0, 15.0
            y_min, y_max = -25.0, 25.0

        if hasattr(env, 'p_target') and env.p_target is not None:
            target = env.p_target[batch_idx].detach().cpu()
            y_min = min(y_min, float(target[1]) - 2.0)
            y_max = max(y_max, float(target[1]) + 2.0)

        # 栅格尺寸
        nx = int(math.ceil((x_max - x_min) / self.resolution))
        ny = int(math.ceil((y_max - y_min) / self.resolution))
        nz = int(math.ceil((self.z_max - self.z_min) / self.resolution))

        self.grid_origin = np.array([x_min, y_min, self.z_min])
        self.grid_shape = (nx, ny, nz)

        # 初始化为空闲
        self.occupancy_grid = np.zeros((nx, ny, nz), dtype=np.uint8)

        # 填充障碍物
        total_margin = float(self.margin) + float(self.drone_radius)

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
    sampled_steps = payload.get('sampled_steps', None)
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
    map_x_max = float(payload.get('map_x_max', 10.0))
    map_y_min = float(payload.get('map_y_min', -13.0))
    map_y_max = float(payload.get('map_y_max', 13.0))

    planner = GlobalPlanner(resolution=resolution, margin=margin, z_min=z_min, z_max=z_max, device='cpu')

    class _EnvShim:
        pass

    env_shim = _EnvShim()
    env_shim.voxels = torch.from_numpy(voxels_np).unsqueeze(0)
    env_shim.balls = torch.from_numpy(balls_np).unsqueeze(0)
    env_shim.cyl = torch.from_numpy(cyl_np).unsqueeze(0)
    env_shim.cyl_h = torch.from_numpy(cyl_h_np).unsqueeze(0)
    env_shim.p_target = torch.from_numpy(goal_np).reshape(1, 3)
    env_shim.map_x_max = map_x_max
    env_shim.map_y_min = map_y_min
    env_shim.map_y_max = map_y_max

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
    sampled_paths = []

    for s in range(S):
        pos_np = sampled_positions[s]
        dist = float(sampled_dist[s])

        step_t = int(sampled_steps[s]) if sampled_steps is not None else int(s)

        if dist < invalid_dist_threshold:
            sampled_paths.append((step_t, None))
            continue

        plan_total += 1
        path = planner.plan_astar(pos_np, goal_np)
        if path is None:
            sampled_paths.append((step_t, None))
            continue
        path = planner.smooth_path(path, window_size=5)
        path_np = np.asarray(path, dtype=np.float32)
        sampled_paths.append((step_t, path_np))

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
        'sampled_paths': sampled_paths,
    }


# 全局规划器实例（在迷宫更新时重新规划）
global_planner = GlobalPlanner(
    resolution=0.15,
    margin=0.07,
    z_min=0.0,
    z_max=5.0,
    drone_radius=POTENTIAL_MAP_DRONE_RADIUS,
)


def compute_guidance_reference_from_planner(env, p, v, p_target, dist_obj, planner: GlobalPlanner,
                                             max_speed=5.0, max_accel=5.0, max_decel=6.0,
                                             lookahead_dist=1.0, invalid_dist_threshold=-0.05,
                                             sampled_steps=None):
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
    sampled_astar_paths = [[] for _ in range(B)]

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
                    'sampled_steps': (
                        np.asarray(sampled_steps[:, b].detach().cpu().numpy(), dtype=np.int64)
                        if sampled_steps is not None else None
                    ),
                    'goal': np.asarray(p_target[b].detach().cpu().numpy(), dtype=np.float32),
                    'resolution': planner.resolution,
                    'margin': planner.margin,
                    'z_min': planner.z_min,
                    'z_max': planner.z_max,
                    'map_x_max': float(getattr(env, 'map_x_max', 10.0)),
                    'map_y_min': float(getattr(env, 'map_y_min', -13.0)),
                    'map_y_max': float(getattr(env, 'map_y_max', 13.0)),
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

                out_paths = out.get('sampled_paths', [])
                for item in out_paths:
                    if item is None:
                        continue
                    step_t, path_s = item
                    sampled_astar_paths[b].append((int(step_t), np.asarray(path_s, dtype=np.float32)))
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

                step_t = int(sampled_steps[s, b].item()) if sampled_steps is not None else int(s)

                if dist < invalid_dist_threshold:
                    sampled_astar_paths[b].append((step_t, None))
                    continue

                sample_plan_total += 1
                path = planner.plan_astar(pos_np, goal_np)
                if path is None:
                    sampled_astar_paths[b].append((step_t, None))
                    continue
                path = planner.smooth_path(path, window_size=5)
                path_np = np.asarray(path, dtype=np.float32)
                sampled_astar_paths[b].append((step_t, path_np))

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
        'sampled_astar_paths': sampled_astar_paths,
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


def _masked_mean(x, mask):
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (x * mask_f).sum() / denom


def compute_guidance_reference_from_potential_map(p_history, map_data, interpolation='trilinear'):
    """Query dense potential value and descending direction from precomputed map field."""
    potential_value, ref_dir, valid_mask = query_potential_guidance(
        map_data=map_data,
        points_world=p_history,
        interpolation=interpolation,
    )
    ref_dir = safe_normalize(ref_dir, dim=-1)
    return potential_value, ref_dir, valid_mask


def compute_potential_guidance_meta_loss(env, p_history, v_history, vec_to_pt, dist_obj,
                                         map_data,
                                         max_speed=5.0,
                                         dir_weight=0.5,
                                         speed_weight=0.3,
                                         lateral_weight=0.3,
                                         escape_weight=1.0,
                                         collision_threshold=-0.05,
                                         accel_weight=0.2,
                                         speed_diff_weight=0.2,
                                         recovery_speed_weight=0.15):
    """Dense guidance loss based on precomputed Dijkstra potential/vector field."""
    T, B, _ = p_history.shape

    potential_value, ref_dir, valid_mask = compute_guidance_reference_from_potential_map(
        p_history=p_history,
        map_data=map_data,
        interpolation=args.potential_interpolation,
    )

    v_dir = safe_normalize(v_history, dim=-1)
    v_speed = v_history.norm(dim=-1)

    loss_dir_align = 1.0 - (v_dir * ref_dir).sum(dim=-1)

    # Simple, stable speed reference: full speed in free/early areas, damp near walls and near low-potential zones.
    finite_mask = torch.isfinite(potential_value)
    valid_mask = valid_mask & finite_mask
    safe_pot = torch.where(valid_mask, potential_value, torch.zeros_like(potential_value))
    pot_max = safe_pot.max(dim=0, keepdim=True).values.clamp_min(1.0)
    pot_ratio = torch.clamp(safe_pot / pot_max, 0.0, 1.0)
    obstacle_factor = torch.sigmoid((dist_obj - 0.8) * 5.0)
    ref_speed = max_speed * (0.25 + 0.75 * obstacle_factor) * (0.30 + 0.70 * pot_ratio)

    loss_overspeed = F.relu(v_speed - ref_speed)
    loss_underspeed = F.relu(ref_speed - v_speed) * 0.3
    loss_speed_diff = loss_overspeed + loss_underspeed

    # Potential descent constraint: penalize local potential increase.
    loss_pot_step = torch.zeros_like(potential_value)
    if T > 1:
        step_valid = valid_mask[:-1] & valid_mask[1:]
        step_term = F.relu(potential_value[1:] - potential_value[:-1] + float(args.potential_delta_margin))
        loss_pot_step[:-1] = step_term * step_valid.float()
    else:
        step_valid = torch.zeros((0, B), dtype=torch.bool, device=p_history.device)

    # Potential absolute term: normalized potential on valid field points.
    loss_pot_abs = torch.where(valid_mask, safe_pot / (pot_max + 1e-6), torch.zeros_like(safe_pot))

    collision_mask = dist_obj < collision_threshold
    invalid_mask = (~valid_mask) & (~collision_mask)
    recovery_mask = collision_mask | invalid_mask
    valid_guidance_mask = valid_mask & (~collision_mask)

    loss_escape, loss_depth = compute_escape_penalty(
        v_history, vec_to_pt, dist_obj, recovery_mask
    )
    loss_recovery_speed = v_speed * invalid_mask.float()

    guidance_for_valid = (
        dir_weight * loss_dir_align
        + speed_weight * loss_overspeed
        + speed_diff_weight * loss_underspeed
        + lateral_weight * loss_pot_abs
        + accel_weight * loss_pot_step
    )
    guidance_for_recovery = (
        escape_weight * (loss_escape + loss_depth)
        + recovery_speed_weight * loss_recovery_speed
    )

    guidance_loss_per_point = torch.where(valid_guidance_mask, guidance_for_valid, guidance_for_recovery)
    guidance_loss = guidance_loss_per_point.mean()

    potential_decrease = torch.tensor(0.0, device=p_history.device, dtype=p_history.dtype)
    if T > 1:
        raw_dec = potential_value[:-1] - potential_value[1:]
        potential_decrease = _masked_mean(raw_dec, step_valid)

    field_dir_align = _masked_mean(1.0 - loss_dir_align, valid_guidance_mask)

    loss_components = {
        'dir_align': loss_dir_align.mean(),
        'speed_diff': loss_speed_diff.mean(),
        'overspeed': loss_overspeed.mean(),
        'underspeed': loss_underspeed.mean(),
        'potential_abs': loss_pot_abs.mean(),
        'potential_step_penalty': loss_pot_step.mean(),
        # compatibility aliases
        'lateral_error': loss_pot_abs.mean(),
        'accel_mismatch': loss_pot_step.mean(),
        'escape': loss_escape.mean(),
        'depth': loss_depth.mean(),
        'recovery_speed': loss_recovery_speed.mean(),
        'valid_ratio': valid_mask.float().mean(),
        'invalid_ratio': invalid_mask.float().mean(),
        'collision_ratio': collision_mask.float().mean(),
        'sample_count': float(T),
        'avg_curvature': 0.0,
        'avg_path_progress': 0.0,
        'avg_lateral_error': loss_pot_abs.mean().item(),
        'max_lateral_error': loss_pot_abs.max().item(),
        'potential_valid_ratio': valid_mask.float().mean(),
        # compatibility alias
        'planner_success_ratio': valid_mask.float().mean().item(),
        'avg_ref_speed': ref_speed.mean().item(),
        'sampled_astar_paths': [],
        'potential_mean': _masked_mean(safe_pot, valid_mask),
        'potential_decrease': potential_decrease,
        'field_dir_align': field_dir_align,
    }
    return guidance_loss, loss_components


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
    # New default path: dense potential-field guidance from precomputed map cache.
    if args.use_precomputed_potential_maps and not args.use_astar_guidance:
        if POTENTIAL_MAP_CACHE is None:
            raise RuntimeError("Precomputed potential map mode is enabled but POTENTIAL_MAP_CACHE is not initialized")
        map_idx = int(getattr(env, 'current_map_idx', 0))
        map_data = POTENTIAL_MAP_CACHE.get_map(map_idx)
        return compute_potential_guidance_meta_loss(
            env=env,
            p_history=p_history,
            v_history=v_history,
            vec_to_pt=vec_to_pt,
            dist_obj=dist_obj,
            map_data=map_data,
            max_speed=max_speed,
            dir_weight=dir_weight,
            speed_weight=speed_weight,
            lateral_weight=lateral_weight,
            escape_weight=escape_weight,
            collision_threshold=collision_threshold,
            accel_weight=accel_weight,
            speed_diff_weight=speed_diff_weight,
            recovery_speed_weight=recovery_speed_weight,
        )

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
        sampled_steps=sample_indices,
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
        'sampled_astar_paths': planner_info.get('sampled_astar_paths', []),
    }

    return guidance_loss, loss_components


@torch.no_grad()
def get_map_view_bounds(env, traj_xy, target_xy=None, pad=0.5):
    """Auto-fit map bounds for both maze-like and random obstacle layouts."""
    x_vals = [traj_xy[:, 0]]
    y_vals = [traj_xy[:, 1]]

    if target_xy is not None:
        x_vals.append(target_xy[0:1])
        y_vals.append(target_xy[1:2])

    if hasattr(env, 'voxels') and env.voxels.numel() > 0:
        walls = env.voxels[0].detach().cpu()
        c = walls[:, :2]
        h = walls[:, 3:5]
        x_vals.extend([c[:, 0] - h[:, 0], c[:, 0] + h[:, 0]])
        y_vals.extend([c[:, 1] - h[:, 1], c[:, 1] + h[:, 1]])

    x_all = torch.cat(x_vals)
    y_all = torch.cat(y_vals)

    x_min = float(x_all.min().item()) - pad
    x_max = float(x_all.max().item()) + pad
    y_min = float(y_all.min().item()) - pad
    y_max = float(y_all.max().item()) + pad

    if (x_max - x_min) < 1e-3:
        x_min -= 0.5
        x_max += 0.5
    if (y_max - y_min) < 1e-3:
        y_min -= 0.5
        y_max += 0.5
    return x_min, x_max, y_min, y_max


def build_potential_xy_debug_figure(map_data, z_world, stride=5):
    """Build XY heatmap + vector field slice from cached potential map for quick debugging."""
    if map_data is None:
        return None

    potential = map_data.get('potential', None)
    occupancy = map_data.get('occupancy', None)
    guide_dir = map_data.get('guide_dir', None)
    origin = map_data.get('grid_origin', None)
    resolution = float(map_data.get('resolution', 0.3))

    if potential is None or occupancy is None or guide_dir is None or origin is None:
        return None

    if isinstance(potential, torch.Tensor):
        potential = potential.detach().cpu().numpy()
    if isinstance(occupancy, torch.Tensor):
        occupancy = occupancy.detach().cpu().numpy()
    if isinstance(guide_dir, torch.Tensor):
        guide_dir = guide_dir.detach().cpu().numpy()
    if isinstance(origin, torch.Tensor):
        origin = origin.detach().cpu().numpy()

    nx, ny, nz = potential.shape
    zi = int(round((float(z_world) - float(origin[2])) / max(resolution, 1e-6)))
    zi = max(0, min(nz - 1, zi))

    pot_xy = potential[:, :, zi]
    occ_xy = occupancy[:, :, zi] > 0
    dir_xy = guide_dir[:, :, zi, :2]

    finite_mask = np.isfinite(pot_xy)
    valid_mask = finite_mask & (~occ_xy)
    if valid_mask.sum() <= 0:
        return None

    # Keep image orientation intuitive: x-axis horizontal, y-axis vertical.
    pot_show = pot_xy.T.copy()
    pot_show[~np.isfinite(pot_show)] = np.nan

    x_min = float(origin[0])
    x_max = x_min + nx * resolution
    y_min = float(origin[1])
    y_max = y_min + ny * resolution

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        pot_show,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap='viridis',
        aspect='auto',
    )
    fig.colorbar(im, ax=ax, label='Potential')

    # Overlay obstacle mask.
    occ_show = np.where(occ_xy.T, 1.0, np.nan)
    ax.imshow(
        occ_show,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap='gray',
        alpha=0.35,
        aspect='auto',
    )

    # Sparse vector field arrows.
    sx = max(1, int(stride))
    xs = []
    ys = []
    us = []
    vs = []
    for ix in range(0, nx, sx):
        for iy in range(0, ny, sx):
            if not valid_mask[ix, iy]:
                continue
            dv = dir_xy[ix, iy]
            dn = float(np.linalg.norm(dv))
            if dn <= 1e-6:
                continue
            px = x_min + (ix + 0.5) * resolution
            py = y_min + (iy + 0.5) * resolution
            xs.append(px)
            ys.append(py)
            us.append(float(dv[0] / dn))
            vs.append(float(dv[1] / dn))

    if len(xs) > 0:
        ax.quiver(xs, ys, us, vs, color='white', alpha=0.85, scale=28, width=0.002)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Potential Field XY Slice (z={z_world:.2f} m, grid_z={zi})')
    return fig


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


def save_interactive_3d_html(html_path, env, p_cpu, v_cpu, R_cpu=None, idx=0, axis_len=0.3, axis_step=5,
                             astar_path=None, astar_paths_sampled=None,
                             potential_map_data=None, show_potential_overlay=False):
    """保存交互式3D轨迹HTML，带有无人机姿态坐标系和A*全局引导轨迹

    Args:
        R_cpu: 姿态矩阵 [T, 3, 3]，如果提供则绘制坐标系
        axis_len: 坐标轴长度(米)
        axis_step: 每隔多少个时间步绘制一次坐标系
        potential_map_data: 势场缓存数据（map_XXX.pt 反序列化对象）
        show_potential_overlay: 是否在 HTML 中叠加势场切片
    """
    if go is None:
        return False

    traj_xyz = p_cpu.numpy()
    speed_cpu = v_cpu.norm(dim=-1).numpy()
    fig = go.Figure()

    traj_hover_data = np.column_stack((np.arange(traj_xyz.shape[0], dtype=np.float32), speed_cpu.astype(np.float32)))

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
        name='Trajectory',
        customdata=traj_hover_data,
        hovertemplate=(
            't=%{customdata[0]:.0f}<br>'
            'x=%{x:.2f}<br>'
            'y=%{y:.2f}<br>'
            'z=%{z:.2f}<br>'
            'speed=%{customdata[1]:.2f} m/s'
            '<extra></extra>'
        )
    ))

    # 势场后端时，在 HTML 中叠加一个 XY 势场切片层。
    potential_overlay_ok = False
    potential_overlay_msg = ""
    if show_potential_overlay and potential_map_data is not None:
        try:
            pot = potential_map_data.get('potential', None)
            occ = potential_map_data.get('occupancy', None)
            origin = potential_map_data.get('grid_origin', None)
            resolution = float(potential_map_data.get('resolution', 0.3))

            if isinstance(pot, torch.Tensor):
                pot = pot.detach().cpu().numpy()
            if isinstance(occ, torch.Tensor):
                occ = occ.detach().cpu().numpy()
            if isinstance(origin, torch.Tensor):
                origin = origin.detach().cpu().numpy()

            if pot is not None and occ is not None and origin is not None and pot.ndim == 3:
                nx, ny, nz = pot.shape
                # 先尝试轨迹中位高度对应切片；若切片无有效点，自动选择有效点最多的切片。
                z_world = float(np.median(traj_xyz[:, 2]))
                zi_guess = int(round((z_world - float(origin[2])) / max(resolution, 1e-6)))
                zi_guess = max(0, min(nz - 1, zi_guess))

                best_zi = zi_guess
                best_valid_cnt = -1
                for zi in range(nz):
                    pot_xy_i = pot[:, :, zi]
                    occ_xy_i = occ[:, :, zi] > 0
                    valid_i = np.isfinite(pot_xy_i) & (~occ_xy_i)
                    cnt_i = int(np.count_nonzero(valid_i))
                    if cnt_i > best_valid_cnt:
                        best_valid_cnt = cnt_i
                        best_zi = zi

                zi = zi_guess
                pot_xy = pot[:, :, zi].copy()
                occ_xy = occ[:, :, zi] > 0
                finite = np.isfinite(pot_xy)
                valid = finite & (~occ_xy)

                if int(np.count_nonzero(valid)) <= 0 and best_valid_cnt > 0:
                    zi = best_zi
                    pot_xy = pot[:, :, zi].copy()
                    occ_xy = occ[:, :, zi] > 0
                    finite = np.isfinite(pot_xy)
                    valid = finite & (~occ_xy)

                if np.any(valid):
                    pot_valid = pot_xy[valid]
                    pmin = float(np.min(pot_valid))
                    pmax = float(np.max(pot_valid))
                    denom = max(1e-6, pmax - pmin)
                    pot_norm = (pot_xy - pmin) / denom

                    z_world_slice = float(origin[2]) + (float(zi) + 0.5) * resolution
                    z_layer = np.full_like(pot_xy, z_world_slice - 0.08, dtype=np.float32)
                    z_layer[~valid] = np.nan
                    pot_norm[~valid] = np.nan

                    x_vec = float(origin[0]) + (np.arange(nx, dtype=np.float32) + 0.5) * resolution
                    y_vec = float(origin[1]) + (np.arange(ny, dtype=np.float32) + 0.5) * resolution
                    X, Y = np.meshgrid(x_vec, y_vec, indexing='ij')

                    fig.add_trace(go.Surface(
                        x=X,
                        y=Y,
                        z=z_layer,
                        surfacecolor=pot_norm,
                        colorscale='Viridis',
                        cmin=0.0,
                        cmax=1.0,
                        opacity=0.62,
                        showscale=True,
                        colorbar=dict(title='Potential (norm)'),
                        name='Potential Slice',
                        hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<br>Potential=%{surfacecolor:.3f}<extra></extra>'
                    ))
                    potential_overlay_ok = True
                    if zi != zi_guess:
                        potential_overlay_msg = (
                            f"Potential overlay: switched z-slice {zi_guess} -> {zi} "
                            f"(valid={int(np.count_nonzero(valid))})."
                        )
                else:
                    # 兜底：当所有 XY 切片都难以直接显示时，渲染稀疏势场点云。
                    finite_all = np.isfinite(pot) & (~(occ > 0))
                    idxs = np.argwhere(finite_all)
                    if idxs.shape[0] > 0:
                        max_pts = 5000
                        if idxs.shape[0] > max_pts:
                            pick = np.random.choice(idxs.shape[0], size=max_pts, replace=False)
                            idxs = idxs[pick]
                        x_pts = float(origin[0]) + (idxs[:, 0].astype(np.float32) + 0.5) * resolution
                        y_pts = float(origin[1]) + (idxs[:, 1].astype(np.float32) + 0.5) * resolution
                        z_pts = float(origin[2]) + (idxs[:, 2].astype(np.float32) + 0.5) * resolution
                        p_pts = pot[idxs[:, 0], idxs[:, 1], idxs[:, 2]].astype(np.float32)
                        pmin = float(np.min(p_pts))
                        pmax = float(np.max(p_pts))
                        pnorm = (p_pts - pmin) / max(1e-6, pmax - pmin)
                        fig.add_trace(go.Scatter3d(
                            x=x_pts,
                            y=y_pts,
                            z=z_pts,
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=pnorm,
                                colorscale='Viridis',
                                opacity=0.42,
                                colorbar=dict(title='Potential (norm)')
                            ),
                            name='Potential Cloud',
                            hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>'
                        ))
                        potential_overlay_ok = True
                        potential_overlay_msg = "Potential overlay: XY slice empty, fallback to sparse 3D cloud."
                    else:
                        potential_overlay_msg = "Potential overlay: no finite free-space potential values."
        except Exception as e:
            potential_overlay_msg = f"Potential overlay error: {e}"
            print(f"[HTML Potential Overlay] {potential_overlay_msg}")
    elif show_potential_overlay and potential_map_data is None:
        potential_overlay_msg = "Potential overlay requested but potential_map_data is None."

    if show_potential_overlay and (not potential_overlay_ok) and potential_overlay_msg:
        fig.add_trace(go.Scatter3d(
            x=[traj_xyz[0, 0]],
            y=[traj_xyz[0, 1]],
            z=[traj_xyz[0, 2]],
            mode='text',
            text=[potential_overlay_msg],
            textposition='top left',
            textfont=dict(color='crimson', size=11),
            showlegend=False,
            name='Potential Overlay Status',
        ))

    # 绘制全局A*引导轨迹（起点到终点）
    if astar_path is not None and len(astar_path) > 1:
        fig.add_trace(go.Scatter3d(
            x=astar_path[:, 0], y=astar_path[:, 1], z=astar_path[:, 2],
            mode='lines+markers',
            marker=dict(size=2, color='gold', symbol='diamond'),
            line=dict(color='orange', width=4, dash='dash'),
            name='A* Global Path'
        ))

    # 绘制采样点对应的所有A*轨迹（直接复用已计算结果，不额外规划）
    if astar_paths_sampled is not None and len(astar_paths_sampled) > 0:
        normalized_paths = []
        for item in astar_paths_sampled:
            if item is None:
                continue
            if isinstance(item, (tuple, list)) and len(item) == 2:
                sample_t, path = item
            else:
                # Fallback for legacy/malformed records: treat as path-only entry.
                sample_t, path = 0, item

            path_arr = None
            if path is not None:
                try:
                    if isinstance(path, torch.Tensor):
                        path_arr = path.detach().cpu().numpy()
                    else:
                        path_arr = np.asarray(path)
                    if path_arr.ndim == 1 and path_arr.size == 3:
                        path_arr = path_arr.reshape(1, 3)
                    if not (path_arr.ndim == 2 and path_arr.shape[1] == 3):
                        path_arr = None
                except Exception:
                    path_arr = None

            normalized_paths.append((sample_t, path_arr))

        num_paths = len(normalized_paths)
        for path_idx, (sample_t, path) in enumerate(normalized_paths):
            color_ratio = path_idx / max(num_paths - 1, 1)
            r = int(100 + 100 * color_ratio)
            g = int(150 * (1 - color_ratio))
            b = int(200 + 55 * color_ratio)
            color = f'rgb({r},{g},{b})'

            if path is not None and len(path) >= 2:
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

            # 标记采样锚点，便于核对“路径从采样点出发”
            try:
                sample_t_int = int(np.asarray(sample_t).reshape(-1)[0])
            except Exception:
                sample_t_int = 0
            t_idx = int(max(0, min(sample_t_int, len(traj_xyz) - 1)))
            anchor = traj_xyz[t_idx]
            fig.add_trace(go.Scatter3d(
                x=[anchor[0]], y=[anchor[1]], z=[anchor[2]],
                mode='markers',
                marker=dict(size=4, color='orange', symbol='cross'),
                showlegend=(path_idx == 0),
                name='A* Sample Anchors' if path_idx == 0 else None,
                legendgroup='astar_sampled_anchor',
                hovertemplate=f'anchor t={sample_t}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>'
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

    z0_vis, z1_vis = 0.0, 5.0
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

    if astar_path is not None and len(astar_path) > 0:
        x_vals.append(astar_path[:, 0])
        y_vals.append(astar_path[:, 1])
        z_vals.append(astar_path[:, 2])

    if astar_paths_sampled is not None:
        for item in astar_paths_sampled:
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                continue
            _, path = item
            if path is None:
                continue
            try:
                path_arr = path.detach().cpu().numpy() if isinstance(path, torch.Tensor) else np.asarray(path)
                if path_arr.ndim == 1 and path_arr.size == 3:
                    path_arr = path_arr.reshape(1, 3)
                if path_arr.ndim == 2 and path_arr.shape[1] == 3 and path_arr.shape[0] > 0:
                    x_vals.append(path_arr[:, 0])
                    y_vals.append(path_arr[:, 1])
                    z_vals.append(path_arr[:, 2])
            except Exception:
                continue

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


@torch.no_grad()
def get_collision_wall_patches(points_xyz, walls, drone_radius, segment_len=0.45, contact_eps=0.02):
    if points_xyz.numel() == 0 or walls.numel() == 0:
        return []

    wall_mask = (
        (walls[:, 2] >= 0.1)
        & (walls[:, 2] <= 1.9)
        & (walls[:, 5] > 0.5)
    )
    walls = walls[wall_mask]
    if walls.numel() == 0:
        return []

    centers = walls[:, :3]
    half = walls[:, 3:]
    wall_min = centers - half
    wall_max = centers + half
    axis_is_x = half[:, 0] <= half[:, 1]

    points_expanded = points_xyz.unsqueeze(1)
    nearest = torch.minimum(torch.maximum(points_expanded, wall_min.unsqueeze(0)), wall_max.unsqueeze(0))
    clearance = (nearest - points_expanded).norm(dim=-1) - float(drone_radius)
    contact_steps = torch.nonzero(clearance.min(dim=1).values <= contact_eps, as_tuple=False).flatten().tolist()

    wall_intervals = defaultdict(list)
    for step_idx in contact_steps:
        wall_idx = int(clearance[step_idx].argmin().item())
        if float(clearance[step_idx, wall_idx].item()) > contact_eps:
            continue

        point = points_xyz[step_idx]
        center = centers[wall_idx]
        wall_half = half[wall_idx]

        if bool(axis_is_x[wall_idx]):
            tangent_min = float(center[1] - wall_half[1])
            tangent_max = float(center[1] + wall_half[1])
            tangent_center = min(max(float(point[1]), tangent_min), tangent_max)
        else:
            tangent_min = float(center[0] - wall_half[0])
            tangent_max = float(center[0] + wall_half[0])
            tangent_center = min(max(float(point[0]), tangent_min), tangent_max)

        seg_half = min(0.5 * float(segment_len), 0.5 * (tangent_max - tangent_min))
        seg_start = max(tangent_min, tangent_center - seg_half)
        seg_end = min(tangent_max, tangent_center + seg_half)
        if seg_end - seg_start <= 1e-4:
            continue
        wall_intervals[wall_idx].append((seg_start, seg_end))

    patches = []
    for wall_idx, intervals in wall_intervals.items():
        center = centers[wall_idx]
        wall_half = half[wall_idx]
        for seg_start, seg_end in merge_intervals(intervals, min_gap=0.02):
            if bool(axis_is_x[wall_idx]):
                patches.append({
                    'xy': (float(center[0] - wall_half[0]), seg_start),
                    'width': float(2.0 * wall_half[0]),
                    'height': float(seg_end - seg_start),
                })
            else:
                patches.append({
                    'xy': (seg_start, float(center[1] - wall_half[1])),
                    'width': float(seg_end - seg_start),
                    'height': float(2.0 * wall_half[1]),
                })
    return patches

def compute_overlap_loss_per_step(p_history, sigma=0.5, time_window=10):
    """
    Step-wise 重叠损失计算
    返回: [Batch, Time] (注意: 调用处需要permute)
    """
    p_history = p_history.permute(1, 0, 2) # [B, T, 3]
    n_batch, n_points, n_dims = p_history.shape

    if n_points < time_window + 1:
        return torch.zeros((n_batch, n_points), device=p_history.device)

    # 二阶梯度稳定性：
    # 避免使用 cdist 的 sqrt 链路（在高阶导下更容易出现数值尖峰），
    # 直接构造平方距离并代入高斯核，语义与 exp(-||x-y||^2 / (2*sigma^2)) 等价。
    sigma_safe = max(float(sigma), 1e-4)
    inv_two_sigma2 = 1.0 / (2.0 * sigma_safe * sigma_safe)
    diff = p_history[:, :, None, :] - p_history[:, None, :, :]  # [B, T, T, 3]
    sq_dist = (diff * diff).sum(dim=-1)  # [B, T, T]
    exponent = (-sq_dist * inv_two_sigma2).clamp(min=-60.0, max=0.0)
    overlap_energy = torch.exp(exponent)

    indices = torch.arange(n_points, device=p_history.device)
    time_diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    mask = (time_diff > time_window).float()

    # 计算每个时间步的能量总和
    energy_sum = (overlap_energy * mask.unsqueeze(0)).sum(dim=2) 
    mask_sum = mask.sum(dim=1).unsqueeze(0) + 1e-6

    # 返回 [Batch, Time]
    loss_per_step = energy_sum / mask_sum
    return loss_per_step


def compute_turn_preference_loss(v_history, speed_threshold=0.2):
    """
    转弯偏好损失：
    仅在水平面 (x-y) 上惩罚“需要改变航向时仍持续保持原方向不变”的行为。
    竖直方向 (z) 机动不计入转弯语义，避免把爬升/下降误判为转弯。

    输入:
        v_history: [T, B, 3]
    输出:
        loss_turn_seq: [T, B]
    """
    T, B, _ = v_history.shape
    device = v_history.device
    dtype = v_history.dtype

    loss_turn_seq = torch.zeros((T, B), device=device, dtype=dtype)
    if T <= 1:
        return loss_turn_seq

    # 仅使用水平速度分量，转弯定义为水平航向变化
    v_xy = v_history[..., :2]  # [T, B, 2]
    v_dir_xy = safe_normalize(v_xy, dim=-1)
    dir_consistency = (v_dir_xy[1:] * v_dir_xy[:-1]).sum(dim=-1).clamp(-1.0, 1.0)

    # 归一化到 [0,1]，水平航向越不变惩罚越大
    loss_core = 0.5 * (dir_consistency + 1.0)

    # 低水平速度时航向不稳定，不施加转弯偏好损失
    speed_now = safe_l2_norm(v_xy[1:], dim=-1)
    speed_prev = safe_l2_norm(v_xy[:-1], dim=-1)
    valid_mask = ((speed_now > speed_threshold) & (speed_prev > speed_threshold)).to(dtype)

    loss_turn_seq[1:] = loss_core * valid_mask
    return loss_turn_seq


def compute_stuck_loss(p_history, collision_depth, stuck_window=15, displacement_threshold=0.3):
    """
    计算卡住惩罚损失

    检测两种卡住状态：
    1. 局部窗口内位移过小
    2. 持续碰撞状态
    """
    T, B, _ = p_history.shape
    device = p_history.device

    loss_stuck = torch.zeros((T, B), device=device)
    if T > stuck_window:
        for t in range(stuck_window, T):
            window_start = t - stuck_window
            displacement = safe_l2_norm(p_history[t] - p_history[window_start], dim=-1)  # [B]
            loss_stuck[t] = F.softplus((displacement_threshold - displacement) * 10.0)

    in_collision = (collision_depth > 0).float()  # [T, B]
    loss_collision_duration = torch.zeros_like(in_collision)

    collision_streak = torch.zeros((B,), device=device)
    for t in range(T):
        collision_streak = collision_streak * in_collision[t] + in_collision[t]
        loss_collision_duration[t] = collision_streak * in_collision[t]

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
    v_preds_val = []
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
        v_preds_val.append(v_pred)
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

    dist_val = sanitize_tensor(safe_l2_norm(vec_val, dim=-1) - env.margin, nan=0.0, posinf=10.0, neginf=-10.0)

    collision_depth_val = F.relu(-dist_val)
    loss_stuck_val, loss_collision_duration_val, stuck_ratio = compute_stuck_loss(
        p_val, collision_depth_val,
        stuck_window=args.stuck_window,
        displacement_threshold=args.stuck_displacement_threshold,
    )

    m_pos  = safe_l2_norm(p_val[-1] - env.p_target, dim=-1).mean()
    with torch.no_grad():
        v_to_pt = torch.ones_like(dist_val)
        if dist_val.shape[0] > 1:
            v_to_pt[1:] = (-torch.diff(dist_val, 1, 0) * 135.0).clamp_min(1.0)
    m_coll = (F.softplus(dist_val.mul(-32.0)) * v_to_pt).mean()
    m_ctrl = safe_l2_norm(act_val, dim=-1).sum()
    m_jerk = act_val.diff(1, 0).mul(15.0).pow(2).sum(-1).mean()
    m_snap = (F.normalize(act_val - env.g_std, dim=-1)
              .diff(1, 0).diff(1, 0).mul(15.0 ** 2).pow(2).sum(-1).mean())
    # Meta rollout 的高度惩罚，与主训练分支保持一致
    m_height = (F.smooth_l1_loss(p_val[:, :, 2], torch.full_like(p_val[:, :, 2], 1.0), reduction='none')
               + F.softplus((p_val[:, :, 2] - 5.0) * 20.0)
               + F.softplus((0.0 - p_val[:, :, 2]) * 20.0)).mean()

    v_preds_val_tensor = torch.stack(v_preds_val)  # [T, B, 3]
    v_val = torch.stack(v_list)  # [T, B, 3]
    m_v_pred = F.mse_loss(v_preds_val_tensor, v_val.detach())
    m_stuck = loss_stuck_val.mean()

    # 全局规划引导损失：始终进入 unrolled 二阶链路
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

    meta_val = (
        m_pos
        + m_coll
        + m_height
        + args.meta_guidance_weight * m_guidance
        + args.meta_smooth_jerk_weight * m_jerk
        + args.meta_smooth_snap_weight * m_snap
        + args.meta_smooth_v_pred_weight * m_v_pred
        + args.stuck_loss_weight * m_stuck
    )
    return meta_val, m_pos, m_coll, m_ctrl

########## 7. 训练主循环 ##########

# 使用命令行参数重新初始化全局规划器
global_planner = GlobalPlanner(
    resolution=args.planner_resolution,
    margin=args.planner_margin,
    z_min=0.0,
    z_max=float(getattr(env, 'map_z_max', 5.0)),
    drone_radius=POTENTIAL_MAP_DRONE_RADIUS,
    device=device
)
print(
    "[GlobalPlanner] Initialized with "
    f"resolution={args.planner_resolution}m, margin={args.planner_margin}m, "
    f"drone_radius={POTENTIAL_MAP_DRONE_RADIUS}m, z=[{global_planner.z_min}, {global_planner.z_max}]"
)

current_precomputed_map_idx = -1

terminal_log_interval = max(1, int(args.terminal_log_interval))
pbar = tqdm(range(args.num_iters), ncols=120, miniters=terminal_log_interval)
B = args.batch_size
cycle_len = args.lgn_steps + args.worker_steps
maze_update_counter = 0
meta_lgn_grad_window = deque(maxlen=20)

state_normalizer.train()

for i in pbar:
    term_log_now = ((i + 1) % terminal_log_interval == 0)
    cycle_pos = i % cycle_len
    train_lgn_phase = cycle_pos < args.lgn_steps
    phase_str = f"LGN ({cycle_pos+1}/{args.lgn_steps})" if train_lgn_phase else f"Work ({cycle_pos-args.lgn_steps+1}/{args.worker_steps})"
    env.set_meta_differentiable_mode(train_lgn_phase)

    if args.use_precomputed_potential_maps:
        if maze_update_counter % args.maze_update_interval == 0:
            current_precomputed_map_idx = (current_precomputed_map_idx + 1) % len(POTENTIAL_MAP_CACHE)
            env.current_map_idx = current_precomputed_map_idx
            map_data_cur = POTENTIAL_MAP_CACHE.get_map(current_precomputed_map_idx)
            _align_env_goal_planes_to_precomputed_map(map_data_cur, env, map_idx_hint=current_precomputed_map_idx)
            env.reset_from_precomputed_map(map_data_cur)
        else:
            env.reset_drone_only()
    else:
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
    v_preds = []
    act_for_diag = None
    dist_obj_history = []
    geom_feat_last = None
    progress_feat_last = None

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
        vec_curr = env.find_vec_to_nearest_pt()
        vec_to_pt_history.append(vec_curr)
        dist_obj_curr = safe_l2_norm(vec_curr, dim=-1) - env.margin
        dist_obj_history.append(dist_obj_curr)
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

        geom_feat_raw = extract_depth_geometry_features(depth)
        geom_feat = geom_normalizer(geom_feat_raw, update=True)
        geom_feat = sanitize_tensor(geom_feat, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        progress_feat_raw = extract_progress_features(
            p_history_list=p_history,
            v_history_list=v_history,
            dist_obj_history_list=dist_obj_history,
            p_target=env.p_target,
            window=8,
        )
        progress_feat = progress_normalizer(progress_feat_raw, update=True)
        progress_feat = sanitize_tensor(progress_feat, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        geom_feat_last = geom_feat
        progress_feat_last = progress_feat

        # LGN Forward（第0维 speed trend 偏好允许有符号，其余维度保持非负）
        current_weights, lgn_hx = lgn(x_pooled, state_tensor, geom_feat, progress_feat, lgn_hx)

        if t == 0 and _diag_should_log(i):
            first_lgn_weight = current_weights[0, 0] if current_weights.numel() > 0 else None
            print(
                f"[DIAG iter={i} t=0] current_weights={_diag_grad_meta(current_weights)}, "
                f"lgn_hx={_diag_grad_meta(lgn_hx)}, "
                f"first_lgn_weight={_diag_grad_meta(first_lgn_weight)}"
            )
        trajectory_lgn_weights.append(current_weights)

        # Worker Forward
        act, _, h = worknet(x_pooled, state_tensor, h)
        act = sanitize_tensor(act, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        act_for_diag = act
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)
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
    weights_seq = torch.stack(trajectory_lgn_weights) # [T, B, 5]
    if _diag_should_log(i):
        print(f"[DIAG iter={i}] weights_seq: {_diag_grad_meta(weights_seq)}")
    rpy_history = torch.stack(rpy_history) # [T, B, 3]
    R_history = torch.stack(R_history)     # [T, B, 3, 3]
    real_act_history = torch.stack(real_act_history) # [T, B, 3]

    vec_to_pt = torch.stack(vec_to_pt_history)
    if vec_to_pt.dim() == 4: vec_to_pt = vec_to_pt.mean(1)
    
    # 1. 计算各项 Raw Loss (保留 [T, B] 维度用于 Step-wise 加权)

    # 碰撞距离
    dist_obj = safe_l2_norm(vec_to_pt, dim=-1) - env.margin  # [T, B]

    # 纯速度变化趋势项（有符号）:
    # speed_delta > 0 表示加速, speed_delta < 0 表示减速
    speed_actual = safe_l2_norm(v_history, dim=-1)  # [T, B]
    loss_speed_seq = torch.zeros_like(speed_actual)
    if speed_actual.shape[0] > 1:
        speed_delta = speed_actual[1:] - speed_actual[:-1]  # [T-1, B]
        loss_speed_seq[1:] = speed_delta

    target_dir = safe_normalize(env.p_target - p_history, dim=-1)
    v_dir = safe_normalize(v_history, dim=-1)
    loss_direction_seq = (1.0 - (v_dir * target_dir).sum(-1))

    with torch.no_grad():
        v_to_pt = torch.ones_like(dist_obj)
        if dist_obj.shape[0] > 1:
            v_to_pt[1:] = (-torch.diff(dist_obj, 1, 0) * 135.0).clamp_min(1.0)

    collision_depth = F.relu(-dist_obj)
    loss_avoidance_seq = v_to_pt * (1.0 - dist_obj).relu().pow(2)
    loss_collision_seq = F.softplus(dist_obj.mul(-32.0)) * v_to_pt

    # 注意: compute_overlap_loss_per_step 返回 [B, T], 需要 permute 成 [T, B]
    loss_exploration_seq = compute_overlap_loss_per_step(
        p_history, sigma=1.0, time_window=int(args.exploration_time_window)
    ).permute(1, 0)
    loss_turn_seq = compute_turn_preference_loss(v_history, speed_threshold=0.2)

    loss_stuck_seq, loss_collision_duration_seq, stuck_ratio = compute_stuck_loss(
        p_history, collision_depth,
        stuck_window=args.stuck_window,
        displacement_threshold=args.stuck_displacement_threshold,
    )
    loss_stuck_total = args.stuck_loss_weight * loss_stuck_seq.mean()
    actual_T = p_history.shape[0]

    # 高度约束损失 (固定权重, 不经LGN控制)
    z_pos = p_history[:, :, 2]  # [T, B]
    z_target = 1.0  # 迷宫中层高度
    z_min, z_max = 0.0, 5.0
    loss_height_seq = (F.smooth_l1_loss(z_pos, torch.full_like(z_pos, z_target), reduction='none')
                       + F.softplus((z_pos - z_max) * 20.0)
                       + F.softplus((z_min - z_pos) * 20.0))

    # 权重策略：speed trend 分量保留符号，其余分量保持非负（由 LGN 输出层约束）
    weights_seq_raw = weights_seq
    if _diag_should_log(i):
        print(f"[DIAG iter={i}] weights_seq_raw: requires_grad={weights_seq_raw.requires_grad}, grad_fn={type(weights_seq_raw.grad_fn).__name__ if weights_seq_raw.grad_fn else 'None'}")
        print(
            f"[DIAG iter={i}] loss_raw requires_grad: speed={loss_speed_seq.requires_grad}, "
            f"dir={loss_direction_seq.requires_grad}, avoid={loss_avoidance_seq.requires_grad}, "
            f"expl={loss_exploration_seq.requires_grad}, turn={loss_turn_seq.requires_grad}"
        )
    speed_coeff = weights_seq_raw[:, :, 0]  # signed
    dir_coeff = weights_seq_raw[:, :, 1]    # non-negative
    avoid_coeff = weights_seq_raw[:, :, 2]  # non-negative
    expl_coeff = weights_seq_raw[:, :, 3]   # non-negative
    turn_coeff = weights_seq_raw[:, :, 4]   # non-negative
    effective_weights_seq = torch.stack(
        [speed_coeff, dir_coeff, avoid_coeff, expl_coeff, turn_coeff],
        dim=-1
    )
    if _diag_should_log(i):
        print(
            f"[DIAG iter={i}] effective_weights: requires_grad={effective_weights_seq.requires_grad}, "
            f"grad_fn={type(effective_weights_seq.grad_fn).__name__ if effective_weights_seq.grad_fn else 'None'}"
        )

    # 2. Step-wise 加权 (Broadcasting: [T, B] * [T, B])
    weighted_loss_map = (
        speed_coeff * loss_speed_seq +
        dir_coeff * loss_direction_seq +
        avoid_coeff * loss_avoidance_seq +
        expl_coeff * loss_exploration_seq +
        turn_coeff * loss_turn_seq
    )

    # 3. 最终 Proxy Loss
    proxy_loss = weighted_loss_map.mean()
    if _diag_should_log(i):
        print(
            f"[DIAG iter={i}] weighted_loss_map: requires_grad={weighted_loss_map.requires_grad}, "
            f"grad_fn={type(weighted_loss_map.grad_fn).__name__ if weighted_loss_map.grad_fn else 'None'}"
        )
        print(
            f"[DIAG iter={i}] proxy_loss: requires_grad={proxy_loss.requires_grad}, "
            f"grad_fn={type(proxy_loss.grad_fn).__name__ if proxy_loss.grad_fn else 'None'}"
        )
        _diag_tensor_finite("act_for_diag", act_for_diag, i)
        _diag_tensor_finite("real_act_history", real_act_history, i)
        _diag_tensor_finite("p_history", p_history, i)
        _diag_tensor_finite("v_history", v_history, i)
        _diag_tensor_finite("a_history", a_history, i)
        _diag_tensor_finite("vec_to_pt", vec_to_pt, i)
        _diag_tensor_finite("dist_obj", dist_obj, i)
        _diag_tensor_finite("loss_speed_seq", loss_speed_seq, i)
        _diag_tensor_finite("loss_direction_seq", loss_direction_seq, i)
        _diag_tensor_finite("loss_avoidance_seq", loss_avoidance_seq, i)
        _diag_tensor_finite("loss_exploration_seq", loss_exploration_seq, i)
        _diag_tensor_finite("loss_turn_seq", loss_turn_seq, i)
        _diag_tensor_finite("weights_seq_raw", weights_seq_raw, i)
        _diag_tensor_finite("weighted_loss_map", weighted_loss_map, i)
        _diag_tensor_finite("proxy_loss", proxy_loss, i)

    # --- Meta Loss Components ---
    loss_meta_pos = safe_l2_norm(p_history[-1] - env.p_target, dim=-1).mean()
    loss_meta_coll = loss_collision_seq.mean()
    loss_meta_ctrl = safe_l2_norm(act_buffer, dim=-1).sum()
    loss_meta_jerk = act_buffer.diff(1, 0).mul(15.0).pow(2).sum(-1).mean()
    loss_meta_snap = (F.normalize(act_buffer - env.g_std, dim=-1)
                      .diff(1, 0).diff(1, 0).mul(15.0 ** 2).pow(2).sum(-1).mean())
    loss_meta_height = loss_height_seq.mean()
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
            'sampled_astar_paths': [],
        }

    # 训练目标仅使用位置/碰撞/高度/引导; 控制项仅用于日志监控
    loss_meta_stuck = loss_stuck_seq.mean()
    meta_loss = (
        loss_meta_pos +
        loss_meta_coll +
        loss_meta_height +
        args.meta_guidance_weight * loss_meta_guidance +
        args.meta_smooth_jerk_weight * loss_meta_jerk +
        args.meta_smooth_snap_weight * loss_meta_snap +
        args.meta_smooth_v_pred_weight * loss_meta_v_pred +
        args.stuck_loss_weight * loss_meta_stuck
    )
    if _diag_should_log(i):
        _diag_tensor_finite("meta_loss", meta_loss, i)
    # Keep root losses untouched for higher-order gradients; skip iteration via finite checks below.

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
    proxy_grad_turn = 0.0
    proxy_grad_speed_nonfinite = 0.0
    proxy_grad_dir_nonfinite = 0.0
    proxy_grad_avoid_nonfinite = 0.0
    proxy_grad_expl_nonfinite = 0.0
    proxy_grad_turn_nonfinite = 0.0
    proxy_grad_speed_elems = 0.0
    proxy_grad_dir_elems = 0.0
    proxy_grad_avoid_elems = 0.0
    proxy_grad_expl_elems = 0.0
    proxy_grad_turn_elems = 0.0
    lgn_grad_norm = 0.0
    lgn_grad_max = 0.0
    lgn_grad_nonfinite = 0.0
    lgn_grad_elems = 0.0
    lgn_clip_pre = 0.0
    meta_grad_window_mean = 0.0
    meta_grad_window_min = 0.0
    meta_grad_window_max = 0.0
    meta_grad_window_finite_ratio = 0.0
    meta_grad_window_nonzero_ratio = 0.0
    lgn_meta_probe_norm = 0.0
    lgn_meta_probe_nonfinite = 0.0
    lgn_meta_probe_elems = 0.0

    rollout_is_finite = bool(
        torch.isfinite(proxy_loss).all()
        and torch.isfinite(meta_loss).all()
        and torch.isfinite(weights_seq).all()
        and torch.isfinite(p_history).all()
        and torch.isfinite(v_history).all()
    )

    if not rollout_is_finite:
        if term_log_now:
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
    proxy_grad_turn, proxy_grad_turn_nonfinite, proxy_grad_turn_elems = \
        get_loss_to_worker_grad_norm(loss_turn_seq.mean(), worker_params)
    if _diag_should_log(i):
        print(
            f"[DIAG iter={i}] loss_to_worker: "
            f"speed={proxy_grad_speed:.6f} (NonFinite={proxy_grad_speed_nonfinite}/{proxy_grad_speed_elems}), "
            f"dir={proxy_grad_dir:.6f} (NonFinite={proxy_grad_dir_nonfinite}/{proxy_grad_dir_elems}), "
            f"avoid={proxy_grad_avoid:.6f} (NonFinite={proxy_grad_avoid_nonfinite}/{proxy_grad_avoid_elems}), "
            f"expl={proxy_grad_expl:.6f} (NonFinite={proxy_grad_expl_nonfinite}/{proxy_grad_expl_elems}), "
            f"turn={proxy_grad_turn:.6f} (NonFinite={proxy_grad_turn_nonfinite}/{proxy_grad_turn_elems})"
        )

    if train_lgn_phase:
        # ===== Unrolled Bilevel: 可微内循环 =====
        # Step 1: 用 proxy_loss 对 worker 做可微梯度下降
        fast_params = dict(worknet.named_parameters())
        inner_update_is_finite = True

        for _inner in range(args.inner_steps):
            inner_grads = torch.autograd.grad(
                proxy_loss, tuple(fast_params.values()),
                create_graph=True, allow_unused=True, retain_graph=True,
            )

            inner_sq_terms = [(g * g).sum() for g in inner_grads if g is not None]
            if inner_sq_terms:
                inner_grad_norm_tensor = torch.sqrt(torch.stack(inner_sq_terms).sum() + 1e-12)
                if not bool(torch.isfinite(inner_grad_norm_tensor).item()):
                    inner_update_is_finite = False
                    break
                inner_scale = (args.inner_grad_clip / (inner_grad_norm_tensor + 1e-6)).clamp(max=1.0)
                inner_grads = tuple((g * inner_scale) if g is not None else None for g in inner_grads)

            if _inner == 0 and _diag_should_log(i):
                fast_param_values = tuple(fast_params.values())

                if args.diag_second_order:
                    # Probe weighted per-term proxy components so each branch truly depends on LGN weights.
                    weighted_speed = (effective_weights_seq[:, :, 0] * loss_speed_seq).mean()
                    weighted_dir = (effective_weights_seq[:, :, 1] * loss_direction_seq).mean()
                    weighted_avoid = (effective_weights_seq[:, :, 2] * loss_avoidance_seq).mean()
                    weighted_expl = (effective_weights_seq[:, :, 3] * loss_exploration_seq).mean()
                    weighted_turn = (effective_weights_seq[:, :, 4] * loss_turn_seq).mean()

                    g_speed = _grad_or_none_tuple(weighted_speed, fast_param_values)
                    g_dir = _grad_or_none_tuple(weighted_dir, fast_param_values)
                    g_avoid = _grad_or_none_tuple(weighted_avoid, fast_param_values)
                    g_expl = _grad_or_none_tuple(weighted_expl, fast_param_values)
                    g_turn = _grad_or_none_tuple(weighted_turn, fast_param_values)

                    lgn_param_list = list(lgn.parameters())
                    _diag_grad_tuple_to_params("speed(weighted) second_order(worker_grad)->lgn", g_speed, lgn_param_list, i)
                    _diag_grad_tuple_to_params("direction(weighted) second_order(worker_grad)->lgn", g_dir, lgn_param_list, i)
                    _diag_grad_tuple_to_params("avoidance(weighted) second_order(worker_grad)->lgn", g_avoid, lgn_param_list, i)
                    _diag_grad_tuple_to_params("exploration(weighted) second_order(worker_grad)->lgn", g_expl, lgn_param_list, i)
                    _diag_grad_tuple_to_params("turn(weighted) second_order(worker_grad)->lgn", g_turn, lgn_param_list, i)
                    _diag_grad_tuple_to_params("proxy_total second_order(worker_grad)->lgn", inner_grads, lgn_param_list, i)

                    act_only_loss = (act_for_diag.pow(2).mean() if act_for_diag is not None else torch.tensor(0.0, device=device))
                    act_only_weighted = effective_weights_seq[:, :, 0].mean() * act_only_loss
                    g_act_only = torch.autograd.grad(
                        act_only_weighted, fast_param_values,
                        create_graph=True, allow_unused=True, retain_graph=True,
                    )
                    _diag_grad_tuple_to_params("act_only(weighted) second_order(worker_grad)->lgn", g_act_only, lgn_param_list, i)

                    _diag_grad_tuple_to_params("inner_grads(sum) -> lgn", inner_grads, lgn_param_list, i)
                    toy_grad = tuple((g.pow(2) if g is not None else None) for g in inner_grads)
                    _diag_grad_tuple_to_params("toy_grad(sqsum) -> lgn", toy_grad, lgn_param_list, i)

                inner_norm, inner_nonfinite, inner_elems = get_grad_norm_from_grads(inner_grads)
                print(f"[DIAG iter={i}] inner_grads finite: nonfinite={inner_nonfinite}/{inner_elems}")

            fast_params = {
                name: (p - args.inner_lr * g
                       if g is not None else p)
                for (name, p), g in zip(fast_params.items(), inner_grads)
            }

        if not inner_update_is_finite:
            if term_log_now:
                pbar.set_description(f"[{phase_str}] non-finite inner-update skipped")
            continue

        if _diag_should_log(i):
            fast_param_vals = list(fast_params.values())
            if len(fast_param_vals) > 0:
                _diag_tensor_finite("first_fast_param", fast_param_vals[0], i)
                _diag_output_to_params_count("fast_params(sum) -> lgn", sum(fp.sum() for fp in fast_param_vals), lgn.parameters(), i)

        # Step 2: 用虚拟更新后的 worker 做验证 rollout → meta_loss
        meta_loss_unrolled, meta_pos_ur, meta_coll_ur, meta_ctrl_ur = \
            unrolled_meta_rollout(env, worknet, fast_params, state_normalizer, args, B, device)
        if not torch.isfinite(meta_loss_unrolled):
            if term_log_now:
                pbar.set_description(f"[{phase_str}] non-finite unroll skipped")
            continue

        if _diag_should_log(i):
            print(
                f"[DIAG iter={i}] meta_loss_unrolled: requires_grad={meta_loss_unrolled.requires_grad}, "
                f"grad_fn={type(meta_loss_unrolled.grad_fn).__name__ if meta_loss_unrolled.grad_fn else 'None'}"
            )
            _diag_tensor_finite("meta_loss_unrolled", meta_loss_unrolled, i)
            _diag_output_to_params_count("meta_loss_unrolled -> lgn", meta_loss_unrolled, lgn.parameters(), i)
            _diag_output_to_params("meta_loss_unrolled -> fast_params", meta_loss_unrolled, fast_params.values(), i)
            _diag_output_to_params("proxy_loss -> lgn", proxy_loss, lgn.parameters(), i)
            _diag_output_to_params("meta_loss -> lgn", meta_loss, lgn.parameters(), i)

        # Step 3: 反向传播贯穿整条链路
        #   meta_loss → fast_params → ∇proxy_loss → LGN weights → LGN params
        # LGN 更新仅使用 unrolled meta loss，不再混入 proxy aux 或 fallback。
        meta_probe_grads = torch.autograd.grad(
            meta_loss_unrolled,
            tuple(lgn.parameters()),
            allow_unused=True,
            retain_graph=True,
            create_graph=False,
        )
        lgn_meta_probe_norm, lgn_meta_probe_nonfinite, lgn_meta_probe_elems = get_grad_norm_from_grads(meta_probe_grads)
        meta_grad_usable = (
            lgn_meta_probe_elems > 0
            and lgn_meta_probe_nonfinite == 0
            and lgn_meta_probe_norm > 0.0
            and math.isfinite(lgn_meta_probe_norm)
        )
        scaled_meta = scale_scalar_objective(meta_loss_unrolled)
        if not meta_grad_usable:
            if _diag_should_log(i):
                print(
                    f"[DIAG iter={i}] skip LGN step: unusable meta gradients "
                    f"(meta_probe_norm={lgn_meta_probe_norm:.6f}, "
                    f"meta_probe_nonfinite={int(lgn_meta_probe_nonfinite)}/{int(lgn_meta_probe_elems)})"
                )
            if term_log_now:
                pbar.set_description(f"[{phase_str}] meta-grad unusable, LGN step skipped")
            continue

        lgn_total = scaled_meta

        lgn_total.backward()
        lgn_grad_norm, lgn_grad_max, lgn_grad_nonfinite, lgn_grad_elems = get_grad_stats(lgn)
        if _diag_should_log(i):
            print(
                f"[DIAG iter={i}] lgn_grads: norm={lgn_grad_norm:.6f}, "
                f"nonfinite={int(lgn_grad_nonfinite)}/{int(lgn_grad_elems)}, max={lgn_grad_max:.6f}"
            )
        if math.isfinite(lgn_grad_norm):
            meta_lgn_grad_window.append(float(lgn_grad_norm))
        else:
            meta_lgn_grad_window.append(float('nan'))

        valid_meta_grad = [g for g in meta_lgn_grad_window if math.isfinite(g)]
        if valid_meta_grad:
            meta_grad_window_mean = float(sum(valid_meta_grad) / len(valid_meta_grad))
            meta_grad_window_min = float(min(valid_meta_grad))
            meta_grad_window_max = float(max(valid_meta_grad))
        meta_grad_window_finite_ratio = float(
            sum(1 for g in meta_lgn_grad_window if math.isfinite(g)) / max(1, len(meta_lgn_grad_window))
        )
        meta_grad_window_nonzero_ratio = float(
            sum(1 for g in meta_lgn_grad_window if math.isfinite(g) and g > 0.0) / max(1, len(meta_lgn_grad_window))
        )

        if _diag_should_log(i):
            print(
                f"[DIAG iter={i}] meta_loss_unrolled->LGN grad window "
                f"len={len(meta_lgn_grad_window)}, finite_ratio={meta_grad_window_finite_ratio:.3f}, "
                f"nonzero_ratio={meta_grad_window_nonzero_ratio:.3f}, mean={meta_grad_window_mean:.6g}, "
                f"min={meta_grad_window_min:.6g}, max={meta_grad_window_max:.6g}"
            )

        lgn_clip_pre = float(nn.utils.clip_grad_norm_(lgn.parameters(), 1.0).item())
        optim_lgn.step()
        sanitize_module_(lgn, clamp_value=5.0)

        lgn_update_loss = meta_loss_unrolled.detach()
    else:
        proxy_loss.backward()
        worker_grad_norm, worker_grad_max, worker_grad_nonfinite, worker_grad_elems = get_grad_stats(worknet)
        worker_clip_pre = float(nn.utils.clip_grad_norm_(worknet.parameters(), 1.0).item())
        optim_worker.step()
        sanitize_module_(worknet, clamp_value=10.0)
        sched.step()

    ###### D. Logging & Saving (Enhanced) ######
    if term_log_now:
        if train_lgn_phase:
            pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Unroll: {lgn_update_loss:.3f}")
        else:
            pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Loss: {meta_loss:.3f}")
    
    with torch.no_grad():
        success = torch.all(dist_obj > 0, 0)
        # 计算平均权重 (用于 Scalar 显示)
        avg_weights_raw = weights_seq.mean(dim=[0, 1]).cpu()  # 原始输出权重
        avg_weights = effective_weights_seq.mean(dim=[0, 1]).cpu()  # 实际使用权重
        avg_effective_weights = effective_weights_seq.mean(dim=[0, 1]).cpu()
        weight_std = effective_weights_seq.std(dim=-1).mean()
        v_norm = v_history.norm(dim=-1)
        avg_speed = v_norm.mean()
        min_speed_threshold = float(env.max_speed) * 0.7
        act_cmd_mean = real_act_history.mean(dim=(0, 1))
        act_cmd_abs_mean = real_act_history.abs().mean(dim=(0, 1))
        act_cmd_norm_mean = real_act_history.norm(dim=-1).mean()
        
        log_data = {
            # === 主要Loss ===
            'Loss/1_Proxy_Total': proxy_loss,
            'Loss/2_Meta_Total': meta_loss,

            # === [增强] 5个权重均值 (实际使用) ===
            'Weights/0_SpeedTrend': avg_weights[0],
            'Weights/1_Direction': avg_weights[1],
            'Weights/2_Avoidance': avg_weights[2],
            'Weights/3_Exploration': avg_weights[3],
            'Weights/4_Turn': avg_weights[4],
            # === 原始输出权重 ===
            'Weights_Raw/0_SpeedTrend': avg_weights_raw[0],
            'Weights_Raw/1_Direction': avg_weights_raw[1],
            'Weights_Raw/2_Avoidance': avg_weights_raw[2],
            'Weights_Raw/3_Exploration': avg_weights_raw[3],
            'Weights_Raw/4_Turn': avg_weights_raw[4],
            'Weights_Effective/0_SpeedTrend': avg_effective_weights[0],
            'Weights_Effective/1_Direction': avg_effective_weights[1],
            'Weights_Effective/2_Avoidance': avg_effective_weights[2],
            'Weights_Effective/3_Exploration': avg_effective_weights[3],
            'Weights_Effective/4_Turn': avg_effective_weights[4],

            # === [新增] 权重分布监控 ===
            'Weight_Stats/Raw_Min': weights_seq.min(),
            'Weight_Stats/Raw_Max': weights_seq.max(),
            'Weight_Stats/Raw_Mean': weights_seq.mean(),
            'Weight_Stats/Std': weight_std,

            # === [增强] Proxy Loss 原始分项 (Average over Time & Batch) ===
            'Proxy_Comp/0_SpeedTrend': loss_speed_seq.mean(),
            'Proxy_Comp/1_Direction': loss_direction_seq.mean(),
            'Proxy_Comp/2_Avoidance': loss_avoidance_seq.mean(),
            'Proxy_Comp/2_1_Collision_Depth': collision_depth.mean(),#穿入墙体深度
            'Proxy_Comp/3_Exploration': loss_exploration_seq.mean(),
            'Proxy_Comp/4_Turn': loss_turn_seq.mean(),
            'Proxy_Comp/5_Height': loss_height_seq.mean(),
            'Diagnostics/Proxy_Stuck': loss_stuck_seq.mean(),
            'Diagnostics/Proxy_Collision_Duration': loss_collision_duration_seq.mean(),
            'Diagnostics/Proxy_Stuck_Total': loss_stuck_total,
            'Stuck/Ratio': stuck_ratio,
            'Stuck/Collision_Streak_Mean': loss_collision_duration_seq.mean(),
            'Stuck/Collision_Streak_Max': loss_collision_duration_seq.max(),

            # === [增强] Meta Loss 分项 ===
            'Meta_Comp/1_Position': loss_meta_pos,
            'Meta_Comp/2_Collision': loss_meta_coll,
            'Meta_Comp/2_1_Collision_Depth': collision_depth.mean(),
            'Meta_Comp/3_Control': loss_meta_ctrl,
            'Meta_Comp/4_Height': loss_meta_height,
            'Meta_Comp/6_Stuck': loss_meta_stuck,
            'Meta_Comp/8_Smooth_Jerk': loss_meta_jerk,
            'Meta_Comp/9_Smooth_Snap': loss_meta_snap,
            'Meta_Comp/10_Smooth_V_Pred': loss_meta_v_pred,

            # === 全局规划引导损失分项 ===
            'Meta_Comp/5_Guidance': loss_meta_guidance,
            'Guidance/Dir_Align': guidance_components['dir_align'],
            'Guidance/Overspeed': guidance_components['overspeed'],
            'Guidance/Underspeed': guidance_components.get('underspeed', 0.0),
            'Guidance/Speed_Diff': guidance_components.get('speed_diff', 0.0),
            'Guidance/Escape': guidance_components['escape'],
            'Guidance/Depth': guidance_components['depth'],
            'Guidance/Recovery_Speed': guidance_components.get('recovery_speed', 0.0),
            'Guidance/Valid_Ratio': guidance_components['valid_ratio'],
            'Guidance/Valid_Guidance_Ratio': guidance_components.get('valid_guidance_ratio', 0.0),
            'Guidance/Invalid_Ratio': guidance_components.get('invalid_ratio', 0.0),
            'Guidance/Collision_Ratio': guidance_components['collision_ratio'],
            'Guidance/Valid_Mean': guidance_components.get('guidance_valid_mean', 0.0),
            'Guidance/Recovery_Mean': guidance_components.get('guidance_recovery_mean', 0.0),
            'Guidance/Boost': guidance_components.get('guidance_boost', 1.0),
            'Guidance/Sample_Count': guidance_components['sample_count'],
            'Guidance/Avg_Curvature': guidance_components.get('avg_curvature', 0.0),
            'Guidance/Avg_Path_Progress': guidance_components.get('avg_path_progress', 0.0),
            'Guidance/Avg_Ref_Speed': guidance_components.get('avg_ref_speed', 0.0),
            'Guidance/Avg_Lateral_Error': guidance_components.get('avg_lateral_error', 0.0),
            'Guidance/Max_Lateral_Error': guidance_components.get('max_lateral_error', 0.0),
            'Guidance/Field_Dir_Align': guidance_components.get('field_dir_align', 0.0),
            'Guidance/Applied_In_Current_Phase': 1.0 if train_lgn_phase else 0.0,

            # === 性能指标 ===
            'Metrics/Success_Rate': success.float().mean(),
            'Metrics/Avg_Speed': avg_speed,
            'Metrics/Speed_Below_Threshold': (avg_speed < min_speed_threshold).float(),
            'Metrics/Min_Speed': v_norm.min(),
            'Metrics/Max_Speed': v_norm.max(),
            'Metrics/Episode_Length': actual_T,
            'Metrics/Speed_Delta_Mean': loss_speed_seq.mean(),
            'Control/Accel_Cmd_Norm_Mean': act_cmd_norm_mean,
            'Control/Accel_Cmd_X_Mean': act_cmd_mean[0],
            'Control/Accel_Cmd_Y_Mean': act_cmd_mean[1],
            'Control/Accel_Cmd_Z_Mean': act_cmd_mean[2],
            'Control/Accel_Cmd_X_AbsMean': act_cmd_abs_mean[0],
            'Control/Accel_Cmd_Y_AbsMean': act_cmd_abs_mean[1],
            'Control/Accel_Cmd_Z_AbsMean': act_cmd_abs_mean[2],

            # === [对齐] 归一化统计命名（与第二脚本风格一致） ===
            'Norm/State_Mean': state_normalizer.mean[0],
            'Norm/State_Var': state_normalizer.var[0],
            'Norm/Update_Count': state_normalizer.count,
            'Norm/Geom_Mean': geom_normalizer.mean[0],
            'Norm/Geom_Var': geom_normalizer.var[0],
            'Norm/Progress_Mean': progress_normalizer.mean[0],
            'Norm/Progress_Var': progress_normalizer.var[0],

            # === [兼容] 保留旧命名 ===
            'Stats/Norm_Mean': state_normalizer.mean[0],
            'Stats/Norm_Var': state_normalizer.var[0],
            'Stats/Norm_Count': state_normalizer.count,

            # === 梯度监控（用于判断梯度爆炸） ===
            'Grad/Worker_Global_Norm': worker_grad_norm,
            'Grad/Worker_Max_Abs': worker_grad_max,
            'Grad/Worker_NonFinite_Count': worker_grad_nonfinite,
            'Grad/Worker_GradElem_Count': worker_grad_elems,
            'Grad/Worker_Clip_PreNorm': worker_clip_pre,
            'Grad/LGN_Global_Norm': lgn_grad_norm,
            'Grad/LGN_Max_Abs': lgn_grad_max,
            'Grad/LGN_NonFinite_Count': lgn_grad_nonfinite,
            'Grad/LGN_GradElem_Count': lgn_grad_elems,
            'Grad/LGN_Clip_PreNorm': lgn_clip_pre,
            'Grad/LGN_MetaWindow_Mean': meta_grad_window_mean,
            'Grad/LGN_MetaWindow_Min': meta_grad_window_min,
            'Grad/LGN_MetaWindow_Max': meta_grad_window_max,
            'Grad/LGN_MetaWindow_FiniteRatio': meta_grad_window_finite_ratio,
            'Grad/LGN_MetaWindow_NonZeroRatio': meta_grad_window_nonzero_ratio,
            'Grad/LGN_MetaProbe_Norm': lgn_meta_probe_norm,
            'Grad/LGN_MetaProbe_NonFinite_Count': lgn_meta_probe_nonfinite,
            'Grad/LGN_MetaProbe_GradElem_Count': lgn_meta_probe_elems,

            # === [新增] 五个代理损失对 Worker 梯度的 norm ===
            'Grad_ProxyWorker/0_SpeedTrend_Norm': proxy_grad_speed,
            'Grad_ProxyWorker/1_Direction_Norm': proxy_grad_dir,
            'Grad_ProxyWorker/2_Avoidance_Norm': proxy_grad_avoid,
            'Grad_ProxyWorker/3_Exploration_Norm': proxy_grad_expl,
            'Grad_ProxyWorker/4_Turn_Norm': proxy_grad_turn,
            'Grad_ProxyWorker/0_SpeedTrend_NonFinite': proxy_grad_speed_nonfinite,
            'Grad_ProxyWorker/1_Direction_NonFinite': proxy_grad_dir_nonfinite,
            'Grad_ProxyWorker/2_Avoidance_NonFinite': proxy_grad_avoid_nonfinite,
            'Grad_ProxyWorker/3_Exploration_NonFinite': proxy_grad_expl_nonfinite,
            'Grad_ProxyWorker/4_Turn_NonFinite': proxy_grad_turn_nonfinite,
            'Grad_ProxyWorker/0_SpeedTrend_GradElem': proxy_grad_speed_elems,
            'Grad_ProxyWorker/1_Direction_GradElem': proxy_grad_dir_elems,
            'Grad_ProxyWorker/2_Avoidance_GradElem': proxy_grad_avoid_elems,
            'Grad_ProxyWorker/3_Exploration_GradElem': proxy_grad_expl_elems,
            'Grad_ProxyWorker/4_Turn_GradElem': proxy_grad_turn_elems
        }

        if train_lgn_phase:
            log_data['Loss/3_LGN_Unrolled_Meta'] = lgn_update_loss
            log_data['Meta_Unrolled/1_Position'] = meta_pos_ur
            log_data['Meta_Unrolled/2_Collision'] = meta_coll_ur
            log_data['Meta_Unrolled/3_Control'] = meta_ctrl_ur

        if geom_feat_last is not None and progress_feat_last is not None:
            log_data['LGN_Input/Geom_Mean'] = geom_feat_last.mean()
            log_data['LGN_Input/Geom_Std'] = geom_feat_last.std(unbiased=False)
            log_data['LGN_Input/Geom_Norm'] = geom_feat_last.norm(dim=-1).mean()
            log_data['LGN_Input/Progress_Mean'] = progress_feat_last.mean()
            log_data['LGN_Input/Progress_Std'] = progress_feat_last.std(unbiased=False)
            log_data['LGN_Input/Progress_Norm'] = progress_feat_last.norm(dim=-1).mean()
            for feat_idx in range(min(4, geom_feat_last.shape[-1])):
                log_data[f'LGN_Input/Geom_{feat_idx}'] = geom_feat_last[:, feat_idx].mean()
            for feat_idx in range(min(4, progress_feat_last.shape[-1])):
                log_data[f'LGN_Input/Progress_{feat_idx}'] = progress_feat_last[:, feat_idx].mean()

        smooth_dict(log_data)

        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()
            writer.add_scalar('Status/Train_Mode', 1.0 if train_lgn_phase else 0.0, i + 1)
            writer.add_scalar('Status/Maze_Age', (maze_update_counter - 1) % args.maze_update_interval, i + 1)

        if is_save_iter(i):
            torch.save(worknet.state_dict(), os.path.join(save_dir, f'worker_ckpt_{i:06d}.pth'))
            torch.save(lgn.state_dict(), os.path.join(save_dir, f'lgn_ckpt_{i:06d}.pth'))
            torch.save(state_normalizer.state_dict(), os.path.join(save_dir, f'norm_ckpt_{i:06d}.pth'))

        if is_save_trajectory_iter(i):
            idx = 0
            
            # 1. 轨迹时序图 (X,Y,Z vs T)
            fig_p, ax = plt.subplots()
            p_cpu = p_history[:, idx].cpu()
            ax.plot(p_cpu[:, 0], label='x'); ax.plot(p_cpu[:, 1], label='y'); ax.plot(p_cpu[:, 2], label='z')
            ax.legend(); ax.set_title(f"Iter {i} Pos (Time Series)")
            writer.add_figure('Trajectory/Position_Series', fig_p, i + 1)
            plt.close(fig_p)

            # 2. [改造] 三维轨迹 + 障碍物显示（非俯视图）
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
            z0_vis, z1_vis = 0.0, 5.0
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

            # 直接复用本轮 guidance 中已经计算过的采样A*轨迹，不做额外A*计算
            astar_paths_all = guidance_components.get('sampled_astar_paths', [])
            astar_paths_sampled = astar_paths_all[idx] if (isinstance(astar_paths_all, list) and idx < len(astar_paths_all)) else []

            if save_interactive_3d_html(
                interactive_html, env, p_cpu, v_cpu, R_cpu=R_cpu, idx=idx,
                astar_path=None, astar_paths_sampled=astar_paths_sampled,
                potential_map_data=None,
                show_potential_overlay=False,
            ):
                writer.add_text('Trajectory/Interactive3D_HTML', interactive_html, i + 1)

            # 3. [新增] 速度时序图 (Vx,Vy,Vz,Speed)
            fig_v, ax = plt.subplots()
            v_cpu = v_history[:, idx].cpu()
            ax.plot(v_cpu[:, 0], label='vx'); ax.plot(v_cpu[:, 1], label='vy'); ax.plot(v_cpu[:, 2], label='vz')
            ax.plot(v_cpu.norm(dim=-1), label='speed', linestyle='--')
            ax.legend(); ax.set_title(f"Iter {i} Velocity (Time Series)")
            writer.add_figure('Trajectory/Velocity_Series', fig_v, i + 1)
            plt.close(fig_v)

            # 3.1 [新增] 姿态时序图 (Roll/Pitch/Yaw, deg)
            fig_rpy, ax = plt.subplots()
            rpy_cpu = rpy_history[:, idx].cpu()
            ax.plot(rpy_cpu[:, 0], label='roll(deg)')
            ax.plot(rpy_cpu[:, 1], label='pitch(deg)')
            ax.plot(rpy_cpu[:, 2], label='yaw(deg)')
            ax.legend(); ax.set_title(f"Iter {i} Attitude RPY (Time Series)")
            writer.add_figure('Trajectory/Attitude_RPY_Series', fig_rpy, i + 1)
            plt.close(fig_rpy)

            # 3.2 [新增] WorkNet映射到真实环境的控制量 (加速度) 时序图
            fig_act, ax = plt.subplots()
            act_cpu = real_act_history[:, idx].cpu()
            ax.plot(act_cpu[:, 0], label='ax_cmd')
            ax.plot(act_cpu[:, 1], label='ay_cmd')
            ax.plot(act_cpu[:, 2], label='az_cmd')
            ax.plot(act_cpu.norm(dim=-1), label='|a_cmd|', linestyle='--')
            ax.legend(); ax.set_title(f"Iter {i} Control Accel Cmd (Time Series)")
            writer.add_figure('Trajectory/Control_Accel_Cmd_Series', fig_act, i + 1)
            plt.close(fig_act)

            # 4. 权重逐时间步变化图：按分量拆分保存
            w_cpu = effective_weights_seq[:, idx, :].cpu() # [T, 5] 实际使用权重（第0维有符号，其余非负）
            labels = ['SpeedTrend', 'Direction', 'Avoidance', 'Exploration', 'Turn']
            tag_suffix = ['0_SpeedTrend', '1_Direction', '2_Avoidance', '3_Exploration', '4_Turn']
            for wi in range(5):
                fig_wi, ax = plt.subplots()
                ax.plot(w_cpu[:, wi], label=labels[wi])
                ax.legend()
                ax.set_title(f"Iter {i} Weight Profile - {labels[wi]} (Per Step)")
                writer.add_figure(f'Debug/Weights_StepWise_{tag_suffix[wi]}', fig_wi, i + 1)
                plt.close(fig_wi)

            # 4.1 [新增] 权重精确值记录（与轨迹同步，用于分析权重动态变化）
            writer.add_scalar('Weights_Snapshot/0_SpeedTrend', avg_weights[0], i + 1)
            writer.add_scalar('Weights_Snapshot/1_Direction', avg_weights[1], i + 1)
            writer.add_scalar('Weights_Snapshot/2_Avoidance', avg_weights[2], i + 1)
            writer.add_scalar('Weights_Snapshot/3_Exploration', avg_weights[3], i + 1)
            writer.add_scalar('Weights_Snapshot/4_Turn', avg_weights[4], i + 1)
            writer.add_scalar('Weights_Snapshot/Std', weight_std, i + 1)
            # 权重统计
            writer.add_scalar('Weights_Snapshot/Raw_Min', weights_seq.min(), i + 1)
            writer.add_scalar('Weights_Snapshot/Raw_Max', weights_seq.max(), i + 1)
            writer.add_scalar('Weights_Snapshot/Raw_Mean', weights_seq.mean(), i + 1)

            # 5. [新增] 深度图视频（保存到本地）
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
