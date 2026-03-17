#5.5.2  在物理反馈仿真基础  INFRA
#       + 每次新撞击单独计一次重罚
#       + 惩罚“贴墙高速擦行

import argparse
import math
from collections import defaultdict
from random import normalvariate
import os
import datetime
import json
import sys

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
import matplotlib.cm as cm
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("[SDP] flash=False, mem_efficient=False, math=True (for higher-order gradients)")

try:
    from env_maze_easy import Env
except ModuleNotFoundError:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from env_maze_easy import Env
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
parser.add_argument('--resume_worker', default="", help='Path to pretrained worker model')
parser.add_argument('--resume_lgn', default="", help='Path to pretrained lgn model')
parser.add_argument('--resume_norm', default="", help='Path to pretrained normalization stats')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=5000000)

# [优化策略参数]
parser.add_argument('--lgn_steps', type=int, default=1)
parser.add_argument('--worker_steps', type=int, default=5)

# 基础物理参数
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--lgn_timesteps', type=int, default=48,
                    help='Rollout steps used in LGN phase; smaller value reduces 2nd-order gradient memory')
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
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')

# 学习率
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lgn_lr', type=float, default=2e-4)
parser.add_argument('--inner_lr', type=float, default=1e-3,
                    help='Inner loop LR for differentiable worker update in LGN phase')
parser.add_argument('--inner_steps', type=int, default=3,
                    help='Number of differentiable inner SGD steps (unrolled bilevel)')
parser.add_argument('--exp_name', type=str, default="default", help="Extra tag for experiment")

# 避障/碰撞超参
parser.add_argument('--avoid_safe_margin', type=float, default=0.35,
                    help='Proxy avoidance rises smoothly inside this clearance to walls')
parser.add_argument('--proxy_avoid_floor', type=float, default=0.8,
                    help='Minimum avoidance weight added to LGN output in proxy loss')
parser.add_argument('--meta_coll_soft_weight', type=float, default=5.0,
                    help='Soft collision term weight in meta loss')
parser.add_argument('--meta_coll_hard_weight', type=float, default=40.0,
                    help='Hard penetration-depth penalty weight in meta loss')
parser.add_argument('--meta_coll_event_weight', type=float, default=80.0,
                    help='Episode-level collision event penalty weight in meta loss')
parser.add_argument('--meta_contact_slide_weight', type=float, default=10.0,
                    help='Penalty weight for wall-assisted sliding/turning in meta loss')
parser.add_argument('--meta_coll_incident_weight', type=float, default=120.0,
                    help='Penalty weight for each new wall-hit incident in meta loss')
parser.add_argument('--meta_wall_scrape_weight', type=float, default=12.0,
                    help='Penalty weight for high-speed wall scraping in meta loss')
parser.add_argument('--meta_coll_event_temp', type=float, default=80.0,
                    help='Sharpness for differentiable episode collision event penalty (sigmoid temperature)')
parser.add_argument('--meta_coll_event_threshold', type=float, default=0.01,
                    help='Penetration-depth threshold (m) where differentiable collision-event penalty turns on')
parser.add_argument('--proxy_contact_slide_weight', type=float, default=0.6,
                    help='Additional proxy penalty for tangential wall sliding while in contact')
parser.add_argument('--proxy_coll_incident_weight', type=float, default=4.0,
                    help='Strong penalty for each new wall-hit incident in proxy loss')
parser.add_argument('--proxy_wall_scrape_weight', type=float, default=1.0,
                    help='Penalty for sustained high-speed wall scraping in proxy loss')
parser.add_argument('--contact_tangent_damping', type=float, default=0.35,
                    help='Tangential damping applied by the environment during wall contacts')
parser.add_argument('--coll_incident_threshold', type=float, default=0.005,
                    help='Penetration threshold (m) that counts as a wall-hit incident')
parser.add_argument('--wall_scrape_speed_limit', type=float, default=1.2,
                    help='Tangential speed limit (m/s) allowed when flying very close to walls')
parser.add_argument('--wall_scrape_clearance', type=float, default=0.22,
                    help='Clearance (m) below which high tangential wall-following speed is penalized')
parser.add_argument('--speed_near_obs_floor', type=float, default=0.05,
                    help='Minimum speed factor near obstacles in adaptive speed target (lower = stronger braking)')

args = parser.parse_args()

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
          random_rotation=args.random_rotation, cam_angle=args.cam_angle)
env.contact_tangent_damping = args.contact_tangent_damping

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
    lgn = LossGenNet(state_dim=state_dim, max_seq_len=args.lgn_max_seq_len).to(device)
except TypeError:
    lgn = LossGenNet(state_dim=state_dim).to(device)
state_normalizer = RunningMeanStd(shape=(state_dim,)).to(device)

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

loss_normalizer = LossNormalizer(5)  # normalize 5 loss components to equal scale

########## 6. 辅助函数 ##########
scaler_q = defaultdict(list)

def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        scaler_q[k].append(float(v))

def is_save_iter(i):
    return (i + 1) % 10000 == 0 if i >= 2000 else (i + 1) % 500 == 0

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

def unrolled_meta_rollout(env, worknet, fast_params, state_normalizer, args, B, device):
    """
    Validation rollout with virtually-updated worker params (via functional_call).
    Computes and returns meta_loss (position + collision + control) plus components.
    LGN is NOT needed here — meta_loss is purely task-performance based.
    Reuses the same maze layout for consistent LGN signal.
    """
    # 保持同一张迷宫布局, 仅重置无人机状态用于验证rollout
    env.reset_drone_only()

    p_list, v_list = [], []
    clearance_list, contact_penetration_list, contact_slide_list = [], [], []
    act_buf = [env.act.detach()] * 2
    h_val = None

    for t in range(args.lgn_timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        depth, flow = env.render(ctl_dt)
        depth = sanitize_tensor(depth, nan=24.0, posinf=24.0, neginf=0.3)

        p_list.append(env.p)
        v_list.append(env.v)

        target_v_raw = env.p_target - env.p.detach()
        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        max_speed = torch.as_tensor(env.max_speed, device=target_v_norm.device,
                                    dtype=target_v_norm.dtype)
        target_v = (target_v_raw / (target_v_norm + 1e-6)) * torch.minimum(target_v_norm, max_speed)

        R = env.R
        state_list = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.current_clearance[:, None]]
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
        real_act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        real_act = sanitize_tensor(real_act, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        act_buf.append(real_act)

        env.run(real_act, ctl_dt, target_v_raw)
        clearance_list.append(env.last_step_clearance)
        contact_penetration_list.append(env.last_contact_penetration)
        contact_slide_list.append(env.last_contact_tangent_speed)

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
    act_val = torch.stack(act_buf)
    clearance_val = sanitize_tensor(torch.stack(clearance_list), nan=0.0, posinf=10.0, neginf=-10.0)
    contact_penetration_val = sanitize_tensor(torch.stack(contact_penetration_list), nan=0.0, posinf=10.0, neginf=0.0)
    contact_slide_val = sanitize_tensor(torch.stack(contact_slide_list), nan=0.0, posinf=30.0, neginf=0.0)

    m_pos = torch.norm(env.p - env.p_target, 2, -1).mean()
    collision_depth_val = torch.maximum(F.relu(-clearance_val), contact_penetration_val)
    contact_prob_val = torch.sigmoid((collision_depth_val - args.coll_incident_threshold) * args.meta_coll_event_temp)
    prev_contact_prob_val = torch.cat([torch.zeros_like(contact_prob_val[:1]), contact_prob_val[:-1]], dim=0)
    collision_incident_val = (contact_prob_val * (1.0 - prev_contact_prob_val)).clamp(0.0, 1.0)
    wall_scrape_excess_val = F.relu(contact_slide_val - args.wall_scrape_speed_limit)
    m_coll_soft = F.softplus(-clearance_val * 32.0).clamp(max=100.0).mean()
    m_coll_hard = collision_depth_val.pow(2).mean()
    m_coll_peak = collision_depth_val.max(dim=0).values
    m_coll_event = torch.sigmoid((m_coll_peak - args.meta_coll_event_threshold) * args.meta_coll_event_temp).mean()
    m_contact_slide = contact_slide_val.mean()
    m_coll_incident = collision_incident_val.sum(dim=0).mean()
    m_wall_scrape = wall_scrape_excess_val.pow(2).mean()
    m_coll = (args.meta_coll_soft_weight * m_coll_soft
              + args.meta_coll_hard_weight * m_coll_hard
              + args.meta_coll_event_weight * m_coll_event
              + args.meta_contact_slide_weight * m_contact_slide
              + args.meta_coll_incident_weight * m_coll_incident
              + args.meta_wall_scrape_weight * m_wall_scrape)
    m_ctrl = act_val.norm(2, -1).sum()
    # [问题1] Meta rollout也加入高度惩罚
    m_height = (F.smooth_l1_loss(p_val[:, :, 2], torch.full_like(p_val[:, :, 2], 1.0), reduction='none')
               + F.softplus((p_val[:, :, 2] - 1.85) * 20.0)
               + F.softplus((0.15 - p_val[:, :, 2]) * 20.0)).mean()

    meta_val = sanitize_tensor(m_pos + m_coll + m_ctrl * 0.000001 + m_height * 2.0,
                               nan=1e3, posinf=1e3, neginf=1e3)
    return meta_val, m_pos, m_coll, m_ctrl

########## 7. 训练主循环 ##########
pbar = tqdm(range(args.num_iters), ncols=120)
B = args.batch_size
cycle_len = args.lgn_steps + args.worker_steps
maze_update_counter = 0

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

    p_history, v_history, target_v_history, vec_to_pt_history = [], [], [], []
    clearance_history, step_clearance_history, contact_penetration_history, contact_slide_history = [], [], [], []
    depth_history = []
    act_buffer = [env.act.detach()] * 2
    trajectory_lgn_weights = []

    h = None
    lgn_hx = None
    do_save_viz = is_save_iter(i)
    rollout_steps = args.lgn_timesteps if train_lgn_phase else args.timesteps

    ###### A. Rollout ######
    for t in range(rollout_steps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        depth, flow = env.render(ctl_dt)
        depth = sanitize_tensor(depth, nan=24.0, posinf=24.0, neginf=0.3)

        if do_save_viz:
            depth_history.append(depth[0].detach().cpu().clone())

        p_history.append(env.p)
        v_history.append(env.v)
        clearance_history.append(env.current_clearance)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())

        target_v_raw_curr = env.p_target - env.p.detach()
        target_v_norm = torch.norm(target_v_raw_curr, 2, -1, keepdim=True)
        max_speed = torch.as_tensor(env.max_speed, device=target_v_norm.device, dtype=target_v_norm.dtype)
        target_v = (target_v_raw_curr / (target_v_norm + 1e-6)) * torch.minimum(target_v_norm, max_speed)
        target_v_history.append(target_v)

        R = env.R
        state_list = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.current_clearance[:, None]]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom: state_list.insert(0, local_v)
        
        raw_state_tensor = sanitize_tensor(torch.cat(state_list, -1), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        state_tensor = state_normalizer(raw_state_tensor, update=True)
        state_tensor = sanitize_tensor(state_tensor, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        x_pooled = F.max_pool2d((3 / depth.clamp(0.3, 24) - 0.6)[:, None], 4, 4)
        x_pooled = sanitize_tensor(x_pooled, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        # LGN Forward
        current_weights, lgn_hx = lgn(x_pooled, state_tensor, lgn_hx)
        current_weights = sanitize_tensor(current_weights, nan=0.2, posinf=1.0, neginf=0.05).clamp(0.05, 1.0)
        trajectory_lgn_weights.append(current_weights)

        # Worker Forward
        act, _, h = worknet(x_pooled, state_tensor, h)
        act = sanitize_tensor(act, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        real_act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        real_act = sanitize_tensor(real_act, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        act_buffer.append(real_act)

        env.run(real_act, ctl_dt, target_v_raw_curr)
        step_clearance_history.append(env.last_step_clearance)
        contact_penetration_history.append(env.last_contact_penetration)
        contact_slide_history.append(env.last_contact_tangent_speed)

        # Early termination when all drones reach their goals
        with torch.no_grad():
            _dist_to_goal = torch.norm(env.p - env.p_target, 2, -1)
            if t >= 10 and (_dist_to_goal < args.goal_radius).all():
                break

        if args.detach_interval > 0 and (t + 1) % args.detach_interval == 0:
            if h is not None:
                h = h.detach()
            if lgn_hx is not None:
                lgn_hx = lgn_hx.detach()

    ###### B. Loss Calculation (Step-wise) ######
    p_history = torch.stack(p_history)     # [T, B, 3]
    v_history = torch.stack(v_history)     # [T, B, 3]
    clearance_history = torch.stack(clearance_history)  # [T, B]
    step_clearance = torch.stack(step_clearance_history)  # [T, B]
    contact_penetration_seq = torch.stack(contact_penetration_history)  # [T, B]
    contact_slide_seq = torch.stack(contact_slide_history)  # [T, B]
    act_buffer = torch.stack(act_buffer)   # [T+2, B, 3]
    weights_seq = torch.stack(trajectory_lgn_weights) # [T, B, 5]

    vec_to_pt = torch.stack(vec_to_pt_history)
    if vec_to_pt.dim() == 4: vec_to_pt = vec_to_pt.mean(1)
    
    # 1. 计算各项 Raw Loss (保留 [T, B] 维度用于 Step-wise 加权)

    # 真实净空: clearance_history 是当前位置净空, step_clearance 是本步动作后的扫掠最小净空
    dist_obj = sanitize_tensor(clearance_history, nan=0.0, posinf=10.0, neginf=-10.0)
    step_clearance = sanitize_tensor(step_clearance, nan=0.0, posinf=10.0, neginf=-10.0)
    contact_penetration_seq = sanitize_tensor(contact_penetration_seq, nan=0.0, posinf=10.0, neginf=0.0)
    contact_slide_seq = sanitize_tensor(contact_slide_seq, nan=0.0, posinf=30.0, neginf=0.0)

    # [问题3] 自适应速度目标: 近障碍物/近终点时自动减速
    speed_actual = v_history.norm(2, -1)  # [T, B]
    dist_to_goal = (env.p_target - p_history).norm(2, -1)  # [T, B]
    v_max = float(env.max_speed)
    speed_factor_obs = torch.sigmoid((dist_obj - 0.8) * 5.0)   # ~0 near wall, ~1 far
    speed_factor_goal = torch.clamp(dist_to_goal / 1.0, 0.0, 1.0)  # 终点1m内线性减速
    v_target_adaptive = v_max * (args.speed_near_obs_floor + (1.0 - args.speed_near_obs_floor) * speed_factor_obs) * speed_factor_goal
    loss_speed_seq = F.smooth_l1_loss(speed_actual, v_target_adaptive.detach(), reduction='none')

    target_dir = safe_normalize(env.p_target - p_history, dim=-1)
    v_dir = safe_normalize(v_history, dim=-1)
    loss_direction_seq = (1.0 - (v_dir * target_dir).sum(-1))

    # [问题2] 多尺度避障 + 前瞻碰撞预测
    vec_to_pt_dir = safe_normalize(vec_to_pt, dim=-1)
    approach_speed = (v_history * vec_to_pt_dir).sum(-1)  # 正值=正在靠近障碍物
    dist_future_02 = dist_obj - F.relu(approach_speed) * 0.2  # 0.2s前瞻
    dist_future_04 = dist_obj - F.relu(approach_speed) * 0.4  # 0.4s前瞻
    collision_depth = torch.maximum(F.relu(-step_clearance), contact_penetration_seq)
    collision_contact_prob = torch.sigmoid((collision_depth - args.coll_incident_threshold) * args.meta_coll_event_temp)
    prev_collision_contact_prob = torch.cat([torch.zeros_like(collision_contact_prob[:1]), collision_contact_prob[:-1]], dim=0)
    collision_incident_seq = (collision_contact_prob * (1.0 - prev_collision_contact_prob)).clamp(0.0, 1.0)
    tangent_speed_near_wall = torch.sqrt((speed_actual.pow(2) - approach_speed.pow(2)).clamp_min(0.0))
    near_wall_gate = torch.sigmoid((args.wall_scrape_clearance - dist_obj) * 16.0)
    wall_scrape_speed_excess = F.relu(tangent_speed_near_wall - args.wall_scrape_speed_limit)
    contact_slide_excess = F.relu(contact_slide_seq - args.wall_scrape_speed_limit)
    loss_wall_scrape_seq = (
        near_wall_gate * wall_scrape_speed_excess.pow(2)
        + 2.0 * contact_slide_excess.pow(2)
    )
    safe_margin = args.avoid_safe_margin
    loss_avoidance_seq = (
        F.softplus((safe_margin - dist_obj) * 12.0) +
        0.3 * F.softplus((safe_margin - dist_future_02) * 10.0) +
        0.2 * F.softplus((safe_margin - dist_future_04) * 10.0) +
        0.8 * F.softplus(-step_clearance * 32.0).clamp(max=100.0) +
        collision_depth.pow(2) +
        args.proxy_contact_slide_weight * contact_slide_seq +
        args.proxy_wall_scrape_weight * loss_wall_scrape_seq
    )

    # 注意: compute_overlap_loss_per_step 返回 [B, T], 需要 permute 成 [T, B]
    loss_exploration_seq = compute_overlap_loss_per_step(p_history, sigma=1.0, time_window=50).permute(1, 0)

    # Smoothness: act_buffer 长度比 timestep 多, 取最后 actual_T 步 (支持 early termination)
    actual_T = p_history.shape[0]
    loss_smooth_seq = act_buffer.diff(1, 0)[-actual_T:].pow(2).sum(-1)

    # [问题1] 高度约束损失 (固定权重, 不经LGN控制)
    z_pos = p_history[:, :, 2]  # [T, B]
    z_target = 1.0  # 迷宫中层高度
    z_min, z_max = 0.15, 1.85
    loss_height_seq = (F.smooth_l1_loss(z_pos, torch.full_like(z_pos, z_target), reduction='none')
                       + F.softplus((z_pos - z_max) * 20.0)
                       + F.softplus((z_min - z_pos) * 20.0))

    # [问题5] 归一化各损失项到相同尺度 (可微除法, stats detached)
    loss_speed_n, loss_dir_n, loss_avoid_n, loss_expl_n, loss_smooth_n = \
        loss_normalizer.normalize(loss_speed_seq, loss_direction_seq, loss_avoidance_seq,
                                  loss_exploration_seq, loss_smooth_seq)

    # 2. Step-wise 加权 (Broadcasting: [T, B] * [T, B])
    weighted_loss_map = (
        weights_seq[:, :, 0] * loss_speed_n +
        weights_seq[:, :, 1] * loss_dir_n +
        (weights_seq[:, :, 2] + args.proxy_avoid_floor) * loss_avoid_n +
        (weights_seq[:, :, 3] + 0.1) * loss_expl_n +
        weights_seq[:, :, 4] * loss_smooth_n
    )

    # 3. 最终 Proxy Loss (含固定权重的高度约束)
    loss_coll_incident = collision_incident_seq.sum(dim=0).mean()
    proxy_loss = (
        weighted_loss_map.mean()
        + 2.0 * loss_height_seq.mean()
        + args.proxy_coll_incident_weight * loss_coll_incident
    )

    # --- Meta Loss Components ---
    loss_meta_pos = torch.norm(env.p - env.p_target, 2, -1).mean()
    loss_meta_coll_soft = F.softplus(-step_clearance * 32.0).clamp(max=100.0).mean()
    loss_meta_coll_hard = collision_depth.pow(2).mean()
    loss_meta_coll_peak = collision_depth.max(dim=0).values
    loss_meta_coll_event = torch.sigmoid(
        (loss_meta_coll_peak - args.meta_coll_event_threshold) * args.meta_coll_event_temp
    ).mean()
    loss_meta_coll_event_rate = (loss_meta_coll_peak > 0).float().mean()
    loss_meta_contact_slide = contact_slide_seq.mean()
    loss_meta_coll_incident = loss_coll_incident
    loss_meta_wall_scrape = loss_wall_scrape_seq.mean()
    loss_meta_coll = (
        args.meta_coll_soft_weight * loss_meta_coll_soft
        + args.meta_coll_hard_weight * loss_meta_coll_hard
        + args.meta_coll_event_weight * loss_meta_coll_event
        + args.meta_contact_slide_weight * loss_meta_contact_slide
        + args.meta_coll_incident_weight * loss_meta_coll_incident
        + args.meta_wall_scrape_weight * loss_meta_wall_scrape
    )
    loss_meta_ctrl = act_buffer.norm(2, -1).sum()
    loss_meta_height = loss_height_seq.mean()

    meta_loss = loss_meta_pos + loss_meta_coll + loss_meta_height * 2.0 #+ loss_meta_ctrl *0  
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
    proxy_grad_smooth = 0.0
    proxy_grad_speed_nonfinite = 0.0
    proxy_grad_dir_nonfinite = 0.0
    proxy_grad_avoid_nonfinite = 0.0
    proxy_grad_expl_nonfinite = 0.0
    proxy_grad_smooth_nonfinite = 0.0
    proxy_grad_speed_elems = 0.0
    proxy_grad_dir_elems = 0.0
    proxy_grad_avoid_elems = 0.0
    proxy_grad_expl_elems = 0.0
    proxy_grad_smooth_elems = 0.0
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
    proxy_grad_smooth, proxy_grad_smooth_nonfinite, proxy_grad_smooth_elems = \
        get_loss_to_worker_grad_norm(loss_smooth_seq.mean(), worker_params)

    if train_lgn_phase:
        # ===== Unrolled Bilevel: 可微内循环 =====
        # Step 1: 用 proxy_loss 对 worker 做可微梯度下降
        fast_params = dict(worknet.named_parameters())

        for _inner in range(args.inner_steps):
            inner_grads = torch.autograd.grad(
                proxy_loss, tuple(fast_params.values()),
                create_graph=True, allow_unused=True, retain_graph=True,
            )
            fast_params = {
                name: (p - args.inner_lr * sanitize_tensor(g, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
                       if g is not None else p)
                for (name, p), g in zip(fast_params.items(), inner_grads)
            }

        # Step 2: 用虚拟更新后的 worker 做验证 rollout → meta_loss
        meta_loss_unrolled, meta_pos_ur, meta_coll_ur, meta_ctrl_ur = \
            unrolled_meta_rollout(env, worknet, fast_params, state_normalizer, args, B, device)
        if not torch.isfinite(meta_loss_unrolled):
            pbar.set_description(f"[{phase_str}] non-finite unroll skipped")
            continue

        # Step 3: 反向传播贯穿整条链路 + 熵正则化
        #   meta_loss → fast_params → ∇proxy_loss → LGN weights → LGN params
        # [问题5] 熵正则化: 鼓励LGN权重多样性, 防止坍缩
        weight_ent = (-weights_seq * torch.log(weights_seq.clamp_min(1e-8))).sum(-1).mean()
        lgn_total = meta_loss_unrolled - 0.1 * weight_ent
        lgn_total.backward()
        lgn_grad_norm, lgn_grad_max, lgn_grad_nonfinite, lgn_grad_elems = get_grad_stats(lgn)
        lgn_clip_pre = float(nn.utils.clip_grad_norm_(lgn.parameters(), 1.0).item())
        optim_lgn.step()
        sanitize_module_(lgn, clamp_value=5.0)

        lgn_update_loss = meta_loss_unrolled.detach()
    else:
        proxy_loss.backward()
        worker_grad_norm, worker_grad_max, worker_grad_nonfinite, worker_grad_elems = get_grad_stats(worknet)
        worker_clip_pre = float(nn.utils.clip_grad_norm_(worknet.parameters(), 5.0).item())
        optim_worker.step()
        sanitize_module_(worknet, clamp_value=10.0)
        sched.step()

    ###### D. Logging & Saving (Enhanced) ######
    if train_lgn_phase:
        pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Unroll: {lgn_update_loss:.3f}")
    else:
        pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Loss: {meta_loss:.3f}")
    
    with torch.no_grad():
        success = torch.all(step_clearance > 0, 0)
        # 计算平均权重 (用于 Scalar 显示)
        avg_weights = weights_seq.mean(dim=[0, 1]).cpu()
        weight_entropy = (-weights_seq * torch.log(weights_seq.clamp_min(1e-8))).sum(dim=-1).mean()
        v_norm = v_history.norm(dim=-1)
        avg_speed = v_norm.mean()
        min_speed_threshold = float(env.max_speed) * 0.7
        
        log_data = {
            # === 主要Loss ===
            'Loss/1_Proxy_Total': proxy_loss,
            'Loss/2_Meta_Total': meta_loss,
            
            # === [增强] 5个权重均值 ===
            'Weights/0_Speed': avg_weights[0],
            'Weights/1_Direction': avg_weights[1],
            'Weights/2_Avoidance': avg_weights[2],
            'Weights/3_Exploration': avg_weights[3],
            'Weights/4_Smoothness': avg_weights[4],

            # === [新增] 权重分布监控 ===
            'Weight_Stats/Min': weights_seq.min(),
            'Weight_Stats/Max': weights_seq.max(),
            'Weight_Stats/Entropy': weight_entropy,

            # === [增强] Proxy Loss 原始分项 (Average over Time & Batch) ===
            'Proxy_Comp/0_Speed': loss_speed_seq.mean(),
            'Proxy_Comp/1_Direction': loss_direction_seq.mean(),
            'Proxy_Comp/2_Avoidance': loss_avoidance_seq.mean(),
            'Proxy_Comp/2_1_Collision_Depth': collision_depth.mean(),#穿入墙体深度
            'Proxy_Comp/2_2_Contact_Slide': contact_slide_seq.mean(),
            'Proxy_Comp/2_3_Contact_Penetration': contact_penetration_seq.mean(),
            'Proxy_Comp/2_4_Collision_Incident': loss_coll_incident,
            'Proxy_Comp/2_5_Wall_Scrape': loss_wall_scrape_seq.mean(),
            'Proxy_Comp/3_Exploration': loss_exploration_seq.mean(),
            'Proxy_Comp/4_Smoothness': loss_smooth_seq.mean(),
            'Proxy_Comp/5_Height': loss_height_seq.mean(),

            # === [增强] Meta Loss 分项 ===
            'Meta_Comp/1_Position': loss_meta_pos,
            'Meta_Comp/2_Collision': loss_meta_coll,
            'Meta_Comp/2_1_Collision_Soft': loss_meta_coll_soft,#软碰撞项（靠墙即升高，连续梯度）
            'Meta_Comp/2_2_Collision_Hard': loss_meta_coll_hard,# 穿墙深度平方项。对“已经进墙”的样本施加强惩罚，且穿得越深罚越重
            'Meta_Comp/2_3_Collision_Event': loss_meta_coll_event,# 可微事件惩罚（用于训练）
            'Meta_Comp/2_4_Collision_Event_Rate': loss_meta_coll_event_rate,# 真实事件率（监控用）
            'Meta_Comp/2_5_Contact_Slide': loss_meta_contact_slide,
            'Meta_Comp/2_6_Collision_Incident': loss_meta_coll_incident,
            'Meta_Comp/2_7_Wall_Scrape': loss_meta_wall_scrape,
            'Meta_Comp/3_Control': loss_meta_ctrl,
            'Meta_Comp/4_Height': loss_meta_height,

            # === 性能指标 ===
            'Metrics/Success_Rate': success.float().mean(),
            'Metrics/Collision_Rate': (collision_depth.max(dim=0).values > 0).float().mean(),
            'Metrics/Collision_Incident_Count': collision_incident_seq.sum(dim=0).mean(),
            'Metrics/Wall_Scrape_Excess': wall_scrape_speed_excess.mean(),
            'Metrics/Min_Clearance': step_clearance.min(),
            'Metrics/Max_Penetration': collision_depth.max(),
            'Metrics/Contact_Slide': contact_slide_seq.mean(),
            'Metrics/Avg_Speed': avg_speed,
            'Metrics/Speed_Below_Threshold': (avg_speed < min_speed_threshold).float(),
            'Metrics/Min_Speed': v_norm.min(),
            'Metrics/Max_Speed': v_norm.max(),
            'Metrics/Episode_Length': actual_T,
            'Metrics/Adaptive_Speed_Target': v_target_adaptive.mean(),

            # === [对齐] 归一化统计命名（与第二脚本风格一致） ===
            'Norm/State_Mean': state_normalizer.mean[0],
            'Norm/State_Var': state_normalizer.var[0],
            'Norm/Update_Count': state_normalizer.count,

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

            # === [新增] 五个代理损失对 Worker 梯度的 norm ===
            'Grad_ProxyWorker/0_Speed_Norm': proxy_grad_speed,
            'Grad_ProxyWorker/1_Direction_Norm': proxy_grad_dir,
            'Grad_ProxyWorker/2_Avoidance_Norm': proxy_grad_avoid,
            'Grad_ProxyWorker/3_Exploration_Norm': proxy_grad_expl,
            'Grad_ProxyWorker/4_Smoothness_Norm': proxy_grad_smooth,
            'Grad_ProxyWorker/0_Speed_NonFinite': proxy_grad_speed_nonfinite,
            'Grad_ProxyWorker/1_Direction_NonFinite': proxy_grad_dir_nonfinite,
            'Grad_ProxyWorker/2_Avoidance_NonFinite': proxy_grad_avoid_nonfinite,
            'Grad_ProxyWorker/3_Exploration_NonFinite': proxy_grad_expl_nonfinite,
            'Grad_ProxyWorker/4_Smoothness_NonFinite': proxy_grad_smooth_nonfinite,
            'Grad_ProxyWorker/0_Speed_GradElem': proxy_grad_speed_elems,
            'Grad_ProxyWorker/1_Direction_GradElem': proxy_grad_dir_elems,
            'Grad_ProxyWorker/2_Avoidance_GradElem': proxy_grad_avoid_elems,
            'Grad_ProxyWorker/3_Exploration_GradElem': proxy_grad_expl_elems,
            'Grad_ProxyWorker/4_Smoothness_GradElem': proxy_grad_smooth_elems
        }
        
        if train_lgn_phase:
            log_data['Loss/3_LGN_Unrolled_Meta'] = lgn_update_loss
            log_data['Meta_Unrolled/1_Position'] = meta_pos_ur
            log_data['Meta_Unrolled/2_Collision'] = meta_coll_ur
            log_data['Meta_Unrolled/3_Control'] = meta_ctrl_ur

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
            
            idx = 0
            
            # 1. 轨迹时序图 (X,Y,Z vs T)
            fig_p, ax = plt.subplots()
            p_cpu = p_history[:, idx].cpu()
            ax.plot(p_cpu[:, 0], label='x'); ax.plot(p_cpu[:, 1], label='y'); ax.plot(p_cpu[:, 2], label='z')
            ax.legend(); ax.set_title(f"Iter {i} Pos (Time Series)")
            writer.add_figure('Trajectory/Position_Series', fig_p, i + 1)
            plt.close(fig_p)

            # 2. [增强] 轨迹俯视图 + 障碍物 + 目标点 (速度热力图)
            fig_map, ax = plt.subplots(figsize=(6, 10))
            wall_patches = []
            if hasattr(env, 'voxels'):
                walls_tensor = env.voxels[0].detach().cpu()
                for w in walls_tensor.numpy():
                    # 过滤地板/天花板，仅显示中间层障碍物
                    if w[2] < 0.1 or w[2] > 1.9:
                        continue
                    rect = plt.Rectangle((w[0] - w[3], w[1] - w[4]), 2 * w[3], 2 * w[4], color='gray', alpha=0.5)
                    ax.add_patch(rect)

                collision_segment_len = 0.35 * float(getattr(env, 'maze_cell_size', 1.5))
                wall_patches = get_collision_wall_patches(
                    p_cpu,
                    walls_tensor,
                    drone_radius=float(getattr(env, 'drone_radius', 0.12)),
                    segment_len=collision_segment_len,
                    contact_eps=0.02,
                )
                for patch_idx, patch in enumerate(wall_patches):
                    rect = plt.Rectangle(
                        patch['xy'], patch['width'], patch['height'],
                        facecolor='red', edgecolor='firebrick', linewidth=0.8,
                        alpha=0.85, zorder=2.5,
                        label='Collision Wall' if patch_idx == 0 else None,
                    )
                    ax.add_patch(rect)

            # 速度热力图轨迹: 蓝(慢) -> 红(快), 范围 0-10 m/s
            v_cpu = v_history[:, idx].cpu()
            speed_cpu = v_cpu.norm(dim=-1).numpy()  # [T]
            points = p_cpu[:, :2].numpy()            # [T, 2] (X, Y)
            segments = []
            seg_speeds = []
            for si in range(len(points) - 1):
                segments.append([points[si], points[si + 1]])
                seg_speeds.append((speed_cpu[si] + speed_cpu[si + 1]) / 2.0)
            seg_speeds = [speed_cpu[0]] if len(seg_speeds) == 0 else seg_speeds

            norm = Normalize(vmin=0.0, vmax=10.0)
            cmap = cm.get_cmap('coolwarm')  # 蓝(低速) -> 红(高速)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2)
            lc.set_array(torch.tensor(seg_speeds).numpy())
            lc.set_zorder(3.0)
            ax.add_collection(lc)
            cbar = fig_map.colorbar(lc, ax=ax, label='Speed (m/s)')

            ax.plot(p_cpu[0, 0], p_cpu[0, 1], 'go', markersize=8, label='Start')   # 起点
            ax.plot(p_cpu[-1, 0], p_cpu[-1, 1], 'kx', markersize=8, label='End')    # 终点
            if hasattr(env, 'p_target'):
                target = env.p_target[idx].detach().cpu().numpy()
                ax.plot(target[0], target[1], 'r*', markersize=10, label='Goal')

            if all(hasattr(env, k) for k in ['maze_cols', 'maze_rows', 'maze_cell_size']):
                maze_w = float(env.maze_cols) * float(env.maze_cell_size)
                maze_h = float(env.maze_rows) * float(env.maze_cell_size)
                ax.set_xlim(0.0, maze_w)
                ax.set_ylim(-maze_h / 2.0, maze_h / 2.0)
            else:
                ax.autoscale_view()

            ax.set_aspect('equal')
            ax.legend(); ax.set_title(f"Iter {i} Map & Trajectory (Speed Heatmap)")
            writer.add_figure('Trajectory/Map_View', fig_map, i + 1)
            plt.close(fig_map)
            
            # 3. [新增] 速度时序图 (Vx,Vy,Vz,Speed)
            fig_v, ax = plt.subplots()
            v_cpu = v_history[:, idx].cpu()
            ax.plot(v_cpu[:, 0], label='vx'); ax.plot(v_cpu[:, 1], label='vy'); ax.plot(v_cpu[:, 2], label='vz')
            ax.plot(v_cpu.norm(dim=-1), label='speed', linestyle='--')
            ax.legend(); ax.set_title(f"Iter {i} Velocity (Time Series)")
            writer.add_figure('Trajectory/Velocity_Series', fig_v, i + 1)
            plt.close(fig_v)

            # 4. [新增] 权重时序变化图 - 验证 Step-wise 效果
            fig_w, ax = plt.subplots()
            w_cpu = weights_seq[:, idx, :].cpu() # [T, 5]
            labels = ['Speed', 'Dir', 'Avoid', 'Expl', 'Smooth']
            for wi in range(5):
                ax.plot(w_cpu[:, wi], label=labels[wi])
            ax.legend(); ax.set_title(f"Iter {i} Weights Profile (Per Step)")
            writer.add_figure('Debug/Weights_StepWise', fig_w, i + 1)
            plt.close(fig_w)

            # 5. [新增] 深度图视频（保存到本地）
            if len(depth_history) > 0:
                depth_stack = torch.stack(depth_history).float()  # [T, H, W], meters
                # 使用逆深度增强近处障碍可见性，并做分位数拉伸避免整段几乎同值导致全黑
                inv_depth = 3.0 / depth_stack.clamp(0.3, 24.0) - 0.6  # 与网络输入一致的尺度
                p2 = torch.quantile(inv_depth, 0.02)
                p98 = torch.quantile(inv_depth, 0.98)
                inv_norm = ((inv_depth - p2) / (p98 - p2 + 1e-6)).clamp(0.0, 1.0)  # [T, H, W]

                # 转为 RGB 彩色帧，避免灰度写视频时编码器/播放器显示发黑
                cmap_np = cm.get_cmap('magma')
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
