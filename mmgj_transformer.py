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

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("[SDP] flash=False, mem_efficient=False, math=True (for higher-order gradients)")

try:
    from env_maze import Env
except ModuleNotFoundError:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from env_maze import Env
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


class LossNormalizer(nn.Module):
    """
    对各项代理损失做运行时均值归一化。
    引入 min_scales (Scale Floor) 防止损失收敛至接近 0 时引发的梯度放大与权重塌陷。
    """
    def __init__(self, num_losses, decay=0.995):
        super().__init__()
        self.num_losses = num_losses
        self.decay = decay
        self.register_buffer('running_mean', torch.zeros(num_losses))
        self.register_buffer('initialized', torch.zeros(num_losses, dtype=torch.bool))

        # [核心] 为不同损失项设置合理的"缩放底线" (依据真实 loss 表现设定)
        # 对应顺序: Speed(~1.5), Dir(~1.3), Avoid(~0.15), Expl(~0.002), Smooth(~0)
        # 当 running_mean 低于此下限时，停止进一步放大梯度
        self.register_buffer('min_scales', torch.tensor([0.5, 0.5, 0.05, 0.02, 0.01]))

    @torch.no_grad()
    def update(self, loss_values):
        """loss_values: [num_losses] — 各项损失在当前 batch 上的均值"""
        for k in range(self.num_losses):
            val = loss_values[k]
            if not self.initialized[k]:
                self.running_mean[k] = val
                self.initialized[k] = True
            else:
                self.running_mean[k] = self.decay * self.running_mean[k] + (1 - self.decay) * val

    def normalize(self, loss_seq_list):
        """
        Returns: list of normalized [T, B] tensors
        """
        normalized = []
        for k, loss_seq in enumerate(loss_seq_list):
            # 用 max(EMA均值, 物理下限) 作为分母，防止收敛项梯度爆炸
            scale_k = torch.maximum(self.running_mean[k], self.min_scales[k])
            normalized.append(loss_seq / scale_k)
        return normalized


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
parser.add_argument('--detach_interval', type=int, default=8,
                    help='Detach temporal memory every N steps to limit graph depth (<=0 disables)')
parser.add_argument('--cam_angle', type=int, default=10)

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
parser.add_argument('--lgn_lr', type=float, default=5e-4)
parser.add_argument('--inner_lr', type=float, default=1e-4,
                    help='Inner loop LR for differentiable worker update in LGN phase')
parser.add_argument('--inner_steps', type=int, default=1,
                    help='Number of differentiable inner SGD steps (unrolled bilevel)')
parser.add_argument('--exp_name', type=str, default="default", help="Extra tag for experiment")

args = parser.parse_args()

########## 2. 目录与日志初始化 ##########
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
script_name = os.path.splitext(os.path.basename(__file__))[0]
save_dir_name = f"{script_name}_{args.exp_name}_{current_time}"
save_dir = os.path.join("..", "checkpoints", save_dir_name)

os.makedirs(save_dir, exist_ok=True)
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

NUM_PROXY_LOSSES = 5  # speed, direction, avoidance, exploration, smoothness
loss_normalizer = LossNormalizer(num_losses=NUM_PROXY_LOSSES, decay=0.995).to(device)
print(f"[LossNorm] min_scales (floor): {loss_normalizer.min_scales.tolist()}")

########## 4. 加载预训练模型 ##########
def load_checkpoint(model, path, name):
    if path and os.path.isfile(path):
        print(f"Loading {name} from {path}")
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
    elif path:
        print(f"Warning: {name} path provided but file not found: {path}")

load_checkpoint(worknet, args.resume_worker, "Worker")
load_checkpoint(lgn, args.resume_lgn, "LGN")
if args.resume_norm:
    load_checkpoint(state_normalizer, args.resume_norm, "Norm Stats")
    # 自动推断 loss_normalizer 路径
    lossnorm_path = args.resume_norm.replace('norm_', 'lossnorm_')
    load_checkpoint(loss_normalizer, lossnorm_path, "Loss Normalizer Stats")
elif args.resume_worker:
    norm_path = args.resume_worker.replace('worker_', 'norm_')
    load_checkpoint(state_normalizer, norm_path, "Auto-inferred Norm Stats")
    lossnorm_path = args.resume_worker.replace('worker_', 'lossnorm_')
    load_checkpoint(loss_normalizer, lossnorm_path, "Auto-inferred Loss Normalizer Stats")

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
    return (i + 1) % 10000 == 0 if i >= 2000 else (i + 1) % 500 == 0

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
    """
    env.reset()

    p_list, v_list, vec_list = [], [], []
    act_buf = [env.act.detach()] * 2
    h_val = None

    for t in range(args.lgn_timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        depth, flow = env.render(ctl_dt)

        p_list.append(env.p)
        v_list.append(env.v)
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

        raw_state = torch.cat(state_list, -1)
        state_t = state_normalizer(raw_state, update=False)  # 不更新统计量

        x_pooled = F.max_pool2d((3 / depth.clamp_(0.3, 24) - 0.6)[:, None], 4, 4)

        # Worker forward with virtually-updated params
        act_out, _, h_val = functional_call(worknet, fast_params, (x_pooled, state_t, h_val))

        a_pred, v_pred, *_ = (R @ act_out.reshape(B, 3, -1)).unbind(-1)
        real_act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buf.append(real_act)

        env.run(real_act, ctl_dt, target_v_raw)

        # 周期性截断以限制显存
        if args.detach_interval > 0 and (t + 1) % args.detach_interval == 0:
            if h_val is not None:
                h_val = h_val.detach()

    # --- 计算 Meta Loss ---
    p_val = torch.stack(p_list)
    act_val = torch.stack(act_buf)
    vec_val = torch.stack(vec_list)
    if vec_val.dim() == 4:
        vec_val = vec_val.mean(1)

    dist_val = vec_val.norm(2, -1) - env.margin

    m_pos  = torch.norm(p_val[-1] - env.p_target, 2, -1).mean()
    m_coll = F.softplus(-dist_val * 32.0).clamp(max=100.0).mean()
    m_ctrl = act_val.norm(2, -1).sum()

    meta_val = m_pos + m_coll * 5.0 + m_ctrl * 0.000001
    return meta_val, m_pos, m_coll, m_ctrl

########## 7. 训练主循环 ##########
pbar = tqdm(range(args.num_iters), ncols=120)
B = args.batch_size
cycle_len = args.lgn_steps + args.worker_steps

state_normalizer.train()

for i in pbar:
    cycle_pos = i % cycle_len
    train_lgn_phase = cycle_pos < args.lgn_steps
    phase_str = f"LGN ({cycle_pos+1}/{args.lgn_steps})" if train_lgn_phase else f"Work ({cycle_pos-args.lgn_steps+1}/{args.worker_steps})"

    env.reset()
    worknet.reset()

    p_history, v_history, target_v_history, vec_to_pt_history = [], [], [], []
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

        if do_save_viz:
            depth_history.append(depth[0].detach().cpu().clone())

        p_history.append(env.p)
        v_history.append(env.v)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())

        target_v_raw_curr = env.p_target - env.p.detach()
        target_v_norm = torch.norm(target_v_raw_curr, 2, -1, keepdim=True)
        max_speed = torch.as_tensor(env.max_speed, device=target_v_norm.device, dtype=target_v_norm.dtype)
        target_v = (target_v_raw_curr / (target_v_norm + 1e-6)) * torch.minimum(target_v_norm, max_speed)
        target_v_history.append(target_v)

        R = env.R
        state_list = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.margin[:, None]]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom: state_list.insert(0, local_v)
        
        raw_state_tensor = torch.cat(state_list, -1)
        state_tensor = state_normalizer(raw_state_tensor, update=True)

        x_pooled = F.max_pool2d((3 / depth.clamp_(0.3, 24) - 0.6)[:, None], 4, 4)

        # LGN Forward
        current_weights, lgn_hx = lgn(x_pooled, state_tensor, lgn_hx)
        trajectory_lgn_weights.append(current_weights)

        # Worker Forward
        act, _, h = worknet(x_pooled, state_tensor, h)
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        real_act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(real_act)

        env.run(real_act, ctl_dt, target_v_raw_curr)

        if args.detach_interval > 0 and (t + 1) % args.detach_interval == 0:
            if h is not None:
                h = h.detach()
            if lgn_hx is not None:
                lgn_hx = lgn_hx.detach()

    ###### B. Loss Calculation (Step-wise) ######
    p_history = torch.stack(p_history)     # [T, B, 3]
    v_history = torch.stack(v_history)     # [T, B, 3]
    act_buffer = torch.stack(act_buffer)   # [T+2, B, 3]
    weights_seq = torch.stack(trajectory_lgn_weights) # [T, B, 5]

    vec_to_pt = torch.stack(vec_to_pt_history)
    if vec_to_pt.dim() == 4: vec_to_pt = vec_to_pt.mean(1)
    
    # 1. 计算各项 Raw Loss (保留 [T, B] 维度用于 Step-wise 加权)
    loss_speed_seq = F.smooth_l1_loss(v_history.norm(2, -1), 
                                      torch.ones_like(v_history.norm(2, -1)) * 5.0, 
                                      reduction='none')
    
    target_dir = F.normalize(env.p_target - p_history, dim=-1)
    v_dir = F.normalize(v_history, dim=-1)
    loss_direction_seq = (1.0 - (v_dir * target_dir).sum(-1))
    
    dist_obj = vec_to_pt.norm(2, -1) - env.margin
    loss_avoidance_seq = F.softplus(-dist_obj * 10.0)
    
    # 注意: compute_overlap_loss_per_step 返回 [B, T], 需要 permute 成 [T, B]
    loss_exploration_seq = compute_overlap_loss_per_step(p_history, sigma=1.0, time_window=50).permute(1, 0)
    
    # Smoothness: act_buffer 长度比 timestep 多, 取最后 rollout_steps 步
    loss_smooth_seq = act_buffer.diff(1, 0)[-rollout_steps:].pow(2).sum(-1)

    # ========== 损失归一化：统一各项量级到 ~1.0 ==========
    with torch.no_grad():
        raw_means = torch.stack([
            loss_speed_seq.mean(),
            loss_direction_seq.mean(),
            loss_avoidance_seq.mean(),
            loss_exploration_seq.mean(),
            loss_smooth_seq.mean(),
        ])
        loss_normalizer.update(raw_means)

    (norm_speed, norm_direction, norm_avoidance,
     norm_exploration, norm_smooth) = loss_normalizer.normalize([
        loss_speed_seq, loss_direction_seq, loss_avoidance_seq,
        loss_exploration_seq, loss_smooth_seq,
    ])

    # 2. Step-wise 加权 (Broadcasting: [T, B] * [T, B])  — 使用归一化后的损失
    weighted_loss_map = (
        weights_seq[:, :, 0] * norm_speed +
        weights_seq[:, :, 1] * norm_direction +
        (weights_seq[:, :, 2] + 0.2) * norm_avoidance +
        (weights_seq[:, :, 3] + 0.1) * norm_exploration +
        weights_seq[:, :, 4] * norm_smooth
    )

    # 3. 最终 Proxy Loss
    proxy_loss = weighted_loss_map.mean()

    # --- Meta Loss Components ---
    loss_meta_pos = torch.norm(p_history[-1] - env.p_target, 2, -1).mean()
    loss_meta_coll = F.softplus(-dist_obj * 32.0).clamp(max=100.0).mean()
    loss_meta_ctrl = act_buffer.norm(2, -1).sum()

    meta_loss = loss_meta_pos + loss_meta_coll * 5.0 + loss_meta_ctrl * 0.000001

    ###### C. Optimization ######
    optim_worker.zero_grad()
    optim_lgn.zero_grad()
    lgn_update_loss = 0.0

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
                name: (p - args.inner_lr * g if g is not None else p)
                for (name, p), g in zip(fast_params.items(), inner_grads)
            }

        # Step 2: 用虚拟更新后的 worker 做验证 rollout → meta_loss
        meta_loss_unrolled, meta_pos_ur, meta_coll_ur, meta_ctrl_ur = \
            unrolled_meta_rollout(env, worknet, fast_params, state_normalizer, args, B, device)

        # Step 3: 反向传播贯穿整条链路
        #   meta_loss → fast_params → ∇proxy_loss → LGN weights → LGN params
        meta_loss_unrolled.backward()
        nn.utils.clip_grad_norm_(lgn.parameters(), 1.0)
        optim_lgn.step()

        lgn_update_loss = meta_loss_unrolled.detach()
    else:
        proxy_loss.backward()
        nn.utils.clip_grad_norm_(worknet.parameters(), 5.0)
        optim_worker.step()
        sched.step()

    ###### D. Logging & Saving (Enhanced) ######
    if train_lgn_phase:
        pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Unroll: {lgn_update_loss:.3f}")
    else:
        pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Loss: {meta_loss:.3f}")
    
    with torch.no_grad():
        success = torch.all(dist_obj > 0, 0)
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
            'Proxy_Comp/0_Speed_Raw': loss_speed_seq.mean(),
            'Proxy_Comp/1_Direction_Raw': loss_direction_seq.mean(),
            'Proxy_Comp/2_Avoidance_Raw': loss_avoidance_seq.mean(),
            'Proxy_Comp/3_Exploration_Raw': loss_exploration_seq.mean(),
            'Proxy_Comp/4_Smoothness_Raw': loss_smooth_seq.mean(),

            # === 归一化后分项 — 验证量级是否统一 ===
            'Proxy_Norm/0_Speed': norm_speed.mean(),
            'Proxy_Norm/1_Direction': norm_direction.mean(),
            'Proxy_Norm/2_Avoidance': norm_avoidance.mean(),
            'Proxy_Norm/3_Exploration': norm_exploration.mean(),
            'Proxy_Norm/4_Smoothness': norm_smooth.mean(),

            # === LossNormalizer 内部统计 ===
            'LossNorm/Running_Mean_Speed': loss_normalizer.running_mean[0],
            'LossNorm/Running_Mean_Dir': loss_normalizer.running_mean[1],
            'LossNorm/Running_Mean_Avoid': loss_normalizer.running_mean[2],
            'LossNorm/Running_Mean_Expl': loss_normalizer.running_mean[3],
            'LossNorm/Running_Mean_Smooth': loss_normalizer.running_mean[4],
            # effective scale = max(running_mean, min_scales)，即实际除数
            'LossNorm/Effective_Scale_Speed': torch.maximum(loss_normalizer.running_mean[0], loss_normalizer.min_scales[0]),
            'LossNorm/Effective_Scale_Dir': torch.maximum(loss_normalizer.running_mean[1], loss_normalizer.min_scales[1]),
            'LossNorm/Effective_Scale_Avoid': torch.maximum(loss_normalizer.running_mean[2], loss_normalizer.min_scales[2]),
            'LossNorm/Effective_Scale_Expl': torch.maximum(loss_normalizer.running_mean[3], loss_normalizer.min_scales[3]),
            'LossNorm/Effective_Scale_Smooth': torch.maximum(loss_normalizer.running_mean[4], loss_normalizer.min_scales[4]),

            # === [增强] Meta Loss 分项 ===
            'Meta_Comp/1_Position': loss_meta_pos,
            'Meta_Comp/2_Collision': loss_meta_coll,
            'Meta_Comp/3_Control': loss_meta_ctrl,

            # === 性能指标 ===
            'Metrics/Success_Rate': success.float().mean(),
            'Metrics/Avg_Speed': avg_speed,
            'Metrics/Speed_Below_Threshold': (avg_speed < min_speed_threshold).float(),
            'Metrics/Min_Speed': v_norm.min(),
            'Metrics/Max_Speed': v_norm.max(),

            # === [对齐] 归一化统计命名（与第二脚本风格一致） ===
            'Norm/State_Mean': state_normalizer.mean[0],
            'Norm/State_Var': state_normalizer.var[0],
            'Norm/Update_Count': state_normalizer.count,

            # === [兼容] 保留旧命名 ===
            'Stats/Norm_Mean': state_normalizer.mean[0],
            'Stats/Norm_Var': state_normalizer.var[0],
            'Stats/Norm_Count': state_normalizer.count
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

        if is_save_iter(i):
            torch.save(worknet.state_dict(), os.path.join(save_dir, f'worker_ckpt_{i:06d}.pth'))
            torch.save(lgn.state_dict(), os.path.join(save_dir, f'lgn_ckpt_{i:06d}.pth'))
            torch.save(state_normalizer.state_dict(), os.path.join(save_dir, f'norm_ckpt_{i:06d}.pth'))
            torch.save(loss_normalizer.state_dict(), os.path.join(save_dir, f'lossnorm_ckpt_{i:06d}.pth'))
            
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
            if hasattr(env, 'voxels'):
                walls = env.voxels[0].detach().cpu().numpy()
                for w in walls:
                    # 过滤地板/天花板，仅显示中间层障碍物
                    if w[2] < 0.1 or w[2] > 1.9:
                        continue
                    rect = plt.Rectangle((w[0] - w[3], w[1] - w[4]), 2 * w[3], 2 * w[4], color='gray', alpha=0.5)
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

            # 5. [新增] 深度图视频
            if len(depth_history) > 0:
                depth_stack = torch.stack(depth_history)
                d_min = depth_stack.min()
                d_max = depth_stack.max()
                depth_norm = (depth_stack - d_min) / (d_max - d_min + 1e-6)
                vid_tensor = depth_norm.unsqueeze(0).unsqueeze(2)  # [1, T, 1, H, W]
                writer.add_video('Video/Depth_View', vid_tensor, i + 1, fps=15)

print(f"Training Finished. Artifacts in: {save_dir}")