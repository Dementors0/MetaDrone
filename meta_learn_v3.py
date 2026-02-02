import argparse
import math
from collections import defaultdict
from random import normalvariate
import os

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt

# 假设这些是你本地的模块
from env_cuda import Env
from WorkNet import WorkNet
from LossGenNet import LossGenNet

########### 1. 参数配置 ##########
parser = argparse.ArgumentParser()
parser.add_argument('--resume_worker',
                    default="/home/robot/validation_code/training_code/multi_pub/worker_ckpt_272999.pth",
                    help='Path to pretrained worker model')
parser.add_argument('--resume_lgn',
                    default="/home/robot/validation_code/training_code/multi_pub/lgn_ckpt_272999.pth",
                    help='Path to pretrained lgn model')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_iters', type=int, default=500000)

# [优化策略参数]
parser.add_argument('--lgn_steps', type=int, default=1, help='LGN 连续优化的步数')
parser.add_argument('--worker_steps', type=int, default=3000, help='Worker 连续优化的步数')

# 基础物理参数
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--cam_angle', type=int, default=10)
# 环境Flag
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')

# 学习率
parser.add_argument('--lr', type=float, default=1e-4, help='Worker Learning Rate')
parser.add_argument('--lgn_lr', type=float, default=5e-4, help='LGN Learning Rate')

args = parser.parse_args()

# 使用更清晰的日志目录名
writer = SummaryWriter()
print(args)

device = torch.device('cuda')

########## 2. 环境初始化 ##########
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle)

# --- Worker Network ---
if args.no_odom:
    worknet = WorkNet(7, 6)
else:
    worknet = WorkNet(7 + 3, 6)
worknet = worknet.to(device)

# --- LGN Network ---
lgn_state_dim = 7 if args.no_odom else 10
lgn = LossGenNet(state_dim=lgn_state_dim).to(device)

########## 4. 加载预训练模型 ##########
"""if args.resume_worker:
    if os.path.isfile(args.resume_worker):
        print(f"Loading pretrained worker from {args.resume_worker}")
        state_dict_worker = torch.load(args.resume_worker, map_location=device)
        worknet.load_state_dict(state_dict_worker, strict=False)
        print(f"Loading pretrained lgn from {args.resume_lgn}")
        state_dict_lgn = torch.load(args.resume_lgn, map_location=device)
        lgn.load_state_dict(state_dict_lgn, strict=False)
    else:
        print(f"Warning: Pretrained model not found at {args.resume}")"""

########## 5. 优化器配置 ##########
optim_worker = AdamW(worknet.parameters(), args.lr)
optim_lgn = AdamW(lgn.parameters(), args.lgn_lr)
sched = CosineAnnealingLR(optim_worker, args.num_iters, args.lr * 0.01)

########## 6. 辅助函数 ##########
scaler_q = defaultdict(list)


def smooth_dict(ori_dict):
    """累积数据用于平滑显示"""
    for k, v in ori_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        scaler_q[k].append(float(v))


def is_save_iter(i):
    return (i + 1) % 10000 == 0 if i >= 2000 else (i + 1) % 500 == 0


def compute_overlap_loss(p_history, sigma=0.5, time_window=10):
    """
    计算轨迹重叠损失 (维度修复版)
    """
    # [修复] 调整维度: [Time, Batch, Dim] -> [Batch, Time, Dim]
    p_history = p_history.permute(1, 0, 2)
    n_batch, n_points, n_dims = p_history.shape

    if n_points < time_window + 1:
        return torch.tensor(0.0, device=p_history.device)

    # 1. 计算两两距离矩阵
    dist_matrix = torch.cdist(p_history, p_history, p=2)

    # 2. 高斯核转换
    overlap_energy = torch.exp(- (dist_matrix ** 2) / (2 * sigma ** 2))

    # 3. 时间掩码
    indices = torch.arange(n_points, device=p_history.device)
    time_diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    mask = (time_diff > time_window).float()

    # 4. 计算损失
    loss_overlap = (overlap_energy * mask).sum() / (mask.sum() * n_batch + 1e-6)
    return loss_overlap


########## 7. 训练主循环 ##########
pbar = tqdm(range(args.num_iters), ncols=120)
B = args.batch_size

# 周期总长度
cycle_len = args.lgn_steps + args.worker_steps

for i in pbar:
    # --- 非对称阶段切换逻辑 ---
    cycle_pos = i % cycle_len
    # 前 lgn_steps 步训练 LGN，之后训练 Worker
    train_lgn_phase = cycle_pos < args.lgn_steps

    if train_lgn_phase:
        phase_str = f"LGN ({cycle_pos + 1}/{args.lgn_steps})"
    else:
        worker_pos = cycle_pos - args.lgn_steps + 1
        phase_str = f"Work ({worker_pos}/{args.worker_steps})"

    # 重置环境
    env.reset()
    worknet.reset()

    # 轨迹容器
    p_history, v_history, target_v_history, vec_to_pt_history = [], [], [], []

    # [修复] 使用 .detach() 断开与上一轮迭代的联系，防止 retain_graph 报错
    act_buffer = [env.act.detach()] * 2

    trajectory_lgn_weights = []
    target_v_raw = env.p_target - env.p
    h = None

    ###### A. Rollout (数据收集) ######
    for t in range(args.timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        depth, flow = env.render(ctl_dt)

        p_history.append(env.p)
        v_history.append(env.v)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())

        target_v_raw_curr = env.p_target - env.p.detach()
        target_v_norm = torch.norm(target_v_raw_curr, 2, -1, keepdim=True)
        target_v = (target_v_raw_curr / (target_v_norm + 1e-6)) * torch.minimum(target_v_norm, env.max_speed)
        target_v_history.append(target_v)

        # 构建状态
        R = env.R
        state_list = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.margin[:, None]]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom: state_list.insert(0, local_v)
        state_tensor = torch.cat(state_list, -1)

        # 视觉特征
        x_pooled = F.max_pool2d((3 / depth.clamp_(0.3, 24) - 0.6)[:, None], 4, 4)

        # --- LGN Forward ---
        current_weights = lgn(x_pooled, state_tensor)
        trajectory_lgn_weights.append(current_weights)

        # --- Worker Forward ---
        act, _, h = worknet(x_pooled, state_tensor, h)
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        real_act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(real_act)

        # 物理步进
        env.run(real_act, ctl_dt, target_v_raw_curr)

    ###### B. 损失计算 ######
    p_history = torch.stack(p_history)
    v_history = torch.stack(v_history)
    target_v_history = torch.stack(target_v_history)
    act_buffer = torch.stack(act_buffer)

    vec_to_pt = torch.stack(vec_to_pt_history)
    if vec_to_pt.dim() == 4: vec_to_pt = vec_to_pt.mean(1)
    dist_obj = vec_to_pt.norm(2, -1) - env.margin

    # --- Proxy Loss Components ---
    loss_speed_norm = F.smooth_l1_loss(v_history.norm(2, -1), torch.ones_like(v_history.norm(2, -1)) * 5.0)
    target_dir = F.normalize(env.p_target - p_history, dim=-1)
    v_dir = F.normalize(v_history, dim=-1)
    loss_direction = (1.0 - (v_dir * target_dir).sum(-1)).mean()
    loss_avoidance = F.softplus(-dist_obj * 10.0).mean()
    loss_exploration = compute_overlap_loss(p_history, sigma=1.0, time_window=50)
    loss_smooth = act_buffer.diff(1, 0).pow(2).mean()

    # --- Meta Loss Components ---
    loss_meta_pos = torch.norm(p_history[-1] - env.p_target, 2, -1).mean()
    loss_meta_coll = F.softplus(-dist_obj * 32.0).clamp(max=100.0).mean()
    # [注意] 使用 sum()，需要配合极小的权重
    loss_meta_ctrl = act_buffer.norm(2, -1).sum()

    # --- 聚合 ---
    avg_weights = torch.stack(trajectory_lgn_weights).mean(dim=[0, 1])

    proxy_loss = avg_weights[0] * loss_speed_norm + \
                 avg_weights[1] * loss_direction + \
                 (avg_weights[2] + 0.2) * loss_avoidance + \
                 (avg_weights[3] + 0.1) * loss_exploration + \
                 avg_weights[4] * loss_smooth

    # [调整权重] ctrl系数降为 1e-6 以适应 sum()
    meta_loss = loss_meta_pos + loss_meta_coll * 5.0 + loss_meta_ctrl * 0.000001

    ###### C. 优化执行 ######
    optim_worker.zero_grad()
    optim_lgn.zero_grad()
    lgn_update_loss = 0.0

    if train_lgn_phase:
        # === 阶段 1: 优化 LGN (Gradient Alignment / 梯度对齐) ===
        # 这里的逻辑是：让 LGN 产生的 Proxy 梯度，方向尽可能接近理想的 Meta 梯度

        # 1. 计算 Proxy 梯度
        # create_graph=True: 必须开启，因为我们需要对这个"梯度"本身进行反向传播（求导）
        grad_proxy = torch.autograd.grad(proxy_loss, worknet.parameters(), create_graph=True, allow_unused=True)

        # 2. 计算 Meta 梯度
        # retain_graph=True: 必须开启，否则计算图会被释放，导致后面无法 backward
        # allow_unused=True: 允许部分参数没有梯度（例如不涉及 Odom 的层）
        grad_meta = torch.autograd.grad(meta_loss, worknet.parameters(), allow_unused=True, retain_graph=True)

        # 3. 计算点积损失 (Dot Product)
        # 我们希望 maximize (grad_proxy · grad_meta)，等价于 minimize -(grad_proxy · grad_meta)
        lgn_update_loss = 0.0
        norm_factor = 0.0  # 用于归一化，防止梯度过大

        for gp, gm in zip(grad_proxy, grad_meta):
            if gp is not None and gm is not None:
                # .detach() 很重要：我们将 Meta 梯度视为"常数目标"，不更新它
                gm_detached = gm.detach()
                lgn_update_loss -= (gp.flatten() * gm_detached.flatten()).sum()
                norm_factor += 1

        # 归一化 (可选，增加稳定性)
        if norm_factor > 0:
            lgn_update_loss = lgn_update_loss / norm_factor

        # 4. 反向传播更新 LGN
        optim_lgn.zero_grad()
        if isinstance(lgn_update_loss, torch.Tensor):
            lgn_update_loss.backward()
            nn.utils.clip_grad_norm_(lgn.parameters(), 1.0)
            optim_lgn.step()

    else:
        # === 阶段 2: 优化 Worker (Standard Update) ===
        # Worker 使用 LGN (刚才阶段1优化过的) 产生的 Proxy Loss 进行真正的更新

        optim_worker.zero_grad()
        proxy_loss.backward()  # 这里的 backward 是标准的累积梯度
        nn.utils.clip_grad_norm_(worknet.parameters(), 5.0)
        optim_worker.step()
        sched.step()

    ###### D. 日志与 TensorBoard ######
    pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Loss: {meta_loss:.3f}")

    with torch.no_grad():
        # 收集数据到 smooth_dict
        success = torch.all(dist_obj > 0, 0)  # 假设 dist_obj > 0 表示无碰撞
        avg_speed = v_history.norm(dim=-1).mean(0)

        smooth_dict({
            'Loss/Proxy_Total': proxy_loss,
            'Loss/Meta_Total': meta_loss,
            'Loss/Proxy_0_Speed': loss_speed_norm,
            'Loss/Proxy_1_Dir': loss_direction,
            'Loss/Proxy_2_Avoid': loss_avoidance,
            'Loss/Proxy_3_Expl': loss_exploration,
            'Loss/Proxy_4_Smooth': loss_smooth,
            'Meta_Comp/Pos': loss_meta_pos,
            'Meta_Comp/Coll': loss_meta_coll,
            'Meta_Comp/Ctrl': loss_meta_ctrl,
            'Metrics/Success_Rate': success.float().mean(),
            'Metrics/Avg_Speed': avg_speed.mean(),
            'Weights/0_Speed': avg_weights[0],
            'Weights/1_Dir': avg_weights[1],
            'Weights/2_Avoid': avg_weights[2],
            'Weights/3_Expl': avg_weights[3],
            'Weights/4_Smooth': avg_weights[4]
        })

        # LGN 阶段特有的日志
        if train_lgn_phase:
            smooth_dict({'Loss/LGN_Dot_Product': lgn_update_loss})

        # 定期写入 TensorBoard (每 25 步)
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()

            # 记录当前状态
            writer.add_scalar('Status/Train_Mode', 1.0 if train_lgn_phase else 0.0, i + 1)

        # 绘图与保存 (is_save_iter)
        if is_save_iter(i):
            torch.save(worknet.state_dict(), f'worker_ckpt_{i:06d}.pth')
            torch.save(lgn.state_dict(), f'lgn_ckpt_{i:06d}.pth')

            # --- 绘图逻辑 (带内存清理) ---
            idx = 0  # 取第一个样本绘图

            # 1. 轨迹图
            fig_p, ax = plt.subplots()
            p_cpu = p_history[:, idx].cpu()
            ax.plot(p_cpu[:, 0], label='x')
            ax.plot(p_cpu[:, 1], label='y')
            ax.plot(p_cpu[:, 2], label='z')
            ax.legend()
            ax.set_title(f"Pos Iter {i}")
            writer.add_figure('Trajectory/Position', fig_p, i + 1)
            plt.close(fig_p)  # [重要] 释放内存

            # 2. 速度图
            fig_v, ax = plt.subplots()
            v_cpu = v_history[:, idx].cpu()
            ax.plot(v_cpu[:, 0], label='vx')
            ax.plot(v_cpu[:, 1], label='vy')
            ax.plot(v_cpu[:, 2], label='vz')
            ax.legend()
            ax.set_title(f"Vel Iter {i}")
            writer.add_figure('Trajectory/Velocity', fig_v, i + 1)
            plt.close(fig_v)

            # 3. 动作图
            fig_a, ax = plt.subplots()
            a_cpu = act_buffer[:, idx].cpu()
            ax.plot(a_cpu[:, 0], label='ax')
            ax.plot(a_cpu[:, 1], label='ay')
            ax.plot(a_cpu[:, 2], label='az')
            ax.legend()
            ax.set_title(f"Act Iter {i}")
            writer.add_figure('Trajectory/Action', fig_a, i + 1)
            plt.close(fig_a)

print("Training Finished.")