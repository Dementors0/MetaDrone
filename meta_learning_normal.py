import argparse
import math
from collections import defaultdict
from random import normalvariate
import os
from datetime import datetime

import torch
import torch.nn as nn
# from torch.func import functional_call # 如果 PyTorch 版本 < 2.0，请注释此行或检查兼容性
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt

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
parser.add_argument('--batch_size', type=int, default=256) 
parser.add_argument('--num_iters', type=int, default=500000)

# [优化策略参数]
parser.add_argument('--lgn_steps', type=int, default=1, help='LGN 连续优化的步数')
parser.add_argument('--worker_steps', type=int, default=1000, help='Worker 连续优化的步数')

# 基础物理参数
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=300)
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

writer = SummaryWriter()
print(args)

# 获取当前时间
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = os.path.join('checkpoints', timestamp)
os.makedirs(save_dir, exist_ok=True)
print(f"--> Models will be saved to: {save_dir}")

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
lgn = LossGenNet(state_dim=lgn_state_dim, hidden_dim=128).to(device)

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

# 逐时间步计算重叠损失
def compute_overlap_loss_per_step(p_history, sigma=0.5, time_window=300):
    p_history = p_history.permute(1, 0, 2)
    n_batch, n_points, n_dims = p_history.shape

    if n_points < time_window + 1:
        return torch.zeros((n_batch, n_points), device=p_history.device)

    dist_matrix = torch.cdist(p_history, p_history, p=2)
    overlap_energy = torch.exp(- (dist_matrix ** 2) / (2 * sigma ** 2))

    indices = torch.arange(n_points, device=p_history.device)
    time_diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    mask = (time_diff > time_window).float()

    energy_sum = (overlap_energy * mask.unsqueeze(0)).sum(dim=2)
    mask_sum = mask.sum(dim=1).unsqueeze(0) + 1e-6
    loss_per_step = energy_sum / mask_sum

    return loss_per_step

########## 7. 训练主循环 ##########
pbar = tqdm(range(args.num_iters), ncols=120)
B = args.batch_size
cycle_len = args.lgn_steps + args.worker_steps

for i in pbar:
    # --- 非对称阶段切换逻辑 ---
    cycle_pos = i % cycle_len
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
    act_buffer = [env.act.detach()] * 2
    trajectory_lgn_weights = []
    
    # 初始化GNU的隐状态
    h_worker = None
    h_lgn = None 

    ###### A. 仿真环境中执行一个Rollout######
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

        x_pooled = F.max_pool2d((3 / depth.clamp_(0.3, 24) - 0.6)[:, None], 4, 4)

        # --- LGN Forward ---
        current_weights, h_lgn = lgn(x_pooled, state_tensor, h_lgn)
        trajectory_lgn_weights.append(current_weights)

        # --- Worker Forward ---
        act, _, h_worker = worknet(x_pooled, state_tensor, h_worker)
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        real_act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(real_act)

        env.run(real_act, ctl_dt, target_v_raw_curr)

    ###### B. 损失计算（逐时间步加权) ######
    p_history = torch.stack(p_history)
    v_history = torch.stack(v_history)
    target_v_history = torch.stack(target_v_history)
    act_buffer = torch.stack(act_buffer)
    weights_seq = torch.stack(trajectory_lgn_weights) 

    vec_to_pt = torch.stack(vec_to_pt_history)
    if vec_to_pt.dim() == 4: vec_to_pt = vec_to_pt.mean(1)
    
    # Raw Losses per Step [T, B]
    loss_speed_seq = F.smooth_l1_loss(v_history.norm(2, -1), torch.ones_like(v_history.norm(2, -1)) * 10.0, reduction='none')
    target_dir = F.normalize(env.p_target - p_history, dim=-1)
    v_dir = F.normalize(v_history, dim=-1)
    loss_direction_seq = (1.0 - (v_dir * target_dir).sum(-1))
    dist_obj = vec_to_pt.norm(2, -1) - env.margin
    loss_avoidance_seq = F.softplus(-dist_obj * 10.0)
    loss_exploration_seq = compute_overlap_loss_per_step(p_history, sigma=1.0, time_window=50).permute(1, 0)
    loss_smooth_seq = act_buffer.diff(1, 0)[-args.timesteps:].pow(2).sum(-1)

    # Element-wise Weighted Sum
    weighted_loss_map = (
        (weights_seq[:, :, 0] + 0.1) * loss_speed_seq +
        (weights_seq[:, :, 1] + 0.1) * loss_direction_seq +
        (weights_seq[:, :, 2] + 0.2) * loss_avoidance_seq +
        (weights_seq[:, :, 3] + 0.1) * loss_exploration_seq +
        (weights_seq[:, :, 4] + 0.1) * loss_smooth_seq
    )
    proxy_loss = weighted_loss_map.mean()

    distance = torch.norm(vec_to_pt, 2, -1)
    distance = distance - env.margin

    # Meta Loss (全局)
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    loss_meta_pos = torch.norm(p_history[-1] - env.p_target, 2, -1).mean()
    loss_meta_coll = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()
    loss_meta_ctrl = act_buffer.norm(2, -1).sum()
    meta_loss = loss_meta_pos + loss_meta_coll * 5.0 + loss_meta_ctrl * 0.001

    ###### C. 优化执行 ######
    optim_worker.zero_grad()
    optim_lgn.zero_grad()

    if train_lgn_phase:
        # === 阶段 1: 优化 LGN (Meta-Gradient) ===
        
        # 1. 计算 Proxy 梯度
        grad_proxy = torch.autograd.grad(proxy_loss, worknet.parameters(), create_graph=True, allow_unused=True)
        
        # 2. 计算 Meta 梯度
        grad_meta = torch.autograd.grad(meta_loss, worknet.parameters(), allow_unused=True, retain_graph=True)
        
        # 3. 过滤 None 值并构建有效梯度对
        valid_grad_proxy = []
        valid_grad_meta_neg = []
        
        # 计算 Meta 梯度模长用于归一化
        meta_grad_norm_sq = 0.0
        for gm in grad_meta:
            if gm is not None:
                meta_grad_norm_sq += gm.norm().item() ** 2
        meta_grad_norm = meta_grad_norm_sq ** 0.5 + 1e-6
        clip_coef = 1.0 / meta_grad_norm

        for gp, gm in zip(grad_proxy, grad_meta):
            if gp is not None and gm is not None:
                valid_grad_proxy.append(gp)
                valid_grad_meta_neg.append(gm.detach().neg() * clip_coef)

        # 4. [修改处] 反向传播：添加 retain_graph=True
        # 因为后续还有 reg_loss.backward() 需要用到 LGN 的计算图
        if valid_grad_proxy:
            torch.autograd.backward(valid_grad_proxy, grad_tensors=valid_grad_meta_neg, retain_graph=True)
        
        # 5. 权重正则化
        avg_weight = weights_seq.mean()
        reg_loss = (avg_weight - 1.0).pow(2) * 0.5 
        reg_loss.backward() # 这里不需要 retain_graph 了，因为是最后一次 backward

        nn.utils.clip_grad_norm_(lgn.parameters(), 1.0)
        optim_lgn.step()

    else:
        # === 阶段 2: 优化 Worker ===
        proxy_loss.backward()
        optim_worker.step()
        sched.step()

    ###### D. 日志与 TensorBoard ######
    pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Loss: {meta_loss:.3f} | Success-rate:{torch.all(dist_obj > 0, 0).float().mean()}")

    with torch.no_grad():
        success = torch.all(dist_obj > 0, 0)
        avg_speed = v_history.norm(dim=-1).mean(0)
        avg_weights = weights_seq.mean(dim=[0, 1])

        smooth_dict({
            'Loss_normalize/Proxy_Total': proxy_loss,
            'Loss_normalize/Meta_Total': meta_loss,
            'Loss_normalize/Raw_Speed': loss_speed_seq.mean(),
            'Loss_normalize/Raw_Dir': loss_direction_seq.mean(),
            'Loss_normalize/Raw_Avoid': loss_avoidance_seq.mean(),
            'Loss_normalize/Meta_pos':loss_meta_pos,
            'Loss_normalize/Meta_coll':loss_meta_coll * 5,
            'Loss_normalize/Meta_ctrl':loss_meta_ctrl * 0.001,
            'Metrics/Success_Rate': success.float().mean(),
            'Metrics/Avg_Speed': avg_speed.mean(),
            'Weights/0_Speed': avg_weights[0],
            'Weights/1_Dir': avg_weights[1],
            'Weights/2_Avoid': avg_weights[2],
            'Weights/3_Expl': avg_weights[3],
            'Weights/4_Smooth': avg_weights[4]
        })
        
        if train_lgn_phase:
             smooth_dict({'Loss/LGN_Reg': reg_loss})

        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()
            writer.add_scalar('Status/Train_Mode', 1.0 if train_lgn_phase else 0.0, i + 1)

        if is_save_iter(i):
            save_path_worker = os.path.join(save_dir, f'worker_ckpt_{i:06d}.pth')
            save_path_lgn = os.path.join(save_dir, f'lgn_ckpt_{i:06d}.pth')
            
            torch.save(worknet.state_dict(), save_path_worker)
            torch.save(lgn.state_dict(), save_path_lgn)
            
            # 绘图逻辑
            idx = 0
            fig_p, ax = plt.subplots()
            p_cpu = p_history[:, idx].cpu()
            ax.plot(p_cpu[:, 0], label='x')
            ax.plot(p_cpu[:, 1], label='y')
            ax.legend()
            ax.set_title(f"Pos Iter {i}")
            writer.add_figure('Trajectory/Position', fig_p, i + 1)
            plt.close(fig_p)

            fig_w, ax = plt.subplots()
            w_cpu = weights_seq[:, idx, :].cpu()
            labels = ['Speed', 'Dir', 'Avoid', 'Expl', 'Smooth']
            for wi in range(5):
                ax.plot(w_cpu[:, wi], label=labels[wi])
            ax.legend()
            ax.set_title(f"Weights Profile Iter {i}")
            writer.add_figure('Debug/Weights_Profile', fig_w, i + 1)
            plt.close(fig_w)

print("Training Finished.")