import argparse
import math
from collections import defaultdict
from random import normalvariate

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt
from env_cuda import Env
from model import Model

########### 参数配置 ##########
parser = argparse.ArgumentParser()
parser.add_argument('--resume',
                    default="/home/robot/validation_code/high_speed_flight_v1/high_speed_flight/src/e2e_planner_v2/base.pth")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=500000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lgn_lr', type=float, default=5e-4)
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')

args = parser.parse_args()
writer = SummaryWriter()
device = torch.device('cuda')

########## 初始化环境与模型 ##########
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle)


class LossGenNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.conv_embed = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        input_dim = 16 * 6 * 8 + state_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Softmax(dim=-1)
        )

    def forward(self, depth_feat, state):
        d_emb = self.conv_embed(depth_feat)
        x = torch.cat([d_emb, state], dim=-1)
        return self.net(x)


if args.no_odom:
    model = Model(7, 6)
else:
    model = Model(7 + 3, 6)
model = model.to(device)

lgn_state_dim = 7 if args.no_odom else 10
lgn = LossGenNet(state_dim=lgn_state_dim).to(device)

if args.resume:
    model.load_state_dict(torch.load(args.resume, map_location=device), False)

optim_worker = AdamW(model.parameters(), args.lr)
optim_lgn = AdamW(lgn.parameters(), args.lgn_lr)
sched = CosineAnnealingLR(optim_worker, args.num_iters, args.lr * 0.01)

scaler_q = defaultdict(list)


def smooth_dict(ori_dict):
    for k, v in ori_dict.items(): scaler_q[k].append(float(v))


def barrier(x, v_to_pt): return (v_to_pt * (1 - x).relu().pow(2)).mean()


def is_save_iter(i): return (i + 1) % 1000 == 0 if i >= 2000 else (i + 1) % 250 == 0


########## 训练循环 ##########
pbar = tqdm(range(args.num_iters), ncols=100)
B = args.batch_size

for i in pbar:
    env.reset()
    model.reset()
    p_history, v_history, target_v_history, vec_to_pt_history = [], [], [], []
    act_buffer = [env.act] * 2
    trajectory_lgn_weights = []
    target_v_raw = env.p_target - env.p
    h = None

    ###### 1. Rollout ######
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

        R = env.R
        state_list = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.margin[:, None]]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom: state_list.insert(0, local_v)
        state_tensor = torch.cat(state_list, -1)

        x_pooled = F.max_pool2d((3 / depth.clamp_(0.3, 24) - 0.6)[:, None], 4, 4)

        # LGN 输出权重
        current_weights = lgn(x_pooled, state_tensor)
        trajectory_lgn_weights.append(current_weights)

        act, _, h = model(x_pooled, state_tensor, h)
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        real_act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(real_act)
        env.run(real_act, ctl_dt, target_v_raw_curr)

    ###### 2. 损失计算 ######
    p_history = torch.stack(p_history)
    v_history = torch.stack(v_history)
    target_v_history = torch.stack(target_v_history)
    act_buffer = torch.stack(act_buffer)

    # --- 代理目标组件 (Proxy Components) ---
    # 1. 速度大小损失 (目标 5m/s)
    loss_speed_norm = F.smooth_l1_loss(v_history.norm(2, -1), torch.ones_like(v_history.norm(2, -1)) * 5.0)

    # 2. 速度方向损失 (朝着终点)
    target_dir = F.normalize(env.p_target - p_history, dim=-1)
    v_dir = F.normalize(v_history, dim=-1)
    loss_direction = (1.0 - (v_dir * target_dir).sum(-1)).mean()

    # 3. 避障与探索损失
    vec_to_pt = torch.stack(vec_to_pt_history)
    if vec_to_pt.dim() == 4: vec_to_pt = vec_to_pt.mean(1)  # 简化多点
    dist_obj = vec_to_pt.norm(2, -1) - env.margin
    loss_avoidance = F.softplus(-dist_obj * 10.0).mean()

    # 位置重叠损失 (探索)
    if p_history.shape[0] > 20:
        loss_exploration = torch.exp(-torch.norm(p_history[20:] - p_history[:-20], 2, -1)).mean()
    else:
        loss_exploration = torch.tensor(0.0, device=device)

    # 4. 平滑损失
    loss_smooth = act_buffer.diff(1, 0).pow(2).mean()

    # --- 元目标组件 (Meta Components) ---
    # M1: 位置损失 (最终距离)
    loss_meta_pos = torch.norm(p_history[-1] - env.p_target, 2, -1).mean()
    # M2: 碰撞损失
    loss_meta_coll = F.softplus(-dist_obj * 32.0).clamp(max=50.0).mean()
    # M3: 控制量损失 (动作大小)
    loss_meta_ctrl = act_buffer.norm(2, -1).mean()

    ###### 3. 双层优化整合 ######
    avg_weights = torch.stack(trajectory_lgn_weights).mean(dim=[0, 1])

    # 代理损失组合
    proxy_loss = avg_weights[0] * loss_speed_norm + \
                 avg_weights[1] * loss_direction + \
                 avg_weights[2] * (loss_avoidance + loss_exploration) + \
                 avg_weights[3] * loss_smooth

    # 元损失组合
    meta_loss = loss_meta_pos + loss_meta_coll * 5.0 + loss_meta_ctrl * 0.1

    ###### 4. 梯度对齐与更新 ######
    optim_worker.zero_grad();
    optim_lgn.zero_grad()

    # 计算梯度
    grad_proxy = torch.autograd.grad(proxy_loss, model.parameters(), create_graph=True, retain_graph=True,
                                     allow_unused=True)
    grad_meta = torch.autograd.grad(meta_loss, model.parameters(), retain_graph=True, allow_unused=True)

    # LGN 更新 (余弦对齐)
    lgn_loss = 0
    for gp, gm in zip(grad_proxy, grad_meta):
        if gp is not None and gm is not None:
            lgn_loss -= (gp.flatten() * gm.flatten().detach()).sum() / (gp.norm() * gm.norm() + 1e-8)

    if i > 2000:  # Warmup
        lgn_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(lgn.parameters(), 1.0)
        optim_lgn.step()

    # Worker 更新
    proxy_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optim_worker.step()
    sched.step()

    ###### 5. 日志 ######
    pbar.set_description_str(f'P-Loss: {proxy_loss:.3f} | M-Loss: {meta_loss:.3f}')
    if (i + 1) % 25 == 0:
        smooth_dict(
            {'w_speed': avg_weights[0], 'w_dir': avg_weights[1], 'w_obs': avg_weights[2], 'w_smth': avg_weights[3],
             'm_pos': loss_meta_pos, 'm_coll': loss_meta_coll})
        for k, v in scaler_q.items(): writer.add_scalar(k, sum(v) / len(v), i + 1)
        scaler_q.clear()