from collections import defaultdict
import datetime
import math
import os
from random import normalvariate
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from env_multi import Env
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
from model import Model

matplotlib.use('Agg', force=True)

try:
    import plotly.graph_objects as go
except Exception:
    go = None

###########参数配置##########

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=500000)
parser.add_argument('--coef_v', type=float, default=1.0, help='smooth l1 of norm(v_set - v_real)')
parser.add_argument('--coef_speed', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_v_pred', type=float, default=2.0, help='mse loss for velocity estimation (no odom)')
parser.add_argument('--coef_collide', type=float, default=2.0, help='softplus loss for collision (large if close to obstacle, zero otherwise)')
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='quadratic clearance loss')
parser.add_argument('--coef_d_acc', type=float, default=0.01, help='control acceleration regularization')
parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='control jerk regularizatinon')
parser.add_argument('--coef_d_snap', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_ground_affinity', type=float, default=0., help='legacy')
parser.add_argument('--coef_bias', type=float, default=0.0, help='legacy')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--obstacle_count_scale', type=float, default=0.5,
                    help='global multiplier for obstacle counts')
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--maze_update_interval', type=int, default=50,
                    help='Regenerate maze every N iterations; drone-only reset in between')
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--terminal_log_interval', type=int, default=500,
                    help='Update terminal progress text every N iterations')
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
parser.add_argument('--include_u_local_optimum', dest='include_u_local_optimum', action='store_true')
parser.add_argument('--no_include_u_local_optimum', dest='include_u_local_optimum', action='store_false')
parser.set_defaults(include_u_local_optimum=False)
parser.add_argument('--compact_two_zone_map', dest='compact_two_zone_map', action='store_true')
parser.add_argument('--no_compact_two_zone_map', dest='compact_two_zone_map', action='store_false')
parser.set_defaults(compact_two_zone_map=False)
parser.add_argument('--wall_physical_feedback', dest='wall_physical_feedback', action='store_true')
parser.add_argument('--no_wall_physical_feedback', dest='wall_physical_feedback', action='store_false')
parser.set_defaults(wall_physical_feedback=False)
parser.add_argument('--exp_name', type=str, default='main_cuda')
args = parser.parse_args()

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = os.path.join('..', 'checkpoints', f"main_cuda_{args.exp_name}_{current_time}")
video_dir = os.path.join(save_dir, 'videos')
os.makedirs(video_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
print(args)
print(f"Training artifacts will be saved to: {save_dir}")

device = torch.device('cuda')

##########初始化仿真环境##########
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle,
          obstacle_count_scale=args.obstacle_count_scale,
          include_u_local_optimum=args.include_u_local_optimum,
          compact_two_zone_map=args.compact_two_zone_map,
          wall_physical_feedback=args.wall_physical_feedback)

##########初始化神经网络##########
if args.no_odom:
    model = Model(7, 6)
else:
    model = Model(7+3, 6)
model = model.to(device)

##########使用预训练模型/继续训练原有的模型##########
if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)

##########优化器##########
optim = AdamW(model.parameters(), args.lr)
##########学习率调度器(余弦曲线Cosine)##########
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

##########控制时间，每秒控制15次##########
ctl_dt = 1 / 15

##########数据收集##########
scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

##########碰撞损失##########
def barrier(x: torch.Tensor, v_to_pt):
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

##########动态保存策略##########
def is_save_iter(i):
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0


def is_save_trajectory_iter(i):
    return i == 0 or (i + 1) % 250 == 0


def rotation_matrix_to_rpy_deg(R):
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]
    r10 = R[..., 1, 0]
    r00 = R[..., 0, 0]

    pitch = torch.asin(torch.clamp(-r20, -1.0, 1.0))
    roll = torch.atan2(r21, r22)
    yaw = torch.atan2(r10, r00)
    return torch.stack([roll, pitch, yaw], dim=-1) * (180.0 / math.pi)


def _plotly_add_cuboid(fig, cx, cy, cz, hx, hy, hz, color='lightgray', opacity=0.68):
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    tri = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
        color=color, opacity=opacity, flatshading=True, hoverinfo='skip', showlegend=False
    ))


def _plotly_add_sphere(fig, cx, cy, cz, r, color='royalblue', opacity=0.58, res_u=18, res_v=14):
    u = np.linspace(0.0, 2.0 * np.pi, res_u)
    v = np.linspace(0.0, np.pi, res_v)
    uu, vv = np.meshgrid(u, v)
    x = cx + r * np.cos(uu) * np.sin(vv)
    y = cy + r * np.sin(uu) * np.sin(vv)
    z = cz + r * np.cos(vv)
    fig.add_trace(go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=np.zeros_like(z),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=opacity,
        hoverinfo='skip',
        showlegend=False,
    ))


def _plotly_add_vertical_cylinder(fig, cx, cy, r, z0, z1, color='orange', opacity=0.52, res_theta=28, res_z=8):
    theta = np.linspace(0.0, 2.0 * np.pi, res_theta)
    z = np.linspace(z0, z1, res_z)
    tt, zz = np.meshgrid(theta, z)
    x = cx + r * np.cos(tt)
    y = cy + r * np.sin(tt)
    fig.add_trace(go.Surface(
        x=x,
        y=y,
        z=zz,
        surfacecolor=np.zeros_like(zz),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=opacity,
        hoverinfo='skip',
        showlegend=False,
    ))


def _is_outer_shell_voxel(env, cx, cy, cz, hx, hy, hz, eps=1e-2):
    x_max = float(getattr(env, 'map_x_max', 10.0))
    y_half = float(getattr(env, 'map_y_half', 12.0))
    y_min = float(getattr(env, 'map_y_min', -y_half))
    y_max = float(getattr(env, 'map_y_max', y_half))
    z_max = float(getattr(env, 'map_z_max', 5.0))
    boundary_half = float(getattr(env, 'boundary_half', 0.05))
    spawn_z_center = float(getattr(env, 'spawn_z_center', 2.5))
    spawn_x_center = float(getattr(env, 'spawn_x_center', x_max * 0.5))

    is_floor = abs(cz - 0.0) <= eps and abs(hz - boundary_half) <= eps
    is_ceiling = abs(cz - z_max) <= eps and abs(hz - boundary_half) <= eps

    is_x_side = (
        abs(hx - boundary_half) <= eps
        and abs(hy - y_half) <= eps
        and abs(cz - spawn_z_center) <= eps
        and (abs(cx - 0.0) <= eps or abs(cx - x_max) <= eps)
    )
    is_y_side = (
        abs(hy - boundary_half) <= eps
        and abs(hx - spawn_x_center) <= eps
        and abs(cz - spawn_z_center) <= eps
        and (abs(cy - y_min) <= eps or abs(cy - y_max) <= eps)
    )

    return is_floor, (is_ceiling or is_x_side or is_y_side)


def save_interactive_3d_html(html_path, env, p_cpu, v_cpu, R_cpu=None, idx=0, axis_len=0.25, axis_step=5):
    if go is None:
        return False

    traj_xyz = p_cpu.numpy()
    speed_cpu = v_cpu.norm(dim=-1).numpy()
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=traj_xyz[:, 0], y=traj_xyz[:, 1], z=traj_xyz[:, 2],
        mode='lines+markers',
        marker=dict(size=3, color=speed_cpu, colorscale='Turbo', colorbar=dict(title='Speed (m/s)')),
        line=dict(color='limegreen', width=5),
        name='Trajectory'
    ))

    if hasattr(env, 'voxels') and env.voxels.numel() > 0:
        vox = env.voxels[idx].detach().cpu().numpy()
        for box in vox[:180]:
            cx, cy, cz, hx, hy, hz = box.tolist()
            if hx > 20 or hy > 20 or hz > 20:
                continue
            is_floor, is_shell_not_floor = _is_outer_shell_voxel(env, cx, cy, cz, hx, hy, hz)
            if is_shell_not_floor:
                continue
            _plotly_add_cuboid(fig, cx, cy, cz, hx, hy, hz)

    if hasattr(env, 'balls') and env.balls.numel() > 0:
        balls = env.balls[idx].detach().cpu().numpy()
        for ball in balls:
            cx, cy, cz, r = ball.tolist()
            if r <= 1e-6:
                continue
            _plotly_add_sphere(fig, cx, cy, cz, r)

    if hasattr(env, 'cyl') and env.cyl.numel() > 0:
        cyls = env.cyl[idx].detach().cpu().numpy()
        z0 = 0.0
        z1 = float(getattr(env, 'map_z_max', 5.0))
        for cyl in cyls:
            cx, cy, r = cyl.tolist()
            if r <= 1e-6:
                continue
            _plotly_add_vertical_cylinder(fig, cx, cy, r, z0=z0, z1=z1)

    if R_cpu is not None:
        R_np = R_cpu.numpy()
        for t in range(0, len(traj_xyz), axis_step):
            pos = traj_xyz[t]
            Rm = R_np[t]
            for c, label, axis_i in [('red', 'X-axis', 0), ('green', 'Y-axis', 1), ('blue', 'Z-axis', 2)]:
                axis = Rm[:, axis_i] * axis_len
                fig.add_trace(go.Scatter3d(
                    x=[pos[0], pos[0] + axis[0]],
                    y=[pos[1], pos[1] + axis[1]],
                    z=[pos[2], pos[2] + axis[2]],
                    mode='lines',
                    line=dict(color=c, width=4),
                    showlegend=(t == 0),
                    name=label if t == 0 else None,
                ))

    fig.add_trace(go.Scatter3d(x=[traj_xyz[0, 0]], y=[traj_xyz[0, 1]], z=[traj_xyz[0, 2]], mode='markers',
                               marker=dict(size=6, color='green', symbol='circle'), name='Start'))
    fig.add_trace(go.Scatter3d(x=[traj_xyz[-1, 0]], y=[traj_xyz[-1, 1]], z=[traj_xyz[-1, 2]], mode='markers',
                               marker=dict(size=6, color='black', symbol='x'), name='End'))
    tgt = env.p_target[idx].detach().cpu().numpy()
    fig.add_trace(go.Scatter3d(x=[tgt[0]], y=[tgt[1]], z=[tgt[2]], mode='markers',
                               marker=dict(size=8, color='red', symbol='diamond'), name='Goal'))

    fig.update_layout(
        title='Interactive 3D Trajectory & Obstacles',
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        template='plotly_white',
        showlegend=True,
        margin=dict(l=5, r=5, t=40, b=5)
    )
    fig.write_html(html_path, include_plotlyjs='cdn')
    return True

##########pbar创建进度条，同时作为迭代器##########
pbar = tqdm(range(args.num_iters), ncols=120, miniters=max(1, int(args.terminal_log_interval)))
# depths = []
# states = []
B = args.batch_size
maze_update_counter = 0
for i in pbar:
    ######重置环境和模型######
    if i == 0 or (maze_update_counter % max(1, int(args.maze_update_interval)) == 0):
        env.reset()
    else:
        env.reset_drone_only()
    maze_update_counter += 1
    model.reset()
    ######初始化数据记录容器######
    p_history = []
    v_history = []
    R_history = []
    rpy_history = []
    act_cmd_history = []
    target_v_history = []
    vec_to_pt_history = []
    act_diff_history = []
    v_preds = []
    vid = []
    v_net_feats = []
    h = None

    ######模拟控制延迟######
    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)
    ######计算初始目标向量######
    target_v_raw = env.p_target - env.p
    ######偏航角角速度偏移######
    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)

    ######开始飞行仿真循环######
    for t in range(args.timesteps):####飞行总步长####
        ####模拟真实硬件控制时间误差####
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        ####生成视觉感知####
        depth, flow = env.render(ctl_dt)
        ####记录位置和最近障碍物向量####
        p_history.append(env.p)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())

        ####保存视频####
        if is_save_iter(i):
            vid.append(depth[4])

        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()

        ####仿真器执行一个时间步####
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        ####构建航向旋转矩阵####
        ##去除了滚转（Roll）和俯仰（Pitch）”的纯偏航（Yaw）旋转矩阵##
        R = env.R
        fwd = env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)

        ####计算理想参考速度####
        ##计算到目标的距离 target_v_raw 的模长##
        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        ##计算方向单位向量 (归一化)##
        target_v_unit = target_v_raw / target_v_norm
        # 速度限幅
        # 如果距离很远 (100m)，不要试图以 100m/s 飞过去，而是限制在 max_speed (比如 10m/s)。
        # 如果距离很近 (0.5m)，则速度就设为 0.5m/s (慢慢靠近)。
        target_v = target_v_unit * torch.clamp(target_v_norm, max=float(env.max_speed))
        ####组装喂给神经网络的“状态包”####
        state = [
            ##第一行相对目标向量##
            torch.squeeze(target_v[:, None] @ R, 1),
            ##机体 Z 轴在世界坐标系下的方向，代表重力向量 / 姿态感##
            env.R[:, 2],
            ##安全半径##
            env.margin[:, None]]
        ####计算 无人机相对于自身机头方向的飞行速度####
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        ####把列表里的所有张量在最后一个维度拼接起来####
        state = torch.cat(state, -1)

        # normalize
        ####视觉预处理####
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        ##最大值池化,长和宽都缩小 4 倍##
        x = F.max_pool2d(x[:, None], 4, 4)
        act, values, h = model(x, state, h)

        #神经网络预测加速度，预测速度，垃圾桶变量1
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
        act_cmd_history.append(act_buffer[t])
        v_net_feats.append(torch.cat([act, local_v, h], -1))
        v_history.append(env.v)
        R_history.append(env.R)
        rpy_history.append(rotation_matrix_to_rpy_deg(env.R))
        target_v_history.append(target_v)

    ####撞地损失####
    p_history = torch.stack(p_history)
    R_history = torch.stack(R_history)
    rpy_history = torch.stack(rpy_history)
    act_cmd_history = torch.stack(act_cmd_history)
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
    act_buffer = torch.stack(act_buffer)

    ####速度大小损失####
    v_history = torch.stack(v_history)
    v_history_cum = v_history.cumsum(0)#对速度累积求和
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30 #最近30帧的平均速度
    target_v_history = torch.stack(target_v_history)
    T, B, _ = v_history.shape
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    ####速度预测误差####
    v_preds = torch.stack(v_preds)
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())

    ####飞行偏离损失####
    ##强迫无人机把速度用在指向目标的方向，不要产生无用的横向漂移##
    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3

    ####飞行平滑损失####
    jerk_history = act_buffer.diff(1, 0).mul(15)
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()
    loss_d_snap = snap_history.pow(2).sum(-1).mean()

    ####避障损失和碰撞损失####
    vec_to_pt_history = torch.stack(vec_to_pt_history)
    distance = torch.norm(vec_to_pt_history, 2, -1)
    distance = distance - env.margin
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

    ####纵向速度误差####
    ##朝着目标方向的有效速度达到一定值##
    speed_history = v_history.norm(2, -1)
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_bias * loss_bias + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_d_snap * loss_d_snap + \
        args.coef_speed * loss_speed + \
        args.coef_v_pred * loss_v_pred + \
        args.coef_collide * loss_collide + \
        args.coef_ground_affinity + loss_ground_affinity

    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    pbar.set_description_str(f'loss: {loss:.3f}')
    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()

    ######接下来运行的代码，不需要计算梯度######
    with torch.no_grad():
        avg_speed = speed_history.mean(0)
        success = torch.all(distance.flatten(0, 1) > 0, 0)
        _success = success.sum() / B
        smooth_dict({
            'loss': loss,
            'loss_v': loss_v,
            'loss_v_pred': loss_v_pred,
            'loss_obj_avoidance': loss_obj_avoidance,
            'loss_d_acc': loss_d_acc,
            'loss_d_jerk': loss_d_jerk,
            'loss_d_snap': loss_d_snap,
            'loss_bias': loss_bias,
            'loss_speed': loss_speed,
            'loss_collide': loss_collide,
            'loss_ground_affinity': loss_ground_affinity,
            'success': _success,
            'max_speed': speed_history.max(0).values.mean(),
            'avg_speed': avg_speed.mean(),
            'ar': (success * avg_speed).mean()})
        log_dict = {}
        if is_save_trajectory_iter(i):
            # vid = torch.stack(vid).cpu().div(10).clamp(0, 1)[None, :, None]
            idx = min(4, B - 1)

            p_cpu = p_history[:, idx].detach().cpu()
            v_cpu = v_history[:, idx].detach().cpu()
            rpy_cpu = rpy_history[:, idx].detach().cpu()
            act_cpu = act_cmd_history[:, idx].detach().cpu()

            fig_p, ax = plt.subplots()
            ax.plot(p_cpu[:, 0], label='x')
            ax.plot(p_cpu[:, 1], label='y')
            ax.plot(p_cpu[:, 2], label='z')
            ax.legend(); ax.set_title(f"Iter {i} Position (Time Series)")

            fig_v, ax = plt.subplots()
            ax.plot(v_cpu[:, 0], label='vx'); ax.plot(v_cpu[:, 1], label='vy'); ax.plot(v_cpu[:, 2], label='vz')
            ax.plot(v_cpu.norm(dim=-1), label='speed', linestyle='--')
            ax.legend(); ax.set_title(f"Iter {i} Velocity (Time Series)")

            fig_rpy, ax = plt.subplots()
            ax.plot(rpy_cpu[:, 0], label='roll(deg)')
            ax.plot(rpy_cpu[:, 1], label='pitch(deg)')
            ax.plot(rpy_cpu[:, 2], label='yaw(deg)')
            ax.legend(); ax.set_title(f"Iter {i} Attitude RPY (Time Series)")

            fig_a, ax = plt.subplots()
            ax.plot(act_cpu[:, 0], label='ax_cmd'); ax.plot(act_cpu[:, 1], label='ay_cmd'); ax.plot(act_cpu[:, 2], label='az_cmd')
            ax.plot(act_cpu.norm(dim=-1), label='|a_cmd|', linestyle='--')
            ax.legend(); ax.set_title(f"Iter {i} Control Accel Cmd (Time Series)")

            interactive_html = os.path.join(video_dir, f'trajectory3d_iter_{i+1:06d}.html')
            R_cpu = R_history[:, idx].detach().cpu()
            if save_interactive_3d_html(interactive_html, env, p_cpu, v_cpu, R_cpu=R_cpu, idx=idx):
                writer.add_text('Trajectory/Interactive3D_HTML', interactive_html, i + 1)

            # writer.add_video('demo', vid, i + 1, 15)
            writer.add_figure('Trajectory/Position_Series', fig_p, i + 1)
            writer.add_figure('Trajectory/Velocity_Series', fig_v, i + 1)
            writer.add_figure('Trajectory/Attitude_RPY_Series', fig_rpy, i + 1)
            writer.add_figure('Trajectory/Control_Accel_Cmd_Series', fig_a, i + 1)
            plt.close(fig_p)
            plt.close(fig_v)
            plt.close(fig_rpy)
            plt.close(fig_a)
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint{i//10000:04d}.pth'))
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()
