import argparse
import datetime
import json
import math
import os
import random
from random import normalvariate
from typing import Dict, List

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env_multi import Env
from model import Model

try:
    import plotly.graph_objects as go
except Exception:
    go = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on precomputed maps and log TensorBoard + HTML trajectories.")
    parser.add_argument("--checkpoint", type=str, default="/home/robot/transformer/multi_pub/checkpoint0004.pth")
    parser.add_argument("--precomputed_map_dir", type=str, default="/home/robot/transformer/precomputed_maps_compact")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=150)
    parser.add_argument("--goal_radius", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_root", type=str, default="../checkpoints")
    parser.add_argument("--exp_name", type=str, default="checkpoint0004_precomputed_eval")
    parser.add_argument("--save_html_every", type=int, default=1, help="Save one trajectory html every N episodes.")
    parser.add_argument("--html_sample_idx", type=int, default=0, help="Which batch index to visualize in html.")
    parser.add_argument("--depth_noise_std", type=float, default=0.02)
    parser.add_argument("--stochastic_ctl_dt", dest="stochastic_ctl_dt", action="store_true")
    parser.add_argument("--no_stochastic_ctl_dt", dest="stochastic_ctl_dt", action="store_false")
    parser.set_defaults(stochastic_ctl_dt=True)
    parser.add_argument("--single", default=False, action="store_true")
    parser.add_argument("--gate", default=False, action="store_true")
    parser.add_argument("--ground_voxels", default=False, action="store_true")
    parser.add_argument("--scaffold", default=False, action="store_true")
    parser.add_argument("--random_rotation", default=False, action="store_true")
    parser.add_argument("--yaw_drift", default=False, action="store_true")
    parser.add_argument("--no_odom", default=False, action="store_true")
    parser.add_argument("--include_u_local_optimum", dest="include_u_local_optimum", action="store_true")
    parser.add_argument("--no_include_u_local_optimum", dest="include_u_local_optimum", action="store_false")
    parser.set_defaults(include_u_local_optimum=False)
    parser.add_argument("--compact_two_zone_map", dest="compact_two_zone_map", action="store_true")
    parser.add_argument("--no_compact_two_zone_map", dest="compact_two_zone_map", action="store_false")
    parser.set_defaults(compact_two_zone_map=True)
    parser.add_argument("--wall_physical_feedback", dest="wall_physical_feedback", action="store_true")
    parser.add_argument("--no_wall_physical_feedback", dest="wall_physical_feedback", action="store_false")
    parser.set_defaults(wall_physical_feedback=False)
    parser.add_argument("--grad_decay", type=float, default=0.4)
    parser.add_argument("--speed_mtp", type=float, default=1.0)
    parser.add_argument("--obstacle_count_scale", type=float, default=0.5)
    parser.add_argument("--fov_x_half_tan", type=float, default=0.53)
    parser.add_argument("--cam_angle", type=int, default=10)
    parser.add_argument("--attitude_model", type=str, default="legacy", choices=["legacy", "v2"])
    parser.add_argument("--yaw_control_source", type=str, default="rule", choices=["rule", "model"])
    parser.add_argument("--yaw_rate_max_deg", type=float, default=150.0)
    parser.add_argument("--yaw_cmd_warmup_iters", type=int, default=2000)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_yaw_frame(R):
    fwd = R[:, :, 0]
    zeros = torch.zeros_like(fwd)
    up = zeros.clone()
    up[:, 2] = 1.0
    fwd_h_raw = torch.stack([fwd[:, 0], fwd[:, 1], torch.zeros_like(fwd[:, 2])], dim=-1)
    fwd_h_norm = torch.norm(fwd_h_raw, 2, -1, keepdim=True)
    fallback = zeros.clone()
    fallback[:, 0] = 1.0
    fwd_h = torch.where(fwd_h_norm > 1e-6, fwd_h_raw / fwd_h_norm.clamp_min(1e-6), fallback)
    left = F.normalize(torch.cross(up, fwd_h, dim=-1), 2, -1, eps=1e-6)
    return torch.stack([fwd_h, left, up], -1)


def compute_heading_reference(env, R_yaw, yaw_rate_max_value, yaw_ref_kp=3.0):
    target_vec = env.p_target - env.p.detach()
    zeros = torch.zeros_like(target_vec[:, 2])
    heading_ref_world = torch.stack([target_vec[:, 0], target_vec[:, 1], zeros], dim=-1)
    heading_norm = torch.norm(heading_ref_world, 2, -1, keepdim=True)
    fallback = R_yaw[:, :, 0]
    heading_ref_world = torch.where(
        heading_norm > 1e-6,
        heading_ref_world / heading_norm.clamp_min(1e-6),
        fallback,
    )
    heading_ref_local = torch.squeeze(heading_ref_world[:, None] @ R_yaw, 1)
    yaw_error = torch.atan2(heading_ref_local[:, 1], heading_ref_local[:, 0]).unsqueeze(-1)
    yaw_rate_ref = torch.clamp(float(yaw_ref_kp) * yaw_error, -float(yaw_rate_max_value), float(yaw_rate_max_value))
    return heading_ref_world, heading_ref_local[:, :2], yaw_rate_ref, yaw_error


def decode_worker_action(act, R_yaw, yaw_rate_max_value):
    B_local = act.shape[0]
    act6 = act[:, :6]
    a_pred, v_pred = (R_yaw @ act6.reshape(B_local, 3, 2)).unbind(-1)
    yaw_rate_cmd = None
    if act.shape[-1] > 6:
        yaw_rate_cmd = torch.tanh(act[:, 6:7]) * float(yaw_rate_max_value)
    return a_pred, v_pred, yaw_rate_cmd


def load_compatible_checkpoint(module, path, device):
    if not path:
        raise ValueError("--checkpoint is empty")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    target_state = module.state_dict()
    compatible_state = {}
    resized_keys = []
    skipped_keys = []
    for key, value in state_dict.items():
        if key not in target_state:
            skipped_keys.append(key)
            continue
        target_value = target_state[key]
        if value.shape == target_value.shape:
            compatible_state[key] = value
            continue
        if value.dim() != target_value.dim():
            skipped_keys.append(key)
            continue
        expanded = torch.zeros_like(target_value)
        slices = tuple(slice(0, min(value.shape[d], target_value.shape[d])) for d in range(value.dim()))
        expanded[slices] = value[slices]
        compatible_state[key] = expanded
        resized_keys.append((key, tuple(value.shape), tuple(target_value.shape)))

    missing_keys, unexpected_keys = module.load_state_dict(compatible_state, strict=False)
    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "resized_keys": resized_keys,
        "skipped_keys": skipped_keys,
    }


def _plotly_add_cuboid(fig, cx, cy, cz, hx, hy, hz, color="lightgray", opacity=0.68):
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    verts = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float32)
    tri = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ], dtype=np.int32)
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=tri[:, 0],
        j=tri[:, 1],
        k=tri[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        hoverinfo="skip",
        showlegend=False,
    ))


def _plotly_add_sphere(fig, cx, cy, cz, r, color="royalblue", opacity=0.58, res_u=18, res_v=14):
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
        hoverinfo="skip",
        showlegend=False,
    ))


def _plotly_add_vertical_cylinder(fig, cx, cy, r, z0, z1, color="orange", opacity=0.52, res_theta=28, res_z=8):
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
        hoverinfo="skip",
        showlegend=False,
    ))


def _is_outer_shell_voxel(env, cx, cy, cz, hx, hy, hz, eps=1e-2):
    x_max = float(getattr(env, "map_x_max", 10.0))
    y_half = float(getattr(env, "map_y_half", 12.0))
    y_min = float(getattr(env, "map_y_min", -y_half))
    y_max = float(getattr(env, "map_y_max", y_half))
    z_max = float(getattr(env, "map_z_max", 5.0))
    boundary_half = float(getattr(env, "boundary_half", 0.05))
    spawn_z_center = float(getattr(env, "spawn_z_center", 2.5))
    spawn_x_center = float(getattr(env, "spawn_x_center", x_max * 0.5))

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
        x=traj_xyz[:, 0],
        y=traj_xyz[:, 1],
        z=traj_xyz[:, 2],
        mode="lines+markers",
        marker=dict(size=3, color=speed_cpu, colorscale="Turbo", colorbar=dict(title="Speed (m/s)")),
        line=dict(color="limegreen", width=5),
        name="Trajectory",
    ))

    if hasattr(env, "voxels") and env.voxels.numel() > 0:
        vox = env.voxels[idx].detach().cpu().numpy()
        for box in vox[:180]:
            cx, cy, cz, hx, hy, hz = box.tolist()
            if hx > 20 or hy > 20 or hz > 20:
                continue
            is_floor, is_shell_not_floor = _is_outer_shell_voxel(env, cx, cy, cz, hx, hy, hz)
            if is_shell_not_floor:
                continue
            if is_floor:
                _plotly_add_cuboid(fig, cx, cy, cz, hx, hy, hz, color="silver", opacity=0.35)
            else:
                _plotly_add_cuboid(fig, cx, cy, cz, hx, hy, hz)

    if hasattr(env, "balls") and env.balls.numel() > 0:
        balls = env.balls[idx].detach().cpu().numpy()
        for ball in balls:
            cx, cy, cz, r = ball.tolist()
            if r <= 1e-6:
                continue
            _plotly_add_sphere(fig, cx, cy, cz, r)

    if hasattr(env, "cyl") and env.cyl.numel() > 0:
        cyls = env.cyl[idx].detach().cpu().numpy()
        z0 = 0.0
        z1 = float(getattr(env, "map_z_max", 5.0))
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
            for c, label, axis_i in [("red", "X-axis", 0), ("green", "Y-axis", 1), ("blue", "Z-axis", 2)]:
                axis = Rm[:, axis_i] * axis_len
                fig.add_trace(go.Scatter3d(
                    x=[pos[0], pos[0] + axis[0]],
                    y=[pos[1], pos[1] + axis[1]],
                    z=[pos[2], pos[2] + axis[2]],
                    mode="lines",
                    line=dict(color=c, width=4),
                    showlegend=(t == 0),
                    name=label if t == 0 else None,
                ))

    fig.add_trace(go.Scatter3d(
        x=[traj_xyz[0, 0]],
        y=[traj_xyz[0, 1]],
        z=[traj_xyz[0, 2]],
        mode="markers",
        marker=dict(size=6, color="green", symbol="circle"),
        name="Start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[traj_xyz[-1, 0]],
        y=[traj_xyz[-1, 1]],
        z=[traj_xyz[-1, 2]],
        mode="markers",
        marker=dict(size=6, color="black", symbol="x"),
        name="End",
    ))
    tgt = env.p_target[idx].detach().cpu().numpy()
    fig.add_trace(go.Scatter3d(
        x=[tgt[0]],
        y=[tgt[1]],
        z=[tgt[2]],
        mode="markers",
        marker=dict(size=8, color="red", symbol="diamond"),
        name="Goal",
    ))

    fig.update_layout(
        title="Interactive 3D Trajectory & Obstacles",
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data"),
        template="plotly_white",
        showlegend=True,
        margin=dict(l=5, r=5, t=40, b=5),
    )
    fig.write_html(html_path, include_plotlyjs="cdn")
    return True


def _align_env_goal_planes_to_precomputed_map(map_data, env_obj):
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

    if start_y_from_map is not None:
        env_obj.spawn_start_y = float(start_y_from_map)
    if goal_y_from_map is not None:
        env_obj.spawn_goal_y = float(goal_y_from_map)


def enforce_eval_drone_specs(env_obj, drone_radius_m=0.13, margin_m=0.07):
    """Force fixed drone radius/margin for evaluation consistency."""
    env_obj.drone_radius = float(drone_radius_m)
    B = int(getattr(env_obj, "batch_size", 1))
    device = getattr(env_obj, "device", None)
    if device is None:
        if hasattr(env_obj, "margin") and isinstance(env_obj.margin, torch.Tensor):
            device = env_obj.margin.device
        else:
            device = torch.device("cpu")
    env_obj.margin = torch.full((B,), float(margin_m), device=device, dtype=torch.float32)


def collect_map_files(map_dir: str) -> List[str]:
    if not os.path.isdir(map_dir):
        return []
    names = []
    for name in sorted(os.listdir(map_dir)):
        if name.startswith("map_") and name.endswith(".pt"):
            names.append(os.path.join(map_dir, name))
    return names


@torch.no_grad()
def run_single_episode(env, model, args, device, use_attitude_v2, yaw_rate_max):
    B = env.batch_size
    model.reset()
    model.eval()

    p_history = []
    v_history = []
    R_history = []
    rpy_history = []
    vec_to_pt_history = []
    yaw_error_history = []
    ctl_dt_history = []
    h = None

    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)
    yaw_rate_buffer = [torch.zeros((B, 1), device=device)] * (act_lag + 1)

    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av),
            -torch.sin(drift_av),
            zeros,
            torch.sin(drift_av),
            torch.cos(drift_av),
            zeros,
            zeros,
            zeros,
            ones,
        ], -1).reshape(B, 3, 3)
    else:
        R_drift = None

    for _ in range(args.timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15) if args.stochastic_ctl_dt else (1 / 15)
        ctl_dt_history.append(float(ctl_dt))
        depth, _ = env.render(ctl_dt)
        p_history.append(env.p.detach())
        vec_to_pt_history.append(env.find_vec_to_nearest_pt().detach())

        if args.yaw_drift and R_drift is not None:
            target_v_raw = env.p_target - env.p.detach()
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()

        if use_attitude_v2:
            R_yaw_pre = build_yaw_frame(env.R)
            heading_ref_world_pre, _, _, _ = compute_heading_reference(env, R_yaw_pre, yaw_rate_max)
            yaw_rate_step = yaw_rate_buffer[-(act_lag + 1)] if args.yaw_control_source == "model" else None
            env.run(
                act_buffer[-(act_lag + 1)],
                ctl_dt,
                heading_ref=heading_ref_world_pre,
                yaw_rate_cmd=yaw_rate_step,
                yaw_rate_max=yaw_rate_max,
            )
        else:
            env.run(act_buffer[-(act_lag + 1)], ctl_dt, target_v_raw)

        R = build_yaw_frame(env.R)
        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm.clamp_min(1e-6)
        target_v = target_v_unit * torch.clamp(target_v_norm, max=float(env.max_speed))

        state = [
            torch.squeeze(target_v[:, None] @ R, 1),
            env.R[:, 2],
            env.margin[:, None],
        ]
        if use_attitude_v2:
            _, heading_ref_local_xy, yaw_rate_ref, yaw_error = compute_heading_reference(env, R, yaw_rate_max)
            yaw_rate_norm = getattr(env, "yaw_rate", torch.zeros((B, 1), device=device)) / float(yaw_rate_max)
            state.extend([heading_ref_local_xy, yaw_rate_norm])
            yaw_error_history.append(yaw_error.detach())

        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)

        x = 3 / depth.clamp(0.3, 24) - 0.6
        if args.depth_noise_std > 0:
            x = x + torch.randn_like(depth) * float(args.depth_noise_std)
        x = F.max_pool2d(x[:, None], 4, 4)
        act, _, h = model(x, state, h)

        if use_attitude_v2:
            a_pred, v_pred, yaw_rate_cmd = decode_worker_action(act, R, yaw_rate_max)
        else:
            a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
            yaw_rate_cmd = None

        act_world = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act_world)

        if use_attitude_v2:
            if args.yaw_control_source == "model":
                yaw_rate_used = yaw_rate_cmd
            else:
                yaw_rate_used = yaw_rate_ref.detach()
            yaw_rate_buffer.append(yaw_rate_used)

        v_history.append(env.v.detach())
        R_history.append(env.R.detach())
        rpy_history.append(rotation_matrix_to_rpy_deg(env.R).detach())

    p_history = torch.stack(p_history, dim=0)
    v_history = torch.stack(v_history, dim=0)
    R_history = torch.stack(R_history, dim=0)
    rpy_history = torch.stack(rpy_history, dim=0)
    vec_to_pt_history = torch.stack(vec_to_pt_history, dim=0)

    p_history_full = torch.cat([p_history, env.p.detach().unsqueeze(0)], dim=0)
    v_history_full = torch.cat([v_history, env.v.detach().unsqueeze(0)], dim=0)
    R_history_full = torch.cat([R_history, env.R.detach().unsqueeze(0)], dim=0)

    distance = torch.norm(vec_to_pt_history, 2, -1) - env.margin.unsqueeze(0)
    collision_free = torch.all(distance > 0, dim=0)
    collision_step_ratio = (distance <= 0).float().mean(dim=0)
    collision_depth_max = F.relu(-distance).max(dim=0).values
    min_clearance = distance.min(dim=0).values

    dist_to_goal = torch.norm(p_history_full - env.p_target.unsqueeze(0), 2, -1)
    reached_goal = torch.any(dist_to_goal < float(args.goal_radius), dim=0)
    final_dist_to_goal = dist_to_goal[-1]
    start_dist_to_goal = dist_to_goal[0]
    progress_ratio = (start_dist_to_goal - final_dist_to_goal) / start_dist_to_goal.clamp_min(1e-6)

    speed_history = v_history_full.norm(2, -1)
    avg_speed = speed_history.mean(dim=0)
    max_speed = speed_history.max(dim=0).values

    path_length = torch.norm(p_history_full[1:] - p_history_full[:-1], 2, -1).sum(dim=0)

    cumulative_t = np.concatenate([[0.0], np.cumsum(np.asarray(ctl_dt_history, dtype=np.float64))], axis=0)
    hit_mask = dist_to_goal < float(args.goal_radius)
    time_to_goal_sec = torch.full((B,), float(cumulative_t[-1]), device=device)
    time_to_goal_step = torch.full((B,), float(args.timesteps + 1), device=device)
    for b in range(B):
        hit_idx = torch.nonzero(hit_mask[:, b], as_tuple=False)
        if hit_idx.numel() > 0:
            first_idx = int(hit_idx[0, 0].item())
            time_to_goal_step[b] = float(first_idx)
            time_to_goal_sec[b] = float(cumulative_t[first_idx])

    if len(yaw_error_history) > 0:
        yaw_error_abs_deg = torch.stack(yaw_error_history).abs().mean(dim=0).squeeze(-1) * (180.0 / math.pi)
    else:
        yaw_error_abs_deg = torch.zeros((B,), device=device)

    return {
        "collision_free": collision_free,
        "reached_goal": reached_goal,
        "collision_step_ratio": collision_step_ratio,
        "collision_depth_max": collision_depth_max,
        "min_clearance": min_clearance,
        "final_dist_to_goal": final_dist_to_goal,
        "progress_ratio": progress_ratio,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "path_length": path_length,
        "time_to_goal_step": time_to_goal_step,
        "time_to_goal_sec": time_to_goal_sec,
        "yaw_error_abs_deg": yaw_error_abs_deg,
        "p_history_full": p_history_full,
        "v_history_full": v_history_full,
        "R_history_full": R_history_full,
        "rpy_history": rpy_history,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    save_dir_name = f"{script_name}_{args.exp_name}_{current_time}"
    save_dir = os.path.join(args.save_root, save_dir_name)
    video_dir = os.path.join(save_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    map_files = collect_map_files(args.precomputed_map_dir)
    if len(map_files) == 0:
        raise RuntimeError(f"No map_*.pt found in {args.precomputed_map_dir}")

    use_attitude_v2 = args.attitude_model == "v2"
    base_state_dim = 7 if args.no_odom else 10
    state_dim = base_state_dim + (3 if use_attitude_v2 else 0)
    action_dim = 7 if use_attitude_v2 else 6
    yaw_rate_max = math.radians(float(args.yaw_rate_max_deg))

    env = Env(
        args.batch_size,
        64,
        48,
        args.grad_decay,
        device,
        fov_x_half_tan=args.fov_x_half_tan,
        single=args.single,
        gate=args.gate,
        ground_voxels=args.ground_voxels,
        scaffold=args.scaffold,
        speed_mtp=args.speed_mtp,
        random_rotation=args.random_rotation,
        cam_angle=args.cam_angle,
        obstacle_count_scale=args.obstacle_count_scale,
        include_u_local_optimum=args.include_u_local_optimum,
        compact_two_zone_map=args.compact_two_zone_map,
        wall_physical_feedback=args.wall_physical_feedback,
    )
    # 评测固定参数：无人机半径 13cm、margin 7cm。
    enforce_eval_drone_specs(env, drone_radius_m=0.13, margin_m=0.07)
    model = Model(state_dim, action_dim).to(device)
    ckpt_info = load_compatible_checkpoint(model, args.checkpoint, device)
    model.eval()

    print(f"Evaluation artifacts will be saved to: {save_dir}")
    print(f"Loaded checkpoint: {args.checkpoint}")
    if len(ckpt_info["resized_keys"]) > 0:
        print("resized_keys:", ckpt_info["resized_keys"])
    if len(ckpt_info["skipped_keys"]) > 0:
        print("skipped_keys:", ckpt_info["skipped_keys"])
    if len(ckpt_info["missing_keys"]) > 0:
        print("missing_keys:", ckpt_info["missing_keys"])
    if len(ckpt_info["unexpected_keys"]) > 0:
        print("unexpected_keys:", ckpt_info["unexpected_keys"])

    per_agent_records: List[Dict] = []
    pbar = tqdm(range(args.num_episodes), ncols=120, desc="Evaluating")

    for epi in pbar:
        map_idx = epi % len(map_files)
        map_path = map_files[map_idx]
        map_data = torch.load(map_path, map_location="cpu")
        _align_env_goal_planes_to_precomputed_map(map_data, env)
        env.reset_from_precomputed_map(map_data)
        enforce_eval_drone_specs(env, drone_radius_m=0.13, margin_m=0.07)

        out = run_single_episode(env, model, args, device, use_attitude_v2, yaw_rate_max)
        B = args.batch_size

        collision_free = out["collision_free"]
        reached_goal = out["reached_goal"]
        batch_no_collision_rate = float(collision_free.float().mean().item())
        batch_reach_goal_rate = float(reached_goal.float().mean().item())
        batch_success_rate_product = batch_no_collision_rate * batch_reach_goal_rate
        batch_success_and = float((collision_free & reached_goal).float().mean().item())

        step = epi + 1
        writer.add_scalar("Episode/No_Collision_Rate_Batch", batch_no_collision_rate, step)
        writer.add_scalar("Episode/Reach_Goal_Rate_Batch", batch_reach_goal_rate, step)
        writer.add_scalar("Episode/Success_Rate_Product_Batch", batch_success_rate_product, step)
        writer.add_scalar("Episode/Success_Rate_AND_Batch", batch_success_and, step)
        writer.add_scalar("Episode/Avg_Speed_Batch", float(out["avg_speed"].mean().item()), step)
        writer.add_scalar("Episode/Fastest_Speed_Batch", float(out["max_speed"].max().item()), step)
        writer.add_scalar("Episode/Final_Dist_To_Goal_Batch", float(out["final_dist_to_goal"].mean().item()), step)
        writer.add_scalar("Episode/Collision_Step_Ratio_Batch", float(out["collision_step_ratio"].mean().item()), step)
        writer.add_scalar("Episode/Min_Clearance_Batch", float(out["min_clearance"].mean().item()), step)
        writer.add_scalar("Episode/Path_Length_Batch", float(out["path_length"].mean().item()), step)
        writer.add_scalar("Episode/Progress_Ratio_Batch", float(out["progress_ratio"].mean().item()), step)
        writer.add_scalar("Episode/Yaw_Error_Abs_Deg_Batch", float(out["yaw_error_abs_deg"].mean().item()), step)
        writer.add_scalar("Episode/Map_Index", float(map_idx), step)

        html_interval = max(1, int(args.save_html_every))
        if (epi % html_interval) == 0:
            idx = int(max(0, min(args.html_sample_idx, B - 1)))
            p_cpu = out["p_history_full"][:, idx].detach().cpu()
            v_cpu = out["v_history_full"][:, idx].detach().cpu()
            R_cpu = out["R_history_full"][:, idx].detach().cpu()
            interactive_html = os.path.join(video_dir, f"trajectory3d_eval_{step:06d}.html")
            if save_interactive_3d_html(interactive_html, env, p_cpu, v_cpu, R_cpu=R_cpu, idx=idx):
                writer.add_text("Trajectory/Interactive3D_HTML", interactive_html, step)

        for b in range(B):
            per_agent_records.append({
                "episode_index": int(epi),
                "map_index": int(map_idx),
                "map_path": map_path,
                "batch_index": int(b),
                "collision_free": bool(collision_free[b].item()),
                "reached_goal": bool(reached_goal[b].item()),
                "success_and": bool((collision_free[b] & reached_goal[b]).item()),
                "avg_speed": float(out["avg_speed"][b].item()),
                "max_speed": float(out["max_speed"][b].item()),
                "final_dist_to_goal": float(out["final_dist_to_goal"][b].item()),
                "collision_step_ratio": float(out["collision_step_ratio"][b].item()),
                "collision_depth_max": float(out["collision_depth_max"][b].item()),
                "min_clearance": float(out["min_clearance"][b].item()),
                "path_length": float(out["path_length"][b].item()),
                "progress_ratio": float(out["progress_ratio"][b].item()),
                "time_to_goal_step": float(out["time_to_goal_step"][b].item()),
                "time_to_goal_sec": float(out["time_to_goal_sec"][b].item()),
                "yaw_error_abs_deg": float(out["yaw_error_abs_deg"][b].item()),
            })

        pbar.set_postfix({
            "no_col": f"{batch_no_collision_rate:.2f}",
            "reach": f"{batch_reach_goal_rate:.2f}",
            "succ_prod": f"{batch_success_rate_product:.2f}",
            "avg_v": f"{float(out['avg_speed'].mean().item()):.2f}",
        })

    collision_vals = np.asarray([float(r["collision_free"]) for r in per_agent_records], dtype=np.float64)
    reach_vals = np.asarray([float(r["reached_goal"]) for r in per_agent_records], dtype=np.float64)
    success_and_vals = np.asarray([float(r["success_and"]) for r in per_agent_records], dtype=np.float64)
    avg_speed_vals = np.asarray([r["avg_speed"] for r in per_agent_records], dtype=np.float64)
    max_speed_vals = np.asarray([r["max_speed"] for r in per_agent_records], dtype=np.float64)
    final_dist_vals = np.asarray([r["final_dist_to_goal"] for r in per_agent_records], dtype=np.float64)
    coll_step_vals = np.asarray([r["collision_step_ratio"] for r in per_agent_records], dtype=np.float64)
    coll_depth_vals = np.asarray([r["collision_depth_max"] for r in per_agent_records], dtype=np.float64)
    clearance_vals = np.asarray([r["min_clearance"] for r in per_agent_records], dtype=np.float64)
    path_len_vals = np.asarray([r["path_length"] for r in per_agent_records], dtype=np.float64)
    progress_vals = np.asarray([r["progress_ratio"] for r in per_agent_records], dtype=np.float64)
    tgoal_sec_vals = np.asarray([r["time_to_goal_sec"] for r in per_agent_records], dtype=np.float64)
    yaw_err_vals = np.asarray([r["yaw_error_abs_deg"] for r in per_agent_records], dtype=np.float64)

    no_collision_rate = float(collision_vals.mean()) if len(collision_vals) > 0 else 0.0
    reach_goal_rate = float(reach_vals.mean()) if len(reach_vals) > 0 else 0.0
    success_rate_product = float(no_collision_rate * reach_goal_rate)
    success_rate_and = float(success_and_vals.mean()) if len(success_and_vals) > 0 else 0.0

    summary_metrics = {
        "num_eval_episodes": int(args.num_episodes),
        "num_eval_agents": int(len(per_agent_records)),
        "no_collision_rate": no_collision_rate,
        "reach_goal_rate": reach_goal_rate,
        "success_rate_product": success_rate_product,
        "success_rate_and": success_rate_and,
        "avg_speed_mean": float(avg_speed_vals.mean()) if len(avg_speed_vals) > 0 else 0.0,
        "fastest_speed": float(max_speed_vals.max()) if len(max_speed_vals) > 0 else 0.0,
        "max_speed_mean": float(max_speed_vals.mean()) if len(max_speed_vals) > 0 else 0.0,
        "final_dist_to_goal_mean": float(final_dist_vals.mean()) if len(final_dist_vals) > 0 else 0.0,
        "final_dist_to_goal_median": float(np.median(final_dist_vals)) if len(final_dist_vals) > 0 else 0.0,
        "collision_step_ratio_mean": float(coll_step_vals.mean()) if len(coll_step_vals) > 0 else 0.0,
        "collision_depth_max_mean": float(coll_depth_vals.mean()) if len(coll_depth_vals) > 0 else 0.0,
        "min_clearance_mean": float(clearance_vals.mean()) if len(clearance_vals) > 0 else 0.0,
        "path_length_mean": float(path_len_vals.mean()) if len(path_len_vals) > 0 else 0.0,
        "progress_ratio_mean": float(progress_vals.mean()) if len(progress_vals) > 0 else 0.0,
        "time_to_goal_sec_mean": float(tgoal_sec_vals.mean()) if len(tgoal_sec_vals) > 0 else 0.0,
        "yaw_error_abs_deg_mean": float(yaw_err_vals.mean()) if len(yaw_err_vals) > 0 else 0.0,
    }

    final_step = int(args.num_episodes)
    writer.add_scalar("Metrics/No_Collision_Rate", summary_metrics["no_collision_rate"], final_step)
    writer.add_scalar("Metrics/Reach_Goal_Rate", summary_metrics["reach_goal_rate"], final_step)
    writer.add_scalar("Metrics/Success_Rate_Product", summary_metrics["success_rate_product"], final_step)
    writer.add_scalar("Metrics/Success_Rate_AND", summary_metrics["success_rate_and"], final_step)
    writer.add_scalar("Metrics/Avg_Speed_Mean", summary_metrics["avg_speed_mean"], final_step)
    writer.add_scalar("Metrics/Fastest_Speed", summary_metrics["fastest_speed"], final_step)
    writer.add_scalar("Metrics/Max_Speed_Mean", summary_metrics["max_speed_mean"], final_step)
    writer.add_scalar("Metrics/Final_Dist_To_Goal_Mean", summary_metrics["final_dist_to_goal_mean"], final_step)
    writer.add_scalar("Metrics/Final_Dist_To_Goal_Median", summary_metrics["final_dist_to_goal_median"], final_step)
    writer.add_scalar("Metrics/Collision_Step_Ratio_Mean", summary_metrics["collision_step_ratio_mean"], final_step)
    writer.add_scalar("Metrics/Collision_Depth_Max_Mean", summary_metrics["collision_depth_max_mean"], final_step)
    writer.add_scalar("Metrics/Min_Clearance_Mean", summary_metrics["min_clearance_mean"], final_step)
    writer.add_scalar("Metrics/Path_Length_Mean", summary_metrics["path_length_mean"], final_step)
    writer.add_scalar("Metrics/Progress_Ratio_Mean", summary_metrics["progress_ratio_mean"], final_step)
    writer.add_scalar("Metrics/Time_To_Goal_Sec_Mean", summary_metrics["time_to_goal_sec_mean"], final_step)
    writer.add_scalar("Metrics/Yaw_Error_Abs_Deg_Mean", summary_metrics["yaw_error_abs_deg_mean"], final_step)

    summary = {
        "checkpoint": args.checkpoint,
        "precomputed_map_dir": args.precomputed_map_dir,
        "save_dir": save_dir,
        "video_dir": video_dir,
        "metrics": summary_metrics,
        "checkpoint_load": {
            "missing_keys": ckpt_info["missing_keys"],
            "unexpected_keys": ckpt_info["unexpected_keys"],
            "resized_keys": ckpt_info["resized_keys"],
            "skipped_keys": ckpt_info["skipped_keys"],
        },
        "per_agent_records": per_agent_records,
    }
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    writer.close()

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary_metrics, indent=2))
    print(f"Summary saved to: {summary_path}")
    print(f"TensorBoard logs: {os.path.join(save_dir, 'logs')}")
    print(f"Trajectory HTML dir: {video_dir}")


if __name__ == "__main__":
    main()
