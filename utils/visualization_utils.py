"""Trajectory and map visualization helpers."""

from collections import defaultdict
import os

import numpy as np
import torch
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

from .logging_utils import is_debug_tb_step
from .tensor_utils import merge_intervals


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


def _is_ceiling_voxel(box_xyz_half, env):
    """Return True if a voxel is the top ceiling slab."""
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
    ceiling = (
        abs(cz - map_z_max) <= tol
        and abs(hz - boundary_half) <= tol
        and hx >= 0.45 * map_x_max
        and hy >= 0.45 * map_y_span
    )
    return bool(ceiling)


def _is_top_ceiling_voxel_relaxed(box_xyz_half, env):
    """Relaxed ceiling check: identify any top boundary slab near map_z_max."""
    cx, cy, cz, hx, hy, hz = [float(v) for v in box_xyz_half]
    if not all(hasattr(env, key) for key in ('map_z_max', 'boundary_half')):
        return False
    map_z_max = float(env.map_z_max)
    boundary_half = float(env.boundary_half)
    tol = max(0.08, boundary_half * 1.8)
    return bool(abs(cz - map_z_max) <= tol and abs(hz - boundary_half) <= tol)


def _is_boundary_wall_voxel(box_xyz_half, env):
    """Heuristic check for outer enclosure side walls (x/y boundaries)."""
    cx, cy, cz, hx, hy, hz = [float(v) for v in box_xyz_half]
    if not all(hasattr(env, key) for key in ('map_x_max', 'map_y_min', 'map_y_max', 'boundary_half')):
        return False

    map_x_max = float(env.map_x_max)
    map_y_min = float(env.map_y_min)
    map_y_max = float(env.map_y_max)
    boundary_half = float(env.boundary_half)

    tol = max(0.08, boundary_half * 1.8)
    side_x = (abs(cx - 0.0) <= tol or abs(cx - map_x_max) <= tol) and abs(hx - boundary_half) <= tol
    side_y = (abs(cy - map_y_min) <= tol or abs(cy - map_y_max) <= tol) and abs(hy - boundary_half) <= tol
    return bool(side_x or side_y)


def save_interactive_3d_html(html_path, env, p_cpu, v_cpu, R_cpu=None, idx=0, axis_len=0.3, axis_step=5,
                             astar_path=None, astar_paths_sampled=None,
                             potential_map_data=None, show_potential_overlay=False,
                             map_type=None):
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

    map_type_norm = str(map_type).strip().lower().replace("-", "_") if map_type is not None else ""
    hide_relaxed_ceiling = map_type_norm in ("u_min", "u_minimal")

    traj_xyz = p_cpu.numpy()
    speed_cpu = v_cpu.norm(dim=-1).numpy()
    traj_labels = [f"t={t}<br>speed={speed_cpu[t]:.2f} m/s" for t in range(len(traj_xyz))]
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
        hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>%{text}<extra></extra>',
        text=traj_labels,
        name='Trajectory'
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
            # X轴 (红色) - 机头方向（真实姿态）
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
            # Y轴 (绿色) - 左侧方向（真实姿态）
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
            # Z轴 (蓝色) - 上方向（真实姿态）
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
            if hide_relaxed_ceiling and _is_top_ceiling_voxel_relaxed(box, env):
                continue
            if _is_ceiling_voxel(box, env) or _is_boundary_wall_voxel(box, env):
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


class _VizEnvSnapshot:
    pass


@torch.no_grad()
def snapshot_env_for_viz(env_obj, idx=0):
    """Capture just the map/target tensors needed by the visualization helpers."""
    snap = _VizEnvSnapshot()
    for name in ('voxels', 'balls', 'cyl', 'cyl_h'):
        if hasattr(env_obj, name):
            value = getattr(env_obj, name)
            if isinstance(value, torch.Tensor):
                setattr(snap, name, value[idx:idx + 1].detach().cpu().clone())
    if hasattr(env_obj, 'p_target') and isinstance(env_obj.p_target, torch.Tensor):
        snap.p_target = env_obj.p_target[idx:idx + 1].detach().cpu().clone()
    for name in ('map_x_max', 'map_y_min', 'map_y_max', 'map_z_max', 'boundary_half'):
        if hasattr(env_obj, name):
            setattr(snap, name, float(getattr(env_obj, name)))
    return snap


def _depth_frames_to_video(
    depth_stack,
    mp4_path,
    gif_path,
    step,
    tb_tag_prefix,
    writer,
):
    if depth_stack is None or depth_stack.numel() == 0:
        return
    depth_stack = depth_stack.float()
    inv_depth = 3.0 / depth_stack.clamp(0.3, 24.0) - 0.6
    p2 = torch.quantile(inv_depth, 0.02)
    p98 = torch.quantile(inv_depth, 0.98)
    inv_norm = ((inv_depth - p2) / (p98 - p2 + 1e-6)).clamp(0.0, 1.0)

    cmap_np = plt.get_cmap('magma')
    inv_np = inv_norm.cpu().numpy()
    frames = []
    for frame_idx in range(inv_np.shape[0]):
        rgb = (cmap_np(inv_np[frame_idx])[..., :3] * 255.0).astype('uint8')
        frames.append(rgb)

    if imageio is not None:
        try:
            imageio.mimsave(mp4_path, frames, fps=15, macro_block_size=None)
            writer.add_text(f'{tb_tag_prefix}/Depth_Video', mp4_path, step)
            return
        except Exception:
            imageio.mimsave(gif_path, frames, format='GIF', fps=15)
            writer.add_text(f'{tb_tag_prefix}/Depth_Video', gif_path, step)
            return

    depth_uint8 = (inv_norm * 255).to(torch.uint8)
    for frame_idx, frame in enumerate(depth_uint8):
        writer.add_image(f'{tb_tag_prefix}/Depth_Frame/{frame_idx:03d}', frame.unsqueeze(0), step)


def save_cached_viz_record(
    record,
    save_step,
    *,
    args,
    potential_map_cache,
    video_dir,
    writer,
    artifact_label=None,
):
    map_type = str(record['map_type'])
    source_iter = int(record['iter'])
    idx = 0
    p_cpu = record['p_cpu']
    v_cpu = record['v_cpu']
    rpy_cpu = record['rpy_cpu']
    R_cpu = record['R_cpu']
    act_cpu = record['act_cpu']
    w_cpu = record['weights_cpu']
    env_snapshot = record['env_snapshot']
    astar_paths_sampled = record.get('astar_paths_sampled', [])

    is_best = artifact_label is not None
    tag_prefix = f'Best/Trajectory/{map_type}' if is_best else f'Trajectory/{map_type}'
    debug_prefix = f'Debug/{map_type}'
    if is_best:
        file_prefix = f'{artifact_label}_{map_type}'
    else:
        # Put save_step first so plain filename sorting follows training chronology.
        file_prefix = f'save_{save_step:06d}_src_{source_iter:06d}_{map_type}'

    fig_p, ax = plt.subplots()
    ax.plot(p_cpu[:, 0], label='x')
    ax.plot(p_cpu[:, 1], label='y')
    ax.plot(p_cpu[:, 2], label='z')
    ax.legend()
    ax.set_title(f"{map_type} source iter {source_iter} Pos")
    writer.add_figure(f'{tag_prefix}/Position_Series', fig_p, save_step)
    plt.close(fig_p)

    potential_map_data = None
    map_idx = int(record.get('map_idx', -1))
    if args.use_precomputed_potential_maps and potential_map_cache is not None and map_idx >= 0:
        potential_map_data = potential_map_cache.get_map(map_idx)

    interactive_html = os.path.join(video_dir, f'trajectory3d_{file_prefix}.html')
    if save_interactive_3d_html(
        interactive_html, env_snapshot, p_cpu, v_cpu, R_cpu=R_cpu, idx=idx,
        astar_path=None, astar_paths_sampled=astar_paths_sampled,
        potential_map_data=potential_map_data,
        show_potential_overlay=potential_map_data is not None,
        map_type=map_type,
    ):
        writer.add_text(f'{tag_prefix}/Interactive3D_HTML', interactive_html, save_step)

    # TensorBoard potential-field figure logging is intentionally disabled.

    fig_v, ax = plt.subplots()
    ax.plot(v_cpu[:, 0], label='vx')
    ax.plot(v_cpu[:, 1], label='vy')
    ax.plot(v_cpu[:, 2], label='vz')
    ax.plot(v_cpu.norm(dim=-1), label='speed', linestyle='--')
    ax.legend()
    ax.set_title(f"{map_type} source iter {source_iter} Velocity")
    writer.add_figure(f'{tag_prefix}/Velocity_Series', fig_v, save_step)
    plt.close(fig_v)

    fig_rpy, ax = plt.subplots()
    ax.plot(rpy_cpu[:, 0], label='roll(deg)')
    ax.plot(rpy_cpu[:, 1], label='pitch(deg)')
    ax.plot(rpy_cpu[:, 2], label='yaw(deg)')
    ax.legend()
    ax.set_title(f"{map_type} source iter {source_iter} Attitude RPY")
    writer.add_figure(f'{tag_prefix}/Attitude_RPY_Series', fig_rpy, save_step)
    plt.close(fig_rpy)

    fig_act, ax = plt.subplots()
    ax.plot(act_cpu[:, 0], label='ax_cmd')
    ax.plot(act_cpu[:, 1], label='ay_cmd')
    ax.plot(act_cpu[:, 2], label='az_cmd')
    ax.plot(act_cpu.norm(dim=-1), label='|a_cmd|', linestyle='--')
    ax.legend()
    ax.set_title(f"{map_type} source iter {source_iter} Control Accel Cmd")
    writer.add_figure(f'{tag_prefix}/Control_Accel_Cmd_Series', fig_act, save_step)
    plt.close(fig_act)

    if not is_best and is_debug_tb_step(save_step, args):
        labels = ['SpeedPref_Signed', 'SpeedPref_Strength', 'Direction', 'Avoidance', 'Exploration', 'Turn', 'VRef']
        tag_suffix = ['0_SpeedPref_Signed', '0_1_SpeedPref_Strength', '1_Direction', '2_Avoidance', '3_Exploration', '4_Turn', '5_VRef']
        for wi in range(min(len(labels), w_cpu.shape[-1])):
            fig_wi, ax = plt.subplots()
            ax.plot(w_cpu[:, wi], label=labels[wi])
            ax.legend()
            ax.set_title(f"{map_type} source iter {source_iter} Weight - {labels[wi]}")
            writer.add_figure(f'{debug_prefix}/Weights_StepWise_{tag_suffix[wi]}', fig_wi, save_step)
            plt.close(fig_wi)

    depth_stack = record.get('depth_stack', None)
    if depth_stack is not None:
        mp4_path = os.path.join(video_dir, f'depth_{file_prefix}.mp4')
        gif_path = os.path.join(video_dir, f'depth_{file_prefix}.gif')
        _depth_frames_to_video(
            depth_stack,
            mp4_path,
            gif_path,
            save_step,
            f'Best/Video/{map_type}' if is_best else f'Video/{map_type}',
            writer,
        )
