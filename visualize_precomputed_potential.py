import argparse
import os

import numpy as np
import torch

try:
    import plotly.graph_objects as go
except Exception:
    go = None


def _collect_map_files(map_dir):
    files = []
    if not os.path.isdir(map_dir):
        return files
    type_order = {
        "hard": 0,
        "easy": 1,
        "u_min": 2,
        "hairpin": 3,
    }

    def _sort_key(name):
        if not name.endswith(".pt"):
            return (9, name)
        stem = name[:-3]
        if "_" in stem:
            prefix, idx_str = stem.rsplit("_", 1)
            if prefix in type_order and idx_str.isdigit():
                return (0, type_order[prefix], int(idx_str), name)
        if stem.startswith("map_"):
            idx_str = stem[4:]
            if idx_str.isdigit():
                return (1, int(idx_str), name)
            return (1, 10**9, name)
        return (9, name)

    for name in sorted(os.listdir(map_dir), key=_sort_key):
        if _sort_key(name)[0] < 9:
            files.append(os.path.join(map_dir, name))
    return files


def _pick_slice_z(z_world, origin_z, resolution, nz):
    if z_world is None:
        return max(0, min(nz - 1, nz // 2))
    zi = int(round((float(z_world) - float(origin_z)) / float(resolution)))
    return max(0, min(nz - 1, zi))


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_on_mask(values, valid_mask):
    if not np.any(valid_mask):
        return np.zeros_like(values, dtype=np.float32)
    v = values[valid_mask]
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    return ((values - vmin) / max(1e-6, vmax - vmin)).astype(np.float32)


def _fallback_dir_from_potential(potential, free_mask):
    # Fallback: use negative potential gradient as descending direction.
    if not np.any(free_mask):
        return np.zeros((*potential.shape, 3), dtype=np.float32)

    valid_vals = potential[free_mask]
    fill_value = float(np.max(valid_vals)) + 1.0
    pot_filled = np.where(np.isfinite(potential), potential, fill_value).astype(np.float32)
    gx, gy, gz = np.gradient(pot_filled)
    d = np.stack([-gx, -gy, -gz], axis=-1).astype(np.float32)
    d[~free_mask] = 0.0
    return d


def _sample_indices_from_mask(mask, step):
    idx = np.argwhere(mask)
    if idx.shape[0] == 0:
        return idx
    s = max(1, int(step))
    return idx[::s]


def _add_cuboid(fig, cx, cy, cz, hx, hy, hz, color="rgba(90,90,90,0.55)"):
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    verts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    tri = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
        color=color, opacity=0.55, flatshading=True,
        name="Obstacles (Solid)", legendgroup="obstacles", showlegend=False,
        hoverinfo="skip",
    ))


def _add_sphere(fig, cx, cy, cz, r, color="rgba(60,120,180,0.55)", res=10):
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    uu, vv = np.meshgrid(u, v)
    x = cx + r * np.cos(uu) * np.sin(vv)
    y = cy + r * np.sin(uu) * np.sin(vv)
    z = cz + r * np.cos(vv)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False, opacity=0.55,
        name="Obstacles (Solid)", legendgroup="obstacles", showlegend=False,
        hoverinfo="skip",
    ))


def _add_cylinder_z(fig, cx, cy, r, z0, z1, color="rgba(0,120,120,0.55)", res=20):
    t = np.linspace(0.0, 2.0 * np.pi, res)
    z = np.array([z0, z1], dtype=np.float32)
    tt, zz = np.meshgrid(t, z)
    x = cx + r * np.cos(tt)
    y = cy + r * np.sin(tt)
    fig.add_trace(go.Surface(
        x=x, y=y, z=zz,
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False, opacity=0.55,
        name="Obstacles (Solid)", legendgroup="obstacles", showlegend=False,
        hoverinfo="skip",
    ))


def _add_cylinder_y(fig, cx, cz, r, y0, y1, color="rgba(180,100,20,0.55)", res=20):
    t = np.linspace(0.0, 2.0 * np.pi, res)
    y = np.array([y0, y1], dtype=np.float32)
    tt, yy = np.meshgrid(t, y)
    x = cx + r * np.cos(tt)
    z = cz + r * np.sin(tt)
    fig.add_trace(go.Surface(
        x=x, y=yy, z=z,
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False, opacity=0.55,
        name="Obstacles (Solid)", legendgroup="obstacles", showlegend=False,
        hoverinfo="skip",
    ))


def _build_3d_figure(
    map_data,
    z_world=None,
    z_band_layers=2,
    stride=4,
    arrow_stride=8,
    max_arrows=1200,
    show_occupancy_points=True,
    show_potential_points=True,
    title_prefix="",
):
    potential = _to_numpy(map_data["potential"]).astype(np.float32)
    occupancy = _to_numpy(map_data["occupancy"]) > 0
    guide_dir = _to_numpy(map_data["guide_dir"]).astype(np.float32)
    origin = _to_numpy(map_data["grid_origin"]).astype(np.float32)
    resolution = float(map_data["resolution"])

    nx, ny, nz = potential.shape
    xs = origin[0] + (np.arange(nx, dtype=np.float32) + 0.5) * resolution
    ys = origin[1] + (np.arange(ny, dtype=np.float32) + 0.5) * resolution
    zs = origin[2] + (np.arange(nz, dtype=np.float32) + 0.5) * resolution

    I, J, K = np.meshgrid(
        np.arange(nx, dtype=np.int32),
        np.arange(ny, dtype=np.int32),
        np.arange(nz, dtype=np.int32),
        indexing="ij",
    )
    X = xs[I]
    Y = ys[J]
    Z = zs[K]

    zi = _pick_slice_z(z_world, origin[2], resolution, nz)
    z_world_used = float(origin[2]) + (zi + 0.5) * resolution

    finite = np.isfinite(potential)
    # reachable_free: finite potential region (reachable from goal)
    reachable_free = (~occupancy) & finite
    # traversable_free: all non-occupied voxels (including inf-potential unreachable region)
    traversable_free = (~occupancy)
    field_mask = traversable_free.copy()
    display_mask = traversable_free.copy()
    sliced_by_z = False
    if z_world is not None:
        band = max(0, int(z_band_layers))
        if band <= 0:
            display_mask = display_mask & (K == zi)
        else:
            display_mask = display_mask & (np.abs(K - zi) <= band)
        sliced_by_z = True

    pot_norm = _normalize_on_mask(potential, reachable_free)
    step = max(1, int(stride))
    arrow_step = max(1, int(arrow_stride))

    occ_mask = occupancy & ((I % step) == 0) & ((J % step) == 0) & ((K % step) == 0)
    # Potential cloud still visualizes only finite values.
    pot_mask = (reachable_free & display_mask) & ((I % step) == 0) & ((J % step) == 0) & ((K % step) == 0)
    # Arrows always cover the whole traversable space (not restricted by z slice).
    arrow_idx = _sample_indices_from_mask(traversable_free, arrow_step)
    fallback_all_z = False
    if arrow_idx.shape[0] == 0 and sliced_by_z:
        # If requested z slice has no drawable arrows, fallback to all-z so the scene is never empty.
        field_mask = traversable_free
        pot_mask = reachable_free & ((I % step) == 0) & ((J % step) == 0) & ((K % step) == 0)
        arrow_idx = _sample_indices_from_mask(traversable_free, arrow_step)
        fallback_all_z = True

    fig = go.Figure()

    # Solid obstacle rendering from precomputed geometry.
    if "voxels" in map_data:
        vox = _to_numpy(map_data["voxels"]).astype(np.float32).reshape(-1, 6)
        for box in vox[:220]:
            _add_cuboid(fig, float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]), float(box[5]))
    if "balls" in map_data:
        balls = _to_numpy(map_data["balls"]).astype(np.float32).reshape(-1, 4)
        for b in balls[:120]:
            if b[3] > 1e-6:
                _add_sphere(fig, float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    z0 = float(map_data.get("z_min", float(zs.min())))
    z1 = float(map_data.get("z_max", float(zs.max())))
    if "cyl" in map_data:
        cyl = _to_numpy(map_data["cyl"]).astype(np.float32).reshape(-1, 3)
        for c in cyl[:140]:
            if c[2] > 1e-6:
                _add_cylinder_z(fig, float(c[0]), float(c[1]), float(c[2]), z0, z1)
    y0 = float(ys.min())
    y1 = float(ys.max())
    if "cyl_h" in map_data:
        cyl_h = _to_numpy(map_data["cyl_h"]).astype(np.float32).reshape(-1, 3)
        for c in cyl_h[:140]:
            if c[2] > 1e-6:
                _add_cylinder_y(fig, float(c[0]), float(c[1]), float(c[2]), y0, y1)

    if show_occupancy_points and np.any(occ_mask):
        fig.add_trace(go.Scatter3d(
            x=X[occ_mask],
            y=Y[occ_mask],
            z=Z[occ_mask],
            mode="markers",
            marker=dict(size=2, color="black", opacity=0.5),
            name="Occupied Voxels",
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        ))

    if show_potential_points and np.any(pot_mask):
        fig.add_trace(go.Scatter3d(
            x=X[pot_mask],
            y=Y[pot_mask],
            z=Z[pot_mask],
            mode="markers",
            marker=dict(
                size=2,
                color=pot_norm[pot_mask],
                colorscale="Viridis",
                cmin=0.0,
                cmax=1.0,
                opacity=0.42,
                colorbar=dict(title="Potential (norm)"),
            ),
            name="Potential Field",
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        ))

    arrow_count_initial = int(arrow_idx.shape[0])
    arrow_count_after_dir = 0
    arrow_count_final = 0
    fallback_goal_used = False
    if arrow_idx.shape[0] > 0:
        if arrow_idx.shape[0] > max_arrows:
            # Uniformly subsample for browser stability and readability.
            pick = np.linspace(0, arrow_idx.shape[0] - 1, num=max_arrows, dtype=np.int64)
            arrow_idx = arrow_idx[pick]
        arrow_count_final = int(arrow_idx.shape[0])

        if arrow_idx.shape[0] > 0:
            ai = arrow_idx[:, 0]
            aj = arrow_idx[:, 1]
            ak = arrow_idx[:, 2]

            ax = X[ai, aj, ak]
            ay = Y[ai, aj, ak]
            az = Z[ai, aj, ak]
            starts = np.stack([ax, ay, az], axis=-1).astype(np.float32)

            # 1) primary direction from cached guide_dir
            dirs = guide_dir[ai, aj, ak].astype(np.float32)

            # 2) per-point fallback to negative potential gradient where guide_dir is too small
            grad_dir_all = _fallback_dir_from_potential(potential, reachable_free)
            norms = np.linalg.norm(dirs, axis=-1)
            weak = norms <= 1e-8
            if np.any(weak):
                dirs[weak] = grad_dir_all[ai[weak], aj[weak], ak[weak]]

            # 3) final per-point fallback to goal direction if still weak
            norms = np.linalg.norm(dirs, axis=-1)
            weak = norms <= 1e-8
            if np.any(weak) and ("goal_world" in map_data):
                goal = _to_numpy(map_data["goal_world"]).astype(np.float32).reshape(-1)
                if goal.size >= 3:
                    goal_xyz = np.array([float(goal[0]), float(goal[1]), float(goal[2])], dtype=np.float32)
                    dirs[weak] = goal_xyz[None, :] - starts[weak]
                    fallback_goal_used = True

            norms = np.linalg.norm(dirs, axis=-1)
            valid = norms > 1e-8
            arrow_count_after_dir = int(np.count_nonzero(valid))
            if not np.any(valid):
                arrow_count_final = 0
            else:
                ai = ai[valid]
                aj = aj[valid]
                ak = ak[valid]
                ax = ax[valid]
                ay = ay[valid]
                az = az[valid]
                starts = starts[valid]
                dirs = dirs[valid]
                dnorm = np.linalg.norm(dirs, axis=-1, keepdims=True)
                dirs = dirs / np.maximum(dnorm, 1e-6)
                arrow_count_final = int(dirs.shape[0])

                shaft_len = max(0.12, 2.0 * resolution)
                head_len = max(0.07, 1.2 * resolution)
                ex = ax + dirs[:, 0] * shaft_len
                ey = ay + dirs[:, 1] * shaft_len
                ez = az + dirs[:, 2] * shaft_len

                x_lines = np.column_stack((ax, ex, np.full_like(ax, np.nan))).reshape(-1)
                y_lines = np.column_stack((ay, ey, np.full_like(ay, np.nan))).reshape(-1)
                z_lines = np.column_stack((az, ez, np.full_like(az, np.nan))).reshape(-1)
                fig.add_trace(go.Scatter3d(
                    x=x_lines,
                    y=y_lines,
                    z=z_lines,
                    mode="lines",
                    line=dict(color="#ff5a00", width=2),
                    opacity=0.95,
                    name="Descent Direction (Shaft)",
                    hoverinfo="skip",
                ))

                fig.add_trace(go.Cone(
                    x=ex,
                    y=ey,
                    z=ez,
                    u=dirs[:, 0] * head_len,
                    v=dirs[:, 1] * head_len,
                    w=dirs[:, 2] * head_len,
                    anchor="tail",
                    sizemode="absolute",
                    sizeref=head_len * 0.55,
                    colorscale=[[0.0, "#ff5a00"], [1.0, "#ff5a00"]],
                    cmin=0.0,
                    cmax=1.0,
                    showscale=False,
                    opacity=0.98,
                    name="Descent Direction (Head)" if not fallback_goal_used else "Goal Direction (Head)",
                    hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
                ))

    print(
        f"[viz] arrows initial={arrow_count_initial}, after_dir={arrow_count_after_dir}, "
        f"final={arrow_count_final}, goal_fallback={int(fallback_goal_used)}, "
        f"z_fallback={int(fallback_all_z)}"
    )

    if "goal_world" in map_data:
        goal = _to_numpy(map_data["goal_world"]).astype(np.float32).reshape(-1)
        if goal.size >= 3:
            fig.add_trace(go.Scatter3d(
                x=[float(goal[0])],
                y=[float(goal[1])],
                z=[float(goal[2])],
                mode="markers",
                marker=dict(size=8, color="red", symbol="diamond"),
                name="Goal",
            ))

    z_desc = "all z" if z_world is None else f"z_idx={zi}, z={z_world_used:.2f}m (arrows: full space)"
    if fallback_all_z:
        z_desc = f"{z_desc} -> fallback all z"
    fig.update_layout(
        title=f"{title_prefix} 3D Potential / Occupancy / Descent Field ({z_desc})",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D potential field from precomputed map files (.pt)"
    )
    parser.add_argument(
        "--map_dir",
        type=str,
        default="/home/robot/transformer/precomputed_maps",
        help="Directory containing precomputed .pt map files",
    )
    parser.add_argument(
        "--map_index",
        type=int,
        default=0,
        help="Index in sorted map file list",
    )
    parser.add_argument(
        "--z_world",
        type=float,
        default=None,
        help="World z (meters) slice center for 3D view; default shows all z",
    )
    parser.add_argument(
        "--z_band_layers",
        type=int,
        default=2,
        help="When --z_world is set, include +/- this many z grid layers (0 means single layer)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Sampling stride for potential points / occupancy points",
    )
    parser.add_argument(
        "--arrow_stride",
        type=int,
        default=8,
        help="Sampling stride for direction arrows (larger means sparser)",
    )
    parser.add_argument(
        "--max_arrows",
        type=int,
        default=1200,
        help="Upper bound of arrows in HTML to keep rendering clear",
    )
    parser.add_argument(
        "--show_potential_points",
        dest="show_potential_points",
        action="store_true",
        help="Show potential point cloud",
    )
    parser.add_argument(
        "--no_show_potential_points",
        dest="show_potential_points",
        action="store_false",
        help="Hide potential point cloud",
    )
    parser.add_argument(
        "--show_occupancy_points",
        dest="show_occupancy_points",
        action="store_true",
        help="Show occupied voxel points",
    )
    parser.add_argument(
        "--no_show_occupancy_points",
        dest="show_occupancy_points",
        action="store_false",
        help="Hide occupied voxel points",
    )
    parser.set_defaults(show_potential_points=True, show_occupancy_points=True)
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional output html path (.html). If empty, open interactive browser window.",
    )

    args = parser.parse_args()

    files = _collect_map_files(args.map_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No .pt map files found in: {args.map_dir}")

    idx = int(args.map_index) % len(files)
    path = files[idx]
    map_data = torch.load(path, map_location="cpu")

    if go is None:
        raise ImportError("plotly is required for 3D HTML visualization. Please install plotly first.")

    fig = _build_3d_figure(
        map_data,
        z_world=args.z_world,
        z_band_layers=args.z_band_layers,
        stride=args.stride,
        arrow_stride=args.arrow_stride,
        max_arrows=max(1, int(args.max_arrows)),
        show_occupancy_points=bool(args.show_occupancy_points),
        show_potential_points=bool(args.show_potential_points),
        title_prefix=f"[{os.path.basename(path)}]",
    )

    if args.save:
        if not args.save.lower().endswith(".html"):
            raise ValueError("--save must end with .html for 3D interactive visualization")
        out_dir = os.path.dirname(args.save)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.write_html(args.save, include_plotlyjs="cdn")
        print(f"Saved: {args.save}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
