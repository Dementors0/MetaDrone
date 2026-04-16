#!/usr/bin/env python3
import argparse
import random
import sys
import webbrowser
from pathlib import Path

import numpy as np
import torch

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from env_multi import Env, DEFAULT_EASY_DENSITY_SCALE, DEFAULT_HARD_DENSITY_SCALE


REGION_COLORS = {
    "easy": "#6BAF5E",
    "hard": "#D97A3A",
    "u-minimal": "#4E79C7",
    "boundary": "#B8BEC7",
}

REGION_FILL_COLORS = {
    "easy": "#CFE8C8",
    "hard": "#F2D3BF",
    "u-minimal": "#D7E3F8",
}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classify_region(y_value: float, region_zones):
    for zone in region_zones:
        if zone["y0"] - 1e-6 <= y_value <= zone["y1"] + 1e-6:
            return zone["type"]
    return "boundary"


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device_arg


def resolve_precomputed_map(args):
    if args.precomputed_map:
        return Path(args.precomputed_map).expanduser().resolve(), "file"

    if not args.precomputed_map_dir:
        return None, None

    map_dir = Path(args.precomputed_map_dir).expanduser().resolve()
    if not map_dir.is_dir():
        raise FileNotFoundError(f"precomputed map dir not found: {map_dir}")

    type_order = {
        "hard": 0,
        "easy": 1,
        "u_min": 2,
        "hairpin": 3,
    }

    def _sort_key(path: Path):
        name = path.name
        stem = path.stem
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

    files = sorted((p for p in map_dir.iterdir() if p.suffix == ".pt" and _sort_key(p)[0] < 9), key=_sort_key)
    if not files:
        raise FileNotFoundError(f"no .pt map files found in: {map_dir}")

    idx = int(args.map_index) % len(files)
    return files[idx], f"dir[{idx}]"


def build_env(args):
    device = choose_device(args.device)
    batch_size = max(1, int(args.batch_index) + 1)
    include_u_local_optimum = not bool(args.disable_u_local_optimum)
    forced_map_type = "" if args.map_type == "cycle" else args.map_type
    env = Env(
        batch_size=batch_size,
        width=320,
        height=240,
        grad_decay=1.0,
        device=device,
        single=True,
        random_rotation=False,
        obstacle_count_scale=float(args.obstacle_count_scale),
        easy_density_scale=float(args.easy_density_scale),
        hard_density_scale=float(args.hard_density_scale),
        include_u_local_optimum=include_u_local_optimum,
        compact_two_zone_map=bool(args.compact_two_zone_map),
        unified_four_maps=bool(args.unified_four_maps),
        forced_map_type=forced_map_type,
    )
    return env, device


def extract_scene_geometry(env, batch_index: int = 0, source_label: str = "generated", seed: int | None = None):
    batch_index = int(batch_index) % max(1, int(env.batch_size))
    order = list(env.region_order[batch_index])
    region_count = max(1, len(order))
    region_span = (float(env.map_y_max) - float(env.map_y_min)) / float(region_count)

    region_zones = []
    for idx, region_type in enumerate(order):
        y0 = float(env.map_y_min + idx * region_span)
        y1 = float(y0 + region_span)
        region_zones.append({
            "type": region_type,
            "y0": y0,
            "y1": y1,
            "centerY": 0.5 * (y0 + y1),
        })

    boundary_count = len(env._build_boundary_voxels())

    balls = []
    for x, y, z, r in env.balls[batch_index].detach().cpu().tolist():
        balls.append({
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "r": float(r),
            "region": classify_region(float(y), region_zones),
        })

    cylinders = []
    for x, y, r in env.cyl[batch_index].detach().cpu().tolist():
        cylinders.append({
            "x": float(x),
            "y": float(y),
            "r": float(r),
            "region": classify_region(float(y), region_zones),
        })

    voxels = []
    for idx, (x, y, z, hx, hy, hz) in enumerate(env.voxels[batch_index].detach().cpu().tolist()):
        region = "boundary" if idx < boundary_count else classify_region(float(y), region_zones)
        voxels.append({
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "hx": float(hx),
            "hy": float(hy),
            "hz": float(hz),
            "region": region,
        })

    start_xyz = env.p[batch_index].detach().cpu().tolist()
    goal_xyz = env.p_target[batch_index].detach().cpu().tolist()
    start = {"x": float(start_xyz[0]), "y": float(start_xyz[1]), "z": float(start_xyz[2])}
    goal = {"x": float(goal_xyz[0]), "y": float(goal_xyz[1]), "z": float(goal_xyz[2])}

    return {
        "seed": None if seed is None else int(seed),
        "source": source_label,
        "order": order,
        "regionZones": region_zones,
        "balls": balls,
        "cylinders": cylinders,
        "voxels": voxels,
        "start": start,
        "goal": goal,
        "config": {
            "mapXMax": float(env.map_x_max),
            "mapYHalf": float(env.map_y_half),
            "mapYMin": float(env.map_y_min),
            "mapYMax": float(env.map_y_max),
            "mapZMax": float(env.map_z_max),
            "regionLength": float(env.region_length),
            "blankLength": float(env.blank_length),
            "spawnXCenter": float(env.spawn_x_center),
            "spawnZCenter": float(env.spawn_z_center),
            "boundaryThickness": float(env.boundary_thickness),
            "boundaryHalf": float(env.boundary_half),
        },
    }


def _add_cuboid(fig, cx, cy, cz, hx, hy, hz, color, opacity, name=None, showlegend=False):
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            flatshading=True,
            hoverinfo="skip",
            showscale=False,
            name=name,
            showlegend=showlegend,
        )
    )


def _add_sphere(fig, cx, cy, cz, r, color, opacity, res=12):
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    c = np.zeros_like(x)
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=c,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=opacity,
            hoverinfo="skip",
        )
    )


def _add_cylinder_z(fig, cx, cy, r, z0, z1, color, opacity, res_theta=16):
    th = np.linspace(0.0, 2.0 * np.pi, res_theta)
    z = np.array([z0, z1], dtype=np.float32)
    th_grid, z_grid = np.meshgrid(th, z)
    x = cx + r * np.cos(th_grid)
    y = cy + r * np.sin(th_grid)
    c = np.zeros_like(x)
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z_grid,
            surfacecolor=c,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=opacity,
            hoverinfo="skip",
        )
    )


def _add_region_floor(fig, scene):
    cfg = scene["config"]
    floor_z = 0.02
    for zone in scene["regionZones"]:
        color = REGION_FILL_COLORS.get(zone["type"], "#E5E7EB")
        _add_cuboid(
            fig,
            cx=0.5 * cfg["mapXMax"],
            cy=zone["centerY"],
            cz=floor_z,
            hx=0.5 * cfg["mapXMax"],
            hy=0.5 * (zone["y1"] - zone["y0"]),
            hz=0.01,
            color=color,
            opacity=0.14,
        )


def _add_region_separators(fig, scene):
    cfg = scene["config"]
    for zone in scene["regionZones"][:-1]:
        y = zone["y1"]
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, cfg["mapXMax"]],
                y=[y, y],
                z=[0.02, 0.02],
                mode="lines",
                line=dict(color="#666666", width=4, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _add_boundary_wireframe(fig, scene):
    cfg = scene["config"]
    x0, x1 = 0.0, cfg["mapXMax"]
    y0, y1 = cfg["mapYMin"], cfg["mapYMax"]
    z0, z1 = 0.0, cfg["mapZMax"]
    edges = [
        ((x0, y0, z0), (x1, y0, z0)),
        ((x1, y0, z0), (x1, y1, z0)),
        ((x1, y1, z0), (x0, y1, z0)),
        ((x0, y1, z0), (x0, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1)),
        ((x1, y0, z1), (x1, y1, z1)),
        ((x1, y1, z1), (x0, y1, z1)),
        ((x0, y1, z1), (x0, y0, z1)),
        ((x0, y0, z0), (x0, y0, z1)),
        ((x1, y0, z0), (x1, y0, z1)),
        ((x1, y1, z0), (x1, y1, z1)),
        ((x0, y1, z0), (x0, y1, z1)),
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs.extend([a[0], b[0], None])
        ys.extend([a[1], b[1], None])
        zs.extend([a[2], b[2], None])
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="#555555", width=4),
            name="Boundary Box",
        )
    )


def render_scene_html(scene, output_html: Path):
    if go is None:
        raise RuntimeError("plotly is not installed. Please install plotly to use preview_map.py")

    fig = go.Figure()
    cfg = scene["config"]

    _add_region_floor(fig, scene)
    _add_region_separators(fig, scene)

    legend_done = set()

    for box in scene["voxels"]:
        region = box["region"]
        color = REGION_COLORS.get(region, "#9CA3AF")
        opacity = 0.14 if region == "boundary" else 0.68
        name = f"{region} voxel"
        showlegend = name not in legend_done and region != "boundary"
        _add_cuboid(fig, box["x"], box["y"], box["z"], box["hx"], box["hy"], box["hz"], color, opacity, name, showlegend)
        if showlegend:
            legend_done.add(name)

    for ball in scene["balls"]:
        region = ball["region"]
        color = REGION_COLORS.get(region, "#9CA3AF")
        _add_sphere(fig, ball["x"], ball["y"], ball["z"], ball["r"], color=color, opacity=0.80, res=10)

    for cyl in scene["cylinders"]:
        region = cyl["region"]
        color = REGION_COLORS.get(region, "#9CA3AF")
        _add_cylinder_z(fig, cyl["x"], cyl["y"], cyl["r"], 0.0, cfg["mapZMax"], color=color, opacity=0.78, res_theta=14)

    _add_boundary_wireframe(fig, scene)

    fig.add_trace(
        go.Scatter3d(
            x=[scene["start"]["x"]],
            y=[scene["start"]["y"]],
            z=[scene["start"]["z"]],
            mode="markers+text",
            marker=dict(size=8, color="#12B76A", symbol="diamond"),
            text=["START"],
            textposition="top center",
            name="Start",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[scene["goal"]["x"]],
            y=[scene["goal"]["y"]],
            z=[scene["goal"]["z"]],
            mode="markers+text",
            marker=dict(size=8, color="#F04438", symbol="x"),
            text=["GOAL"],
            textposition="top center",
            name="Goal",
        )
    )

    title_parts = [
        "Env Map Preview",
        f"source={scene['source']}",
        f"regions={'/'.join(scene['order'])}",
    ]
    if scene["seed"] is not None:
        title_parts.append(f"seed={scene['seed']}")

    fig.update_layout(
        title=" | ".join(title_parts),
        template="plotly_white",
        showlegend=True,
        scene=dict(
            xaxis=dict(title="X", range=[-0.2, cfg["mapXMax"] + 0.2], backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Y", range=[cfg["mapYMin"] - 0.4, cfg["mapYMax"] + 0.4], backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="Z", range=[-0.1, cfg["mapZMax"] + 0.3], backgroundcolor="rgba(0,0,0,0)"),
            # Keep true world-scale proportions so cubes do not look stretched.
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=-1.65, z=1.15)),
        ),
        margin=dict(l=10, r=10, b=10, t=50),
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs=True, full_html=True, auto_open=False)


def maybe_open_browser(path: Path, open_browser: bool) -> None:
    if not open_browser:
        return
    try:
        webbrowser.open(path.resolve().as_uri(), new=2)
    except Exception as exc:
        print(f"[preview_map] failed to open browser automatically: {exc}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(description="Preview env_multi.py maps in Plotly 3D without starting training.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for on-the-fly map generation")
    parser.add_argument("--batch-index", type=int, default=0, help="Which batch sample to preview")
    parser.add_argument("--compact-two-zone-map", action="store_true", help="Use compact easy/hard two-zone layout")
    parser.add_argument("--disable-u-local-optimum", action="store_true", help="Disable the u-minimal region when not using compact layout")
    parser.add_argument("--unified-four-maps", dest="unified_four_maps", action="store_true",
                        help="Enable unified four-map mode (easy/hard/u-min/hairpin)")
    parser.add_argument("--no-unified-four-maps", dest="unified_four_maps", action="store_false",
                        help="Disable unified four-map mode and use legacy multi-region layout")
    parser.set_defaults(unified_four_maps=True)
    parser.add_argument("--map-type", type=str, default="cycle",
                        choices=["cycle", "easy", "hard", "u-min", "u_min", "hairpin"],
                        help="Force one map type when unified four-map mode is enabled; cycle rotates each reset")
    parser.add_argument("--obstacle-count-scale", type=float, default=0.5, help="Obstacle count scale; 0.5 matches mmgj_transformer default")
    parser.add_argument("--easy-density-scale", type=float, default=float(DEFAULT_EASY_DENSITY_SCALE), help="Density multiplier for easy-region obstacle generation (default follows mmgj_transformer.py)")
    parser.add_argument("--hard-density-scale", type=float, default=float(DEFAULT_HARD_DENSITY_SCALE), help="Density multiplier for hard-region obstacle generation (default follows mmgj_transformer.py)")
    parser.add_argument("--output-html", type=str, default="preview_map.html", help="Output HTML path; overwritten on each run")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Env device for preview generation")
    parser.add_argument("--no-open-browser", action="store_true", help="Write HTML only, do not open a browser tab")
    parser.add_argument(
        "--source-mode",
        choices=["generated", "precomputed", "auto"],
        default="generated",
        help=(
            "Map source mode: generated=always env.reset() (default), "
            "precomputed=always load .pt, auto=load .pt if provided else generate"
        ),
    )
    parser.add_argument("--precomputed-map", type=str, default="", help="Optional path to a single precomputed .pt map to preview")
    parser.add_argument("--precomputed-map-dir", type=str, default="", help="Optional directory containing precomputed .pt map files")
    parser.add_argument("--map-index", type=int, default=0, help="Index in sorted precomputed map dir when --precomputed-map-dir is used")
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(int(args.seed))
    env, device = build_env(args)

    source_label = "generated:env.reset"
    seed_used = int(args.seed)

    if args.source_mode == "generated":
        if args.precomputed_map or args.precomputed_map_dir:
            print(
                "[preview_map] source-mode=generated: ignore --precomputed-map/--precomputed-map-dir",
                file=sys.stderr,
            )
        env.reset()
    else:
        map_path, map_mode = resolve_precomputed_map(args)
        if args.source_mode == "precomputed" and map_path is None:
            raise ValueError("source-mode=precomputed requires --precomputed-map or --precomputed-map-dir")

        if map_path is not None:
            map_data = torch.load(str(map_path), map_location=device)
            env.reset_from_precomputed_map(map_data)
            source_label = (
                f"precomputed:{map_path.name}"
                if map_mode == "file"
                else f"precomputed:{map_path.parent.name}/{map_path.name}"
            )
            seed_used = None
        else:
            env.reset()

    scene = extract_scene_geometry(
        env=env,
        batch_index=args.batch_index,
        source_label=source_label,
        seed=seed_used,
    )

    output_html = Path(args.output_html).expanduser().resolve()
    render_scene_html(scene, output_html)
    maybe_open_browser(output_html, open_browser=not args.no_open_browser)

    print(f"[preview_map] wrote: {output_html}")
    print(f"[preview_map] source: {scene['source']}")
    print(f"[preview_map] order: {scene['order']}")
    print(
        "[preview_map] density_scales: "
        f"easy={float(args.easy_density_scale):.3f}, hard={float(args.hard_density_scale):.3f}"
    )
    print(f"[preview_map] device: {device}")


if __name__ == "__main__":
    main()
