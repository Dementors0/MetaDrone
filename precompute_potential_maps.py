import argparse
import json
import math
import multiprocessing as mp
import os
import random
import traceback
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from env_multi import Env, DEFAULT_EASY_DENSITY_SCALE, DEFAULT_HARD_DENSITY_SCALE
from potential_map_utils import (
    build_occupancy_grid_from_obstacles,
    compute_descending_vector_field,
    compute_dijkstra_potential,
    world_to_grid_index,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


MAP_TYPE_ORDER = ("hard", "easy", "u-min", "hairpin")
_MAP_TYPE_PREFIX = {
    "hard": "hard",
    "easy": "easy",
    "u-min": "u_min",
    "hairpin": "hairpin",
}


def _map_type_prefix(map_type: str) -> str:
    if map_type not in _MAP_TYPE_PREFIX:
        raise ValueError(f"Unsupported map_type: {map_type}")
    return _MAP_TYPE_PREFIX[map_type]


def _ensure_np(data, cols: int) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, cols), dtype=np.float32)
    return arr.reshape(-1, cols).astype(np.float32)


def _format_map_id(map_id) -> str:
    if isinstance(map_id, int):
        return f"{map_id:03d}"
    return str(map_id)


def _has_reachable_free_near_index(
    potential: np.ndarray,
    occupancy: np.ndarray,
    center_idx: Tuple[int, int, int],
    radius_xy: int = 6,
    radius_z: int = 2,
) -> bool:
    cx, cy, cz = center_idx
    nx, ny, nz = potential.shape
    x0 = max(0, int(cx) - int(radius_xy))
    x1 = min(nx, int(cx) + int(radius_xy) + 1)
    y0 = max(0, int(cy) - int(radius_xy))
    y1 = min(ny, int(cy) + int(radius_xy) + 1)
    z0 = max(0, int(cz) - int(radius_z))
    z1 = min(nz, int(cz) + int(radius_z) + 1)

    local_p = potential[x0:x1, y0:y1, z0:z1]
    local_occ = occupancy[x0:x1, y0:y1, z0:z1]
    if local_p.size == 0 or local_occ.size == 0:
        return False
    return bool(np.any(np.isfinite(local_p) & (local_occ == 0)))


def _make_env(
    include_u_local_optimum: bool,
    compact_two_zone_map: bool,
    easy_density_scale: float = DEFAULT_EASY_DENSITY_SCALE,
    hard_density_scale: float = DEFAULT_HARD_DENSITY_SCALE,
):
    return Env(
        batch_size=1,
        width=64,
        height=48,
        grad_decay=0.4,
        device="cpu",
        fov_x_half_tan=0.53,
        single=True,
        gate=False,
        ground_voxels=False,
        scaffold=False,
        speed_mtp=1.0,
        scene_scale=1.0,
        random_rotation=False,
        cam_angle=10,
        obstacle_count_scale=0.5,
        easy_density_scale=float(easy_density_scale),
        hard_density_scale=float(hard_density_scale),
        speed_limit_softness=0.05,
        max_speed_ceiling=10.0,
        hard_vpred_clip=20.0,
        hard_speed_clip=30.0,
        start_goal_plane_y_abs=25.0,
        include_u_local_optimum=include_u_local_optimum,
        compact_two_zone_map=compact_two_zone_map,
        wall_physical_feedback=False,
    )


def _build_boundary_voxels_for_length(env: Env, map_length: float) -> List[List[float]]:
    y_half = 0.5 * float(map_length)
    return [
        [0.0, 0.0, env.spawn_z_center, env.boundary_half, y_half, env.spawn_z_center],
        [env.map_x_max, 0.0, env.spawn_z_center, env.boundary_half, y_half, env.spawn_z_center],
        [env.spawn_x_center, -y_half, env.spawn_z_center, env.spawn_x_center, env.boundary_half, env.spawn_z_center],
        [env.spawn_x_center, y_half, env.spawn_z_center, env.spawn_x_center, env.boundary_half, env.spawn_z_center],
        [env.spawn_x_center, 0.0, 0.0, env.spawn_x_center, y_half, env.boundary_half],
        [env.spawn_x_center, 0.0, env.map_z_max, env.spawn_x_center, y_half, env.boundary_half],
    ]


def _distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-8:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def _distance_to_polyline(px: float, py: float, polyline: Sequence[Tuple[float, float]]) -> float:
    if len(polyline) <= 1:
        return float("inf")
    best = float("inf")
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        d = _distance_point_to_segment(px, py, ax, ay, bx, by)
        if d < best:
            best = d
    return best


def _build_dense_fill_region(env: Env, y_min: float, y_max: float, map_length: float):
    bounds = {
        "x_lo": 0.45,
        "x_hi": env.map_x_max - 0.45,
        "y_lo": y_min + 0.35,
        "y_hi": y_max - 0.35,
    }

    area = max(1.0, (bounds["x_hi"] - bounds["x_lo"]) * (bounds["y_hi"] - bounds["y_lo"]))
    trunk_count = int(round(0.42 * area))
    trunk_count = max(48, min(110, trunk_count))
    if map_length <= 10.0:
        trunk_count = max(36, int(round(0.7 * trunk_count)))

    pair_gap = 0.10
    placed = []

    for _ in range(trunk_count):
        spec = env._make_scatter_spec("cyl")
        if env._try_place_forest_obstacle(spec, placed, bounds, pair_gap, max_tries=90):
            continue
        env._try_place_forest_obstacle(spec, placed, bounds, pair_gap * 0.65, max_tries=45)

    placed = env._attach_objects_to_cylinders(placed, bounds, mode="scatter", difficulty="hard")

    extra_count = max(8, int(round(0.22 * trunk_count)))
    for _ in range(extra_count):
        kind = "ball" if random.random() < 0.58 else "box"
        spec = env._make_scatter_spec(kind)
        if env._try_place_forest_obstacle(spec, placed, bounds, pair_gap * 0.50, max_tries=50):
            continue
        env._try_place_forest_obstacle(spec, placed, bounds, pair_gap * 0.30, max_tries=25)

    return placed


def _carve_path_from_obstacles(
    placed: Sequence[Dict],
    centerline_xy: Sequence[Tuple[float, float]],
    path_width: float,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
):
    half_w = 0.5 * float(path_width)
    carve_margin = 0.05
    start_goal_pad = 0.24

    kept = []
    for obs in placed:
        ox = float(obs["x"])
        oy = float(obs["y"])
        r = float(obs.get("footprint_r", 0.25))

        d_path = _distance_to_polyline(ox, oy, centerline_xy)
        if d_path <= half_w + r + carve_margin:
            continue

        if math.hypot(ox - start_xy[0], oy - start_xy[1]) <= half_w + r + start_goal_pad:
            continue
        if math.hypot(ox - goal_xy[0], oy - goal_xy[1]) <= half_w + r + start_goal_pad:
            continue

        kept.append(obs)

    return kept


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _append_voxel_box(
    env: Env,
    voxels: List[List[float]],
    x_center: float,
    y_center: float,
    hx: float,
    hy: float,
    hz: float = None,
    z_center: float = None,
):
    if hx <= 1e-4 or hy <= 1e-4:
        return
    voxels.append([
        float(x_center),
        float(y_center),
        float(env.spawn_z_center if z_center is None else z_center),
        float(hx),
        float(hy),
        float(env.inner_wall_hz if hz is None else hz),
    ])


def _append_vertical_wall_from_boundary(
    env: Env,
    voxels: List[List[float]],
    x_boundary: float,
    outward_sign: float,
    y0: float,
    y1: float,
    wall_thickness: float,
):
    y_lo = min(y0, y1)
    y_hi = max(y0, y1)
    if y_hi <= y_lo:
        return
    wall_half = 0.5 * float(wall_thickness)
    _append_voxel_box(
        env=env,
        voxels=voxels,
        x_center=float(x_boundary) + float(outward_sign) * wall_half,
        y_center=0.5 * (y_lo + y_hi),
        hx=wall_half,
        hy=0.5 * (y_hi - y_lo),
    )


def _append_horizontal_wall_from_boundary(
    env: Env,
    voxels: List[List[float]],
    y_boundary: float,
    outward_sign: float,
    x0: float,
    x1: float,
    wall_thickness: float,
):
    x_lo = min(x0, x1)
    x_hi = max(x0, x1)
    if x_hi <= x_lo:
        return
    wall_half = 0.5 * float(wall_thickness)
    _append_voxel_box(
        env=env,
        voxels=voxels,
        x_center=0.5 * (x_lo + x_hi),
        y_center=float(y_boundary) + float(outward_sign) * wall_half,
        hx=0.5 * (x_hi - x_lo),
        hy=wall_half,
    )


def _design_corridor_params(env: Env):
    # Use conservative max training radius so every map honors the minimum-clearance rule.
    drone_radius_ref = 0.15
    drone_width = 2.0 * drone_radius_ref
    corridor_width = max(float(env.two_drone_passage_width), 3.0 * drone_width)
    corridor_width = _clamp(corridor_width, 0.65, float(env.map_x_max) - 1.40)
    min_clear_width = 1.5 * drone_radius_ref
    return drone_radius_ref, corridor_width, min_clear_width


def _add_embedded_narrowings_vertical(
    env: Env,
    voxels: List[List[float]],
    balls: List[List[float]],
    x_center: float,
    y0: float,
    y1: float,
    corridor_width: float,
    min_clear_width: float,
) -> float:
    if y1 - y0 <= 1.2:
        return corridor_width

    half_w = 0.5 * corridor_width
    left_inner_x = x_center - half_w
    right_inner_x = x_center + half_w
    min_gap_seen = corridor_width

    for frac in (0.24, 0.50, 0.76):
        y_mid = y0 + frac * (y1 - y0) + random.uniform(-0.16, 0.16)
        y_mid = _clamp(y_mid, y0 + 0.45, y1 - 0.45)

        gap_lo = min(corridor_width - 0.10, min_clear_width + 0.14)
        gap_hi = max(gap_lo, corridor_width - 0.16)
        gap = random.uniform(gap_lo, gap_hi) if gap_hi > gap_lo + 1e-6 else gap_lo
        gap = _clamp(gap, min_clear_width + 0.10, corridor_width - 0.08)

        removable = max(0.0, corridor_width - gap)
        if removable <= 1e-6:
            continue

        if random.random() < 0.55:
            split = random.uniform(0.30, 0.70)
            left_depth = removable * split
            right_depth = removable - left_depth
        elif random.random() < 0.5:
            left_depth, right_depth = removable, 0.0
        else:
            left_depth, right_depth = 0.0, removable

        local_hy = random.uniform(0.22, 0.36)
        if left_depth > 0.06:
            _append_voxel_box(
                env=env,
                voxels=voxels,
                x_center=left_inner_x + 0.5 * left_depth,
                y_center=y_mid,
                hx=0.5 * left_depth,
                hy=local_hy,
            )
        if right_depth > 0.06:
            _append_voxel_box(
                env=env,
                voxels=voxels,
                x_center=right_inner_x - 0.5 * right_depth,
                y_center=y_mid,
                hx=0.5 * right_depth,
                hy=local_hy,
            )
        min_gap_seen = min(min_gap_seen, corridor_width - left_depth - right_depth)

    for frac in (0.36, 0.68):
        y_mid = y0 + frac * (y1 - y0) + random.uniform(-0.14, 0.14)
        y_mid = _clamp(y_mid, y0 + 0.35, y1 - 0.35)
        r = random.uniform(0.10, 0.14)
        intrusion = min(0.05, 0.42 * r)
        if random.random() < 0.5:
            cx = left_inner_x - (r - intrusion)
        else:
            cx = right_inner_x + (r - intrusion)
        balls.append([float(cx), float(y_mid), float(env.spawn_z_center), float(r)])

    return max(min_clear_width, min_gap_seen - 0.02)


def _add_embedded_narrowings_horizontal(
    env: Env,
    voxels: List[List[float]],
    balls: List[List[float]],
    x0: float,
    x1: float,
    y_center: float,
    corridor_width: float,
    min_clear_width: float,
) -> float:
    x_lo = min(x0, x1)
    x_hi = max(x0, x1)
    if x_hi - x_lo <= 1.2:
        return corridor_width

    half_w = 0.5 * corridor_width
    bottom_inner_y = y_center - half_w
    top_inner_y = y_center + half_w
    min_gap_seen = corridor_width

    for frac in (0.30, 0.58, 0.82):
        x_mid = x_lo + frac * (x_hi - x_lo) + random.uniform(-0.12, 0.12)
        x_mid = _clamp(x_mid, x_lo + 0.30, x_hi - 0.30)

        gap_lo = min(corridor_width - 0.10, min_clear_width + 0.14)
        gap_hi = max(gap_lo, corridor_width - 0.16)
        gap = random.uniform(gap_lo, gap_hi) if gap_hi > gap_lo + 1e-6 else gap_lo
        gap = _clamp(gap, min_clear_width + 0.10, corridor_width - 0.08)

        removable = max(0.0, corridor_width - gap)
        if removable <= 1e-6:
            continue

        if random.random() < 0.55:
            split = random.uniform(0.30, 0.70)
            top_depth = removable * split
            bottom_depth = removable - top_depth
        elif random.random() < 0.5:
            top_depth, bottom_depth = removable, 0.0
        else:
            top_depth, bottom_depth = 0.0, removable

        local_hx = random.uniform(0.24, 0.38)
        if top_depth > 0.06:
            _append_voxel_box(
                env=env,
                voxels=voxels,
                x_center=x_mid,
                y_center=top_inner_y - 0.5 * top_depth,
                hx=local_hx,
                hy=0.5 * top_depth,
            )
        if bottom_depth > 0.06:
            _append_voxel_box(
                env=env,
                voxels=voxels,
                x_center=x_mid,
                y_center=bottom_inner_y + 0.5 * bottom_depth,
                hx=local_hx,
                hy=0.5 * bottom_depth,
            )
        min_gap_seen = min(min_gap_seen, corridor_width - top_depth - bottom_depth)

    for frac in (0.42, 0.74):
        x_mid = x_lo + frac * (x_hi - x_lo) + random.uniform(-0.10, 0.10)
        x_mid = _clamp(x_mid, x_lo + 0.25, x_hi - 0.25)
        r = random.uniform(0.10, 0.14)
        intrusion = min(0.05, 0.42 * r)
        if random.random() < 0.5:
            cy = top_inner_y + (r - intrusion)
        else:
            cy = bottom_inner_y - (r - intrusion)
        balls.append([float(x_mid), float(cy), float(env.spawn_z_center), float(r)])

    return max(min_clear_width, min_gap_seen - 0.02)


def _build_tight_semicircle_cylinders(
    center_x: float,
    center_y: float,
    bend_radius: float,
    cyl_radius: float,
    upward: bool = True,
) -> List[List[float]]:
    if bend_radius <= 1e-4 or cyl_radius <= 1e-4:
        return []
    arc_len = math.pi * bend_radius
    step = max(1e-4, 2.0 * cyl_radius * 0.98)
    steps = max(10, int(math.ceil(arc_len / step)))
    angle_offset = 0.0 if upward else math.pi

    cyl = []
    for i in range(steps + 1):
        theta = angle_offset + (math.pi * i / float(steps))
        x = center_x + bend_radius * math.cos(theta)
        y = center_y + bend_radius * math.sin(theta)
        cyl.append([float(x), float(y), float(cyl_radius)])
    return cyl


def _embed_small_obstacles_on_u_inner_side(
    env: Env,
    voxels: List[List[float]],
    balls: List[List[float]],
    center_x: float,
    center_y: float,
    bend_radius: float,
    wall_cyl_radius: float,
    y_min: float,
    y_max: float,
):
    """Embed small boxes/balls on the concave (inner) side of U-bend wall."""
    if bend_radius <= 1e-4 or wall_cyl_radius <= 1e-4:
        return

    placed_xy: List[Tuple[float, float]] = []
    # Dense textured surface: push obstacle count up aggressively.
    target_count = random.randint(16, 24)
    placed_count = 0
    attempts = 0
    max_attempts = target_count * 22
    while placed_count < target_count and attempts < max_attempts:
        attempts += 1
        theta = random.uniform(0.14 * math.pi, 0.86 * math.pi)
        wall_x = center_x + bend_radius * math.cos(theta)
        wall_y = center_y + bend_radius * math.sin(theta)

        # Concave side points toward the semicircle center.
        nx = (center_x - wall_x) / max(1e-6, bend_radius)
        ny = (center_y - wall_y) / max(1e-6, bend_radius)

        is_ball = (random.random() < 0.50)
        if is_ball:
            r = random.uniform(0.15, 0.27)
            intrusion = random.uniform(0.02, 0.05)
            cx = wall_x + nx * (wall_cyl_radius - intrusion + r)
            cy = wall_y + ny * (wall_cyl_radius - intrusion + r)
            min_sep = 0.225
            if any(math.hypot(cx - px, cy - py) < min_sep for px, py in placed_xy):
                continue
            cx = _clamp(cx, 0.35 + r, float(env.map_x_max) - 0.35 - r)
            cy = _clamp(cy, y_min + 0.35 + r, y_max - 0.35 - r)
            z_lo = float(env.ground_z) + r + 0.04
            z_hi = float(env.ceiling_z) - r - 0.04
            if z_hi <= z_lo:
                cz = float(env.spawn_z_center)
            else:
                # 上下分布：让障碍在墙面上有高度层次，不是一排。
                band = random.random()
                if band < 0.34:
                    cz = random.uniform(z_lo, z_lo + 0.35 * (z_hi - z_lo))
                elif band < 0.68:
                    cz = random.uniform(z_lo + 0.30 * (z_hi - z_lo), z_lo + 0.70 * (z_hi - z_lo))
                else:
                    cz = random.uniform(z_lo + 0.65 * (z_hi - z_lo), z_hi)
            balls.append([float(cx), float(cy), float(cz), float(r)])
            placed_xy.append((cx, cy))
            placed_count += 1
            continue

        # True cube: hx == hy == hz, and size close to ball diameter scale.
        half_side = random.uniform(0.15, 0.27)
        intrusion = random.uniform(0.02, 0.05)
        radial_extent = half_side
        cx = wall_x + nx * (wall_cyl_radius - intrusion + radial_extent)
        cy = wall_y + ny * (wall_cyl_radius - intrusion + radial_extent)
        min_sep = 0.225
        if any(math.hypot(cx - px, cy - py) < min_sep for px, py in placed_xy):
            continue
        cx = _clamp(cx, 0.35 + half_side, float(env.map_x_max) - 0.35 - half_side)
        cy = _clamp(cy, y_min + 0.35 + half_side, y_max - 0.35 - half_side)
        z_lo = float(env.ground_z) + half_side + 0.04
        z_hi = float(env.ceiling_z) - half_side - 0.04
        if z_hi <= z_lo:
            cz = float(env.spawn_z_center)
        else:
            band = random.random()
            if band < 0.34:
                cz = random.uniform(z_lo, z_lo + 0.35 * (z_hi - z_lo))
            elif band < 0.68:
                cz = random.uniform(z_lo + 0.30 * (z_hi - z_lo), z_lo + 0.70 * (z_hi - z_lo))
            else:
                cz = random.uniform(z_lo + 0.65 * (z_hi - z_lo), z_hi)
        _append_voxel_box(
            env=env,
            voxels=voxels,
            x_center=cx,
            y_center=cy,
            hx=half_side,
            hy=half_side,
            hz=half_side,
            z_center=cz,
        )
        placed_xy.append((cx, cy))
        placed_count += 1


def _build_easy_or_hard_geometry(env: Env, map_type: str) -> Dict:
    if map_type not in ("easy", "hard"):
        raise ValueError(f"invalid map_type for random geometry: {map_type}")

    map_length = 16.0 if map_type == "easy" else 8.0
    y_half = 0.5 * map_length
    y_min = -y_half
    y_max = y_half

    balls, cyls, inner_voxels = env._generate_random_region(map_type, y_min, y_max)
    boundary_voxels = _build_boundary_voxels_for_length(env, map_length)
    voxels = boundary_voxels + inner_voxels

    # Keep easy/hard start-goal sampling rules aligned with current config.
    start_goal_offset = 0.55 if map_type == "easy" else 0.50
    spawn_start = (env.spawn_x_center, y_min + start_goal_offset, env.spawn_z_center)
    spawn_goal = (env.spawn_x_center, y_max - start_goal_offset, env.spawn_z_center)

    style = "random_sparse" if map_type == "easy" else "random_compact"
    map_meta = {
        "style": style,
        "map_type": map_type,
    }
    if map_type == "hard":
        map_meta["base_style"] = "easy"

    return {
        "map_type": map_type,
        "map_length": map_length,
        "map_x_max": float(env.map_x_max),
        "map_y_min": float(y_min),
        "map_y_max": float(y_max),
        "map_z_max": float(env.map_z_max),
        "balls": _ensure_np(balls, 4),
        "cyl": _ensure_np(cyls, 3),
        "voxels": _ensure_np(voxels, 6),
        "cyl_h": _ensure_np([], 3),
        "region_order": (map_type,),
        "u_meta": {"map_type": map_type},
        "map_meta": map_meta,
        "spawn_start": spawn_start,
        "spawn_goal": spawn_goal,
        "spawn_start_x_half_span": float(env.fixed_spawn_half_span),
        "spawn_goal_x_half_span": float(env.fixed_spawn_half_span),
        "spawn_start_z_half_span": float(env.fixed_spawn_half_span),
        "spawn_goal_z_half_span": float(env.fixed_spawn_half_span),
    }


def _build_u_min_geometry(env: Env) -> Dict:
    map_type = "u-min"
    map_length = 16.0
    y_half = 0.5 * map_length
    y_min = -y_half
    y_max = y_half

    _, corridor_width, min_clear_width = _design_corridor_params(env)
    wall_thickness = 0.24
    straight_length = min(10.0, map_length - 5.0)

    start_x = float(env.spawn_x_center)
    start_y = y_min + 0.60
    straight_end_y = _clamp(start_y + straight_length, y_min + 5.50, y_max - 3.20)
    wall_entry_y0 = y_min - float(env.boundary_half)

    voxels: List[List[float]] = []
    balls: List[List[float]] = []

    corridor_half = 0.5 * corridor_width
    _append_vertical_wall_from_boundary(
        env=env,
        voxels=voxels,
        x_boundary=start_x - corridor_half,
        outward_sign=-1.0,
        y0=wall_entry_y0,
        y1=straight_end_y,
        wall_thickness=wall_thickness,
    )
    _append_vertical_wall_from_boundary(
        env=env,
        voxels=voxels,
        x_boundary=start_x + corridor_half,
        outward_sign=1.0,
        y0=wall_entry_y0,
        y1=straight_end_y,
        wall_thickness=wall_thickness,
    )

    min_gap_straight = _add_embedded_narrowings_vertical(
        env=env,
        voxels=voxels,
        balls=balls,
        x_center=start_x,
        y0=start_y + 0.85,
        y1=straight_end_y - 0.45,
        corridor_width=corridor_width,
        min_clear_width=min_clear_width,
    )

    u_diameter = 4.0
    u_radius = 0.5 * u_diameter
    u_center_x = start_x
    u_center_y = straight_end_y
    cyls = _build_tight_semicircle_cylinders(
        center_x=u_center_x,
        center_y=u_center_y,
        bend_radius=u_radius,
        cyl_radius=float(env.cyl_tree_radius),
        upward=True,
    )
    _embed_small_obstacles_on_u_inner_side(
        env=env,
        voxels=voxels,
        balls=balls,
        center_x=u_center_x,
        center_y=u_center_y,
        bend_radius=u_radius,
        wall_cyl_radius=float(env.cyl_tree_radius),
        y_min=y_min,
        y_max=y_max,
    )

    goal_y = _clamp(u_center_y + u_radius + 1.35, y_min + 2.20, y_max - 0.85)
    goal_xy = (start_x, goal_y)
    spawn_start = (start_x, start_y, env.spawn_z_center)
    spawn_goal = (goal_xy[0], goal_xy[1], env.spawn_z_center)

    side = 1.0 if random.random() < 0.5 else -1.0
    side_label = "right" if side > 0 else "left"
    route_side_x = _clamp(start_x + side * (u_radius + 0.75), 0.80, env.map_x_max - 0.80)
    centerline = [
        (start_x, start_y),
        (start_x, straight_end_y),
        (route_side_x, u_center_y + 0.26),
        (route_side_x, u_center_y + u_radius + 0.36),
        (goal_xy[0], goal_xy[1]),
    ]
    width_profile = [corridor_width for _ in centerline]

    boundary_voxels = _build_boundary_voxels_for_length(env, map_length)
    min_gap_design = max(min_clear_width, min_gap_straight)
    u_meta = {
        "map_type": map_type,
        "style": "structured_corridor_u",
        "centerline": [(float(x), float(y)) for x, y in centerline],
        "width_profile": [float(w) for w in width_profile],
        "u_meta": {
            "corridor_width": float(corridor_width),
            "straight_length": float(straight_end_y - start_y),
            "wall_entry_y0": float(wall_entry_y0),
            "minimum_clearance_constraint": float(min_clear_width),
            "minimum_clearance_design": float(min_gap_design),
            "u_diameter": float(u_diameter),
            "u_radius": float(u_radius),
            "u_center": [float(u_center_x), float(u_center_y)],
            "u_open_direction": "toward_start",
            "reference_route_side": side_label,
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "corridor_exit_xy": [float(start_x), float(straight_end_y)],
        },
    }

    return {
        "map_type": map_type,
        "map_length": map_length,
        "map_x_max": float(env.map_x_max),
        "map_y_min": float(y_min),
        "map_y_max": float(y_max),
        "map_z_max": float(env.map_z_max),
        "balls": _ensure_np(balls, 4),
        "cyl": _ensure_np(cyls, 3),
        "voxels": _ensure_np(boundary_voxels + voxels, 6),
        "cyl_h": _ensure_np([], 3),
        "region_order": (map_type,),
        "u_meta": u_meta,
        "map_meta": u_meta,
        "spawn_start": spawn_start,
        "spawn_goal": spawn_goal,
        "spawn_start_x_half_span": 0.16,
        "spawn_goal_x_half_span": 0.16,
        "spawn_start_z_half_span": 0.16,
        "spawn_goal_z_half_span": 0.16,
    }


def _build_hairpin_geometry(env: Env) -> Dict:
    map_type = "hairpin"
    map_length = 16.0
    y_half = 0.5 * map_length
    y_min = -y_half
    y_max = y_half

    _, corridor_width, min_clear_width = _design_corridor_params(env)
    wall_thickness = 0.24
    post_turn_length = 3.0

    start_x = float(env.spawn_x_center)
    start_y = y_min + 0.60
    turn_distance_from_map_start = 15.0
    turn_y = y_min + turn_distance_from_map_start
    turn_dir = 1.0 if random.random() < 0.5 else -1.0
    turn_label = "right" if turn_dir > 0 else "left"
    wall_entry_y0 = y_min - float(env.boundary_half)

    x_turn_end = _clamp(start_x + turn_dir * post_turn_length, 0.90, env.map_x_max - 0.90)
    goal_x = _clamp(start_x + turn_dir * (post_turn_length - 0.45), 0.90, env.map_x_max - 0.90)
    goal_y = turn_y

    corridor_half = 0.5 * corridor_width
    x_left = start_x - corridor_half
    x_right = start_x + corridor_half
    y_bot = turn_y - corridor_half
    y_top = turn_y + corridor_half
    x_turn_side = x_right if turn_dir > 0 else x_left
    x_far_side = x_left if turn_dir > 0 else x_right
    outward_turn = 1.0 if turn_dir > 0 else -1.0
    outward_far = -outward_turn

    voxels: List[List[float]] = []
    balls: List[List[float]] = []
    cyls: List[List[float]] = []

    far_extra = 0.65
    near_retract = 0.14

    # 远端墙：更长，封死直行出口；近端墙：更短，给拐弯留入口。
    _append_vertical_wall_from_boundary(
        env=env,
        voxels=voxels,
        x_boundary=x_far_side,
        outward_sign=outward_far,
        y0=wall_entry_y0,
        y1=min(y_top + far_extra, y_max + float(env.boundary_half)),
        wall_thickness=wall_thickness,
    )
    _append_vertical_wall_from_boundary(
        env=env,
        voxels=voxels,
        x_boundary=x_turn_side,
        outward_sign=outward_turn,
        y0=wall_entry_y0,
        y1=y_bot - near_retract,
        wall_thickness=wall_thickness,
    )

    far_x0 = min(x_far_side, x_turn_end) - wall_thickness
    far_x1 = max(x_far_side, x_turn_end) + wall_thickness
    _append_horizontal_wall_from_boundary(
        env=env,
        voxels=voxels,
        y_boundary=y_top,
        outward_sign=1.0,
        x0=far_x0,
        x1=far_x1,
        wall_thickness=wall_thickness,
    )

    near_release = max(0.26, 1.10 * wall_thickness)
    if turn_dir > 0:
        near_x0 = x_turn_side + near_release
        near_x1 = x_turn_end + wall_thickness
    else:
        near_x0 = x_turn_end - wall_thickness
        near_x1 = x_turn_side - near_release
    _append_horizontal_wall_from_boundary(
        env=env,
        voxels=voxels,
        y_boundary=y_bot,
        outward_sign=-1.0,
        x0=near_x0,
        x1=near_x1,
        wall_thickness=wall_thickness,
    )

    # 弯后 3m 通道端部封口，避免从远端直接漏出。
    _append_vertical_wall_from_boundary(
        env=env,
        voxels=voxels,
        x_boundary=x_turn_end,
        outward_sign=turn_dir,
        y0=y_bot,
        y1=y_top,
        wall_thickness=wall_thickness,
    )

    min_gap_straight = _add_embedded_narrowings_vertical(
        env=env,
        voxels=voxels,
        balls=balls,
        x_center=start_x,
        y0=start_y + 0.85,
        y1=turn_y - 0.50,
        corridor_width=corridor_width,
        min_clear_width=min_clear_width,
    )
    # Do not embed extra obstacles in the hairpin turning segment.
    min_gap_turn = corridor_width

    centerline = [
        (start_x, start_y),
        (start_x, turn_y),
        (goal_x, goal_y),
    ]
    boundary_voxels = _build_boundary_voxels_for_length(env, map_length)
    spawn_start = (start_x, start_y, env.spawn_z_center)
    spawn_goal = (goal_x, goal_y, env.spawn_z_center)
    width_profile = [corridor_width for _ in centerline]
    min_gap_design = max(min_clear_width, min(min_gap_straight, min_gap_turn))

    hairpin_meta = {
        "map_type": map_type,
        "style": "structured_corridor_hairpin",
        "centerline": [(float(x), float(y)) for x, y in centerline],
        "width_profile": [float(w) for w in width_profile],
        "turn_meta": {
            "turn_direction": turn_label,
            "corridor_width": float(corridor_width),
            "straight_length": float(turn_y - start_y),
            "turn_distance_from_map_start": float(turn_distance_from_map_start),
            "post_turn_length": float(abs(x_turn_end - start_x)),
            "wall_entry_y0": float(wall_entry_y0),
            "near_release": float(near_release),
            "far_extra": float(far_extra),
            "near_retract": float(near_retract),
            "minimum_clearance_constraint": float(min_clear_width),
            "minimum_clearance_design": float(min_gap_design),
            "turn_point_xy": [float(start_x), float(turn_y)],
            "goal_xy": [float(goal_x), float(goal_y)],
        },
    }

    return {
        "map_type": map_type,
        "map_length": map_length,
        "map_x_max": float(env.map_x_max),
        "map_y_min": float(y_min),
        "map_y_max": float(y_max),
        "map_z_max": float(env.map_z_max),
        "balls": _ensure_np(balls, 4),
        "cyl": _ensure_np(cyls, 3),
        "voxels": _ensure_np(boundary_voxels + voxels, 6),
        "cyl_h": _ensure_np([], 3),
        "region_order": (map_type,),
        "u_meta": hairpin_meta,
        "map_meta": hairpin_meta,
        "spawn_start": spawn_start,
        "spawn_goal": spawn_goal,
        "spawn_start_x_half_span": 0.16,
        "spawn_goal_x_half_span": 0.16,
        "spawn_start_z_half_span": 0.16,
        "spawn_goal_z_half_span": 0.16,
    }


def _build_unified_geometry(env: Env, map_type: str) -> Dict:
    if map_type in ("easy", "hard"):
        return _build_easy_or_hard_geometry(env, map_type)
    if map_type == "u-min":
        return _build_u_min_geometry(env)
    if map_type == "hairpin":
        return _build_hairpin_geometry(env)
    raise ValueError(f"Unsupported map type: {map_type}")


def _build_single_map_legacy(task: Dict):
    map_id = int(task["map_id"])
    save_dir = task["save_dir"]
    seed = int(task["seed"])
    include_u = bool(task["include_u_local_optimum"])
    compact_two_zone = bool(task.get("compact_two_zone_map", False))
    easy_density_scale = float(task.get("easy_density_scale", 1.0))
    hard_density_scale = float(task.get("hard_density_scale", 1.0))

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        env = _make_env(
            include_u_local_optimum=include_u,
            compact_two_zone_map=compact_two_zone,
            easy_density_scale=easy_density_scale,
            hard_density_scale=hard_density_scale,
        )

        bounds = {
            "x_min": -0.5,
            "x_max": float(env.map_x_max) + 0.5,
            "y_min": float(env.map_y_min) - 1.0,
            "y_max": float(env.map_y_max) + 1.0,
        }

        env.reset()

        voxels = env.voxels[0].detach().cpu().numpy().astype(np.float32)
        balls = env.balls[0].detach().cpu().numpy().astype(np.float32)
        cyl = env.cyl[0].detach().cpu().numpy().astype(np.float32)
        cyl_h = env.cyl_h[0].detach().cpu().numpy().astype(np.float32)

        occupancy, origin, shape = build_occupancy_grid_from_obstacles(
            voxels=voxels,
            balls=balls,
            cyl=cyl,
            cyl_h=cyl_h,
            resolution=float(task["resolution"]),
            margin=float(task["margin"]),
            bounds=bounds,
            z_min=float(task["z_min"]),
            z_max=float(task["z_max"]),
        )

        goal_world = np.asarray([env.spawn_x_center, float(env.spawn_goal_y), env.spawn_z_center], dtype=np.float32)
        goal_idx = world_to_grid_index(goal_world, origin, shape, float(task["resolution"]))

        potential, goal_idx_used = compute_dijkstra_potential(
            occupancy=occupancy,
            goal_idx=goal_idx,
            resolution=float(task["resolution"]),
        )
        guide_dir = compute_descending_vector_field(potential, occupancy)

        save_obj = {
            "map_id": map_id,
            "resolution": float(task["resolution"]),
            "margin": float(task["margin"]),
            "z_min": float(task["z_min"]),
            "z_max": float(task["z_max"]),
            "bounds": bounds,
            "grid_origin": torch.from_numpy(origin),
            "grid_shape": torch.tensor(shape, dtype=torch.long),
            "goal_world": torch.from_numpy(goal_world),
            "goal_idx": torch.tensor(goal_idx, dtype=torch.long),
            "goal_idx_used": torch.tensor(goal_idx_used, dtype=torch.long),
            "occupancy": torch.from_numpy(occupancy),
            "potential": torch.from_numpy(potential),
            "guide_dir": torch.from_numpy(guide_dir),
            "voxels": torch.from_numpy(voxels),
            "balls": torch.from_numpy(balls),
            "cyl": torch.from_numpy(cyl),
            "cyl_h": torch.from_numpy(cyl_h),
            "region_order": env.region_order[0] if len(env.region_order) > 0 else tuple(),
            "u_meta": env.u_meta[0] if len(env.u_meta) > 0 else {},
            "spawn_start_bounds": env._spawn_start_bounds[0].detach().cpu(),
            "spawn_goal_bounds": env._spawn_goal_bounds[0].detach().cpu(),
            "spawn_start_y": float(env.spawn_start_y),
            "spawn_goal_y": float(env.spawn_goal_y),
        }

        out_path = os.path.join(save_dir, f"map_{map_id:03d}.pt")
        torch.save(save_obj, out_path)
        reachable = int(np.isfinite(potential).sum())
        total = int(potential.size)
        return {
            "ok": True,
            "map_id": map_id,
            "out_path": out_path,
            "reachable": reachable,
            "total": total,
            "error": "",
        }
    except Exception:
        return {
            "ok": False,
            "map_id": map_id,
            "out_path": "",
            "reachable": 0,
            "total": 0,
            "error": traceback.format_exc(),
        }


def _build_single_map_unified(task: Dict):
    map_id = str(task["map_id"])
    map_type = str(task["map_type"])
    save_dir = task["save_dir"]
    seed = int(task["seed"])
    easy_density_scale = float(task.get("easy_density_scale", 1.0))
    hard_density_scale = float(task.get("hard_density_scale", 1.0))

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        env = _make_env(
            include_u_local_optimum=False,
            compact_two_zone_map=True,
            easy_density_scale=easy_density_scale,
            hard_density_scale=hard_density_scale,
        )
        geom = _build_unified_geometry(env, map_type=map_type)

        bounds = {
            "x_min": -0.5,
            "x_max": float(geom["map_x_max"]) + 0.5,
            "y_min": float(geom["map_y_min"]) - 1.0,
            "y_max": float(geom["map_y_max"]) + 1.0,
        }

        voxels = _ensure_np(geom["voxels"], 6)
        balls = _ensure_np(geom["balls"], 4)
        cyl = _ensure_np(geom["cyl"], 3)
        cyl_h = _ensure_np(geom.get("cyl_h", []), 3)

        occupancy, origin, shape = build_occupancy_grid_from_obstacles(
            voxels=voxels,
            balls=balls,
            cyl=cyl,
            cyl_h=cyl_h,
            resolution=float(task["resolution"]),
            margin=float(task["margin"]),
            bounds=bounds,
            z_min=float(task["z_min"]),
            z_max=float(task["z_max"]),
        )

        spawn_start = tuple(float(v) for v in geom["spawn_start"])
        spawn_goal = tuple(float(v) for v in geom["spawn_goal"])
        goal_world = np.asarray(spawn_goal, dtype=np.float32)
        start_world = np.asarray(spawn_start, dtype=np.float32)

        goal_idx = world_to_grid_index(goal_world, origin, shape, float(task["resolution"]))
        potential, goal_idx_used = compute_dijkstra_potential(
            occupancy=occupancy,
            goal_idx=goal_idx,
            resolution=float(task["resolution"]),
        )
        guide_dir = compute_descending_vector_field(potential, occupancy)

        start_idx = world_to_grid_index(start_world, origin, shape, float(task["resolution"]))
        start_reachable_near = _has_reachable_free_near_index(
            potential=potential,
            occupancy=occupancy,
            center_idx=start_idx,
            radius_xy=6,
            radius_z=2,
        )
        if not start_reachable_near:
            print(
                f"[WARN] no reachable-free voxel near start: "
                f"map_id={map_id}, map_type={map_type}, start_idx={start_idx}"
            )

        start_y = float(spawn_start[1])
        goal_y = float(spawn_goal[1])

        save_obj = {
            "map_id": map_id,
            "map_type": map_type,
            "map_index_within_type": int(task["map_index_within_type"]),
            "global_index": int(task["global_index"]),
            "seed": seed,
            "resolution": float(task["resolution"]),
            "margin": float(task["margin"]),
            "z_min": float(task["z_min"]),
            "z_max": float(task["z_max"]),
            "bounds": bounds,
            "grid_origin": torch.from_numpy(origin),
            "grid_shape": torch.tensor(shape, dtype=torch.long),
            "goal_world": torch.from_numpy(goal_world),
            "goal_idx": torch.tensor(goal_idx, dtype=torch.long),
            "goal_idx_used": torch.tensor(goal_idx_used, dtype=torch.long),
            "start_idx": torch.tensor(start_idx, dtype=torch.long),
            "start_reachable_near": bool(start_reachable_near),
            "occupancy": torch.from_numpy(occupancy),
            "potential": torch.from_numpy(potential),
            "guide_dir": torch.from_numpy(guide_dir),
            "voxels": torch.from_numpy(voxels),
            "balls": torch.from_numpy(balls),
            "cyl": torch.from_numpy(cyl),
            "cyl_h": torch.from_numpy(cyl_h),
            "region_order": tuple(geom.get("region_order", (map_type,))),
            "u_meta": geom.get("u_meta", {"map_type": map_type}),
            "spawn_start_bounds": torch.tensor([start_y - 0.05, start_y + 0.05], dtype=torch.float32),
            "spawn_goal_bounds": torch.tensor([goal_y - 0.05, goal_y + 0.05], dtype=torch.float32),
            "spawn_start_y": start_y,
            "spawn_goal_y": goal_y,
            "spawn_start_x": float(spawn_start[0]),
            "spawn_goal_x": float(spawn_goal[0]),
            "spawn_start_z": float(spawn_start[2]),
            "spawn_goal_z": float(spawn_goal[2]),
            "spawn_start_x_half_span": float(geom["spawn_start_x_half_span"]),
            "spawn_goal_x_half_span": float(geom["spawn_goal_x_half_span"]),
            "spawn_start_z_half_span": float(geom["spawn_start_z_half_span"]),
            "spawn_goal_z_half_span": float(geom["spawn_goal_z_half_span"]),
            "map_x_max": float(geom["map_x_max"]),
            "map_y_min": float(geom["map_y_min"]),
            "map_y_max": float(geom["map_y_max"]),
            "map_z_max": float(geom["map_z_max"]),
            "map_length": float(geom["map_length"]),
            "map_meta": geom.get("map_meta", {"map_type": map_type}),
        }

        out_name = str(task["out_name"])
        out_path = os.path.join(save_dir, out_name)
        torch.save(save_obj, out_path)

        reachable = int(np.isfinite(potential).sum())
        total = int(potential.size)
        return {
            "ok": True,
            "map_id": map_id,
            "out_path": out_path,
            "reachable": reachable,
            "total": total,
            "error": "",
        }
    except Exception:
        return {
            "ok": False,
            "map_id": map_id,
            "out_path": "",
            "reachable": 0,
            "total": 0,
            "error": traceback.format_exc(),
        }


def _build_single_map(task: Dict):
    if bool(task.get("unified_dataset_mode", False)):
        return _build_single_map_unified(task)
    return _build_single_map_legacy(task)


def _default_num_workers(user_value: int) -> int:
    if user_value > 0:
        return user_value
    cpu_total = os.cpu_count() or 8
    return max(1, min(28, cpu_total - 2))


def _run_round(tasks: List[Dict], num_workers: int, chunksize: int):
    results = []
    if num_workers <= 1:
        iterator = tasks
        if tqdm is not None:
            iterator = tqdm(tasks, total=len(tasks), desc="Precompute", ncols=110)
        for t in iterator:
            results.append(_build_single_map(t))
        return results

    start_method = "fork" if os.name == "posix" else "spawn"
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(_build_single_map, tasks, chunksize=max(1, chunksize))
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(tasks), desc=f"Precompute x{num_workers}", ncols=110)
        for out in iterator:
            results.append(out)
    return results


def _build_legacy_tasks(args) -> List[Dict]:
    tasks = []
    for map_id in range(int(args.num_maps)):
        tasks.append({
            "map_id": int(map_id),
            "save_dir": args.save_dir,
            "seed_base": int(args.seed) + int(map_id),
            "resolution": float(args.resolution),
            "margin": float(args.margin),
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "easy_density_scale": float(args.easy_density_scale),
            "hard_density_scale": float(args.hard_density_scale),
            "include_u_local_optimum": bool(args.include_u_local_optimum),
            "compact_two_zone_map": bool(args.compact_two_zone_map),
            "unified_dataset_mode": False,
        })
    return tasks


def _build_unified_tasks(args) -> List[Dict]:
    maps_per_type = int(args.maps_per_type)
    if maps_per_type <= 0:
        raise ValueError("--maps_per_type must be > 0 in unified dataset mode")

    tasks = []
    global_index = 0
    for map_type in MAP_TYPE_ORDER:
        prefix = _map_type_prefix(map_type)
        for local_idx in range(maps_per_type):
            stem = f"{prefix}_{local_idx:03d}"
            tasks.append({
                "map_id": stem,
                "out_name": f"{stem}.pt",
                "map_type": map_type,
                "map_index_within_type": int(local_idx),
                "global_index": int(global_index),
                "save_dir": args.save_dir,
                "seed_base": int(args.seed) + int(global_index),
                "resolution": float(args.resolution),
                "margin": float(args.margin),
                "z_min": float(args.z_min),
                "z_max": float(args.z_max),
                "easy_density_scale": float(args.easy_density_scale),
                "hard_density_scale": float(args.hard_density_scale),
                "unified_dataset_mode": True,
            })
            global_index += 1
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_maps", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="../precomputed_maps")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resolution", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=5.0)
    parser.add_argument(
        "--easy_density_scale",
        type=float,
        default=float(DEFAULT_EASY_DENSITY_SCALE),
        help="Density multiplier for easy-region obstacle generation (default follows mmgj_transformer.py)",
    )
    parser.add_argument(
        "--hard_density_scale",
        type=float,
        default=float(DEFAULT_HARD_DENSITY_SCALE),
        help="Density multiplier for hard-region obstacle generation (default follows mmgj_transformer.py)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Worker process count (<=0 means auto)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help="Task chunksize for pool.imap_unordered (<=0 means auto)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Retry rounds for failed map builds",
    )
    parser.add_argument("--include_u_local_optimum", dest="include_u_local_optimum", action="store_true")
    parser.add_argument("--no_include_u_local_optimum", dest="include_u_local_optimum", action="store_false")
    parser.add_argument("--compact_two_zone_map", dest="compact_two_zone_map", action="store_true")
    parser.add_argument("--no_compact_two_zone_map", dest="compact_two_zone_map", action="store_false")
    parser.set_defaults(include_u_local_optimum=False)
    parser.set_defaults(compact_two_zone_map=False)

    parser.add_argument(
        "--unified_dataset_mode",
        action="store_true",
        help=(
            "Enable unified four-type dataset generation in fixed order: "
            "hard -> easy -> u-min -> hairpin"
        ),
    )
    parser.add_argument(
        "--maps_per_type",
        type=int,
        default=0,
        help="Maps generated for each type when --unified_dataset_mode is enabled",
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    fail_log_path = os.path.join(args.save_dir, "precompute_failures.log")

    if bool(args.unified_dataset_mode):
        tasks_template = _build_unified_tasks(args)
        # Fixed-order contiguous generation is required by unified mode.
        num_workers = 1
    else:
        tasks_template = _build_legacy_tasks(args)
        num_workers = _default_num_workers(int(args.num_workers))

    total_tasks = len(tasks_template)
    auto_chunksize = max(1, total_tasks // max(1, num_workers * 8))
    chunksize = int(args.chunksize) if int(args.chunksize) > 0 else auto_chunksize

    mode_name = "unified" if bool(args.unified_dataset_mode) else "legacy"
    print(
        f"[Precompute] mode={mode_name}, total={total_tasks}, workers={num_workers}, "
        f"chunksize={chunksize}, max_retries={args.max_retries}, save_dir={args.save_dir}"
    )
    print(
        "[Precompute] density_scales: "
        f"easy={float(args.easy_density_scale):.3f}, hard={float(args.hard_density_scale):.3f}"
    )
    if bool(args.unified_dataset_mode):
        print(
            "[Precompute] unified order fixed: "
            "hard -> easy -> u-min -> hairpin "
            f"(maps_per_type={int(args.maps_per_type)})"
        )

    task_by_id = {t["map_id"]: t for t in tasks_template}
    pending_ids = [t["map_id"] for t in tasks_template]

    round_idx = 0
    while len(pending_ids) > 0 and round_idx <= int(args.max_retries):
        tasks = []
        for map_id in pending_ids:
            base = dict(task_by_id[map_id])
            base["seed"] = int(base["seed_base"]) + 100000 * round_idx
            tasks.append(base)

        print(f"[Precompute] round={round_idx}, pending={len(pending_ids)}")
        results = _run_round(tasks, num_workers=num_workers, chunksize=chunksize)

        next_pending = []
        ok_cnt = 0
        fail_cnt = 0
        for out in results:
            rid = out["map_id"]
            rid_show = _format_map_id(rid)
            if out["ok"]:
                ok_cnt += 1
                print(
                    f"[OK] map={rid_show} saved={out['out_path']} "
                    f"reachable={out['reachable']}/{out['total']}"
                )
            else:
                fail_cnt += 1
                next_pending.append(rid)
                err_msg = {
                    "round": round_idx,
                    "map_id": str(rid),
                    "error": out["error"],
                }
                with open(fail_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(err_msg, ensure_ascii=False) + "\n")
                print(f"[FAIL] map={rid_show} (logged to {fail_log_path})")

        print(f"[Precompute] round={round_idx} done, ok={ok_cnt}, fail={fail_cnt}")
        pending_ids = list(dict.fromkeys(next_pending))
        round_idx += 1

    if len(pending_ids) > 0:
        print(f"[Precompute] unfinished maps after retries: {[str(x) for x in pending_ids]}")
        print(f"[Precompute] see failure log: {fail_log_path}")
        raise SystemExit(2)

    print(f"[Precompute] all maps completed: {total_tasks}")


if __name__ == "__main__":
    main()
