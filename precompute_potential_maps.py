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

from env_multi import Env
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


def _make_env(include_u_local_optimum: bool, compact_two_zone_map: bool):
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

    placed = _build_dense_fill_region(env, y_min, y_max, map_length)

    side = 1.0 if random.random() < 0.5 else -1.0
    side_label = "right" if side > 0 else "left"

    start_x = env.spawn_x_center + random.uniform(-0.15, 0.15)
    start_y = y_min + 0.60

    y_straight = y_max - 2.55
    y_u_top = y_max - 1.35
    y_u_bottom = y_max - 3.45

    x_inner = env.spawn_x_center + side * random.uniform(1.05, 1.45)
    # Keep U bend clearly away from outer boundary walls.
    outer_wall_clearance = 1.60
    x_outer = env.map_x_max - outer_wall_clearance if side > 0 else outer_wall_clearance
    x_hidden = x_outer + side * random.uniform(0.85, 1.05)
    x_hidden = max(0.70, min(env.map_x_max - 0.70, x_hidden))

    goal_y = y_u_bottom - random.uniform(0.18, 0.38)
    goal_y = max(y_min + 2.60, min(y_max - 1.00, goal_y))

    centerline = [
        (start_x, start_y),
        (start_x, y_straight),
        (x_inner, y_straight + 0.55),
        (x_inner, y_u_top),
        (x_outer, y_u_top),
        (x_outer, y_u_bottom),
        (x_hidden, goal_y + 0.32),
        (x_hidden, goal_y),
    ]

    path_width = float(env.two_drone_passage_width)
    goal_xy = (x_hidden, goal_y)
    start_xy = (start_x, start_y)
    kept = _carve_path_from_obstacles(
        placed=placed,
        centerline_xy=centerline,
        path_width=path_width,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )

    balls, cyls, vox = env._packed_obstacle_lists(kept)
    boundary_voxels = _build_boundary_voxels_for_length(env, map_length)

    # Structural U outer wall to create the hidden pocket zone near map boundary.
    wall_x = x_outer - side * 0.55
    wall_half_thickness = 0.12
    u_struct_voxels = [
        [
            wall_x,
            0.5 * (y_u_top + y_u_bottom),
            env.spawn_z_center,
            wall_half_thickness,
            0.5 * (y_u_top - y_u_bottom),
            env.inner_wall_hz,
        ],
        [
            0.5 * (x_inner + wall_x),
            y_u_bottom,
            env.spawn_z_center,
            0.5 * abs(x_inner - wall_x),
            wall_half_thickness,
            env.inner_wall_hz,
        ],
    ]

    spawn_start = (start_x, start_y, env.spawn_z_center)
    spawn_goal = (goal_xy[0], goal_xy[1], env.spawn_z_center)

    width_profile = [path_width for _ in centerline]
    u_meta = {
        "map_type": map_type,
        "centerline": [(float(x), float(y)) for x, y in centerline],
        "width_profile": [float(w) for w in width_profile],
        "u_meta": {
            "open_side": side_label,
            "exit_span": [float(y_u_bottom), float(y_u_top)],
            "u_span": [float(y_u_bottom), float(y_u_top)],
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "entry_xy": [float(x_inner), float(y_straight + 0.55)],
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
        "voxels": _ensure_np(boundary_voxels + vox + u_struct_voxels, 6),
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

    placed = _build_dense_fill_region(env, y_min, y_max, map_length)

    side = 1.0 if random.random() < 0.5 else -1.0
    start_x = env.spawn_x_center + random.uniform(-0.18, 0.18)
    start_y = y_min + 0.55

    y1 = y_min + 3.40
    y2 = y_min + 6.80
    y3 = y_min + 9.30
    y4 = y_min + 10.70
    y5 = y_min + 11.50

    x1 = start_x + random.uniform(-0.24, 0.24)
    x2 = x1 + side * random.uniform(0.25, 0.55)
    x3 = env.spawn_x_center + side * random.uniform(1.90, 2.45)
    x4 = x3 - side * random.uniform(0.45, 0.75)
    x5 = env.spawn_x_center - side * random.uniform(0.55, 0.95)

    centerline = [
        (start_x, start_y),
        (x1, y1),
        (x2, y2),
        (x3, y3),
        (x4, y4),
        (x5, y5),
    ]

    path_width = float(env.two_drone_passage_width)
    start_xy = (start_x, start_y)
    goal_xy = (x5, y5)
    kept = _carve_path_from_obstacles(
        placed=placed,
        centerline_xy=centerline,
        path_width=path_width,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )

    balls, cyls, vox = env._packed_obstacle_lists(kept)
    boundary_voxels = _build_boundary_voxels_for_length(env, map_length)

    # Mid-wall strengthens the sharp-turn + short-turnback hairpin behavior.
    wall_half_thickness = 0.12
    mid_wall_x = env.spawn_x_center + side * 1.05
    mid_wall = [
        [
            mid_wall_x,
            0.5 * (y2 + y4),
            env.spawn_z_center,
            wall_half_thickness,
            0.5 * (y4 - y2),
            env.inner_wall_hz,
        ]
    ]

    spawn_start = (start_x, start_y, env.spawn_z_center)
    spawn_goal = (goal_xy[0], goal_xy[1], env.spawn_z_center)
    width_profile = [path_width for _ in centerline]

    hairpin_meta = {
        "map_type": map_type,
        "centerline": [(float(x), float(y)) for x, y in centerline],
        "width_profile": [float(w) for w in width_profile],
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
        "voxels": _ensure_np(boundary_voxels + vox + mid_wall, 6),
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

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        env = _make_env(include_u_local_optimum=include_u, compact_two_zone_map=compact_two_zone)

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

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        env = _make_env(include_u_local_optimum=False, compact_two_zone_map=True)
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
