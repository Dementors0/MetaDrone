import argparse
import json
import multiprocessing as mp
import os
import random
import traceback
from typing import Dict, List

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


def _make_env(
    include_u_local_optimum: bool,
    compact_two_zone_map: bool,
    obstacle_count_scale: float,
    scene_scale: float,
    start_goal_plane_y_abs: float,
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
        scene_scale=float(scene_scale),
        random_rotation=False,
        cam_angle=10,
        obstacle_count_scale=float(obstacle_count_scale),
        speed_limit_softness=0.05,
        max_speed_ceiling=10.0,
        hard_vpred_clip=20.0,
        hard_speed_clip=30.0,
        start_goal_plane_y_abs=float(start_goal_plane_y_abs),
        include_u_local_optimum=include_u_local_optimum,
        compact_two_zone_map=compact_two_zone_map,
        wall_physical_feedback=False,
    )


def _axis_window(center: float, half_span: float, origin_axis: float, resolution: float, axis_size: int):
    lo = int(np.floor((center - half_span - origin_axis) / resolution))
    hi = int(np.ceil((center + half_span - origin_axis) / resolution))
    lo = max(0, min(axis_size - 1, lo))
    hi = max(0, min(axis_size - 1, hi))
    if hi < lo:
        hi = lo
    return lo, hi


def _axis_index(value: float, origin_axis: float, resolution: float, axis_size: int):
    idx = int(np.floor((float(value) - float(origin_axis)) / float(resolution)))
    return max(0, min(axis_size - 1, idx))


def _spawn_band_valid_ratio(
    valid_mask: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    y_world: float,
    x_center: float,
    z_center: float,
    half_span: float,
) -> float:
    nx, ny, nz = valid_mask.shape
    y_idx = _axis_index(float(y_world), float(origin[1]), float(resolution), ny)
    x0, x1 = _axis_window(float(x_center), float(half_span), float(origin[0]), float(resolution), nx)
    z0, z1 = _axis_window(float(z_center), float(half_span), float(origin[2]), float(resolution), nz)
    band = valid_mask[x0 : x1 + 1, y_idx, z0 : z1 + 1]
    if band.size <= 0:
        return 0.0
    return float(band.mean())


def _spawn_band_reachable_ratio(
    potential: np.ndarray,
    occupancy: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    y_world: float,
    x_center: float,
    z_center: float,
    half_span: float,
) -> float:
    reachable_mask = np.isfinite(potential) & (occupancy == 0)
    return _spawn_band_valid_ratio(
        valid_mask=reachable_mask,
        origin=origin,
        resolution=resolution,
        y_world=y_world,
        x_center=x_center,
        z_center=z_center,
        half_span=half_span,
    )


def _collect_goal_band_sources(
    occupancy: np.ndarray,
    origin: np.ndarray,
    resolution: float,
    y_world: float,
    x_center: float,
    z_center: float,
    half_span: float,
    y_pad_cells: int = 1,
    max_sources: int = 4096,
):
    nx, ny, nz = occupancy.shape
    y_idx = _axis_index(float(y_world), float(origin[1]), float(resolution), ny)
    x0, x1 = _axis_window(float(x_center), float(half_span), float(origin[0]), float(resolution), nx)
    z0, z1 = _axis_window(float(z_center), float(half_span), float(origin[2]), float(resolution), nz)

    ys = list(range(max(0, y_idx - int(y_pad_cells)), min(ny - 1, y_idx + int(y_pad_cells)) + 1))
    sources = []
    for yy in ys:
        for xx in range(x0, x1 + 1):
            for zz in range(z0, z1 + 1):
                if occupancy[xx, yy, zz] == 0:
                    sources.append((xx, yy, zz))
                    if len(sources) >= int(max_sources):
                        return sources
    return sources


def _candidate_spawn_bands(env) -> List[Dict]:
    bands = []
    seen = set()
    order = env.region_order[0] if len(env.region_order) > 0 else tuple(env.region_types)
    for slot_idx, region_type in enumerate(order):
        y0 = float(env.map_y_min + slot_idx * env.region_length)
        y1 = float(y0 + env.region_length)
        for lo, hi, edge in (
            (y0, y0 + env.blank_length, "lo"),
            (y1 - env.blank_length, y1, "hi"),
        ):
            center = 0.5 * (lo + hi)
            key = round(center, 6)
            if key in seen:
                continue
            seen.add(key)
            bands.append({
                "y_lo": float(lo),
                "y_hi": float(hi),
                "y": float(center),
                "region": str(region_type),
                "slot": int(slot_idx),
                "edge": edge,
            })
    return bands


def _build_single_map(task: Dict):
    """Worker entry: generate one map and save to disk. Returns status dict."""
    map_id = int(task["map_id"])
    save_dir = task["save_dir"]
    seed = int(task["seed"])
    include_u = bool(task["include_u_local_optimum"])
    compact_two_zone = bool(task.get("compact_two_zone_map", False))

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    try:
        env = _make_env(
            include_u_local_optimum=include_u,
            compact_two_zone_map=compact_two_zone,
            obstacle_count_scale=float(task.get("obstacle_count_scale", 0.3)),
            scene_scale=float(task.get("scene_scale", 0.5)),
            start_goal_plane_y_abs=float(task.get("start_goal_plane_y_abs", 25.0)),
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
            extra_inflate=float(task.get("extra_inflate", 0.0)),
            bounds=bounds,
            z_min=float(task["z_min"]),
            z_max=float(task["z_max"]),
        )

        spawn_half_span = float(getattr(env, "fixed_spawn_half_span", 1.0))
        min_global = float(task.get("min_global_valid_ratio", 0.0))
        min_start = float(task.get("min_start_valid_ratio", 0.0))
        min_goal = float(task.get("min_goal_valid_ratio", 0.0))

        candidates = _candidate_spawn_bands(env)
        best = None
        best_diag = None
        for goal_band in candidates:
            goal_world_candidate = np.asarray(
                [env.spawn_x_center, float(goal_band["y"]), env.spawn_z_center],
                dtype=np.float32,
            )
            goal_idx_candidate = world_to_grid_index(goal_world_candidate, origin, shape, float(task["resolution"]))
            goal_sources_candidate = _collect_goal_band_sources(
                occupancy=occupancy,
                origin=origin,
                resolution=float(task["resolution"]),
                y_world=float(goal_band["y"]),
                x_center=float(env.spawn_x_center),
                z_center=float(env.spawn_z_center),
                half_span=spawn_half_span,
                y_pad_cells=1,
                max_sources=int(task.get("max_goal_sources", 4096)),
            )
            if len(goal_sources_candidate) == 0:
                continue

            potential_candidate, goal_idx_used_candidate = compute_dijkstra_potential(
                occupancy=occupancy,
                goal_idx=goal_idx_candidate,
                resolution=float(task["resolution"]),
                goal_sources=goal_sources_candidate,
            )
            guide_dir_candidate = compute_descending_vector_field(potential_candidate, occupancy)
            dir_norm = np.linalg.norm(guide_dir_candidate, axis=-1)
            valid_mask = np.isfinite(potential_candidate) & (occupancy == 0) & (dir_norm > 1e-6)
            global_valid_ratio_candidate = float(valid_mask.mean())
            goal_valid_ratio_candidate = _spawn_band_valid_ratio(
                valid_mask=valid_mask,
                origin=origin,
                resolution=float(task["resolution"]),
                y_world=float(goal_band["y"]),
                x_center=float(env.spawn_x_center),
                z_center=float(env.spawn_z_center),
                half_span=spawn_half_span,
            )
            goal_reachable_ratio_candidate = _spawn_band_reachable_ratio(
                potential=potential_candidate,
                occupancy=occupancy,
                origin=origin,
                resolution=float(task["resolution"]),
                y_world=float(goal_band["y"]),
                x_center=float(env.spawn_x_center),
                z_center=float(env.spawn_z_center),
                half_span=spawn_half_span,
            )

            for start_band in candidates:
                if abs(float(start_band["y"]) - float(goal_band["y"])) < 1e-6:
                    continue
                start_valid_ratio_candidate = _spawn_band_valid_ratio(
                    valid_mask=valid_mask,
                    origin=origin,
                    resolution=float(task["resolution"]),
                    y_world=float(start_band["y"]),
                    x_center=float(env.spawn_x_center),
                    z_center=float(env.spawn_z_center),
                    half_span=spawn_half_span,
                )
                start_reachable_ratio_candidate = _spawn_band_reachable_ratio(
                    potential=potential_candidate,
                    occupancy=occupancy,
                    origin=origin,
                    resolution=float(task["resolution"]),
                    y_world=float(start_band["y"]),
                    x_center=float(env.spawn_x_center),
                    z_center=float(env.spawn_z_center),
                    half_span=spawn_half_span,
                )
                distance = abs(float(start_band["y"]) - float(goal_band["y"]))
                diag = {
                    "start_band": start_band,
                    "goal_band": goal_band,
                    "goal_world": goal_world_candidate,
                    "goal_idx": goal_idx_candidate,
                    "goal_idx_used": goal_idx_used_candidate,
                    "goal_sources": goal_sources_candidate,
                    "potential": potential_candidate,
                    "guide_dir": guide_dir_candidate,
                    "global_valid_ratio": global_valid_ratio_candidate,
                    "start_valid_ratio": float(start_valid_ratio_candidate),
                    "start_reachable_ratio": float(start_reachable_ratio_candidate),
                    "goal_valid_ratio": float(goal_valid_ratio_candidate),
                    "goal_reachable_ratio": float(goal_reachable_ratio_candidate),
                    "distance": float(distance),
                }
                if best_diag is None or (
                    start_reachable_ratio_candidate,
                    start_valid_ratio_candidate,
                    goal_reachable_ratio_candidate,
                    distance,
                ) > (
                    best_diag["start_reachable_ratio"],
                    best_diag["start_valid_ratio"],
                    best_diag["goal_reachable_ratio"],
                    best_diag["distance"],
                ):
                    best_diag = diag
                if (
                    global_valid_ratio_candidate >= min_global
                    and start_valid_ratio_candidate >= min_start
                    and goal_reachable_ratio_candidate >= min_goal
                ):
                    score = (distance, start_valid_ratio_candidate, goal_reachable_ratio_candidate)
                    if best is None or score > best["score"]:
                        best = dict(diag)
                        best["score"] = score

        if best is None:
            best = best_diag

        if best is None:
            potential = np.full(shape, np.inf, dtype=np.float32)
            guide_dir = np.zeros(shape + (3,), dtype=np.float32)
            goal_world = np.asarray([env.spawn_x_center, env.spawn_goal_y, env.spawn_z_center], dtype=np.float32)
            goal_idx = world_to_grid_index(goal_world, origin, shape, float(task["resolution"]))
            goal_idx_used = goal_idx
            goal_sources = []
            global_valid_ratio = 0.0
            start_valid_ratio = 0.0
            start_reachable_ratio = 0.0
            goal_valid_ratio = 0.0
            goal_reachable_ratio = 0.0
        else:
            env.spawn_start_y = float(best["start_band"]["y"])
            env.spawn_goal_y = float(best["goal_band"]["y"])
            env._spawn_start_bounds = torch.tensor(
                [[best["start_band"]["y_lo"], best["start_band"]["y_hi"]]],
                device=env.device,
                dtype=torch.float32,
            )
            env._spawn_goal_bounds = torch.tensor(
                [[best["goal_band"]["y_lo"], best["goal_band"]["y_hi"]]],
                device=env.device,
                dtype=torch.float32,
            )
            env.reset_drone_only()

            potential = best["potential"]
            guide_dir = best["guide_dir"]
            goal_world = best["goal_world"]
            goal_idx = best["goal_idx"]
            goal_idx_used = best["goal_idx_used"]
            goal_sources = best["goal_sources"]
            global_valid_ratio = best["global_valid_ratio"]
            start_valid_ratio = best["start_valid_ratio"]
            start_reachable_ratio = best["start_reachable_ratio"]
            goal_valid_ratio = best["goal_valid_ratio"]
            goal_reachable_ratio = best["goal_reachable_ratio"]

        save_obj = {
            "map_id": map_id,
            "resolution": float(task["resolution"]),
            "margin": float(task["margin"]),
            "extra_inflate": float(task.get("extra_inflate", 0.0)),
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
            "goal_source_count": int(len(goal_sources)),
            "global_valid_ratio": float(global_valid_ratio),
            "start_band_valid_ratio": float(start_valid_ratio),
            "start_band_reachable_ratio": float(start_reachable_ratio),
            "goal_band_valid_ratio": float(goal_valid_ratio),
            "goal_band_reachable_ratio": float(goal_reachable_ratio),
        }

        reachable = int(np.isfinite(potential).sum())
        total = int(potential.size)
        min_global = float(task.get("min_global_valid_ratio", 0.0))
        min_start = float(task.get("min_start_valid_ratio", 0.0))
        min_goal = float(task.get("min_goal_valid_ratio", 0.0))
        quality_ok = (
            global_valid_ratio >= min_global
            and start_valid_ratio >= min_start
            and goal_reachable_ratio >= min_goal
        )
        if not quality_ok:
            return {
                "ok": False,
                "map_id": map_id,
                "out_path": "",
                "reachable": reachable,
                "total": total,
                "global_valid_ratio": global_valid_ratio,
                "start_valid_ratio": start_valid_ratio,
                "start_reachable_ratio": start_reachable_ratio,
                "goal_valid_ratio": goal_valid_ratio,
                "goal_reachable_ratio": goal_reachable_ratio,
                "spawn_start_y": float(env.spawn_start_y),
                "spawn_goal_y": float(env.spawn_goal_y),
                "goal_source_count": int(len(goal_sources)),
                "error": (
                    "map quality check failed: "
                    f"global_valid_ratio={global_valid_ratio:.6f} (<{min_global:.6f}) or "
                    f"start_valid_ratio={start_valid_ratio:.6f} (<{min_start:.6f}) or "
                    f"goal_reachable_ratio={goal_reachable_ratio:.6f} (<{min_goal:.6f})"
                ),
            }

        out_path = os.path.join(save_dir, f"map_{map_id:03d}.pt")
        torch.save(save_obj, out_path)
        return {
            "ok": True,
            "map_id": map_id,
            "out_path": out_path,
            "reachable": reachable,
            "total": total,
            "global_valid_ratio": global_valid_ratio,
            "start_valid_ratio": start_valid_ratio,
            "start_reachable_ratio": start_reachable_ratio,
            "goal_valid_ratio": goal_valid_ratio,
            "goal_reachable_ratio": goal_reachable_ratio,
            "spawn_start_y": float(env.spawn_start_y),
            "spawn_goal_y": float(env.spawn_goal_y),
            "goal_source_count": int(len(goal_sources)),
            "error": "",
        }
    except Exception:
        return {
            "ok": False,
            "map_id": map_id,
            "out_path": "",
            "reachable": 0,
            "total": 0,
            "global_valid_ratio": 0.0,
            "start_valid_ratio": 0.0,
            "start_reachable_ratio": 0.0,
            "goal_valid_ratio": 0.0,
            "goal_reachable_ratio": 0.0,
            "spawn_start_y": 0.0,
            "spawn_goal_y": 0.0,
            "goal_source_count": 0,
            "error": traceback.format_exc(),
        }


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_maps", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="../precomputed_maps")
    parser.add_argument("--seed", type=int, default=1234)
    # 栅格分辨率（米）：越小越精细，势场更准确，但预构建更慢、缓存更大。
    # 建议先用 0.10~0.15 跑通，再根据需要降到 0.08 左右。
    parser.add_argument("--resolution", type=float, default=0.3)
    # 障碍膨胀边距（米）：给墙体外扩安全缓冲，值越大路径越保守。
    parser.add_argument("--margin", type=float, default=0.15)
    # 额外膨胀项（米）：默认 0，避免对 margin 进行隐式双重膨胀。
    parser.add_argument("--extra_inflate", type=float, default=0.0)
    # 地图质量阈值（设为 0 可关闭对应约束）
    parser.add_argument("--min_global_valid_ratio", type=float, default=0.001,
                        help="Minimum ratio of globally valid guidance voxels (0 disables)")
    parser.add_argument("--min_start_valid_ratio", type=float, default=0.01,
                        help="Minimum valid guidance ratio inside start spawn band (0 disables)")
    parser.add_argument("--min_goal_valid_ratio", type=float, default=0.01,
                        help="Minimum valid guidance ratio inside goal spawn band (0 disables)")
    parser.add_argument("--max_goal_sources", type=int, default=4096,
                        help="Maximum number of free cells sampled in goal spawn band as Dijkstra seeds")
    # 与训练环境对齐的地图生成参数（建议与训练脚本保持一致）
    parser.add_argument("--obstacle_count_scale", type=float, default=0.3,
                        help="Obstacle density multiplier for precomputed-map env generation")
    parser.add_argument("--scene_scale", type=float, default=0.5,
                        help="Scene scale for precomputed-map env generation")
    parser.add_argument("--start_goal_plane_y_abs", type=float, default=25.0,
                        help="Start/goal plane y abs value forwarded to env generation")
    parser.add_argument("--spawn_plane_inset", type=float, default=1.5,
                        help="Deprecated; spawn y planes now use generated spawn bounds")
    # 势场 z 方向覆盖范围（米）：应覆盖飞行高度和目标高度。
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=5.0)
    # 并行进程数：<=0 自动按 CPU 估算；1 为串行；>1 为并行预构建。
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Worker process count (<=0 means auto)")
    # 任务分片大小：每个进程每次领取的地图数量。
    # <=0 自动估算；通常 2~8 比较稳妥。
    parser.add_argument("--chunksize", type=int, default=0,
                        help="Task chunksize for pool.imap_unordered (<=0 means auto)")
    # 单张地图失败后的重试轮数（用于偶发失败容错）。
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Retry rounds for failed map builds")
    # 地图结构开关：与训练脚本保持一致。
    # - 默认关闭：使用当前默认布局（可含三分区，取决于 include_u_local_optimum）
    # - 开启后：仅生成 easy/hard 两分区紧凑地图，Y 向尺寸与起终点平面随之调整
    parser.add_argument("--include_u_local_optimum", dest="include_u_local_optimum", action="store_true")
    parser.add_argument("--no_include_u_local_optimum", dest="include_u_local_optimum", action="store_false")
    parser.add_argument("--compact_two_zone_map", dest="compact_two_zone_map", action="store_true")
    parser.add_argument("--no_compact_two_zone_map", dest="compact_two_zone_map", action="store_false")
    parser.set_defaults(include_u_local_optimum=False)
    parser.set_defaults(compact_two_zone_map=False)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    fail_log_path = os.path.join(args.save_dir, "precompute_failures.log")

    num_workers = _default_num_workers(int(args.num_workers))
    all_ids = list(range(int(args.num_maps)))
    auto_chunksize = max(1, len(all_ids) // max(1, num_workers * 8))
    chunksize = int(args.chunksize) if int(args.chunksize) > 0 else auto_chunksize

    print(
        f"[Precompute] num_maps={args.num_maps}, workers={num_workers}, "
        f"chunksize={chunksize}, max_retries={args.max_retries}, save_dir={args.save_dir}, "
        f"margin={args.margin}, extra_inflate={args.extra_inflate}, "
        f"spawn_source=generated_bounds, "
        f"min_valid(global/start/goal)=({args.min_global_valid_ratio}, {args.min_start_valid_ratio}, {args.min_goal_valid_ratio})"
    )

    pending = all_ids[:]
    round_idx = 0
    while len(pending) > 0 and round_idx <= int(args.max_retries):
        tasks = []
        for map_id in pending:
            tasks.append({
                "map_id": int(map_id),
                "save_dir": args.save_dir,
                "seed": int(args.seed) + int(map_id) + 100000 * round_idx,
                "resolution": float(args.resolution),
                "margin": float(args.margin),
                "extra_inflate": float(args.extra_inflate),
                "z_min": float(args.z_min),
                "z_max": float(args.z_max),
                "min_global_valid_ratio": float(args.min_global_valid_ratio),
                "min_start_valid_ratio": float(args.min_start_valid_ratio),
                "min_goal_valid_ratio": float(args.min_goal_valid_ratio),
                "max_goal_sources": int(args.max_goal_sources),
                "obstacle_count_scale": float(args.obstacle_count_scale),
                "scene_scale": float(args.scene_scale),
                "start_goal_plane_y_abs": float(args.start_goal_plane_y_abs),
                "spawn_plane_inset": float(args.spawn_plane_inset),
                "include_u_local_optimum": bool(args.include_u_local_optimum),
                "compact_two_zone_map": bool(args.compact_two_zone_map),
            })

        print(f"[Precompute] round={round_idx}, pending={len(pending)}")
        results = _run_round(tasks, num_workers=num_workers, chunksize=chunksize)

        next_pending = []
        ok_cnt = 0
        fail_cnt = 0
        for out in results:
            if out["ok"]:
                ok_cnt += 1
                print(
                    f"[OK] map={out['map_id']:03d} saved={out['out_path']} "
                    f"reachable={out['reachable']}/{out['total']} "
                    f"valid(global/start/goal)=({out.get('global_valid_ratio', 0.0):.4f}/"
                    f"{out.get('start_valid_ratio', 0.0):.4f}/{out.get('goal_valid_ratio', 0.0):.4f}) "
                    f"start_reachable={out.get('start_reachable_ratio', 0.0):.4f} "
                    f"goal_reachable={out.get('goal_reachable_ratio', 0.0):.4f} "
                    f"spawn_y=({out.get('spawn_start_y', 0.0):.2f}->{out.get('spawn_goal_y', 0.0):.2f}) "
                    f"goal_sources={int(out.get('goal_source_count', 0))}"
                )
            else:
                fail_cnt += 1
                next_pending.append(int(out["map_id"]))
                err_msg = {
                    "round": round_idx,
                    "map_id": int(out["map_id"]),
                    "global_valid_ratio": float(out.get("global_valid_ratio", 0.0)),
                    "start_valid_ratio": float(out.get("start_valid_ratio", 0.0)),
                    "start_reachable_ratio": float(out.get("start_reachable_ratio", 0.0)),
                    "goal_valid_ratio": float(out.get("goal_valid_ratio", 0.0)),
                    "goal_reachable_ratio": float(out.get("goal_reachable_ratio", 0.0)),
                    "spawn_start_y": float(out.get("spawn_start_y", 0.0)),
                    "spawn_goal_y": float(out.get("spawn_goal_y", 0.0)),
                    "goal_source_count": int(out.get("goal_source_count", 0)),
                    "error": out["error"],
                }
                with open(fail_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(err_msg, ensure_ascii=False) + "\n")
                print(
                    f"[FAIL] map={out['map_id']:03d} "
                    f"valid(global/start/goal)=({out.get('global_valid_ratio', 0.0):.4f}/"
                    f"{out.get('start_valid_ratio', 0.0):.4f}/{out.get('goal_valid_ratio', 0.0):.4f}) "
                    f"start_reachable={out.get('start_reachable_ratio', 0.0):.4f} "
                    f"goal_reachable={out.get('goal_reachable_ratio', 0.0):.4f} "
                    f"spawn_y=({out.get('spawn_start_y', 0.0):.2f}->{out.get('spawn_goal_y', 0.0):.2f}) "
                    f"goal_sources={int(out.get('goal_source_count', 0))} "
                    f"(logged to {fail_log_path})"
                )

        print(f"[Precompute] round={round_idx} done, ok={ok_cnt}, fail={fail_cnt}")
        pending = sorted(set(next_pending))
        round_idx += 1

    if len(pending) > 0:
        print(f"[Precompute] unfinished maps after retries: {pending}")
        print(f"[Precompute] see failure log: {fail_log_path}")
        raise SystemExit(2)

    print(f"[Precompute] all maps completed: {args.num_maps}")


if __name__ == "__main__":
    main()
