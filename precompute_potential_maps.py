import argparse
import json
import multiprocessing as mp
import os
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


def _build_single_map(task: Dict):
    """Worker entry: generate one map and save to disk. Returns status dict."""
    map_id = int(task["map_id"])
    save_dir = task["save_dir"]
    seed = int(task["seed"])
    include_u = bool(task["include_u_local_optimum"])
    compact_two_zone = bool(task.get("compact_two_zone_map", False))

    np.random.seed(seed)
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
        f"chunksize={chunksize}, max_retries={args.max_retries}, save_dir={args.save_dir}"
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
                "z_min": float(args.z_min),
                "z_max": float(args.z_max),
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
                    f"reachable={out['reachable']}/{out['total']}"
                )
            else:
                fail_cnt += 1
                next_pending.append(int(out["map_id"]))
                err_msg = {
                    "round": round_idx,
                    "map_id": int(out["map_id"]),
                    "error": out["error"],
                }
                with open(fail_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(err_msg, ensure_ascii=False) + "\n")
                print(f"[FAIL] map={out['map_id']:03d} (logged to {fail_log_path})")

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
