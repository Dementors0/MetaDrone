"""Precomputed-map classification and curriculum helpers."""

from collections import defaultdict
import os

import numpy as np
import torch


def _precomputed_map_type_from_path(path):
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    if stem.startswith("easy_"):
        return "easy"
    if stem.startswith("hairpin_"):
        return "hairpin"
    if stem.startswith("u_min_"):
        return "u_min"
    if stem.startswith("hard_"):
        return "hard"
    if stem.startswith("map_"):
        return "legacy"
    return "unknown"


def _build_precomputed_map_type_indices(map_cache):
    groups = defaultdict(list)
    for idx, path in enumerate(getattr(map_cache, "map_files", [])):
        groups[_precomputed_map_type_from_path(path)].append(int(idx))
    return {k: v for k, v in groups.items() if len(v) > 0}


def _precomputed_curriculum_stage(iter_idx):
    """Keep every training iteration in the easy-map stage."""
    _ = iter_idx
    return "easy_only", ("easy",)


def _select_precomputed_curriculum_map(active_types, stage_update_count, type_offsets, type_indices):
    map_type = active_types[int(stage_update_count) % len(active_types)]
    candidates = type_indices.get(map_type, [])
    if len(candidates) == 0:
        raise RuntimeError(f"No precomputed maps available for curriculum type: {map_type}")
    offset = int(type_offsets[map_type])
    map_idx = candidates[offset % len(candidates)]
    type_offsets[map_type] = offset + 1
    return int(map_idx), map_type


def _align_env_goal_planes_to_precomputed_map(map_data, env_obj, map_idx_hint=None, tol=1e-3):
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

    if start_y_from_map is None and goal_y_from_map is None:
        return

    old_start = float(getattr(env_obj, "spawn_start_y", -11.5))
    old_goal = float(getattr(env_obj, "spawn_goal_y", 11.5))
    new_start = old_start if start_y_from_map is None else float(start_y_from_map)
    new_goal = old_goal if goal_y_from_map is None else float(goal_y_from_map)

    env_obj.spawn_start_y = new_start
    env_obj.spawn_goal_y = new_goal

    if abs(new_start - old_start) > float(tol) or abs(new_goal - old_goal) > float(tol):
        idx_msg = "unknown" if map_idx_hint is None else str(int(map_idx_hint))
        print(
            f"[PotentialMap] Align start/goal planes to map_idx={idx_msg}: "
            f"start_y {old_start:.3f}->{new_start:.3f}, goal_y {old_goal:.3f}->{new_goal:.3f}"
        )
