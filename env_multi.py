import json
import math
import random
import re
from pathlib import Path

import torch
import quadsim_cuda

from env import (
    Env as BaseEnv,
    run as differentiable_run,
    run_torch as differentiable_run_torch,
    safe_normalize,
    update_state_vec_torch,
    update_state_vec_torch_v2,
)


def _extract_float_default_from_mmgj(src_text: str, arg_name: str):
    pat = (
        r"parser\.add_argument\(\s*['\"]--"
        + re.escape(arg_name)
        + r"['\"].*?default\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )
    m = re.search(pat, src_text, flags=re.S)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def load_density_defaults_from_mmgj(
    fallback_easy: float = 1.0,
    fallback_hard: float = 1.0,
):
    """Read easy/hard density defaults from mmgj_transformer.py.

    This keeps map preview/precompute defaults synchronized with mmgj defaults
    without importing mmgj (which has heavy side effects at import time).
    """
    easy = float(fallback_easy)
    hard = float(fallback_hard)

    sync_path = Path(__file__).resolve().with_name(".mmgj_density_defaults.json")
    if sync_path.exists():
        try:
            payload = json.loads(sync_path.read_text(encoding="utf-8"))
            easy_sync = float(payload.get("easy_density_scale", easy))
            hard_sync = float(payload.get("hard_density_scale", hard))
            return easy_sync, hard_sync
        except Exception:
            pass

    mmgj_path = Path(__file__).resolve().with_name("mmgj_transformer.py")
    try:
        src_text = mmgj_path.read_text(encoding="utf-8")
    except Exception:
        return easy, hard

    easy_from_file = _extract_float_default_from_mmgj(src_text, "easy_density_scale")
    hard_from_file = _extract_float_default_from_mmgj(src_text, "hard_density_scale")
    if easy_from_file is not None:
        easy = float(easy_from_file)
    if hard_from_file is not None:
        hard = float(hard_from_file)
    return easy, hard


DEFAULT_EASY_DENSITY_SCALE, DEFAULT_HARD_DENSITY_SCALE = load_density_defaults_from_mmgj()


class Env(BaseEnv):
    """Three-zone obstacle field for joint meta-learning generalization."""

    def __init__(
        self,
        batch_size,
        width,
        height,
        grad_decay,
        device="cpu",
        fov_x_half_tan=0.53,
        single=False,
        gate=False,
        ground_voxels=False,
        scaffold=False,
        speed_mtp=1,
        scene_scale=1.0,
        random_rotation=False,
        cam_angle=10,
        obstacle_count_scale=1.0,
        easy_density_scale=DEFAULT_EASY_DENSITY_SCALE,
        hard_density_scale=DEFAULT_HARD_DENSITY_SCALE,
        speed_limit_softness=0.05,
        max_speed_ceiling=5.0,
        hard_vpred_clip=20.0,
        hard_speed_clip=30.0,
        start_goal_plane_y_abs=50.0,
        include_u_local_optimum=False,
        compact_two_zone_map=True,
        unified_four_maps=False,
        forced_map_type="",
        unified_map_easy_enable=True,
        unified_map_hard_enable=True,
        unified_map_u_min_enable=True,
        unified_map_hairpin_enable=True,
        unified_map_easy_count=1,
        unified_map_hard_count=1,
        unified_map_u_min_count=1,
        unified_map_hairpin_count=1,
        wall_physical_feedback=False,
    ):
        self.map_x_max = 10.0
        self.compact_two_zone_map = bool(compact_two_zone_map) or bool(unified_four_maps)
        self.map_y_half = 8.0 if self.compact_two_zone_map else 12.0
        self.map_y_min = -self.map_y_half
        self.map_y_max = self.map_y_half
        self.map_z_max = 5.0
        self.ground_z = 0.0
        self.ceiling_z = self.map_z_max
        self.cyl_tree_radius = 0.15  # ~30 cm diameter
        self.object_height = self.map_z_max / 3.0
        self.object_half_height = 0.5 * self.object_height
        self.two_drone_passage_width = 0.60
        self.easy_density_scale = max(0.05, float(easy_density_scale))
        self.hard_density_scale = max(0.05, float(hard_density_scale))
        self.region_types = ("easy", "hard") if self.compact_two_zone_map else (
            ("easy", "hard", "u-minimal") if bool(include_u_local_optimum) else ("hard", "easy", "easy")
        )
        self.region_length = (2.0 * self.map_y_half) / float(len(self.region_types))
        self.blank_length = 1.0
        self.spawn_x_center = 5.0
        self.spawn_z_center = 2.5
        self.spawn_x_half_span = 2.0
        self.spawn_z_half_span = 2.0
        self.fixed_spawn_half_span = 1.0
        self.spawn_start_x = self.spawn_x_center
        self.spawn_goal_x = self.spawn_x_center
        self.spawn_start_z = self.spawn_z_center
        self.spawn_goal_z = self.spawn_z_center
        self.spawn_start_x_half_span = self.fixed_spawn_half_span
        self.spawn_goal_x_half_span = self.fixed_spawn_half_span
        self.spawn_start_z_half_span = self.fixed_spawn_half_span
        self.spawn_goal_z_half_span = self.fixed_spawn_half_span
        self.boundary_thickness = 0.10
        self.boundary_half = 0.5 * self.boundary_thickness
        self.full_wall_hz = 2.45
        self.inner_wall_hz = 2.30
        self.include_u_local_optimum = bool(include_u_local_optimum)
        self.unified_four_maps = bool(unified_four_maps)
        self._four_map_types = ("easy", "hard", "u-min", "hairpin")
        self._four_map_cycle_idx = 0
        self.forced_map_type = self._normalize_map_type_name(forced_map_type)
        self._four_map_enable = {
            "easy": bool(unified_map_easy_enable),
            "hard": bool(unified_map_hard_enable),
            "u-min": bool(unified_map_u_min_enable),
            "hairpin": bool(unified_map_hairpin_enable),
        }
        self._four_map_count = {
            "easy": max(1, int(unified_map_easy_count)),
            "hard": max(1, int(unified_map_hard_count)),
            "u-min": max(1, int(unified_map_u_min_count)),
            "hairpin": max(1, int(unified_map_hairpin_count)),
        }
        self._four_map_enabled_types = [m for m in self._four_map_types if self._four_map_enable.get(m, False)]
        if len(self._four_map_enabled_types) == 0:
            raise ValueError(
                "At least one unified map type must be enabled among easy/hard/u-min/hairpin."
            )
        self._four_map_block_order = []
        self._four_map_block_cursor = 0
        self._four_map_block_remaining = 0
        self._four_map_active_type = ""
        self._unified_builder_fn = None
        # 起终点固定在上下边界内缩 0.5m 的平面上；紧凑地图时会自动变为 y=±7.5。
        self.spawn_start_y = self.map_y_min + 0.5
        self.spawn_goal_y = self.map_y_max - 0.5
        self.precomputed_maps = []
        self.current_map_idx = -1
        self.current_map_type = ""

        super().__init__(
            batch_size=batch_size,
            width=width,
            height=height,
            grad_decay=grad_decay,
            device=device,
            fov_x_half_tan=fov_x_half_tan,
            single=single,
            gate=gate,
            ground_voxels=ground_voxels,
            scaffold=scaffold,
            speed_mtp=speed_mtp,
            scene_scale=scene_scale,
            random_rotation=random_rotation,
            cam_angle=cam_angle,
            obstacle_count_scale=obstacle_count_scale,
            speed_limit_softness=speed_limit_softness,
            max_speed_ceiling=max_speed_ceiling,
            hard_vpred_clip=hard_vpred_clip,
            hard_speed_clip=hard_speed_clip,
            start_goal_plane_y_abs=start_goal_plane_y_abs,
            wall_physical_feedback=wall_physical_feedback,
        )

        self.scene_x_half = self.map_x_max
        self.scene_y_half = self.map_y_half

    def _scaled_region_count(self, base_count, min_count=0):
        return max(min_count, int(round(base_count * self.obstacle_count_scale)))

    def _density_scale_for_difficulty(self, difficulty):
        if difficulty == "easy":
            return self.easy_density_scale
        return self.hard_density_scale

    def _normalize_map_type_name(self, map_type):
        if map_type is None:
            return ""
        name = str(map_type).strip().lower()
        if name in ("", "cycle", "auto"):
            return ""
        if name == "u_min":
            return "u-min"
        if name not in self._four_map_types:
            raise ValueError(f"Unsupported map_type={map_type}. expected one of {self._four_map_types} or cycle")
        return name

    def _pick_unified_map_type(self):
        if self.forced_map_type:
            return self.forced_map_type
        if self._four_map_block_remaining > 0 and self._four_map_active_type:
            self._four_map_block_remaining -= 1
            return self._four_map_active_type

        if self._four_map_block_cursor >= len(self._four_map_block_order):
            enabled_types = list(self._four_map_enabled_types)
            if len(enabled_types) == 1:
                self._four_map_block_order = [enabled_types[0]]
            else:
                # Use CUDA RNG for map-order randomization when available.
                perm = torch.randperm(len(enabled_types), device=self.device).tolist()
                self._four_map_block_order = [enabled_types[idx] for idx in perm]
            self._four_map_block_cursor = 0

        map_type = self._four_map_block_order[self._four_map_block_cursor]
        self._four_map_block_cursor += 1
        self._four_map_active_type = map_type
        self._four_map_block_remaining = max(0, int(self._four_map_count.get(map_type, 1)) - 1)
        self._four_map_cycle_idx += 1
        return map_type

    def _get_unified_builder(self):
        if self._unified_builder_fn is None:
            from precompute_potential_maps import _build_unified_geometry

            self._unified_builder_fn = _build_unified_geometry
        return self._unified_builder_fn

    def _build_boundary_voxels(self):
        # Keep only floor slab; remove side walls and top ceiling enclosure.
        return [
            [self.spawn_x_center, 0.0, 0.0, self.spawn_x_center, self.map_y_half, self.boundary_half],
        ]

    def _should_keep_ceiling_for_current_map(self) -> bool:
        map_type = str(getattr(self, "current_map_type", "")).strip().lower().replace("_", "-")
        return map_type in ("hairpin", "u-min", "u-minimal")

    def _should_keep_side_walls_for_current_map(self) -> bool:
        map_type = str(getattr(self, "current_map_type", "")).strip().lower().replace("_", "-")
        return map_type in ("hairpin",)

    def _strip_side_walls_and_ceiling(
        self,
        voxels: torch.Tensor,
        keep_ceiling: bool = None,
        keep_side_walls: bool = None,
        y_min: float = None,
        y_max: float = None,
    ) -> torch.Tensor:
        """Remove enclosure voxels by map policy: side walls / ceiling can be retained by map type."""
        if not isinstance(voxels, torch.Tensor) or voxels.numel() == 0:
            return voxels
        if keep_ceiling is None:
            keep_ceiling = self._should_keep_ceiling_for_current_map()
        if keep_side_walls is None:
            keep_side_walls = self._should_keep_side_walls_for_current_map()

        squeeze_batch = False
        if voxels.dim() == 2:
            vox = voxels.unsqueeze(0)
            squeeze_batch = True
        elif voxels.dim() == 3:
            vox = voxels
        else:
            return voxels

        tol = max(0.08, float(self.boundary_half) * 1.8)
        local_y_min = float(self.map_y_min) if y_min is None else float(y_min)
        local_y_max = float(self.map_y_max) if y_max is None else float(y_max)
        cx, cy, cz = vox[..., 0], vox[..., 1], vox[..., 2]
        hx, hy, hz = vox[..., 3], vox[..., 4], vox[..., 5]

        side_x = (
            ((cx - 0.0).abs() <= tol) | ((cx - float(self.map_x_max)).abs() <= tol)
        ) & ((hx - float(self.boundary_half)).abs() <= tol)
        side_y = (
            ((cy - local_y_min).abs() <= tol) | ((cy - local_y_max).abs() <= tol)
        ) & ((hy - float(self.boundary_half)).abs() <= tol)
        ceiling = (
            (cz - float(self.map_z_max)).abs() <= tol
        ) & ((hz - float(self.boundary_half)).abs() <= tol)

        if bool(keep_side_walls):
            remove_mask = torch.zeros_like(side_x, dtype=torch.bool)
        else:
            remove_mask = side_x | side_y
        if not bool(keep_ceiling):
            remove_mask = remove_mask | ceiling
        keep_mask = ~remove_mask
        keep_counts = keep_mask.sum(dim=1)
        min_keep = int(keep_counts.min().item())
        max_keep = int(keep_counts.max().item())

        if max_keep <= 0:
            filtered = vox.new_zeros((vox.shape[0], 0, vox.shape[-1]))
        else:
            rows = [vox[b, keep_mask[b], :] for b in range(vox.shape[0])]
            if min_keep == max_keep:
                filtered = torch.stack(rows, dim=0)
            else:
                # Fallback for rare mismatched counts across batch.
                filtered = torch.stack([r[:min_keep] for r in rows], dim=0)

        return filtered[0] if squeeze_batch else filtered

    def _ensure_top_ceiling_if_needed(self, voxels: torch.Tensor) -> torch.Tensor:
        """For hairpin/u-min maps, ensure there is one top ceiling slab even with old caches."""
        if not isinstance(voxels, torch.Tensor) or voxels.dim() != 3:
            return voxels
        if not self._should_keep_ceiling_for_current_map():
            return voxels

        tol = max(0.08, float(self.boundary_half) * 1.8)
        map_y_span = max(1e-6, float(self.map_y_max) - float(self.map_y_min))
        if voxels.numel() > 0:
            cz = voxels[..., 2]
            hx = voxels[..., 3]
            hy = voxels[..., 4]
            hz = voxels[..., 5]
            has_ceiling = (
                (cz - float(self.map_z_max)).abs() <= tol
            ) & ((hz - float(self.boundary_half)).abs() <= tol) & (
                hx >= 0.45 * float(self.map_x_max)
            ) & (
                hy >= 0.45 * map_y_span
            )
            if bool(has_ceiling.any().item()):
                return voxels

        B = int(voxels.shape[0])
        ceiling_row = torch.tensor(
            [self.spawn_x_center, 0.0, self.map_z_max, self.spawn_x_center, self.map_y_half, self.boundary_half],
            device=voxels.device,
            dtype=voxels.dtype,
        ).view(1, 1, 6).repeat(B, 1, 1)
        return torch.cat([voxels, ceiling_row], dim=1)

    def _make_blank_zone(self, y0, y1):
        return {
            "y_lo": y0,
            "y_hi": y1,
            "y_center": 0.5 * (y0 + y1),
        }

    def _select_spawn_pair(self, easy_zones, hard_zones):
        best_pairs = []
        best_dist = -1.0
        for ez in easy_zones:
            for hz in hard_zones:
                dist = abs(ez["y_center"] - hz["y_center"])
                if dist > best_dist + 1e-6:
                    best_pairs = [(ez, hz)]
                    best_dist = dist
                elif abs(dist - best_dist) <= 1e-6:
                    best_pairs.append((ez, hz))
        zone_a, zone_b = random.choice(best_pairs)
        if zone_a["y_center"] <= zone_b["y_center"]:
            return zone_a, zone_b
        return zone_b, zone_a

    def _easy_corridor_segments(self, y0, y1):
        usable_y0 = y0 + self.blank_length
        usable_y1 = y1 - self.blank_length
        direction = random.choice([-1.0, 1.0])
        c1 = self.spawn_x_center - 1.10 * direction + random.uniform(-0.12, 0.12)
        c2 = self.spawn_x_center + 1.10 * direction + random.uniform(-0.12, 0.12)
        c3 = self.spawn_x_center + random.uniform(-0.18, 0.18)
        y_mid0 = usable_y0 + 2.0
        y_mid1 = usable_y0 + 4.0
        return [
            (usable_y0, y_mid0, max(2.8, min(7.2, c1)), 1.25),
            (y_mid0, y_mid1, max(2.8, min(7.2, c2)), 1.20),
            (y_mid1, usable_y1, max(3.2, min(6.8, c3)), 1.35),
        ]

    def _hard_corridor_segments(self, y0, y1):
        direction = random.choice([-1.0, 1.0])
        c1 = 5.0 - 1.45 * direction + random.uniform(-0.15, 0.15)
        c2 = 5.0 + 1.45 * direction + random.uniform(-0.15, 0.15)
        c3 = 5.0 - 0.55 * direction + random.uniform(-0.10, 0.10)
        return [
            (y0 + 1.0, y0 + 2.9, max(3.0, min(7.0, c1)), 1.08),
            (y0 + 3.1, y0 + 5.0, max(3.0, min(7.0, c2)), 1.00),
            (y0 + 5.2, y1 - 1.0, max(3.0, min(7.0, c3)), 1.08),
        ]

    def _sample_x_outside_corridor(self, x_extent, y_center, y_half, segments, clearance, hug_boundary):
        y_lo = y_center - y_half
        y_hi = y_center + y_half
        reserved = []
        for seg_y0, seg_y1, seg_center, seg_half in segments:
            if y_hi > seg_y0 and y_lo < seg_y1:
                reserved.append((seg_center - seg_half - clearance, seg_center + seg_half + clearance))

        x_margin = 0.30
        x_lo = x_margin + x_extent
        x_hi = self.map_x_max - x_margin - x_extent
        if x_hi <= x_lo:
            return self.spawn_x_center

        if not reserved:
            reserved = [(self.spawn_x_center - 1.2, self.spawn_x_center + 1.2)]

        clamped = []
        for lo, hi in reserved:
            lo = max(x_lo, lo)
            hi = min(x_hi, hi)
            if hi > lo:
                clamped.append((lo, hi))

        if not clamped:
            return random.uniform(x_lo, x_hi)

        clamped.sort(key=lambda interval: interval[0])
        merged = []
        for lo, hi in clamped:
            if not merged or lo > merged[-1][1]:
                merged.append([lo, hi])
            else:
                merged[-1][1] = max(merged[-1][1], hi)

        intervals = []
        cursor = x_lo
        for lo, hi in merged:
            if lo > cursor + 1e-6:
                intervals.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < x_hi - 1e-6:
            intervals.append((cursor, x_hi))

        intervals = [(lo, hi) for lo, hi in intervals if hi - lo > 1e-4]
        if not intervals:
            return self.spawn_x_center

        if not hug_boundary:
            spans = [hi - lo for lo, hi in intervals]
            total = sum(spans)
            pick = random.random() * total
            for (lo, hi), span in zip(intervals, spans):
                if pick <= span:
                    return random.uniform(lo, hi)
                pick -= span
            lo, hi = intervals[-1]
            return random.uniform(lo, hi)

        lo, hi = random.choice(intervals)
        focus = min(0.9, hi - lo)
        if abs(lo - x_lo) <= abs(hi - x_hi):
            return random.uniform(max(lo, hi - focus), hi)
        return random.uniform(lo, min(hi, lo + focus))

    def _sample_vertical_center(self, half_z, prefer_extremes):
        z_margin = 0.10
        lo = half_z + z_margin
        hi = self.map_z_max - half_z - z_margin
        if hi <= lo:
            return self.spawn_z_center
        if not prefer_extremes:
            return random.uniform(lo, hi)
        mode = random.random()
        if mode < 0.33:
            return lo
        if mode < 0.66:
            return hi
        return random.uniform(lo, hi)

    def _sample_y_inside_segments(self, segments, half_y):
        margin = 0.10
        valid = []
        for seg_y0, seg_y1, _, _ in segments:
            lo = seg_y0 + half_y + margin
            hi = seg_y1 - half_y - margin
            if hi > lo:
                valid.append((lo, hi))
        if not valid:
            seg_y0, seg_y1, _, _ = random.choice(segments)
            return 0.5 * (seg_y0 + seg_y1)
        lo, hi = random.choice(valid)
        return random.uniform(lo, hi)

    def _sample_y_uniform_in_usable(self, y0, y1, half_y):
        """在中间6米可用区域内均匀采样 y 坐标，确保障碍物不侵入留白区。"""
        usable_y0 = y0 + self.blank_length
        usable_y1 = y1 - self.blank_length
        margin = 0.20  # 额外安全边距
        lo = usable_y0 + half_y + margin
        hi = usable_y1 - half_y - margin
        if hi <= lo:
            return 0.5 * (usable_y0 + usable_y1)
        return random.uniform(lo, hi)

    def _sample_embedded_center_z(self, half_h):
        # Allow partial embedding into floor/ceiling, but keep at least a small part inside.
        eps_inside = 0.02
        lo = self.ground_z - half_h + eps_inside
        hi = self.ceiling_z + half_h - eps_inside
        if hi <= lo:
            return 0.5 * (self.ground_z + self.ceiling_z)
        return random.uniform(lo, hi)

    def _make_scatter_spec(self, kind):
        if kind == "ball":
            r = self.object_half_height
            return {
                "kind": "ball",
                "r": r,
                "z": self._sample_embedded_center_z(r),
                "extent_x": r,
                "extent_y": r,
                "footprint_r": r,
            }
        if kind == "cyl":
            r = self.cyl_tree_radius
            return {
                "kind": "cyl",
                "r": r,
                "extent_x": r,
                "extent_y": r,
                "footprint_r": r,
            }
        hx = random.uniform(0.26, 0.62)
        hy = random.uniform(0.26, 0.72)
        hz = self.object_half_height
        return {
            "kind": "box",
            "hx": hx,
            "hy": hy,
            "hz": hz,
            "z": self._sample_embedded_center_z(hz),
            "extent_x": hx,
            "extent_y": hy,
            "footprint_r": max(hx, hy),
        }

    def _build_scatter_specs(self, difficulty):
        base_density = 0.55 if difficulty == "easy" else 1.0
        density_scale = self._density_scale_for_difficulty(difficulty)
        density = base_density * density_scale
        base_counts = {
            "ball": 3,
            "cyl": 18,
            "box": 6,
        }
        min_counts = {
            "easy": {"ball": 1, "cyl": 4, "box": 2},
            "hard": {"ball": 2, "cyl": 8, "box": 3},
        }

        # 只生成圆柱主干；球体/立方体通过附着流程生成，避免其独立出现。
        total_base = sum(base_counts.values())
        total_min = sum(min_counts[difficulty].values())
        scaled_min = max(1, int(round(total_min * max(0.30, density_scale))))
        trunk_count = self._scaled_region_count(max(1, int(round(total_base * density))), min_count=scaled_min)

        specs = []
        for _ in range(trunk_count):
            specs.append(self._make_scatter_spec("cyl"))
        random.shuffle(specs)
        return specs

    def _attach_objects_to_cylinders(self, placed, bounds, mode="scatter", difficulty="hard"):
        """在圆柱主干上附着固定数量球体/立方体，确保每张图障碍数一致。"""
        if not placed:
            return placed

        trunks = [obs for obs in placed if obs.get("kind") == "cyl"]
        if not trunks:
            return placed

        if mode == "easy_forest":
            make_spec = self._make_easy_forest_spec
            ball_count = int(round(len(trunks) * 0.45))
            box_count = int(round(len(trunks) * 0.40))
        else:
            make_spec = self._make_scatter_spec
            if difficulty == "easy":
                ball_count = int(round(len(trunks) * 0.35))
                box_count = int(round(len(trunks) * 0.30))
            else:
                ball_count = int(round(len(trunks) * 0.50))
                box_count = int(round(len(trunks) * 0.45))

        ball_count = max(0, min(len(trunks), ball_count))
        box_count = max(0, min(len(trunks), box_count))

        def _clamp_xy(obs):
            x_min = bounds["x_lo"] + obs["extent_x"]
            x_max = bounds["x_hi"] - obs["extent_x"]
            y_min = bounds["y_lo"] + obs["extent_y"]
            y_max = bounds["y_hi"] - obs["extent_y"]
            obs["x"] = min(max(obs["x"], x_min), x_max)
            obs["y"] = min(max(obs["y"], y_min), y_max)

        ball_trunks = random.sample(trunks, ball_count) if ball_count > 0 else []
        box_trunks = random.sample(trunks, box_count) if box_count > 0 else []

        attached = []
        for trunk in ball_trunks:
            jitter = min(0.12, float(trunk["r"]) * 0.6)
            ball = make_spec("ball")
            ball["x"] = trunk["x"] + random.uniform(-jitter, jitter)
            ball["y"] = trunk["y"] + random.uniform(-jitter, jitter)
            _clamp_xy(ball)
            attached.append(ball)

        for trunk in box_trunks:
            jitter = min(0.12, float(trunk["r"]) * 0.6)
            box = make_spec("box")
            box["x"] = trunk["x"] + random.uniform(-jitter, jitter)
            box["y"] = trunk["y"] + random.uniform(-jitter, jitter)
            _clamp_xy(box)
            attached.append(box)

        placed.extend(attached)
        return placed

    def _place_scatter_specs(self, specs, bounds, pair_gap):
        placed = []
        for spec in specs:
            obs = dict(spec)
            if self._try_place_forest_obstacle(obs, placed, bounds, pair_gap, max_tries=56):
                continue
            if self._try_place_forest_obstacle(obs, placed, bounds, pair_gap * 0.75, max_tries=28):
                continue

            # Fallback: still place uniformly to keep per-batch obstacle tensor shape stable.
            x_min = bounds["x_lo"] + obs["extent_x"]
            x_max = bounds["x_hi"] - obs["extent_x"]
            y_min = bounds["y_lo"] + obs["extent_y"]
            y_max = bounds["y_hi"] - obs["extent_y"]
            if x_max <= x_min:
                obs["x"] = 0.5 * (bounds["x_lo"] + bounds["x_hi"])
            else:
                obs["x"] = random.uniform(x_min, x_max)
            if y_max <= y_min:
                obs["y"] = 0.5 * (bounds["y_lo"] + bounds["y_hi"])
            else:
                obs["y"] = random.uniform(y_min, y_max)
            placed.append(obs)
        return placed

    def _obstacle_blocks_nav_slice(self, obstacle, nav_z):
        kind = obstacle["kind"]
        if kind == "cyl":
            return True
        if kind == "ball":
            return abs(obstacle["z"] - nav_z) <= obstacle["r"]
        return abs(obstacle["z"] - nav_z) <= obstacle["hz"]

    def _build_scatter_nav_grid(self, obstacles, y0, y1, resolution=0.35, inflate=0.30):
        x_lo = 0.35
        x_hi = self.map_x_max - 0.35
        y_lo = y0 + self.blank_length
        y_hi = y1 - self.blank_length
        nx = max(12, int(math.ceil((x_hi - x_lo) / resolution)))
        ny = max(12, int(math.ceil((y_hi - y_lo) / resolution)))
        occ = [[False for _ in range(nx)] for _ in range(ny)]
        x_centers = [x_lo + (ix + 0.5) * resolution for ix in range(nx)]
        y_centers = [y_lo + (iy + 0.5) * resolution for iy in range(ny)]

        blockers = [obs for obs in obstacles if self._obstacle_blocks_nav_slice(obs, self.spawn_z_center)]
        for iy, y_c in enumerate(y_centers):
            row = occ[iy]
            for ix, x_c in enumerate(x_centers):
                blocked = False
                for obs in blockers:
                    dx = x_c - obs["x"]
                    dy = y_c - obs["y"]
                    if obs["kind"] == "box":
                        if abs(dx) <= obs["hx"] + inflate and abs(dy) <= obs["hy"] + inflate:
                            blocked = True
                            break
                    else:
                        rr = obs["r"] + inflate
                        if dx * dx + dy * dy <= rr * rr:
                            blocked = True
                            break
                row[ix] = blocked
        return occ

    @staticmethod
    def _blocked_ratio(occ):
        total = len(occ) * len(occ[0]) if occ and occ[0] else 1
        blocked = sum(1 for row in occ for cell in row if cell)
        return blocked / float(total)

    @staticmethod
    def _packed_obstacle_lists(placed):
        balls = []
        cyls = []
        voxels = []
        for obs in placed:
            if obs["kind"] == "ball":
                balls.append([obs["x"], obs["y"], obs["z"], obs["r"]])
            elif obs["kind"] == "cyl":
                cyls.append([obs["x"], obs["y"], obs["r"]])
            else:
                voxels.append([obs["x"], obs["y"], obs["z"], obs["hx"], obs["hy"], obs["hz"]])
        return balls, cyls, voxels

    def _make_easy_forest_spec(self, kind):
        if kind == "ball":
            r = random.uniform(0.18, 0.34)
            return {
                "kind": "ball",
                "r": r,
                "z": self._sample_vertical_center(r, False),
                "extent_x": r,
                "extent_y": r,
                "footprint_r": r,
            }
        if kind == "cyl":
            r = random.uniform(0.16, 0.30)
            return {
                "kind": "cyl",
                "r": r,
                "extent_x": r,
                "extent_y": r,
                "footprint_r": r,
            }
        hx = random.uniform(0.22, 0.42)
        hy = random.uniform(0.22, 0.48)
        hz = random.uniform(0.45, 1.00)
        return {
            "kind": "box",
            "hx": hx,
            "hy": hy,
            "hz": hz,
            "z": self._sample_vertical_center(hz, False),
            "extent_x": hx,
            "extent_y": hy,
            "footprint_r": max(hx, hy),
        }

    def _build_easy_forest_specs(self):
        # easy 区域同样仅生成圆柱主干，附着物后续再挂载。
        trunk_count = self._scaled_region_count(25, min_count=12)
        specs = []
        for _ in range(trunk_count):
            specs.append(self._make_easy_forest_spec("cyl"))
        random.shuffle(specs)
        return specs

    def _try_place_forest_obstacle(self, spec, placed, bounds, pair_gap, max_tries=48, anchor=None, anchor_jitter=0.9):
        x_min = bounds["x_lo"] + spec["extent_x"]
        x_max = bounds["x_hi"] - spec["extent_x"]
        y_min = bounds["y_lo"] + spec["extent_y"]
        y_max = bounds["y_hi"] - spec["extent_y"]
        if x_max <= x_min or y_max <= y_min:
            return False

        best_xy = None
        best_score = None
        for trial_idx in range(max_tries):
            if anchor is not None and trial_idx < int(0.7 * max_tries):
                x = min(max(anchor[0] + random.uniform(-anchor_jitter, anchor_jitter), x_min), x_max)
                y = min(max(anchor[1] + random.uniform(-anchor_jitter, anchor_jitter), y_min), y_max)
            else:
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)

            nearest_margin = 10.0
            valid = True
            for other in placed:
                required = spec["footprint_r"] + other["footprint_r"] + pair_gap
                dist = math.hypot(x - other["x"], y - other["y"])
                if dist < required:
                    valid = False
                    break
                nearest_margin = min(nearest_margin, dist - required)
            if not valid:
                continue

            boundary_margin = min(x - x_min, x_max - x, y - y_min, y_max - y)
            score = nearest_margin + 0.35 * boundary_margin + 0.01 * random.random()
            if best_score is None or score > best_score:
                best_score = score
                best_xy = (x, y)

        if best_xy is None:
            return False

        spec["x"], spec["y"] = best_xy
        placed.append(spec)
        return True

    def _easy_obstacle_blocks_nav_band(self, obstacle):
        z_ref = self.spawn_z_center
        z_slack = 0.75
        kind = obstacle["kind"]
        if kind == "cyl":
            return True
        if kind == "ball":
            return abs(obstacle["z"] - z_ref) <= obstacle["r"] + z_slack
        return abs(obstacle["z"] - z_ref) <= obstacle["hz"] + z_slack

    def _build_easy_nav_grid(self, obstacles, y0, y1, resolution=0.40, inflate=0.42):
        x_lo = 0.35
        x_hi = self.map_x_max - 0.35
        y_lo = y0 + self.blank_length
        y_hi = y1 - self.blank_length
        nx = max(12, int(math.ceil((x_hi - x_lo) / resolution)))
        ny = max(12, int(math.ceil((y_hi - y_lo) / resolution)))
        occ = [[False for _ in range(nx)] for _ in range(ny)]
        x_centers = [x_lo + (ix + 0.5) * resolution for ix in range(nx)]
        y_centers = [y_lo + (iy + 0.5) * resolution for iy in range(ny)]

        blockers = [obs for obs in obstacles if self._easy_obstacle_blocks_nav_band(obs)]
        for iy, y_c in enumerate(y_centers):
            row = occ[iy]
            for ix, x_c in enumerate(x_centers):
                blocked = False
                for obs in blockers:
                    dx = x_c - obs["x"]
                    dy = y_c - obs["y"]
                    if obs["kind"] == "box":
                        if abs(dx) <= obs["hx"] + inflate and abs(dy) <= obs["hy"] + inflate:
                            blocked = True
                            break
                    else:
                        rr = obs["r"] + inflate
                        if dx * dx + dy * dy <= rr * rr:
                            blocked = True
                            break
                row[ix] = blocked
        return occ, x_centers, y_centers, resolution

    def _easy_nav_has_connection(self, occ):
        ny = len(occ)
        nx = len(occ[0]) if ny > 0 else 0
        if nx == 0:
            return True

        queue = []
        visited = [[False for _ in range(nx)] for _ in range(ny)]
        for ix in range(nx):
            if not occ[0][ix]:
                visited[0][ix] = True
                queue.append((0, ix))

        head = 0
        while head < len(queue):
            iy, ix = queue[head]
            head += 1
            if iy == ny - 1:
                return True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nyi = iy + dy
                nxi = ix + dx
                if 0 <= nyi < ny and 0 <= nxi < nx and (not visited[nyi][nxi]) and (not occ[nyi][nxi]):
                    visited[nyi][nxi] = True
                    queue.append((nyi, nxi))
        return False

    def _find_easy_direct_band(self, occ, x_centers, y_centers, resolution):
        ny = len(occ)
        nx = len(occ[0]) if ny > 0 else 0
        if nx == 0:
            return None

        min_run = max(6, int(round(0.82 * ny)))
        min_width = max(3, int(math.ceil(1.6 / max(resolution, 1e-6))))

        col_runs = [None for _ in range(nx)]
        for ix in range(nx):
            best_len = 0
            best_center = 0.5 * (y_centers[0] + y_centers[-1])
            run_start = None
            for iy in range(ny):
                is_open = not occ[iy][ix]
                if is_open and run_start is None:
                    run_start = iy
                if (not is_open or iy == ny - 1) and run_start is not None:
                    run_end = iy if is_open and iy == ny - 1 else iy - 1
                    run_len = run_end - run_start + 1
                    if run_len > best_len:
                        best_len = run_len
                        best_center = 0.5 * (y_centers[run_start] + y_centers[run_end])
                    run_start = None
            if best_len >= min_run:
                col_runs[ix] = (best_len, best_center)

        best_band = None
        band_start = None
        for ix in range(nx + 1):
            active = ix < nx and col_runs[ix] is not None
            if active and band_start is None:
                band_start = ix
            if (not active) and band_start is not None:
                band_end = ix - 1
                band_width = band_end - band_start + 1
                if band_width >= min_width:
                    x_anchor = 0.5 * (x_centers[band_start] + x_centers[band_end])
                    y_anchor = sum(col_runs[j][1] for j in range(band_start, band_end + 1)) / float(band_width)
                    score = band_width * sum(col_runs[j][0] for j in range(band_start, band_end + 1))
                    candidate = (score, x_anchor, y_anchor)
                    if best_band is None or candidate[0] > best_band[0]:
                        best_band = candidate
                band_start = None

        if best_band is None:
            return None
        return best_band[1], best_band[2]

    def _densest_easy_window_center(self, occ, x_centers, y_centers, resolution):
        ny = len(occ)
        nx = len(occ[0]) if ny > 0 else 0
        if nx == 0:
            return self.spawn_x_center, 0.0, 0.0

        win_x = max(3, int(math.ceil(1.6 / max(resolution, 1e-6))))
        win_y = max(3, int(math.ceil(1.4 / max(resolution, 1e-6))))
        best_density = -1.0
        best_center = (self.spawn_x_center, 0.5 * (y_centers[0] + y_centers[-1]), 0.0)

        for iy0 in range(0, max(1, ny - win_y + 1)):
            iy1 = min(ny, iy0 + win_y)
            for ix0 in range(0, max(1, nx - win_x + 1)):
                ix1 = min(nx, ix0 + win_x)
                total = (iy1 - iy0) * (ix1 - ix0)
                blocked = 0
                for iy in range(iy0, iy1):
                    for ix in range(ix0, ix1):
                        blocked += 1 if occ[iy][ix] else 0
                density = blocked / float(max(1, total))
                if density > best_density:
                    x_c = 0.5 * (x_centers[ix0] + x_centers[ix1 - 1])
                    y_c = 0.5 * (y_centers[iy0] + y_centers[iy1 - 1])
                    best_density = density
                    best_center = (x_c, y_c, density)
        return best_center

    def _relax_easy_forest_obstacle(self, placed, target_xy=None, remove=False):
        if not placed:
            return False

        best_idx = None
        best_score = None
        tx, ty = (target_xy if target_xy is not None else (self.spawn_x_center, 0.0))
        for idx, obs in enumerate(placed):
            size = obs["footprint_r"]
            dist = math.hypot(obs["x"] - tx, obs["y"] - ty)
            score = size - 0.20 * dist
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            return False

        obs = placed[best_idx]
        if remove or obs["footprint_r"] < 0.22:
            del placed[best_idx]
            return True

        if obs["kind"] == "box":
            obs["hx"] *= 0.82
            obs["hy"] *= 0.82
            obs["hz"] *= 0.90
            obs["extent_x"] = obs["hx"]
            obs["extent_y"] = obs["hy"]
            obs["footprint_r"] = max(obs["hx"], obs["hy"])
            obs["z"] = min(max(obs["z"], obs["hz"] + 0.10), self.map_z_max - obs["hz"] - 0.10)
        else:
            obs["r"] *= 0.84
            obs["extent_x"] = obs["r"]
            obs["extent_y"] = obs["r"]
            obs["footprint_r"] = obs["r"]
            if obs["kind"] == "ball":
                obs["z"] = min(max(obs["z"], obs["r"] + 0.10), self.map_z_max - obs["r"] - 0.10)
        return True

    def _make_easy_band_breaker(self):
        return self._make_easy_forest_spec("cyl")

    def _generate_easy_forest_region(self, y0, y1):
        bounds = {
            "x_lo": 0.45,
            "x_hi": self.map_x_max - 0.45,
            "y_lo": y0 + self.blank_length + 0.25,
            "y_hi": y1 - self.blank_length - 0.25,
        }
        pair_gap = 0.78
        placed = []

        for spec in self._build_easy_forest_specs():
            if self._try_place_forest_obstacle(spec, placed, bounds, pair_gap, max_tries=56):
                continue
            self._try_place_forest_obstacle(spec, placed, bounds, pair_gap * 0.88, max_tries=32)

        placed = self._attach_objects_to_cylinders(placed, bounds, mode="easy_forest")

        for _ in range(6):
            occ, x_centers, y_centers, res = self._build_easy_nav_grid(placed, y0, y1)
            total_cells = len(occ) * len(occ[0]) if occ and occ[0] else 1
            blocked_cells = sum(1 for row in occ for cell in row if cell)
            blocked_ratio = blocked_cells / float(total_cells)
            connected = self._easy_nav_has_connection(occ)
            band_anchor = self._find_easy_direct_band(occ, x_centers, y_centers, res)
            dense_x, dense_y, dense_ratio = self._densest_easy_window_center(occ, x_centers, y_centers, res)

            if band_anchor is not None and blocked_ratio < 0.32:
                breaker = self._make_easy_band_breaker()
                if self._try_place_forest_obstacle(
                    breaker, placed, bounds, pair_gap * 0.82, max_tries=40, anchor=band_anchor, anchor_jitter=0.65
                ):
                    continue

            if not connected:
                if self._relax_easy_forest_obstacle(placed, target_xy=(dense_x, dense_y), remove=True):
                    continue

            if blocked_ratio > 0.34 or dense_ratio > 0.58:
                if self._relax_easy_forest_obstacle(placed, target_xy=(dense_x, dense_y), remove=False):
                    continue

            break

        balls = []
        cyls = []
        voxels = []
        for obs in placed:
            if obs["kind"] == "ball":
                balls.append([obs["x"], obs["y"], obs["z"], obs["r"]])
            elif obs["kind"] == "cyl":
                cyls.append([obs["x"], obs["y"], obs["r"]])
            else:
                voxels.append([obs["x"], obs["y"], obs["z"], obs["hx"], obs["hy"], obs["hz"]])
        return balls, cyls, voxels

    def _generate_random_region(self, difficulty, y0, y1):
        if difficulty not in ("easy", "hard"):
            return [], [], []

        bounds = {
            "x_lo": 0.45,
            "x_hi": self.map_x_max - 0.45,
            "y_lo": y0 + self.blank_length + 0.20,
            "y_hi": y1 - self.blank_length - 0.20,
        }
        pair_gap = 0.24
        nav_inflate = 0.5 * self.two_drone_passage_width
        # easy/hard only differ in density target.
        density_scale = self._density_scale_for_difficulty(difficulty)
        blocked_lo = (0.06 if difficulty == "easy" else 0.18) * density_scale
        blocked_hi = (0.22 if difficulty == "easy" else 0.38) * density_scale
        blocked_lo = max(0.02, min(0.80, blocked_lo))
        blocked_hi = max(blocked_lo + 0.04, min(0.92, blocked_hi))

        best_placed = None
        best_score = -1e9
        for _ in range(18):
            specs = self._build_scatter_specs(difficulty)
            placed = self._place_scatter_specs(specs, bounds, pair_gap)
            placed = self._attach_objects_to_cylinders(placed, bounds, mode="scatter", difficulty=difficulty)
            occ = self._build_scatter_nav_grid(placed, y0, y1, resolution=0.35, inflate=nav_inflate)
            connected = self._easy_nav_has_connection(occ)
            blocked_ratio = self._blocked_ratio(occ)

            penalty_dense = max(0.0, blocked_ratio - blocked_hi)
            penalty_sparse = max(0.0, blocked_lo - blocked_ratio)
            score = (1.0 if connected else 0.0) - 2.0 * penalty_dense - 1.2 * penalty_sparse
            if score > best_score:
                best_score = score
                best_placed = placed

            if connected and blocked_lo <= blocked_ratio <= blocked_hi:
                return self._packed_obstacle_lists(placed)

        if best_placed is None:
            return [], [], []
        return self._packed_obstacle_lists(best_placed)

    def _append_wall_box(self, walls, x_center, y_center, hx, hy, hz=None):
        if hx <= 1e-4 or hy <= 1e-4:
            return
        walls.append([
            x_center,
            y_center,
            self.spawn_z_center,
            hx,
            hy,
            self.inner_wall_hz if hz is None else hz,
        ])

    def _append_horizontal_wall(self, walls, x0, x1, y_center, half_thickness):
        x_lo = min(x0, x1)
        x_hi = max(x0, x1)
        self._append_wall_box(walls, 0.5 * (x_lo + x_hi), y_center, 0.5 * (x_hi - x_lo), half_thickness)

    def _append_vertical_wall(self, walls, x_center, y0, y1, half_thickness):
        y_lo = min(y0, y1)
        y_hi = max(y0, y1)
        self._append_wall_box(walls, x_center, 0.5 * (y_lo + y_hi), half_thickness, 0.5 * (y_hi - y_lo))

    def _append_stepped_wall(self, walls, x0, y0, x1, y1, half_thickness, steps=8):
        """构造斜墙离散段，沿 x/y 双向加密封重叠，避免穿缝。"""
        total_steps = max(2, int(steps))
        seg_dx = (x1 - x0) / total_steps
        seg_dy = (y1 - y0) / total_steps

        # y 向重叠用于封住段间上下缝，x 向重叠用于封住斜率造成的横向锯齿缝。
        overlap_y = 0.08
        overlap_x = 0.04
        hx = half_thickness + 0.5 * abs(seg_dx) + overlap_x
        hy = 0.5 * abs(seg_dy) + overlap_y

        for idx in range(total_steps):
            seg_y0 = y0 + seg_dy * idx
            seg_y1 = y0 + seg_dy * (idx + 1)
            y_center = 0.5 * (seg_y0 + seg_y1)
            t = (idx + 0.5) / total_steps
            x_center = x0 + (x1 - x0) * t
            self._append_wall_box(walls, x_center, y_center, hx, hy)

        # 端点封口，防止与相邻竖墙/边界拼接处出现小孔。
        cap_hx = half_thickness + 0.5 * abs(seg_dx) + overlap_x
        cap_hy = half_thickness + 0.04
        self._append_wall_box(walls, x0, y0, cap_hx, cap_hy)
        self._append_wall_box(walls, x1, y1, cap_hx, cap_hy)

    def _generate_u_region(self, y0, y1):
        """生成 U 型局部最优陷阱区域。"""
        walls = []
        wall_half = 0.14
        corridor_half_width = 1.0  # 通道半宽，总宽 2 米
        corridor_x_left = self.spawn_x_center - corridor_half_width   # 4.0
        corridor_x_right = self.spawn_x_center + corridor_half_width  # 6.0
        outer_left_x = wall_half
        outer_right_x = self.map_x_max - wall_half

        # 漏斗区域 y 范围
        funnel_y0 = y0 + 0.04
        funnel_y1 = y0 + 2.35
        corridor_y0 = funnel_y1
        corridor_y1 = y0 + 6.45
        deadend_y = corridor_y1 + wall_half

        # 出口参数：2 米宽出口
        exit_gap = 2.0
        exit_center_y = corridor_y0 + 2.0
        exit_y0 = exit_center_y - 0.5 * exit_gap
        exit_y1 = exit_center_y + 0.5 * exit_gap

        # 漏斗入口斜墙（无间隙）
        self._append_stepped_wall(walls, outer_left_x, funnel_y0, corridor_x_left, funnel_y1, wall_half, steps=12)
        self._append_stepped_wall(walls, outer_right_x, funnel_y0, corridor_x_right, funnel_y1, wall_half, steps=12)

        open_left = random.random() < 0.5
        if open_left:
            # 左墙有出口：分成两段
            self._append_vertical_wall(walls, corridor_x_left, corridor_y0, exit_y0, wall_half)
            self._append_vertical_wall(walls, corridor_x_left, exit_y1, corridor_y1, wall_half)
            # 右墙完整
            self._append_vertical_wall(walls, corridor_x_right, corridor_y0, corridor_y1, wall_half)

            # 出口外侧引导墙：保留 2 米间距
            guide_x = corridor_x_left - 2.0 - wall_half  # 距离出口 2 米
            self._append_vertical_wall(walls, guide_x, exit_y0, exit_y1 + 1.0, wall_half)
        else:
            # 左墙完整
            self._append_vertical_wall(walls, corridor_x_left, corridor_y0, corridor_y1, wall_half)
            # 右墙有出口：分成两段
            self._append_vertical_wall(walls, corridor_x_right, corridor_y0, exit_y0, wall_half)
            self._append_vertical_wall(walls, corridor_x_right, exit_y1, corridor_y1, wall_half)

            # 出口外侧引导墙：保留 2 米间距
            guide_x = corridor_x_right + 2.0 + wall_half  # 距离出口 2 米
            self._append_vertical_wall(walls, guide_x, exit_y0, exit_y1 + 1.0, wall_half)

        # 死胡同尽头的墙
        self._append_horizontal_wall(walls, corridor_x_left - 0.12, corridor_x_right + 0.12, deadend_y, wall_half)

        return walls, {
            "open_left": open_left,
            "exit_side": "left" if open_left else "right",
            "exit_y": exit_center_y,
            "exit_span": [exit_y0, exit_y1],
            "corridor_span": [corridor_y0, corridor_y1],
        }

    def _expand_obs_to_batch(self, arr, cols, B, device):
        t = torch.as_tensor(arr, device=device, dtype=torch.float32)
        if t.numel() == 0:
            t = torch.zeros((0, cols), device=device, dtype=torch.float32)
        else:
            t = t.reshape(-1, cols)
        t = t.unsqueeze(0)
        if B > 1:
            t = t.repeat(B, 1, 1)
        return t

    def _reset_unified_four_map(self):
        B = self.batch_size
        device = self.device
        map_type = self._pick_unified_map_type()
        self.current_map_type = str(map_type)
        builder = self._get_unified_builder()
        geom = builder(self, map_type)

        self.balls = self._expand_obs_to_batch(geom.get("balls", []), 4, B, device)
        self.cyl = self._expand_obs_to_batch(geom.get("cyl", []), 3, B, device)
        self.voxels = self._expand_obs_to_batch(geom.get("voxels", []), 6, B, device)
        self.voxels = self._strip_side_walls_and_ceiling(
            self.voxels,
            y_min=float(geom.get("map_y_min", self.map_y_min)),
            y_max=float(geom.get("map_y_max", self.map_y_max)),
        )
        self.voxels = self._ensure_top_ceiling_if_needed(self.voxels)
        self.cyl_h = self._expand_obs_to_batch(geom.get("cyl_h", []), 3, B, device)

        spawn_start = tuple(float(v) for v in geom.get("spawn_start", (self.spawn_x_center, self.map_y_min + 0.5, self.spawn_z_center)))
        spawn_goal = tuple(float(v) for v in geom.get("spawn_goal", (self.spawn_x_center, self.map_y_max - 0.5, self.spawn_z_center)))
        self.spawn_start_y = float(spawn_start[1])
        self.spawn_goal_y = float(spawn_goal[1])
        self.spawn_start_x = float(spawn_start[0])
        self.spawn_goal_x = float(spawn_goal[0])
        self.spawn_start_z = float(spawn_start[2])
        self.spawn_goal_z = float(spawn_goal[2])
        self.spawn_start_x_half_span = float(geom.get("spawn_start_x_half_span", self.fixed_spawn_half_span))
        self.spawn_goal_x_half_span = float(geom.get("spawn_goal_x_half_span", self.fixed_spawn_half_span))
        self.spawn_start_z_half_span = float(geom.get("spawn_start_z_half_span", self.fixed_spawn_half_span))
        self.spawn_goal_z_half_span = float(geom.get("spawn_goal_z_half_span", self.fixed_spawn_half_span))

        start_bounds = torch.tensor(
            [[self.spawn_start_y - 0.05, self.spawn_start_y + 0.05]],
            device=device,
            dtype=torch.float32,
        )
        goal_bounds = torch.tensor(
            [[self.spawn_goal_y - 0.05, self.spawn_goal_y + 0.05]],
            device=device,
            dtype=torch.float32,
        )
        self._spawn_start_bounds = start_bounds.repeat(B, 1)
        self._spawn_goal_bounds = goal_bounds.repeat(B, 1)

        region_order = tuple(geom.get("region_order", (map_type,)))
        self.region_order = [region_order for _ in range(B)]
        u_meta = geom.get("u_meta", {"map_type": map_type})
        self.u_meta = [u_meta for _ in range(B)]
        self.current_map_idx = -1
        self._maze_rotation = None

    def reset(self):
        B = self.batch_size
        device = self.device

        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180.0
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack(
            [
                torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
                zeros, ones, zeros,
                torch.sin(cam_angle), zeros, torch.cos(cam_angle),
            ],
            -1,
        ).reshape(B, 3, 3)

        self.maze_cols = int(self.map_x_max)
        self.maze_rows = int(self.region_length * len(self.region_types))
        self.maze_cell_size = 1.0

        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.n_drones_per_group = 1 if self.single else random.choice([4, 8])
        self.drone_radius = 0.13
        self.max_speed = float(min(5.0 * self.speed_mtp, self.max_speed_ceiling))
        self._obstacle_scale = torch.ones((B, 1), device=device)

        if self.unified_four_maps:
            self._reset_unified_four_map()
            self._reset_drone_state(self._obstacle_scale)
            if self.random_rotation:
                self._maze_rotation = None
            return

        balls_batch = []
        cyl_batch = []
        voxel_batch = []
        start_bounds = []
        goal_bounds = []
        region_orders = []
        u_meta_batch = []

        boundary_voxels = self._build_boundary_voxels()

        for _ in range(B):
            order = list(self.region_types)
            random.shuffle(order)
            region_orders.append(tuple(order))

            balls = []
            cyls = []
            voxels = list(boundary_voxels)
            easy_zones = []
            hard_zones = []
            u_meta = None

            for slot_idx, region_type in enumerate(order):
                region_y0 = self.map_y_min + slot_idx * self.region_length
                region_y1 = region_y0 + self.region_length

                if region_type == "easy":
                    region_balls, region_cyls, region_voxels = self._generate_random_region("easy", region_y0, region_y1)
                    easy_zones = [
                        self._make_blank_zone(region_y0, region_y0 + self.blank_length),
                        self._make_blank_zone(region_y1 - self.blank_length, region_y1),
                    ]
                elif region_type == "hard":
                    region_balls, region_cyls, region_voxels = self._generate_random_region("hard", region_y0, region_y1)
                    hard_zones = [
                        self._make_blank_zone(region_y0, region_y0 + self.blank_length),
                        self._make_blank_zone(region_y1 - self.blank_length, region_y1),
                    ]
                else:
                    region_balls = []
                    region_cyls = []
                    region_voxels, u_meta = self._generate_u_region(region_y0, region_y1)

                balls.extend(region_balls)
                cyls.extend(region_cyls)
                voxels.extend(region_voxels)

            start_zone, goal_zone = self._select_spawn_pair(easy_zones, hard_zones)
            start_bounds.append([start_zone["y_lo"], start_zone["y_hi"]])
            goal_bounds.append([goal_zone["y_lo"], goal_zone["y_hi"]])

            balls_batch.append(balls)
            cyl_batch.append(cyls)
            voxel_batch.append(voxels)
            u_meta_batch.append(u_meta or {"open_left": None, "exit_side": "unknown"})

        self.region_order = region_orders
        self.current_map_type = "mixed-random"
        self.balls = torch.tensor(balls_batch, device=device, dtype=torch.float32)
        self.cyl = torch.tensor(cyl_batch, device=device, dtype=torch.float32)
        self.voxels = torch.tensor(voxel_batch, device=device, dtype=torch.float32)
        self.voxels = self._strip_side_walls_and_ceiling(self.voxels)
        self.voxels = self._ensure_top_ceiling_if_needed(self.voxels)
        self.cyl_h = torch.zeros((B, 0, 3), device=device, dtype=torch.float32)
        self._spawn_start_bounds = torch.tensor(start_bounds, device=device, dtype=torch.float32)
        self._spawn_goal_bounds = torch.tensor(goal_bounds, device=device, dtype=torch.float32)
        self.u_meta = u_meta_batch

        # Randomly generated maps use default x/z spawn planes around the scene center.
        self.spawn_start_x = self.spawn_x_center
        self.spawn_goal_x = self.spawn_x_center
        self.spawn_start_z = self.spawn_z_center
        self.spawn_goal_z = self.spawn_z_center
        self.spawn_start_x_half_span = self.fixed_spawn_half_span
        self.spawn_goal_x_half_span = self.fixed_spawn_half_span
        self.spawn_start_z_half_span = self.fixed_spawn_half_span
        self.spawn_goal_z_half_span = self.fixed_spawn_half_span

        self._maze_rotation = None
        self._reset_drone_state(self._obstacle_scale)

        if self.random_rotation:
            # Keep start/goal constrained to the fixed y=-11 / y=11 planes.
            self._maze_rotation = None

    def set_precomputed_maps(self, map_list):
        self.precomputed_maps = list(map_list) if map_list is not None else []

    def load_precomputed_map(self, map_data):
        """Load obstacle/layout tensors from cached map data without regenerating geometry."""
        B = self.batch_size
        device = self.device

        map_type_raw = map_data.get("map_type", None)
        if not map_type_raw:
            map_type_raw = map_data.get("u_meta", {}).get("map_type", None) if isinstance(map_data.get("u_meta", None), dict) else None
        if not map_type_raw:
            map_type_raw = map_data.get("map_meta", {}).get("map_type", None) if isinstance(map_data.get("map_meta", None), dict) else None
        self.current_map_type = str(map_type_raw or "").strip().lower().replace("_", "-")

        def _to_device_tensor(key, fallback_shape):
            val = map_data.get(key, None)
            if val is None:
                return torch.zeros(fallback_shape, device=device, dtype=torch.float32)
            if isinstance(val, torch.Tensor):
                t = val.detach().to(device=device, dtype=torch.float32)
            else:
                t = torch.as_tensor(val, device=device, dtype=torch.float32)
            if t.dim() == len(fallback_shape) - 1:
                t = t.unsqueeze(0)
            if t.shape[0] == 1 and B > 1:
                t = t.repeat(B, *([1] * (t.dim() - 1)))
            return t

        self.balls = _to_device_tensor("balls", (B, 0, 4))
        self.cyl = _to_device_tensor("cyl", (B, 0, 3))
        self.voxels = _to_device_tensor("voxels", (B, 0, 6))
        self.voxels = self._strip_side_walls_and_ceiling(
            self.voxels,
            y_min=float(map_data.get("map_y_min", self.map_y_min)),
            y_max=float(map_data.get("map_y_max", self.map_y_max)),
        )
        self.voxels = self._ensure_top_ceiling_if_needed(self.voxels)
        self.cyl_h = _to_device_tensor("cyl_h", (B, 0, 3))

        start_bounds = map_data.get("spawn_start_bounds", torch.tensor([self.map_y_min, self.map_y_min + self.blank_length]))
        goal_bounds = map_data.get("spawn_goal_bounds", torch.tensor([self.map_y_max - self.blank_length, self.map_y_max]))

        start_bounds = torch.as_tensor(start_bounds, device=device, dtype=torch.float32).view(1, 2).repeat(B, 1)
        goal_bounds = torch.as_tensor(goal_bounds, device=device, dtype=torch.float32).view(1, 2).repeat(B, 1)
        self._spawn_start_bounds = start_bounds
        self._spawn_goal_bounds = goal_bounds

        region_order = map_data.get("region_order", tuple(self.region_types))
        self.region_order = [tuple(region_order) for _ in range(B)]
        u_meta = map_data.get("u_meta", {"open_left": None, "exit_side": "unknown"})
        self.u_meta = [u_meta for _ in range(B)]

        if "spawn_start_y" in map_data:
            self.spawn_start_y = float(map_data["spawn_start_y"])
        if "spawn_goal_y" in map_data:
            self.spawn_goal_y = float(map_data["spawn_goal_y"])

        self.spawn_start_x = float(map_data.get("spawn_start_x", self.spawn_x_center))
        self.spawn_goal_x = float(map_data.get("spawn_goal_x", self.spawn_x_center))
        self.spawn_start_z = float(map_data.get("spawn_start_z", self.spawn_z_center))
        self.spawn_goal_z = float(map_data.get("spawn_goal_z", self.spawn_z_center))
        self.spawn_start_x_half_span = float(map_data.get("spawn_start_x_half_span", self.fixed_spawn_half_span))
        self.spawn_goal_x_half_span = float(map_data.get("spawn_goal_x_half_span", self.fixed_spawn_half_span))
        self.spawn_start_z_half_span = float(map_data.get("spawn_start_z_half_span", self.fixed_spawn_half_span))
        self.spawn_goal_z_half_span = float(map_data.get("spawn_goal_z_half_span", self.fixed_spawn_half_span))

        self._obstacle_scale = torch.ones((B, 1), device=device)
        self.scene_x_half = self.map_x_max
        self.scene_y_half = self.map_y_half
        self._maze_rotation = None

    def reset_from_precomputed_map(self, map_data):
        """Load cached map geometry and then reset only drone state."""
        self.load_precomputed_map(map_data)
        self.reset_drone_only()

    def set_map_by_index(self, map_idx):
        if len(self.precomputed_maps) == 0:
            raise ValueError("No precomputed maps registered in env.precomputed_maps")
        self.current_map_idx = int(map_idx) % len(self.precomputed_maps)
        self.reset_from_precomputed_map(self.precomputed_maps[self.current_map_idx])

    def reset_drone_only(self):
        B = self.batch_size
        device = self.device

        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180.0
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack(
            [
                torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
                zeros, ones, zeros,
                torch.sin(cam_angle), zeros, torch.cos(cam_angle),
            ],
            -1,
        ).reshape(B, 3, 3)

        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.drone_radius = 0.13
        self.max_speed = float(min(5.0 * self.speed_mtp, self.max_speed_ceiling))
        self._reset_drone_state(getattr(self, "_obstacle_scale", None))


    def _reset_drone_state(self, obstacle_scale):
        B = self.batch_size
        device = self.device

        del obstacle_scale

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        start_x_center = float(getattr(self, "spawn_start_x", self.spawn_x_center))
        goal_x_center = float(getattr(self, "spawn_goal_x", self.spawn_x_center))
        start_z_center = float(getattr(self, "spawn_start_z", self.spawn_z_center))
        goal_z_center = float(getattr(self, "spawn_goal_z", self.spawn_z_center))
        start_x_half = max(0.0, float(getattr(self, "spawn_start_x_half_span", self.fixed_spawn_half_span)))
        goal_x_half = max(0.0, float(getattr(self, "spawn_goal_x_half_span", self.fixed_spawn_half_span)))
        start_z_half = max(0.0, float(getattr(self, "spawn_start_z_half_span", self.fixed_spawn_half_span)))
        goal_z_half = max(0.0, float(getattr(self, "spawn_goal_z_half_span", self.fixed_spawn_half_span)))

        x = start_x_center + (torch.rand(B, device=device) * 2.0 - 1.0) * start_x_half
        z = start_z_center + (torch.rand(B, device=device) * 2.0 - 1.0) * start_z_half
        x_goal = goal_x_center + (torch.rand(B, device=device) * 2.0 - 1.0) * goal_x_half
        z_goal = goal_z_center + (torch.rand(B, device=device) * 2.0 - 1.0) * goal_z_half

        y = torch.full((B,), float(self.spawn_start_y), device=device)
        y_goal = torch.full((B,), float(self.spawn_goal_y), device=device)

        self.p = torch.stack([x, y, z], dim=-1)
        self.p_target = torch.stack([x_goal, y_goal, z_goal], dim=-1)

        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)
        self.yaw_rate = torch.zeros((B, 1), device=device)
        self.v = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2

        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(
            R,
            self.act,
            safe_normalize(torch.randn((B, 3), device=device) * torch.tensor([1.0, 1.0, 0.0], device=device)),
            torch.zeros_like(self.yaw_ctl_delay),
            2,
        )
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.full((B,), 0.07, device=device)
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)

    def run(
        self,
        act_pred,
        ctl_dt=1 / 15,
        v_pred=None,
        heading_ref=None,
        yaw_rate_cmd=None,
        yaw_rate_max=None,
    ):
        act_pred = torch.nan_to_num(act_pred, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        if v_pred is not None:
            v_pred = torch.nan_to_num(
                v_pred,
                nan=0.0,
                posinf=self.hard_vpred_clip,
                neginf=-self.hard_vpred_clip,
            ).clamp(-self.hard_vpred_clip, self.hard_vpred_clip)
        use_explicit_yaw = heading_ref is not None or yaw_rate_cmd is not None
        if use_explicit_yaw:
            if heading_ref is None:
                heading_ref = v_pred if v_pred is not None else self.R[:, :, 0].detach()
            heading_ref = torch.nan_to_num(
                heading_ref,
                nan=0.0,
                posinf=self.hard_vpred_clip,
                neginf=-self.hard_vpred_clip,
            ).clamp(-self.hard_vpred_clip, self.hard_vpred_clip)
            if yaw_rate_cmd is not None:
                max_yaw = math.radians(150.0) if yaw_rate_max is None else float(yaw_rate_max)
                yaw_rate_cmd = torch.nan_to_num(
                    yaw_rate_cmd,
                    nan=0.0,
                    posinf=max_yaw,
                    neginf=-max_yaw,
                ).clamp(-max_yaw, max_yaw)

        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        dyn_fn = differentiable_run_torch if self.use_meta_differentiable_dynamics else differentiable_run
        self.act, p_free, v_free, a_free = dyn_fn(
            self.R,
            self.dg,
            self.z_drag_coef,
            self.drag_2,
            self.pitch_ctl_delay,
            act_pred,
            self.act,
            self.p,
            self.v,
            self.v_wind,
            self.a,
            self.grad_decay,
            ctl_dt,
            0.5,
        )
        self.act = torch.nan_to_num(self.act, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        p_free = torch.nan_to_num(p_free, nan=0.0, posinf=100.0, neginf=-100.0)
        v_free = torch.nan_to_num(
            v_free,
            nan=0.0,
            posinf=self.hard_speed_clip,
            neginf=-self.hard_speed_clip,
        ).clamp(-self.hard_speed_clip, self.hard_speed_clip)
        a_free = torch.nan_to_num(a_free, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        if self.wall_physical_feedback:
            self.p, self.v, self.a = self._apply_soft_contacts(self.p_old, p_free, v_free, a_free, ctl_dt)
        else:
            self.p, self.v, self.a = p_free, v_free, a_free
        self.p, self.v, self.a = self._apply_speed_limit(self.p_old, self.p, self.v, self.a, ctl_dt)
        self.p = torch.nan_to_num(self.p, nan=0.0, posinf=100.0, neginf=-100.0)
        self.v = torch.nan_to_num(
            self.v,
            nan=0.0,
            posinf=self.hard_speed_clip,
            neginf=-self.hard_speed_clip,
        ).clamp(-self.hard_speed_clip, self.hard_speed_clip)
        self.a = torch.nan_to_num(self.a, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)

        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        if use_explicit_yaw:
            if not hasattr(self, "yaw_rate"):
                self.yaw_rate = torch.zeros((self.batch_size, 1), device=self.device)
            max_yaw = math.radians(150.0) if yaw_rate_max is None else float(yaw_rate_max)
            yaw_rate_cmd_arg = torch.zeros_like(self.yaw_rate) if yaw_rate_cmd is None else yaw_rate_cmd
            use_torch_attitude = (
                self.use_meta_differentiable_dynamics
                or yaw_rate_cmd_arg.requires_grad
                or not hasattr(quadsim_cuda, "update_state_vec_v2")
            )
            if use_torch_attitude:
                self.R, self.yaw_rate = update_state_vec_torch_v2(
                    self.R,
                    self.act,
                    heading_ref,
                    alpha,
                    self.yaw_rate,
                    yaw_rate_cmd=yaw_rate_cmd_arg,
                    ctl_dt=ctl_dt,
                    yaw_rate_max=max_yaw,
                )
            else:
                self.R, self.yaw_rate = quadsim_cuda.update_state_vec_v2(
                    self.R,
                    self.act,
                    heading_ref,
                    self.yaw_rate,
                    yaw_rate_cmd_arg,
                    alpha,
                    float(ctl_dt),
                    max_yaw,
                )
        elif self.use_meta_differentiable_dynamics:
            self.R = update_state_vec_torch(self.R, self.act, v_pred, alpha, 2)
        else:
            self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 2)
        self.R = torch.nan_to_num(self.R, nan=0.0, posinf=1.0, neginf=-1.0)
