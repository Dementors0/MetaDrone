import math
import random

import torch
import quadsim_cuda

from env import Env as BaseEnv, run as differentiable_run, run_torch as differentiable_run_torch, safe_normalize, update_state_vec_torch


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
        speed_limit_softness=0.05,
        max_speed_ceiling=5.0,
        hard_vpred_clip=20.0,
        hard_speed_clip=30.0,
        start_goal_plane_y_abs=50.0,
    ):
        self.map_x_max = 10.0
        self.map_y_half = 12.0
        self.map_y_min = -self.map_y_half
        self.map_y_max = self.map_y_half
        self.map_z_max = 5.0
        self.region_length = 8.0
        self.blank_length = 1.0
        self.spawn_x_center = 5.0
        self.spawn_z_center = 2.5
        self.spawn_x_half_span = 2.0
        self.spawn_z_half_span = 2.0
        self.fixed_spawn_half_span = 1.0
        self.boundary_thickness = 0.10
        self.boundary_half = 0.5 * self.boundary_thickness
        self.full_wall_hz = 2.45
        self.inner_wall_hz = 2.30
        self.region_types = ("easy", "hard", "u-minimal")

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
        )

        self.scene_x_half = self.map_x_max
        self.scene_y_half = self.map_y_half

    def _scaled_region_count(self, base_count, min_count=0):
        return max(min_count, int(round(base_count * self.obstacle_count_scale)))

    def _build_boundary_voxels(self):
        return [
            [0.0, 0.0, self.spawn_z_center, self.boundary_half, self.map_y_half, self.spawn_z_center],
            [self.map_x_max, 0.0, self.spawn_z_center, self.boundary_half, self.map_y_half, self.spawn_z_center],
            [self.spawn_x_center, self.map_y_min, self.spawn_z_center, self.spawn_x_center, self.boundary_half, self.spawn_z_center],
            [self.spawn_x_center, self.map_y_max, self.spawn_z_center, self.spawn_x_center, self.boundary_half, self.spawn_z_center],
            [self.spawn_x_center, 0.0, 0.0, self.spawn_x_center, self.map_y_half, self.boundary_half],
            [self.spawn_x_center, 0.0, self.map_z_max, self.spawn_x_center, self.map_y_half, self.boundary_half],
        ]

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

    def _generate_random_region(self, difficulty, y0, y1):
        if difficulty == "easy":
            segments = self._easy_corridor_segments(y0, y1)
            # 增加 easy 区域的障碍物数量
            ball_count = self._scaled_region_count(8, min_count=4)
            cyl_count = self._scaled_region_count(8, min_count=4)
            box_count = self._scaled_region_count(12, min_count=6)
            cfg = {
                "ball_r": (0.35, 0.60),
                "cyl_r": (0.24, 0.45),
                "box_hx": (0.35, 0.70),
                "box_hy": (0.30, 0.80),
                "box_hz": (0.40, 1.10),
                "clearance": 0.28,
                "hug_boundary": False,
                "prefer_extremes": False,
                "full_height_prob": 0.0,
            }
        else:
            segments = self._hard_corridor_segments(y0, y1)
            ball_count = self._scaled_region_count(6, min_count=3)
            cyl_count = self._scaled_region_count(6, min_count=3)
            box_count = self._scaled_region_count(10, min_count=6)
            cfg = {
                "ball_r": (0.45, 0.90),
                "cyl_r": (0.35, 0.70),
                "box_hx": (0.45, 1.20),
                "box_hy": (0.45, 1.45),
                "box_hz": (0.60, 1.80),
                "clearance": 0.22,
                "hug_boundary": True,
                "prefer_extremes": True,
                "full_height_prob": 0.55,
            }

        balls = []
        cyls = []
        voxels = []

        for _ in range(ball_count):
            radius = random.uniform(*cfg["ball_r"])
            half_y = radius
            # 使用新的均匀采样方法，确保在中间6米区域内且不侵入留白区
            y_center = self._sample_y_uniform_in_usable(y0, y1, half_y)
            x_center = self._sample_x_outside_corridor(
                radius, y_center, half_y, segments, cfg["clearance"], cfg["hug_boundary"]
            )
            z_center = self._sample_vertical_center(radius, cfg["prefer_extremes"])
            balls.append([x_center, y_center, z_center, radius])

        for _ in range(cyl_count):
            radius = random.uniform(*cfg["cyl_r"])
            half_y = max(radius, random.uniform(0.20, 0.55))
            y_center = self._sample_y_uniform_in_usable(y0, y1, half_y)
            x_center = self._sample_x_outside_corridor(
                radius, y_center, half_y, segments, cfg["clearance"], cfg["hug_boundary"]
            )
            cyls.append([x_center, y_center, radius])

        for _ in range(box_count):
            hx = random.uniform(*cfg["box_hx"])
            hy = random.uniform(*cfg["box_hy"])
            if random.random() < cfg["full_height_prob"]:
                hz = self.inner_wall_hz
                cz = self.spawn_z_center
            else:
                hz = random.uniform(*cfg["box_hz"])
                cz = self._sample_vertical_center(hz, cfg["prefer_extremes"])
            y_center = self._sample_y_uniform_in_usable(y0, y1, hy)
            x_center = self._sample_x_outside_corridor(
                hx, y_center, hy, segments, cfg["clearance"], cfg["hug_boundary"]
            )
            voxels.append([x_center, y_center, cz, hx, hy, hz])

        return balls, cyls, voxels

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
        self.maze_rows = int(self.region_length * 3)
        self.maze_cell_size = 1.0

        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.n_drones_per_group = 1 if self.single else random.choice([4, 8])
        self.drone_radius = random.uniform(0.10, 0.15)
        self.max_speed = float(min(5.0 * self.speed_mtp, self.max_speed_ceiling))
        self._obstacle_scale = torch.ones((B, 1), device=device)

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
        self.balls = torch.tensor(balls_batch, device=device, dtype=torch.float32)
        self.cyl = torch.tensor(cyl_batch, device=device, dtype=torch.float32)
        self.voxels = torch.tensor(voxel_batch, device=device, dtype=torch.float32)
        self.cyl_h = torch.zeros((B, 0, 3), device=device, dtype=torch.float32)
        self._spawn_start_bounds = torch.tensor(start_bounds, device=device, dtype=torch.float32)
        self._spawn_goal_bounds = torch.tensor(goal_bounds, device=device, dtype=torch.float32)
        self.u_meta = u_meta_batch

        self._maze_rotation = None
        self._reset_drone_state(self._obstacle_scale)

        if self.random_rotation:
            # Keep start/goal constrained to the fixed y=-11 / y=11 planes.
            self._maze_rotation = None

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
        self.drone_radius = random.uniform(0.10, 0.15)
        self.max_speed = float(min(5.0 * self.speed_mtp, self.max_speed_ceiling))
        self._reset_drone_state(getattr(self, "_obstacle_scale", None))


    def _reset_drone_state(self, obstacle_scale):
        B = self.batch_size
        device = self.device

        del obstacle_scale

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        x = self.spawn_x_center + (torch.rand(B, device=device) * 2.0 - 1.0) * self.fixed_spawn_half_span
        z = self.spawn_z_center + (torch.rand(B, device=device) * 2.0 - 1.0) * self.fixed_spawn_half_span
        x_goal = self.spawn_x_center + (torch.rand(B, device=device) * 2.0 - 1.0) * self.fixed_spawn_half_span
        z_goal = self.spawn_z_center + (torch.rand(B, device=device) * 2.0 - 1.0) * self.fixed_spawn_half_span

        y = torch.full((B,), -11.5, device=device)
        y_goal = torch.full((B,), 11.5, device=device)

        self.p = torch.stack([x, y, z], dim=-1)
        self.p_target = torch.stack([x_goal, y_goal, z_goal], dim=-1)

        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)
        self.v = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2

        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(
            R,
            self.act,
            torch.randn((B, 3), device=device) * 0.2 + safe_normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay),
            2,
        )
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)

    def run(self, act_pred, ctl_dt=1 / 15, v_pred=None):
        act_pred = torch.nan_to_num(act_pred, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        if v_pred is not None:
            v_pred = torch.nan_to_num(
                v_pred,
                nan=0.0,
                posinf=self.hard_vpred_clip,
                neginf=-self.hard_vpred_clip,
            ).clamp(-self.hard_vpred_clip, self.hard_vpred_clip)

        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        dyn_fn = differentiable_run_torch if self.use_meta_fallback else differentiable_run
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
        self.p, self.v, self.a = self._apply_soft_contacts(self.p_old, p_free, v_free, a_free, ctl_dt)
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
        if self.use_meta_fallback:
            self.R = update_state_vec_torch(self.R, self.act, v_pred, alpha, 2)
        else:
            self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 2)
        self.R = torch.nan_to_num(self.R, nan=0.0, posinf=1.0, neginf=-1.0)
