import random
import torch

import quadsim_cuda
from env import Env as BaseEnv, safe_normalize


class Env(BaseEnv):
    """Environment for validating local-minimum behavior.

    The layout builds a deterministic U-shaped dead-end in front of the start-goal
    straight line. A policy that greedily follows goal direction tends to enter the
    pocket and gets trapped near the dead-end, while a better policy should detour
    around side gaps to reach the goal.
    """

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
        trap_width_ratio=0.15,
        trap_depth_ratio=0.30,
        trap_open_y_ratio=0.20,
    ):
        self.trap_width_ratio = float(trap_width_ratio)
        self.trap_depth_ratio = float(trap_depth_ratio)
        self.trap_open_y_ratio = float(trap_open_y_ratio)

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

    def reset(self):
        super().reset()
        self._apply_local_minimum_layout()
        # Disable rotating cached maze layout in reset_drone_only for this task env.
        self._maze_rotation = None

    def _apply_local_minimum_layout(self):
        B = self.batch_size
        device = self.device
        dtype = torch.float32

        # Keep only voxel walls in this benchmark layout.
        self.balls = torch.zeros((B, 0, 4), device=device, dtype=dtype)
        self.cyl = torch.zeros((B, 0, 3), device=device, dtype=dtype)
        self.cyl_h = torch.zeros((B, 0, 3), device=device, dtype=dtype)

        sx = float(self.scene_x_half)
        sy = float(self.scene_y_half)
        x_mid = 0.5 * sx

        # Wall dimensions: world z in approximately [-0.2, 2.2]
        hz = 1.2
        cz = 1.0
        th = 0.08

        walls = []

        # Outer boundary walls
        walls.append([0.0, 0.0, cz, th, sy, hz])
        walls.append([sx, 0.0, cz, th, sy, hz])
        walls.append([x_mid, sy, cz, sx * 0.5, th, hz])
        walls.append([x_mid, -sy, cz, sx * 0.5, th, hz])

        # U-shaped dead-end trap in the central corridor.
        trap_half_w = max(0.6, sx * self.trap_width_ratio)
        open_y = sy * self.trap_open_y_ratio
        deadend_y = open_y - max(2.0, sy * self.trap_depth_ratio)
        arm_half_y = max(0.6, 0.5 * (open_y - deadend_y))
        arm_center_y = 0.5 * (open_y + deadend_y)

        # U left/right arms and dead-end back wall.
        walls.append([x_mid - trap_half_w, arm_center_y, cz, th, arm_half_y, hz])
        walls.append([x_mid + trap_half_w, arm_center_y, cz, th, arm_half_y, hz])
        walls.append([x_mid, deadend_y, cz, trap_half_w, th, hz])

        # A lower splitter encourages side detour and prevents trivial straight pass.
        splitter_y = deadend_y - max(1.2, 0.10 * sy)
        walls.append([x_mid + 0.2 * trap_half_w, splitter_y, cz, trap_half_w * 0.8, th, hz])

        # Floor and ceiling slabs
        walls.append([x_mid, 0.0, -0.2, sx * 0.5, sy, 0.2])
        walls.append([x_mid, 0.0, 2.2, sx * 0.5, sy, 0.2])

        vox = torch.tensor(walls, device=device, dtype=dtype)
        self.voxels = vox.unsqueeze(0).repeat(B, 1, 1)

    def _reset_drone_state(self, obstacle_scale):
        B = self.batch_size
        device = self.device

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        # Start and goal aligned with the trap's longitudinal axis.
        sx = float(self.scene_x_half)
        sy = float(self.scene_y_half)
        x_mid = 0.5 * sx

        x_jitter = (torch.rand(B, device=device) - 0.5) * (0.30 * sx)
        x_start = (x_mid + x_jitter).clamp(0.6, sx - 0.6)
        x_goal = (x_mid + 0.15 * x_jitter).clamp(0.6, sx - 0.6)

        y_start = torch.full((B,), sy * 0.75, device=device)
        y_goal = torch.full((B,), -sy * 0.75, device=device)

        z_start = 0.9 + torch.rand(B, device=device) * 0.4
        z_goal = 0.9 + torch.rand(B, device=device) * 0.4

        self.p = torch.stack([x_start, y_start, z_start], dim=-1)
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
