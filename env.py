import math
import random
import time
import torch
import torch.nn.functional as F
import quadsim_cuda


class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply


class RunFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode):
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, airmode)
        ctx.save_for_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = ctx.saved_tensors
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next,
            ctx.grad_decay, ctx.ctl_dt)
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None

run = RunFunction.apply


def safe_normalize(x, dim=-1, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, p=2, dim=dim, eps=eps)


class Env:
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10) -> None:
        self.device = device
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.grad_decay = grad_decay
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device)
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device)
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)
        self.ground_voxel_w = torch.tensor([8., 18,  0, 2.9, 2.9, 1.9], device=device)
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device)
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device)
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)
        self.gate_w = torch.tensor([2.,  2,  1.0, 0.5], device=device)
        self.gate_b = torch.tensor([3., -1,  0.0, 0.5], device=device)
        self.v_wind_w = torch.tensor([1,  1,  0.2], device=device)
        self.g_std = torch.tensor([0., 0, -9.80665], device=device)
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)
        self.p_init = torch.as_tensor([
            [-1.5, -3.,  1],
            [ 9.5, -3.,  1],
            [-0.5,  1.,  1],
            [ 8.5,  1.,  1],
            [ 0.0,  3.,  1],
            [ 8.0,  3.,  1],
            [-1.0, -1.,  1],
            [ 9.0, -1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.p_end = torch.as_tensor([
            [8.,  3.,  1],
            [0.,  3.,  1],
            [8., -1.,  1],
            [0., -1.,  1],
            [8., -3.,  1],
            [0., -3.,  1],
            [8.,  1.,  1],
            [0.,  1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.flow = torch.empty((batch_size, 0, height, width), device=device)
        self.single = single
        self.gate = gate
        self.ground_voxels = ground_voxels
        self.scaffold = scaffold
        self.speed_mtp = speed_mtp
        self.random_rotation = random_rotation
        self.cam_angle = cam_angle
        self.fov_x_half_tan = fov_x_half_tan
        self.contact_buffer = 0.02
        self.contact_softness = 0.02
        self.contact_gate_softness = 0.04
        self.contact_velocity_softness = 0.10
        self.contact_normal_damping = 1.0
        self.speed_limit_softness = 0.05
        self.reset()
        # self.obj_avoid_grad_mtp = torch.tensor([0.5, 2., 1.], device=device)

    def reset(self):
        B = self.batch_size
        device = self.device

        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        # Keep maze metadata for downstream consumers that probe these attributes.
        self.maze_cols = 8
        self.maze_rows = 18
        self.maze_cell_size = 1.0

        # env obstacles (aligned with env_cuda)
        self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b
        self.voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b
        self.cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b
        self.cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b

        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.n_drones_per_group = random.choice([4, 8])
        self.drone_radius = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1

        rd_speed = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        speed_profile = (0.75 + 2.5 * rd_speed) * self.speed_mtp
        obstacle_scale = (speed_profile - 0.5).clamp_min(1)
        self._obstacle_scale = obstacle_scale
        self.max_speed = float(min(5.0 * self.speed_mtp, 5.0))

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        roof = torch.rand((B,), device=device) < 0.5
        self.balls[~roof, :15, :2] = self.cyl[~roof, :15, :2]
        self.voxels[~roof, :15, :2] = self.cyl[~roof, 15:, :2]
        self.balls[~roof, :15] = self.balls[~roof, :15] + self.roof_add[:4]
        self.voxels[~roof, :15] = self.voxels[~roof, :15] + self.roof_add
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / obstacle_scale), 8 - 0.3 / obstacle_scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / obstacle_scale), 8 - 0.3 / obstacle_scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / obstacle_scale), 8 - 0.3 / obstacle_scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / obstacle_scale), 8 - 0.3 / obstacle_scale - self.cyl_h[..., 2])
        self.voxels[roof, 0, 2] = self.voxels[roof, 0, 2] * 0.5 + 201
        self.voxels[roof, 0, 3:] = 200

        if self.ground_voxels:
            ground_balls_r = 8 + torch.rand((B, 2), device=device) * 6
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            self.balls[:, :2, 3] = ground_balls_r
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r - 1

            ground_voxels = torch.rand((B, 10, 6), device=device) * self.ground_voxel_w + self.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1
            self.voxels = torch.cat([self.voxels, ground_voxels], 1)

        self.voxels[:, :, 1] *= (speed_profile + 4) / obstacle_scale
        self.balls[:, :, 1] *= (speed_profile + 4) / obstacle_scale
        self.cyl[:, :, 1] *= (speed_profile + 4) / obstacle_scale

        if self.gate:
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b
            p = gate[None, :, :3]
            nearest_pt = torch.empty_like(p)
            quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, 1)
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50
            ones = torch.ones_like(gate_x)
            gate = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1),
            ], 1)
            self.voxels = torch.cat([self.voxels, gate], 1)

        self.voxels[..., 0] *= obstacle_scale
        self.balls[..., 0] *= obstacle_scale
        self.cyl[..., 0] *= obstacle_scale
        self.cyl_h[..., 0] *= obstacle_scale
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), obstacle_scale * 8 - 0.3 - ground_balls_r_ground)

        if self.scaffold and random.random() < 0.5:
            x = torch.arange(1, 6, dtype=torch.float, device=device)
            y = torch.arange(-3, 4, dtype=torch.float, device=device)
            z = torch.arange(1, 4, dtype=torch.float, device=device)
            _x, _y = torch.meshgrid(x, y)
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            x_bias = torch.rand_like(speed_profile) * speed_profile
            scale_scaf = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale_scaf + torch.stack([
                x_bias,
                torch.randn_like(speed_profile),
                torch.rand_like(speed_profile) * 0.01
            ], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)
            _x, _z = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            scaf_h = scaf_h * scale_scaf + torch.stack([
                x_bias,
                torch.randn_like(speed_profile) * 0.1,
                torch.rand_like(speed_profile) * 0.01
            ], -1)
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)

        self._maze_rotation = None
        self._reset_drone_state(obstacle_scale)

        if self.random_rotation:
            yaw_bias = torch.rand(B // self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias)
            o = torch.zeros_like(yaw_bias)
            R_rot = torch.stack([c, -s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3)
            self._maze_rotation = R_rot
            self.p = torch.squeeze(R_rot @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R_rot @ self.p_target[..., None], -1)
            self.voxels[..., :3] = (R_rot @ self.voxels[..., :3].transpose(1, 2)).transpose(1, 2)
            self.balls[..., :3] = (R_rot @ self.balls[..., :3].transpose(1, 2)).transpose(1, 2)
            self.cyl[..., :3] = (R_rot @ self.cyl[..., :3].transpose(1, 2)).transpose(1, 2)

    def reset_drone_only(self):
        B = self.batch_size
        device = self.device

        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.drone_radius = random.uniform(0.1, 0.15)
        self.max_speed = float(min(5.0 * self.speed_mtp, 5.0))

        obstacle_scale = getattr(self, '_obstacle_scale', None)
        if obstacle_scale is None:
            rd_speed = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
            speed_profile = (0.75 + 2.5 * rd_speed) * self.speed_mtp
            obstacle_scale = (speed_profile - 0.5).clamp_min(1)
            self._obstacle_scale = obstacle_scale

        self._reset_drone_state(obstacle_scale)

        if self.random_rotation and getattr(self, '_maze_rotation', None) is not None:
            R_rot = self._maze_rotation
            self.p = torch.squeeze(R_rot @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R_rot @ self.p_target[..., None], -1)

    def _reset_drone_state(self, obstacle_scale):
        B = self.batch_size
        device = self.device

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        # 随机将起点和终点分别放置在环境长边(Y方向)的两端
        # 环境 Y 范围: -9 到 9 (rows=18, y_offset=9)
        # 环境 X 范围: 0 到 8 (cols=8)
        # 起点放在 Y 负端 (Y ≈ -8.5)，终点放在 Y 正端 (Y ≈ 8.5)
        x_start = torch.rand(B, device=device) * 6 + 1  # X in [1, 7] 避开边界
        y_start = torch.rand(B, device=device) * 1.0 - 8.5  # Y in [-8.5, -7.5]
        z_start = torch.ones(B, device=device) * 1.0  # Z = 1

        x_end = torch.rand(B, device=device) * 6 + 1  # X in [1, 7]
        y_end = torch.rand(B, device=device) * 1.0 + 7.5  # Y in [7.5, 8.5]
        z_end = torch.ones(B, device=device) * 1.0  # Z = 1

        self.p = torch.stack([x_start, y_start, z_start], dim=-1) * obstacle_scale + torch.randn(B, 3, device=device) * 0.1
        self.p_target = torch.stack([x_end, y_end, z_end], dim=-1) * obstacle_scale + torch.randn(B, 3, device=device) * 0.1

        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)

        self.v = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2

        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(
            R, self.act,
            torch.randn((B, 3), device=device) * 0.2 + safe_normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 2)
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1

        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)

    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=2):
        self_forward_vec = R[..., 0]
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        a_thr = a_thr - g_std
        thrust = torch.norm(a_thr, 2, -1, True)
        self_up_vec = a_thr / thrust
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[2]
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)
        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1)

    def _apply_soft_contacts(self, p_prev, p_free, v_free, a_free, ctl_dt):
        if self.voxels.numel() == 0:
            return p_free, v_free, a_free

        vox_centers = self.voxels[..., :3]
        vox_half = self.voxels[..., 3:]
        dtype = p_free.dtype
        radius = torch.as_tensor(float(self.drone_radius), device=p_free.device, dtype=dtype)
        buffer = torch.as_tensor(self.contact_buffer, device=p_free.device, dtype=dtype)
        softness = torch.as_tensor(self.contact_softness, device=p_free.device, dtype=dtype)
        gate_softness = torch.as_tensor(self.contact_gate_softness, device=p_free.device, dtype=dtype)
        velocity_softness = torch.as_tensor(self.contact_velocity_softness, device=p_free.device, dtype=dtype)
        dt = torch.as_tensor(max(float(ctl_dt), 1e-4), device=p_free.device, dtype=dtype)

        dp = p_free - p_prev
        p_mid = 0.5 * (p_prev + p_free)
        sweep_half = 0.5 * dp.abs().unsqueeze(1) + vox_half + radius
        overlap_margin = sweep_half - (p_mid.unsqueeze(1) - vox_centers).abs()

        normal_axis = vox_half.argmin(dim=-1)
        axis_idx = normal_axis.unsqueeze(-1)
        axis_mask = F.one_hot(normal_axis, num_classes=3).to(dtype=dtype)

        point_expand = (-1, vox_centers.size(1), -1)
        p_prev_expanded = p_prev.unsqueeze(1).expand(*point_expand)
        p_free_expanded = p_free.unsqueeze(1).expand(*point_expand)
        v_free_expanded = v_free.unsqueeze(1).expand(*point_expand)

        center_n = vox_centers.gather(2, axis_idx).squeeze(-1)
        half_n = vox_half.gather(2, axis_idx).squeeze(-1)
        p_prev_n = p_prev_expanded.gather(2, axis_idx).squeeze(-1)
        p_free_n = p_free_expanded.gather(2, axis_idx).squeeze(-1)
        v_free_n = v_free_expanded.gather(2, axis_idx).squeeze(-1)

        side = torch.where(p_prev_n >= center_n, torch.ones_like(p_prev_n), -torch.ones_like(p_prev_n))
        boundary = center_n + side * (half_n + radius + buffer)
        clearance_free = side * (p_free_n - boundary)
        penetration = softness * F.softplus(-clearance_free / softness)

        overlap_gate = torch.sigmoid(overlap_margin / gate_softness)
        overlap_gate = torch.where(axis_mask.bool(), torch.ones_like(overlap_gate), overlap_gate)
        tangential_gate = overlap_gate.prod(dim=-1)
        contact_gate = tangential_gate * torch.sigmoid(-clearance_free / gate_softness)

        pos_corr_mag = penetration * tangential_gate
        pos_corr = (axis_mask * (side * pos_corr_mag).unsqueeze(-1)).sum(dim=1)
        p_contact = p_free + pos_corr

        inward_speed = velocity_softness * F.softplus(-(side * v_free_n) / velocity_softness)
        vel_corr_mag = self.contact_normal_damping * inward_speed * contact_gate
        vel_corr = (axis_mask * (side * vel_corr_mag).unsqueeze(-1)).sum(dim=1)
        v_contact = v_free + vel_corr
        a_contact = a_free + (v_contact - v_free) / dt

        return p_contact, v_contact, a_contact

    def _smooth_cap_magnitude(self, vec, cap, softness):
        norm = torch.norm(vec, 2, -1, keepdim=True)
        cap = torch.as_tensor(cap, device=vec.device, dtype=vec.dtype)
        softness = torch.as_tensor(softness, device=vec.device, dtype=vec.dtype)
        capped_norm = norm - softness * F.softplus((norm - cap) / softness)
        scale = capped_norm / (norm + 1e-6)
        return vec * scale

    def _apply_speed_limit(self, p_prev, p_curr, v_curr, a_curr, ctl_dt):
        dt = torch.as_tensor(max(float(ctl_dt), 1e-4), device=v_curr.device, dtype=v_curr.dtype)
        speed_cap = torch.as_tensor(float(self.max_speed), device=v_curr.device, dtype=v_curr.dtype)
        speed_softness = torch.as_tensor(self.speed_limit_softness, device=v_curr.device, dtype=v_curr.dtype)
        disp_softness = torch.clamp(speed_softness * dt, min=1e-4)

        p_disp_limited = self._smooth_cap_magnitude(p_curr - p_prev, speed_cap * dt, disp_softness)
        v_limited = self._smooth_cap_magnitude(v_curr, speed_cap, speed_softness)
        p_limited = p_prev + p_disp_limited
        a_limited = a_curr + (v_limited - v_curr) / dt
        return p_limited, v_limited, a_limited

    def render(self, ctl_dt):
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        # assert canvas.is_contiguous()
        # assert nearest_pt.is_contiguous()
        # assert self.balls.is_contiguous()
        # assert self.cyl.is_contiguous()
        # assert self.voxels.is_contiguous()
        # assert Rt.is_contiguous()
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
                            self.p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan)
        return canvas, None

    def find_vec_to_nearest_pt(self):
        p = self.p + self.v * self.sub_div
        nearest_pt = torch.empty_like(p)
        quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, self.n_drones_per_group)
        return nearest_pt - p

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        act_pred = torch.nan_to_num(act_pred, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        if v_pred is not None:
            v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        self.act, p_free, v_free, a_free = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
        self.act = torch.nan_to_num(self.act, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        p_free = torch.nan_to_num(p_free, nan=0.0, posinf=100.0, neginf=-100.0)
        v_free = torch.nan_to_num(v_free, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        a_free = torch.nan_to_num(a_free, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        # 去掉碰撞物理反馈，直接使用自由运动结果
        # self.p, self.v, self.a = self._apply_soft_contacts(self.p_old, p_free, v_free, a_free, ctl_dt)
        self.p, self.v, self.a = p_free, v_free, a_free
        self.p, self.v, self.a = self._apply_speed_limit(self.p_old, self.p, self.v, self.a, ctl_dt)
        self.p = torch.nan_to_num(self.p, nan=0.0, posinf=100.0, neginf=-100.0)
        self.v = torch.nan_to_num(self.v, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        self.a = torch.nan_to_num(self.a, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 2)
        self.R = torch.nan_to_num(self.R, nan=0.0, posinf=1.0, neginf=-1.0)

    def _run(self, act_pred, ctl_dt=1/15, v_pred=None):
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        z_drag = 0
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        a_next = self.act + self.dg - z_drag - drag
        self.p_old = self.p
        p_free = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        v_free = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.p, self.v, self.a = self._apply_soft_contacts(self.p_old, p_free, v_free, a_next, ctl_dt)
        self.p, self.v, self.a = self._apply_speed_limit(self.p_old, self.p, self.v, self.a, ctl_dt)

        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # ---- 迷宫参数（与 reset() 一致） ----
    cols, rows = 8, 18
    cell_size = 1.0          # env_maze: 1m 单元格
    y_offset = 9.0           # rows / 2
    th = 0.1                 # 墙壁半厚度

    # ---- DFS 生成完整迷宫（保留所有内部墙壁） ----
    visited = set()
    stack = [(0, 0)]
    visited.add((0, 0))
    passages = set()

    while stack:
        c, r = stack[-1]
        neighbors = []
        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < cols and 0 <= nr < rows and (nc, nr) not in visited:
                neighbors.append((nc, nr))
        if neighbors:
            nc, nr = random.choice(neighbors)
            visited.add((nc, nr))
            stack.append((nc, nr))
            passages.add(tuple(sorted(((c, r), (nc, nr)))))
        else:
            stack.pop()

    # ---- 收集墙壁 (center_x, center_y, half_dx, half_dy) ----
    walls = []
    # 竖直墙（沿 x 方向的边界）
    for r in range(rows):
        for c in range(cols + 1):
            is_wall = (c == 0 or c == cols)
            if not is_wall and tuple(sorted(((c - 1, r), (c, r)))) not in passages:
                is_wall = True
            if is_wall:
                cx = float(c) * cell_size
                cy = (r + 0.5) * cell_size - y_offset
                walls.append((cx, cy, th, 0.5 * cell_size))
    # 水平墙（沿 y 方向的边界）
    for c in range(cols):
        for r in range(rows + 1):
            is_wall = (r == 0 or r == rows)
            if not is_wall and tuple(sorted(((c, r - 1), (c, r)))) not in passages:
                is_wall = True
            if is_wall:
                cx = (c + 0.5) * cell_size
                cy = float(r) * cell_size - y_offset
                walls.append((cx, cy, 0.5 * cell_size, th))

    # ---- 随机起点/终点 ----
    sc, sr = random.randint(0, cols - 1), random.randint(0, rows - 1)
    ec, er = random.randint(0, cols - 1), random.randint(0, rows - 1)
    while (sc, sr) == (ec, er):
        ec, er = random.randint(0, cols - 1), random.randint(0, rows - 1)
    start = ((sc + 0.5) * cell_size, (sr + 0.5) * cell_size - y_offset)
    goal  = ((ec + 0.5) * cell_size, (er + 0.5) * cell_size - y_offset)

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(6, 12))
    ax.set_title('env_maze.py — Full Maze (8×18, cell=1m, wall=0.2m)', fontsize=13)

    for cx, cy, hdx, hdy in walls:
        rect = patches.Rectangle((cx - hdx, cy - hdy), 2 * hdx, 2 * hdy,
                                  linewidth=0.3, edgecolor='black', facecolor='#4a4a4a')
        ax.add_patch(rect)

    ax.plot(*start, 'go', markersize=10, label='Start')
    ax.plot(*goal,  'r*', markersize=14, label='Goal')

    ax.set_xlim(-0.5, cols * cell_size + 0.5)
    ax.set_ylim(-y_offset - 0.5, y_offset + 0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    out_path = 'maze_full_topview.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
