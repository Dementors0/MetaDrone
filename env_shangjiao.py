import math
import random
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


class Env:
    """
    Random-obstacle environment based on file 1, with a thin compatibility layer for code
    that previously used file 2.

    Kept compatible items:
      - reset_drone_only()
      - current_clearance / last_step_clearance
      - last_contact_strength / penetration / tangent_speed / normal_speed
      - compute_geodesic_distance()  (degraded to Euclidean distance)

    Intentionally removed:
      - soft contact / voxel-only contact correction
      - maze graph / true geodesic on grid
    """

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
        self.ground_voxel_w = torch.tensor([8., 18, 0, 2.9, 2.9, 1.9], device=device)
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device)
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device)
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)
        self.gate_w = torch.tensor([2., 2, 1.0, 0.5], device=device)
        self.gate_b = torch.tensor([3., -1, 0.0, 0.5], device=device)
        self.v_wind_w = torch.tensor([1, 1, 0.2], device=device)
        self.g_std = torch.tensor([0., 0, -9.80665], device=device)
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)
        self.p_init = torch.as_tensor([
            [-1.5, -3., 1],
            [9.5, -3., 1],
            [-0.5, 1., 1],
            [8.5, 1., 1],
            [0.0, 3., 1],
            [8.0, 3., 1],
            [-1.0, -1., 1],
            [9.0, -1., 1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.p_end = torch.as_tensor([
            [8., 3., 1],
            [0., 3., 1],
            [8., -1., 1],
            [0., -1., 1],
            [8., -3., 1],
            [0., -3., 1],
            [8., 1., 1],
            [0., 1., 1],
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
        self.reset()

    def _sample_camera(self):
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

    def _sample_layout(self):
        B = self.batch_size
        device = self.device

        self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b
        self.voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b
        self.cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b
        self.cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b

        self.n_drones_per_group = random.choice([4, 8])
        self.drone_radius = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp
        scale = (self.max_speed - 0.5).clamp_min(1)
        self._layout_scale = scale

        roof = torch.rand((B,), device=device) < 0.5
        self.balls[~roof, :15, :2] = self.cyl[~roof, :15, :2]
        self.voxels[~roof, :15, :2] = self.cyl[~roof, 15:, :2]
        self.balls[~roof, :15] = self.balls[~roof, :15] + self.roof_add[:4]
        self.voxels[~roof, :15] = self.voxels[~roof, :15] + self.roof_add
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl_h[..., 2])
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

        self.voxels[:, :, 1] *= (self.max_speed + 4) / scale
        self.balls[:, :, 1] *= (self.max_speed + 4) / scale
        self.cyl[:, :, 1] *= (self.max_speed + 4) / scale

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

        self.voxels[..., 0] *= scale
        self.balls[..., 0] *= scale
        self.cyl[..., 0] *= scale
        self.cyl_h[..., 0] *= scale
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), scale * 8 - 0.3 - ground_balls_r_ground)

    def _sample_drone_task(self):
        B = self.batch_size
        device = self.device

        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)
        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        scale = self._layout_scale
        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        task_scale = torch.cat([
            scale,
            rd + 0.5,
            torch.rand_like(scale) - 0.5,
        ], -1)
        self.p = self.p_init * task_scale + torch.randn_like(task_scale) * 0.1
        self.p_target = self.p_end * task_scale + torch.randn_like(task_scale) * 0.1

        if self.random_rotation:
            yaw_bias = torch.rand(B // self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            ones = torch.ones_like(c)
            zeros = torch.zeros_like(c)
            R_bias = torch.stack([
                c, -s, zeros,
                s, c, zeros,
                zeros, zeros, ones,
            ], -1).reshape(B, 3, 3)
            center = 0.5 * (self.p + self.p_target)
            self.p = (R_bias @ (self.p - center)[..., None]).squeeze(-1) + center
            self.p_target = (R_bias @ (self.p_target - center)[..., None]).squeeze(-1) + center

        self.v = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2

        R = torch.zeros((B, 3, 3), device=device)
        v_ref = torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p, dim=-1)
        self.R = quadsim_cuda.update_state_vec(R, self.act, v_ref, torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1

        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)

    def reset(self):
        self._sample_camera()
        self._sample_layout()
        self._sample_drone_task()
        self._reset_contact_cache()

    def reset_drone_only(self):
        """Reset drone states/targets while keeping the current obstacle layout unchanged."""
        self._sample_camera()
        self.drone_radius = random.uniform(0.1, 0.15)
        self._sample_drone_task()
        self._reset_contact_cache()

    def _compute_signed_clearance(self, points):
        """
        Approximate clearance based on nearest surface point returned by CUDA helper.
        Positive => distance to nearest obstacle surface.
        Near zero => almost touching.

        This is not the maze voxel signed distance from file 2, but it gives a useful
        all-obstacle safety margin for logging / light loss usage.
        """
        points = points.contiguous()
        nearest_pt = torch.empty_like(points)
        quadsim_cuda.find_nearest_pt(
            nearest_pt,
            self.balls,
            self.cyl,
            self.cyl_h,
            self.voxels,
            points,
            self.drone_radius,
            self.n_drones_per_group,
        )
        clearance = (nearest_pt - points).norm(2, -1) - float(self.drone_radius)
        return torch.nan_to_num(clearance, nan=0.0, posinf=10.0, neginf=-10.0)

    def _compute_swept_clearance(self, p_start, p_end):
        alphas = torch.linspace(0.0, 1.0, 9, device=p_start.device, dtype=p_start.dtype).reshape(-1, 1, 1)
        samples = p_start.unsqueeze(0) * (1 - alphas) + p_end.unsqueeze(0) * alphas
        clearance = self._compute_signed_clearance(samples.reshape(-1, 3)).reshape(alphas.shape[0], self.batch_size)
        return clearance.min(dim=0).values

    def _reset_contact_cache(self):
        zeros = torch.zeros((self.batch_size,), device=self.device)
        self.current_clearance = self._compute_signed_clearance(self.p)
        self.last_step_clearance = self.current_clearance.clone()
        self.last_contact_strength = zeros.clone()
        self.last_contact_penetration = zeros.clone()
        self.last_contact_tangent_speed = zeros.clone()
        self.last_contact_normal_speed = zeros.clone()

    def compute_geodesic_distance(self, p_curr, p_goal):
        """Compatibility stub: in random obstacle maps this degrades to Euclidean distance."""
        return torch.norm(p_curr - p_goal, dim=-1)

    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        self_forward_vec = R[..., 0]
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        a_thr = a_thr - g_std
        thrust = torch.norm(a_thr, 2, -1, True)
        self_up_vec = a_thr / thrust
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[:, 2]
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)
        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1)

    def render(self, ctl_dt):
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
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

    def _update_compat_metrics(self, prev_p, new_p, new_v):
        self.current_clearance = torch.nan_to_num(
            self._compute_signed_clearance(new_p), nan=0.0, posinf=10.0, neginf=-10.0
        ).clamp(-10.0, 10.0)
        self.last_step_clearance = torch.nan_to_num(
            torch.minimum(self._compute_swept_clearance(prev_p, new_p), self.current_clearance),
            nan=0.0, posinf=10.0, neginf=-10.0,
        ).clamp(-10.0, 10.0)

        penetration = (-self.current_clearance).clamp_min(0.0)
        self.last_contact_penetration = penetration
        self.last_contact_strength = (penetration > 0).to(new_p.dtype)

        motion = new_p - prev_p
        motion_norm = motion.norm(2, -1).clamp_min(1e-6)
        toward_obs = -(self.find_vec_to_nearest_pt()[:, 0])
        toward_obs_norm = toward_obs.norm(2, -1).clamp_min(1e-6)
        cos_sim = torch.sum(motion * toward_obs, dim=-1) / (motion_norm * toward_obs_norm)
        normal_speed = torch.clamp(cos_sim, min=0.0) * new_v.norm(2, -1)
        tangent_speed = torch.sqrt(torch.clamp(new_v.norm(2, -1).pow(2) - normal_speed.pow(2), min=0.0))
        self.last_contact_normal_speed = torch.where(self.last_contact_strength > 0, normal_speed, torch.zeros_like(normal_speed))
        self.last_contact_tangent_speed = torch.where(self.last_contact_strength > 0, tangent_speed, torch.zeros_like(tangent_speed))

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        act_pred = torch.nan_to_num(act_pred, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        if v_pred is not None:
            v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
        self.act = torch.nan_to_num(self.act, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        self.p = torch.nan_to_num(self.p, nan=0.0, posinf=100.0, neginf=-100.0)
        self.v = torch.nan_to_num(self.v, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        self.a = torch.nan_to_num(self.a, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)

        self._update_compat_metrics(self.p_old, self.p, self.v)

        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)
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
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt ** 2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next

        self.p = torch.nan_to_num(self.p, nan=0.0, posinf=100.0, neginf=-100.0)
        self.v = torch.nan_to_num(self.v, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        self.a = torch.nan_to_num(self.a, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        self._update_compat_metrics(self.p_old, self.p, self.v)

        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)
        self.R = torch.nan_to_num(self.R, nan=0.0, posinf=1.0, neginf=-1.0)