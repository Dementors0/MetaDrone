import math
import random
import time
import torch
import torch.nn.functional as F
import quadsim_cuda
import collections

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

        # Clear obstacles for maze environment where only walls matter
        self.balls = torch.zeros((B, 0, 4), device=device)
        self.cyl = torch.zeros((B, 0, 3), device=device)
        self.cyl_h = torch.zeros((B, 0, 3), device=device)

        # ---------------------------------------------------------------------
        # Maze Generation  (Easy difficulty)
        # ---------------------------------------------------------------------
        # Grid: 8x18 cells, with 1.5m corridor width
        cols, rows = 8, 18
        cell_size = 1.5
        self.maze_cols = cols
        self.maze_rows = rows
        self.maze_cell_size = cell_size
        y_center_offset = rows * cell_size / 2.0   # 13.5
        
        # Initialize Graph (Fully connected grid)
        self.graph = {}
        for c in range(cols):
            for r in range(rows):
                neighbors = []
                for dc, dr in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nc, nr = c+dc, r+dr
                    if 0 <= nc < cols and 0 <= nr < rows:
                        neighbors.append((nc, nr))
                self.graph[(c,r)] = set(neighbors)

        # Iterative DFS for maze generation (single layout for batch efficiency)
        visited = set()
        stack = [(0, 0)]
        visited.add((0, 0))
        passages = set()
        
        while stack:
            c, r = stack[-1]
            neighbors = []
            for dc, dr in [(-1,0), (1,0), (0,-1), (0,1)]:
                nc, nr = c+dc, r+dr
                if 0 <= nc < cols and 0 <= nr < rows and (nc, nr) not in visited:
                    neighbors.append((nc, nr))
            
            if neighbors:
                nc, nr = random.choice(neighbors)
                visited.add((nc, nr))
                stack.append((nc, nr))
                passages.add(tuple(sorted(((c, r), (nc, nr)))))
            else:
                stack.pop()
        
        wall_list = []
        # Wall dimensions: CenterZ=1.0, HalfSizeZ=1.0 => 0m to 2m height
        c_z = 1.0
        h_z = 1.0
        th = 0.1  # Half thickness of wall (same as env_maze)
        wall_keep_prob = 0.65  # Keep 65% of non-passage internal walls
        
        # Vertical walls (YZ plane)
        for r in range(rows):
            for c in range(cols + 1):
                is_wall = (c == 0 or c == cols)  # Perimeter always kept
                if not is_wall and tuple(sorted(((c-1, r), (c, r)))) not in passages:
                    # Randomly keep 65% of internal walls
                    if random.random() < wall_keep_prob:
                        is_wall = True
                
                if is_wall:
                    wall_list.append([
                        float(c) * cell_size,
                        (r + 0.5) * cell_size - y_center_offset,
                        c_z,
                        th,
                        0.5 * cell_size,
                        h_z
                    ])
                    # Update Graph: Remove connection between (c-1, r) and (c, r)
                    if 0 < c < cols:
                        if (c, r) in self.graph.get((c-1, r), set()):
                            self.graph[(c-1, r)].remove((c, r))
                        if (c-1, r) in self.graph.get((c, r), set()):
                            self.graph[(c, r)].remove((c-1, r))
        
        # Horizontal walls (XZ plane)
        for c in range(cols):
            for r in range(rows + 1):
                is_wall = (r == 0 or r == rows)  # Perimeter always kept
                if not is_wall and tuple(sorted(((c, r-1), (c, r)))) not in passages:
                    # Randomly keep 65% of internal walls
                    if random.random() < wall_keep_prob:
                        is_wall = True
                
                if is_wall:
                    wall_list.append([
                        (c + 0.5) * cell_size,
                        float(r) * cell_size - y_center_offset,
                        c_z,
                        0.5 * cell_size,
                        th,
                        h_z
                    ])
                    # Update Graph: Remove connection between (c, r-1) and (c, r)
                    if 0 < r < rows:
                        if (c, r) in self.graph.get((c, r-1), set()):
                            self.graph[(c, r-1)].remove((c, r))
                        if (c, r-1) in self.graph.get((c, r), set()):
                            self.graph[(c, r)].remove((c, r-1))
                    
        walls_tensor = torch.tensor(wall_list, device=device, dtype=torch.float32)

        # Add floor and ceiling boundaries to prevent flying out
        maze_half_x = cols * cell_size / 2.0 + 1.0
        maze_half_y = rows * cell_size / 2.0 + 1.0
        maze_center_x = cols * cell_size / 2.0
        bounds = torch.tensor([
            [maze_center_x, 0.0, -0.5, maze_half_x, maze_half_y, 0.5], # Floor
            [maze_center_x, 0.0,  2.5, maze_half_x, maze_half_y, 0.5]  # Ceiling
        ], device=device)
        
        if walls_tensor.nelement() == 0:
            walls_tensor = bounds
        else:
            walls_tensor = torch.cat([walls_tensor, bounds], dim=0)

        # Replicate for batch: [B, N_walls, 6]
        self.voxels = walls_tensor.unsqueeze(0).repeat(B, 1, 1)

        # ---------------------------------------------------------------------
        # Drone & Path Initialization
        # ---------------------------------------------------------------------
        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.drone_radius = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1
        else:
            self.n_drones_per_group = 8 # Default to 8 or choice

        self.max_speed = 5.0 * self.speed_mtp
        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01
        
        # Generate random start/end points
        starts = []
        ends = []
        for _ in range(B):
            sc = random.randint(0, cols-1)
            sr = random.randint(0, rows-1)
            ec = random.randint(0, cols-1)
            er = random.randint(0, rows-1)
            # Ensure start != end
            while (sc, sr) == (ec, er):
                 ec = random.randint(0, cols-1)
                 er = random.randint(0, rows-1)
            
            starts.append([(sc + 0.5) * cell_size, (sr + 0.5) * cell_size - y_center_offset, 1.0])
            ends.append([(ec + 0.5) * cell_size, (er + 0.5) * cell_size - y_center_offset, 1.0])
            
        p = torch.tensor(starts, device=device)
        self.p_target = torch.tensor(ends, device=device)
        self.p = p + torch.randn_like(p) * 0.1 # Add small noise to start

        # Timings & Dynamics (align with env_maze)
        self.pitch_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)
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

        # drag coef
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)

    def reset_drone_only(self):
        """Reset drone states while keeping the current easy-maze layout unchanged."""
        B = self.batch_size
        device = self.device
        cols = self.maze_cols
        rows = self.maze_rows
        cell_size = self.maze_cell_size
        y_center_offset = rows * cell_size / 2.0

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
        if self.single:
            self.n_drones_per_group = 1
        else:
            self.n_drones_per_group = 8

        self.max_speed = 5.0 * self.speed_mtp
        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        starts = []
        ends = []
        for _ in range(B):
            sc = random.randint(0, cols - 1)
            sr = random.randint(0, rows - 1)
            ec = random.randint(0, cols - 1)
            er = random.randint(0, rows - 1)
            while (sc, sr) == (ec, er):
                ec = random.randint(0, cols - 1)
                er = random.randint(0, rows - 1)
            starts.append([(sc + 0.5) * cell_size, (sr + 0.5) * cell_size - y_center_offset, 1.0])
            ends.append([(ec + 0.5) * cell_size, (er + 0.5) * cell_size - y_center_offset, 1.0])

        p = torch.tensor(starts, device=device)
        self.p_target = torch.tensor(ends, device=device)
        self.p = p + torch.randn_like(p) * 0.1

        self.pitch_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)
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

    def get_grid_coords(self, p):
        # p: [B, 3] or [N, 3]
        cell_size = getattr(self, 'maze_cell_size', 1.0)
        cols = getattr(self, 'maze_cols', 8)
        rows = getattr(self, 'maze_rows', 18)
        y_center_offset = rows * cell_size / 2.0
        # x range [0, cols*cell_size] -> c = floor(x / cell_size)
        # y range [-rows*cell_size/2, rows*cell_size/2] -> r = floor((y + y_center_offset) / cell_size)
        c = torch.floor(p[..., 0] / cell_size).long().clamp(0, cols - 1)
        r = torch.floor((p[..., 1] + y_center_offset) / cell_size).long().clamp(0, rows - 1)
        return c, r

    def compute_geodesic_distance(self, p_curr, p_goal):
        # This function computes the shortest path distance on the grid graph
        # Returns tensor [B]
        
        B = p_curr.shape[0]
        c_curr, r_curr = self.get_grid_coords(p_curr)
        c_goal, r_goal = self.get_grid_coords(p_goal)
        
        c_curr = c_curr.cpu().numpy()
        r_curr = r_curr.cpu().numpy()
        c_goal = c_goal.cpu().numpy()
        r_goal = r_goal.cpu().numpy()
        
        dists = []
        for i in range(B):
            start_node = (c_curr[i], r_curr[i])
            end_node = (c_goal[i], r_goal[i])
            
            if start_node == end_node:
                # Within same cell, use Euclidean
                dists.append(torch.norm(p_curr[i] - p_goal[i]))
                continue
                
            # BFS on CPU
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            found = False
            
            # Simple BFS
            while queue:
                current, dist = queue.popleft()
                if current == end_node:
                    # Found, return grid distance + residual Euclidean for endpoints?
                    # Convert grid steps to metric distance
                    cell_size = getattr(self, 'maze_cell_size', 1.0)
                    dists.append(torch.tensor(dist * cell_size))
                    found = True
                    break
                
                # Expand
                neighbors = self.graph.get(current, set())
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            if not found:
                 # Fallback to Euclidean (should not happen if graph is connected)
                 dists.append(torch.norm(p_curr[i] - p_goal[i]))

        return torch.stack(dists).to(self.device).float()

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
        self.p, self.v, self.a = self._apply_soft_contacts(self.p_old, p_free, v_free, a_free, ctl_dt)
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

        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # ---- 迷宫参数（与 reset() 一致） ----
    cols, rows = 8, 18
    cell_size = 1.5          # env_maze_easy: 1.5m 走廊
    y_offset = rows * cell_size / 2.0   # 13.5
    th = 0.1                 # 墙壁半厚度
    wall_keep_prob = 0.65    # 保留 65% 内部墙

    # ---- DFS 生成迷宫 ----
    visited = set()
    stack_dfs = [(0, 0)]
    visited.add((0, 0))
    passages = set()

    while stack_dfs:
        c, r = stack_dfs[-1]
        neighbors = []
        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < cols and 0 <= nr < rows and (nc, nr) not in visited:
                neighbors.append((nc, nr))
        if neighbors:
            nc, nr = random.choice(neighbors)
            visited.add((nc, nr))
            stack_dfs.append((nc, nr))
            passages.add(tuple(sorted(((c, r), (nc, nr)))))
        else:
            stack_dfs.pop()

    # ---- 收集墙壁（保留 65% 内部墙壁） ----
    walls = []
    # 竖直墙
    for r in range(rows):
        for c in range(cols + 1):
            is_wall = (c == 0 or c == cols)
            if not is_wall and tuple(sorted(((c - 1, r), (c, r)))) not in passages:
                if random.random() < wall_keep_prob:
                    is_wall = True
            if is_wall:
                cx = float(c) * cell_size
                cy = (r + 0.5) * cell_size - y_offset
                walls.append((cx, cy, th, 0.5 * cell_size))
    # 水平墙
    for c in range(cols):
        for r in range(rows + 1):
            is_wall = (r == 0 or r == rows)
            if not is_wall and tuple(sorted(((c, r - 1), (c, r)))) not in passages:
                if random.random() < wall_keep_prob:
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
    fig, ax = plt.subplots(figsize=(6, 14))
    ax.set_title('env_maze_easy.py — Medium Maze\n(8×18, cell=1.5m, wall=0.2m, 65%% internal walls)',
                 fontsize=12)

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

    out_path = 'maze_easy_topview.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
