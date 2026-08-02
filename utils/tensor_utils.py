"""Tensor, feature extraction, and loss helpers."""

from collections import defaultdict

import torch
from torch.nn import functional as F

from turn_loss_utils import compute_direction_stability_loss_3d


def safe_normalize(x, dim=-1, eps=1e-6):
    return F.normalize(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dim=dim, eps=eps)


def safe_l2_norm(x, dim=-1, keepdim=False, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim) + eps)


def compute_arrival_reward(p_history, p_target, radius=0.5):
    """Per-step positive reward inside goal ball; longer stay yields larger total reward."""
    radius = max(float(radius), 1e-6)
    target = p_target.unsqueeze(0) if p_history.dim() == 3 else p_target
    dist_to_goal = safe_l2_norm(p_history - target, dim=-1)
    reward_per_step = F.relu(radius - dist_to_goal) / radius
    # Time-average reward per agent so every in-goal step contributes.
    reward_per_agent = reward_per_step.mean(dim=0)
    reward = reward_per_agent.mean()
    with torch.no_grad():
        hit_rate = (dist_to_goal <= radius).any(dim=0).float().mean()
        best_dist = dist_to_goal.min(dim=0).values.mean()
    return reward, hit_rate, best_dist


def build_yaw_frame(R):
    fwd = R[:, :, 0]
    zeros = torch.zeros_like(fwd)
    up = zeros.clone()
    up[:, 2] = 1.0
    fwd_h_raw = torch.stack([fwd[:, 0], fwd[:, 1], torch.zeros_like(fwd[:, 2])], dim=-1)
    fwd_h_norm = safe_l2_norm(fwd_h_raw, dim=-1, keepdim=True)
    fallback = zeros.clone()
    fallback[:, 0] = 1.0
    fwd_h = torch.where(fwd_h_norm > 1e-6, fwd_h_raw / fwd_h_norm.clamp_min(1e-6), fallback)
    left = safe_normalize(torch.cross(up, fwd_h, dim=-1), dim=-1)
    return torch.stack([fwd_h, left, up], -1)


def compute_heading_reference(env, R_yaw):
    target_vec = env.p_target - env.p.detach()
    zeros = torch.zeros_like(target_vec[:, 2])
    heading_ref_world = torch.stack([target_vec[:, 0], target_vec[:, 1], zeros], dim=-1)
    heading_norm = safe_l2_norm(heading_ref_world, dim=-1, keepdim=True)
    fallback = R_yaw[:, :, 0]
    heading_ref_world = torch.where(
        heading_norm > 1e-6,
        heading_ref_world / heading_norm.clamp_min(1e-6),
        fallback,
    )
    heading_ref_local = torch.squeeze(heading_ref_world[:, None] @ R_yaw, 1)
    yaw_error = torch.atan2(heading_ref_local[:, 1], heading_ref_local[:, 0]).unsqueeze(-1)
    return heading_ref_world, heading_ref_local[:, :2], yaw_error


def compute_velocity_heading_command(
    R_yaw,
    v_ref_world,
    yaw_rate_max_value,
    yaw_kp=4.0,
    min_speed=0.25,
):
    """
    根据期望速度方向计算机头参考方向和 yaw_rate_cmd。

    逻辑：
    1. 只使用水平面速度方向，不让 z 方向影响 yaw；
    2. 机头参考方向始终来自速度方向（不再做低速保持当前机头）；
    3. yaw_rate_cmd = yaw_kp * yaw_error，并限制在 [-yaw_rate_max, yaw_rate_max]；
    4. 即使 v_ref_world 反向，也不会让机头瞬间跳 180°，而是通过 yaw_rate_max 平滑转过去。
    """
    _ = min_speed  # kept for call-site compatibility; no low-speed heading hold is applied.

    v_xy = torch.stack([
        v_ref_world[:, 0],
        v_ref_world[:, 1],
        torch.zeros_like(v_ref_world[:, 2]),
    ], dim=-1)
    speed_xy = safe_l2_norm(v_xy, dim=-1, keepdim=True)
    heading_ref_world = safe_normalize(v_xy, dim=-1)

    heading_ref_local = torch.squeeze(heading_ref_world[:, None] @ R_yaw, 1)
    yaw_error = torch.atan2(
        heading_ref_local[:, 1],
        heading_ref_local[:, 0],
    ).unsqueeze(-1)

    yaw_rate_cmd = torch.clamp(
        float(yaw_kp) * yaw_error,
        -float(yaw_rate_max_value),
        float(yaw_rate_max_value),
    )

    return heading_ref_world, heading_ref_local[:, :2], yaw_error, yaw_rate_cmd, speed_xy


def decode_worker_action(act, R_yaw, yaw_rate_max_value):
    """Decode body-frame acceleration and optional yaw-rate residual."""
    accel_body = act[:, :3]
    a_pred = torch.squeeze(R_yaw @ accel_body.unsqueeze(-1), -1)
    yaw_rate_cmd = None
    if act.shape[-1] > 3:
        yaw_rate_cmd = torch.tanh(act[:, 3:4]) * float(yaw_rate_max_value)
    return a_pred, yaw_rate_cmd


def sanitize_tensor(x, nan=0.0, posinf=1e3, neginf=-1e3):
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def extract_depth_geometry_features(depth, near_threshold=1.5):
    """
    从原始深度图提取显式几何/风险统计特征。

    输入:
        depth: [B, H, W]
    输出:
        geom_feat: [B, 19]
    """
    d = depth.clamp(0.3, 24.0)
    B, H, W = d.shape

    w1 = max(1, W // 3)
    w2 = max(w1 + 1, (2 * W) // 3)
    w2 = min(w2, W - 1) if W > 2 else w2

    h1 = max(1, H // 3)
    h2 = max(h1 + 1, (2 * H) // 3)
    h2 = min(h2, H - 1) if H > 2 else h2

    left = d[:, :, :w1]
    center = d[:, :, w1:w2]
    right = d[:, :, w2:]

    upper = d[:, :h1, :]
    middle = d[:, h1:h2, :]
    lower = d[:, h2:, :]

    def _mean_min_ratio(region):
        flat = region.reshape(B, -1)
        mean_v = flat.mean(dim=-1)
        min_v = flat.min(dim=-1).values
        near_ratio = (flat < near_threshold).float().mean(dim=-1)
        return mean_v, min_v, near_ratio

    def _mean_ratio(region):
        flat = region.reshape(B, -1)
        mean_v = flat.mean(dim=-1)
        near_ratio = (flat < near_threshold).float().mean(dim=-1)
        return mean_v, near_ratio

    l_mean, l_min, l_ratio = _mean_min_ratio(left)
    c_mean, c_min, c_ratio = _mean_min_ratio(center)
    r_mean, r_min, r_ratio = _mean_min_ratio(right)

    u_mean, u_ratio = _mean_ratio(upper)
    m_mean, m_ratio = _mean_ratio(middle)
    lo_mean, lo_ratio = _mean_ratio(lower)

    flat_all = d.reshape(B, -1)
    g_mean = flat_all.mean(dim=-1)
    g_std = flat_all.std(dim=-1, unbiased=False)
    lr_diff = l_mean - r_mean
    center_vs_side = c_mean - 0.5 * (l_mean + r_mean)

    geom_feat = torch.stack([
        l_mean, l_min, l_ratio,
        c_mean, c_min, c_ratio,
        r_mean, r_min, r_ratio,
        u_mean, u_ratio,
        m_mean, m_ratio,
        lo_mean, lo_ratio,
        g_mean, g_std, lr_diff, center_vs_side,
    ], dim=-1)
    return sanitize_tensor(geom_feat, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)


def extract_progress_features(p_history_list, v_history_list, dist_obj_history_list, p_target, window=8):
    """
    构造近期进展/卡住摘要特征。

    输出维度: [B, 8]
    - progress_to_goal
    - disp_k
    - speed_mean_k
    - collision_depth_mean_k
    - stuck_score
    - progress_efficiency
    - tortuosity
    - heading_align_improvement
    """
    if len(p_history_list) == 0:
        B = p_target.shape[0]
        return torch.zeros((B, 8), device=p_target.device, dtype=p_target.dtype)

    p_now = p_history_list[-1]
    device = p_now.device
    dtype = p_now.dtype

    k = max(1, min(int(window), len(p_history_list)))
    p_prev = p_history_list[-k]

    dist_now = safe_l2_norm(p_target - p_now, dim=-1)
    dist_prev = safe_l2_norm(p_target - p_prev, dim=-1)
    progress_to_goal = dist_prev - dist_now

    disp_k = safe_l2_norm(p_now - p_prev, dim=-1)

    v_tail = torch.stack(v_history_list[-k:], dim=0)
    speed_mean_k = safe_l2_norm(v_tail, dim=-1).mean(dim=0)

    if len(dist_obj_history_list) > 0:
        dist_tail = torch.stack(dist_obj_history_list[-k:], dim=0)
        depth_tail = F.relu(-dist_tail)
        # dist_tail 可能是 [k, B] 或 [k, sub_div, B]，统一压缩到 [B]
        while depth_tail.dim() > 1:
            depth_tail = depth_tail.mean(dim=0)
        collision_depth_mean_k = depth_tail
    else:
        collision_depth_mean_k = torch.zeros_like(progress_to_goal)

    stuck_score = F.softplus((0.3 - disp_k) * 10.0)
    # 效率比: 跑了多少净位移是否真正转化为接近目标
    progress_efficiency = progress_to_goal / (disp_k + 1e-6)

    # 曲折度: 窗口路径长度 / 窗口净位移，越大表示越绕/打转
    p_tail = torch.stack(p_history_list[-k:], dim=0)  # [k, B, 3]
    if k > 1:
        path_len_k = safe_l2_norm(p_tail[1:] - p_tail[:-1], dim=-1).sum(dim=0)
    else:
        path_len_k = torch.zeros_like(disp_k)
    tortuosity = path_len_k / (disp_k + 1e-6)

    v_now = v_history_list[-1]
    target_dir_now = safe_normalize(p_target - p_now, dim=-1)
    v_dir_now = safe_normalize(v_now, dim=-1)
    heading_align_now = (v_dir_now * target_dir_now).sum(dim=-1)

    if len(v_history_list) >= k:
        v_prev = v_history_list[-k]
        target_dir_prev = safe_normalize(p_target - p_prev, dim=-1)
        v_dir_prev = safe_normalize(v_prev, dim=-1)
        heading_align_prev = (v_dir_prev * target_dir_prev).sum(dim=-1)
        heading_align_improvement = heading_align_now - heading_align_prev
    else:
        heading_align_improvement = torch.zeros_like(heading_align_now)

    progress_feat = torch.stack([
        progress_to_goal,
        disp_k,
        speed_mean_k,
        collision_depth_mean_k,
        stuck_score,
        progress_efficiency,
        tortuosity,
        heading_align_improvement,
    ], dim=-1)

    progress_feat = sanitize_tensor(progress_feat, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
    return progress_feat.to(device=device, dtype=dtype)


@torch.no_grad()
def sanitize_module_(module, clamp_value=10.0):
    for p in module.parameters():
        p.data = sanitize_tensor(p.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value).clamp(-clamp_value, clamp_value)


def rotation_matrix_to_rpy_deg(R):
    """Convert rotation matrix to roll-pitch-yaw in degrees (ZYX convention)."""
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]
    r10 = R[..., 1, 0]
    r00 = R[..., 0, 0]

    pitch = torch.asin(torch.clamp(-r20, -1.0, 1.0))
    roll = torch.atan2(r21, r22)
    yaw = torch.atan2(r10, r00)
    return torch.rad2deg(torch.stack([roll, pitch, yaw], dim=-1))


def merge_intervals(intervals, min_gap=1e-4):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + min_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


@torch.no_grad()
def get_collision_wall_patches(points_xyz, walls, drone_radius, segment_len=0.45, contact_eps=0.02):
    if points_xyz.numel() == 0 or walls.numel() == 0:
        return []

    wall_mask = (
        (walls[:, 2] >= 0.1)
        & (walls[:, 2] <= 1.9)
        & (walls[:, 5] > 0.5)
    )
    walls = walls[wall_mask]
    if walls.numel() == 0:
        return []

    centers = walls[:, :3]
    half = walls[:, 3:]
    wall_min = centers - half
    wall_max = centers + half
    axis_is_x = half[:, 0] <= half[:, 1]

    points_expanded = points_xyz.unsqueeze(1)
    nearest = torch.minimum(torch.maximum(points_expanded, wall_min.unsqueeze(0)), wall_max.unsqueeze(0))
    clearance = (nearest - points_expanded).norm(dim=-1) - float(drone_radius)
    contact_steps = torch.nonzero(clearance.min(dim=1).values <= contact_eps, as_tuple=False).flatten().tolist()

    wall_intervals = defaultdict(list)
    for step_idx in contact_steps:
        wall_idx = int(clearance[step_idx].argmin().item())
        if float(clearance[step_idx, wall_idx].item()) > contact_eps:
            continue

        point = points_xyz[step_idx]
        center = centers[wall_idx]
        wall_half = half[wall_idx]

        if bool(axis_is_x[wall_idx]):
            tangent_min = float(center[1] - wall_half[1])
            tangent_max = float(center[1] + wall_half[1])
            tangent_center = min(max(float(point[1]), tangent_min), tangent_max)
        else:
            tangent_min = float(center[0] - wall_half[0])
            tangent_max = float(center[0] + wall_half[0])
            tangent_center = min(max(float(point[0]), tangent_min), tangent_max)

        seg_half = min(0.5 * float(segment_len), 0.5 * (tangent_max - tangent_min))
        seg_start = max(tangent_min, tangent_center - seg_half)
        seg_end = min(tangent_max, tangent_center + seg_half)
        if seg_end - seg_start <= 1e-4:
            continue
        wall_intervals[wall_idx].append((seg_start, seg_end))

    patches = []
    for wall_idx, intervals in wall_intervals.items():
        center = centers[wall_idx]
        wall_half = half[wall_idx]
        for seg_start, seg_end in merge_intervals(intervals, min_gap=0.02):
            if bool(axis_is_x[wall_idx]):
                patches.append({
                    'xy': (float(center[0] - wall_half[0]), seg_start),
                    'width': float(2.0 * wall_half[0]),
                    'height': float(seg_end - seg_start),
                })
            else:
                patches.append({
                    'xy': (seg_start, float(center[1] - wall_half[1])),
                    'width': float(seg_end - seg_start),
                    'height': float(2.0 * wall_half[1]),
                })
    return patches


def compute_overlap_loss_per_step(p_history, sigma=0.5, time_window=10):
    """
    Step-wise 重叠损失计算
    返回: [Batch, Time] (注意: 调用处需要permute)
    """
    # Use squared-distance RBF directly (without sqrt/cdist) so 2nd-order gradients
    # stay well-behaved when trajectory points overlap exactly.
    p_history = p_history.permute(1, 0, 2)  # [B, T, 3]
    n_batch, n_points, _ = p_history.shape
    device = p_history.device
    dtype = p_history.dtype

    time_window = max(0, int(time_window))
    sigma = max(float(sigma), 1e-4)

    if n_points <= 1:
        return torch.zeros((n_batch, n_points), device=device, dtype=dtype)
    # Keep at least one valid long-range pair. Otherwise the mask becomes all-zero
    # and exploration term is permanently zero when time_window ~= rollout length.
    max_effective_window = max(0, n_points - 2)
    time_window = min(time_window, max_effective_window)

    # Pairwise squared distances: [B, T, T]
    pair_delta = p_history[:, :, None, :] - p_history[:, None, :, :]
    sq_dist = (pair_delta * pair_delta).sum(dim=-1)
    sq_dist = sanitize_tensor(sq_dist, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
    inv_two_sigma2 = 0.5 / (sigma * sigma)
    overlap_energy = torch.exp(-sq_dist * inv_two_sigma2)

    indices = torch.arange(n_points, device=device)
    time_diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    mask = (time_diff > time_window).to(dtype=dtype)  # [T, T]

    # Step-wise mean overlap energy with temporal exclusion mask.
    energy_sum = (overlap_energy * mask.unsqueeze(0)).sum(dim=2)  # [B, T]
    mask_sum = mask.sum(dim=1).unsqueeze(0).clamp_min(1.0)  # [1, T]

    return energy_sum / mask_sum


def compute_turn_preference_loss(
    v_history,
    speed_threshold=0.2,
    speed_softness=0.01,
    soft_angle_deg=10.0,
):
    """
    三维速度方向稳定损失。

    相邻方向一致时损失为零，损失随夹角单调增加；小角度区域
    使用二次惩罚，并通过可微低速软掩码降低低速方向噪声的影响。

    输入:
        v_history: [T, B, 3]
    输出:
        loss_turn_seq: [T, B]
    """
    return compute_direction_stability_loss_3d(
        v_history=v_history,
        speed_threshold=speed_threshold,
        speed_softness=speed_softness,
        soft_angle_deg=soft_angle_deg,
    )


def compute_stuck_loss(p_history, collision_depth, stuck_window=15, displacement_threshold=0.3):
    """
    计算卡住惩罚损失

    检测两种卡住状态：
    1. 局部窗口内位移过小
    2. 持续碰撞状态
    """
    T, B, _ = p_history.shape
    device = p_history.device

    loss_stuck = torch.zeros((T, B), device=device)
    if T > stuck_window:
        for t in range(stuck_window, T):
            window_start = t - stuck_window
            displacement = safe_l2_norm(p_history[t] - p_history[window_start], dim=-1)  # [B]
            loss_stuck[t] = F.softplus((displacement_threshold - displacement) * 10.0)

    in_collision = (collision_depth > 0).float()  # [T, B]
    loss_collision_duration = torch.zeros_like(in_collision)

    collision_streak = torch.zeros((B,), device=device)
    for t in range(T):
        collision_streak = collision_streak * in_collision[t] + in_collision[t]
        loss_collision_duration[t] = collision_streak * in_collision[t]

    with torch.no_grad():
        stuck_mask = loss_stuck > 0.5
        stuck_ratio = stuck_mask.float().mean()

    return loss_stuck, loss_collision_duration, stuck_ratio
