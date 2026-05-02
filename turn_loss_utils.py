import torch
from torch.nn import functional as F


def _safe_normalize(x, dim=-1, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, dim=dim, eps=eps)


def _safe_l2_norm(x, dim=-1, keepdim=False, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim) + eps)


def _wrap_angle(delta):
    """Wrap angle to [-pi, pi] to avoid discontinuity at the branch cut."""
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def compute_turn_preference_loss_xy_unit(v_history, speed_threshold=0.2):
    """
    Turn-preference loss with horizontal-plane unit-direction dot product.

    Behavior contract:
    - Only x-y velocity is used to define turn behavior.
    - z-axis velocity changes do not affect turn preference.
    - Direction consistency is computed from unit vectors:
      dot(normalize(v_xy[t]), normalize(v_xy[t-1])).
    - Validity mask is based on horizontal speed magnitudes only.
    """
    T, B, _ = v_history.shape
    device = v_history.device
    dtype = v_history.dtype

    loss_turn_seq = torch.zeros((T, B), device=device, dtype=dtype)
    if T <= 1:
        return loss_turn_seq

    v_xy = v_history[..., :2]  # [T, B, 2]
    v_dir_xy = _safe_normalize(v_xy, dim=-1)
    dir_consistency = (v_dir_xy[1:] * v_dir_xy[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
    loss_core = 0.5 * (dir_consistency + 1.0)

    speed_now = _safe_l2_norm(v_xy[1:], dim=-1)
    speed_prev = _safe_l2_norm(v_xy[:-1], dim=-1)
    valid_mask = ((speed_now > speed_threshold) & (speed_prev > speed_threshold)).to(dtype)

    loss_turn_seq[1:] = loss_core * valid_mask
    return loss_turn_seq


def compute_turn_preference_loss_xy_windowed(
    v_history,
    speed_threshold=0.2,
    window=8,
):
    """
    Turn-preference loss that encourages multi-step cumulative turning.

    Behavior contract:
    - Only x-y velocity is used.
    - Per-step heading change uses wrapped angle:
      dtheta[t] = wrap(yaw[t] - yaw[t-1]).
    - Cumulative turn uses vector-sum magnitude over a fixed recent window:
      |sum(dtheta)|, where signed dtheta keeps turn direction.
    - Low-speed pairs are masked out using horizontal-speed threshold.
    - Output is [T, B], compatible with existing weighted loss map.
    """
    T, B, _ = v_history.shape
    device = v_history.device
    dtype = v_history.dtype

    loss_turn_seq = torch.zeros((T, B), device=device, dtype=dtype)
    if T <= 1:
        return loss_turn_seq

    w = max(1, int(window))
    v_xy = v_history[..., :2]
    v_xy = torch.nan_to_num(v_xy, nan=0.0, posinf=0.0, neginf=0.0)

    yaw = torch.atan2(v_xy[..., 1], v_xy[..., 0])  # [T, B]
    dtheta = _wrap_angle(yaw[1:] - yaw[:-1])  # [T-1, B], signed in [-pi, pi]

    speed_now = _safe_l2_norm(v_xy[1:], dim=-1)
    speed_prev = _safe_l2_norm(v_xy[:-1], dim=-1)
    valid_mask = ((speed_now > speed_threshold) & (speed_prev > speed_threshold)).to(dtype)  # [T-1, B]

    dtheta_masked = dtheta * valid_mask

    # Prefix sums for O(T) window aggregation.
    turn_prefix = torch.cat(
        [torch.zeros((1, B), device=device, dtype=dtype), dtheta_masked.cumsum(dim=0)],
        dim=0,
    )  # [T, B]
    cnt_prefix = torch.cat(
        [torch.zeros((1, B), device=device, dtype=dtype), valid_mask.cumsum(dim=0)],
        dim=0,
    )  # [T, B]

    idx = torch.arange(T, device=device)
    left = torch.clamp(idx - w, min=0)  # [T]

    # Window over step transitions: (left[t], t] on prefix axis.
    turn_sum_signed = turn_prefix[idx] - turn_prefix[left]  # [T, B]
    turn_cnt_raw = cnt_prefix[idx] - cnt_prefix[left]  # [T, B]
    turn_cnt = turn_cnt_raw.clamp_min(1.0)
    mean_turn = turn_sum_signed.abs() / turn_cnt  # [T, B], radians in [0, pi]

    # Encourage larger cumulative turning by minimizing 1 - normalized turn.
    loss_turn_seq = 1.0 - torch.clamp(mean_turn / torch.pi, 0.0, 1.0)
    # Keep t=0 neutral (no valid transition yet).
    loss_turn_seq[0] = 0.0
    # No valid low-speed transitions in window => no contribution.
    loss_turn_seq = loss_turn_seq * (turn_cnt_raw > 0).to(dtype)
    return loss_turn_seq
