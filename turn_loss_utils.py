import torch
from torch.nn import functional as F


def _safe_normalize(x, dim=-1, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, dim=dim, eps=eps)


def _safe_l2_norm(x, dim=-1, keepdim=False, eps=1e-6):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim) + eps)


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
