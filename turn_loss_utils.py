"""Differentiable three-dimensional direction-stability loss."""

import math

import torch


def compute_direction_stability_loss_3d(
    v_history,
    speed_threshold=0.2,
    speed_softness=0.01,
    soft_angle_deg=10.0,
):
    """
    Penalize changes between consecutive three-dimensional velocity directions.

    The angular penalty is quadratic for small angles and linear for larger
    angles. A differentiable speed gate suppresses unreliable directions at
    low speed without cutting the gradient path through the velocity.

    Args:
        v_history: Velocity history with shape [T, B, 3].
        speed_threshold: Center speed of the differentiable low-speed gate.
        speed_softness: Width of the sigmoid transition around the threshold.
        soft_angle_deg: Boundary between quadratic and linear angular penalty.

    Returns:
        Per-step direction-stability loss with shape [T, B].
    """
    T, B, _ = v_history.shape
    device = v_history.device
    dtype = v_history.dtype

    first_step_zero = torch.zeros((1, B), device=device, dtype=dtype)
    if T <= 1:
        return first_step_zero[:T]

    velocity = torch.nan_to_num(
        v_history,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    direction = velocity / speed.unsqueeze(-1).clamp_min(1e-6)

    direction_prev = direction[:-1]
    direction_now = direction[1:]

    dot = (direction_prev * direction_now).sum(dim=-1).clamp(-1.0, 1.0)
    cross_norm = torch.linalg.vector_norm(
        torch.cross(direction_prev, direction_now, dim=-1),
        dim=-1,
    )
    angle = torch.atan2(cross_norm, dot)

    beta = math.radians(float(soft_angle_deg))
    beta = min(max(beta, 1e-6), math.pi - 1e-6)
    small_angle_loss = 0.5 * angle.square() / beta
    large_angle_loss = angle - 0.5 * beta
    angular_loss = torch.where(
        angle <= beta,
        small_angle_loss,
        large_angle_loss,
    )
    angular_loss = angular_loss / (math.pi - 0.5 * beta)

    softness = max(float(speed_softness), 1e-6)
    gate_prev = torch.sigmoid((speed[:-1] - float(speed_threshold)) / softness)
    gate_now = torch.sigmoid((speed[1:] - float(speed_threshold)) / softness)
    speed_gate = gate_prev * gate_now

    return torch.cat([first_step_zero, angular_loss * speed_gate], dim=0)
