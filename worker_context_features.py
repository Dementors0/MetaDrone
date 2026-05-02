import torch


WORKER_CONTEXT_LAGS = (1, 2, 4, 8, 16)
WORKER_CONTEXT_FEATURE_DIM = 4 + len(WORKER_CONTEXT_LAGS) * 4


def _safe_l2_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim) + eps)


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = _safe_l2_norm(x, dim=dim, keepdim=True, eps=eps)
    return x / n.clamp_min(eps)


def extract_worker_context_features(
    p_history_list,
    p_target: torch.Tensor,
    R_current: torch.Tensor,
    lags=WORKER_CONTEXT_LAGS,
):
    """
    Build deployable local context features for Worker/LGN.

    Output shape: [B, 24] with default lags [1, 2, 4, 8, 16].
    Layout:
    - goal_dir_body (3)
    - goal_dist_log (1)
    - delta_p_body[k] for each lag (5 * 3)
    - delta_goal_dist[k] for each lag (5)
    """
    B = p_target.shape[0]
    device = p_target.device
    dtype = p_target.dtype

    if len(p_history_list) == 0:
        return torch.zeros((B, WORKER_CONTEXT_FEATURE_DIM), device=device, dtype=dtype)

    p_now = p_history_list[-1]
    n_hist = len(p_history_list)

    goal_vec_world = p_target - p_now
    goal_dist = _safe_l2_norm(goal_vec_world, dim=-1)
    goal_dir_world = _safe_normalize(goal_vec_world, dim=-1)
    goal_dir_body = torch.squeeze(goal_dir_world[:, None] @ R_current, 1)
    goal_dist_log = torch.log1p(goal_dist).unsqueeze(-1)

    delta_p_body_list = []
    delta_goal_dist_list = []
    for lag in lags:
        idx_prev = max(0, n_hist - 1 - int(lag))
        p_prev = p_history_list[idx_prev]

        # Breadcrumb semantics: where historical point lies in current body frame.
        delta_p_world = p_prev - p_now
        delta_p_body = torch.squeeze(delta_p_world[:, None] @ R_current, 1)
        delta_p_body_list.append(delta_p_body)

        dist_prev = _safe_l2_norm(p_target - p_prev, dim=-1)
        delta_goal_dist = (dist_prev - goal_dist).unsqueeze(-1)
        delta_goal_dist_list.append(delta_goal_dist)

    feat = torch.cat(
        [goal_dir_body, goal_dist_log] + delta_p_body_list + delta_goal_dist_list,
        dim=-1,
    )
    feat = torch.nan_to_num(feat, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
    return feat.to(device=device, dtype=dtype)
