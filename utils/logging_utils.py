"""Scalar logging and gradient diagnostic helpers."""

from collections import defaultdict
import math

import torch


def _resolve_map_log_key(map_type_name, map_writers):
    key = str(map_type_name).strip().lower().replace("-", "_")
    return key if key in map_writers else "global"


def _resolve_tb_writer(map_type_name, writer, map_writers):
    key = _resolve_map_log_key(map_type_name, map_writers)
    if key == "global":
        return writer
    return map_writers[key]


def smooth_dict(ori_dict, scaler_q_by_map, map_log_key="global"):
    q = scaler_q_by_map[map_log_key]
    for k, v in ori_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        q[k].append(float(v))


def is_artifact_save_iter(i, args):
    """Use one interval for checkpoints, trajectories, and videos."""
    interval = int(args.artifact_save_interval)
    return interval > 0 and (i + 1) % interval == 0


def is_debug_tb_step(step, args):
    interval = int(args.debug_tb_interval)
    return interval > 0 and (step % interval == 0 or step == args.num_iters)


def get_grad_stats(module):
    total_sq = 0.0
    max_abs = 0.0
    nonfinite_cnt = 0
    grad_elem_cnt = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        finite_mask = torch.isfinite(g)
        nonfinite_cnt += int((~finite_mask).sum().item())
        if finite_mask.any():
            g_finite = g[finite_mask]
            total_sq += float((g_finite * g_finite).sum().item())
            max_abs = max(max_abs, float(g_finite.abs().max().item()))
        grad_elem_cnt += g.numel()
    global_norm = math.sqrt(total_sq)
    return global_norm, max_abs, nonfinite_cnt, grad_elem_cnt


def get_grad_norm_from_grads(grads):
    total_sq = 0.0
    nonfinite_cnt = 0
    grad_elem_cnt = 0
    for g in grads:
        if g is None:
            continue
        g = g.detach()
        finite_mask = torch.isfinite(g)
        nonfinite_cnt += int((~finite_mask).sum().item())
        if finite_mask.any():
            g_finite = g[finite_mask]
            total_sq += float((g_finite * g_finite).sum().item())
        grad_elem_cnt += g.numel()
    return math.sqrt(total_sq), nonfinite_cnt, grad_elem_cnt


def scale_scalar_objective(x, target_mag=10.0, eps=1e-6):
    """Rescale scalar objective by detached magnitude to avoid gradient blow-up."""
    if x is None:
        return x
    denom = x.detach().abs().clamp_min(eps)
    return x * (float(target_mag) / denom)


def _diag_should_log(iter_idx, args):
    return args.diag_interval > 0 and (iter_idx % args.diag_interval == 0)


def _diag_grad_meta(x):
    if x is None:
        return "None"
    gfn = type(x.grad_fn).__name__ if getattr(x, 'grad_fn', None) is not None else "None"
    return f"requires_grad={x.requires_grad}, is_leaf={x.is_leaf}, grad_fn={gfn}"


def _diag_tensor_finite(tag, x, iter_idx):
    if x is None:
        print(f"[DIAG iter={iter_idx}] {tag}: None")
        return
    with torch.no_grad():
        xd = x.detach()
        finite_mask = torch.isfinite(xd)
        finite_cnt = int(finite_mask.sum().item())
        total_cnt = int(xd.numel())
        nonfinite_cnt = total_cnt - finite_cnt
        if finite_cnt > 0:
            vals = xd[finite_mask]
            vmin = float(vals.min().item())
            vmax = float(vals.max().item())
        else:
            vmin = float('nan')
            vmax = float('nan')
    print(
        f"[DIAG iter={iter_idx}] {tag}: finite={finite_cnt}/{total_cnt}, "
        f"nonfinite={nonfinite_cnt}/{total_cnt}, min={vmin:.6g}, max={vmax:.6g}"
    )


def _diag_grad_tuple_to_params(tag, grad_tuple, params, iter_idx, retain_graph=True):
    params = list(params)
    total_params = len(params)
    if total_params == 0:
        print(f"[DIAG iter={iter_idx}] {tag}: None=0/0, NonZero=0/0, Norm=0.000000, NonFinite=0/0")
        return

    if grad_tuple is None:
        print(f"[DIAG iter={iter_idx}] {tag}: None={total_params}/{total_params}, NonZero=0/{total_params}, Norm=0.000000, NonFinite=0/0")
        return

    grads = [g for g in grad_tuple if g is not None]
    if len(grads) == 0:
        print(f"[DIAG iter={iter_idx}] {tag}: None={total_params}/{total_params}, NonZero=0/{total_params}, Norm=0.000000, NonFinite=0/0")
        return

    probe = None
    for g in grads:
        s = g.sum()
        probe = s if probe is None else (probe + s)

    try:
        mapped = torch.autograd.grad(
            probe,
            params,
            allow_unused=True,
            retain_graph=retain_graph,
            create_graph=False,
        )
    except Exception as e:
        print(f"[DIAG iter={iter_idx}] {tag}: grad-check failed: {e}")
        return

    none_cnt = sum(g is None for g in mapped)
    nonzero_cnt = 0
    total_sq = 0.0
    nonfinite = 0
    total_elems = 0
    for g in mapped:
        if g is None:
            continue
        gd = g.detach()
        finite_mask = torch.isfinite(gd)
        nonfinite += int((~finite_mask).sum().item())
        total_elems += gd.numel()
        if finite_mask.any():
            vals = gd[finite_mask]
            total_sq += float((vals * vals).sum().item())
            if float(vals.abs().sum().item()) > 1e-12:
                nonzero_cnt += 1

    print(
        f"[DIAG iter={iter_idx}] {tag}: None={none_cnt}/{total_params}, "
        f"NonZero={nonzero_cnt}/{total_params}, Norm={math.sqrt(total_sq):.6f}, "
        f"NonFinite={nonfinite}/{total_elems}"
    )


def _diag_output_to_params(tag, output, params, iter_idx, retain_graph=True):
    params = list(params)
    total_params = len(params)
    if total_params == 0:
        print(f"[DIAG iter={iter_idx}] {tag}: norm=0.000000, NonFinite=0/0")
        return
    try:
        grads = torch.autograd.grad(
            output,
            params,
            allow_unused=True,
            retain_graph=retain_graph,
            create_graph=False,
        )
    except Exception as e:
        print(f"[DIAG iter={iter_idx}] {tag}: grad-check failed: {e}")
        return

    norm, nonfinite, grad_elems = get_grad_norm_from_grads(grads)
    print(f"[DIAG iter={iter_idx}] {tag}: norm={norm:.6f}, NonFinite={nonfinite}/{grad_elems}")


def _diag_output_to_params_count(tag, output, params, iter_idx, retain_graph=True):
    params = list(params)
    total_params = len(params)
    if total_params == 0:
        print(f"[DIAG iter={iter_idx}] {tag}: None=0/0, NonZero=0/0")
        return
    try:
        grads = torch.autograd.grad(
            output,
            params,
            allow_unused=True,
            retain_graph=retain_graph,
            create_graph=False,
        )
    except Exception as e:
        print(f"[DIAG iter={iter_idx}] {tag}: grad-check failed: {e}")
        return

    none_cnt = sum(g is None for g in grads)
    nonzero_cnt = 0
    for g in grads:
        if g is None:
            continue
        if float(g.detach().abs().sum().item()) > 1e-12:
            nonzero_cnt += 1
    print(f"[DIAG iter={iter_idx}] {tag}: None={none_cnt}/{total_params}, NonZero={nonzero_cnt}/{total_params}")


def _grad_or_none_tuple(loss, params, create_graph=True, retain_graph=True):
    params = tuple(params)
    if not getattr(loss, "requires_grad", False):
        return tuple(None for _ in params)
    return torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        allow_unused=True,
        retain_graph=retain_graph,
    )
