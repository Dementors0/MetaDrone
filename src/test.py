import math
import os
import sys
import torch
import quadsim_cuda

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from env import run as custom_cuda_run, run_torch, update_state_vec_torch


class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply

R = torch.randn((64, 3, 3), dtype=torch.double, device='cuda')
dg = torch.randn((64, 3), dtype=torch.double, device='cuda')
z_drag_coef = torch.randn((64, 1), dtype=torch.double, device='cuda')
drag_2 = torch.randn((64, 2), dtype=torch.double, device='cuda')
pitch_ctl_delay = torch.randn((64, 1), dtype=torch.double, device='cuda')
g_std = torch.tensor([[0, 0, -9.80665]], dtype=torch.double, device='cuda')
act_pred = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
act = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
p = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
v = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
v_wind = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
a = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)

grad_decay = 0.4
ctl_dt = 1/15

def run_forward_pytorch(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt):
    alpha = torch.exp(-pitch_ctl_delay * ctl_dt)
    act_next = act_pred * (1 - alpha) + act * alpha
    # dg = dg * math.sqrt(1 - ctl_dt) + torch.randn_like(dg) * 0.2 * math.sqrt(ctl_dt)
    v_fwd_s, v_left_s, v_up_s = (v.add(-v_wind)[:, None] @ R).unbind(-1)
    # 0.047 = (4*rotor_drag_coefficient*motor_velocity_real) / sqrt(9.8)
    drag = drag_2[:, :1] * (v_fwd_s.abs() * v_fwd_s * R[..., 0] + v_left_s.abs() * v_left_s * R[..., 1] + v_up_s.abs() * v_up_s * R[..., 2] * z_drag_coef)
    drag += drag_2[:, 1:] * (v_fwd_s * R[..., 0] + v_left_s * R[..., 1] + v_up_s * R[..., 2] * z_drag_coef)
    a_next = act_next + dg - drag
    p_next = g_decay(p, grad_decay ** ctl_dt) + v * ctl_dt + 0.5 * a * ctl_dt**2
    v_next = g_decay(v, grad_decay ** ctl_dt) + (a + a_next) / 2 * ctl_dt
    return act_next, p_next, v_next, a_next

act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, 0)

_act_next, _p_next, _v_next, _a_next = run_forward_pytorch(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt)

assert torch.allclose(act_next, _act_next)
assert torch.allclose(a_next, _a_next)
assert torch.allclose(p_next, _p_next)
assert torch.allclose(v_next, _v_next)

d_act_next = torch.randn_like(act_next)
d_p_next = torch.randn_like(p_next)
d_v_next = torch.randn_like(v_next)
d_a_next = torch.randn_like(a_next)

torch.autograd.backward(
    (_act_next, _p_next, _v_next, _a_next),
    (d_act_next, d_p_next, d_v_next, d_a_next),
)

d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next, grad_decay, ctl_dt)

assert torch.allclose(d_act_pred, act_pred.grad)
assert torch.allclose(d_act, act.grad)
assert torch.allclose(d_p, p.grad)
assert torch.allclose(d_v, v.grad)
assert torch.allclose(d_a, a.grad)


# --- Path-level higher-order probe ---
# Fallback path: should support second-order gradients.
w = torch.randn((1,), dtype=torch.double, device='cuda', requires_grad=True)
act_pred_meta = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True) + w
act_meta = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
p_meta = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
v_meta = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
a_meta = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
v_wind_meta = torch.randn((64, 3), dtype=torch.double, device='cuda')
R_meta = torch.randn((64, 3, 3), dtype=torch.double, device='cuda')

act_1, p_1, v_1, a_1 = run_torch(
    R_meta, dg, z_drag_coef, drag_2, pitch_ctl_delay,
    act_pred_meta, act_meta, p_meta, v_meta, v_wind_meta, a_meta,
    grad_decay, ctl_dt, 0.5,
)
alpha_yaw = torch.exp(-pitch_ctl_delay * ctl_dt)
v_pred_meta = torch.randn((64, 3), dtype=torch.double, device='cuda')
R_1 = update_state_vec_torch(R_meta, act_1, v_pred_meta, alpha_yaw, 2)
act_2_pred = 0.5 * act_pred_meta + R_1[:, :, 2]
_, p_2, v_2, _ = run_torch(
    R_1, dg, z_drag_coef, drag_2, pitch_ctl_delay,
    act_2_pred, act_1, p_1, v_1, v_wind_meta, a_1,
    grad_decay, ctl_dt, 0.5,
)
meta_loss = (p_2.pow(2).mean() + v_2.pow(2).mean())
inner_grad = torch.autograd.grad(meta_loss, act_pred_meta, create_graph=True, allow_unused=False)[0]
second_grad = torch.autograd.grad(inner_grad.sum(), w, allow_unused=True)[0]
assert second_grad is not None
assert torch.isfinite(second_grad).all()

# CUDA custom backward path: still first-order only for higher-order chain.
w2 = torch.randn((1,), dtype=torch.double, device='cuda', requires_grad=True)
act_pred_cuda = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True) + w2
act_cuda = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
p_cuda = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
v_cuda = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
a_cuda = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
act_next_cuda, p_next_cuda, v_next_cuda, a_next_cuda = custom_cuda_run(
    R_meta, dg, z_drag_coef, drag_2, pitch_ctl_delay,
    act_pred_cuda, act_cuda, p_cuda, v_cuda, v_wind_meta, a_cuda,
    grad_decay, ctl_dt, 0.5,
)
loss_cuda = p_next_cuda.pow(2).mean() + v_next_cuda.pow(2).mean() + a_next_cuda.pow(2).mean()
inner_grad_cuda = torch.autograd.grad(loss_cuda, act_pred_cuda, create_graph=True, allow_unused=True)[0]
second_grad_cuda = None
if inner_grad_cuda is not None:
    try:
        second_grad_cuda = torch.autograd.grad(inner_grad_cuda.sum(), w2, allow_unused=True)[0]
    except RuntimeError:
        second_grad_cuda = None
assert second_grad_cuda is None
