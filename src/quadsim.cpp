#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void render_cuda(
    torch::Tensor canvas,
    torch::Tensor flow,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor R_old,
    torch::Tensor pos,
    torch::Tensor pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan);

void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan);

void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group);

torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia);

std::vector<torch::Tensor> update_state_vec_v2_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor heading_ref,
    torch::Tensor yaw_rate,
    torch::Tensor yaw_rate_cmd,
    torch::Tensor alpha,
    float ctl_dt,
    float yaw_rate_max);

std::vector<torch::Tensor> run_forward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor act_pred,
    torch::Tensor act,
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor a,
    float ctl_dt,
    float airmode_av2a);

std::vector<torch::Tensor> run_backward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor act_next,
    torch::Tensor _d_act_next,
    torch::Tensor d_p_next,
    torch::Tensor d_v_next,
    torch::Tensor _d_a_next,
    float grad_decay,
    float ctl_dt);

// Explicitly named wrappers for API clarity.
std::vector<torch::Tensor> run_forward_first_order_cuda(
        torch::Tensor R,
        torch::Tensor dg,
        torch::Tensor z_drag_coef,
        torch::Tensor drag_2,
        torch::Tensor pitch_ctl_delay,
        torch::Tensor act_pred,
        torch::Tensor act,
        torch::Tensor p,
        torch::Tensor v,
        torch::Tensor v_wind,
        torch::Tensor a,
        float ctl_dt,
        float airmode_av2a) {
    return run_forward_cuda(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            act_pred, act, p, v, v_wind, a, ctl_dt, airmode_av2a);
}

std::vector<torch::Tensor> run_backward_first_order_cuda(
        torch::Tensor R,
        torch::Tensor dg,
        torch::Tensor z_drag_coef,
        torch::Tensor drag_2,
        torch::Tensor pitch_ctl_delay,
        torch::Tensor v,
        torch::Tensor v_wind,
        torch::Tensor act_next,
        torch::Tensor _d_act_next,
        torch::Tensor d_p_next,
        torch::Tensor d_v_next,
        torch::Tensor _d_a_next,
        float grad_decay,
        float ctl_dt) {
    return run_backward_cuda(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay,
            v, v_wind, act_next, _d_act_next, d_p_next, d_v_next, _d_a_next,
            grad_decay, ctl_dt);
}

torch::Tensor update_state_vec_first_order_cuda(
        torch::Tensor R,
        torch::Tensor a_thr,
        torch::Tensor v_pred,
        torch::Tensor alpha,
        float yaw_inertia) {
    return update_state_vec_cuda(R, a_thr, v_pred, alpha, yaw_inertia);
}

std::vector<torch::Tensor> update_state_vec_v2_first_order_cuda(
        torch::Tensor R,
        torch::Tensor a_thr,
        torch::Tensor heading_ref,
        torch::Tensor yaw_rate,
        torch::Tensor yaw_rate_cmd,
        torch::Tensor alpha,
        float ctl_dt,
        float yaw_rate_max) {
    return update_state_vec_v2_cuda(
            R, a_thr, heading_ref, yaw_rate, yaw_rate_cmd, alpha,
            ctl_dt, yaw_rate_max);
}

// C++ interface

// // NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
// #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// void render(
//     torch::Tensor canvas,
//     torch::Tensor nearest_pt,
//     torch::Tensor balls,
//     torch::Tensor cylinders,
//     torch::Tensor voxels,
//     torch::Tensor Rt) {
//   CHECK_INPUT(input);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(bias);
//   CHECK_INPUT(old_h);
//   CHECK_INPUT(old_cell);

//   return render_cuda(input, weights, bias, old_h, old_cell);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render", &render_cuda, "render (CUDA)");
  m.def("find_nearest_pt", &find_nearest_pt_cuda, "find_nearest_pt (CUDA)");
  m.def("update_state_vec", &update_state_vec_cuda, "update_state_vec (CUDA)");
  m.def("update_state_vec_v2", &update_state_vec_v2_cuda, "update_state_vec_v2 explicit yaw-rate (CUDA)");
  m.def("run_forward", &run_forward_cuda, "run_forward_cuda (CUDA)");
  m.def("run_backward", &run_backward_cuda, "run_backward_cuda (CUDA)");
  m.def("rerender_backward", &rerender_backward_cuda, "rerender_backward_cuda (CUDA)");

    // Explicit first-order aliases for training scripts to avoid API ambiguity.
    m.def("run_forward_first_order", &run_forward_first_order_cuda, "run_forward first-order (CUDA)");
    m.def("run_backward_first_order", &run_backward_first_order_cuda, "run_backward first-order (CUDA)");
    m.def("update_state_vec_first_order", &update_state_vec_first_order_cuda, "update_state_vec first-order (CUDA)");
    m.def("update_state_vec_v2_first_order", &update_state_vec_v2_first_order_cuda, "update_state_vec_v2 first-order (CUDA)");
}
