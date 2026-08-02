#9.2.7
# 增加感知与 LGN 动态参考方向
# LGN 输出方向直接指导真实速度；Worker 直接输出三维加速度动作，不再预测速度
# 三种地图的 log 分开，去掉低速保持机头方向修正
#变速偏好自己决定速度


import argparse
import math
from collections import defaultdict
import os
import datetime
import json
import sys
import matplotlib

matplotlib.use('Agg', force=True)

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("[SDP] flash=False, mem_efficient=False, math=True (for higher-order gradients)")

try:
    from env_multi import Env
except ModuleNotFoundError:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from env_multi import Env
try:
    from potential_map_utils import PotentialMapCache, query_potential_guidance
except ModuleNotFoundError:
    PotentialMapCache = None
    query_potential_guidance = None
from env import probe_update_state_vec_common_upstream
try:
    from WorkNet_transformer import WorkNet
    from LossGenNet_transformer import LossGenNet
except ModuleNotFoundError:
    from WorkNet import WorkNet
    from LossGenNet import LossGenNet
from worker_context_features import (
    WORKER_CONTEXT_FEATURE_DIM,
    extract_worker_context_features,
)
from utils.io_utils import (
    create_unique_experiment_dir,
    load_compatible_checkpoint,
    sync_multi_pub_to_checkpoint_dir,
)
from utils.logging_utils import (
    _diag_grad_meta,
    _diag_grad_tuple_to_params,
    _diag_output_to_params,
    _diag_output_to_params_count,
    _diag_should_log,
    _diag_tensor_finite,
    _grad_or_none_tuple,
    _resolve_map_log_key,
    _resolve_tb_writer,
    get_grad_norm_from_grads,
    get_grad_stats,
    is_artifact_save_iter,
    scale_scalar_objective,
    smooth_dict,
)
from utils.map_utils import (
    _align_env_goal_planes_to_precomputed_map,
    _build_precomputed_map_type_indices,
    _precomputed_curriculum_stage,
    _select_precomputed_curriculum_map,
)
from utils.planner_utils import (
    GlobalPlanner,
    configure_planner_pool,
    compute_global_guidance_meta_loss,
    compute_lgn_potential_vref_sync_loss,
)
from utils.rollout_utils import unrolled_meta_rollout
from utils.tensor_utils import (
    build_yaw_frame,
    compute_arrival_reward,
    compute_heading_reference,
    compute_overlap_loss_per_step,
    compute_stuck_loss,
    compute_turn_preference_loss,
    compute_velocity_heading_command,
    decode_worker_action,
    extract_depth_geometry_features,
    extract_progress_features,
    rotation_matrix_to_rpy_deg,
    safe_l2_norm,
    safe_normalize,
    sanitize_module_,
    sanitize_tensor,
)
from utils.visualization_utils import save_cached_viz_record, snapshot_env_for_viz


########### 1. 参数配置 ##########
parser = argparse.ArgumentParser()
parser.add_argument('--resume_worker', default="", help='Path to pretrained worker model')
parser.add_argument('--resume_lgn', default="", help='Path to pretrained lgn model')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_iters', type=int, default=20000)

# [优化策略参数]
parser.add_argument('--lgn_steps', type=int, default=1)
parser.add_argument('--worker_steps', type=int, default=1)

# 基础物理参数
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--scene_scale', type=float, default=0.5,#調節環境大小
                    help='Global scene size scale for obstacle field extent and spawn area')
parser.add_argument('--obstacle_count_scale', type=float, default=0.3,#調節障礙物數量
                    help='Global multiplier for obstacle counts (balls/voxels/cylinders)')
parser.add_argument('--soft_speed_limit_softness', type=float, default=0.05,#物理軟限速的平滑度（越小越接近硬截斷，越大越平滑）。
                    help='Softness for physical speed cap in env._apply_speed_limit (smaller = harder cap)')
parser.add_argument('--max_speed_ceiling', type=float, default=15.0,#env.max_speed 的上限值（軟限速的速度天花板）。
                    help='Upper bound of env max_speed used by soft speed limiter')
parser.add_argument('--hard_vpred_clip', type=float, default=20.0,#env.run 中 v_pred 的硬截斷閾值（原本固定 20）。
                    help='Hard clip magnitude for v_pred in env.run')
parser.add_argument('--hard_speed_clip', type=float, default=30.0,#env.run 中 v_free/self.v 的硬截斷閾值（原本固定 30）。
                    help='Hard clip magnitude for velocity tensors (v_free/self.v) in env.run')
parser.add_argument('--start_goal_plane_y_abs', type=float, default=25,#調節起點和終點的位置
                    help='Start/goal planes are set to +Y and -Y using this absolute value')
parser.add_argument('--attitude_model', type=str, default='v2', choices=['legacy', 'v2'],
                    help='legacy uses goal-projected heading; v2 uses explicit yaw-rate dynamics')
parser.add_argument('--yaw_rate_max_deg', type=float, default=150.0)
parser.add_argument('--coef_yaw_cmd', type=float, default=0.2)
parser.add_argument('--coef_yaw_smooth', type=float, default=0.01)

# ===================== v2 机头跟踪参数（中文详解） =====================
# 这些参数只在 attitude_model='v2' 的显式 yaw 动力学下生效，用于控制：
# 1) 机头追踪真实速度方向的“转头力度”；
# 2) 是否允许网络输出的 yaw_rate 作为小幅残差；
# 3) 对“倒飞/侧后飞”的惩罚强度。
#
# 调参建议（经验）：
# - 机头转得慢：增大 heading_yaw_kp 或 yaw_rate_max_deg。
# - 低速抖头：增大 heading_min_speed，或降低 heading_yaw_kp。
# - 不希望倒飞：增大 coef_backward_penalty，必要时把 backward_cos_limit 调高到 0.1~0.3。
# - 想保留少量学习型偏航微调：把 heading_residual_scale 从 0.0 提到 0.1/0.2。
parser.add_argument('--heading_track_mode', type=str, default='actual_v',
                    choices=['actual_v'],
                    help='Compatibility option; Worker no longer predicts velocity, so heading tracks actual velocity')
parser.add_argument('--heading_min_speed', type=float, default=0.25,
                    help='Minimum horizontal speed for heading-alignment safety metrics/losses')
parser.add_argument('--heading_yaw_kp', type=float, default=4.0,
                    help='机头跟踪比例增益：yaw_rate_rule = heading_yaw_kp * yaw_error。越大转头越积极，但过大可能引起振荡')
parser.add_argument('--heading_residual_scale', type=float, default=0.0,
                    help='网络残差偏航比例：最终 yaw_rate = 规则项 + scale*网络项。0 表示仅规则跟踪，关闭学习残差')
parser.add_argument('--backward_cos_limit', type=float, default=0.0,
                    help='倒飞判定余弦阈值：当 cos(heading,vel) 低于该值时施加倒飞惩罚。0 表示主要惩罚超过90度的后向飞行')
parser.add_argument('--coef_heading_align', type=float, default=0.05,
                    help='机头-速度对齐损失权重（固定项，不经LGN调权）。增大可强化“机头朝着运动方向”')
parser.add_argument('--coef_backward_penalty', type=float, default=0.2,
                    help='倒飞惩罚权重（固定项，不经LGN调权）。增大可更强抑制“摄像头朝前但机体向后飞”')
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--lgn_timesteps', type=int, default=150,
                    help='Rollout steps used in LGN phase; smaller value reduces 2nd-order gradient memory')
parser.add_argument('--exploration_time_window', type=int, default=1,
                    help='Look-back gap for exploration overlap loss; effective window is auto-clipped to keep valid long-range pairs')
parser.add_argument('--turn_speed_threshold', type=float, default=0.2,
                    help='Center speed (m/s) of the differentiable low-speed gate for turn loss')
parser.add_argument('--turn_speed_softness', type=float, default=0.01,
                    help='Sigmoid transition width (m/s) of the low-speed gate for turn loss')
parser.add_argument('--turn_soft_angle_deg', type=float, default=10.0,
                    help='Quadratic-to-linear angle boundary (degrees) for 3D turn loss')
parser.add_argument('--detach_interval', type=int, default=12,
                    help='Detach temporal memory every N steps to limit graph depth (<=0 disables)')
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--goal_radius', type=float, default=1.0,
                    help='Episode terminates when all drones are within this radius of their goal')
parser.add_argument('--meta_arrival_reward_radius', type=float, default=0.5,
                    help='Goal-ball radius where arrival reward starts reducing meta loss')
parser.add_argument('--meta_arrival_reward_weight', type=float, default=1.0,
                    help='Weight of the arrival reward subtracted from meta loss')
parser.add_argument('--maze_update_interval', type=int, default=50,
                    help='Regenerate maze every N iterations; drone-only reset in between for stable LGN signal')

# Transformer memory参数
parser.add_argument('--worker_max_seq_len', type=int, default=32)
parser.add_argument('--lgn_max_seq_len', type=int, default=32)

# 环境Flag
parser.add_argument('--single', default=True, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
# [开关1] U 型局部最优陷阱地图开关（默认: 关闭）
# - 默认行为：不传任何参数时 include_u_local_optimum=False。
# - 显式开启：--include_u_local_optimum
# - 关闭陷阱：--no_include_u_local_optimum（地图为 hard/easy/easy 三块随机重排，且不再包含 U 区）
# - 说明：这两个参数写在同一行时，以最后一个为准（argparse store_true/store_false 同目标变量）。
parser.add_argument('--include_u_local_optimum', dest='include_u_local_optimum', action='store_true',
                    help='Include U-shaped local-optimum trap region in three-zone map')
parser.add_argument('--no_include_u_local_optimum', dest='include_u_local_optimum', action='store_false',
                    help='Disable U-shaped trap region and use shuffled hard/easy/easy region order')
parser.set_defaults(include_u_local_optimum=False)
# [开关1.1] 两分区紧凑地图开关（默认: 关闭）
# - 默认行为：不传任何参数时 compact_two_zone_map=False，保持当前三分区地图逻辑不变。
# - 显式开启：--compact_two_zone_map（仅保留 easy/hard 两种地图，Y 向尺寸缩小，起终点平面随之调整）
# - 显式关闭：--no_compact_two_zone_map
# - 说明：与 include_u_local_optimum 共存时，开启紧凑两分区会优先使用两分区布局（不再包含 U 区）。
parser.add_argument('--compact_two_zone_map', dest='compact_two_zone_map', action='store_true',
                    help='Use compact two-zone map (easy+hard only), with smaller map and adjusted start/goal planes')
parser.add_argument('--no_compact_two_zone_map', dest='compact_two_zone_map', action='store_false',
                    help='Use default map layout (current behavior)')
parser.set_defaults(compact_two_zone_map=True)
# [开关2] 墙壁物理反馈开关（默认: 关闭）
# - 默认行为：不传任何参数时 wall_physical_feedback=False，采用自由运动结果（当前代码行为）。
# - 开启反馈：--wall_physical_feedback（启用软接触反馈，修正穿墙/贴墙时的位置与速度）
# - 显式关闭：--no_wall_physical_feedback
# - 典型组合：
#   1) 保持当前基线：不加这两个开关（或显式 --include_u_local_optimum --no_wall_physical_feedback）
#   2) 仅去掉 U 陷阱：--no_include_u_local_optimum
#   3) 仅加墙体反馈：--wall_physical_feedback
#   4) 同时去陷阱+加反馈：--no_include_u_local_optimum --wall_physical_feedback
parser.add_argument('--wall_physical_feedback', dest='wall_physical_feedback', action='store_true',
                    help='Enable wall-contact physical feedback correction in env dynamics')
parser.add_argument('--no_wall_physical_feedback', dest='wall_physical_feedback', action='store_false',
                    help='Disable wall-contact physical feedback and use free-motion result')
parser.set_defaults(wall_physical_feedback=False)

# 学习率
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--lgn_lr', type=float, default=2e-4)
parser.add_argument('--inner_lr', type=float, default=5e-4,
                    help='Inner loop LR for differentiable worker update in LGN phase')
parser.add_argument('--inner_steps', type=int, default=1,
                    help='Number of differentiable inner SGD steps (unrolled bilevel)')
parser.add_argument('--inner_grad_clip', type=float, default=10.0,
                    help='Global-norm cap for differentiable inner gradients to stabilize second-order chain')
parser.add_argument('--exp_name', type=str, default="default", help="Extra tag for experiment")

# 避障/碰撞超参
parser.add_argument('--avoid_safe_margin', type=float, default=0.35,
                    help='Proxy avoidance rises smoothly inside this clearance to walls')
parser.add_argument('--lgn_output_temperature', type=float, default=1.0,
                    help='Compatibility arg (currently not used): LGN weights are constrained to be non-negative')
parser.add_argument('--lgn_weight_floor', type=float, default=0.01,
                    help='Compatibility arg (unused): no extra floor is applied beyond non-negative constraint')
parser.add_argument('--lgn_weight_ceiling', type=float, default=100.0,
                    help='Compatibility arg (unused): no ceiling constraint is applied to LGN weights')
parser.add_argument('--lgn_potential_vref_weight', type=float, default=0.5,
                    help='Weight for LGN potential-field vref alignment loss (LGN phase only)')
parser.add_argument('--speed_goal_slow_dist', type=float, default=2.5,
                    help='Distance-to-goal (m) where speed target starts linearly reducing to prevent straight-line rushing')
parser.add_argument('--meta_coll_soft_weight', type=float, default=5.0,
                    help='Soft collision term weight in meta loss')
parser.add_argument('--meta_coll_hard_weight', type=float, default=40.0,
                    help='Hard penetration-depth penalty weight in meta loss')
parser.add_argument('--meta_coll_event_weight', type=float, default=80.0,
                    help='Episode-level collision event penalty weight in meta loss')
parser.add_argument('--meta_coll_event_temp', type=float, default=80.0,
                    help='Sharpness for differentiable episode collision event penalty (sigmoid temperature)')
parser.add_argument('--meta_coll_event_threshold', type=float, default=0.01,
                    help='Penetration-depth threshold (m) where differentiable collision-event penalty turns on')
parser.add_argument('--speed_near_obs_floor', type=float, default=0.05,
                    help='Minimum speed factor near obstacles in adaptive speed target (lower = stronger braking)')
parser.add_argument('--stuck_loss_weight', type=float, default=2.0,
                    help='Weight for local displacement-based stuck penalty')
parser.add_argument('--stuck_window', type=int, default=15,
                    help='Window size for stuck detection (steps)')
parser.add_argument('--stuck_displacement_threshold', type=float, default=0.3,
                    help='Minimum displacement in window before stuck penalty activates (m)')
parser.add_argument('--collision_duration_weight', type=float, default=10.0,
                    help='Weight for collision-duration diagnostic penalty')
parser.add_argument('--meta_smooth_jerk_weight', type=float, default=0.001,
                    help='Meta loss weight for first-order action difference (jerk) smoothing')
parser.add_argument('--meta_smooth_snap_weight', type=float, default=0.0002,
                    help='Meta loss weight for second-order normalized action difference (snap) smoothing')

# 全局规划引导元损失参数
parser.add_argument('--meta_guidance_weight', type=float, default=0.5,
                    help='Weight for global guidance meta loss (path-guiding dense supervision)')
parser.add_argument('--guide_sample_count', type=int, default=10,
                    help='Number of keypoints to sample for guidance loss computation')
parser.add_argument('--guide_sample_strategy', type=str, default='random',
                    choices=['random', 'uniform', 'adaptive', 'critical'],
                    help='Sampling strategy: random/uniform/adaptive(danger+curvature)/critical(start+end+danger)')
parser.add_argument('--guide_max_accel', type=float, default=5.0,
                    help='Max acceleration for trapezoidal velocity profile (m/s^2)')
parser.add_argument('--guide_max_decel', type=float, default=6.0,
                    help='Max deceleration for trapezoidal velocity profile (m/s^2)')
parser.add_argument('--guide_dir_weight', type=float, default=0.5,
                    help='Weight for direction alignment loss within guidance loss')
parser.add_argument('--guide_speed_weight', type=float, default=0.3,
                    help='Weight for overspeed penalty within guidance loss')
parser.add_argument('--guide_lateral_weight', type=float, default=0.3,
                    help='Weight for lateral error penalty (geometric distance to planned path)')
parser.add_argument('--guide_speed_diff_weight', type=float, default=0.2,
                    help='Weight for speed difference penalty (overspeed + underspeed)')
parser.add_argument('--guide_escape_weight', type=float, default=1.0,
                    help='Weight for escape penalty on collided points')
parser.add_argument('--guide_recovery_speed_weight', type=float, default=0.15,
                    help='Extra speed damping weight on planner-invalid sampled points')
parser.add_argument('--guide_collision_threshold', type=float, default=-0.05,
                    help='Penetration threshold below which point is considered collided')
parser.add_argument('--guide_accel_weight', type=float, default=0.1,
                    help='Weight for acceleration mismatch penalty (deceleration requirement)')
parser.add_argument('--planner_resolution', type=float, default=0.3,
                    help='Resolution of the occupancy grid for A* planning (meters)')
parser.add_argument('--planner_margin', type=float, default=0.15,
                    help='Safety margin for obstacle inflation in planner (meters)')
parser.add_argument('--planner_parallel', dest='planner_parallel', action='store_true',
                    help='Enable sample-level parallel global planning with multiprocessing pool')
parser.add_argument('--no_planner_parallel', dest='planner_parallel', action='store_false',
                    help='Disable sample-level parallel global planning')
parser.set_defaults(planner_parallel=True)
parser.add_argument('--planner_workers', type=int, default=0,
                    help='Number of planner worker processes (<=0 means auto)')
parser.add_argument('--planner_pool_maxtasks', type=int, default=256,
                    help='maxtasksperchild for planner process pool to avoid long-run memory growth')
# [引导后端切换开关]
# - astar: 使用在线 A* 规划引导（原有方案）
# - dijkstra_potential: 使用离线缓存 Dijkstra 势场引导（新方案）
# 使用方法：
# - 默认不写时为 dijkstra_potential（离线缓存势场 + 预计算地图）。
# - 势场模式默认读取 /home/robot/transformer/precomputed_maps_turn_encouragement
#   需要配合预计算地图目录，例如：
#   --precomputed_map_dir /home/robot/transformer/precomputed_maps_turn_encouragement --num_precomputed_maps 0
# - 切回 A* 模式：--guidance_backend astar
# 说明：这是统一开关，优先于旧的 use_precomputed_potential_maps/use_astar_guidance 组合语义。
parser.add_argument('--guidance_backend', type=str, default='dijkstra_potential',
                    choices=['astar', 'dijkstra_potential'],
                    help='Switch guidance backend between online A* and cached Dijkstra potential field')
parser.add_argument('--use_precomputed_potential_maps', default=False, action='store_true',
                    help='Use precomputed Dijkstra potential-map guidance instead of online A* planning')
parser.add_argument('--precomputed_map_dir', type=str, default='/home/robot/transformer/precomputed_maps_turn_encouragement',
                    help='Directory containing precomputed potential cache .pt files')
parser.add_argument('--num_precomputed_maps', type=int, default=0,
                    help='Max number of precomputed maps to load from precomputed_map_dir (<=0 means all)')
# [势场查询参数]
# - trilinear: 三线性插值，点落在栅格内部时按8个角点加权，训练更平滑（推荐）
# - nearest: 最近邻查询，调试方便但梯度更离散
parser.add_argument('--potential_interpolation', type=str, default='trilinear',
                    choices=['nearest', 'trilinear'],
                    help='Interpolation mode for querying potential/vector field at continuous positions')
# [势场下降约束参数]
# ReLU(phi[t+1]-phi[t]+delta) 中的 delta。
# 取负值(如 -0.01)表示“允许极小上升噪声，但整体应下降”。
parser.add_argument('--potential_delta_margin', type=float, default=-0.01,
                    help='Delta margin in potential decrease loss: ReLU(phi[t+1]-phi[t]+delta)')
parser.add_argument('--use_astar_guidance', default=False, action='store_true',
                    help='Force legacy online A* guidance even when precomputed maps are enabled')
parser.add_argument('--guidance_all_phases', dest='guidance_all_phases', action='store_true',
                    help='Compute guidance loss/metrics in worker phase too (A* backend may be slower)')
parser.add_argument('--no_guidance_all_phases', dest='guidance_all_phases', action='store_false',
                    help='Compute guidance only in LGN phase unless potential backend auto-enables worker phase')
parser.set_defaults(guidance_all_phases=False)
parser.add_argument('--diag_interval', type=int, default=100,
                    help='Print detailed DIAG logs every N iterations (<=0 disables)')
parser.add_argument('--diag_second_order', dest='diag_second_order', action='store_true',
                    help='Enable heavy second-order diagnostic probes (can be noisy and slow)')
parser.add_argument('--no_diag_second_order', dest='diag_second_order', action='store_false',
                    help='Disable heavy second-order diagnostic probes')
parser.set_defaults(diag_second_order=False)
parser.add_argument('--terminal_log_interval', type=int, default=500,
                    help='Update terminal progress/log text every N iterations')
parser.add_argument('--debug_scalar_interval', type=int, default=25,
                    help='Unified TensorBoard scalar logging interval in iterations (<=0 disables periodic scalar writes)')
parser.add_argument('--debug_tb_interval', type=int, default=500,
                    help='TensorBoard Debug/* logging interval in iterations (<=0 disables Debug tag writes)')
parser.add_argument('--artifact_save_interval', type=int, default=1000,
                    help='Unified checkpoint, trajectory, and video save interval (<=0 disables periodic saves)')
parser.add_argument('--trajectory_save_interval', type=int, default=None,
                    help='Deprecated alias for --artifact_save_interval')

args = parser.parse_args()
if args.trajectory_save_interval is not None:
    args.artifact_save_interval = int(args.trajectory_save_interval)
yaw_rate_max = math.radians(float(args.yaw_rate_max_deg))
use_attitude_v2 = args.attitude_model == 'v2'

# 统一 guidance 开关生效：根据 guidance_backend 映射为旧布尔参数，保持后续逻辑兼容。
if args.guidance_backend == 'dijkstra_potential':
    args.use_precomputed_potential_maps = True
    args.use_astar_guidance = False
else:
    args.use_precomputed_potential_maps = False
    args.use_astar_guidance = True

# 势场后端查询代价较低，默认在 worker phase 也计算 guidance 指标/损失。
args.guidance_all_phases = bool(args.guidance_all_phases or args.use_precomputed_potential_maps)

# Planner parallel runtime config (used by guidance reference computation)
configure_planner_pool(
    enabled=args.planner_parallel,
    num_workers=args.planner_workers,
    maxtasks_per_child=args.planner_pool_maxtasks,
)
POTENTIAL_MAP_CACHE = None

########## 2. 目录与日志初始化 ##########
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
script_name = os.path.splitext(os.path.basename(__file__))[0]
save_dir_name = "代码9.2.7自己决定速度"
save_dir = create_unique_experiment_dir(
    os.path.join("..", "checkpoints"),
    save_dir_name,
)
video_dir = os.path.join(save_dir, 'videos')

os.makedirs(video_dir, exist_ok=True)
print(f"Training artifacts will be saved to: {save_dir}")

with open(os.path.join(save_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
map_writers = {}
print(f"[TensorBoard] Unified log directory: {os.path.join(save_dir, 'logs')}")


device = torch.device('cuda')

########## 3. 环境初始化 ##########
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          scene_scale=args.scene_scale,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle,
          obstacle_count_scale=args.obstacle_count_scale,
          speed_limit_softness=args.soft_speed_limit_softness,
          max_speed_ceiling=args.max_speed_ceiling,
          hard_vpred_clip=args.hard_vpred_clip,
          hard_speed_clip=args.hard_speed_clip,
          start_goal_plane_y_abs=args.start_goal_plane_y_abs,
          include_u_local_optimum=args.include_u_local_optimum,
          compact_two_zone_map=args.compact_two_zone_map,
          wall_physical_feedback=args.wall_physical_feedback)


TRAIN_MAP_TYPES = ("easy",)
PRECOMPUTED_CURRICULUM_REQUIRED_TYPES = TRAIN_MAP_TYPES
VIZ_MAP_TYPES = TRAIN_MAP_TYPES
PRECOMPUTED_MAP_TYPE_CODES = {
    "none": -1,
    "easy": 0,
    "hairpin": 1,
    "u_min": 2,
    "hard": 3,
    "legacy": 4,
}
PRECOMPUTED_CURRICULUM_STAGE_CODES = {
    "none": -1,
    "easy_only": 0,
    "easy_hairpin": 1,
    "easy_hairpin_u_min": 2,
}


PRECOMPUTED_MAP_TYPE_INDICES = {}
INITIAL_PRECOMPUTED_MAP_IDX = -1
INITIAL_PRECOMPUTED_MAP_TYPE = "none"
INITIAL_PRECOMPUTED_MAP_FILE = ""

if args.use_precomputed_potential_maps:
    if PotentialMapCache is None or query_potential_guidance is None:
        raise RuntimeError("potential_map_utils.py is required for --use_precomputed_potential_maps")
    POTENTIAL_MAP_CACHE = PotentialMapCache(
        map_dir=args.precomputed_map_dir,
        num_maps=args.num_precomputed_maps,
    )
    if len(POTENTIAL_MAP_CACHE) <= 0:
        raise RuntimeError(
            f"No precomputed maps found in {args.precomputed_map_dir}. "
            f"Please run precompute_potential_maps.py first."
        )
    PRECOMPUTED_MAP_TYPE_INDICES = _build_precomputed_map_type_indices(POTENTIAL_MAP_CACHE)
    missing_types = [
        map_type for map_type in PRECOMPUTED_CURRICULUM_REQUIRED_TYPES
        if len(PRECOMPUTED_MAP_TYPE_INDICES.get(map_type, [])) == 0
    ]
    if missing_types:
        counts_msg = ", ".join(
            f"{k}={len(v)}" for k, v in sorted(PRECOMPUTED_MAP_TYPE_INDICES.items())
        )
        raise RuntimeError(
            "Easy-only training requires at least one easy precomputed map. "
            f"Missing: {','.join(missing_types)}. loaded_counts: {counts_msg}. "
            "If --num_precomputed_maps is set, increase it or use 0 to load all maps."
        )

    INITIAL_PRECOMPUTED_MAP_TYPE = "easy"
    INITIAL_PRECOMPUTED_MAP_IDX = PRECOMPUTED_MAP_TYPE_INDICES[INITIAL_PRECOMPUTED_MAP_TYPE][0]
    INITIAL_PRECOMPUTED_MAP_FILE = os.path.basename(POTENTIAL_MAP_CACHE.map_files[INITIAL_PRECOMPUTED_MAP_IDX])
    first_map = POTENTIAL_MAP_CACHE.get_map(INITIAL_PRECOMPUTED_MAP_IDX)
    _align_env_goal_planes_to_precomputed_map(first_map, env, map_idx_hint=INITIAL_PRECOMPUTED_MAP_IDX)
    env.reset_from_precomputed_map(first_map)
    env.current_map_idx = INITIAL_PRECOMPUTED_MAP_IDX
    counts_msg = ", ".join(
        f"{k}={len(v)}" for k, v in sorted(PRECOMPUTED_MAP_TYPE_INDICES.items())
    )
    print(
        f"[PotentialMap] Enabled. loaded={len(POTENTIAL_MAP_CACHE)} "
        f"from {args.precomputed_map_dir}, initial={INITIAL_PRECOMPUTED_MAP_FILE}, "
        f"counts=({counts_msg})"
    )

_upstream_probe = probe_update_state_vec_common_upstream(device)
env.update_state_vec_in_meta_path = bool(_upstream_probe["is_common_upstream"])
print(
    f"[Phase1 Probe] update_state_vec common-upstream="
    f"{env.update_state_vec_in_meta_path} (delta={_upstream_probe['delta']:.6g})"
)

base_state_dim = 7 if args.no_odom else 10
state_dim = base_state_dim + (3 if use_attitude_v2 else 0)
action_dim = 4 if use_attitude_v2 else 3
geom_dim = 19
progress_dim = 8
progress_dim += WORKER_CONTEXT_FEATURE_DIM
worker_state_dim = state_dim + geom_dim + progress_dim

if args.no_odom:
    try:
        worknet = WorkNet(worker_state_dim, action_dim, max_seq_len=args.worker_max_seq_len)
    except TypeError:
        worknet = WorkNet(worker_state_dim, action_dim)
else:
    try:
        worknet = WorkNet(worker_state_dim, action_dim, max_seq_len=args.worker_max_seq_len)
    except TypeError:
        worknet = WorkNet(worker_state_dim, action_dim)
worknet = worknet.to(device)

try:
    lgn = LossGenNet(
        state_dim=state_dim,
        geom_dim=geom_dim,
        progress_dim=progress_dim,
        max_seq_len=args.lgn_max_seq_len,
        output_temperature=args.lgn_output_temperature,
        weight_floor=args.lgn_weight_floor,
    ).to(device)
except TypeError:
    lgn = LossGenNet(state_dim=state_dim).to(device)
########## 4. 加载预训练模型 ##########


worker_output_row_indices = [0, 2, 4, 6] if use_attitude_v2 else [0, 2, 4]
load_compatible_checkpoint(
    worknet,
    args.resume_worker,
    "Worker",
    device,
    zero_expanded=True,
    output_row_indices=worker_output_row_indices,
)
load_compatible_checkpoint(lgn, args.resume_lgn, "LGN", device, zero_expanded=True)

########## 5. 优化器配置 ##########
optim_worker = AdamW(worknet.parameters(), args.lr)
optim_lgn = AdamW(lgn.parameters(), args.lgn_lr)
sched = CosineAnnealingLR(optim_worker, args.num_iters, args.lr * 0.01)

########## 6. 日志状态 ##########
scaler_q_by_map = defaultdict(lambda: defaultdict(list))

########## 7. 训练主循环 ##########

# 使用命令行参数重新初始化全局规划器
global_planner = GlobalPlanner(
    resolution=args.planner_resolution,
    margin=args.planner_margin,
    device=device
)
print(f"[GlobalPlanner] Initialized with resolution={args.planner_resolution}m, margin={args.planner_margin}m")

current_precomputed_map_idx = int(INITIAL_PRECOMPUTED_MAP_IDX)
current_precomputed_map_type = str(INITIAL_PRECOMPUTED_MAP_TYPE)
current_precomputed_map_file = str(INITIAL_PRECOMPUTED_MAP_FILE)
current_precomputed_stage = "none"
current_precomputed_stage_update_count = 0
current_precomputed_active_types = ""
precomputed_type_offsets = defaultdict(int)
latest_viz_by_map_type = {map_type: None for map_type in VIZ_MAP_TYPES}
best_meta_loss = float('inf')
best_meta_loss_step = 0

terminal_log_interval = max(1, int(args.terminal_log_interval))
tb_scalar_interval = int(args.debug_scalar_interval)
pbar = tqdm(range(args.num_iters), ncols=120, miniters=terminal_log_interval)
B = args.batch_size
cycle_len = args.lgn_steps + args.worker_steps
maze_update_counter = 0

for i in pbar:
    term_log_now = ((i + 1) % terminal_log_interval == 0)
    tb_log_now = (
        tb_scalar_interval > 0 and (
            i == 0
            or ((i + 1) % tb_scalar_interval == 0)
            or ((i + 1) == args.num_iters)
        )
    )
    cycle_pos = i % cycle_len
    train_lgn_phase = cycle_pos < args.lgn_steps
    phase_str = f"LGN ({cycle_pos+1}/{args.lgn_steps})" if train_lgn_phase else f"Work ({cycle_pos-args.lgn_steps+1}/{args.worker_steps})"
    env.set_meta_differentiable_mode(train_lgn_phase)
    if _diag_should_log(i, args):
        print(
            f"[DIAG iter={i}] phase={phase_str}, train_lgn_phase={train_lgn_phase}, "
            f"cycle_pos={cycle_pos}, cycle_len={cycle_len}, lgn_steps={args.lgn_steps}, worker_steps={args.worker_steps}"
        )

    if args.use_precomputed_potential_maps:
        stage_name, active_map_types = _precomputed_curriculum_stage(i)
        stage_changed = stage_name != current_precomputed_stage
        update_precomputed_map = (maze_update_counter % args.maze_update_interval == 0) or stage_changed
        if update_precomputed_map:
            if stage_changed:
                current_precomputed_stage = stage_name
                current_precomputed_stage_update_count = 0
                current_precomputed_active_types = ",".join(active_map_types)
                stage_msg = (
                    f"iter={i + 1}, stage={stage_name}, "
                    f"active_types={current_precomputed_active_types}"
                )
                print(f"[PotentialMapCurriculum] {stage_msg}")
                if tb_log_now or stage_changed:
                    writer.add_text("Map/Curriculum_Stage", stage_msg, i + 1)

            current_precomputed_map_idx, current_precomputed_map_type = _select_precomputed_curriculum_map(
                active_types=active_map_types,
                stage_update_count=current_precomputed_stage_update_count,
                type_offsets=precomputed_type_offsets,
                type_indices=PRECOMPUTED_MAP_TYPE_INDICES,
            )
            current_precomputed_stage_update_count += 1
            current_precomputed_map_file = os.path.basename(POTENTIAL_MAP_CACHE.map_files[current_precomputed_map_idx])
            env.current_map_idx = current_precomputed_map_idx
            map_data_cur = POTENTIAL_MAP_CACHE.get_map(current_precomputed_map_idx)
            _align_env_goal_planes_to_precomputed_map(map_data_cur, env, map_idx_hint=current_precomputed_map_idx)
            env.reset_from_precomputed_map(map_data_cur)
            map_msg = (
                f"iter={i + 1}, idx={current_precomputed_map_idx}, "
                f"type={current_precomputed_map_type}, file={current_precomputed_map_file}"
            )
            if tb_log_now or stage_changed:
                _resolve_tb_writer(current_precomputed_map_type, writer, map_writers).add_text(
                    "Map/Current_Precomputed_File",
                    map_msg,
                    i + 1,
                )
            if term_log_now or stage_changed:
                print(f"[PotentialMapCurriculum] {map_msg}")
        else:
            env.reset_drone_only()
    else:
        if maze_update_counter % args.maze_update_interval == 0:
            env.reset()          # full reset: new maze + new drones
        else:
            env.reset_drone_only()  # keep maze, reset drones only
    maze_update_counter += 1
    worknet.reset()

    p_history, v_history, a_history, vec_to_pt_history = [], [], [], []
    rpy_history = []
    R_history = []  # 记录姿态矩阵用于可视化
    R_loss_history = []  # 保留可反向传播姿态用于安全损失
    real_act_history = []
    depth_history = []
    act_buffer = [env.act.detach()] * 2
    trajectory_lgn_weights = []
    trajectory_lgn_vrefs = []
    R_proxy_history = []
    yaw_rate_cmd_history = []
    yaw_error_history = []
    act_for_diag = None
    dist_obj_history = []
    geom_feat_last = None
    progress_feat_last = None

    h = None
    lgn_hx = None
    do_save_viz = current_precomputed_map_type in VIZ_MAP_TYPES
    rollout_steps = args.lgn_timesteps if train_lgn_phase else args.timesteps

    ###### A. Rollout ######
    for t in range(rollout_steps):
        ctl_dt = 1.0 / 15.0
        depth, flow = env.render(ctl_dt)
        depth = sanitize_tensor(depth, nan=24.0, posinf=24.0, neginf=0.3)

        if do_save_viz:
            depth_history.append(depth[0].detach().cpu().clone())

        p_history.append(env.p)
        v_history.append(env.v)
        a_history.append(env.a)
        vec_curr = env.find_vec_to_nearest_pt()
        vec_to_pt_history.append(vec_curr)
        dist_obj_curr = safe_l2_norm(vec_curr, dim=-1) - env.margin
        dist_obj_history.append(dist_obj_curr)
        rpy_history.append(rotation_matrix_to_rpy_deg(env.R).detach())
        R_history.append(env.R.detach().clone())  # 保存姿态矩阵
        R_loss_history.append(env.R)

        target_v_raw_curr = env.p_target - env.p.detach()
        target_v_norm = torch.norm(target_v_raw_curr, 2, -1, keepdim=True)
        max_speed = torch.as_tensor(env.max_speed, device=target_v_norm.device, dtype=target_v_norm.dtype)
        target_v = (target_v_raw_curr / (target_v_norm + 1e-6)) * torch.minimum(target_v_norm, max_speed)

        R = build_yaw_frame(env.R) if use_attitude_v2 else env.R
        R_proxy_history.append(R)
        state_list = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.margin[:, None]]
        if use_attitude_v2:
            heading_ref_world, heading_ref_local_xy, yaw_error = compute_heading_reference(env, R)
            yaw_rate_norm = getattr(env, "yaw_rate", torch.zeros((B, 1), device=device)) / float(yaw_rate_max)
            state_list.extend([heading_ref_local_xy, yaw_rate_norm])
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom: state_list.insert(0, local_v)
        
        state_tensor = sanitize_tensor(
            torch.cat(state_list, -1),
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).clamp(-10.0, 10.0)

        x_pooled = F.max_pool2d((3 / depth.clamp(0.3, 24) - 0.6)[:, None], 4, 4)
        x_pooled = sanitize_tensor(x_pooled, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

        geom_feat = extract_depth_geometry_features(depth)
        geom_feat = sanitize_tensor(
            geom_feat,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).clamp(-10.0, 10.0)

        progress_feat_base_raw = extract_progress_features(
            p_history_list=p_history,
            v_history_list=v_history,
            dist_obj_history_list=dist_obj_history,
            p_target=env.p_target,
            window=8,
        )
        context_feat_raw = extract_worker_context_features(
            p_history_list=p_history,
            p_target=env.p_target,
            R_current=R,
        )
        progress_feat = torch.cat([progress_feat_base_raw, context_feat_raw], dim=-1)
        progress_feat = sanitize_tensor(
            progress_feat,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        ).clamp(-10.0, 10.0)
        geom_feat_last = geom_feat
        progress_feat_last = progress_feat

        # LGN forward:
        # - LGN phase: keep graph for unrolled second-order path.
        # - Worker phase: freeze LGN graph to avoid unnecessary memory overhead.
        if train_lgn_phase:
            current_weights, current_vref_body, lgn_hx = lgn(
                x_pooled,
                state_tensor,
                geom_feat,
                progress_feat,
                lgn_hx,
            )
        else:
            with torch.no_grad():
                current_weights, current_vref_body, lgn_hx = lgn(
                    x_pooled,
                    state_tensor,
                    geom_feat,
                    progress_feat,
                    lgn_hx,
                )

        if t == 0 and _diag_should_log(i, args):
            first_lgn_weight = current_weights[0, 0] if current_weights.numel() > 0 else None
            print(
                f"[DIAG iter={i} t=0] current_weights={_diag_grad_meta(current_weights)}, "
                f"lgn_hx={_diag_grad_meta(lgn_hx)}, "
                f"first_lgn_weight={_diag_grad_meta(first_lgn_weight)}"
            )
            expected = "requires_grad=True" if train_lgn_phase else "requires_grad=False"
            actual = bool(current_weights.requires_grad)
            print(
                f"[DIAG iter={i} t=0] LGN mode expectation: {expected}, "
                f"actual_requires_grad={actual}"
            )
            if train_lgn_phase and not actual:
                print(
                    f"[DIAG iter={i} t=0][ALERT] LGN phase but current_weights.requires_grad=False; "
                    "possible second-order path break."
                )
        trajectory_lgn_weights.append(current_weights)
        trajectory_lgn_vrefs.append(current_vref_body)

        # Worker Forward: consume state + geometry + progress features.
        worker_input = torch.cat([state_tensor, geom_feat, progress_feat], dim=-1)
        act, _, h = worknet(x_pooled, worker_input, h)
        act = sanitize_tensor(act, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        act_for_diag = act
        a_pred, yaw_rate_cmd = decode_worker_action(act, R, yaw_rate_max)
        real_act = a_pred
        real_act = sanitize_tensor(real_act, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        real_act_history.append(real_act.detach())
        act_buffer.append(real_act)

        if use_attitude_v2:
            heading_v_ref = env.v

            heading_ref_world, heading_ref_local_xy, yaw_error_vel, yaw_rate_rule, heading_speed_xy = \
                compute_velocity_heading_command(
                    R_yaw=R,
                    v_ref_world=heading_v_ref,
                    yaw_rate_max_value=yaw_rate_max,
                    yaw_kp=args.heading_yaw_kp,
                    min_speed=args.heading_min_speed,
                )

            if yaw_rate_cmd is None:
                yaw_rate_residual = torch.zeros((B, 1), device=device, dtype=real_act.dtype)
            else:
                yaw_rate_residual = yaw_rate_cmd

            yaw_rate_cmd_final = yaw_rate_rule + float(args.heading_residual_scale) * yaw_rate_residual
            yaw_rate_cmd_final = torch.clamp(
                yaw_rate_cmd_final,
                -float(yaw_rate_max),
                float(yaw_rate_max),
            )

            yaw_rate_cmd_history.append(yaw_rate_cmd_final)
            yaw_error_history.append(yaw_error_vel.detach())
            env.run(
                real_act,
                ctl_dt,
                heading_ref=heading_ref_world,
                yaw_rate_cmd=yaw_rate_cmd_final,
                yaw_rate_max=yaw_rate_max,
            )
        else:
            env.run(real_act, ctl_dt, target_v_raw_curr)

        # Keep full horizon so in-goal staying can continuously accumulate arrival reward.

        if args.detach_interval > 0 and (t + 1) % args.detach_interval == 0:
            if h is not None:
                h = h.detach()
            # LGN phase 时保留 lgn_hx 梯度，Worker phase 时截断
            if lgn_hx is not None and not train_lgn_phase:
                lgn_hx = lgn_hx.detach()

    ###### B. Loss Calculation (Step-wise) ######
    p_history = torch.stack(p_history)     # [T, B, 3]
    v_history = torch.stack(v_history)     # [T, B, 3]
    a_history = torch.stack(a_history)     # [T, B, 3]
    act_buffer = torch.stack(act_buffer)   # [T+2, B, 3]
    weights_seq = torch.stack(trajectory_lgn_weights)  # [T, B, 7]
    vref_body_seq = torch.stack(trajectory_lgn_vrefs)  # [T, B, 3]
    R_proxy_seq = torch.stack(R_proxy_history)         # [T, B, 3, 3]
    v_real_body_seq = torch.squeeze(v_history[:, :, None, :] @ R_proxy_seq, 2)
    v_real_body_dir = safe_normalize(v_real_body_seq, dim=-1)
    vref_body_dir = safe_normalize(vref_body_seq, dim=-1)
    lgn_vref_real_cos_seq = (v_real_body_dir * vref_body_dir).sum(dim=-1).clamp(-1.0, 1.0)
    loss_lgn_vref_seq = (1.0 - lgn_vref_real_cos_seq).clamp(0.0, 2.0)
    loss_lgn_potential_vref, lgn_potential_vref_components = compute_lgn_potential_vref_sync_loss(
        env=env,
        p_history=p_history,
        vref_body_seq=vref_body_seq,
        R_proxy_seq=R_proxy_seq,
        config=args,
        potential_map_cache=POTENTIAL_MAP_CACHE,
    )
    if _diag_should_log(i, args):
        print(f"[DIAG iter={i}] weights_seq: {_diag_grad_meta(weights_seq)}")
        if train_lgn_phase and not weights_seq.requires_grad:
            print(
                f"[DIAG iter={i}][ALERT] LGN phase but weights_seq.requires_grad=False; "
                "possible second-order path break."
            )
    rpy_history = torch.stack(rpy_history) # [T, B, 3]
    R_history = torch.stack(R_history)     # [T, B, 3, 3]
    R_loss_history = torch.stack(R_loss_history)  # [T, B, 3, 3], keep graph
    real_act_history = torch.stack(real_act_history) # [T, B, 3]

    vec_to_pt = torch.stack(vec_to_pt_history)
    if vec_to_pt.dim() == 4: vec_to_pt = vec_to_pt.mean(1)
    
    # 1. 计算各项 Raw Loss (保留 [T, B] 维度用于 Step-wise 加权)

    # 碰撞距离 (先计算, 后续速度目标依赖它)
    dist_obj = safe_l2_norm(vec_to_pt, dim=-1) - env.margin  # [T, B]

    # 速度变化响应项：不使用自适应速度大小目标。
    speed_actual = safe_l2_norm(v_history, dim=-1)  # [T, B]
    dist_to_goal = safe_l2_norm(env.p_target - p_history, dim=-1)  # [T, B]
    delta_speed_signed = torch.zeros_like(speed_actual)
    if speed_actual.shape[0] > 1:
        delta_speed_signed[1:] = torch.diff(speed_actual, dim=0)

    target_dir = safe_normalize(env.p_target - p_history, dim=-1)
    v_dir = safe_normalize(v_history, dim=-1)
    loss_direction_seq = (1.0 - (v_dir * target_dir).sum(-1))

    with torch.no_grad():
        v_to_pt = torch.ones_like(dist_obj)
        if dist_obj.shape[0] > 1:
            v_to_pt[1:] = (-torch.diff(dist_obj, 1, 0) * 135.0).clamp_min(1.0)

    collision_depth = F.relu(-dist_obj)
    loss_avoidance_seq = v_to_pt * (1.0 - dist_obj).relu().pow(2)
    loss_collision_seq = F.softplus(dist_obj.mul(-32.0)) * v_to_pt

    # 注意: compute_overlap_loss_per_step 返回 [B, T], 需要 permute 成 [T, B]
    loss_exploration_seq = compute_overlap_loss_per_step(
        p_history, sigma=1.0, time_window=int(args.exploration_time_window)
    ).permute(1, 0)
    loss_turn_base_seq = compute_turn_preference_loss(
        v_history,
        speed_threshold=args.turn_speed_threshold,
        speed_softness=args.turn_speed_softness,
        soft_angle_deg=args.turn_soft_angle_deg,
    )
    loss_turn_seq = loss_turn_base_seq
    if use_attitude_v2 and len(yaw_rate_cmd_history) > 0:
        yaw_rate_cmd_seq = torch.stack(yaw_rate_cmd_history)  # [T, B, 1]
        if yaw_rate_cmd_seq.shape[0] > 1:
            loss_yaw_smooth = yaw_rate_cmd_seq.diff(1, 0).pow(2).mean()
        else:
            loss_yaw_smooth = torch.tensor(0.0, device=device, dtype=v_history.dtype)
    else:
        loss_yaw_smooth = torch.tensor(0.0, device=device, dtype=v_history.dtype)

    # ========== Heading-Velocity Alignment Safety Loss ==========
    heading_world = R_loss_history[:, :, :, 0]  # [T, B, 3]

    heading_xy = torch.stack([
        heading_world[:, :, 0],
        heading_world[:, :, 1],
        torch.zeros_like(heading_world[:, :, 2]),
    ], dim=-1)
    heading_xy = safe_normalize(heading_xy, dim=-1)

    v_xy = torch.stack([
        v_history[:, :, 0],
        v_history[:, :, 1],
        torch.zeros_like(v_history[:, :, 2]),
    ], dim=-1)

    speed_xy = safe_l2_norm(v_xy, dim=-1)
    v_dir_xy = safe_normalize(v_xy, dim=-1)

    heading_vel_cos = (heading_xy * v_dir_xy).sum(dim=-1).clamp(-1.0, 1.0)

    valid_heading_speed = (speed_xy > float(args.heading_min_speed)).float()

    loss_heading_align_seq = valid_heading_speed * (1.0 - heading_vel_cos)

    loss_backward_seq = valid_heading_speed * F.relu(
        float(args.backward_cos_limit) - heading_vel_cos
    ).pow(2)

    loss_heading_safety = (
        float(args.coef_heading_align) * loss_heading_align_seq.mean()
        + float(args.coef_backward_penalty) * loss_backward_seq.mean()
    )

    loss_stuck_seq, loss_collision_duration_seq, stuck_ratio = compute_stuck_loss(
        p_history, collision_depth,
        stuck_window=args.stuck_window,
        displacement_threshold=args.stuck_displacement_threshold,
    )
    actual_T = p_history.shape[0]

    # 高度约束损失 (固定权重, 不经LGN控制)
    z_pos = p_history[:, :, 2]  # [T, B]
    z_target = 1.0  # 迷宫中层高度
    z_min, z_max = 0.0, 5.0
    loss_height_seq = (F.smooth_l1_loss(z_pos, torch.full_like(z_pos, z_target), reduction='none')
                       + F.softplus((z_pos - z_max) * 20.0)
                       + F.softplus((z_min - z_pos) * 20.0))

    # 非负权重策略：所有分量均不允许为负
    weights_seq_raw = weights_seq
    if _diag_should_log(i, args):
        print(f"[DIAG iter={i}] weights_seq_raw: requires_grad={weights_seq_raw.requires_grad}, grad_fn={type(weights_seq_raw.grad_fn).__name__ if weights_seq_raw.grad_fn else 'None'}")
        print(
            f"[DIAG iter={i}] loss_raw requires_grad: delta_speed={delta_speed_signed.requires_grad}, "
            f"dir={loss_direction_seq.requires_grad}, avoid={loss_avoidance_seq.requires_grad}, "
            f"expl={loss_exploration_seq.requires_grad}, turn={loss_turn_seq.requires_grad}, "
            f"vref={loss_lgn_vref_seq.requires_grad}"
        )
    # LGN 输出已做约束：speed_pref_sign in [-1,1]，其余权重非负。
    effective_weights_seq = weights_seq_raw
    if _diag_should_log(i, args):
        print(
            f"[DIAG iter={i}] effective_weights: requires_grad={effective_weights_seq.requires_grad}, "
            f"grad_fn={type(effective_weights_seq.grad_fn).__name__ if effective_weights_seq.grad_fn else 'None'}"
        )

    speed_pref_signed = effective_weights_seq[:, :, 0].clamp(-1.0, 1.0)
    speed_pref_strength = effective_weights_seq[:, :, 1].clamp_min(0.0)
    delta_speed_scale = 0.10 * float(env.max_speed) + 1e-6
    speed_response_seq = torch.tanh(delta_speed_signed / delta_speed_scale)
    weighted_speed_pref_reward_seq = -speed_pref_strength * speed_pref_signed * speed_response_seq

    # 2. Step-wise 加权 (Broadcasting: [T, B] * [T, B])
    weighted_loss_map = (
        weighted_speed_pref_reward_seq +
        effective_weights_seq[:, :, 2] * loss_direction_seq +
        effective_weights_seq[:, :, 3] * loss_avoidance_seq +
        effective_weights_seq[:, :, 4] * loss_exploration_seq +
        effective_weights_seq[:, :, 5] * loss_turn_seq +
        effective_weights_seq[:, :, 6] * loss_lgn_vref_seq
    )

    # 3. 最终 Proxy Loss
    proxy_loss = weighted_loss_map.mean() + loss_heading_safety
    if _diag_should_log(i, args):
        print(
            f"[DIAG iter={i}] weighted_loss_map: requires_grad={weighted_loss_map.requires_grad}, "
            f"grad_fn={type(weighted_loss_map.grad_fn).__name__ if weighted_loss_map.grad_fn else 'None'}"
        )
        print(
            f"[DIAG iter={i}] proxy_loss: requires_grad={proxy_loss.requires_grad}, "
            f"grad_fn={type(proxy_loss.grad_fn).__name__ if proxy_loss.grad_fn else 'None'}"
        )
        _diag_tensor_finite("act_for_diag", act_for_diag, i)
        _diag_tensor_finite("real_act_history", real_act_history, i)
        _diag_tensor_finite("p_history", p_history, i)
        _diag_tensor_finite("v_history", v_history, i)
        _diag_tensor_finite("a_history", a_history, i)
        _diag_tensor_finite("vec_to_pt", vec_to_pt, i)
        _diag_tensor_finite("dist_obj", dist_obj, i)
        _diag_tensor_finite("loss_direction_seq", loss_direction_seq, i)
        _diag_tensor_finite("loss_avoidance_seq", loss_avoidance_seq, i)
        _diag_tensor_finite("loss_exploration_seq", loss_exploration_seq, i)
        _diag_tensor_finite("loss_turn_seq", loss_turn_seq, i)
        _diag_tensor_finite("loss_lgn_vref_seq", loss_lgn_vref_seq, i)
        _diag_tensor_finite("loss_yaw_smooth", loss_yaw_smooth, i)
        _diag_tensor_finite("loss_heading_safety", loss_heading_safety, i)
        _diag_tensor_finite("weights_seq_raw", weights_seq_raw, i)
        _diag_tensor_finite("weighted_loss_map", weighted_loss_map, i)
        _diag_tensor_finite("proxy_loss", proxy_loss, i)

    # --- Meta Loss Components ---
    # Worker phase does not backprop through meta_loss; detach to avoid unnecessary graph build.
    if train_lgn_phase:
        meta_p_history = p_history
        meta_v_history = v_history
        meta_a_history = a_history
        meta_act_buffer = act_buffer
        meta_loss_collision_seq = loss_collision_seq
        meta_loss_stuck_seq = loss_stuck_seq
        meta_loss_height_seq = loss_height_seq
    else:
        meta_p_history = p_history.detach()
        meta_v_history = v_history.detach()
        meta_a_history = a_history.detach()
        meta_act_buffer = act_buffer.detach()
        meta_loss_collision_seq = loss_collision_seq.detach()
        meta_loss_stuck_seq = loss_stuck_seq.detach()
        meta_loss_height_seq = loss_height_seq.detach()

    loss_meta_pos = safe_l2_norm(meta_p_history[-1] - env.p_target, dim=-1).mean()
    loss_meta_arrival_reward, meta_arrival_hit_rate, meta_arrival_best_dist = compute_arrival_reward(
        meta_p_history,
        env.p_target,
        radius=args.meta_arrival_reward_radius,
    )
    loss_meta_coll = meta_loss_collision_seq.mean()
    loss_meta_ctrl = safe_l2_norm(meta_act_buffer, dim=-1).sum()
    loss_meta_jerk = meta_act_buffer.diff(1, 0).mul(15.0).pow(2).sum(-1).mean()
    loss_meta_snap = (F.normalize(meta_act_buffer - env.g_std, dim=-1)
                      .diff(1, 0).diff(1, 0).mul(15.0 ** 2).pow(2).sum(-1).mean())
    loss_meta_height = meta_loss_height_seq.mean()

    # --- 全局规划引导损失 ---
    guidance_active_this_phase = bool(train_lgn_phase or args.guidance_all_phases)
    if guidance_active_this_phase:
        if train_lgn_phase:
            loss_meta_guidance, guidance_components = compute_global_guidance_meta_loss(
                env, meta_p_history, meta_v_history, env.p_target, vec_to_pt, dist_obj,
                config=args,
                potential_map_cache=POTENTIAL_MAP_CACHE,
                planner=global_planner,
                a_history=meta_a_history,
                sample_count=args.guide_sample_count,
                strategy=args.guide_sample_strategy,
                max_speed=float(env.max_speed),
                max_accel=args.guide_max_accel,
                max_decel=args.guide_max_decel,
                dir_weight=args.guide_dir_weight,
                speed_weight=args.guide_speed_weight,
                lateral_weight=args.guide_lateral_weight,
                escape_weight=args.guide_escape_weight,
                collision_threshold=args.guide_collision_threshold,
                accel_weight=args.guide_accel_weight,
                speed_diff_weight=args.guide_speed_diff_weight,
                recovery_speed_weight=args.guide_recovery_speed_weight,
            )
        else:
            with torch.no_grad():
                loss_meta_guidance, guidance_components = compute_global_guidance_meta_loss(
                    env,
                    meta_p_history.detach(),
                    meta_v_history.detach(),
                    env.p_target,
                    vec_to_pt.detach(),
                    dist_obj.detach(),
                    config=args,
                    potential_map_cache=POTENTIAL_MAP_CACHE,
                    planner=global_planner,
                    a_history=meta_a_history.detach(),
                    sample_count=args.guide_sample_count,
                    strategy=args.guide_sample_strategy,
                    max_speed=float(env.max_speed),
                    max_accel=args.guide_max_accel,
                    max_decel=args.guide_max_decel,
                    dir_weight=args.guide_dir_weight,
                    speed_weight=args.guide_speed_weight,
                    lateral_weight=args.guide_lateral_weight,
                    escape_weight=args.guide_escape_weight,
                    collision_threshold=args.guide_collision_threshold,
                    accel_weight=args.guide_accel_weight,
                    speed_diff_weight=args.guide_speed_diff_weight,
                    recovery_speed_weight=args.guide_recovery_speed_weight,
                )
            loss_meta_guidance = loss_meta_guidance.detach()
    else:
        zero = torch.tensor(0.0, device=p_history.device)
        loss_meta_guidance = zero
        guidance_components = {
            'dir_align': zero,
            'speed_diff': zero,
            'overspeed': zero,
            'underspeed': zero,
            'lateral_error': zero,
            'accel_mismatch': zero,
            'escape': zero,
            'depth': zero,
            'recovery_speed': zero,
            'valid_ratio': zero,
            'invalid_ratio': zero,
            'collision_ratio': zero,
            'sample_count': 0.0,
            'avg_curvature': 0.0,
            'avg_path_progress': 0.0,
            'avg_lateral_error': 0.0,
            'max_lateral_error': 0.0,
            'planner_success_ratio': 0.0,
            'avg_ref_speed': 0.0,
            'sampled_astar_paths': [],
            'field_dir_align': 0.0,
        }

    # 训练目标仅使用位置/碰撞/高度/引导; 控制项仅用于日志监控
    loss_meta_stuck = meta_loss_stuck_seq.mean()
    meta_loss = (
        loss_meta_pos +
        loss_meta_coll +
        loss_meta_height +
        args.meta_guidance_weight * loss_meta_guidance +
        args.meta_smooth_jerk_weight * loss_meta_jerk +
        args.meta_smooth_snap_weight * loss_meta_snap +
        args.stuck_loss_weight * loss_meta_stuck +
        loss_heading_safety +
        args.coef_yaw_smooth * loss_yaw_smooth
        - args.meta_arrival_reward_weight * loss_meta_arrival_reward
    )
    if _diag_should_log(i, args):
        _diag_tensor_finite("loss_meta_pos", loss_meta_pos, i)
        _diag_tensor_finite("loss_meta_arrival_reward", loss_meta_arrival_reward, i)
        _diag_tensor_finite("loss_meta_coll", loss_meta_coll, i)
        _diag_tensor_finite("loss_meta_height", loss_meta_height, i)
        _diag_tensor_finite("loss_meta_guidance", loss_meta_guidance, i)
        _diag_tensor_finite("loss_meta_jerk", loss_meta_jerk, i)
        _diag_tensor_finite("loss_meta_snap", loss_meta_snap, i)
        _diag_tensor_finite("loss_meta_stuck", loss_meta_stuck, i)
        _diag_tensor_finite("loss_heading_safety", loss_heading_safety, i)
        _diag_tensor_finite("loss_yaw_smooth", loss_yaw_smooth, i)
        _diag_tensor_finite("meta_loss", meta_loss, i)
    # Keep root losses untouched for higher-order gradients; skip iteration via finite checks below.

    ###### C. Optimization ######
    optim_worker.zero_grad()
    optim_lgn.zero_grad()
    lgn_update_loss = 0.0
    worker_grad_norm = 0.0
    worker_grad_max = 0.0
    worker_grad_nonfinite = 0.0
    worker_grad_elems = 0.0
    worker_clip_pre = 0.0
    lgn_grad_norm = 0.0
    lgn_grad_max = 0.0
    lgn_grad_nonfinite = 0.0
    lgn_grad_elems = 0.0
    lgn_clip_pre = 0.0
    lgn_meta_probe_norm = 0.0
    lgn_meta_probe_nonfinite = 0.0
    lgn_meta_probe_elems = 0.0
    meta_pos_ur = torch.tensor(0.0, device=device)
    meta_coll_ur = torch.tensor(0.0, device=device)
    meta_ctrl_ur = torch.tensor(0.0, device=device)
    meta_arrival_reward_ur = torch.tensor(0.0, device=device)
    meta_arrival_hit_rate_ur = torch.tensor(0.0, device=device)
    meta_arrival_best_dist_ur = torch.tensor(0.0, device=device)

    rollout_is_finite = bool(
        torch.isfinite(proxy_loss).all()
        and torch.isfinite(meta_loss).all()
        and torch.isfinite(weights_seq).all()
        and torch.isfinite(vref_body_seq).all()
        and torch.isfinite(p_history).all()
        and torch.isfinite(v_history).all()
    )

    if not rollout_is_finite:
        if term_log_now:
            pbar.set_description(f"[{phase_str}] non-finite rollout skipped")
        continue

    if train_lgn_phase:
        # ===== Unrolled Bilevel: 可微内循环 =====
        # Step 1: 用 proxy_loss 对 worker 做可微梯度下降
        fast_params = dict(worknet.named_parameters())
        inner_update_is_finite = True

        for _inner in range(args.inner_steps):
            inner_grads = torch.autograd.grad(
                proxy_loss, tuple(fast_params.values()),
                create_graph=True, allow_unused=True, retain_graph=True,
            )

            inner_sq_terms = [(g * g).sum() for g in inner_grads if g is not None]
            if inner_sq_terms:
                inner_grad_norm_tensor = torch.sqrt(torch.stack(inner_sq_terms).sum() + 1e-12)
                if not bool(torch.isfinite(inner_grad_norm_tensor).item()):
                    inner_update_is_finite = False
                    break
                inner_scale = (args.inner_grad_clip / (inner_grad_norm_tensor + 1e-6)).clamp(max=1.0)
                inner_grads = tuple((g * inner_scale) if g is not None else None for g in inner_grads)

            if _inner == 0 and _diag_should_log(i, args):
                fast_param_values = tuple(fast_params.values())

                if args.diag_second_order:
                    # Probe weighted per-term proxy components so each branch truly depends on LGN weights.
                    weighted_speed_pref_reward = weighted_speed_pref_reward_seq.mean()
                    weighted_dir = (effective_weights_seq[:, :, 2] * loss_direction_seq).mean()
                    weighted_avoid = (effective_weights_seq[:, :, 3] * loss_avoidance_seq).mean()
                    weighted_expl = (effective_weights_seq[:, :, 4] * loss_exploration_seq).mean()
                    weighted_turn = (effective_weights_seq[:, :, 5] * loss_turn_seq).mean()
                    weighted_vref = (effective_weights_seq[:, :, 6] * loss_lgn_vref_seq).mean()

                    g_speed_pref_reward = _grad_or_none_tuple(weighted_speed_pref_reward, fast_param_values)
                    g_dir = _grad_or_none_tuple(weighted_dir, fast_param_values)
                    g_avoid = _grad_or_none_tuple(weighted_avoid, fast_param_values)
                    g_expl = _grad_or_none_tuple(weighted_expl, fast_param_values)
                    g_turn = _grad_or_none_tuple(weighted_turn, fast_param_values)
                    g_vref = _grad_or_none_tuple(weighted_vref, fast_param_values)

                    lgn_param_list = list(lgn.parameters())
                    _diag_grad_tuple_to_params("speed_pref_reward(weighted) second_order(worker_grad)->lgn", g_speed_pref_reward, lgn_param_list, i)
                    _diag_grad_tuple_to_params("direction(weighted) second_order(worker_grad)->lgn", g_dir, lgn_param_list, i)
                    _diag_grad_tuple_to_params("avoidance(weighted) second_order(worker_grad)->lgn", g_avoid, lgn_param_list, i)
                    _diag_grad_tuple_to_params("exploration(weighted) second_order(worker_grad)->lgn", g_expl, lgn_param_list, i)
                    _diag_grad_tuple_to_params("turn(weighted) second_order(worker_grad)->lgn", g_turn, lgn_param_list, i)
                    _diag_grad_tuple_to_params("vref(weighted) second_order(worker_grad)->lgn", g_vref, lgn_param_list, i)
                    _diag_grad_tuple_to_params("proxy_total second_order(worker_grad)->lgn", inner_grads, lgn_param_list, i)

                    act_only_loss = (act_for_diag.pow(2).mean() if act_for_diag is not None else torch.tensor(0.0, device=device))
                    act_only_weighted = speed_pref_strength.mean() * act_only_loss
                    g_act_only = torch.autograd.grad(
                        act_only_weighted, fast_param_values,
                        create_graph=True, allow_unused=True, retain_graph=True,
                    )
                    _diag_grad_tuple_to_params("act_only(weighted) second_order(worker_grad)->lgn", g_act_only, lgn_param_list, i)

                    _diag_grad_tuple_to_params("inner_grads(sum) -> lgn", inner_grads, lgn_param_list, i)
                    toy_grad = tuple((g.pow(2) if g is not None else None) for g in inner_grads)
                    _diag_grad_tuple_to_params("toy_grad(sqsum) -> lgn", toy_grad, lgn_param_list, i)

                inner_norm, inner_nonfinite, inner_elems = get_grad_norm_from_grads(inner_grads)
                print(f"[DIAG iter={i}] inner_grads finite: nonfinite={inner_nonfinite}/{inner_elems}")

            fast_params = {
                name: (p - args.inner_lr * g
                       if g is not None else p)
                for (name, p), g in zip(fast_params.items(), inner_grads)
            }

        if not inner_update_is_finite:
            if term_log_now:
                pbar.set_description(f"[{phase_str}] non-finite inner-update skipped")
        else:
            if _diag_should_log(i, args):
                fast_param_vals = list(fast_params.values())
                if len(fast_param_vals) > 0:
                    _diag_tensor_finite("first_fast_param", fast_param_vals[0], i)
                    _diag_output_to_params_count("fast_params(sum) -> lgn", sum(fp.sum() for fp in fast_param_vals), lgn.parameters(), i)

            # Step 2: 用虚拟更新后的 worker 做验证 rollout → meta_loss
            (meta_loss_unrolled, meta_pos_ur, meta_coll_ur, meta_ctrl_ur,
             meta_arrival_reward_ur, meta_arrival_hit_rate_ur, meta_arrival_best_dist_ur) = \
                unrolled_meta_rollout(
                    env,
                    worknet,
                    fast_params,
                    args,
                    B,
                    device,
                    POTENTIAL_MAP_CACHE,
                    global_planner,
                    iter_idx=i,
                )
            if not torch.isfinite(meta_loss_unrolled):
                if term_log_now:
                    pbar.set_description(f"[{phase_str}] non-finite unroll skipped")
            else:
                if _diag_should_log(i, args):
                    print(
                        f"[DIAG iter={i}] meta_loss_unrolled: requires_grad={meta_loss_unrolled.requires_grad}, "
                        f"grad_fn={type(meta_loss_unrolled.grad_fn).__name__ if meta_loss_unrolled.grad_fn else 'None'}"
                    )
                    _diag_tensor_finite("meta_loss_unrolled", meta_loss_unrolled, i)
                    _diag_output_to_params_count("meta_loss_unrolled -> lgn", meta_loss_unrolled, lgn.parameters(), i)
                    _diag_output_to_params("meta_loss_unrolled -> fast_params", meta_loss_unrolled, fast_params.values(), i)
                    _diag_output_to_params("proxy_loss -> lgn", proxy_loss, lgn.parameters(), i)
                    _diag_output_to_params("meta_loss -> lgn", meta_loss, lgn.parameters(), i)

                # Step 3: 反向传播贯穿整条链路
                #   meta_loss → fast_params → ∇proxy_loss → LGN weights → LGN params
                # LGN 更新仅使用 unrolled meta loss，不再混入 proxy aux 或 fallback。
                meta_probe_grads = torch.autograd.grad(
                    meta_loss_unrolled,
                    tuple(lgn.parameters()),
                    allow_unused=True,
                    retain_graph=True,
                    create_graph=False,
                )
                lgn_meta_probe_norm, lgn_meta_probe_nonfinite, lgn_meta_probe_elems = get_grad_norm_from_grads(meta_probe_grads)
                meta_grad_usable = (
                    lgn_meta_probe_elems > 0
                    and lgn_meta_probe_nonfinite == 0
                    and lgn_meta_probe_norm > 0.0
                    and math.isfinite(lgn_meta_probe_norm)
                )
                scaled_meta = scale_scalar_objective(meta_loss_unrolled)
                if not meta_grad_usable:
                    if _diag_should_log(i, args):
                        print(
                            f"[DIAG iter={i}] skip LGN step: unusable meta gradients "
                            f"(meta_probe_norm={lgn_meta_probe_norm:.6f}, "
                            f"meta_probe_nonfinite={int(lgn_meta_probe_nonfinite)}/{int(lgn_meta_probe_elems)})"
                        )
                    if term_log_now:
                        pbar.set_description(f"[{phase_str}] meta-grad unusable, LGN step skipped")
                else:
                    lgn_total = scaled_meta + float(args.lgn_potential_vref_weight) * loss_lgn_potential_vref

                    lgn_total.backward()
                    lgn_grad_norm, lgn_grad_max, lgn_grad_nonfinite, lgn_grad_elems = get_grad_stats(lgn)
                    if _diag_should_log(i, args):
                        print(
                            f"[DIAG iter={i}] lgn_grads: norm={lgn_grad_norm:.6f}, "
                            f"nonfinite={int(lgn_grad_nonfinite)}/{int(lgn_grad_elems)}, max={lgn_grad_max:.6f}"
                        )

                    lgn_clip_pre = float(nn.utils.clip_grad_norm_(lgn.parameters(), 1.0).item())
                    optim_lgn.step()
                    sanitize_module_(lgn, clamp_value=5.0)

                    lgn_update_loss = meta_loss_unrolled.detach()
    else:
        proxy_loss.backward()
        worker_grad_norm, worker_grad_max, worker_grad_nonfinite, worker_grad_elems = get_grad_stats(worknet)
        worker_clip_pre = float(nn.utils.clip_grad_norm_(worknet.parameters(), 1.0).item())
        optim_worker.step()
        sanitize_module_(worknet, clamp_value=10.0)
        sched.step()

    ###### D. Logging & Saving (Enhanced) ######
    if term_log_now:
        if train_lgn_phase:
            pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Unroll: {lgn_update_loss:.3f}")
        else:
            pbar.set_description(f"[{phase_str}] P-Loss: {proxy_loss:.3f} | M-Loss: {meta_loss:.3f}")
    
    with torch.no_grad():
        weights_raw_tb = weights_seq_raw.detach()
        weights_eff_tb = effective_weights_seq.detach()
        weights_raw_mean_tb = weights_raw_tb.mean(dim=(0, 1))
        weights_eff_mean_tb = weights_eff_tb.mean(dim=(0, 1))
        if weights_eff_tb.shape[0] > 0:
            weights_snapshot_eff_tb = weights_eff_tb[-1].mean(dim=0)
            snapshot_raw_flat_tb = weights_raw_tb[-1].reshape(-1)
        else:
            weights_snapshot_eff_tb = weights_eff_mean_tb
            snapshot_raw_flat_tb = weights_raw_tb.reshape(-1)
        collision_free = torch.all(dist_obj > 0, dim=0)
        reached_goal = torch.any(dist_to_goal < args.goal_radius, dim=0)
        success = collision_free & reached_goal
        final_dist_to_goal = dist_to_goal[-1]
        v_norm = v_history.norm(dim=-1)
        avg_speed = v_norm.mean()
        min_speed_threshold = float(env.max_speed) * 0.7
        act_cmd_mean = real_act_history.mean(dim=(0, 1))
        act_cmd_abs_mean = real_act_history.abs().mean(dim=(0, 1))
        act_cmd_norm_mean = real_act_history.norm(dim=-1).mean()
        if len(yaw_error_history) > 0:
            yaw_error_abs_deg = torch.stack(yaw_error_history).abs().mean() * (180.0 / math.pi)
            yaw_rate_env_abs_deg = getattr(env, "yaw_rate", torch.zeros((B, 1), device=device)).abs().mean() * (180.0 / math.pi)
        else:
            yaw_error_abs_deg = torch.tensor(0.0, device=device)
            yaw_rate_env_abs_deg = torch.tensor(0.0, device=device)
        if use_attitude_v2 and len(yaw_rate_cmd_history) > 0:
            yaw_rate_cmd_abs_deg = torch.stack(yaw_rate_cmd_history).abs().mean() * (180.0 / math.pi)
        else:
            yaw_rate_cmd_abs_deg = torch.tensor(0.0, device=device)
        heading_vel_error_deg = torch.rad2deg(torch.acos(
            heading_vel_cos.clamp(-1.0, 1.0)
        ))
        backward_ratio = ((heading_vel_cos < 0.0) & (speed_xy > float(args.heading_min_speed))).float().mean()
        side_backward_ratio = ((heading_vel_cos < 0.5) & (speed_xy > float(args.heading_min_speed))).float().mean()
        exploration_window_effective = float(min(max(0, int(args.exploration_time_window)), max(0, actual_T - 2)))
        guidance_backprop_in_phase = 1.0 if (train_lgn_phase and guidance_active_this_phase) else 0.0
        map_type_code = PRECOMPUTED_MAP_TYPE_CODES.get(current_precomputed_map_type, -2)
        map_stage_code = PRECOMPUTED_CURRICULUM_STAGE_CODES.get(current_precomputed_stage, -2)

        log_data = {
            # === 主要Loss ===
            'Loss/1_Proxy_Total': proxy_loss,
            'Loss/2_Meta_Total': meta_loss,

            # === [增强] Proxy Loss 原始分项 (Average over Time & Batch) ===
            'Proxy_Comp/0_SpeedPrefReward': weighted_speed_pref_reward_seq.mean(),
            'Proxy_Comp/0_1_SpeedResponse': speed_response_seq.mean(),
            'Proxy_Comp/0_2_DeltaSpeedSigned': delta_speed_signed.mean(),
            'Proxy_Comp/0_3_DeltaSpeedAbs': delta_speed_signed.abs().mean(),
            'Proxy_Comp/1_Direction': loss_direction_seq.mean(),
            'Proxy_Comp/2_Avoidance': loss_avoidance_seq.mean(),
            'Proxy_Comp/2_1_Collision_Depth': collision_depth.mean(),#穿入墙体深度
            'Proxy_Comp/3_Exploration': loss_exploration_seq.mean(),
            'Proxy_Comp/4_Turn': loss_turn_seq.mean(),
            'Proxy_Comp/5_LGN_VRef': loss_lgn_vref_seq.mean(),
            'Proxy_Comp/4_0_Turn_Base': loss_turn_base_seq.mean(),
            'Proxy_Comp/4_2_Yaw_Smooth': loss_yaw_smooth,
            'Proxy_Comp/5_Height': loss_height_seq.mean(),
            'Loss/Proxy_LGN_VRef': loss_lgn_vref_seq.mean(),
            'LGN/VRef_Weight': weights_eff_mean_tb[6],
            'LGN/VRef_Norm': safe_l2_norm(vref_body_seq, dim=-1).mean(),
            'LGN/VRef_Real_Cos': lgn_vref_real_cos_seq.mean(),
            'LGN/VRef_Worker_Cos': lgn_vref_real_cos_seq.mean(),
            'LGN/Potential_VRef_Loss': loss_lgn_potential_vref,
            'LGN/Potential_VRef_Weight': float(args.lgn_potential_vref_weight),
            'LGN/Potential_VRef_Valid_Ratio': lgn_potential_vref_components['valid_ratio'],
            'Stuck/Ratio': stuck_ratio,
            'Stuck/Collision_Streak_Mean': loss_collision_duration_seq.mean(),
            'Stuck/Collision_Streak_Max': loss_collision_duration_seq.max(),
            'Yaw/Heading_Vel_Error_Deg': heading_vel_error_deg.mean(),
            'Yaw/Backward_Ratio': backward_ratio,
            'Yaw/Side_Backward_Ratio': side_backward_ratio,
            'Yaw/Heading_Safety_Loss': loss_heading_safety,
            'Yaw/Heading_Align_Loss': loss_heading_align_seq.mean(),
            'Yaw/Backward_Loss': loss_backward_seq.mean(),
            'Yaw/Yaw_Smooth_Loss': loss_yaw_smooth,

            # === [增强] Meta Loss 分项 ===
            'Meta_Comp/1_Position': loss_meta_pos,
            'Meta_Comp/0_Arrival_Reward': loss_meta_arrival_reward,
            'Meta_Comp/0_Arrival_Term': -args.meta_arrival_reward_weight * loss_meta_arrival_reward,
            'Meta_Comp/2_Collision': loss_meta_coll,
            'Meta_Comp/2_1_Collision_Depth': collision_depth.mean(),
            'Meta_Comp/3_Control': loss_meta_ctrl,
            'Meta_Comp/4_Height': loss_meta_height,
            'Meta_Comp/6_Stuck': loss_meta_stuck,
            'Meta_Comp/8_Smooth_Jerk': loss_meta_jerk,
            'Meta_Comp/9_Smooth_Snap': loss_meta_snap,
            'Meta_Comp/11_Heading_Safety': loss_heading_safety,
            'Meta_Comp/12_Yaw_Smooth': loss_yaw_smooth,
            'Guidance/Applied_In_Current_Phase': 1.0 if guidance_active_this_phase else 0.0,
            'Guidance/Backprop_In_Current_Phase': guidance_backprop_in_phase,

            # === 性能指标 ===
            'Metrics/Success_Rate': success.float().mean(),
            'Metrics/No_Collision_Rate': collision_free.float().mean(),
            'Metrics/Reach_Goal_Rate': reached_goal.float().mean(),
            'Metrics/Arrival_Reward_Hit_Rate': meta_arrival_hit_rate,
            'Metrics/Arrival_Reward_Best_Dist': meta_arrival_best_dist,
            'Metrics/Final_Dist_To_Goal': final_dist_to_goal.mean(),
            'Metrics/Final_Dist_To_Goal_Min': final_dist_to_goal.min(),
            'Metrics/Final_Dist_To_Goal_Max': final_dist_to_goal.max(),
            'Metrics/Avg_Speed': avg_speed,
            'Metrics/Speed_Below_Threshold': (avg_speed < min_speed_threshold).float(),
            'Metrics/Min_Speed': v_norm.min(),
            'Metrics/Max_Speed': v_norm.max(),
            'Metrics/Episode_Length': actual_T,
            'Control/Accel_Cmd_Norm_Mean': act_cmd_norm_mean,
            'Control/Accel_Cmd_X_Mean': act_cmd_mean[0],
            'Control/Accel_Cmd_Y_Mean': act_cmd_mean[1],
            'Control/Accel_Cmd_Z_Mean': act_cmd_mean[2],
            'Control/Accel_Cmd_X_AbsMean': act_cmd_abs_mean[0],
            'Control/Accel_Cmd_Y_AbsMean': act_cmd_abs_mean[1],
            'Control/Accel_Cmd_Z_AbsMean': act_cmd_abs_mean[2],

            'Status/Exploration_Window_Effective': exploration_window_effective,
            'Status/Guidance_All_Phases': 1.0 if args.guidance_all_phases else 0.0,

            # === Precomputed map curriculum ===
            'Map/Precomputed_Enabled': 1.0 if args.use_precomputed_potential_maps else 0.0,
            'Map/Current_Index': current_precomputed_map_idx,
            'Map/Type_Code': map_type_code,
            'Map/Curriculum_Stage_Code': map_stage_code,
            'Map/Is_Easy': 1.0 if current_precomputed_map_type == "easy" else 0.0,
            'Map/Is_Hairpin': 1.0 if current_precomputed_map_type == "hairpin" else 0.0,
            'Map/Is_U_Min': 1.0 if current_precomputed_map_type == "u_min" else 0.0,

            # === [兼容旧日志] Debug 权重动态 ===
            'Weights/0_SpeedPref_Signed': weights_eff_mean_tb[0],
            'Weights/0_1_SpeedPref_Strength': weights_eff_mean_tb[1],
            'Weights/1_Direction': weights_eff_mean_tb[2],
            'Weights/2_Avoidance': weights_eff_mean_tb[3],
            'Weights/3_Exploration': weights_eff_mean_tb[4],
            'Weights/4_Turn': weights_eff_mean_tb[5],
            'Weights/5_VRef': weights_eff_mean_tb[6],
            'Weights_Raw/0_SpeedPref_Signed': weights_raw_mean_tb[0],
            'Weights_Raw/0_1_SpeedPref_Strength': weights_raw_mean_tb[1],
            'Weights_Raw/1_Direction': weights_raw_mean_tb[2],
            'Weights_Raw/2_Avoidance': weights_raw_mean_tb[3],
            'Weights_Raw/3_Exploration': weights_raw_mean_tb[4],
            'Weights_Raw/4_Turn': weights_raw_mean_tb[5],
            'Weights_Raw/5_VRef': weights_raw_mean_tb[6],
            'Weights_Effective/0_SpeedPref_Signed': weights_eff_mean_tb[0],
            'Weights_Effective/0_1_SpeedPref_Strength': weights_eff_mean_tb[1],
            'Weights_Effective/1_Direction': weights_eff_mean_tb[2],
            'Weights_Effective/2_Avoidance': weights_eff_mean_tb[3],
            'Weights_Effective/3_Exploration': weights_eff_mean_tb[4],
            'Weights_Effective/4_Turn': weights_eff_mean_tb[5],
            'Weights_Effective/5_VRef': weights_eff_mean_tb[6],
            'Weight_Stats/Raw_Min': weights_raw_tb.min(),
            'Weight_Stats/Raw_Max': weights_raw_tb.max(),
            'Weight_Stats/Raw_Mean': weights_raw_tb.mean(),
            'Weight_Stats/Std': weights_eff_tb.std(unbiased=False),
            'Weights_Snapshot/0_SpeedPref_Signed': weights_snapshot_eff_tb[0],
            'Weights_Snapshot/0_1_SpeedPref_Strength': weights_snapshot_eff_tb[1],
            'Weights_Snapshot/1_Direction': weights_snapshot_eff_tb[2],
            'Weights_Snapshot/2_Avoidance': weights_snapshot_eff_tb[3],
            'Weights_Snapshot/3_Exploration': weights_snapshot_eff_tb[4],
            'Weights_Snapshot/4_Turn': weights_snapshot_eff_tb[5],
            'Weights_Snapshot/5_VRef': weights_snapshot_eff_tb[6],
            'Weights_Snapshot/Std': weights_snapshot_eff_tb.std(unbiased=False),
            'Weights_Snapshot/Raw_Min': snapshot_raw_flat_tb.min(),
            'Weights_Snapshot/Raw_Max': snapshot_raw_flat_tb.max(),
            'Weights_Snapshot/Raw_Mean': snapshot_raw_flat_tb.mean(),
        }

        if use_attitude_v2:
            log_data['Heading/Yaw_Error_Abs_Deg'] = yaw_error_abs_deg
            log_data['Heading/Yaw_Rate_Env_Abs_Deg'] = yaw_rate_env_abs_deg
            log_data['Heading/Yaw_Rate_Cmd_Abs_Deg'] = yaw_rate_cmd_abs_deg

        if guidance_active_this_phase:
            log_data.update({
                'Meta_Comp/5_Guidance': loss_meta_guidance,
                'Guidance/Dir_Align': guidance_components['dir_align'],
                'Guidance/Overspeed': guidance_components['overspeed'],
                'Guidance/Underspeed': guidance_components.get('underspeed', 0.0),
                'Guidance/Speed_Diff': guidance_components.get('speed_diff', 0.0),
                'Guidance/Escape': guidance_components['escape'],
                'Guidance/Depth': guidance_components['depth'],
                'Guidance/Valid_Ratio': guidance_components['valid_ratio'],
                'Guidance/Collision_Ratio': guidance_components['collision_ratio'],
                'Guidance/Boost': guidance_components.get('guidance_boost', 1.0),
                'Guidance/Sample_Count': guidance_components['sample_count'],
                'Guidance/Avg_Ref_Speed': guidance_components.get('avg_ref_speed', 0.0),
                'Guidance/Avg_Lateral_Error': guidance_components.get('avg_lateral_error', 0.0),
                'Guidance/Max_Lateral_Error': guidance_components.get('max_lateral_error', 0.0),
                'Guidance/Field_Dir_Align': guidance_components.get('field_dir_align', 0.0),
            })

        if train_lgn_phase:
            log_data.update({
                'Grad/LGN_Max_Abs': lgn_grad_max,
                'Grad/LGN_NonFinite_Count': lgn_grad_nonfinite,
                'Grad/LGN_GradElem_Count': lgn_grad_elems,
                'Grad/LGN_Clip_PreNorm': lgn_clip_pre,
                'Grad/LGN_MetaProbe_NonFinite_Count': lgn_meta_probe_nonfinite,
                'Grad/LGN_MetaProbe_GradElem_Count': lgn_meta_probe_elems,
            })
        else:
            log_data.update({
                'Grad/Worker_Global_Norm': worker_grad_norm,
                'Grad/Worker_Max_Abs': worker_grad_max,
                'Grad/Worker_NonFinite_Count': worker_grad_nonfinite,
                'Grad/Worker_GradElem_Count': worker_grad_elems,
                'Grad/Worker_Clip_PreNorm': worker_clip_pre,
            })

        if train_lgn_phase:
            log_data['Loss/3_LGN_Unrolled_Meta'] = lgn_update_loss
            log_data['Meta_Unrolled/1_Position'] = meta_pos_ur
            log_data['Meta_Unrolled/0_Arrival_Reward'] = meta_arrival_reward_ur
            log_data['Meta_Unrolled/0_Arrival_Hit_Rate'] = meta_arrival_hit_rate_ur
            log_data['Meta_Unrolled/0_Arrival_Best_Dist'] = meta_arrival_best_dist_ur
            log_data['Meta_Unrolled/2_Collision'] = meta_coll_ur
            log_data['Meta_Unrolled/3_Control'] = meta_ctrl_ur

        if geom_feat_last is not None and progress_feat_last is not None:
            log_data['LGN_Input/Geom_Mean'] = geom_feat_last.mean()
            log_data['LGN_Input/Geom_Std'] = geom_feat_last.std(unbiased=False)
            log_data['LGN_Input/Geom_Norm'] = geom_feat_last.norm(dim=-1).mean()
            log_data['LGN_Input/Progress_Mean'] = progress_feat_last.mean()
            log_data['LGN_Input/Progress_Std'] = progress_feat_last.std(unbiased=False)
            log_data['LGN_Input/Progress_Norm'] = progress_feat_last.norm(dim=-1).mean()
            for feat_idx in range(min(4, geom_feat_last.shape[-1])):
                log_data[f'LGN_Input/Geom_{feat_idx}'] = geom_feat_last[:, feat_idx].mean()
            for feat_idx in range(min(4, progress_feat_last.shape[-1])):
                log_data[f'LGN_Input/Progress_{feat_idx}'] = progress_feat_last[:, feat_idx].mean()

        active_map_log_key = _resolve_map_log_key(current_precomputed_map_type, map_writers)
        active_tb_writer = _resolve_tb_writer(current_precomputed_map_type, writer, map_writers)
        smooth_dict(log_data, scaler_q_by_map, map_log_key=active_map_log_key)
        if tb_log_now:
            writer_q = scaler_q_by_map[active_map_log_key]
            active_tb_writer.add_scalar("Status_Raw/Train_LGN_Phase", float(train_lgn_phase), i + 1)
            active_tb_writer.add_scalar('Status/Train_Mode', 1.0 if train_lgn_phase else 0.0, i + 1)
            active_tb_writer.add_scalar('Status/Maze_Age', (maze_update_counter - 1) % args.maze_update_interval, i + 1)
            for k, v in writer_q.items():
                active_tb_writer.add_scalar(k, sum(v) / len(v), i + 1)
            writer_q.clear()

        if current_precomputed_map_type in VIZ_MAP_TYPES:
            idx = 0
            astar_paths_all = guidance_components.get('sampled_astar_paths', [])
            astar_paths_sampled = astar_paths_all[idx] if (isinstance(astar_paths_all, list) and idx < len(astar_paths_all)) else []
            depth_stack = torch.stack(depth_history).float() if len(depth_history) > 0 else None
            latest_viz_by_map_type[current_precomputed_map_type] = {
                'iter': i + 1,
                'map_type': current_precomputed_map_type,
                'map_idx': int(current_precomputed_map_idx),
                'map_file': str(current_precomputed_map_file),
                'stage': str(current_precomputed_stage),
                'env_snapshot': snapshot_env_for_viz(env, idx=idx),
                'p_cpu': p_history[:, idx].detach().cpu().clone(),
                'v_cpu': v_history[:, idx].detach().cpu().clone(),
                'rpy_cpu': rpy_history[:, idx].detach().cpu().clone(),
                'R_cpu': R_history[:, idx].detach().cpu().clone(),
                'act_cpu': real_act_history[:, idx].detach().cpu().clone(),
                'weights_cpu': effective_weights_seq[:, idx, :].detach().cpu().clone(),
                'depth_stack': depth_stack,
                'astar_paths_sampled': astar_paths_sampled,
            }

        if is_artifact_save_iter(i, args):
            save_step = i + 1
            selection_loss = float(meta_loss.detach().item())
            is_new_best = math.isfinite(selection_loss) and selection_loss < best_meta_loss

            torch.save(worknet.state_dict(), os.path.join(save_dir, f'worker_ckpt_{save_step:06d}.pth'))
            torch.save(lgn.state_dict(), os.path.join(save_dir, f'lgn_ckpt_{save_step:06d}.pth'))

            for map_type in VIZ_MAP_TYPES:
                record = latest_viz_by_map_type.get(map_type)
                if record is None:
                    continue
                save_cached_viz_record(
                    record,
                    i + 1,
                    args=args,
                    potential_map_cache=POTENTIAL_MAP_CACHE,
                    video_dir=video_dir,
                    writer=writer,
                )

            if is_new_best:
                best_meta_loss = selection_loss
                best_meta_loss_step = save_step
                best_artifact_label = f'best_step_{save_step:06d}'
                best_worker_file = f'worker_{best_artifact_label}.pth'
                best_lgn_file = f'lgn_{best_artifact_label}.pth'
                torch.save(worknet.state_dict(), os.path.join(save_dir, best_worker_file))
                torch.save(lgn.state_dict(), os.path.join(save_dir, best_lgn_file))

                best_record = latest_viz_by_map_type.get(current_precomputed_map_type)
                if best_record is not None:
                    save_cached_viz_record(
                        best_record,
                        save_step,
                        args=args,
                        potential_map_cache=POTENTIAL_MAP_CACHE,
                        video_dir=video_dir,
                        writer=writer,
                        artifact_label=best_artifact_label,
                    )

                best_metadata = {
                    'step': best_meta_loss_step,
                    'meta_loss': best_meta_loss,
                    'worker_checkpoint': best_worker_file,
                    'lgn_checkpoint': best_lgn_file,
                    'artifact_label': best_artifact_label,
                    'map_type': current_precomputed_map_type,
                    'map_index': int(current_precomputed_map_idx),
                    'map_file': str(current_precomputed_map_file),
                }
                with open(os.path.join(save_dir, 'best_checkpoint.json'), 'w') as f:
                    json.dump(best_metadata, f, indent=4)
                writer.add_scalar('Best/Meta_Loss', best_meta_loss, save_step)
                writer.add_scalar('Best/Step', best_meta_loss_step, save_step)
                print(
                    f"[BestCheckpoint] iter={save_step} "
                    f"meta_loss={best_meta_loss:.6f} "
                    f"map={current_precomputed_map_type}:{current_precomputed_map_idx}"
                )

            try:
                sync_stats = sync_multi_pub_to_checkpoint_dir(save_dir)
                print(
                    f"[CheckpointSync] iter={save_step} "
                    f"src={sync_stats['src_root']} -> dst={sync_stats['dst_root']} "
                    f"copied={sync_stats['files_copied']} "
                    f"deleted_files={sync_stats['files_deleted']} "
                    f"deleted_dirs={sync_stats['dirs_deleted']}"
                )
            except Exception as e:
                print(f"[CheckpointSync][WARN] iter={save_step} sync failed: {e}")
            writer.flush()

print(f"Training Finished. Artifacts in: {save_dir}")
for _map_writer in map_writers.values():
    _map_writer.flush()
    _map_writer.close()
writer.flush()
writer.close()
