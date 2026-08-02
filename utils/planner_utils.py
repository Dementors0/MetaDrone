"""Global planning and guidance-loss helpers."""

import atexit
import heapq
import math
import multiprocessing as mp
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

try:
    from potential_map_utils import query_potential_guidance
except ModuleNotFoundError:
    query_potential_guidance = None

from .tensor_utils import safe_normalize, sanitize_tensor


PLANNER_PARALLEL_ENABLE = False
PLANNER_NUM_WORKERS = 0
PLANNER_POOL_MAXTASKS = 256
_PLANNER_POOL = None
_PLANNER_POOL_SIZE = 0


def compute_lgn_potential_vref_sync_loss(
    env,
    p_history,
    vref_body_seq,
    R_proxy_seq,
    config,
    potential_map_cache,
):
    loss = torch.tensor(0.0, device=vref_body_seq.device, dtype=vref_body_seq.dtype)
    components = {
        'valid_ratio': torch.tensor(0.0, device=vref_body_seq.device, dtype=vref_body_seq.dtype),
    }
    if (not config.use_precomputed_potential_maps) or potential_map_cache is None:
        return loss, components

    map_idx = int(getattr(env, 'current_map_idx', 0))
    map_data = potential_map_cache.get_map(map_idx)
    _, ref_dir_world, valid_mask = compute_guidance_reference_from_potential_map(
        p_history=p_history.detach(),
        map_data=map_data,
        interpolation=config.potential_interpolation,
    )
    ref_dir_body = torch.squeeze(ref_dir_world[:, :, None, :] @ R_proxy_seq.detach(), 2)
    cos_align = (
        safe_normalize(vref_body_seq, dim=-1) * safe_normalize(ref_dir_body, dim=-1)
    ).sum(-1).clamp(-1.0, 1.0)
    loss_map = (1.0 - cos_align).clamp(0.0, 2.0)
    loss = _masked_mean(loss_map, valid_mask)
    components['valid_ratio'] = valid_mask.float().mean()
    return loss, components


class GlobalPlanner:
    """
    3D A* 全局路径规划器

    构建占用栅格地图并使用 A* 算法规划从起点到终点的最优路径，
    然后从路径中提取参考方向、速度和加速度。
    """

    def __init__(self, resolution: float = 0.3, margin: float = 0.15,
                 z_min: float = 0.0, z_max: float = 2.5, device='cuda'):
        """
        Args:
            resolution: 栅格分辨率 (米)
            margin: 安全边距 (米)，障碍物膨胀量
            z_min, z_max: Z轴范围
            device: 计算设备
        """
        self.resolution = resolution
        self.margin = margin
        self.z_min = z_min
        self.z_max = z_max
        self.device = device

        # 缓存的占用栅格地图
        self.occupancy_grid = None
        self.grid_origin = None  # [x_min, y_min, z_min]
        self.grid_shape = None   # [nx, ny, nz]

        # 缓存的规划路径 (每个 batch 元素一条路径)
        self.cached_paths = {}   # batch_idx -> path tensor [N, 3]
        self.plan_stats = {'success': 0, 'total': 0}

        # 3D 邻居偏移 (26-连通)
        self._neighbors_26 = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cost = math.sqrt(dx**2 + dy**2 + dz**2)
                    self._neighbors_26.append((dx, dy, dz, cost))

    def build_occupancy_grid(self, env, batch_idx: int = 0):
        """
        从环境障碍物构建 3D 占用栅格地图

        Args:
            env: 环境对象，包含 voxels, balls, cyl, cyl_h 等障碍物
            batch_idx: batch 索引
        """
        # 确定地图边界
        x_min, x_max = -15.0, 15.0
        y_min, y_max = -25.0, 25.0

        if hasattr(env, 'p_target') and env.p_target is not None:
            target = env.p_target[batch_idx].detach().cpu()
            y_min = min(y_min, float(target[1]) - 5.0)
            y_max = max(y_max, float(target[1]) + 5.0)

        # 栅格尺寸
        nx = int(math.ceil((x_max - x_min) / self.resolution))
        ny = int(math.ceil((y_max - y_min) / self.resolution))
        nz = int(math.ceil((self.z_max - self.z_min) / self.resolution))

        self.grid_origin = np.array([x_min, y_min, self.z_min])
        self.grid_shape = (nx, ny, nz)

        # 初始化为空闲
        self.occupancy_grid = np.zeros((nx, ny, nz), dtype=np.uint8)

        # 填充障碍物
        total_margin = self.margin + 0.15  # 额外的安全边距

        # 1. 体素盒体障碍物
        if hasattr(env, 'voxels') and env.voxels.numel() > 0:
            voxels = env.voxels[batch_idx].detach().cpu().numpy()
            for vox in voxels:
                cx, cy, cz, hx, hy, hz = vox[:6]
                if hx > 15 or hy > 15 or hz > 15:  # 跳过占位大盒体
                    continue
                # 膨胀障碍物
                x0 = max(0, int((cx - hx - total_margin - x_min) / self.resolution))
                x1 = min(nx, int((cx + hx + total_margin - x_min) / self.resolution) + 1)
                y0 = max(0, int((cy - hy - total_margin - y_min) / self.resolution))
                y1 = min(ny, int((cy + hy + total_margin - y_min) / self.resolution) + 1)
                z0 = max(0, int((cz - hz - total_margin - self.z_min) / self.resolution))
                z1 = min(nz, int((cz + hz + total_margin - self.z_min) / self.resolution) + 1)
                self.occupancy_grid[x0:x1, y0:y1, z0:z1] = 1

        # 2. 球形障碍物
        if hasattr(env, 'balls') and env.balls.numel() > 0:
            balls = env.balls[batch_idx].detach().cpu().numpy()
            for ball in balls:
                bx, by, bz, br = ball[:4]
                r_inflated = br + total_margin
                # 球的包围盒
                x0 = max(0, int((bx - r_inflated - x_min) / self.resolution))
                x1 = min(nx, int((bx + r_inflated - x_min) / self.resolution) + 1)
                y0 = max(0, int((by - r_inflated - y_min) / self.resolution))
                y1 = min(ny, int((by + r_inflated - y_min) / self.resolution) + 1)
                z0 = max(0, int((bz - r_inflated - self.z_min) / self.resolution))
                z1 = min(nz, int((bz + r_inflated - self.z_min) / self.resolution) + 1)
                # 精确球形检测
                for ix in range(x0, x1):
                    for iy in range(y0, y1):
                        for iz in range(z0, z1):
                            px = x_min + (ix + 0.5) * self.resolution
                            py = y_min + (iy + 0.5) * self.resolution
                            pz = self.z_min + (iz + 0.5) * self.resolution
                            if (px - bx)**2 + (py - by)**2 + (pz - bz)**2 < r_inflated**2:
                                self.occupancy_grid[ix, iy, iz] = 1

        # 3. 竖直圆柱障碍物 (沿Z轴)
        if hasattr(env, 'cyl') and env.cyl.numel() > 0:
            cyl = env.cyl[batch_idx].detach().cpu().numpy()
            for c in cyl:
                cx, cy, cr = c[:3]
                r_inflated = cr + total_margin
                x0 = max(0, int((cx - r_inflated - x_min) / self.resolution))
                x1 = min(nx, int((cx + r_inflated - x_min) / self.resolution) + 1)
                y0 = max(0, int((cy - r_inflated - y_min) / self.resolution))
                y1 = min(ny, int((cy + r_inflated - y_min) / self.resolution) + 1)
                for ix in range(x0, x1):
                    for iy in range(y0, y1):
                        px = x_min + (ix + 0.5) * self.resolution
                        py = y_min + (iy + 0.5) * self.resolution
                        if (px - cx)**2 + (py - cy)**2 < r_inflated**2:
                            self.occupancy_grid[ix, iy, :] = 1

        # 4. 水平圆柱障碍物 (沿Y轴)
        if hasattr(env, 'cyl_h') and env.cyl_h.numel() > 0:
            cyl_h = env.cyl_h[batch_idx].detach().cpu().numpy()
            for c in cyl_h:
                cx, cz, cr = c[:3]
                r_inflated = cr + total_margin
                x0 = max(0, int((cx - r_inflated - x_min) / self.resolution))
                x1 = min(nx, int((cx + r_inflated - x_min) / self.resolution) + 1)
                z0 = max(0, int((cz - r_inflated - self.z_min) / self.resolution))
                z1 = min(nz, int((cz + r_inflated - self.z_min) / self.resolution) + 1)
                for ix in range(x0, x1):
                    for iz in range(z0, z1):
                        px = x_min + (ix + 0.5) * self.resolution
                        pz = self.z_min + (iz + 0.5) * self.resolution
                        if (px - cx)**2 + (pz - cz)**2 < r_inflated**2:
                            self.occupancy_grid[ix, :, iz] = 1

        return self.occupancy_grid

    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """世界坐标转栅格索引"""
        idx = ((pos - self.grid_origin) / self.resolution).astype(int)
        return tuple(np.clip(idx, 0, np.array(self.grid_shape) - 1))

    def grid_to_world(self, idx: Tuple[int, int, int]) -> np.ndarray:
        """栅格索引转世界坐标（单元格中心）"""
        return self.grid_origin + (np.array(idx) + 0.5) * self.resolution

    def is_valid(self, idx: Tuple[int, int, int]) -> bool:
        """检查栅格索引是否有效且空闲"""
        nx, ny, nz = self.grid_shape
        if not (0 <= idx[0] < nx and 0 <= idx[1] < ny and 0 <= idx[2] < nz):
            return False
        return self.occupancy_grid[idx[0], idx[1], idx[2]] == 0

    def heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """A* 启发式函数（欧几里得距离）"""
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) * self.resolution

    def plan_astar(self, start_pos: np.ndarray, goal_pos: np.ndarray,
                   max_iterations: int = 50000) -> Optional[List[np.ndarray]]:
        """
        3D A* 路径规划

        Args:
            start_pos: 起点世界坐标 [3]
            goal_pos: 终点世界坐标 [3]
            max_iterations: 最大迭代次数

        Returns:
            path: 路径点列表 [N, 3] 或 None（规划失败）
        """
        if self.occupancy_grid is None:
            return None

        start_idx = self.world_to_grid(start_pos)
        goal_idx = self.world_to_grid(goal_pos)

        # 如果起点在障碍物内，尝试找到最近的空闲点
        if not self.is_valid(start_idx):
            start_idx = self._find_nearest_free(start_idx)
            if start_idx is None:
                return None

        # 如果终点在障碍物内，尝试找到最近的空闲点
        if not self.is_valid(goal_idx):
            goal_idx = self._find_nearest_free(goal_idx)
            if goal_idx is None:
                return None

        # A* 搜索
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start_idx, goal_idx), 0, start_idx))

        came_from = {}
        g_score = {start_idx: 0}

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current_g, current = heapq.heappop(open_set)

            # 到达目标
            if current == goal_idx:
                # 重建路径
                path = [self.grid_to_world(current)]
                while current in came_from:
                    current = came_from[current]
                    path.append(self.grid_to_world(current))
                path.reverse()
                return path

            # 跳过过期节点
            if current_g > g_score.get(current, float('inf')):
                continue

            # 扩展邻居
            for dx, dy, dz, move_cost in self._neighbors_26:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                if not self.is_valid(neighbor):
                    continue

                tentative_g = g_score[current] + move_cost * self.resolution

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal_idx)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # 规划失败
        return None

    def _find_nearest_free(self, idx: Tuple[int, int, int], search_radius: int = 10) -> Optional[Tuple[int, int, int]]:
        """寻找最近的空闲栅格"""
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if abs(dx) == r or abs(dy) == r or abs(dz) == r:
                            neighbor = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                            if self.is_valid(neighbor):
                                return neighbor
        return None

    def smooth_path(self, path: List[np.ndarray], window_size: int = 5) -> np.ndarray:
        """
        路径平滑处理（移动平均）

        Args:
            path: 原始路径点列表
            window_size: 平滑窗口大小

        Returns:
            smoothed_path: 平滑后的路径 [N, 3]
        """
        if len(path) < 3:
            return np.array(path)

        path_array = np.array(path)
        smoothed = np.copy(path_array)

        half_w = window_size // 2
        for i in range(half_w, len(path) - half_w):
            smoothed[i] = path_array[max(0, i-half_w):min(len(path), i+half_w+1)].mean(axis=0)

        # 保持起点和终点不变
        smoothed[0] = path_array[0]
        smoothed[-1] = path_array[-1]

        return smoothed

    def extract_reference_from_path(self, current_pos: np.ndarray, path: np.ndarray,
                                    max_speed: float = 5.0,
                                    max_accel: float = 5.0,
                                    max_decel: float = 6.0,
                                    lookahead_dist: float = 1.0) -> Dict:
        """
        从规划路径中提取当前位置的参考方向、速度和加速度

        使用梯形速度剖面和动力学约束计算参考值：
        - 速度剖面考虑：最大速度、曲率限制、终点减速
        - 加速度基于速度剖面的变化率计算
        - 横向偏差为到路径的真正几何距离

        Args:
            current_pos: 当前位置 [3]
            path: 规划路径 [N, 3]
            max_speed: 最大速度 (m/s)
            max_accel: 最大加速度 (m/s^2)
            max_decel: 最大减速度 (m/s^2)，正值
            lookahead_dist: 前瞻距离 (m)

        Returns:
            dict: 包含参考方向、速度、加速度、曲率、横向偏差等
        """
        if path is None or len(path) < 2:
            return {
                'direction': np.array([0.0, 1.0, 0.0]),
                'speed': max_speed * 0.5,
                'acceleration': np.array([0.0, 0.0, 0.0]),
                'curvature': 0.0,
                'path_progress': 0.0,
                'dist_to_goal': 10.0,
                'lateral_error': 0.0,  # 横向偏差
                'valid': False
            }

        # ========== 1. 找到路径上最近点并计算横向偏差 ==========
        distances = np.linalg.norm(path - current_pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_point = path[nearest_idx]

        # 横向偏差：当前位置到路径最近点的距离
        lateral_error = distances[nearest_idx]

        # 更精确的横向偏差：投影到路径线段上
        if 0 < nearest_idx < len(path) - 1:
            # 检查是否应该投影到前一段或后一段
            for seg_start, seg_end in [(nearest_idx - 1, nearest_idx), (nearest_idx, nearest_idx + 1)]:
                if seg_end >= len(path):
                    continue
                p0 = path[seg_start]
                p1 = path[seg_end]
                seg_vec = p1 - p0
                seg_len = np.linalg.norm(seg_vec)
                if seg_len > 1e-6:
                    seg_dir = seg_vec / seg_len
                    t = np.clip(np.dot(current_pos - p0, seg_dir), 0, seg_len)
                    proj_point = p0 + t * seg_dir
                    proj_dist = np.linalg.norm(current_pos - proj_point)
                    if proj_dist < lateral_error:
                        lateral_error = proj_dist
                        nearest_point = proj_point

        # ========== 2. 计算路径进度和剩余路径长度 ==========
        path_progress = nearest_idx / max(len(path) - 1, 1)

        # 计算从当前点到终点的路径长度
        remaining_path_length = 0.0
        for i in range(nearest_idx, len(path) - 1):
            remaining_path_length += np.linalg.norm(path[i+1] - path[i])
        dist_to_goal = remaining_path_length

        # ========== 3. 计算前瞻点和参考方向 ==========
        lookahead_idx = nearest_idx
        accumulated_dist = 0.0
        for i in range(nearest_idx, len(path) - 1):
            segment_dist = np.linalg.norm(path[i+1] - path[i])
            accumulated_dist += segment_dist
            if accumulated_dist >= lookahead_dist:
                lookahead_idx = i + 1
                break
        else:
            lookahead_idx = len(path) - 1

        lookahead_point = path[lookahead_idx]
        direction_vec = lookahead_point - current_pos
        direction_norm = np.linalg.norm(direction_vec)
        if direction_norm > 1e-6:
            direction = direction_vec / direction_norm
        else:
            if nearest_idx < len(path) - 1:
                direction = path[nearest_idx + 1] - path[nearest_idx]
                direction = direction / (np.linalg.norm(direction) + 1e-6)
            else:
                direction = np.array([0.0, 1.0, 0.0])

        # ========== 4. 计算局部曲率（用于速度限制）==========
        curvature = 0.0
        if 1 <= nearest_idx < len(path) - 1:
            v1 = path[nearest_idx] - path[nearest_idx - 1]
            v2 = path[nearest_idx + 1] - path[nearest_idx]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm > 1e-6 and v2_norm > 1e-6:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle_change = math.acos(cos_angle)
                curvature = angle_change / (self.resolution + 1e-6)

        # ========== 5. 计算梯形速度剖面 ==========
        # 5.1 曲率速度限制：v_max_curve = sqrt(a_lateral_max / curvature)
        a_lateral_max = 4.0  # 最大侧向加速度 (m/s^2)
        if curvature > 0.1:
            v_curve_limit = min(max_speed, math.sqrt(a_lateral_max / (curvature + 1e-6)))
        else:
            v_curve_limit = max_speed

        # 5.2 终点减速限制：v^2 = 2 * a_decel * d
        # 在终点速度应为 0，所以 v_max_decel = sqrt(2 * max_decel * dist_to_goal)
        v_decel_limit = math.sqrt(2.0 * max_decel * max(dist_to_goal, 0.01))

        # 5.3 前方曲率预瞰（提前减速）
        v_lookahead_limit = max_speed
        lookahead_curvature_dist = 2.0  # 向前看 2m
        accumulated = 0.0
        max_future_curvature = 0.0
        for i in range(nearest_idx, min(nearest_idx + 20, len(path) - 1)):
            seg_len = np.linalg.norm(path[i+1] - path[i])
            accumulated += seg_len
            if accumulated > lookahead_curvature_dist:
                break
            if 1 <= i < len(path) - 1:
                v1 = path[i] - path[i-1]
                v2 = path[i+1] - path[i]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_a = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
                    future_curv = math.acos(cos_a) / (self.resolution + 1e-6)
                    max_future_curvature = max(max_future_curvature, future_curv)

        if max_future_curvature > 0.1:
            v_future = math.sqrt(a_lateral_max / (max_future_curvature + 1e-6))
            # 需要在 lookahead_curvature_dist 内减速到 v_future
            v_lookahead_limit = math.sqrt(v_future**2 + 2.0 * max_decel * lookahead_curvature_dist)

        # 5.4 综合速度限制
        ref_speed = min(max_speed, v_curve_limit, v_decel_limit, v_lookahead_limit)
        ref_speed = max(ref_speed, 0.1)  # 最小速度

        # ========== 6. 计算参考加速度（基于速度剖面变化）==========
        # 切向加速度：基于当前位置的速度限制变化
        tangent_accel = 0.0

        # 如果当前速度限制低于最大速度，说明需要减速
        if ref_speed < max_speed * 0.9:
            # 估算需要的减速度
            if dist_to_goal > 0.1:
                # v_final^2 - v_current^2 = 2 * a * d
                # 假设需要在 dist_to_goal 内减到 0
                tangent_accel = -(ref_speed ** 2) / (2 * max(dist_to_goal, 0.1))
                tangent_accel = max(tangent_accel, -max_decel)
            else:
                tangent_accel = -max_decel

        # 如果在弯道，额外减速
        if curvature > 0.5:
            curve_decel = -curvature * 1.5  # 曲率越大，减速越多
            tangent_accel = min(tangent_accel, curve_decel)

        # 限制加速度范围
        tangent_accel = np.clip(tangent_accel, -max_decel, max_accel)

        # 加速度向量 = 切向加速度 * 方向
        acceleration = direction * tangent_accel

        return {
            'direction': direction,
            'speed': ref_speed,
            'acceleration': acceleration,
            'curvature': curvature,
            'path_progress': path_progress,
            'dist_to_goal': dist_to_goal,
            'lateral_error': lateral_error,  # 新增：横向偏差
            'valid': True
        }

    def plan_and_cache(self, env, start_positions: torch.Tensor,
                       goal_positions: torch.Tensor, batch_indices: List[int] = None):
        """
        为多个 batch 元素规划并缓存路径

        Args:
            env: 环境对象
            start_positions: 起点位置 [B, 3]
            goal_positions: 目标位置 [B, 3]
            batch_indices: 要规划的 batch 索引列表，None 表示全部
        """
        B = start_positions.shape[0]
        if batch_indices is None:
            batch_indices = list(range(B))

        success_count = 0

        for b in batch_indices:
            # 为该 batch 构建占用栅格
            self.build_occupancy_grid(env, batch_idx=b)

            start = start_positions[b].detach().cpu().numpy()
            goal = goal_positions[b].detach().cpu().numpy()

            # A* 规划
            path = self.plan_astar(start, goal)

            if path is not None:
                # 路径平滑
                path = self.smooth_path(path, window_size=5)
                self.cached_paths[b] = path
                success_count += 1
            else:
                # 规划失败：标记为 None，由恢复项接管，避免强制拟合不可靠参考
                self.cached_paths[b] = None

        self.plan_stats = {
            'success': int(success_count),
            'total': int(len(batch_indices)),
        }

    def clear_cache(self):
        """清除缓存的路径"""
        self.cached_paths.clear()
        self.occupancy_grid = None


def configure_planner_pool(enabled, num_workers=0, maxtasks_per_child=256):
    """Configure the lazily-created multiprocessing planner pool."""
    global PLANNER_PARALLEL_ENABLE, PLANNER_NUM_WORKERS, PLANNER_POOL_MAXTASKS
    new_config = (
        bool(enabled),
        int(num_workers),
        max(1, int(maxtasks_per_child)),
    )
    old_config = (
        PLANNER_PARALLEL_ENABLE,
        PLANNER_NUM_WORKERS,
        PLANNER_POOL_MAXTASKS,
    )
    if new_config != old_config:
        _shutdown_planner_pool()
    (
        PLANNER_PARALLEL_ENABLE,
        PLANNER_NUM_WORKERS,
        PLANNER_POOL_MAXTASKS,
    ) = new_config


def _compute_planner_worker_count():
    if PLANNER_NUM_WORKERS > 0:
        return PLANNER_NUM_WORKERS
    cpu_total = os.cpu_count() or 8
    # 在高核机器上预留少量核给训练主进程与系统线程
    return max(1, min(28, cpu_total - 3))


def _shutdown_planner_pool():
    global _PLANNER_POOL, _PLANNER_POOL_SIZE
    if _PLANNER_POOL is not None:
        _PLANNER_POOL.close()
        _PLANNER_POOL.join()
        _PLANNER_POOL = None
        _PLANNER_POOL_SIZE = 0


atexit.register(_shutdown_planner_pool)


def _get_planner_pool():
    global _PLANNER_POOL, _PLANNER_POOL_SIZE
    if not PLANNER_PARALLEL_ENABLE:
        return None

    target_size = _compute_planner_worker_count()
    if _PLANNER_POOL is not None and _PLANNER_POOL_SIZE == target_size:
        return _PLANNER_POOL

    if _PLANNER_POOL is not None:
        _shutdown_planner_pool()

    # 本脚本顶层直接启动训练，spawn 会重入模块；Linux 下优先 fork。
    start_method = 'fork' if sys.platform.startswith('linux') else 'spawn'
    ctx = mp.get_context(start_method)
    _PLANNER_POOL = ctx.Pool(processes=target_size, maxtasksperchild=PLANNER_POOL_MAXTASKS)
    _PLANNER_POOL_SIZE = target_size
    return _PLANNER_POOL


def _plan_sample_points_worker(payload):
    """Per-sample planning worker: build occupancy once and solve all sampled points."""
    voxels_np = payload['voxels']
    balls_np = payload['balls']
    cyl_np = payload['cyl']
    cyl_h_np = payload['cyl_h']
    sampled_positions = payload['sampled_positions']  # [S, 3]
    sampled_dist = payload['sampled_dist']            # [S]
    sampled_steps = payload.get('sampled_steps', None)
    goal_np = payload['goal']                         # [3]
    resolution = float(payload['resolution'])
    margin = float(payload['margin'])
    z_min = float(payload['z_min'])
    z_max = float(payload['z_max'])
    max_speed = float(payload['max_speed'])
    max_accel = float(payload['max_accel'])
    max_decel = float(payload['max_decel'])
    lookahead_dist = float(payload['lookahead_dist'])
    invalid_dist_threshold = float(payload['invalid_dist_threshold'])

    planner = GlobalPlanner(resolution=resolution, margin=margin, z_min=z_min, z_max=z_max, device='cpu')

    class _EnvShim:
        pass

    env_shim = _EnvShim()
    env_shim.voxels = torch.from_numpy(voxels_np).unsqueeze(0)
    env_shim.balls = torch.from_numpy(balls_np).unsqueeze(0)
    env_shim.cyl = torch.from_numpy(cyl_np).unsqueeze(0)
    env_shim.cyl_h = torch.from_numpy(cyl_h_np).unsqueeze(0)
    env_shim.p_target = torch.from_numpy(goal_np).reshape(1, 3)

    planner.build_occupancy_grid(env_shim, batch_idx=0)

    S = sampled_positions.shape[0]
    ref_direction = np.zeros((S, 3), dtype=np.float32)
    ref_speed = np.zeros((S,), dtype=np.float32)
    ref_acceleration = np.zeros((S, 3), dtype=np.float32)
    lateral_error = np.zeros((S,), dtype=np.float32)
    valid_mask = np.zeros((S,), dtype=np.bool_)

    plan_total = 0
    plan_success = 0
    curv_sum = 0.0
    prog_sum = 0.0
    lat_sum = 0.0
    lat_max = 0.0
    metric_count = 0
    sampled_paths = []

    for s in range(S):
        pos_np = sampled_positions[s]
        dist = float(sampled_dist[s])

        step_t = int(sampled_steps[s]) if sampled_steps is not None else int(s)

        if dist < invalid_dist_threshold:
            sampled_paths.append((step_t, None))
            continue

        plan_total += 1
        path = planner.plan_astar(pos_np, goal_np)
        if path is None:
            sampled_paths.append((step_t, None))
            continue
        path = planner.smooth_path(path, window_size=5)
        path_np = np.asarray(path, dtype=np.float32)
        sampled_paths.append((step_t, path_np))

        ref_info = planner.extract_reference_from_path(
            pos_np,
            path,
            max_speed=max_speed,
            max_accel=max_accel,
            max_decel=max_decel,
            lookahead_dist=lookahead_dist,
        )

        ref_direction[s] = np.asarray(ref_info['direction'], dtype=np.float32)
        ref_speed[s] = float(ref_info['speed'])
        ref_acceleration[s] = np.asarray(ref_info['acceleration'], dtype=np.float32)
        lateral_error[s] = float(ref_info.get('lateral_error', 0.0))
        valid_mask[s] = bool(ref_info['valid'])
        if valid_mask[s]:
            plan_success += 1

        curv_sum += float(ref_info.get('curvature', 0.0))
        prog_sum += float(ref_info.get('path_progress', 0.0))
        lat = float(ref_info.get('lateral_error', 0.0))
        lat_sum += lat
        lat_max = max(lat_max, lat)
        metric_count += 1

    return {
        'ref_direction': ref_direction,
        'ref_speed': ref_speed,
        'ref_acceleration': ref_acceleration,
        'lateral_error': lateral_error,
        'valid_mask': valid_mask,
        'plan_total': int(plan_total),
        'plan_success': int(plan_success),
        'curv_sum': float(curv_sum),
        'prog_sum': float(prog_sum),
        'lat_sum': float(lat_sum),
        'lat_max': float(lat_max),
        'metric_count': int(metric_count),
        'sampled_paths': sampled_paths,
    }


def compute_guidance_reference_from_planner(env, p, v, p_target, dist_obj, planner: GlobalPlanner,
                                             max_speed=5.0, max_accel=5.0, max_decel=6.0,
                                             lookahead_dist=1.0, invalid_dist_threshold=-0.05,
                                             sampled_steps=None):
    """
    使用全局规划器生成参考方向、速度、加速度和横向偏差

    基于 A* 规划路径和梯形速度剖面计算动力学可行的参考值：
    - 方向：指向前瞻点
    - 速度：考虑曲率、终点减速、动力学约束
    - 加速度：基于速度剖面变化率
    - 横向偏差：到规划路径的真正几何距离

    Args:
        p: 位置 [S, B, 3] 或 [B, 3]
        v: 速度 [S, B, 3] 或 [B, 3]
        p_target: 目标位置 [B, 3]
        dist_obj: 到障碍物的距离 [S, B] 或 [B]
        planner: 全局规划器实例
        max_speed: 最大速度 (m/s)
        max_accel: 最大加速度 (m/s^2)
        max_decel: 最大减速度 (m/s^2)
        lookahead_dist: 前瞻距离 (m)

    Returns:
        ref_direction: 参考方向 [S, B, 3]
        ref_speed: 参考速度 [S, B]
        ref_acceleration: 参考加速度 [S, B, 3]
        lateral_error: 横向偏差 [S, B]，到规划路径的距离
        valid_mask: 有效性掩码 [S, B]
        planner_info: 规划器信息字典
    """
    squeeze_output = False
    if p.dim() == 2:
        p = p.unsqueeze(0)
        v = v.unsqueeze(0)
        dist_obj = dist_obj.unsqueeze(0)
        squeeze_output = True

    S, B, _ = p.shape
    device = p.device

    ref_direction = torch.zeros(S, B, 3, device=device)
    ref_speed = torch.zeros(S, B, device=device)
    ref_acceleration = torch.zeros(S, B, 3, device=device)
    lateral_error = torch.zeros(S, B, device=device)
    valid_mask = torch.zeros(S, B, dtype=torch.bool, device=device)

    curv_sum = 0.0
    prog_sum = 0.0
    lat_sum = 0.0
    lat_max = 0.0
    metric_count = 0
    sample_plan_total = 0
    sample_plan_success = 0
    sampled_astar_paths = [[] for _ in range(B)]

    used_parallel = False
    pool = _get_planner_pool()
    if pool is not None and B > 1:
        try:
            payloads = []
            for b in range(B):
                vox_np = env.voxels[b].detach().cpu().numpy() if hasattr(env, 'voxels') else np.zeros((0, 6), dtype=np.float32)
                balls_np = env.balls[b].detach().cpu().numpy() if hasattr(env, 'balls') else np.zeros((0, 4), dtype=np.float32)
                cyl_np = env.cyl[b].detach().cpu().numpy() if hasattr(env, 'cyl') else np.zeros((0, 3), dtype=np.float32)
                cyl_h_np = env.cyl_h[b].detach().cpu().numpy() if hasattr(env, 'cyl_h') else np.zeros((0, 3), dtype=np.float32)

                payloads.append({
                    'voxels': np.asarray(vox_np, dtype=np.float32),
                    'balls': np.asarray(balls_np, dtype=np.float32),
                    'cyl': np.asarray(cyl_np, dtype=np.float32),
                    'cyl_h': np.asarray(cyl_h_np, dtype=np.float32),
                    'sampled_positions': np.asarray(p[:, b].detach().cpu().numpy(), dtype=np.float32),
                    'sampled_dist': np.asarray(dist_obj[:, b].detach().cpu().numpy(), dtype=np.float32),
                    'sampled_steps': (
                        np.asarray(sampled_steps[:, b].detach().cpu().numpy(), dtype=np.int64)
                        if sampled_steps is not None else None
                    ),
                    'goal': np.asarray(p_target[b].detach().cpu().numpy(), dtype=np.float32),
                    'resolution': planner.resolution,
                    'margin': planner.margin,
                    'z_min': planner.z_min,
                    'z_max': planner.z_max,
                    'max_speed': max_speed,
                    'max_accel': max_accel,
                    'max_decel': max_decel,
                    'lookahead_dist': lookahead_dist,
                    'invalid_dist_threshold': invalid_dist_threshold,
                })

            results = pool.map(_plan_sample_points_worker, payloads)
            used_parallel = True

            for b, out in enumerate(results):
                ref_direction[:, b] = torch.from_numpy(out['ref_direction']).to(device=device, dtype=p.dtype)
                ref_speed[:, b] = torch.from_numpy(out['ref_speed']).to(device=device, dtype=p.dtype)
                ref_acceleration[:, b] = torch.from_numpy(out['ref_acceleration']).to(device=device, dtype=p.dtype)
                lateral_error[:, b] = torch.from_numpy(out['lateral_error']).to(device=device, dtype=p.dtype)
                valid_mask[:, b] = torch.from_numpy(out['valid_mask']).to(device=device)

                sample_plan_total += int(out['plan_total'])
                sample_plan_success += int(out['plan_success'])
                curv_sum += float(out['curv_sum'])
                prog_sum += float(out['prog_sum'])
                lat_sum += float(out['lat_sum'])
                lat_max = max(lat_max, float(out['lat_max']))
                metric_count += int(out['metric_count'])

                out_paths = out.get('sampled_paths', [])
                for item in out_paths:
                    if item is None:
                        continue
                    step_t, path_s = item
                    sampled_astar_paths[b].append((int(step_t), np.asarray(path_s, dtype=np.float32)))
        except Exception:
            used_parallel = False

    if not used_parallel:
        # 串行回退路径：每个 batch 仅构建一次占用栅格，随后复用
        occupancy_cache = {}
        for b in range(B):
            planner.build_occupancy_grid(env, batch_idx=b)
            occupancy_cache[b] = (
                planner.occupancy_grid,
                planner.grid_origin.copy(),
                planner.grid_shape,
            )

        for b in range(B):
            planner.occupancy_grid, planner.grid_origin, planner.grid_shape = occupancy_cache[b]
            goal_np = p_target[b].detach().cpu().numpy()

            for s in range(S):
                pos_np = p[s, b].detach().cpu().numpy()
                dist = dist_obj[s, b].item()

                step_t = int(sampled_steps[s, b].item()) if sampled_steps is not None else int(s)

                if dist < invalid_dist_threshold:
                    sampled_astar_paths[b].append((step_t, None))
                    continue

                sample_plan_total += 1
                path = planner.plan_astar(pos_np, goal_np)
                if path is None:
                    sampled_astar_paths[b].append((step_t, None))
                    continue
                path = planner.smooth_path(path, window_size=5)
                path_np = np.asarray(path, dtype=np.float32)
                sampled_astar_paths[b].append((step_t, path_np))

                ref_info = planner.extract_reference_from_path(
                    pos_np, path,
                    max_speed=max_speed,
                    max_accel=max_accel,
                    max_decel=max_decel,
                    lookahead_dist=lookahead_dist,
                )

                ref_direction[s, b] = torch.tensor(ref_info['direction'], device=device, dtype=p.dtype)
                ref_speed[s, b] = ref_info['speed']
                ref_acceleration[s, b] = torch.tensor(ref_info['acceleration'], device=device, dtype=p.dtype)
                lateral_error[s, b] = ref_info.get('lateral_error', 0.0)
                valid_mask[s, b] = bool(ref_info['valid'])
                if bool(ref_info['valid']):
                    sample_plan_success += 1

                curv_sum += float(ref_info.get('curvature', 0.0))
                prog_sum += float(ref_info.get('path_progress', 0.0))
                lat = float(ref_info.get('lateral_error', 0.0))
                lat_sum += lat
                lat_max = max(lat_max, lat)
                metric_count += 1

    plan_total = max(1, int(sample_plan_total))
    plan_success = int(sample_plan_success)
    planner_info = {
        'avg_curvature': (curv_sum / metric_count) if metric_count > 0 else 0.0,
        'avg_path_progress': (prog_sum / metric_count) if metric_count > 0 else 0.0,
        'avg_lateral_error': (lat_sum / metric_count) if metric_count > 0 else 0.0,
        'max_lateral_error': lat_max if metric_count > 0 else 0.0,
        'planner_success_ratio': float(plan_success) / float(plan_total),
        'reference_valid_ratio': float(valid_mask.float().mean().item()),
        'sample_plan_total': int(sample_plan_total),
        'sample_plan_success': int(sample_plan_success),
        'sampled_astar_paths': sampled_astar_paths,
    }

    if squeeze_output:
        ref_direction = ref_direction.squeeze(0)
        ref_speed = ref_speed.squeeze(0)
        ref_acceleration = ref_acceleration.squeeze(0)
        lateral_error = lateral_error.squeeze(0)
        valid_mask = valid_mask.squeeze(0)

    return ref_direction, ref_speed, ref_acceleration, lateral_error, valid_mask, planner_info


def compute_escape_penalty(v, vec_to_pt, dist_obj, collision_mask):
    """
    对已碰撞点，计算逃逸惩罚而非规划引导
    鼓励速度方向与逃逸方向（远离障碍物内部）一致

    Args:
        v: 速度 [S, B, 3]
        vec_to_pt: 指向最近障碍物表面的向量 [S, B, 3]
        dist_obj: 到障碍物的距离 [S, B]
        collision_mask: 碰撞掩码 [S, B]

    Returns:
        escape_loss: 逃逸方向一致性损失 [S, B]
        depth_penalty: 碰撞深度惩罚 [S, B]
    """
    # 逃逸方向：vec_to_pt 指向障碍物表面最近点
    # 当在障碍物内时，应该朝 vec_to_pt 的方向移动以逃出
    escape_dir = safe_normalize(vec_to_pt, dim=-1)  # [S, B, 3]
    v_dir = safe_normalize(v, dim=-1)  # [S, B, 3]

    # 逃逸方向一致性损失：1 - cos(v, escape_dir)
    # 当速度方向与逃逸方向一致时，损失为0
    escape_alignment = 1.0 - (v_dir * escape_dir).sum(dim=-1)  # [S, B]

    # 只对碰撞点计算
    escape_loss = escape_alignment * collision_mask.float()  # [S, B]

    # 额外惩罚：碰撞深度越大，惩罚越重
    depth_penalty = F.relu(-dist_obj).pow(2) * collision_mask.float()  # [S, B]

    return escape_loss, depth_penalty


def sample_guidance_points(p_history, v_history, dist_obj, sample_count, strategy='random'):
    """
    智能采样轨迹上的关键点

    Args:
        p_history: 位置历史 [T, B, 3]
        v_history: 速度历史 [T, B, 3]
        dist_obj: 到障碍物的距离 [T, B]
        sample_count: 采样点数
        strategy: 采样策略 ('random', 'uniform', 'adaptive', 'critical')

    Returns:
        indices: 采样点索引 tensor [S, B]（每个 batch 独立采样）
    """
    T, B = p_history.shape[:2]
    device = p_history.device

    # 确保采样数不超过总时间步数
    sample_count = min(sample_count, T)

    if strategy == 'random':
        # 随机采样时间步（不放回），每个 batch 独立采样并按时间排序
        if sample_count >= T:
            base = torch.arange(T, device=device, dtype=torch.long)
            indices = base[:, None].expand(T, B).clone()
        else:
            cols = []
            for b in range(B):
                idx_b = torch.randperm(T, device=device)[:sample_count].sort().values
                cols.append(idx_b)
            indices = torch.stack(cols, dim=1)

    elif strategy == 'uniform':
        # 均匀采样（时间步一致，但按 batch 维度展开）
        base = torch.linspace(0, T - 1, sample_count, device=device).long()
        indices = base[:, None].expand(sample_count, B).clone()

    elif strategy == 'adaptive':
        # 自适应采样：优先采样危险点和变化大的点（每个 batch 独立）
        with torch.no_grad():
            # 危险度：dist_obj 越小越危险
            danger_score = F.softplus(-dist_obj * 5.0)  # [T, B]

            # 速度变化度（曲率指标）
            v_diff = (v_history[1:] - v_history[:-1]).norm(dim=-1)  # [T-1, B]
            v_diff = F.pad(v_diff, (0, 0, 0, 1), value=0.0)  # [T, B]

            # 综合分数
            importance = danger_score + v_diff  # [T, B]

            # 每个 batch 选择最重要的点
            k = min(sample_count, T)
            cols = []
            for b in range(B):
                _, top_indices = importance[:, b].topk(k)
                cols.append(top_indices.sort().values)
            indices = torch.stack(cols, dim=1)

    elif strategy == 'critical':
        # 只采样关键时刻：轨迹的起点、终点、最危险点（每个 batch 独立）
        with torch.no_grad():
            danger = F.softplus(-dist_obj * 5.0)  # [T, B]
            cols = []
            for b in range(B):
                critical_indices = {0, T - 1}
                remaining_count = sample_count - len(critical_indices)
                if remaining_count > 0 and T > 2:
                    danger_mid = danger[1:-1, b]
                    k = min(remaining_count, int(danger_mid.numel()))
                    if k > 0:
                        _, top_danger = danger_mid.topk(k)
                        for idx in (top_danger + 1).tolist():
                            critical_indices.add(int(idx))

                idx_b = torch.tensor(sorted(critical_indices), device=device, dtype=torch.long)
                if idx_b.numel() < sample_count:
                    pad = idx_b[-1].repeat(sample_count - idx_b.numel())
                    idx_b = torch.cat([idx_b, pad], dim=0)
                elif idx_b.numel() > sample_count:
                    idx_b = idx_b[:sample_count]
                cols.append(idx_b)

            indices = torch.stack(cols, dim=1)

    else:
        # 默认均匀采样
        base = torch.linspace(0, T - 1, sample_count, device=device).long()
        indices = base[:, None].expand(sample_count, B).clone()

    return indices


def _masked_mean(x, mask):
    mask_bool = mask.bool()
    mask_f = mask_bool.float()
    denom = mask_f.sum().clamp_min(1.0)
    x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(mask_bool, x_safe, torch.zeros_like(x_safe)).sum() / denom


def compute_guidance_reference_from_potential_map(p_history, map_data, interpolation='trilinear'):
    """Query dense potential value and descending direction from precomputed map field."""
    potential_value, ref_dir, valid_mask = query_potential_guidance(
        map_data=map_data,
        points_world=p_history,
        interpolation=interpolation,
    )
    ref_dir = safe_normalize(ref_dir, dim=-1)
    return potential_value, ref_dir, valid_mask


def compute_potential_guidance_meta_loss(env, p_history, v_history, vec_to_pt, dist_obj,
                                         map_data,
                                         config,
                                         max_speed=5.0,
                                         dir_weight=0.5,
                                         speed_weight=0.3,
                                         lateral_weight=0.3,
                                         escape_weight=1.0,
                                         collision_threshold=-0.05,
                                         accel_weight=0.2,
                                         speed_diff_weight=0.2,
                                         recovery_speed_weight=0.15):
    """Dense guidance loss based on precomputed Dijkstra potential/vector field."""
    T, B, _ = p_history.shape

    potential_value, ref_dir, valid_mask = compute_guidance_reference_from_potential_map(
        p_history=p_history,
        map_data=map_data,
        interpolation=config.potential_interpolation,
    )

    ref_dir = sanitize_tensor(ref_dir, nan=0.0, posinf=0.0, neginf=0.0)
    v_dir = safe_normalize(v_history, dim=-1)
    v_speed = v_history.norm(dim=-1)

    loss_dir_align = sanitize_tensor(
        1.0 - (v_dir * ref_dir).sum(dim=-1),
        nan=0.0,
        posinf=2.0,
        neginf=0.0,
    ).clamp(0.0, 2.0)

    # Simple, stable speed reference: full speed in free/early areas, damp near walls and near low-potential zones.
    finite_mask = torch.isfinite(potential_value)
    valid_mask = valid_mask & finite_mask
    potential_value_safe = torch.where(finite_mask, potential_value, torch.zeros_like(potential_value))
    safe_pot = torch.where(valid_mask, potential_value_safe, torch.zeros_like(potential_value_safe))
    pot_max = safe_pot.max(dim=0, keepdim=True).values.clamp_min(1.0)
    pot_ratio = torch.clamp(safe_pot / pot_max, 0.0, 1.0)
    obstacle_factor = torch.sigmoid((dist_obj - 0.8) * 5.0)
    ref_speed = max_speed * (0.25 + 0.75 * obstacle_factor) * (0.30 + 0.70 * pot_ratio)

    loss_overspeed = F.relu(v_speed - ref_speed)
    loss_underspeed = F.relu(ref_speed - v_speed) * 0.3
    loss_speed_diff = loss_overspeed + loss_underspeed

    # Potential descent constraint: penalize local potential increase.
    loss_pot_step = torch.zeros_like(potential_value)
    if T > 1:
        step_valid = valid_mask[:-1] & valid_mask[1:]
        step_term = F.relu(
            potential_value_safe[1:]
            - potential_value_safe[:-1]
            + float(config.potential_delta_margin)
        )
        loss_pot_step[:-1] = torch.where(step_valid, step_term, torch.zeros_like(step_term))
    else:
        step_valid = torch.zeros((0, B), dtype=torch.bool, device=p_history.device)

    # Potential absolute term: normalized potential on valid field points.
    loss_pot_abs = torch.where(valid_mask, safe_pot / (pot_max + 1e-6), torch.zeros_like(safe_pot))

    collision_mask = dist_obj < collision_threshold
    invalid_mask = (~valid_mask) & (~collision_mask)
    recovery_mask = collision_mask | invalid_mask
    valid_guidance_mask = valid_mask & (~collision_mask)

    loss_escape, loss_depth = compute_escape_penalty(
        v_history, vec_to_pt, dist_obj, recovery_mask
    )
    loss_recovery_speed = v_speed * invalid_mask.float()

    guidance_for_valid = (
        dir_weight * loss_dir_align
        + lateral_weight * loss_pot_abs
    )
    guidance_for_recovery = (
        escape_weight * (loss_escape + loss_depth)
    )

    guidance_loss_per_point = torch.where(valid_guidance_mask, guidance_for_valid, guidance_for_recovery)
    guidance_loss_per_point = sanitize_tensor(guidance_loss_per_point, nan=0.0, posinf=1e3, neginf=0.0)
    guidance_loss = guidance_loss_per_point.mean()

    potential_decrease = torch.tensor(0.0, device=p_history.device, dtype=p_history.dtype)
    if T > 1:
        raw_dec = potential_value_safe[:-1] - potential_value_safe[1:]
        potential_decrease = _masked_mean(raw_dec, step_valid)

    field_dir_align = _masked_mean(1.0 - loss_dir_align, valid_guidance_mask)

    loss_components = {
        'dir_align': loss_dir_align.mean(),
        'speed_diff': loss_speed_diff.mean(),
        'overspeed': loss_overspeed.mean(),
        'underspeed': loss_underspeed.mean(),
        'potential_abs': loss_pot_abs.mean(),
        'potential_step_penalty': loss_pot_step.mean(),
        # compatibility aliases
        'lateral_error': loss_pot_abs.mean(),
        'accel_mismatch': loss_pot_step.mean(),
        'escape': loss_escape.mean(),
        'depth': loss_depth.mean(),
        'recovery_speed': loss_recovery_speed.mean(),
        'valid_ratio': valid_mask.float().mean(),
        'invalid_ratio': invalid_mask.float().mean(),
        'collision_ratio': collision_mask.float().mean(),
        'sample_count': float(T),
        'avg_curvature': 0.0,
        'avg_path_progress': 0.0,
        'avg_lateral_error': loss_pot_abs.mean().item(),
        'max_lateral_error': loss_pot_abs.max().item(),
        'potential_valid_ratio': valid_mask.float().mean(),
        # compatibility alias
        'planner_success_ratio': valid_mask.float().mean().item(),
        'avg_ref_speed': ref_speed.mean().item(),
        'sampled_astar_paths': [],
        'potential_mean': _masked_mean(safe_pot, valid_mask),
        'potential_decrease': potential_decrease,
        'field_dir_align': field_dir_align,
    }
    return guidance_loss, loss_components


def compute_global_guidance_meta_loss(env, p_history, v_history, p_target, vec_to_pt, dist_obj,
                                       config, potential_map_cache, planner,
                                       a_history=None,
                                       sample_count=10, strategy='random',
                                       max_speed=5.0, max_accel=5.0, max_decel=6.0,
                                       dir_weight=0.5, speed_weight=0.3, lateral_weight=0.3,
                                       escape_weight=1.0, collision_threshold=-0.05,
                                       accel_weight=0.2, speed_diff_weight=0.2,
                                       recovery_speed_weight=0.15):
    """
    全局规划器引导的元损失，使用 A* 算法规划的全局路径作为参考

    基于梯形速度剖面和动力学约束计算损失：
    - 方向一致性：速度方向与规划方向的夹角
    - 速度偏差：实际速度与规划参考速度的差异（双向惩罚）
    - 横向偏差：到规划路径的真正几何距离
    - 加速度偏差：实际加速度与规划参考加速度的差异

    Args:
        p_history: 位置历史 [T, B, 3]
        v_history: 速度历史 [T, B, 3]
        p_target: 目标位置 [B, 3]
        vec_to_pt: 指向最近障碍物的向量 [T, B, 3]
        dist_obj: 到障碍物的距离 [T, B]
        a_history: 可选的环境真实加速度历史 [T, B, 3]，提供时优先用于加速度监督
        sample_count: 采样点数
        strategy: 采样策略
        max_speed: 最大速度 (m/s)
        max_accel: 最大加速度 (m/s^2)
        max_decel: 最大减速度 (m/s^2)
        dir_weight: 方向一致性损失权重
        speed_weight: 速度偏差惩罚权重
        lateral_weight: 横向偏差惩罚权重（到路径的几何距离）
        escape_weight: 逃逸惩罚权重
        collision_threshold: 碰撞判定阈值
        accel_weight: 加速度偏差权重
        speed_diff_weight: 速度差（超速/低速）惩罚权重

    Returns:
        guidance_loss: 标量损失
        loss_components: 各分项损失的字典
    """
    # New default path: dense potential-field guidance from precomputed map cache.
    if config.use_precomputed_potential_maps and not config.use_astar_guidance:
        if potential_map_cache is None:
            raise RuntimeError("Precomputed potential map mode is enabled but POTENTIAL_MAP_CACHE is not initialized")
        map_idx = int(getattr(env, 'current_map_idx', 0))
        map_data = potential_map_cache.get_map(map_idx)
        return compute_potential_guidance_meta_loss(
            env=env,
            p_history=p_history,
            v_history=v_history,
            vec_to_pt=vec_to_pt,
            dist_obj=dist_obj,
            map_data=map_data,
            config=config,
            max_speed=max_speed,
            dir_weight=dir_weight,
            speed_weight=speed_weight,
            lateral_weight=lateral_weight,
            escape_weight=escape_weight,
            collision_threshold=collision_threshold,
            accel_weight=accel_weight,
            speed_diff_weight=speed_diff_weight,
            recovery_speed_weight=recovery_speed_weight,
        )

    T, B, _ = p_history.shape

    # 1. 采样关键点
    sample_indices = sample_guidance_points(
        p_history, v_history, dist_obj, sample_count, strategy
    )
    S = sample_indices.shape[0]

    # 2. 提取采样点的状态
    b_idx = torch.arange(B, device=p_history.device).unsqueeze(0).expand(S, B)
    p_sampled = p_history[sample_indices, b_idx]      # [S, B, 3]
    v_sampled = v_history[sample_indices, b_idx]      # [S, B, 3]
    vec_sampled = vec_to_pt[sample_indices, b_idx]    # [S, B, 3]
    dist_sampled = dist_obj[sample_indices, b_idx]    # [S, B]
    a_sampled = None
    if a_history is not None:
        a_sampled = a_history[sample_indices, b_idx]  # [S, B, 3]

    # 3. 使用全局规划器计算参考（A* 规划 + 梯形速度剖面）
    ref_dir, ref_speed, ref_accel, lateral_error, valid_mask, planner_info = compute_guidance_reference_from_planner(
        env, p_sampled, v_sampled, p_target, dist_sampled, planner,
        max_speed=max_speed, max_accel=max_accel, max_decel=max_decel,
        lookahead_dist=1.0,
        invalid_dist_threshold=collision_threshold,
        sampled_steps=sample_indices,
    )

    # 4. 计算各项损失
    v_dir = safe_normalize(v_sampled, dim=-1)  # [S, B, 3]
    v_speed = v_sampled.norm(dim=-1)  # [S, B]

    # 4.1 方向一致性损失：1 - cos(v_dir, ref_dir)
    loss_dir_align = 1.0 - (v_dir * ref_dir).sum(dim=-1)  # [S, B]

    # 4.2 速度偏差：双向惩罚（超速和低速都惩罚）
    # 超速惩罚更重，低速惩罚较轻
    loss_overspeed = F.relu(v_speed - ref_speed)  # [S, B]
    loss_underspeed = F.relu(ref_speed - v_speed) * 0.3  # 低速惩罚较轻
    loss_speed_diff = loss_overspeed + loss_underspeed  # [S, B]

    # 4.3 横向偏差：到规划路径的真正几何距离
    # 使用平滑的 L1 损失，对小偏差不过度惩罚
    loss_lateral = F.smooth_l1_loss(lateral_error, torch.zeros_like(lateral_error), reduction='none')  # [S, B]

    # 4.4 加速度偏差惩罚
    if S > 1:
        # 实际加速度：优先使用环境提供的 a_history，缺失时回退到速度差分估算
        if a_sampled is not None:
            v_diff = a_sampled
        else:
            v_diff = torch.zeros_like(v_sampled)
            for i in range(S - 1):
                dt_approx = (sample_indices[i + 1] - sample_indices[i]).to(v_sampled.dtype) / 15.0  # [B]
                step_acc = (v_sampled[i + 1] - v_sampled[i]) / (dt_approx[:, None] + 1e-6)
                valid_dt = dt_approx > 0
                if valid_dt.any():
                    v_diff[i, valid_dt] = step_acc[valid_dt]
            v_diff[-1] = v_diff[-2] if S > 1 else torch.zeros_like(v_diff[-1])

        # 加速度偏差：实际加速度与参考加速度的差异
        # ref_accel 是基于速度剖面计算的参考加速度（通常是减速）
        accel_error = (v_diff - ref_accel).norm(dim=-1)  # [S, B]

        # 只惩罚明显的加速度偏差（阈值 0.5 m/s^2）
        loss_accel_mismatch = F.relu(accel_error - 0.5)  # [S, B]

        # 额外检查：需要减速时是否真的在减速
        ref_accel_mag = ref_accel.norm(dim=-1)  # [S, B]
        ref_accel_dir = safe_normalize(ref_accel, dim=-1)
        actual_accel_along_ref = (v_diff * ref_accel_dir).sum(dim=-1)  # [S, B]
        # 如果规划器要求减速（ref_accel 有显著幅度且为负）但实际在加速
        need_decel = ref_accel_mag > 0.5
        not_deceling = actual_accel_along_ref < -0.5  # 实际加速度与减速方向相反
        loss_decel_violation = need_decel.float() * not_deceling.float() * ref_accel_mag
        loss_accel_mismatch = loss_accel_mismatch + loss_decel_violation
    else:
        loss_accel_mismatch = torch.zeros_like(loss_dir_align)

    # 5. 对不可规划点/碰撞点的恢复处理
    collision_mask = dist_sampled < collision_threshold  # [S, B]
    invalid_mask = (~valid_mask) & (~collision_mask)
    recovery_mask = collision_mask | invalid_mask
    valid_guidance_mask = valid_mask & (~collision_mask)

    loss_escape, loss_depth = compute_escape_penalty(
        v_sampled, vec_sampled, dist_sampled, recovery_mask
    )
    loss_recovery_speed = v_speed * invalid_mask.float()

    # 6. 组合损失
    # 有效点（非碰撞）用规划引导；速度/加速度动态项仅保留日志，不进入 loss。
    guidance_for_valid = (
        dir_weight * loss_dir_align +
        lateral_weight * loss_lateral
    )  # [S, B]

    # 碰撞点和不可规划点用恢复惩罚
    guidance_for_recovery = (
        escape_weight * (loss_escape + loss_depth)
    )  # [S, B]

    # 根据点状态选择损失
    guidance_loss_per_point = torch.where(
        valid_guidance_mask,
        guidance_for_valid,
        guidance_for_recovery,
    )  # [S, B]

    # 总损失
    guidance_loss = guidance_loss_per_point.mean()

    # 返回各分项用于日志
    loss_components = {
        'dir_align': loss_dir_align.mean(),
        'speed_diff': loss_speed_diff.mean(),
        'overspeed': loss_overspeed.mean(),
        'underspeed': loss_underspeed.mean(),
        'lateral_error': loss_lateral.mean(),
        'accel_mismatch': loss_accel_mismatch.mean(),
        'escape': loss_escape.mean(),
        'depth': loss_depth.mean(),
        'recovery_speed': loss_recovery_speed.mean(),
        'valid_ratio': valid_mask.float().mean(),
        'invalid_ratio': invalid_mask.float().mean(),
        'collision_ratio': collision_mask.float().mean(),
        'sample_count': S,
        'avg_curvature': planner_info.get('avg_curvature', 0.0),
        'avg_path_progress': planner_info.get('avg_path_progress', 0.0),
        'avg_lateral_error': planner_info.get('avg_lateral_error', 0.0),
        'max_lateral_error': planner_info.get('max_lateral_error', 0.0),
        'planner_success_ratio': planner_info.get('planner_success_ratio', 0.0),
        'avg_ref_speed': ref_speed.mean().item(),
        'sampled_astar_paths': planner_info.get('sampled_astar_paths', []),
    }

    return guidance_loss, loss_components
