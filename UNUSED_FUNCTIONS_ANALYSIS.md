# mmgj_transformer.py 完整代码分析报告

## 📋 文件概况
- **文件路径**: `/home/robot/transformer/multi_pub/mmgj_transformer.py`
- **代码行数**: ~3400+ 行
- **文件大小**: 约 121 KB
- **核心功能**: Transformer 无人机控制 + LGN 元学习 + 全局规划引导

---

## 📊 定义统计

| 类别 | 数量 | 被使用 | 未使用 | 使用率 |
|------|------|--------|---------|--------|
| 类定义 | 3 | 3 | 0 | 100% |
| 函数定义 | 33 | 31 | 2 | 93.9% |
| **总计** | **36** | **34** | **2** | **94.4%** |

---

## 🏗️ 所有类定义及使用情况

### ✅ 使用中的类（3个）

#### 1. RunningMeanStd (第 62-89 行)
**归一化工具类 - 动态运行均值/方差计算**

```python
class RunningMeanStd(nn.Module):
    """
    维护运行中的均值和方差，用于在线状态归一化。
    继承自 nn.Module，支持 GPU/CPU 切换。
    """
```

**定义位置**: [L62-89](mmgj_transformer.py#L62-L89)
- **关键方法**: `forward(x, update=True)` - 数据归一化并可选更新统计量
- **核心属性**: 
  - `mean`: 运行均值 [shape]
  - `var`: 运行方差 [shape]
  - `count`: 更新计数
  - `epsilon`: 数值稳定性

**使用位置**:
- [L403](mmgj_transformer.py#L403): 创建实例 `state_normalizer = RunningMeanStd(shape=(state_dim,)).to(device)`
- [L2185](mmgj_transformer.py#L2185): 元 rollout 中调用（不更新统计）
- [L2284](mmgj_transformer.py#L2284): `state_normalizer.train()` 设置训练模式
- [L2339](mmgj_transformer.py#L2339): 主训练循环中调用（更新统计）
- **多处**: 梯度日志中记录 `state_normalizer.mean`, `state_normalizer.var`, `state_normalizer.count`

**重要性**: ⭐⭐⭐⭐⭐ **核心**

---

#### 2. LossNormalizer (第 92-111 行)
**损失缩放器 - 多损失分量的动态归一化**

```python
class LossNormalizer:
    """
    追踪每个损失分量的运行标准差，确保各损失分量贡献均衡。
    可微分的缩放，统计量分离（不参与反向传播）。
    """
```

**定义位置**: [L92-111](mmgj_transformer.py#L92-L111)
- **关键方法**: `normalize(*losses)` - 使用运行标准差归一化损失
- **核心属性**:
  - `running_std`: 各损失分量的运行标准差列表
  - `momentum`: 动量参数

**使用位置**:
- [L410](mmgj_transformer.py#L410): 创建实例 `loss_normalizer = LossNormalizer(4)` (4个损失分量)
- [L2382-2385](mmgj_transformer.py#L2382-L2385): 主训练循环中调用
  ```python
  loss_speed_n, loss_dir_n, loss_avoid_n, loss_expl_n = \
      loss_normalizer.normalize(
          loss_speed_seq, loss_direction_seq, loss_avoidance_seq, loss_exploration_seq
      )
  ```

**重要性**: ⭐⭐⭐⭐⭐ **核心**

---

#### 3. GlobalPlanner (第 504-1017 行)
**3D A* 全局规划器 - 路径规划 & 引导生成**

```python
class GlobalPlanner:
    """
    3D A* 路径规划器，构建占用栅格地图并规划从起点到终点的最优路径。
    支持球形、立方体、圆柱障碍物。
    提供梯形速度剖面和加速度参考值。
    包含并行处理支持和结果缓存。
    """
```

**定义位置**: [L504-1017](mmgj_transformer.py#L504-L1017)
- **关键方法**:
  - `build_occupancy_grid()` - 构建 3D 占用栅格
  - `plan_astar()` - A* 路径规划
  - `smooth_path()` - 路径平滑处理
  - `extract_reference_from_path()` - 从路径提取参考方向/速度/加速度
  - `plan_and_cache()` - 批量规划并缓存
  - `clear_cache()` - 清除缓存

**使用位置**:
- [L1056](mmgj_transformer.py#L1056): 创建全局实例
  ```python
  global_planner = GlobalPlanner(resolution=0.3, margin=0.15)
  ```
- [L2217-2269](mmgj_transformer.py#L2217-L2269): 在 `_plan_sample_points_worker` 中创建副本并调用
- [L2406, L2437](mmgj_transformer.py#L2406-L2437): 在 `compute_global_guidance_meta_loss` 中调用
- [L2955](mmgj_transformer.py#L2955): 构建占用栅格用于 A* 路径可视化

**重要性**: ⭐⭐⭐⭐⭐ **核心**

---

## 🔧 所有函数定义及使用情况

### ✅ 使用中的函数（31个）

#### 第1组: 基础工具函数（7个）

**1. safe_normalize (第 113-114 行) ✅**
```python
def safe_normalize(x, dim=-1, eps=1e-6):
    """安全向量标准化，处理 NaN/Inf"""
    return F.normalize(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dim=dim, eps=eps)
```
- **定义**: [L113-114](mmgj_transformer.py#L113-L114)
- **使用**:
  - [L1361](mmgj_transformer.py#L1361): `escape_dir = safe_normalize(vec_to_pt, dim=-1)`
  - [L1362](mmgj_transformer.py#L1362): `v_dir = safe_normalize(v, dim=-1)`
  - [L1537](mmgj_transformer.py#L1537): `v_dir = safe_normalize(v_sampled, dim=-1)`
  - [L1577](mmgj_transformer.py#L1577): `ref_accel_dir = safe_normalize(ref_accel, dim=-1)`
- **使用次数**: 4 次

---

**2. sanitize_tensor (第 117-119 行) ✅**
```python
def sanitize_tensor(x, nan=0.0, posinf=1e3, neginf=-1e3):
    """清理张量中的 NaN/Inf 值"""
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
```
- **定义**: [L117-119](mmgj_transformer.py#L117-L119)
- **使用**: **广泛使用** ~35+ 处
  - 关键位置: 数据清理、深度特征处理、状态/动作处理
- **使用次数**: 35+ 次

---

**3. sanitize_module_ (第 122-127 行) ✅**
```python
@torch.no_grad()
def sanitize_module_(module, clamp_value=10.0):
    """清理模型参数中的 NaN/Inf"""
```
- **定义**: [L122-127](mmgj_transformer.py#L122-L127)
- **使用**:
  - [L3195](mmgj_transformer.py#L3195): `sanitize_module_(lgn, clamp_value=5.0)` (LGN 梯度爆炸保护)
  - [L3208](mmgj_transformer.py#L3208): `sanitize_module_(worknet, clamp_value=10.0)` (Worker 梯度爆炸保护)
- **使用次数**: 2 次

---

**4. smooth_dict (第 415-419 行) ✅**
```python
def smooth_dict(ori_dict):
    """将字典值添加到标量滑动平均队列"""
    for k, v in ori_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        scaler_q[k].append(float(v))
```
- **定义**: [L415-419](mmgj_transformer.py#L415-L419)
- **使用**: [L3277](mmgj_transformer.py#L3277) - 在日志收集中
- **使用次数**: 1 次

---

**5. is_save_iter (第 421-423 行) ✅**
```python
def is_save_iter(i):
    """判断是否应保存模型检查点"""
    return (i + 1) % 10000 == 0 if i >= 2000 else (i + 1) % 500 == 0
```
- **定义**: [L421-423](mmgj_transformer.py#L421-L423)
- **使用**: [L3267](mmgj_transformer.py#L3267)
- **使用次数**: 1 次

---

**6. is_save_trajectory_iter (第 425-427 行) ✅**
```python
def is_save_trajectory_iter(i):
    """判断是否应保存轨迹可视化"""
    return i == 0 or (i + 1) % 500 == 0
```
- **定义**: [L425-427](mmgj_transformer.py#L425-L427)
- **使用**: [L2305](mmgj_transformer.py#L2305)
- **使用次数**: 1 次

---

**7. rotation_matrix_to_rpy_deg (第 429-439 行) ✅**
```python
def rotation_matrix_to_rpy_deg(R):
    """将旋转矩阵转换为欧拉角（度）"""
    # 返回 roll, pitch, yaw (度数)
```
- **定义**: [L429-439](mmgj_transformer.py#L429-L439)
- **使用**:
  - [L2324](mmgj_transformer.py#L2324): `rpy_history.append(rotation_matrix_to_rpy_deg(env.R))`
- **使用次数**: 1 次
- **目的**: 记录无人机姿态用于轨迹可视化

---

#### 第2组: 梯度监控函数（3个）

**8. get_grad_stats (第 442-460 行) ✅**
```python
def get_grad_stats(module):
    """获取模型梯度的全局范数、最大值、非有限元素数等"""
```
- **定义**: [L442-460](mmgj_transformer.py#L442-L460)
- **使用**:
  - [L3199](mmgj_transformer.py#L3199): `worker_grad_norm, worker_grad_max, ... = get_grad_stats(worknet)` (Worker 阶段)
  - [L3192](mmgj_transformer.py#L3192): `lgn_grad_norm, lgn_grad_max, ... = get_grad_stats(lgn)` (LGN 阶段)
- **使用次数**: 2 次
- **目的**: 梯度爆炸检测和诊断

---

**9. get_grad_norm_from_grads (第 461-476 行) ✅**
```python
def get_grad_norm_from_grads(grads):
    """从梯度列表计算全局范数"""
```
- **定义**: [L461-476](mmgj_transformer.py#L461-L476)
- **使用**: [L520](mmgj_transformer.py#L520) - 在 `get_loss_to_worker_grad_norm` 中调用
- **使用次数**: 1 次 (间接)

---

**10. get_loss_to_worker_grad_norm (第 477-485 行) ✅**
```python
def get_loss_to_worker_grad_norm(loss, params):
    """计算单个损失项对 Worker 梯度的贡献"""
```
- **定义**: [L477-485](mmgj_transformer.py#L477-L485)
- **使用**:
  - [L3046-3050](mmgj_transformer.py#L3046-L3050): 计算四个损失分项的梯度贡献
    ```python
    proxy_grad_speed, ... = get_loss_to_worker_grad_norm(loss_speed_seq.mean(), worker_params)
    proxy_grad_dir, ... = get_loss_to_worker_grad_norm(loss_direction_seq.mean(), worker_params)
    proxy_grad_avoid, ... = get_loss_to_worker_grad_norm(loss_avoidance_seq.mean(), worker_params)
    proxy_grad_expl, ... = get_loss_to_worker_grad_norm(loss_exploration_seq.mean(), worker_params)
    ```
- **使用次数**: 4 次

---

**11. merge_intervals (第 486-502 行) ✅**
```python
def merge_intervals(intervals, min_gap=1e-4):
    """合并间隔序列中重叠或相邻的区间"""
```
- **定义**: [L486-502](mmgj_transformer.py#L486-L502)
- **使用**: [L2043](mmgj_transformer.py#L2043) - 在 `get_collision_wall_patches` 中
  ```python
  for wall_idx, intervals in wall_intervals.items():
      ...
      for seg_start, seg_end in merge_intervals(intervals, min_gap=0.02):
  ```
- **使用次数**: 1 次

---

#### 第3组: 全局规划相关函数（5个）

**12. _compute_planner_worker_count (第 1020-1026 行) ✅**
```python
def _compute_planner_worker_count():
    """计算规划器进程池的进程数"""
    if PLANNER_NUM_WORKERS > 0:
        return PLANNER_NUM_WORKERS
    cpu_total = os.cpu_count() or 8
    return max(1, min(28, cpu_total - 3))
```
- **定义**: [L1020-1026](mmgj_transformer.py#L1020-L1026)
- **使用**: [L1042](mmgj_transformer.py#L1042) - 在 `_get_planner_pool` 中调用
- **使用次数**: 1 次

---

**13. _shutdown_planner_pool (第 1028-1035 行) ✅**
```python
@torch.no_grad()
def _shutdown_planner_pool():
    """关闭并清理规划器进程池"""
```
- **定义**: [L1028-1035](mmgj_transformer.py#L1028-L1035)
- **使用**:
  - [L1047](mmgj_transformer.py#L1047): 在 `_get_planner_pool` 中调用（清理旧池）
  - [L1057](mmgj_transformer.py#L1057): `atexit.register(_shutdown_planner_pool)` - 注册为退出处理程序
- **使用次数**: 2 次

---

**14. _get_planner_pool (第 1037-1055 行) ✅**
```python
def _get_planner_pool():
    """获取或创建规划器进程池（线程安全）"""
```
- **定义**: [L1037-1055](mmgj_transformer.py#L1037-L1055)
- **使用**: [L1281](mmgj_transformer.py#L1281) - 在 `compute_guidance_reference_from_planner` 中调用
- **使用次数**: 1 次

---

**15. _plan_sample_points_worker (第 1060-1259 行) ✅**
```python
def _plan_sample_points_worker(payload):
    """Worker 进程函数：对多个采样点进行 A* 规划和参考值提取"""
```
- **定义**: [L1060-1259](mmgj_transformer.py#L1060-L1259)
- **使用**: [L1283](mmgj_transformer.py#L1283) - 在 `compute_guidance_reference_from_planner` 中作为 `pool.map(_plan_sample_points_worker, payloads)`
- **使用次数**: 1 次
- **备注**: 多进程规划的关键 Worker 函数

---

#### 第4组: 引导损失相关函数（4个）

**16. compute_guidance_reference_from_planner (第 1165-1343 行) ✅**
```python
def compute_guidance_reference_from_planner(env, p, v, p_target, dist_obj, planner: GlobalPlanner, ...):
    """使用全局规划器生成参考方向、速度、加速度和横向偏差"""
```
- **定义**: [L1165-1343](mmgj_transformer.py#L1165-L1343)
- **使用**: [L1535](mmgj_transformer.py#L1535) - 在 `compute_global_guidance_meta_loss` 中调用
- **使用次数**: 1 次
- **返回值**: ref_direction, ref_speed, ref_acceleration, lateral_error, valid_mask, planner_info

---

**17. compute_escape_penalty (第 1344-1375 行) ✅**
```python
def compute_escape_penalty(v, vec_to_pt, dist_obj, collision_mask):
    """对已碰撞点计算逃逸惩罚（指向逃逸方向的一致性损失）"""
```
- **定义**: [L1344-1375](mmgj_transformer.py#L1344-L1375)
- **使用**: [L1599](mmgj_transformer.py#L1599) - 在 `compute_global_guidance_meta_loss` 中调用
  ```python
  loss_escape, loss_depth = compute_escape_penalty(v_sampled, vec_sampled, dist_sampled, recovery_mask)
  ```
- **使用次数**: 1 次

---

**18. sample_guidance_points (第 1377-1467 行) ✅**
```python
def sample_guidance_points(p_history, v_history, dist_obj, sample_count, strategy='random'):
    """智能采样轨迹上的关键点（支持多种采样策略）"""
```
- **定义**: [L1377-1467](mmgj_transformer.py#L1377-L1467)
- **采样策略**: 'random', 'uniform', 'adaptive', 'critical'
- **使用**: [L1512](mmgj_transformer.py#L1512) - 在 `compute_global_guidance_meta_loss` 中调用
- **使用次数**: 1 次

---

**19. compute_global_guidance_meta_loss (第 1469-1649 行) ✅**
```python
def compute_global_guidance_meta_loss(env, p_history, v_history, p_target, vec_to_pt, dist_obj, ...):
    """全局规划器引导的元损失（A* + 梯形速度剓面）"""
```
- **定义**: [L1469-1649](mmgj_transformer.py#L1469-L1649)
- **使用**:
  - [L2398](mmgj_transformer.py#L2398): 主训练循环中（LGN 阶段）
  - [L2437](mmgj_transformer.py#L2437): `unrolled_meta_rollout` 中
- **使用次数**: 2 次
- **返回**: guidance_loss (标量), loss_components (字典)

---

#### 第5组: 可视化函数（7个）

**20. draw_sphere (第 1684-1691 行) ✅**
```python
def draw_sphere(ax, cx, cy, cz, r, color='royalblue', alpha=0.18, res=12):
    """在 Matplotlib 3D 坐标轴上绘制球形障碍物"""
```
- **定义**: [L1684-1691](mmgj_transformer.py#L1684-L1691)
- **使用**: [L3309](mmgj_transformer.py#L3309) - 在轨迹可视化中
- **使用次数**: 1 次

---

**21. draw_cylinder_z (第 1693-1700 行) ✅**
```python
def draw_cylinder_z(ax, cx, cy, r, z0, z1, color='teal', alpha=0.14, res_theta=18, res_h=2):
    """绘制竖直圆柱障碍物"""
```
- **定义**: [L1693-1700](mmgj_transformer.py#L1693-L1700)
- **使用**: [L3323](mmgj_transformer.py#L3323)
- **使用次数**: 1 次

---

**22. draw_cylinder_y (第 1702-1709 行) ✅**
```python
def draw_cylinder_y(ax, cx, zc, r, y0, y1, color='darkorange', alpha=0.16, res_theta=18, res_h=2):
    """绘制水平圆柱障碍物"""
```
- **定义**: [L1702-L1709](mmgj_transformer.py#L1702-L1709)
- **使用**: [L3336](mmgj_transformer.py#L3336)
- **使用次数**: 1 次

---

**23. _plotly_add_cuboid (第 1711-1724 行) ✅**
```python
def _plotly_add_cuboid(fig, cx, cy, cz, hx, hy, hz, color='lightgray', opacity=0.65):
    """添加立方体几何到 Plotly 3D 图形"""
```
- **定义**: [L1711-1724](mmgj_transformer.py#L1711-L1724)
- **使用**: [L1919](mmgj_transformer.py#L1919) - 在 `save_interactive_3d_html` 中
- **使用次数**: 1 次

---

**24. _plotly_add_sphere (第 1726-1736 行) ✅**
```python
def _plotly_add_sphere(fig, cx, cy, cz, r, color='royalblue', opacity=0.75, res=16):
    """添加球体到 Plotly 图形"""
```
- **定义**: [L1726-1736](mmgj_transformer.py#L1726-L1736)
- **使用**: [L1927](mmgj_transformer.py#L1927)
- **使用次数**: 1 次

---

**25. _plotly_add_cylinder_z (第 1738-1748 行) ✅**
```python
def _plotly_add_cylinder_z(fig, cx, cy, r, z0, z1, color='teal', opacity=0.72, res_theta=22):
    """添加竖直圆柱到 Plotly 图形"""
```
- **定义**: [L1738-1748](mmgj_transformer.py#L1738-L1748)
- **使用**: [L1936](mmgj_transformer.py#L1936)
- **使用次数**: 1 次

---

**26. _plotly_add_cylinder_y (第 1750-1760 行) ✅**
```python
def _plotly_add_cylinder_y(fig, cx, zc, r, y0, y1, color='darkorange', opacity=0.72, res_theta=22):
    """添加水平圆柱到 Plotly 图形"""
```
- **定义**: [L1750-1760](mmgj_transformer.py#L1750-L1760)
- **使用**: [L1944](mmgj_transformer.py#L1944)
- **使用次数**: 1 次

---

**27. _is_ceiling_or_side_boundary_voxel (第 1762-1794 行) ✅**
```python
def _is_ceiling_or_side_boundary_voxel(box_xyz_half, env):
    """检查体素是否为边界/天花板（用于隐藏）"""
```
- **定义**: [L1762-1794](mmgj_transformer.py#L1762-L1794)
- **使用**: [L1915](mmgj_transformer.py#L1915) - 在 `save_interactive_3d_html` 中的过滤条件
- **使用次数**: 1 次

---

**28. save_interactive_3d_html (第 1796-1995 行) ✅**
```python
def save_interactive_3d_html(html_path, env, p_cpu, v_cpu, R_cpu=None, idx=0, axis_len=0.3, ...):
    """保存交互式 3D 轨迹 HTML（包含障碍物、速度着色、A* 路径等）"""
```
- **定义**: [L1796-1995](mmgj_transformer.py#L1796-L1995)
- **使用**: [L2954](mmgj_transformer.py#L2954) - 在轨迹保存中
- **使用次数**: 1 次
- **特色功能**:
  - 速度着色的 3D 轨迹
  - 无人机姿态坐标系（XYZ 轴）
  - A* 规划路径和采样路径
  - 所有障碍物类型的 3D 渲染
  - Plotly 交互式可视化

---

#### 第6组: 损失计算函数（3个）

**29. compute_overlap_loss_per_step (第 2066-2092 行) ✅**
```python
def compute_overlap_loss_per_step(p_history, sigma=0.5, time_window=10):
    """计算轨迹重叠惩罚（Step-wise 散度）"""
```
- **定义**: [L2066-2092](mmgj_transformer.py#L2066-L2092)
- **使用**: [L2354](mmgj_transformer.py#L2354) - 在主训练循环中
  ```python
  loss_exploration_seq = compute_overlap_loss_per_step(p_history, sigma=1.0, time_window=50).permute(1, 0)
  ```
- **使用次数**: 1 次

---

**30. compute_stuck_loss (第 2094-2145 行) ✅**
```python
def compute_stuck_loss(p_history, collision_depth, stuck_window=15, displacement_threshold=0.3):
    """
    计算卡住和碰撞时间惩罚。
    检测两种卡住状态：
    1. 局部窗口内位移过小
    2. 持续碰撞状态
    """
```
- **定义**: [L2094-2145](mmgj_transformer.py#L2094-L2145)
- **使用**: [L2360](mmgj_transformer.py#L2360) - 在主训练循环中
  ```python
  loss_stuck_seq, loss_collision_duration_seq, stuck_ratio = compute_stuck_loss(
      p_history, collision_depth,
      stuck_window=args.stuck_window,
      displacement_threshold=args.stuck_displacement_threshold
  )
  ```
- **使用次数**: 1 次

---

**31. unrolled_meta_rollout (第 2147-2262 行) ✅**
```python
def unrolled_meta_rollout(env, worknet, fast_params, state_normalizer, args, B, device):
    """
    验证 Rollout（使用虚拟更新的 Worker 参数）。
    计算元损失（位置 + 碰撞 + 高度 + 引导）。
    """
```
- **定义**: [L2147-2262](mmgj_transformer.py#L2147-L2262)
- **使用**: [L3170](mmgj_transformer.py#L3170) - 在 LGN 阶段的双层优化中
  ```python
  meta_loss_unrolled, meta_pos_ur, meta_coll_ur, meta_ctrl_ur = \
      unrolled_meta_rollout(env, worknet, fast_params, state_normalizer, args, B, device)
  ```
- **使用次数**: 1 次
- **关键作用**: 计算用于 LGN 梯度反向传播的元损失

---

### ❌ 未被使用的函数（2个）

#### 1. ⚠️ get_collision_wall_patches (第 1997-2064 行)

**未使用！** 💥

```python
@torch.no_grad()
def get_collision_wall_patches(points_xyz, walls, drone_radius, segment_len=0.45, contact_eps=0.02):
    """
    获取轨迹与墙面碰撞的补丁坐标。
    
    对接触墙体的轨迹点，计算其与墙体接触的区间段。
    用于碰撞可视化或碰撞分析。
    """
```

**定义位置**: [L1997-2064](mmgj_transformer.py#L1997-L2064)
- **代码行数**: 68 行
- **功能描述**:
  - 实现轨迹-墙体碰撞检测
  - 提取碰撞接触补丁
  - 支持矩形补丁合并（minimum_gap 参数）
- **关键操作**:
  1. 过滤表示行为高度的墙体
  2. 计算轨迹点到每个墙体的最近点距离
  3. 找出接触阈值内的步骤
  4. 为每个接触提取墙面参数化坐标
  5. 合并重叠的接触区间

**为什么未被使用**:
- 代码中通过 `save_interactive_3d_html` 可视化碰撞
- 但该函数在碰撞可视化中未被调用
- 可能是早期功能，已被其他方法替代
- **可能是遗留代码（Legacy Code）** ⚠️

**调用签名**:
```python
patches = get_collision_wall_patches(
    points_xyz,      # [T, 3] 轨迹点
    walls,           # [N_walls, 6] 墙体 [cx,cy,cz,hx,hy,hz]
    drone_radius,    # 无人机半径
    segment_len=0.45,     # 补丁长度
    contact_eps=0.02      # 接触判定阈值
)
```

**返回值**:
```python
patches = [
    {'xy': (x, y), 'width': w, 'height': h},
    ...
]
```

---

#### 2. ⚠️ get_map_view_bounds (第 1651-1682 行)

**未使用！** 💥

```python
@torch.no_grad()
def get_map_view_bounds(env, traj_xy, target_xy=None, pad=0.5):
    """
    自动计算包含轨迹和主要障碍物的地图视图边界。
    
    提供智能的坐标轴范围选择，适配 maze-like 和
    随机障碍物布局。
    """
```

**定义位置**: [L1651-1682](mmgj_transformer.py#L1651-L1682)
- **代码行数**: 32 行
- **功能描述**:
  - 自动计算 2D 地图视图范围
  - 支持轨迹和障碍物的边界包含
  - 自动填充（padding）处理
  - 处理退化情况（轨迹为单点等）
- **关键操作**:
  1. 收集轨迹的 X、Y 坐标
  2. 可选添加目标点坐标
  3. 提取所有墙体边界
  4. 计算 min/max 范围并应用边距
  5. 处理范围过小的情况

**为什么未被使用**:
- 主训练循环中直接计算范围
- 在 `save_interactive_3d_html` 中使用手动范围计算
- 该函数设计用于简化用户代码
- 可能是用于外部调用的文档 API
- **可能是公共工具函数（Public API），但内部未使用** ⚠️

**调用签名**:
```python
x_min, x_max, y_min, y_max = get_map_view_bounds(
    env,           # 环境对象
    traj_xy,       # [T, 2] 轨迹的 XY 平面坐标
    target_xy=None,    # [2] 可选的目标点坐标
    pad=0.5        # 边距（米）
)
```

**返回值**:
```python
(x_min, x_max, y_min, y_max)  # 2D 地图边界
```

---

## 🔍 正则化/归一化相关代码完整分析

### 归一化相关类和函数

#### 1. 状态归一化 (RunningMeanStd)
- **位置**: [L62-89](mmgj_transformer.py#L62-L89)
- **用途**: 在线计算状态的运行均值和方差，进行状态标准化
- **创建**: [L403](mmgj_transformer.py#L403)
- **更新**: [L2339](mmgj_transformer.py#L2339) - 在每个时间步中（update=True）
- **使用**: [L2185](mmgj_transformer.py#L2185) - 元 rollout 中（update=False）

#### 2. 损失归一化 (LossNormalizer)
- **位置**: [L92-111](mmgj_transformer.py#L92-L111)
- **用途**: 确保四个损失分量（速度、方向、避障、探索）贡献均衡
- **创建**: [L410](mmgj_transformer.py#L410) - 创建实例处理 4 个损失分量
- **使用**: [L2382-L2385](mmgj_transformer.py#L2382-L2385) - 主训练循环中
- **应用**:
  ```python
  loss_speed_n, loss_dir_n, loss_avoid_n, loss_expl_n = \
      loss_normalizer.normalize(
          loss_speed_seq, loss_direction_seq, loss_avoidance_seq, loss_exploration_seq
      )
  ```

#### 3. 安全向量归一化 (safe_normalize)
- **位置**: [L113-114](mmgj_transformer.py#L113-L114)
- **用途**: PyTorch F.normalize 的安全包装，预处理 NaN/Inf
- **使用位置**: [L1361](mmgj_transformer.py#L1361), [L1362](mmgj_transformer.py#L1362), [L1537](mmgj_transformer.py#L1537), [L1577](mmgj_transformer.py#L1577)

### 与正则化相关的关键参数
```python
# 状态归一化器配置
state_normalizer = RunningMeanStd(shape=(state_dim,)).to(device)

# 损失归一化器配置（4个损失成分）
loss_normalizer = LossNormalizer(4)  # normalize 4 loss components to equal scale
```

### 缺失的正则化
- **❌ L2 正则化**: 代码中未使用 L2 正则化项
- **❌ 权重衰减**: AdamW 优化器中可能有隐含权重衰减，但未在配置中显式设置
- **❌ 批量归一化**: 未使用 BatchNorm 层
- **❌ Dropout**: 代码中未使用 Dropout 正则化

---

## 📈 函数调用关系图

```
主训练循环 (L2285+)
├── 环境交互 (Rollout)
│   ├── env.render() 
│   ├── state_normalizer() ✅ L2339
│   ├── lgn() → current_weights
│   ├── worknet()
│   └── env.run()
│
├── 损失计算
│   ├── compute_overlap_loss_per_step() ✅ L2354
│   ├── compute_stuck_loss() ✅ L2360
│   ├── loss_normalizer.normalize() ✅ L2382
│   ├── compute_global_guidance_meta_loss() ✅ L2398
│   │   ├── sample_guidance_points() ✅ L1512
│   │   ├── compute_guidance_reference_from_planner() ✅ L1535
│   │   │   ├── _get_planner_pool() ✅ L1281
│   │   │   │   ├── _compute_planner_worker_count() ✅ L1042
│   │   │   │   └── _shutdown_planner_pool() ✅ L1047
│   │   │   └── _plan_sample_points_worker() ✅ 作为 pool.map 函数
│   │   └── compute_escape_penalty() ✅ L1599
│   │
│   └── unrolled_meta_rollout() (仅 LGN 阶段) ✅ L3170
│       └── compute_global_guidance_meta_loss() (在元 rollout 中调用)
│
├── 梯度保护
│   ├── get_grad_stats() ✅ L3192, L3199
│   ├── sanitize_module_() ✅ L3195, L3208
│   └── get_loss_to_worker_grad_norm() (4次) ✅ L3046-L3050
│
├── 日志记录
│   ├── smooth_dict() ✅ L3277
│   └── 日志收集 (scaler_q)
│
└── 轨迹保存 (is_save_trajectory_iter)
    ├── is_save_trajectory_iter() ✅ L2305
    ├── rotation_matrix_to_rpy_deg() ✅ L2324
    ├── save_interactive_3d_html() ✅ L2954
    │   ├── draw_sphere() ✅ L3309
    │   ├── draw_cylinder_z() ✅ L3323
    │   ├── draw_cylinder_y() ✅ L3336
    │   ├── _plotly_add_cuboid() ✅ L1919
    │   ├── _plotly_add_sphere() ✅ L1927
    │   ├── _plotly_add_cylinder_z() ✅ L1936
    │   ├── _plotly_add_cylinder_y() ✅ L1944
    │   ├── _is_ceiling_or_side_boundary_voxel() ✅ L1915
    │   └── GlobalPlanner.build_occupancy_grid() ✅ (用于 A* 路径展示)
    │
    └── ❌ 未使用的函数
        ├── get_collision_wall_patches() (未被调用)
        └── get_map_view_bounds() (未被调用)
```

---

## 📊 使用统计表

| 分类 | 函数名 | 定义行 | 使用次数 | 状态 | 重要性 |
|------|--------|--------|---------|------|--------|
| **工具** | safe_normalize | 113 | 4 | ✅ | ⭐⭐⭐ |
| | sanitize_tensor | 117 | 35+ | ✅ | ⭐⭐⭐⭐ |
| | sanitize_module_ | 122 | 2 | ✅ | ⭐⭐⭐ |
| | smooth_dict | 415 | 1 | ✅ | ⭐⭐ |
| | is_save_iter | 421 | 1 | ✅ | ⭐⭐ |
| | is_save_trajectory_iter | 425 | 1 | ✅ | ⭐⭐ |
| | rotation_matrix_to_rpy_deg | 429 | 1 | ✅ | ⭐⭐ |
| **梯度** | get_grad_stats | 442 | 2 | ✅ | ⭐⭐⭐ |
| | get_grad_norm_from_grads | 461 | 1 | ✅ | ⭐⭐ |
| | get_loss_to_worker_grad_norm | 477 | 4 | ✅ | ⭐⭐⭐ |
| | merge_intervals | 486 | 1 | ✅ | ⭐ |
| **规划** | _compute_planner_worker_count | 1020 | 1 | ✅ | ⭐⭐ |
| | _shutdown_planner_pool | 1028 | 2 | ✅ | ⭐⭐ |
| | _get_planner_pool | 1037 | 1 | ✅ | ⭐⭐⭐ |
| | _plan_sample_points_worker | 1060 | 1 | ✅ | ⭐⭐⭐⭐ |
| | compute_guidance_reference_from_planner | 1165 | 1 | ✅ | ⭐⭐⭐⭐⭐ |
| **损失** | compute_escape_penalty | 1344 | 1 | ✅ | ⭐⭐ |
| | sample_guidance_points | 1377 | 1 | ✅ | ⭐⭐⭐⭐ |
| | compute_global_guidance_meta_loss | 1469 | 2 | ✅ | ⭐⭐⭐⭐⭐ |
| **可视化** | draw_sphere | 1684 | 1 | ✅ | ⭐⭐ |
| | draw_cylinder_z | 1693 | 1 | ✅ | ⭐⭐ |
| | draw_cylinder_y | 1702 | 1 | ✅ | ⭐⭐ |
| | _plotly_add_cuboid | 1711 | 1 | ✅ | ⭐⭐ |
| | _plotly_add_sphere | 1726 | 1 | ✅ | ⭐⭐ |
| | _plotly_add_cylinder_z | 1738 | 1 | ✅ | ⭐⭐ |
| | _plotly_add_cylinder_y | 1750 | 1 | ✅ | ⭐⭐ |
| | _is_ceiling_or_side_boundary_voxel | 1762 | 1 | ✅ | ⭐ |
| | save_interactive_3d_html | 1796 | 1 | ✅ | ⭐⭐⭐ |
| **❌ 遗留** | ❌ get_collision_wall_patches | 1997 | 0 | ❌ | ⭐ |
| **❌ 公共工具** | ❌ get_map_view_bounds | 1651 | 0 | ❌ | ⭐⭐ |
| **最终损失** | compute_overlap_loss_per_step | 2066 | 1 | ✅ | ⭐⭐⭐ |
| | compute_stuck_loss | 2094 | 1 | ✅ | ⭐⭐⭐⭐ |
| | unrolled_meta_rollout | 2147 | 1 | ✅ | ⭐⭐⭐⭐⭐ |
| **总计** | | | | **31/33** | |

---

## ✨ 关键发现

### 🎯 核心发现
1. **使用率高** (93.9%) - 大部分代码都在使用中
2. **未使用函数少** (2个) - 可能是遗留代码或公共 API
3. **完整的正则化体系**:
   - ✅ 状态归一化 (RunningMeanStd)
   - ✅ 损失归一化 (LossNormalizer)
   - ✅ 张量清理 (sanitize_tensor)
4. **高度模块化**:
   - 独立的全局规划器
   - 可视化工具集中
   - 梯度监控完整

### ⚠️ 建议
1. **删除未使用函数**:
   - `get_collision_wall_patches` - 代码行数: 68 行（如果确实没有计划使用）
   - `get_map_view_bounds` - 代码行数: 32 行（or 保留作为公共 API 文档）

2. **添加的正则化**:
   - 考虑添加 L2 正则化项到元损失
   - 考虑在网络中添加 BatchNorm/Dropout

3. **代码段清理**:
   - 若 `get_collision_wall_patches` 是遗留代码，建议清理
   - 若 `get_map_view_bounds` 是公共工具，建议添加文档示例

---

## 📝 附录：完整定义位置索引

| # | 函数/类 | 行号范围 | 类型 |
|---|--------|---------|------|
| 1 | RunningMeanStd | 62-89 | 类 ✅ |
| 2 | LossNormalizer | 92-111 | 类 ✅ |
| 3 | safe_normalize | 113-114 | 函数 ✅ |
| 4 | sanitize_tensor | 117-119 | 函数 ✅ |
| 5 | sanitize_module_ | 122-127 | 函数 ✅ |
| 6 | smooth_dict | 415-419 | 函数 ✅ |
| 7 | is_save_iter | 421-423 | 函数 ✅ |
| 8 | is_save_trajectory_iter | 425-427 | 函数 ✅ |
| 9 | rotation_matrix_to_rpy_deg | 429-439 | 函数 ✅ |
| 10 | get_grad_stats | 442-460 | 函数 ✅ |
| 11 | get_grad_norm_from_grads | 461-476 | 函数 ✅ |
| 12 | get_loss_to_worker_grad_norm | 477-485 | 函数 ✅ |
| 13 | merge_intervals | 486-502 | 函数 ✅ |
| 14 | GlobalPlanner | 504-1017 | 类 ✅ |
| 15 | _compute_planner_worker_count | 1020-1026 | 函数 ✅ |
| 16 | _shutdown_planner_pool | 1028-1035 | 函数 ✅ |
| 17 | _get_planner_pool | 1037-1055 | 函数 ✅ |
| 18 | _plan_sample_points_worker | 1060-1259 | 函数 ✅ |
| 19 | compute_guidance_reference_from_planner | 1165-1343 | 函数 ✅ |
| 20 | compute_escape_penalty | 1344-1375 | 函数 ✅ |
| 21 | sample_guidance_points | 1377-1467 | 函数 ✅ |
| 22 | compute_global_guidance_meta_loss | 1469-1649 | 函数 ✅ |
| 23 | **❌ get_map_view_bounds** | 1651-1682 | 函数 ❌ |
| 24 | draw_sphere | 1684-1691 | 函数 ✅ |
| 25 | draw_cylinder_z | 1693-1700 | 函数 ✅ |
| 26 | draw_cylinder_y | 1702-1709 | 函数 ✅ |
| 27 | _plotly_add_cuboid | 1711-1724 | 函数 ✅ |
| 28 | _plotly_add_sphere | 1726-1736 | 函数 ✅ |
| 29 | _plotly_add_cylinder_z | 1738-1748 | 函数 ✅ |
| 30 | _plotly_add_cylinder_y | 1750-1760 | 函数 ✅ |
| 31 | _is_ceiling_or_side_boundary_voxel | 1762-1794 | 函数 ✅ |
| 32 | save_interactive_3d_html | 1796-1995 | 函数 ✅ |
| 33 | **❌ get_collision_wall_patches** | 1997-2064 | 函数 ❌ |
| 34 | compute_overlap_loss_per_step | 2066-2092 | 函数 ✅ |
| 35 | compute_stuck_loss | 2094-2145 | 函数 ✅ |
| 36 | unrolled_meta_rollout | 2147-2262 | 函数 ✅ |

---

**报告生成时间**: 2026-04-01
**分析工具**: 语义搜索 + 调用链检索
