# TensorBoard 参数详解与训练指标

本文档详细说明了 mmgj_transformer.py 中记录到 TensorBoard 的所有参数含义及其应该观察到的现象。

## 📊 主要 Loss 指标

### Loss/1_Proxy_Total
- **含义**：Worker 策略网络的总代理损失
- **现象**：初期波动较大，训练中期应该趋势向下
- **异常**：持续增加则需要调整学习率或参数

### Loss/2_Meta_Total
- **含义**：LGN 元学习器在反向 unroll 中的总优化目标
- **现象**：通常小于 proxy_loss，训练中期应该相对稳定
- **异常**：大的波动表示元学习不稳定

### Loss/3_LGN_Unrolled_Meta (仅在 LGN phase)
- **含义**：LGN 网络的 unroll 优化目标
- **现象**：应该逐步下降

---

## 🎯 权重监控（LGN 输出）

LGN 网络输出 4 个权重，控制代理损失的相对重要性。这些权重经过 softmax 和温度变换归一化到 [0, 1] 区间。

### Weights/0_Speed (速度权重)
- **含义**：速度目标的权重
- **现象**：快速任务中应该较大，慢速避障中应该较小

### Weights/1_Direction (方向权重)
- **含义**：方向一致性的权重
- **现象**：狭窄通道中应该较大，开阔空间中应该较小

### Weights/2_Avoidance (避障权重)
- **含义**：碰撞避免的权重
- **现象**：复杂障碍环境中应该较大，空旷环境中应该较小

### Weights/3_Exploration (探索权重)
- **含义**：鼓励多样化的权重
- **现象**：初期可能较小，若陷入局部最优会增大

### Weights_Raw/* (原始 logits)
- **含义**：softmax 前的原始权重 logits
- **诊断用**：logits 的 max-min 差距大表示权重分布尖锐

### Weights_Effective/* (有效权重)
- **含义**：应用了 lgn_weight_floor 下限保护后的权重
- **现象**：应该 >= lgn_weight_floor (通常 0.01)

### Weight_Stats/* (权重统计)
- **Raw_Min**: 应该 >= lgn_weight_floor
- **Raw_Max**: 应该接近 1.0
- **Raw_Mean**: 应该在 0.2-0.8 之间
- **Entropy**: 高熵 (>0.8) 表示均匀，低熵 (<0.5) 表示某权重主导

---

## 🔴 Proxy Loss 分项

代理损失是 Worker 网络的训练目标。

### Proxy_Comp/0_Speed
- **含义**：鼓励无人机达到自适应速度目标
- **现象**：初期可能较大，应逐渐减小
- **异常**：持续增加表示无人机无法跟上速度

### Proxy_Comp/1_Direction
- **含义**：方向与参考路径对齐的损失
- **现象**：应该相对较小 (<0.5)
- **异常**：太大表示规划引导效果差

### Proxy_Comp/2_Avoidance
- **含义**：碰撞惩罚项（安全余度内）
- **现象**：应该较小且趋势向下
- **异常**：>1.0 说明无人机频繁接近障碍物

### Proxy_Comp/2_1_Collision_Depth
- **含义**：实际穿入障碍的深度（米）
- **现象**：应该接近 0
- **异常**：>0.1m 说明有明显穿墙

### Proxy_Comp/3_Exploration
- **含义**：鼓励多样化的散度惩罚
- **现象**：通常较小，训练后减小说明策略收敛

### Proxy_Comp/4_Height
- **含义**：与参考路径的高度偏差
- **现象**：应该较小
- **异常**：>0.5 说明高度控制不当

---

## 🧱 卡住 (Stuck) 损失

检测无人机移动缓慢或被困的情况。

### Proxy_Comp/5_Stuck
- **含义**：局部移动过慢的惩罚
- **现象**：应该接近 0
- **异常**：>0.1 说明经常卡住

### Proxy_Comp/6_Collision_Duration
- **含义**：连续碰撞时长的惩罚
- **现象**：应该接近 0
- **异常**：>0.2 说明碰撞后难以恢复

### Stuck/Ratio
- **含义**：Episode 中卡住步数占比 (0-1)
- **现象**：应该 < 0.1
- **异常**：>0.3 说明导航效率很低

### Stuck/Collision_Streak_Mean
- **含义**：单次碰撞的平均持续步数
- **现象**：应该 < 5 步
- **异常**：>10 步说明碰撞后无法脱离

---

## 🛡️ Meta Loss 分项

元学习器的训练目标，用于优化 Worker 的损失权重。

### Meta_Comp/1_Position
- **含义**：到达终点的惩罚
- **现象**：应该相对较小

### Meta_Comp/2_Collision (及其子项)
- **2_Collision_Soft**: 靠近墙体的连续梯度
- **2_Collision_Hard**: 穿入深度的平方惩罚
- **2_Collision_Event**: Episode 级碰撞事件
- **2_Collision_Event_Rate**: 真实碰撞比例，应该逐渐 <0.1

### Meta_Comp/3_Control
- **含义**：对平滑控制的惩罚
- **异常**：>0.5 说明控制量抖动过大

### Meta_Comp/4_Height
- **含义**：高度偏差惩罚
- **异常**：>0.3 说明高度控制不稳定

---

## 🗺️ 全局规划引导损失

使用 A* 规划器的参考路径来指导学习。

### Meta_Comp/5_Guidance
- **含义**：整个全局规划引导加权损失
- **现象**：应该相对较小但稳定
- **异常**：剧烈波动说明 A* 规划不稳定

### Guidance/* (详细分项)

| 参数 | 含义 | 正常范围 |
|------|------|--------|
| Dir_Align | 方向对齐误差 | <0.5 |
| Overspeed | 超速惩罚 | 较小 |
| Speed_Diff | 速度差异 | <0.5 |
| Lateral_Error | 到规划路径的距离 | <1.0m |
| Valid_Ratio | 规划成功比例 | >0.9 |
| Collision_Ratio | 规划路径碰撞比例 | ~0 |
| Planner_Success_Ratio | A* 规划成功率 | >0.95 |

---

## 📈 性能指标

### Metrics/Success_Rate
- **含义**：到达目标的成功率
- **现象**：初期 ~0，训练后 >0.8，好的模型 >0.95

### Metrics/Avg_Speed
- **含义**：Episode 中平均速度
- **现象**：3-8 m/s，初期可能 0.5-2.0，应逐步加快

### Metrics/Episode_Length
- **含义**：Episode 持续步数
- **现象**：<timesteps 参数，通常 100-280 步

### Metrics/Max_Speed
- **含义**：最大瞬间速度
- **异常**：>15 m/s 说明速度限制失效

### Metrics/Min_Speed
- **含义**：最小瞬间速度
- **异常**：接近 0 说明无人机卡住或悬停

---

## 🎮 控制信号

### Control/Accel_Cmd_Norm_Mean
- **含义**：加速度命令平均模长
- **现象**：1-5 m/s²

### Control/Accel_Cmd_*_Mean
- **含义**：各轴加速度的平均值（可正可负）
- **现象**：应该接近 0（平衡）

### Control/Accel_Cmd_*_AbsMean
- **含义**：各轴加速度的绝对值平均
- **现象**：X/Y <5 m/s²，Z <3 m/s²

---

## 📊 状态归一化

### Norm/State_Mean
- **含义**：运行平均的状态均值
- **现象**：应该接近 0

### Norm/State_Var
- **含义**：运行平均的状态方差
- **现象**：应该接近 1.0

---

## 🔍 梯度监控

### Grad/Worker_Global_Norm
- **含义**：Worker 全局梯度范数
- **现象**：<100 为正常，1-50 是理想范围
- **异常**：>1000 表示梯度爆炸

### Grad/Worker_NonFinite_Count
- **含义**：非有限梯度 (NaN/Inf) 数量
- **现象**：应该 = 0
- **异常**：>0 表示数值不稳定

### Grad/LGN_*
- **含义**：LGN 网络的梯度统计
- **现象**：通常大于 Worker（二阶梯度）

---

## 🛡️ 梯度爆炸保护

### Grad_Protection/Explosion_Count_Total
- **含义**：检测到梯度爆炸的总次数
- **现象**：应该接近 0 或增速很慢

### Grad_Protection/*_Consecutive_Explosions
- **含义**：连续爆炸次数
- **现象**：应该 < grad_explosion_skip_window

---

## 🔗 LGN 梯度链路诊断

### LGN_Diag/Grad_Coverage_Ratio
- **含义**：有梯度的 LGN 参数比例
- **现象**：应该接近 1.0
- **异常**：<0.95 说明存在断梯问题

### LGN_Diag/Weights_Seq_Requires_Grad
- **含义**：权重序列是否启用梯度
- **现象**：LGN phase = 1.0，Worker phase = 0.0

---

## 💡 训练建议

1. **快速检查**：看 Loss/1_Proxy_Total 和 Loss/2_Meta_Total 是否下降
2. **权重学习**：Weights/* 应该随环境动态变化，不应该固定
3. **成功率**：Metrics/Success_Rate 是最直观的指标
4. **梯度健康**：Grad/ 系列应该稳定，非有限计数应该为 0
5. **避障性能**：Proxy_Comp/2_Avoidance 和碰撞率应该逐步降低
6. **速度控制**：Metrics/Avg_Speed 应该在合理范围内逐步增加

---

## 📝 状态/Train_Mode 和 Status/Maze_Age

每 25 次迭代记录一次：
- **Status/Train_Mode**: 1 = LGN phase，0 = Worker phase
- **Status/Maze_Age**: 当前迷宫的年龄（多少次迭代后会重新生成）
