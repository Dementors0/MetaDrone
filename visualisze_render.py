import torch
import matplotlib.pyplot as plt
from env_cuda import Env
import numpy as np


def debug_render(env):
    # ----------------------------------------
    # 1. 强制设置无人机位置和姿态 (关键步骤)
    # ----------------------------------------
    print("正在重置无人机位置...")

    # 强制位置：移到高处 (x=0, y=0, z=5)，防止卡在任何墙里
    env.p = torch.tensor([[0.0, 0.0, 5.0]], device='cuda')

    # 强制姿态：水平放置，不旋转
    env.R = torch.eye(3, device='cuda').unsqueeze(0)  # [1, 3, 3] 单位矩阵

    # 强制速度：设为0
    env.v = torch.zeros_like(env.p)

    # 强制旧位置：为了光流计算不报错，设为和当前位置一样
    env.p_old = env.p.clone()
    env.R_old = env.R.clone()

    # ----------------------------------------
    # 2. 渲染图像
    # ----------------------------------------
    print("正在渲染...")
    # ctl_dt 稍微大一点，确保物理状态更新
    canvas, _ = env.render(ctl_dt=0.02)

    # 取出数据 [Batch, H, W] -> [H, W]
    depth_map = canvas[0].detach().cpu().numpy()

    # ----------------------------------------
    # 3. 数据诊断 (这里会告诉你为什么黑)
    # ----------------------------------------
    print("-" * 30)
    print(f"【数据诊断报告】")
    print(f"图像尺寸: {depth_map.shape}")
    print(f"最大值 (Max): {depth_map.max()}")
    print(f"最小值 (Min): {depth_map.min()}")
    print(f"平均值 (Mean): {depth_map.mean()}")

    unique_vals = np.unique(depth_map)
    if len(unique_vals) < 10:
        print(f"警告：整个图像只有 {len(unique_vals)} 种数值: {unique_vals}")
        if np.all(unique_vals == 0):
            print("结论：图像全是 0 -> 相机可能被完全遮挡，或渲染器认为前方无限远返回0。")
    else:
        print("结论：图像包含丰富的数据，应该能看到东西。")
    print("-" * 30)

    # ----------------------------------------
    # 4. 智能绘图
    # ----------------------------------------
    plt.figure(figsize=(10, 5))

    # 子图 1: 原始深度图
    plt.subplot(1, 2, 1)
    # 使用 'magma' 或 'inferno' 配色，对暗部细节更敏感
    # 强制 vmin/vmax 防止单一值导致全屏同色
    if depth_map.max() > 0:
        plt.imshow(depth_map, cmap='magma', vmin=0, vmax=depth_map.max())
    else:
        plt.imshow(depth_map, cmap='gray')

    plt.colorbar(label='Depth')
    plt.title("Depth Map (Forced View)")

    # 子图 2: 直方图 (看看数值分布)
    plt.subplot(1, 2, 2)
    plt.hist(depth_map.flatten(), bins=50, color='gray')
    plt.title("Value Distribution")
    plt.yscale('log')  # 对数坐标，防止 0 值太多掩盖其他值

    plt.savefig("debug_result.png")
    print("✅ 诊断图片已保存为: debug_result.png")


if __name__ == "__main__":
    try:
        # 初始化环境
        # 注意：这里我们生成一些障碍物，width/height 设大一点看清楚
        env = Env(batch_size=1, width=256, height=256, grad_decay=0.99, device='cuda', single=True)
        env.reset()

        # 为了保证有东西可看，我们手动造一个大球在无人机正下方
        # 无人机在 z=5, 球在 z=0, 半径 2
        env.balls[0, 0] = torch.tensor([0.0, 0.0, 0.0, 2.0], device='cuda')

        debug_render(env)

    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback

        traceback.print_exc()