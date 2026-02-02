import open3d as o3d
import torch
import argparse
from env_cuda import Env


def visualize_3d_open3d(env, batch_idx=0):
    geometries = []

    # 获取数据并转为 numpy
    voxels = env.voxels[batch_idx].detach().cpu().numpy()
    balls = env.balls[batch_idx].detach().cpu().numpy()  # [N, 4] -> x,y,z,r
    cyls = env.cyl[batch_idx].detach().cpu().numpy()  # [N, 3] -> x,y,r (垂直圆柱)

    # ---------------------------
    # 1. 绘制方块障碍物 (Voxels) / 墙壁
    # ---------------------------
    for v in voxels:
        x, y, z, w, l, h = v
        # 过滤掉无效数据 (-50 通常是无效填充值)
        if x < -40: continue

        # 创建 Box
        box = o3d.geometry.TriangleMesh.create_box(width=2 * w, height=2 * l, depth=2 * h)
        box.translate((x - w, y - l, z - h))
        box.compute_vertex_normals()  # 关键：计算法线，否则没有光影

        # 如果尺寸很大（可能是墙壁或地板），给深灰色；小的给浅灰色
        if w > 3 or l > 3:
            box.paint_uniform_color([0.3, 0.3, 0.3])  # 深灰外墙
        else:
            box.paint_uniform_color([0.7, 0.7, 0.7])  # 浅灰障碍

        geometries.append(box)

    # ---------------------------
    # 2. 绘制球体障碍物 (Balls)
    # ---------------------------
    for b in balls:
        x, y, z, r = b
        # 过滤无效半径或坐标
        if r <= 0 or x < -40: continue

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere.translate((x, y, z))
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.1, 0.5, 0.8])  # 蓝色球体
        geometries.append(sphere)

    # ---------------------------
    # 3. 绘制垂直圆柱体 (Cylinders)
    # env_cuda 中 cyl 只有 (x, y, r)，高度通常是贯穿的
    # ---------------------------
    for c in cyls:
        x, y, r = c
        if r <= 0 or x < -40: continue

        # 假设圆柱体高度为 6 (覆盖大部分空间)
        height = 6.0
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=height)
        # Open3D 圆柱默认中心在原点，轴向是 Y 轴? 不，通常是 Z 轴
        # 我们先移动到位置。注意 env_cuda 的圆柱好像没有 Z 坐标，说明是无限高的柱子
        # 我们把它放在 Z=0 附近
        cylinder.translate((x, y, 0))
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color([0.8, 0.5, 0.1])  # 橙色圆柱
        geometries.append(cylinder)

    # ---------------------------
    # 4. 起点 (绿色) 和 终点 (红色)
    # ---------------------------
    if hasattr(env, 'p_init'):
        p_start = env.p_init[batch_idx].detach().cpu().numpy()
        s_start = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        s_start.translate(p_start)
        s_start.compute_vertex_normals()
        s_start.paint_uniform_color([0, 1, 0])  # 绿
        geometries.append(s_start)

    if hasattr(env, 'p_end'):
        p_end = env.p_end[batch_idx].detach().cpu().numpy()
        s_end = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        s_end.translate(p_end)
        s_end.compute_vertex_normals()
        s_end.paint_uniform_color([1, 0, 0])  # 红
        geometries.append(s_end)

    # ---------------------------
    # 5. 坐标轴
    # ---------------------------
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    geometries.append(mesh_frame)

    # ---------------------------
    # 6. 开始可视化
    # ---------------------------
    print("------------------------------------------------")
    print("可视化窗口已打开！操作指南：")
    print("1. 鼠标左键拖动：旋转视角")
    print("2. 鼠标滚轮：缩放 (如果你看到灰色方块，请用力滚动滚轮进入内部！)")
    print("3. Ctrl + 左键拖动：平移")
    print("4. 按 'W' 键：切换线框模式 (可以看到内部)")
    print("------------------------------------------------")

    o3d.visualization.draw_geometries(geometries,
                                      window_name=f"DiffPhysDrone Visualization",
                                      width=1024, height=768)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=1)
    args = parser.parse_args()

    try:
        # 初始化环境 (single=True 修复 batch_size=1 的问题)
        env = Env(batch_size=1, width=64, height=64, grad_decay=0.99, device='cuda', single=True)
        env.reset()
        print("环境生成完毕，启动 Open3D...")

        visualize_3d_open3d(env, batch_idx=0)

    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()