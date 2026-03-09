"""
独立脚本：分别按 env_maze.py 和 env_maze_easy.py 的迷宫生成规则
各生成一张俯视图，保存为 maze_full_topview.png 和 maze_easy_topview.png
"""
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_maze_dfs(cols, rows):
    """DFS 生成迷宫，返回通道集合 passages"""
    visited = set()
    stack = [(0, 0)]
    visited.add((0, 0))
    passages = set()
    while stack:
        c, r = stack[-1]
        neighbors = []
        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < cols and 0 <= nr < rows and (nc, nr) not in visited:
                neighbors.append((nc, nr))
        if neighbors:
            nc, nr = random.choice(neighbors)
            visited.add((nc, nr))
            stack.append((nc, nr))
            passages.add(tuple(sorted(((c, r), (nc, nr)))))
        else:
            stack.pop()
    return passages


# ============================================================
#  1. env_maze.py — 完整迷宫 (Full Maze)
# ============================================================
def draw_full_maze():
    cols, rows = 8, 18
    cell_size = 1.0
    y_offset = 9.0
    th = 0.1

    passages = generate_maze_dfs(cols, rows)

    walls = []
    # 竖直墙
    for r in range(rows):
        for c in range(cols + 1):
            is_wall = (c == 0 or c == cols)
            if not is_wall and tuple(sorted(((c - 1, r), (c, r)))) not in passages:
                is_wall = True
            if is_wall:
                cx = float(c) * cell_size
                cy = (r + 0.5) * cell_size - y_offset
                walls.append((cx, cy, th, 0.5 * cell_size))
    # 水平墙
    for c in range(cols):
        for r in range(rows + 1):
            is_wall = (r == 0 or r == rows)
            if not is_wall and tuple(sorted(((c, r - 1), (c, r)))) not in passages:
                is_wall = True
            if is_wall:
                cx = (c + 0.5) * cell_size
                cy = float(r) * cell_size - y_offset
                walls.append((cx, cy, 0.5 * cell_size, th))

    # 随机起点/终点
    sc, sr = random.randint(0, cols - 1), random.randint(0, rows - 1)
    ec, er = random.randint(0, cols - 1), random.randint(0, rows - 1)
    while (sc, sr) == (ec, er):
        ec, er = random.randint(0, cols - 1), random.randint(0, rows - 1)
    start = ((sc + 0.5) * cell_size, (sr + 0.5) * cell_size - y_offset)
    goal  = ((ec + 0.5) * cell_size, (er + 0.5) * cell_size - y_offset)

    fig, ax = plt.subplots(figsize=(6, 12))
    ax.set_title('env_maze.py — Full Maze\n(8×18, cell=1 m, wall_thickness=0.2 m, 100% internal walls)',
                 fontsize=11)
    for cx, cy, hdx, hdy in walls:
        rect = patches.Rectangle((cx - hdx, cy - hdy), 2 * hdx, 2 * hdy,
                                  linewidth=0.3, edgecolor='black', facecolor='#4a4a4a')
        ax.add_patch(rect)
    ax.plot(*start, 'go', markersize=10, label='Start')
    ax.plot(*goal,  'r*', markersize=14, label='Goal')
    ax.set_xlim(-0.5, cols * cell_size + 0.5)
    ax.set_ylim(-y_offset - 0.5, y_offset + 0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    n_walls = len(walls)
    out = 'maze_full_topview.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Full Maze] walls={n_walls}  saved → {out}')
    return n_walls


# ============================================================
#  2. env_maze_easy.py — 简易迷宫 (Easy Maze)
# ============================================================
def draw_easy_maze():
    cols, rows = 8, 18
    cell_size = 1.5
    y_offset = rows * cell_size / 2.0  # 13.5
    th = 0.1
    wall_keep_prob = 0.65

    passages = generate_maze_dfs(cols, rows)

    walls = []
    # 竖直墙（保留 65% 内部墙）
    for r in range(rows):
        for c in range(cols + 1):
            is_wall = (c == 0 or c == cols)
            if not is_wall and tuple(sorted(((c - 1, r), (c, r)))) not in passages:
                if random.random() < wall_keep_prob:
                    is_wall = True
            if is_wall:
                cx = float(c) * cell_size
                cy = (r + 0.5) * cell_size - y_offset
                walls.append((cx, cy, th, 0.5 * cell_size))
    # 水平墙（保留 65% 内部墙）
    for c in range(cols):
        for r in range(rows + 1):
            is_wall = (r == 0 or r == rows)
            if not is_wall and tuple(sorted(((c, r - 1), (c, r)))) not in passages:
                if random.random() < wall_keep_prob:
                    is_wall = True
            if is_wall:
                cx = (c + 0.5) * cell_size
                cy = float(r) * cell_size - y_offset
                walls.append((cx, cy, 0.5 * cell_size, th))

    # 随机起点/终点
    sc, sr = random.randint(0, cols - 1), random.randint(0, rows - 1)
    ec, er = random.randint(0, cols - 1), random.randint(0, rows - 1)
    while (sc, sr) == (ec, er):
        ec, er = random.randint(0, cols - 1), random.randint(0, rows - 1)
    start = ((sc + 0.5) * cell_size, (sr + 0.5) * cell_size - y_offset)
    goal  = ((ec + 0.5) * cell_size, (er + 0.5) * cell_size - y_offset)

    fig, ax = plt.subplots(figsize=(6, 14))
    ax.set_title('env_maze_easy.py — Medium Maze\n(8×18, cell=1.5 m, wall_thickness=0.2 m, 65% internal walls)',
                 fontsize=11)
    for cx, cy, hdx, hdy in walls:
        rect = patches.Rectangle((cx - hdx, cy - hdy), 2 * hdx, 2 * hdy,
                                  linewidth=0.3, edgecolor='black', facecolor='#4a4a4a')
        ax.add_patch(rect)
    ax.plot(*start, 'go', markersize=10, label='Start')
    ax.plot(*goal,  'r*', markersize=14, label='Goal')
    ax.set_xlim(-0.5, cols * cell_size + 0.5)
    ax.set_ylim(-y_offset - 0.5, y_offset + 0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    n_walls = len(walls)
    out = 'maze_easy_topview.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Easy Maze] walls={n_walls}  saved → {out}')
    return n_walls


if __name__ == '__main__':
    n_full = draw_full_maze()
    n_easy = draw_easy_maze()
    print(f'\n=== Summary ===')
    print(f'Full Maze: {n_full} wall segments')
    print(f'Easy Maze: {n_easy} wall segments')
    print(f'Difference: Full has ~{n_full - n_easy} more walls')
