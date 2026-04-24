import heapq
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PLANNER_DRONE_RADIUS = 0.13
_DIJKSTRA_EPS = 1e-12


def build_occupancy_grid_from_obstacles(
    voxels: np.ndarray,
    balls: np.ndarray,
    cyl: np.ndarray,
    cyl_h: np.ndarray,
    resolution: float = 0.3,
    margin: float = 0.07,
    drone_radius: float = PLANNER_DRONE_RADIUS,
    bounds: Optional[Dict[str, float]] = None,
    z_min: float = 0.0,
    z_max: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Build occupancy grid from env obstacle tensors using planner-compatible inflation semantics."""
    if bounds is None:
        bounds = {
            "x_min": -1.0,
            "x_max": 11.0,
            "y_min": -13.0,
            "y_max": 13.0,
        }

    x_min = float(bounds["x_min"])
    x_max = float(bounds["x_max"])
    y_min = float(bounds["y_min"])
    y_max = float(bounds["y_max"])

    nx = int(math.ceil((x_max - x_min) / resolution))
    ny = int(math.ceil((y_max - y_min) / resolution))
    nz = int(math.ceil((z_max - z_min) / resolution))

    grid = np.zeros((nx, ny, nz), dtype=np.uint8)
    origin = np.asarray([x_min, y_min, z_min], dtype=np.float32)
    shape = (nx, ny, nz)
    total_margin = max(0.0, float(margin)) + max(0.0, float(drone_radius))

    # Voxels
    if voxels is not None and voxels.size > 0:
        for vox in voxels:
            cx, cy, cz, hx, hy, hz = [float(v) for v in vox[:6]]
            if hx > 15 or hy > 15 or hz > 15:
                continue
            x0 = max(0, int((cx - hx - total_margin - x_min) / resolution))
            x1 = min(nx, int((cx + hx + total_margin - x_min) / resolution) + 1)
            y0 = max(0, int((cy - hy - total_margin - y_min) / resolution))
            y1 = min(ny, int((cy + hy + total_margin - y_min) / resolution) + 1)
            z0 = max(0, int((cz - hz - total_margin - z_min) / resolution))
            z1 = min(nz, int((cz + hz + total_margin - z_min) / resolution) + 1)
            grid[x0:x1, y0:y1, z0:z1] = 1

    # Balls
    if balls is not None and balls.size > 0:
        for b in balls:
            bx, by, bz, br = [float(v) for v in b[:4]]
            r = br + total_margin
            x0 = max(0, int((bx - r - x_min) / resolution))
            x1 = min(nx, int((bx + r - x_min) / resolution) + 1)
            y0 = max(0, int((by - r - y_min) / resolution))
            y1 = min(ny, int((by + r - y_min) / resolution) + 1)
            z0 = max(0, int((bz - r - z_min) / resolution))
            z1 = min(nz, int((bz + r - z_min) / resolution) + 1)
            for ix in range(x0, x1):
                px = x_min + (ix + 0.5) * resolution
                for iy in range(y0, y1):
                    py = y_min + (iy + 0.5) * resolution
                    for iz in range(z0, z1):
                        pz = z_min + (iz + 0.5) * resolution
                        if (px - bx) ** 2 + (py - by) ** 2 + (pz - bz) ** 2 < r * r:
                            grid[ix, iy, iz] = 1

    # Vertical cylinders (z axis)
    if cyl is not None and cyl.size > 0:
        for c in cyl:
            cx, cy, cr = [float(v) for v in c[:3]]
            r = cr + total_margin
            x0 = max(0, int((cx - r - x_min) / resolution))
            x1 = min(nx, int((cx + r - x_min) / resolution) + 1)
            y0 = max(0, int((cy - r - y_min) / resolution))
            y1 = min(ny, int((cy + r - y_min) / resolution) + 1)
            for ix in range(x0, x1):
                px = x_min + (ix + 0.5) * resolution
                for iy in range(y0, y1):
                    py = y_min + (iy + 0.5) * resolution
                    if (px - cx) ** 2 + (py - cy) ** 2 < r * r:
                        grid[ix, iy, :] = 1

    # Horizontal cylinders (y axis)
    if cyl_h is not None and cyl_h.size > 0:
        for c in cyl_h:
            cx, cz, cr = [float(v) for v in c[:3]]
            r = cr + total_margin
            x0 = max(0, int((cx - r - x_min) / resolution))
            x1 = min(nx, int((cx + r - x_min) / resolution) + 1)
            z0 = max(0, int((cz - r - z_min) / resolution))
            z1 = min(nz, int((cz + r - z_min) / resolution) + 1)
            for ix in range(x0, x1):
                px = x_min + (ix + 0.5) * resolution
                for iz in range(z0, z1):
                    pz = z_min + (iz + 0.5) * resolution
                    if (px - cx) ** 2 + (pz - cz) ** 2 < r * r:
                        grid[ix, :, iz] = 1

    return grid, origin, shape


def world_to_grid_float(points: torch.Tensor, origin: torch.Tensor, resolution: float) -> torch.Tensor:
    return (points - origin) / float(resolution)


def world_to_grid_index(pos: np.ndarray, origin: np.ndarray, shape: Tuple[int, int, int], resolution: float) -> Tuple[int, int, int]:
    idx = ((pos - origin) / float(resolution)).astype(np.int64)
    idx = np.clip(idx, 0, np.asarray(shape, dtype=np.int64) - 1)
    return int(idx[0]), int(idx[1]), int(idx[2])


def _neighbors26() -> List[Tuple[int, int, int, float]]:
    n = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                n.append((dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz)))
    return n


_NEI26 = _neighbors26()


def _find_nearest_free(occupancy: np.ndarray, start: Tuple[int, int, int], max_r: int = 20) -> Optional[Tuple[int, int, int]]:
    sx, sy, sz = start
    nx, ny, nz = occupancy.shape
    if occupancy[sx, sy, sz] == 0:
        return start
    for r in range(1, max_r + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r and abs(dz) != r:
                        continue
                    x = sx + dx
                    y = sy + dy
                    z = sz + dz
                    if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
                        continue
                    if occupancy[x, y, z] == 0:
                        return (x, y, z)
    return None


def compute_dijkstra_potential(
    occupancy: np.ndarray,
    goal_idx: Tuple[int, int, int],
    resolution: float,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Compute shortest-path distance field from goal over free voxels."""
    nx, ny, nz = occupancy.shape
    # Use float64 during propagation to reduce precision loss on large grids.
    potential64 = np.full((nx, ny, nz), np.inf, dtype=np.float64)

    g = _find_nearest_free(occupancy, goal_idx)
    if g is None:
        return potential64.astype(np.float32), goal_idx

    pq: List[Tuple[float, Tuple[int, int, int]]] = []
    potential64[g] = 0.0
    heapq.heappush(pq, (0.0, g))

    while pq:
        cur_d, (x, y, z) = heapq.heappop(pq)
        if cur_d > float(potential64[x, y, z]) + _DIJKSTRA_EPS:
            continue
        for dx, dy, dz, cost in _NEI26:
            nx_i = x + dx
            ny_i = y + dy
            nz_i = z + dz
            if nx_i < 0 or ny_i < 0 or nz_i < 0 or nx_i >= nx or ny_i >= ny or nz_i >= nz:
                continue
            if occupancy[nx_i, ny_i, nz_i] != 0:
                continue
            nd = cur_d + float(cost) * float(resolution)
            if nd + _DIJKSTRA_EPS < float(potential64[nx_i, ny_i, nz_i]):
                potential64[nx_i, ny_i, nz_i] = nd
                heapq.heappush(pq, (nd, (nx_i, ny_i, nz_i)))

    potential64[occupancy != 0] = np.inf
    return potential64.astype(np.float32), g


def compute_descending_vector_field(potential: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """For each free voxel, pick 26-neighbor with smallest potential as descent direction."""
    nx, ny, nz = potential.shape
    vf = np.zeros((nx, ny, nz, 3), dtype=np.float32)

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if occupancy[x, y, z] != 0:
                    continue
                p0 = float(potential[x, y, z])
                if not math.isfinite(p0):
                    continue
                best = None
                best_val = p0
                for dx, dy, dz, _ in _NEI26:
                    xi = x + dx
                    yi = y + dy
                    zi = z + dz
                    if xi < 0 or yi < 0 or zi < 0 or xi >= nx or yi >= ny or zi >= nz:
                        continue
                    if occupancy[xi, yi, zi] != 0:
                        continue
                    pv = float(potential[xi, yi, zi])
                    if not math.isfinite(pv):
                        continue
                    if pv < best_val:
                        best_val = pv
                        best = (dx, dy, dz)
                if best is None:
                    continue
                d = np.asarray(best, dtype=np.float32)
                n = np.linalg.norm(d)
                if n > 1e-6:
                    vf[x, y, z] = d / n

    return vf


def query_potential_guidance(
    map_data: Dict,
    points_world: torch.Tensor,
    interpolation: str = "nearest",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Query potential value and descending direction for continuous points.
    Returns: potential, guide_dir, valid_mask
    """
    device = points_world.device
    dtype = points_world.dtype

    potential = map_data["potential"].to(device=device, dtype=dtype)
    guide_dir = map_data["guide_dir"].to(device=device, dtype=dtype)
    occupancy = map_data["occupancy"].to(device=device)
    origin = map_data["grid_origin"].to(device=device, dtype=dtype)
    resolution = float(map_data["resolution"])

    p = points_world
    flat = p.reshape(-1, 3)
    g = world_to_grid_float(flat, origin, resolution)

    nx, ny, nz = potential.shape
    x = g[:, 0]
    y = g[:, 1]
    z = g[:, 2]

    if interpolation == "trilinear":
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        z0 = torch.floor(z).long()
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        x0c = x0.clamp(0, nx - 1)
        y0c = y0.clamp(0, ny - 1)
        z0c = z0.clamp(0, nz - 1)
        x1c = x1.clamp(0, nx - 1)
        y1c = y1.clamp(0, ny - 1)
        z1c = z1.clamp(0, nz - 1)

        fx = (x - x0.float()).clamp(0.0, 1.0)
        fy = (y - y0.float()).clamp(0.0, 1.0)
        fz = (z - z0.float()).clamp(0.0, 1.0)

        def sample(ix, iy, iz):
            pot = potential[ix, iy, iz]
            d = guide_dir[ix, iy, iz]
            occ = occupancy[ix, iy, iz]
            return pot, d, occ

        corners = [
            (x0c, y0c, z0c, (1 - fx) * (1 - fy) * (1 - fz)),
            (x1c, y0c, z0c, fx * (1 - fy) * (1 - fz)),
            (x0c, y1c, z0c, (1 - fx) * fy * (1 - fz)),
            (x1c, y1c, z0c, fx * fy * (1 - fz)),
            (x0c, y0c, z1c, (1 - fx) * (1 - fy) * fz),
            (x1c, y0c, z1c, fx * (1 - fy) * fz),
            (x0c, y1c, z1c, (1 - fx) * fy * fz),
            (x1c, y1c, z1c, fx * fy * fz),
        ]

        pot_acc = torch.zeros_like(x)
        dir_acc = torch.zeros((x.shape[0], 3), device=device, dtype=dtype)
        valid_weight = torch.zeros_like(x)

        for ix, iy, iz, w in corners:
            pot_c, dir_c, occ_c = sample(ix, iy, iz)
            finite = torch.isfinite(pot_c)
            free = occ_c == 0
            good = finite & free
            w_eff = w * good.float()
            pot_acc = pot_acc + torch.where(good, pot_c, torch.zeros_like(pot_c)) * w_eff
            dir_acc = dir_acc + dir_c * w_eff.unsqueeze(-1)
            valid_weight = valid_weight + w_eff

        valid_mask = valid_weight > 1e-6
        pot_out = torch.where(valid_mask, pot_acc / valid_weight.clamp_min(1e-6), torch.full_like(pot_acc, float("inf")))
        dir_norm = dir_acc.norm(dim=-1, keepdim=True)
        dir_out = dir_acc / dir_norm.clamp_min(1e-6)
        valid_mask = valid_mask & (dir_norm.squeeze(-1) > 1e-6)
    else:
        xi = torch.round(x).long().clamp(0, nx - 1)
        yi = torch.round(y).long().clamp(0, ny - 1)
        zi = torch.round(z).long().clamp(0, nz - 1)

        pot_out = potential[xi, yi, zi]
        dir_out = guide_dir[xi, yi, zi]
        free = occupancy[xi, yi, zi] == 0
        finite = torch.isfinite(pot_out)
        nonzero_dir = dir_out.norm(dim=-1) > 1e-6
        valid_mask = free & finite & nonzero_dir

    pot_out = pot_out.reshape(p.shape[:-1])
    dir_out = dir_out.reshape(*p.shape[:-1], 3)
    valid_mask = valid_mask.reshape(p.shape[:-1])
    return pot_out, dir_out, valid_mask


class PotentialMapCache:
    """Lazy on-disk loader for precomputed potential maps."""

    def __init__(self, map_dir: str, num_maps: int = 0):
        self.map_dir = map_dir
        self.map_files = self._collect_map_files(map_dir, num_maps)
        self._cache: Dict[int, Dict] = {}

    @staticmethod
    def _collect_map_files(map_dir: str, num_maps: int) -> List[str]:
        if not os.path.isdir(map_dir):
            return []
        files = []

        type_order = {
            "hard": 0,
            "easy": 1,
            "u_min": 2,
            "hairpin": 3,
        }

        def sort_key(name: str):
            if not name.endswith(".pt"):
                return (9, name)
            stem = name[:-3]

            # Unified dataset naming: <type_prefix>_<idx>.pt
            if "_" in stem:
                prefix, idx_str = stem.rsplit("_", 1)
                if prefix in type_order and idx_str.isdigit():
                    return (0, type_order[prefix], int(idx_str), name)

            # Legacy naming: map_<idx>.pt
            if stem.startswith("map_"):
                idx_str = stem[4:]
                if idx_str.isdigit():
                    return (1, int(idx_str), name)
                return (1, 10**9, name)

            # Fallback for any other .pt maps.
            return (9, name)

        for name in sorted(os.listdir(map_dir), key=sort_key):
            if sort_key(name)[0] < 9:
                files.append(os.path.join(map_dir, name))
        if num_maps > 0:
            files = files[:num_maps]
        return files

    def __len__(self) -> int:
        return len(self.map_files)

    def get_map(self, idx: int) -> Dict:
        if len(self.map_files) == 0:
            raise IndexError("No precomputed maps available")
        ridx = int(idx) % len(self.map_files)
        if ridx not in self._cache:
            self._cache[ridx] = torch.load(self.map_files[ridx], map_location="cpu")
        return self._cache[ridx]
