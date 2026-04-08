import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def _collect_map_files(map_dir):
    files = []
    if not os.path.isdir(map_dir):
        return files
    for name in sorted(os.listdir(map_dir)):
        if name.startswith("map_") and name.endswith(".pt"):
            files.append(os.path.join(map_dir, name))
    return files


def _pick_slice_z(z_world, origin_z, resolution, nz):
    if z_world is None:
        return max(0, min(nz - 1, nz // 2))
    zi = int(round((float(z_world) - float(origin_z)) / float(resolution)))
    return max(0, min(nz - 1, zi))


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _build_xy_figure(map_data, z_world=None, stride=5, title_prefix=""):
    potential = _to_numpy(map_data["potential"]).astype(np.float32)
    occupancy = _to_numpy(map_data["occupancy"]) > 0
    guide_dir = _to_numpy(map_data["guide_dir"]).astype(np.float32)
    origin = _to_numpy(map_data["grid_origin"]).astype(np.float32)
    resolution = float(map_data["resolution"])

    nx, ny, nz = potential.shape
    zi = _pick_slice_z(z_world, origin[2], resolution, nz)
    z_world_used = float(origin[2]) + (zi + 0.5) * resolution

    pot_xy = potential[:, :, zi].copy()
    occ_xy = occupancy[:, :, zi]
    dir_xy = guide_dir[:, :, zi, :2].copy()

    valid = np.isfinite(pot_xy) & (~occ_xy)
    if np.any(valid):
        pmin = float(np.min(pot_xy[valid]))
        pmax = float(np.max(pot_xy[valid]))
        denom = max(1e-6, pmax - pmin)
        pot_norm = (pot_xy - pmin) / denom
        pot_plot = np.ma.array(pot_norm, mask=~valid)
    else:
        pot_plot = np.ma.array(np.zeros_like(pot_xy), mask=np.ones_like(pot_xy, dtype=bool))

    x = origin[0] + (np.arange(nx, dtype=np.float32) + 0.5) * resolution
    y = origin[1] + (np.arange(ny, dtype=np.float32) + 0.5) * resolution
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(
        pot_plot.T,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized Potential")

    occ_yx = np.where(occ_xy.T)
    if occ_yx[0].size > 0:
        oy = y[occ_yx[0]]
        ox = x[occ_yx[1]]
        ax.scatter(ox, oy, s=4, c="black", alpha=0.25, label="Occupied")

    step = max(1, int(stride))
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Us = dir_xy[::step, ::step, 0]
    Vs = dir_xy[::step, ::step, 1]
    Ms = valid[::step, ::step]

    Us = np.where(Ms, Us, 0.0)
    Vs = np.where(Ms, Vs, 0.0)

    ax.quiver(
        Xs,
        Ys,
        Us,
        Vs,
        color="white",
        alpha=0.8,
        scale=35,
        width=0.0025,
        headwidth=3,
        headlength=4,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"{title_prefix} Potential Slice + Descent Field | z_idx={zi}, z={z_world_used:.2f}m"
    )
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize potential field from precomputed map_XXX.pt"
    )
    parser.add_argument(
        "--map_dir",
        type=str,
        default="/home/robot/transformer/precomputed_maps",
        help="Directory containing map_XXX.pt",
    )
    parser.add_argument(
        "--map_index",
        type=int,
        default=0,
        help="Index in sorted map_XXX.pt list",
    )
    parser.add_argument(
        "--z_world",
        type=float,
        default=None,
        help="World z (meters) for XY slice; default uses mid z",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Quiver sampling stride",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional output image path (.png). If empty, show interactive window.",
    )

    args = parser.parse_args()

    files = _collect_map_files(args.map_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No map_XXX.pt found in: {args.map_dir}")

    idx = int(args.map_index) % len(files)
    path = files[idx]
    map_data = torch.load(path, map_location="cpu")

    fig = _build_xy_figure(
        map_data,
        z_world=args.z_world,
        stride=args.stride,
        title_prefix=f"[{os.path.basename(path)}]",
    )

    if args.save:
        out_dir = os.path.dirname(args.save)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(args.save, dpi=180)
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
