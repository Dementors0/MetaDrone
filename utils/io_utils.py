"""Checkpoint and artifact I/O helpers."""

import os
import shutil

import torch


def create_unique_experiment_dir(parent_dir, base_name):
    """
    Create and return a non-conflicting experiment directory.

    The first run uses ``<parent_dir>/<base_name>``. Later runs use
    ``<base_name>_1``, ``<base_name>_2``, ... without reusing old artifacts.
    """
    os.makedirs(parent_dir, exist_ok=True)
    suffix = 0
    while True:
        dir_name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        candidate = os.path.join(parent_dir, dir_name)
        try:
            os.makedirs(candidate)
            return candidate
        except FileExistsError:
            suffix += 1


def sync_multi_pub_to_checkpoint_dir(save_dir):
    """
    Mirror current multi_pub workspace into <save_dir>/multi_pub.
    Best-effort sync for experiment reproducibility.
    """
    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dst_root = os.path.abspath(os.path.join(save_dir, "multi_pub"))

    ignore_names = {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".idea",
        ".vscode",
    }

    os.makedirs(dst_root, exist_ok=True)

    files_copied = 0
    files_deleted = 0
    dirs_deleted = 0

    for root, dirs, files in os.walk(src_root):
        rel = os.path.relpath(root, src_root)
        dst_dir = dst_root if rel == "." else os.path.join(dst_root, rel)

        dirs[:] = [d for d in dirs if d not in ignore_names]
        files = [f for f in files if f not in ignore_names]

        os.makedirs(dst_dir, exist_ok=True)

        src_entries = set(dirs) | set(files)
        try:
            dst_entries = os.listdir(dst_dir)
        except FileNotFoundError:
            dst_entries = []

        for name in dst_entries:
            if name in src_entries:
                continue
            stale_path = os.path.join(dst_dir, name)
            if os.path.isdir(stale_path):
                shutil.rmtree(stale_path)
                dirs_deleted += 1
            else:
                os.remove(stale_path)
                files_deleted += 1

        for fname in files:
            src_file = os.path.join(root, fname)
            dst_file = os.path.join(dst_dir, fname)
            if (
                (not os.path.exists(dst_file))
                or (os.path.getsize(src_file) != os.path.getsize(dst_file))
                or (int(os.path.getmtime(src_file)) != int(os.path.getmtime(dst_file)))
            ):
                shutil.copy2(src_file, dst_file)
                files_copied += 1

    return {
        "src_root": src_root,
        "dst_root": dst_root,
        "files_copied": int(files_copied),
        "files_deleted": int(files_deleted),
        "dirs_deleted": int(dirs_deleted),
    }


def load_compatible_checkpoint(
    module,
    path,
    name,
    device,
    zero_expanded=True,
    output_row_indices=None,
):
    if not path:
        return
    if not os.path.isfile(path):
        print(f"Warning: {name} path provided but file not found: {path}")
        return
    print(f"Loading {name} from {path}")
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    target_state = module.state_dict()
    compatible_state = {}
    resized_keys = []
    skipped_keys = []
    for key, value in state_dict.items():
        if key not in target_state:
            skipped_keys.append(key)
            continue
        target_value = target_state[key]
        if value.shape == target_value.shape:
            compatible_state[key] = value
            continue
        if (
            output_row_indices is not None
            and key == 'fc.weight'
            and value.dim() == 2
            and target_value.dim() == 2
            and target_value.shape[0] == len(output_row_indices)
            and value.shape[1] == target_value.shape[1]
            and max(output_row_indices, default=-1) < value.shape[0]
        ):
            compatible_state[key] = value[output_row_indices].clone()
            resized_keys.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue
        if value.dim() != target_value.dim():
            skipped_keys.append(key)
            continue
        expanded = torch.zeros_like(target_value) if zero_expanded else target_value.clone()
        slices = tuple(slice(0, min(value.shape[d], target_value.shape[d])) for d in range(value.dim()))
        expanded[slices] = value[slices]
        compatible_state[key] = expanded
        resized_keys.append((key, tuple(value.shape), tuple(target_value.shape)))

    missing_keys, unexpected_keys = module.load_state_dict(compatible_state, strict=False)
    if missing_keys:
        print(f"{name} missing_keys:", missing_keys)
    if unexpected_keys:
        print(f"{name} unexpected_keys:", unexpected_keys)
    if resized_keys:
        print(f"{name} resized_keys:", resized_keys)
    if skipped_keys:
        print(f"{name} skipped_keys:", skipped_keys)
