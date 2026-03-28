#!/usr/bin/env python3
import argparse
import importlib
import json
import random
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "env_multi_demo.html"
MODULE_NAME = "env_multi"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def load_env_module():
    importlib.invalidate_caches()
    if MODULE_NAME in sys.modules:
        return importlib.reload(sys.modules[MODULE_NAME])
    return importlib.import_module(MODULE_NAME)


def classify_region(y_value, region_zones):
    for zone in region_zones:
        if zone["y0"] - 1e-6 <= y_value <= zone["y1"] + 1e-6:
            return zone["type"]
    return "boundary"


def extract_u_meta(env, voxels):
    if getattr(env, "u_meta", None):
        raw = env.u_meta[0]
        if isinstance(raw, dict):
            return {
                "openLeft": raw.get("open_left"),
                "exitSide": raw.get("exit_side"),
                "exitY": raw.get("exit_y"),
                "exitSpan": raw.get("exit_span"),
                "corridorSpan": raw.get("corridor_span"),
            }

    candidates = [
        box for box in voxels
        if box["region"] == "u-minimal" and box["hy"] > 1.5 and box["hx"] < 0.3
    ]
    guessed = None
    for box in candidates:
        if box["x"] < 5.0:
            guessed = True
            break
        if box["x"] > 5.0:
            guessed = False
            break
    return {"openLeft": guessed, "exitSide": "left" if guessed else "right" if guessed is not None else "unknown"}


def build_scene(seed, obstacle_scale):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    module = load_env_module()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = module.Env(
        batch_size=1,
        width=320,
        height=240,
        grad_decay=1.0,
        device=device,
        single=True,
        random_rotation=False,
        obstacle_count_scale=obstacle_scale,
    )

    order = list(env.region_order[0])
    region_zones = []
    for idx, region_type in enumerate(order):
        y0 = float(env.map_y_min + idx * env.region_length)
        y1 = float(y0 + env.region_length)
        region_zones.append({
            "type": region_type,
            "y0": y0,
            "y1": y1,
            "centerY": 0.5 * (y0 + y1),
        })

    balls = []
    for x, y, z, r in env.balls[0].detach().cpu().tolist():
        balls.append({
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "r": float(r),
            "region": classify_region(float(y), region_zones),
        })

    cylinders = []
    for x, y, r in env.cyl[0].detach().cpu().tolist():
        cylinders.append({
            "x": float(x),
            "y": float(y),
            "r": float(r),
            "region": classify_region(float(y), region_zones),
        })

    boundary_count = len(env._build_boundary_voxels())
    voxels = []
    for idx, (x, y, z, hx, hy, hz) in enumerate(env.voxels[0].detach().cpu().tolist()):
        region = "boundary" if idx < boundary_count else classify_region(float(y), region_zones)
        voxels.append({
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "hx": float(hx),
            "hy": float(hy),
            "hz": float(hz),
            "region": region,
        })

    start_xyz = env.p[0].detach().cpu().tolist()
    goal_xyz = env.p_target[0].detach().cpu().tolist()
    start = {"x": float(start_xyz[0]), "y": float(start_xyz[1]), "z": float(start_xyz[2])}
    goal = {"x": float(goal_xyz[0]), "y": float(goal_xyz[1]), "z": float(goal_xyz[2])}

    return {
        "seed": int(seed),
        "obstacleScale": float(obstacle_scale),
        "order": order,
        "regionZones": region_zones,
        "voxels": voxels,
        "balls": balls,
        "cylinders": cylinders,
        "start": start,
        "goal": goal,
        "ySpan": abs(goal["y"] - start["y"]),
        "uMeta": extract_u_meta(env, voxels),
        "device": device,
        "config": {
            "mapXMax": float(env.map_x_max),
            "mapYHalf": float(env.map_y_half),
            "mapYMin": float(env.map_y_min),
            "mapYMax": float(env.map_y_max),
            "mapZMax": float(env.map_z_max),
            "regionLength": float(env.region_length),
            "blankLength": float(env.blank_length),
            "spawnXCenter": float(env.spawn_x_center),
            "spawnZCenter": float(env.spawn_z_center),
            "fixedSpawnHalfSpan": float(env.fixed_spawn_half_span),
            "boundaryThickness": float(env.boundary_thickness),
            "boundaryHalf": float(env.boundary_half),
            "innerWallHz": float(env.inner_wall_hz),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send_bytes(self, status_code, body, content_type):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(status_code, body, "application/json; charset=utf-8")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html", "/env_multi_demo.html"):
            self._send_bytes(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/health":
            self._send_json(200, {"ok": True})
            return
        if parsed.path == "/api/scene":
            query = parse_qs(parsed.query)
            seed_raw = query.get("seed", ["0"])[0]
            scale_raw = query.get("scale", ["1.0"])[0]
            try:
                seed = int(seed_raw)
            except ValueError:
                seed = 0
            try:
                scale = float(scale_raw)
            except ValueError:
                scale = 1.0
            try:
                payload = build_scene(seed, scale)
            except Exception as exc:
                self._send_json(500, {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                return
            self._send_json(200, payload)
            return
        self._send_json(404, {"error": f"Unknown path: {parsed.path}"})

    def log_message(self, format, *args):
        sys.stderr.write("[env_multi_viewer] " + (format % args) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Serve live env_multi.py scenes for HTML visualization.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"env_multi viewer running at http://{args.host}:{args.port}/")
    print("Edit env_multi.py, then refresh the page or click '重新生成' to see the latest scene.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping env_multi viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
