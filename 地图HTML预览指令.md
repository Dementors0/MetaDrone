# 地图 HTML 预览指令

本文档给出统一四类型地图的生成与 HTML 预览命令。

## 1. 重新生成地图（统一模式）

```bash
python3 precompute_potential_maps.py \
  --unified_dataset_mode \
  --maps_per_type 10 \
  --save_dir ./generated_maps \
  --seed 42 \
  --resolution 0.3 \
  --margin 0.15 \
  --z_min 0.0 \
  --z_max 5.0
```

说明：
- 固定顺序为 `hard -> easy -> u-min -> hairpin`。
- 文件名示例：`hard_000.pt`、`easy_000.pt`、`u_min_000.pt`、`hairpin_000.pt`。

如果你希望先清空旧失败日志：

```bash
rm -f ./generated_maps/precompute_failures.log
```

## 2. 预览场景几何（不看势场，只看地图布局）

### 2.1 预览单个 map 文件

```bash
python3 preview_map.py \
  --precomputed-map ./generated_maps/hard_000.pt \
  --output-html ./generated_maps/preview_hard_000_scene.html \
  --no-open-browser
```
```bash
python3 preview_map.py \
  --precomputed-map ./generated_maps/easy_000.pt \
  --output-html ./generated_maps/preview_easy_000_scene.html \
  --no-open-browser
```
### 2.2 按目录 + 索引预览

```bash
python3 preview_map.py \
  --precomputed-map-dir ./generated_maps \
  --map-index 0 \
  --output-html ./generated_maps/preview_idx0_scene.html \
  --no-open-browser
```

### 2.3 一次预览四种地图（单条命令）

```bash
for t in hard easy u_min hairpin; do python3 preview_map.py --precomputed-map ./generated_maps/${t}_000.pt --output-html ./generated_maps/preview_${t}_000_scene.html --no-open-browser; done
```

执行后会一次生成 4 个 HTML：
- `./generated_maps/preview_hard_000_scene.html`
- `./generated_maps/preview_easy_000_scene.html`
- `./generated_maps/preview_u_min_000_scene.html`
- `./generated_maps/preview_hairpin_000_scene.html`

如果想一键打开这 4 个页面：

```bash
for t in hard easy u_min hairpin; do xdg-open ./generated_maps/preview_${t}_000_scene.html; done
```

索引范围（`maps_per_type = x` 时）：
- `hard`: `0 ~ x-1`
- `easy`: `x ~ 2x-1`
- `u_min`: `2x ~ 3x-1`
- `hairpin`: `3x ~ 4x-1`

## 3. 预览势场 + 方向场（3D）

### 3.1 保存为 HTML

```bash
python3 visualize_precomputed_potential.py \
  --map_dir ./generated_maps \
  --map_index 0 \
  --arrow_stride 8 \
  --stride 4 \
  --save ./generated_maps/preview_idx0_potential.html
```

### 3.2 指定 z 截面带宽

```bash
python3 visualize_precomputed_potential.py \
  --map_dir ./generated_maps \
  --map_index 0 \
  --z_world 2.5 \
  --z_band_layers 2 \
  --save ./generated_maps/preview_idx0_z2p5_potential.html
```

## 4. 打开 HTML 文件

```bash
xdg-open ./generated_maps/preview_hard_000_scene.html
xdg-open ./generated_maps/preview_idx0_potential.html
```

如果是远程服务器，可直接下载 HTML 到本地浏览器打开。
