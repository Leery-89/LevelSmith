# Mesh Complexity Integration -- Diagnostic Report

Generated: 2026-04-01

---

## 1. style_registry.py -- OUTPUT_PARAMS 参数定义

**Status: ✅ 23 parameters defined**

| Group | Count | Parameters |
|-------|-------|-----------|
| Structural (original) | 10 | height_range_min/max, wall_thickness, floor_thickness, door_width/height, win_width/height/density, subdivision |
| Visual (new) | 10 | roof_type, roof_pitch, wall_color_r/g/b, has_battlements, has_arch, eave_overhang, column_count, window_shape |
| Geometry (new) | 3 | mesh_complexity, detail_density, simple_ratio |

`OUTPUT_DIM = len(OUTPUT_PARAMS) = 23` (style_registry.py:65)

---

## 2. 20 种风格 mesh_complexity / detail_density / simple_ratio 覆盖情况

**Status: ✅ All 20 styles have base values + bounds for all 3 params**

All data stored in `STYLE_EXTRA_PARAMS` dict (style_registry.py:1166-1408).

### 真实 W3 数据 (7 styles)

| Style | mesh_complexity | detail_density | simple_ratio | W3 Source |
|-------|----------------|---------------|-------------|-----------|
| medieval | 0.257 | 0.637 | 0.393 | avg_faces=3624, chunks=5.8, s/m/c=228/312/40 |
| modern | 0.532 | 0.681 | 0.313 | avg_faces=7522, chunks=6.2, s/m/c=130/252/33 |
| industrial | 0.138 | 0.396 | 0.652 | avg_faces=1954, chunks=3.6, s/m/c=58/31/0 |
| horror | 0.541 | 1.000 | 0.400 | avg_faces=7635, chunks=9.1, s/m/c=30/25/20 |
| medieval_keep | 0.637 | 0.846 | 0.274 | avg_faces=9003, chunks=7.7, s/m/c=68/126/54 |
| modern_loft | 0.399 | 0.560 | 0.464 | avg_faces=5630, chunks=5.1, s/m/c=83/86/10 |
| fantasy_palace | 0.268 | 0.615 | 0.381 | avg_faces=3783, chunks=5.6, s/m/c=103/145/22 |

### 从父风格估算 (13 styles)

| Style | Derived From | mesh_complexity | detail_density | simple_ratio |
|-------|-------------|----------------|---------------|-------------|
| fantasy | fantasy_palace W3 | 0.350 | 0.615 | 0.381 |
| japanese | medieval (similar complexity) | 0.300 | 0.650 | 0.350 |
| desert | industrial (simple geometry) | 0.150 | 0.400 | 0.620 |
| medieval_chapel | medieval parent | 0.300 | 0.660 | 0.360 |
| modern_villa | modern parent | 0.480 | 0.620 | 0.340 |
| industrial_workshop | industrial parent (+complex) | 0.180 | 0.450 | 0.600 |
| industrial_powerplant | industrial (simplest) | 0.120 | 0.350 | 0.700 |
| fantasy_dungeon | fantasy parent (medium) | 0.320 | 0.700 | 0.350 |
| horror_asylum | horror parent | 0.500 | 0.900 | 0.420 |
| horror_crypt | horror parent (+dense) | 0.580 | 0.950 | 0.380 |
| japanese_temple | japanese parent (+complex) | 0.380 | 0.720 | 0.300 |
| japanese_machiya | japanese parent (simpler) | 0.220 | 0.550 | 0.420 |
| desert_palace | desert parent (+complex) | 0.280 | 0.550 | 0.450 |

---

## 3. model.py -- Output Dimension

**Status: ⚠️ StyleParamMLP=23, but build_model() default=20**

- `StyleParamMLP.__init__` default: `output_dim=23` (model.py:27) ✅
- `build_model()` default: `output_dim=20` (model.py:92) ⚠️ **stale default**
- Architecture: `16 -> 128 -> 64 -> 32 -> 23` (with BatchNorm + Dropout + Sigmoid)
- Total trainable parameters: **13,719**

**Impact**: train.py explicitly passes `output_dim=OUTPUT_DIM` (=23) so training is correct.
But any code calling `build_model()` without arguments gets 20 instead of 23.

---

## 4. generate_level.py -- mesh_complexity / detail_density / simple_ratio 控制逻辑

**Status: ✅ All 3 parameters integrated into build_room()**

Source: generate_level.py:1281-1303

### mesh_complexity (default 0.3)
```
if mesh_complexity > 0.7:
    column_count = max(column_count, 2)
    has_arch = True
```
- **Effect**: Adds decorative columns and arches to complex buildings

### detail_density (default 0.5)
```
if detail_density > 0.6 and "win_spec" in params:
    boost = (detail_density - 0.6) * 0.5  # max +0.2
    win_density = min(0.95, old_density + boost)
```
- **Effect**: Increases window density for detail-rich styles

### simple_ratio (default 0.4)
```
if simple_ratio > 0.6:
    has_battlements = False
    column_count = min(column_count, 2)
if simple_ratio > 0.75:
    roof_type = 0  # force flat roof
```
- **Effect**: Simplifies geometry for industrial/utilitarian styles

---

## 5. train.py -- 最近一次训练结果

**Status: ✅ Converged**

| Metric | Value |
|--------|-------|
| Epochs | 150 (full run, no early stopping) |
| Best val_loss | **0.001891** (epoch ~136) |
| Final val_loss | 0.001946 (epoch 150) |
| Final val_MAE | 0.03145 |
| Final train_loss | 0.005564 |
| LR schedule | 1e-3 -> 5e-4 -> 2.5e-4 -> 1.25e-4 -> 6.25e-5 -> 3.125e-5 |
| Batch size | 256 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Loss function | MSE + subdivision penalty (weight=0.1) |
| Hidden dims | [128, 64, 32] |
| Dropout | 0.2 |
| Trainable params | **13,719** |

---

## 6. 发现的不一致和潜在问题

### ⚠️ P1: build_model() 默认 output_dim=20 (应为 23)

- **位置**: model.py:92
- **现象**: `build_model()` 函数签名默认值 `output_dim=20`，与 `StyleParamMLP.__init__` 默认值 `output_dim=23` 和 `OUTPUT_DIM=23` 不一致
- **影响**: train.py 显式传参所以训练正确，但其他调用者（如 model.py 的 `__main__` 测试 L147）会创建 20 维输出模型
- **修复**: 将 model.py:92 的 `output_dim: int = 20` 改为 `output_dim: int = 23`

### ⚠️ P2: train.py 导出 JSON 缺少 mesh_complexity 三参数

- **位置**: train.py:196-216 `export_trained_params()`
- **现象**: 导出的 `params` 字典只包含原有 10 + 新增 10 = 20 个可读参数，未包含 mesh_complexity / detail_density / simple_ratio
- **影响**: `trained_style_params.json` 的 `params` 字段缺少 3 个参数的反归一化物理值（但 `normalized` 字段包含完整 23 维向量）
- **修复**: 在 `params` 字典中添加 `"mesh_complexity"`, `"detail_density"`, `"simple_ratio"` 三个键

### ⚠️ P3: mesh_complexity > 0.7 阈值不可达

- **位置**: generate_level.py:1286
- **现象**: 所有 20 种风格的 mesh_complexity base 值范围为 0.120 ~ 0.637，无一超过 0.7
- **影响**: 拱门/柱子增强逻辑在正常风格生成中永远不会触发
- **建议**: 降低阈值至 0.5，或改为渐进式缩放而非二值门控

### ⚠️ P4: get_param_vector() 文档注释不准确

- **位置**: style_registry.py:1473
- **现象**: 注释写 "新增10个来自 STYLE_EXTRA_PARAMS"，实际应为 "新增13个"（10 visual + 3 mesh）
- **影响**: 仅文档错误，不影响功能

### ✅ 无问题项

- OUTPUT_PARAMS 定义 23 个参数，OUTPUT_DIM 正确
- 所有 20 种风格均有 mesh_complexity/detail_density/simple_ratio 的 base + bounds
- 7 种风格使用真实 W3 数据，13 种从父风格合理估算
- 训练收敛良好（val_loss 0.00189，val_MAE 0.031）
- generate_level.py 的 detail_density 和 simple_ratio 阈值工作正常

---

## 总结

| Check | Status |
|-------|--------|
| OUTPUT_PARAMS = 23 个参数定义 | ✅ |
| OUTPUT_DIM = 23 | ✅ |
| 20 styles base + bounds 完整 | ✅ |
| 7 W3 真实数据源 | ✅ |
| 13 estimated styles | ✅ |
| StyleParamMLP output_dim = 23 | ✅ |
| build_model() default output_dim | ⚠️ stale default=20 |
| mesh_complexity 用于 build_room | ✅ |
| detail_density 用于 build_room | ✅ |
| simple_ratio 用于 build_room | ✅ |
| mesh_complexity 阈值合理性 | ⚠️ 0.7 不可达 |
| 训练收敛 | ✅ val_loss=0.00189 |
| 导出 JSON 完整性 | ⚠️ 缺少 3 参数 |
| 文档准确性 | ⚠️ 注释说 10 实为 13 |
