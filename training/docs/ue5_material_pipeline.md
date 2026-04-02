# LevelSmith — UE5 材质管线实施方案

> 状态：规划中，灰模+中尺度构件完成后启动
> 前置条件：style lock 基线冻结、中尺度构件增强完成

---

## 实施顺序

```
Day 1: M_Master_Architecture（父材质）
Day 2: 3 个 Material Instances（MI_Japanese / MI_Medieval / MI_Industrial）
Day 3: 3 个 Material Functions（MF_DirtBottom / MF_EdgeWear / MF_SurfaceVariation）
Day 4: 升级为 Material Layers（如果需要长期扩展）
```

---

## Step 1: 父材质 M_Master_Architecture

范围限定：只做建筑外表面的 4 类
- 墙面 (wall)
- 屋顶 (roof)
- Trim/边框 (trim)
- 地面接触脏污 (ground contact)

暴露的关键参数：
- Base Color / Tint
- Normal 强度
- Roughness 偏移（最关键！决定质感差异）
- AO
- Macro Variation（宏观色差）
- Dirt（脏污强度）
- Edge Wear（边缘磨损）
- Moss/Rust Mask（苔藓/锈蚀遮罩）

核心原则：风格的"质感差异"主要靠 Roughness 分布和 layered masks 拉开，不只是靠颜色。

---

## Step 2: 3 个 Material Instances

### MI_Japanese_Base
材质组合：木 + 白灰/plaster + 深色瓦/木瓦
特征：
- 木材纤维感
- 较细腻的 roughness 变化
- 檐口和柱梁的材质层次
- 墙面轻微污渍
- 整体感觉：克制、整洁、材质分层清楚

### MI_Medieval_Base
材质组合：石块 + 砂浆 + 苔藓/墙根脏污
特征：
- 更重的 roughness
- 更明显的 normal（块石拼缝）
- 裂缝、湿痕
- 地面返脏、边角磨损
- 整体感觉：厚重、老化、表面不均匀

### MI_Industrial_Base
材质组合：红砖/混凝土 + 涂漆金属 + 锈蚀/油污
特征：
- 砖或混凝土的重复模数
- 金属件的 roughness 对比
- 局部锈迹、脏污流痕
- 边缘掉漆
- 整体感觉：规则、功能化、使用痕迹强

---

## Step 3: Material Functions

### MF_DirtBottom
- 世界空间 Z 轴渐变
- 底部脏污强度可调
- 用于墙根返脏、地面接触处

### MF_EdgeWear
- 基于曲率/法线的边缘检测
- 边缘磨损强度可调
- 用于转角、窗框边缘、屋顶边缘

### MF_SurfaceVariation
- 世界空间平铺的宏观色差
- 破除大面积材质的"数字感"
- 噪声频率和强度可调

---

## Step 4: Material Layers（可选升级）

如果长期扩展需要：
- 基材层（Base Material Layer）
- 老化层（Aging Layer）
- 脏污层（Dirt Layer）

使用 Material Layers 把纹理和属性按层混合。

---

## 素材来源

### 推荐平台
- Poly Haven — CC0，PBR 纹理/HDRI/模型
- ambientCG — CC0，PBR 材质

### 9 类核心素材（每类 1 主材 + 1 辅材 + 1 老化层）

| 风格 | 主材 | 辅材 | 老化层 |
|------|------|------|--------|
| Japanese | wood | plaster | subtle_dirt |
| Medieval | stone_block | mortar | moss_cracks |
| Industrial | brick/concrete | painted_metal | rust_oil |

先每类只挑 3 个素材，跑通三套稳定母版，不做几十种变体。

---

## 光照预设（配合材质一起做）

每个 style 固定一套最小光照预设：

| 风格 | 时段 | 色温 | 阴影 | 雾 |
|------|------|------|------|-----|
| Japanese | 清晨柔光 | warm_soft | 柔和 (0.4) | 薄雾 (0.2) |
| Medieval | 午后阴天 | neutral_cool | 强硬 (0.7) | 轻雾 (0.15) |
| Industrial | 正午顶光 | cool_grey | 高对比 (0.6) | 无 (0.05) |

---

## 与 Python 端的对接

Python 端导出的 FBX/GLB 中需要携带的信息：
- 顶点色：编码 wall/roof/trim/base 的材质区域 ID
- 或 UV2：编码世界空间位置（用于 MF_DirtBottom 和 MF_SurfaceVariation）
- metadata JSON：每栋建筑的 style_key + aging_intensity + material_palette

UE5 导入后：
- 根据顶点色或 UV2 自动分配 Material Instance
- 根据 metadata 调整 Instance 参数

---

## 质感差异的核心不是颜色

三种风格的最终渲染差异应该主要来自：
1. Roughness 分布（Japanese 细腻 vs Medieval 粗糙 vs Industrial 对比强）
2. Normal 强度（Medieval 最强、石缝明显）
3. 老化 mask 分布（Medieval 苔藓+裂缝、Industrial 锈蚀+油污、Japanese 轻微自然老化）
4. Edge wear 强度
5. Macro variation 频率

颜色只是基调，质感才是风格。
