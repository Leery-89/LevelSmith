# LevelSmith — 训练数据源参考

> 状态：规划中，配合 model_training_roadmap.md 使用
> 原则：分层数据栈，每层用最适合的数据源，不依赖单一数据集

---

## 数据栈总览

| 层 | 学什么 | 推荐数据源 | 优先级 |
|---|--------|-----------|--------|
| 建筑群布局 | layout family, density, road hierarchy, courtyard ratio, height variation | OSM, Overture, Google Open Buildings, Microsoft Footprints, 3DBAG, SpaceNet | 🔴 最高 |
| 立面细节 | opening grammar, facade rhythm, street edge, 构件语法 | CMP Facade DB, eTRIMS, Mapillary Vistas, ADE20K | 🟡 高 |
| 材质质感 | wall/roof/trim/dirt/roughness/aging | Poly Haven | 🟡 高 |
| 风格标签 | style supervision (japanese/medieval/industrial 分类) | TOBuilt, Historic England, NSW Heritage, 手工整理集 | 🟢 中 |

---

## 第一类：建筑群 / 道路 / Footprint / 高度

这是最该优先做的一层，解决 layout family、density、road hierarchy、courtyard ratio、height variation。

### OpenStreetMap (OSM)
- 字段：building, height, building:levels, roof:levels 等标签
- 优势：全球覆盖，标签丰富，社区活跃
- 项目已有：~60,000 条（卡尔卡松/布鲁日/约克/京都）

### Overture Maps
- 全球标准化 Buildings 主题
- 比 OSM 更规范的 schema

### Google Open Buildings v3
- 18 亿级建筑轮廓
- 2024 年 2.5D Temporal 版本：2016-2023 年度变化 + 高度信息

### Microsoft Building Footprints
- 9.99 亿级全球建筑足迹

### 3DBAG (荷兰)
- 全国范围、定期更新、多 LOD 三维建筑
- BAG 源数据带建筑用途、建造日期等属性

### SpaceNet
- 高分辨率影像 + 1100 万建筑 footprint + 道路标签
- 非常适合学"路—地块—建筑群"的关系

---

## 第二类：立面 / 门窗 / 柱廊 / 檐口 / 细部语法

### CMP Facade Database
- 606 张人工标注整正立面图
- 类别：window, door, pillar, balcony, shop, deco
- 直接可用于 opening grammar 学习

### eTRIMS
- 110 + 200 张全标注 facade 图
- 老牌数据集，学术引用广泛

### ADE20K
- 泛化场景解析
- 适合做更广义的建筑元素检测

### Mapillary Vistas
- 全球街景级别，25,000 张高分辨率标注图
- 适合补充真实街道立面分布

---

## 第三类：材质 / PBR / 质感先验

### Poly Haven
- 可直接用的 PBR 纹理、HDRI 和模型
- CC0 许可，对训练和工程整合都友好
- 建议：单独建材质库，不要把材质混在建筑标签里

---

## 第四类：风格标签 / 建筑史元数据

这一层最难——高质量"建筑风格分类大库"稀缺。

### TOBuilt (Toronto)
- 开放建筑数据库
- 包含建筑风格分类

### Historic England
- 官方开放数据
- Listed building 数据可下载

### NSW State Heritage Inventory
- 本地资源（Sydney 区域）

### 注意事项
- Kaggle 上的"25 styles dataset"等混合抓取集适合预训练/弱监督，不适合金标准
- 优先用官方/半官方遗产数据库做强标签

---

## 推荐组合方案

```
布局模型训练数据：
  OSM / Overture / Google / Microsoft / 3DBAG / SpaceNet
  → layout family, density, road hierarchy, courtyard ratio

立面模型训练数据：
  CMP / eTRIMS / Mapillary
  → opening grammar, facade rhythm, 构件语法

材质模型训练数据：
  Poly Haven
  → wall/roof/trim/dirt/roughness/aging

风格分类与评估：
  TOBuilt / Historic England / NSW Heritage + 手工筛选参考图集
  → japanese / medieval / industrial 风格监督
```

---

## 与现有数据的关系

| 现有数据 | 记录数 | 覆盖层 | 补充需求 |
|---------|-------|--------|---------|
| 合成数据 | 131,020 | 单体风格参数 | 足够，无需补充 |
| OSM 真实数据 | ~60,000 | 建筑群布局 | 可扩展更多城市 |
| W3 布局数据 | 11,903 | 建筑群布局 | 游戏特有，保留 |
| W3 Mesh 数据 | 1,856 | 几何复杂度 | 游戏特有，保留 |
| CMP / eTRIMS | 0 | 立面细节 | **需要新增** |
| Poly Haven | 0 | 材质质感 | **需要新增** |
| 风格标签 | 0 | 风格分类 | **需要新增** |
