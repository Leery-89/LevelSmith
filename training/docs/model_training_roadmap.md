# LevelSmith — 模型训练路线图

> 状态：规划中，待 v2 架构稳定后统一实施
> 依赖：archetype agent pipeline 完成、地块系统完成

---

## 核心目标

让模型从"参数匹配器"升级为"风格理解器"——不只是输出 23 个数值，而是理解建筑群的组织规律、风格继承关系和视觉语言。

---

## 三条训练路线

### 路线一：先学中间表示，再生成

文本/风格描述 → 风格群落参数 → 建筑群布局 → 单体几何与材质

中间表示（群落级别的抽象参数）：

```json
{
  "style_key": "japanese",
  "layout_family": "organic",
  "enclosure_strength": 0.45,
  "density_center": 0.72,
  "density_edge": 0.38,
  "landmark_bias": 0.55,
  "symmetry_bias": 0.30,
  "road_hierarchy": 0.40,
  "courtyard_ratio": 0.42,
  "height_variation": 0.28,
  "material_palette": ["dark_wood", "plaster", "stone_base"],
  "aging_intensity": 0.33
}
```

模型先学会"这类建筑群的抽象组织规律"，再去生成具体几何，稳定性更高。

### 路线二：两层监督

不要只监督最终输出，分两层：

**第一层：风格组织监督**
- 布局类型
- 主建筑位置
- 道路模式
- 围合类型
- 庭院比例
- 高度层次
- 主次建筑比例

**第二层：视觉风格监督**
- 屋顶语言
- 开窗语言
- 材料语言
- 老化语言
- 色彩语言

这样模型才知道"为什么这是这种风格的建筑群"，而不是只会模仿外观。

### 路线三：规则底座 + 学习偏移（推荐）

不让模型从零学所有风格，先给规则底座，模型学偏移：

```
final_style_cluster = base_style_rules + learned_offsets
```

示例 — japanese 默认底座：
- organic / courtyard-friendly
- low height variation
- strong eave language
- high material restraint
- medium enclosure

用户说"更庄严、更密、更压抑"时，模型输出偏移：
- enclosure_strength +0.2
- landmark_bias +0.15
- density_center +0.1
- contrast +0.1

与现有系统的对接：archetype agent 的 JSON plan = 规则底座，模型学习的 = 用户意图到偏移量的映射。

---

## 数据准备（三类）

### 第一类：群体布局数据

教模型"建筑群怎么组织"：
- 建筑 footprint
- 道路网络
- 围墙边界
- 开放空间
- 主次建筑类别
- 朝向
- 高度

格式：2D 平面、block graph、topology graph、JSON

### 第二类：单体风格数据

教模型"风格单体长什么样"：
- 屋顶参数
- 门窗参数
- 墙体厚度
- 材料类型
- 装饰密度

### 第三类：风格说明数据

教模型"自然语言怎么翻译成参数"：
- "日式町屋街道"
- "中世纪围合型聚落"
- "工业风规则厂区"
- "沙漠宫殿式中心广场"

每条配上对应的结构字段。

---

## 风格继承规则

模型必须学会风格在建筑层级间的继承衰减：

| 层级 | 风格表达强度 | 特征 |
|------|------------|------|
| primary | 最强 | 轮廓、材质、装饰都最完整 |
| secondary | 继承并简化 | 保留主风格核心，减少装饰 |
| tertiary | 保留核心语言 | 只保留主风格的关键特征（屋顶形式、材料） |
| ambient | 只延续材质 | 局部语言，不抢主建筑视觉焦点 |

不学这套继承逻辑会导致两个问题：
- 所有建筑太像 → 复制粘贴感
- 每栋都想表达自己 → 整体风格散掉

---

## 实施顺序

1. 先完成 v2 架构（archetype agent + 地块系统 + style director）
2. 用 archetype agent 的 JSON 输出作为路线三的"规则底座"数据
3. 从 W3 + OSM 数据中提取群体布局标注（第一类数据）
4. 训练中间表示模型（路线一 + 路线三结合）
5. 分两层监督训练最终模型（路线二）

---

## 与现有 agent 的关系

| 组件 | 当前角色 | 模型训练后的角色 |
|------|---------|----------------|
| Archetype Agent | LLM 推理生成规则底座 | 可被训练后的轻量模型替代（推理更快） |
| Distribution Agent | 代码规则驱动 | 可引入学习偏移控制密度/间距 |
| Style Director | LLM 推理生成视觉规则 | 核心训练目标：学会风格语言映射 |
| StyleParamMLP | 16→23 参数直接映射 | 升级为中间表示模型（输出群落参数） |
