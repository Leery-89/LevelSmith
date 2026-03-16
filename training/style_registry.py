"""
LevelSmith Style Registry
定义建筑风格及其特征向量，用于结构参数预测模型的训练。

特征向量维度 (16-dim):
 0  architectural_period    建筑年代 (0=古代, 0.33=中世纪, 0.66=工业, 1=现代)
 1  wall_material_density   墙体材料密度 (石材1.0, 砖0.8, 混凝土0.7, 玻璃0.3)
 2  structural_complexity   结构复杂度 (0-1)
 3  ornament_level          装饰程度 (0=朴素, 1=华丽)
 4  ceiling_type            天花板类型 (0=平顶, 0.5=拱顶, 1=穹顶)
 5  lighting_type           照明类型 (0=火把/蜡烛, 0.5=窗户采光, 1=电气)
 6  symmetry_level          对称程度 (0-1)
 7  thermal_mass            热质量 (0=轻质, 1=厚重)
 8  window_density          窗户密度 (0=无窗, 1=全玻璃)
 9  door_formality          门的正式程度 (0=简陋, 1=宏伟)
10  floor_material          地板材料 (0=泥土, 0.25=石板, 0.5=木材, 0.75=混凝土, 1=瓷砖)
11  roof_type               屋顶类型 (0=平屋顶, 0.5=坡屋顶, 1=穹顶)
12  interior_division       内部分割 (0=开放, 0.5=混合, 1=单元式)
13  climate_adaptation      气候适应 (0=热带, 0.5=温带, 1=寒冷)
14  security_level          安全等级 (0=开放, 1=要塞)
15  vertical_emphasis       垂直强调 (0=横向延伸, 1=高耸)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

FEATURE_DIM = 16
FEATURE_NAMES = [
    "architectural_period", "wall_material_density", "structural_complexity",
    "ornament_level", "ceiling_type", "lighting_type", "symmetry_level",
    "thermal_mass", "window_density", "door_formality", "floor_material",
    "roof_type", "interior_division", "climate_adaptation", "security_level",
    "vertical_emphasis",
]

# 输出参数说明
OUTPUT_PARAMS = {
    "height_range_min":    {"unit": "m", "range": (2.0,  6.0)},
    "height_range_max":    {"unit": "m", "range": (3.0, 20.0)},
    "wall_thickness":      {"unit": "m", "range": (0.1,  1.5)},
    "floor_thickness":     {"unit": "m", "range": (0.1,  0.6)},
    "door_width":          {"unit": "m", "range": (0.6,  3.0)},
    "door_height":         {"unit": "m", "range": (1.8,  5.0)},
    "win_width":           {"unit": "m", "range": (0.3,  3.0)},
    "win_height":          {"unit": "m", "range": (0.4,  3.0)},
    "win_density":         {"unit": "",  "range": (0.0,  1.0)},
    "subdivision":         {"unit": "",  "range": (1,    8)},
}
OUTPUT_DIM = len(OUTPUT_PARAMS)
OUTPUT_KEYS = list(OUTPUT_PARAMS.keys())


@dataclass
class StyleDefinition:
    name: str
    description: str
    feature_vector: List[float]        # 16维特征向量 (归一化 0-1)
    base_params: Dict[str, float]      # 基准输出参数 (真实物理值)
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # 风格级物理值上下界
    param_noise_scale: Dict[str, float] = field(default_factory=dict)  # 扩增噪声幅度


def _default_noise() -> Dict[str, float]:
    return {
        "height_range_min": 0.3,
        "height_range_max": 1.0,
        "wall_thickness":   0.05,
        "floor_thickness":  0.02,
        "door_width":       0.1,
        "door_height":      0.2,
        "win_width":        0.15,
        "win_height":       0.15,
        "win_density":      0.08,
        "subdivision":      1.0,
    }


# ─────────────────────────────────────────────
#  风格 1: 中世纪 (Medieval)
#  厚重石墙、低矮天花、小窗、重型木门、强安全感
# ─────────────────────────────────────────────
MEDIEVAL = StyleDefinition(
    name="medieval",
    description="中世纪城堡/教堂风格：厚重石墙，拱形天花，小窗，宏伟木门",
    feature_vector=[
        0.33,   # architectural_period
        1.00,   # wall_material_density (石材)
        0.70,   # structural_complexity
        0.60,   # ornament_level
        0.50,   # ceiling_type (拱顶)
        0.20,   # lighting_type (窗户少)
        0.75,   # symmetry_level
        1.00,   # thermal_mass (厚重)
        0.15,   # window_density (窗少)
        0.80,   # door_formality (宏伟)
        0.25,   # floor_material (石板)
        0.50,   # roof_type (坡屋顶)
        0.80,   # interior_division (单元式)
        0.50,   # climate_adaptation (温带)
        0.90,   # security_level (高安全)
        0.60,   # vertical_emphasis
    ],
    base_params={
        "height_range_min": 3.5,
        "height_range_max": 8.0,
        "wall_thickness":   0.80,
        "floor_thickness":  0.35,
        "door_width":       1.20,
        "door_height":      2.80,
        "win_width":        0.60,
        "win_height":       1.20,
        "win_density":      0.20,
        "subdivision":      4.0,
    },
    param_bounds={
        # 中世纪石砌建筑的合理物理范围（比全局 OUTPUT_PARAMS 更紧）
        "height_range_min": (2.5,  5.0),   # 厅室最低天花不低于2.5m，一般不超5m
        "height_range_max": (5.0, 11.0),   # 城堡大厅通常5-11m，超过11m为教堂特例
        "wall_thickness":   (0.50, 1.20),  # 石墙厚度范围
        "floor_thickness":  (0.20, 0.50),
        "door_width":       (0.90, 1.60),
        "door_height":      (2.20, 3.60),
        "win_width":        (0.30, 0.90),  # 中世纪窗小，不超0.9m
        "win_height":       (0.60, 1.60),
        "win_density":      (0.05, 0.35),  # 窗少，密度低
        "subdivision":      (2.0,  6.0),
    },
    param_noise_scale={**_default_noise(), "wall_thickness": 0.10, "subdivision": 1.5},
)

# ─────────────────────────────────────────────
#  风格 2: 现代 (Modern)
#  轻薄墙体、高天花、大窗、简约门、开放空间
# ─────────────────────────────────────────────
MODERN = StyleDefinition(
    name="modern",
    description="现代简约风格：薄混凝土/玻璃墙，高天花，大面积玻璃窗，简约门",
    feature_vector=[
        1.00,   # architectural_period
        0.50,   # wall_material_density (混凝土+玻璃)
        0.40,   # structural_complexity
        0.20,   # ornament_level (极简)
        0.00,   # ceiling_type (平顶)
        1.00,   # lighting_type (电气+自然光)
        0.90,   # symmetry_level
        0.40,   # thermal_mass
        0.75,   # window_density (大窗)
        0.40,   # door_formality (简约)
        1.00,   # floor_material (瓷砖/混凝土)
        0.00,   # roof_type (平屋顶)
        0.20,   # interior_division (开放)
        0.50,   # climate_adaptation
        0.20,   # security_level
        0.70,   # vertical_emphasis
    ],
    base_params={
        "height_range_min": 2.8,
        "height_range_max": 4.5,
        "wall_thickness":   0.20,
        "floor_thickness":  0.20,
        "door_width":       1.00,
        "door_height":      2.20,
        "win_width":        1.80,
        "win_height":       2.00,
        "win_density":      0.70,
        "subdivision":      2.0,
    },
    param_bounds={
        "height_range_min": (2.4,  3.5),   # 现代住宅/办公层高范围
        "height_range_max": (3.5,  6.0),   # 现代建筑净高一般不超6m
        "wall_thickness":   (0.12, 0.35),  # 现代薄墙
        "floor_thickness":  (0.15, 0.28),
        "door_width":       (0.75, 1.20),
        "door_height":      (2.00, 2.50),
        "win_width":        (1.20, 2.80),  # 现代大窗
        "win_height":       (1.40, 2.60),
        "win_density":      (0.50, 0.90),  # 窗多，密度高
        "subdivision":      (1.0,  3.0),
    },
    param_noise_scale={**_default_noise(), "win_density": 0.12},
)

# ─────────────────────────────────────────────
#  风格 3: 工业 (Industrial)
#  裸露金属/混凝土、超高天花、工业窗、重型钢门
# ─────────────────────────────────────────────
INDUSTRIAL = StyleDefinition(
    name="industrial",
    description="工业仓库/工厂风格：裸露钢铁与混凝土，超高空间，工业格窗，重型钢门",
    feature_vector=[
        0.66,   # architectural_period
        0.70,   # wall_material_density (混凝土+钢)
        0.55,   # structural_complexity
        0.05,   # ornament_level (无装饰)
        0.00,   # ceiling_type (平顶/桁架)
        0.70,   # lighting_type (工业照明)
        0.50,   # symmetry_level
        0.75,   # thermal_mass
        0.40,   # window_density (工业格窗)
        0.50,   # door_formality (功能性)
        0.75,   # floor_material (混凝土)
        0.00,   # roof_type (平屋顶/锯齿形)
        0.60,   # interior_division
        0.50,   # climate_adaptation
        0.60,   # security_level
        0.80,   # vertical_emphasis (高层)
    ],
    base_params={
        "height_range_min": 4.0,
        "height_range_max": 12.0,
        "wall_thickness":   0.40,
        "floor_thickness":  0.25,
        "door_width":       2.00,
        "door_height":      4.00,
        "win_width":        1.00,
        "win_height":       0.80,
        "win_density":      0.35,
        "subdivision":      3.0,
    },
    param_bounds={
        "height_range_min": (3.0,  5.5),   # 工业层高起点通常3-5.5m
        "height_range_max": (7.0, 17.0),   # 仓库/工厂净高7-17m
        "wall_thickness":   (0.25, 0.65),  # 工业混凝土/钢墙
        "floor_thickness":  (0.18, 0.38),
        "door_width":       (1.50, 2.60),  # 工业大门
        "door_height":      (3.00, 5.00),
        "win_width":        (0.70, 1.40),  # 工业格窗
        "win_height":       (0.50, 1.20),
        "win_density":      (0.20, 0.55),
        "subdivision":      (2.0,  5.0),
    },
    param_noise_scale={**_default_noise(), "height_range_max": 2.0, "door_height": 0.5},
)

# ─────────────────────────────────────────────
#  风格 4: 奇幻 (Fantasy)
#  魔法城堡/RPG 地牢：高耸尖塔，华丽装饰，哥特拱窗，宏大门洞
# ─────────────────────────────────────────────
FANTASY = StyleDefinition(
    name="fantasy",
    description="奇幻魔法城堡/RPG风格：高耸尖塔，极致装饰，哥特拱窗，宏伟门洞",
    feature_vector=[
        0.28,   # architectural_period (介于古代与中世纪)
        0.80,   # wall_material_density (魔法石材/水晶)
        0.88,   # structural_complexity (极复杂)
        0.92,   # ornament_level (极度华丽)
        0.70,   # ceiling_type (拱顶趋向穹顶)
        0.30,   # lighting_type (火把+魔法光)
        0.85,   # symmetry_level
        0.65,   # thermal_mass
        0.45,   # window_density (高拱窗)
        0.88,   # door_formality (宏伟门洞)
        0.30,   # floor_material (魔法石板)
        0.75,   # roof_type (尖塔屋顶)
        0.55,   # interior_division
        0.45,   # climate_adaptation
        0.65,   # security_level
        0.88,   # vertical_emphasis (尖塔高耸)
    ],
    base_params={
        "height_range_min": 4.0,
        "height_range_max": 10.0,
        "wall_thickness":   0.60,
        "floor_thickness":  0.30,
        "door_width":       1.50,
        "door_height":      3.50,
        "win_width":        1.00,
        "win_height":       2.00,
        "win_density":      0.45,
        "subdivision":      3.0,
    },
    param_bounds={
        "height_range_min": (3.0,  6.0),
        "height_range_max": (7.0, 14.0),   # 奇幻建筑允许夸张高度
        "wall_thickness":   (0.40, 0.90),
        "floor_thickness":  (0.20, 0.45),
        "door_width":       (1.10, 2.00),
        "door_height":      (2.80, 4.50),  # 超高门洞
        "win_width":        (0.70, 1.40),
        "win_height":       (1.50, 2.80),  # 细长哥特窗
        "win_density":      (0.30, 0.65),
        "subdivision":      (2.0,  5.0),
    },
    param_noise_scale={**_default_noise(), "door_height": 0.4, "win_height": 0.3},
)

# ─────────────────────────────────────────────
#  风格 5: 恐怖 (Horror)
#  哥特废墟/维多利亚鬼屋：压抑低矮空间，极少窗户，迷宫般分割
# ─────────────────────────────────────────────
HORROR = StyleDefinition(
    name="horror",
    description="哥特废墟/维多利亚鬼屋风格：压抑低矮，极暗，迷宫分割，腐朽墙体",
    feature_vector=[
        0.35,   # architectural_period (维多利亚/哥特)
        0.88,   # wall_material_density (厚重石砖)
        0.60,   # structural_complexity
        0.30,   # ornament_level (腐朽装饰)
        0.35,   # ceiling_type (低矮拱顶)
        0.05,   # lighting_type (极暗，仅蜡烛)
        0.30,   # symmetry_level (不规则)
        0.92,   # thermal_mass (极厚重冰冷)
        0.10,   # window_density (极少窗)
        0.42,   # door_formality (沉重破旧)
        0.22,   # floor_material (腐木/石板)
        0.40,   # roof_type
        0.88,   # interior_division (极多小房间)
        0.85,   # climate_adaptation (阴冷)
        0.80,   # security_level (厚重锁门)
        0.25,   # vertical_emphasis (低压迫感)
    ],
    base_params={
        "height_range_min": 2.5,
        "height_range_max": 5.0,
        "wall_thickness":   0.65,
        "floor_thickness":  0.30,
        "door_width":       0.80,
        "door_height":      2.30,
        "win_width":        0.40,
        "win_height":       0.60,
        "win_density":      0.12,
        "subdivision":      5.0,
    },
    param_bounds={
        "height_range_min": (2.2,  3.5),
        "height_range_max": (3.5,  7.0),
        "wall_thickness":   (0.45, 0.95),
        "floor_thickness":  (0.20, 0.48),
        "door_width":       (0.65, 1.00),  # 窄门，压迫感
        "door_height":      (1.90, 2.80),
        "win_width":        (0.25, 0.60),  # 极小窗
        "win_height":       (0.30, 0.80),
        "win_density":      (0.05, 0.22),  # 极少窗
        "subdivision":      (3.0,  7.0),   # 迷宫式多分割
    },
    param_noise_scale={**_default_noise(), "subdivision": 1.5, "win_density": 0.04},
)

# ─────────────────────────────────────────────
#  风格 6: 日式 (Japanese)
#  传统和风建筑：低矮水平，轻薄木构，障子格窗，推拉门
# ─────────────────────────────────────────────
JAPANESE = StyleDefinition(
    name="japanese",
    description="传统和风建筑：低矮水平，轻薄木构/障子，自然采光，推拉门",
    feature_vector=[
        0.42,   # architectural_period (传统，介于中世纪与工业)
        0.35,   # wall_material_density (轻薄木材/纸)
        0.55,   # structural_complexity
        0.52,   # ornament_level (素雅精致)
        0.10,   # ceiling_type (低平顶，微斜)
        0.48,   # lighting_type (障子自然漫射光)
        0.88,   # symmetry_level (高度对称)
        0.22,   # thermal_mass (轻木构)
        0.68,   # window_density (大量障子窗)
        0.52,   # door_formality (典雅推拉门)
        0.48,   # floor_material (木材/榻榻米)
        0.55,   # roof_type (坡屋顶，起翘)
        0.62,   # interior_division (模块化榻榻米间)
        0.62,   # climate_adaptation (温带湿润)
        0.22,   # security_level (开放)
        0.12,   # vertical_emphasis (强调水平延伸)
    ],
    base_params={
        "height_range_min": 2.2,
        "height_range_max": 3.2,
        "wall_thickness":   0.15,
        "floor_thickness":  0.25,
        "door_width":       0.90,
        "door_height":      2.00,
        "win_width":        0.80,
        "win_height":       1.20,
        "win_density":      0.65,
        "subdivision":      3.0,
    },
    param_bounds={
        "height_range_min": (2.0,  2.8),   # 传统日式低矮
        "height_range_max": (2.8,  4.0),   # 净高严格控制
        "wall_thickness":   (0.10, 0.22),  # 极薄木/纸壁
        "floor_thickness":  (0.18, 0.35),
        "door_width":       (0.70, 1.10),  # 推拉门
        "door_height":      (1.80, 2.20),
        "win_width":        (0.55, 1.10),  # 障子格窗
        "win_height":       (0.80, 1.50),
        "win_density":      (0.50, 0.85),  # 大量采光窗
        "subdivision":      (2.0,  5.0),
    },
    param_noise_scale={**_default_noise(), "wall_thickness": 0.02, "win_density": 0.08},
)

# ─────────────────────────────────────────────
#  风格 7: 沙漠 (Desert)
#  中东土坯/沙漠堡垒：极厚隔热土墙，极少小窗，平顶
# ─────────────────────────────────────────────
DESERT = StyleDefinition(
    name="desert",
    description="中东土坯/沙漠堡垒风格：极厚隔热土墙，极少小窗，平屋顶，热带遮阳",
    feature_vector=[
        0.15,   # architectural_period (古代)
        0.88,   # wall_material_density (土坯/泥砖)
        0.28,   # structural_complexity (简单)
        0.28,   # ornament_level (几何纹样)
        0.02,   # ceiling_type (绝对平顶)
        0.18,   # lighting_type (极少小窗)
        0.72,   # symmetry_level
        0.98,   # thermal_mass (最大热质量，隔热)
        0.08,   # window_density (极少窗)
        0.32,   # door_formality
        0.12,   # floor_material (夯土/石)
        0.05,   # roof_type (平屋顶)
        0.48,   # interior_division
        0.02,   # climate_adaptation (炎热沙漠)
        0.55,   # security_level
        0.12,   # vertical_emphasis (水平低矮)
    ],
    base_params={
        "height_range_min": 2.8,
        "height_range_max": 4.0,
        "wall_thickness":   0.70,
        "floor_thickness":  0.18,
        "door_width":       0.85,
        "door_height":      2.10,
        "win_width":        0.35,
        "win_height":       0.50,
        "win_density":      0.15,
        "subdivision":      2.0,
    },
    param_bounds={
        "height_range_min": (2.4,  3.5),
        "height_range_max": (3.5,  5.5),
        "wall_thickness":   (0.50, 1.10),  # 极厚隔热墙
        "floor_thickness":  (0.12, 0.28),
        "door_width":       (0.65, 1.05),  # 窄门减少热交换
        "door_height":      (1.90, 2.50),
        "win_width":        (0.22, 0.55),  # 极小遮阳窗
        "win_height":       (0.28, 0.70),
        "win_density":      (0.05, 0.28),  # 极稀疏
        "subdivision":      (1.0,  4.0),
    },
    param_noise_scale={**_default_noise(), "wall_thickness": 0.12, "win_density": 0.04},
)

# ─────────────────────────────────────────────
#  风格 8: 中世纪礼拜堂 (Medieval Chapel)
#  小型单间礼拜堂：尖拱细窗，低矮亲切，石砌温馨
# ─────────────────────────────────────────────
MEDIEVAL_CHAPEL = StyleDefinition(
    name="medieval_chapel",
    description="中世纪小礼拜堂：尖拱细高窗，低矮亲切空间，石砌温馨氛围",
    feature_vector=[
        0.33,   # architectural_period
        0.95,   # wall_material_density (厚重石材)
        0.50,   # structural_complexity
        0.70,   # ornament_level (宗教装饰)
        0.60,   # ceiling_type (尖拱顶)
        0.25,   # lighting_type (彩窗弱光)
        0.90,   # symmetry_level (高度对称)
        0.90,   # thermal_mass (厚重)
        0.25,   # window_density (少量尖拱窗)
        0.65,   # door_formality (庄重入口)
        0.25,   # floor_material (石板)
        0.65,   # roof_type (尖坡屋顶)
        0.60,   # interior_division (单间为主)
        0.50,   # climate_adaptation
        0.60,   # security_level
        0.70,   # vertical_emphasis (纵向感强)
    ],
    base_params={
        "height_range_min": 3.0,
        "height_range_max": 6.0,
        "wall_thickness":   0.60,
        "floor_thickness":  0.30,
        "door_width":       1.00,
        "door_height":      2.50,
        "win_width":        0.40,
        "win_height":       1.80,
        "win_density":      0.25,
        "subdivision":      3.0,
    },
    param_bounds={
        "height_range_min": (2.5,  4.0),
        "height_range_max": (4.5,  8.0),
        "wall_thickness":   (0.40, 0.80),
        "floor_thickness":  (0.20, 0.42),
        "door_width":       (0.85, 1.20),
        "door_height":      (2.20, 3.00),
        "win_width":        (0.30, 0.55),
        "win_height":       (1.20, 2.40),  # 细高尖拱窗
        "win_density":      (0.15, 0.38),
        "subdivision":      (2.0,  5.0),
    },
    param_noise_scale={**_default_noise(), "win_height": 0.25},
)

# ─────────────────────────────────────────────
#  风格 9: 中世纪防御塔 (Medieval Keep)
#  要塞主塔：极厚石墙，箭孔微窗，超高多层，最大安全等级
# ─────────────────────────────────────────────
MEDIEVAL_KEEP = StyleDefinition(
    name="medieval_keep",
    description="中世纪防御主塔：极厚石墙，箭孔微窗，超高多层，要塞级防御",
    feature_vector=[
        0.33,   # architectural_period
        1.00,   # wall_material_density (最厚重石材)
        0.60,   # structural_complexity
        0.20,   # ornament_level (几乎无装饰)
        0.40,   # ceiling_type (低拱顶)
        0.10,   # lighting_type (极暗)
        0.70,   # symmetry_level
        1.00,   # thermal_mass (最厚重)
        0.08,   # window_density (箭孔)
        0.50,   # door_formality (厚重功能性)
        0.25,   # floor_material (石板)
        0.60,   # roof_type
        0.90,   # interior_division (多层分割)
        0.60,   # climate_adaptation
        1.00,   # security_level (最高安全)
        0.95,   # vertical_emphasis (高耸)
    ],
    base_params={
        "height_range_min": 5.0,
        "height_range_max": 15.0,
        "wall_thickness":   1.20,
        "floor_thickness":  0.45,
        "door_width":       0.90,
        "door_height":      2.20,
        "win_width":        0.30,
        "win_height":       0.60,
        "win_density":      0.08,
        "subdivision":      6.0,
    },
    param_bounds={
        "height_range_min": (4.0,  6.0),
        "height_range_max": (10.0, 20.0),
        "wall_thickness":   (0.90, 1.50),  # 极厚防御石墙
        "floor_thickness":  (0.30, 0.55),
        "door_width":       (0.70, 1.10),  # 窄防御门
        "door_height":      (1.90, 2.60),
        "win_width":        (0.30, 0.45),  # 箭孔
        "win_height":       (0.40, 0.90),
        "win_density":      (0.03, 0.15),
        "subdivision":      (4.0,  8.0),
    },
    param_noise_scale={**_default_noise(), "height_range_max": 2.0, "wall_thickness": 0.12},
)

# ─────────────────────────────────────────────
#  风格 10: 工业Loft公寓 (Modern Loft)
#  高挑裸露混凝土，开放大空间，工业大窗
# ─────────────────────────────────────────────
MODERN_LOFT = StyleDefinition(
    name="modern_loft",
    description="现代工业Loft公寓：高挑裸露混凝土，开放大空间，工业大窗",
    feature_vector=[
        0.85,   # architectural_period
        0.65,   # wall_material_density (混凝土)
        0.35,   # structural_complexity
        0.15,   # ornament_level (极简工业)
        0.00,   # ceiling_type (平顶/裸露管道)
        0.90,   # lighting_type
        0.60,   # symmetry_level
        0.55,   # thermal_mass
        0.65,   # window_density (大工业窗)
        0.50,   # door_formality
        0.75,   # floor_material (混凝土/木)
        0.00,   # roof_type (平屋顶)
        0.10,   # interior_division (全开放)
        0.50,   # climate_adaptation
        0.25,   # security_level
        0.80,   # vertical_emphasis (高挑)
    ],
    base_params={
        "height_range_min": 3.5,
        "height_range_max": 6.0,
        "wall_thickness":   0.25,
        "floor_thickness":  0.22,
        "door_width":       1.20,
        "door_height":      2.50,
        "win_width":        2.00,
        "win_height":       2.20,
        "win_density":      0.65,
        "subdivision":      1.0,
    },
    param_bounds={
        "height_range_min": (3.0,  4.5),
        "height_range_max": (5.0,  8.0),
        "wall_thickness":   (0.18, 0.35),
        "floor_thickness":  (0.18, 0.28),
        "door_width":       (0.95, 1.50),
        "door_height":      (2.20, 3.00),
        "win_width":        (1.50, 2.80),  # 大工业窗
        "win_height":       (1.80, 2.60),
        "win_density":      (0.50, 0.80),
        "subdivision":      (1.0,  2.0),
    },
    param_noise_scale={**_default_noise(), "win_density": 0.10, "win_width": 0.20},
)

# ─────────────────────────────────────────────
#  风格 11: 现代豪华别墅 (Modern Villa)
#  落地玻璃幕墙，极薄围护，奢华开敞
# ─────────────────────────────────────────────
MODERN_VILLA = StyleDefinition(
    name="modern_villa",
    description="现代豪华别墅：落地玻璃幕墙，极薄围护，奢华开敞，高端材料",
    feature_vector=[
        1.00,   # architectural_period
        0.35,   # wall_material_density (玻璃+薄混凝土)
        0.55,   # structural_complexity
        0.60,   # ornament_level (精致高端)
        0.00,   # ceiling_type (平顶)
        1.00,   # lighting_type
        0.80,   # symmetry_level
        0.30,   # thermal_mass (轻质)
        0.85,   # window_density (近全玻璃)
        0.70,   # door_formality (精致入口)
        1.00,   # floor_material (高端瓷砖/石材)
        0.00,   # roof_type (平屋顶)
        0.25,   # interior_division (开放式)
        0.40,   # climate_adaptation
        0.30,   # security_level
        0.65,   # vertical_emphasis
    ],
    base_params={
        "height_range_min": 3.0,
        "height_range_max": 5.0,
        "wall_thickness":   0.18,
        "floor_thickness":  0.25,
        "door_width":       1.20,
        "door_height":      2.40,
        "win_width":        2.50,
        "win_height":       2.80,
        "win_density":      0.85,
        "subdivision":      2.0,
    },
    param_bounds={
        "height_range_min": (2.7,  3.8),
        "height_range_max": (4.0,  6.5),
        "wall_thickness":   (0.12, 0.25),  # 极薄围护
        "floor_thickness":  (0.18, 0.32),
        "door_width":       (0.95, 1.50),
        "door_height":      (2.10, 2.80),
        "win_width":        (1.80, 3.00),  # 近落地大窗
        "win_height":       (2.20, 3.00),
        "win_density":      (0.70, 0.95),  # 高密度玻璃
        "subdivision":      (1.0,  3.0),
    },
    param_noise_scale={**_default_noise(), "win_width": 0.25, "win_density": 0.06},
)

# ─────────────────────────────────────────────
#  风格 12: 工业机械工坊 (Industrial Workshop)
#  中型砖墙工坊，格窗，多功能分区
# ─────────────────────────────────────────────
INDUSTRIAL_WORKSHOP = StyleDefinition(
    name="industrial_workshop",
    description="工业小型机械工坊：砖墙格窗，中等层高，多功能工作区",
    feature_vector=[
        0.66,   # architectural_period
        0.80,   # wall_material_density (砖墙)
        0.45,   # structural_complexity
        0.05,   # ornament_level (无装饰)
        0.00,   # ceiling_type (平顶/桁架)
        0.60,   # lighting_type
        0.60,   # symmetry_level
        0.70,   # thermal_mass
        0.30,   # window_density (工业格窗)
        0.55,   # door_formality (功能性)
        0.75,   # floor_material (混凝土)
        0.10,   # roof_type
        0.55,   # interior_division (多工区)
        0.50,   # climate_adaptation
        0.50,   # security_level
        0.55,   # vertical_emphasis
    ],
    base_params={
        "height_range_min": 3.5,
        "height_range_max": 7.0,
        "wall_thickness":   0.45,
        "floor_thickness":  0.20,
        "door_width":       1.80,
        "door_height":      3.00,
        "win_width":        0.80,
        "win_height":       1.00,
        "win_density":      0.30,
        "subdivision":      4.0,
    },
    param_bounds={
        "height_range_min": (3.0,  4.5),
        "height_range_max": (5.5,  9.0),
        "wall_thickness":   (0.30, 0.60),
        "floor_thickness":  (0.15, 0.28),
        "door_width":       (1.40, 2.30),
        "door_height":      (2.50, 4.00),
        "win_width":        (0.60, 1.10),
        "win_height":       (0.70, 1.40),
        "win_density":      (0.18, 0.45),
        "subdivision":      (2.0,  6.0),
    },
    param_noise_scale={**_default_noise()},
)

# ─────────────────────────────────────────────
#  风格 13: 重工业发电厂 (Industrial Powerplant)
#  超高净空发电站/重工厂房，极少窗，设备级大门
# ─────────────────────────────────────────────
INDUSTRIAL_POWERPLANT = StyleDefinition(
    name="industrial_powerplant",
    description="重工业发电站/厂房：超高净空，极厚混凝土，极少高位窗，设备级大门",
    feature_vector=[
        0.75,   # architectural_period
        0.90,   # wall_material_density (重型混凝土)
        0.70,   # structural_complexity
        0.02,   # ornament_level (无装饰)
        0.00,   # ceiling_type (平顶/桁架)
        0.55,   # lighting_type (工业照明)
        0.65,   # symmetry_level
        0.90,   # thermal_mass (极厚重)
        0.15,   # window_density (极少高位窗)
        0.60,   # door_formality (重型功能门)
        0.75,   # floor_material (重型混凝土)
        0.05,   # roof_type (平屋顶)
        0.35,   # interior_division
        0.50,   # climate_adaptation
        0.70,   # security_level
        0.95,   # vertical_emphasis (超高)
    ],
    base_params={
        "height_range_min": 6.0,
        "height_range_max": 20.0,
        "wall_thickness":   0.60,
        "floor_thickness":  0.35,
        "door_width":       2.50,
        "door_height":      5.00,
        "win_width":        0.70,
        "win_height":       0.60,
        "win_density":      0.15,
        "subdivision":      2.0,
    },
    param_bounds={
        "height_range_min": (5.0,  6.0),
        "height_range_max": (15.0, 20.0),  # 超高净空
        "wall_thickness":   (0.45, 0.80),
        "floor_thickness":  (0.25, 0.45),
        "door_width":       (2.00, 3.00),  # 设备级大门
        "door_height":      (4.00, 5.00),
        "win_width":        (0.50, 1.00),
        "win_height":       (0.40, 0.90),
        "win_density":      (0.08, 0.25),
        "subdivision":      (1.0,  4.0),
    },
    param_noise_scale={**_default_noise(), "height_range_max": 2.5, "door_height": 0.4},
)

# ─────────────────────────────────────────────
#  风格 14: 奇幻地下城 (Fantasy Dungeon)
#  黑暗迷宫地牢：极低压抑，极厚石墙，近乎无窗，最大迷宫分割
# ─────────────────────────────────────────────
FANTASY_DUNGEON = StyleDefinition(
    name="fantasy_dungeon",
    description="奇幻地下城/地牢：极低压抑，极厚石墙，近乎无窗，最大迷宫分割",
    feature_vector=[
        0.20,   # architectural_period (古代)
        1.00,   # wall_material_density (最厚重石材)
        0.65,   # structural_complexity
        0.30,   # ornament_level (阴暗纹饰)
        0.30,   # ceiling_type (低拱顶)
        0.05,   # lighting_type (极暗/火把)
        0.20,   # symmetry_level (不规则迷宫)
        1.00,   # thermal_mass (极厚重)
        0.05,   # window_density (几乎无窗)
        0.25,   # door_formality (沉重铁门)
        0.10,   # floor_material (泥土/石)
        0.15,   # roof_type
        1.00,   # interior_division (最大迷宫)
        0.70,   # climate_adaptation (阴冷)
        0.95,   # security_level (极高安全)
        0.10,   # vertical_emphasis (低矮压迫)
    ],
    base_params={
        "height_range_min": 2.2,
        "height_range_max": 4.0,
        "wall_thickness":   0.90,
        "floor_thickness":  0.35,
        "door_width":       0.75,
        "door_height":      2.00,
        "win_width":        0.30,
        "win_height":       0.40,
        "win_density":      0.05,
        "subdivision":      8.0,
    },
    param_bounds={
        "height_range_min": (2.0,  3.0),
        "height_range_max": (3.0,  5.5),
        "wall_thickness":   (0.65, 1.20),
        "floor_thickness":  (0.22, 0.48),
        "door_width":       (0.60, 0.95),  # 窄暗门
        "door_height":      (1.80, 2.40),
        "win_width":        (0.30, 0.45),  # 通风缝
        "win_height":       (0.40, 0.65),
        "win_density":      (0.02, 0.12),
        "subdivision":      (5.0,  8.0),   # 极高迷宫度
    },
    param_noise_scale={**_default_noise(), "subdivision": 1.0, "win_density": 0.03},
)

# ─────────────────────────────────────────────
#  风格 15: 奇幻魔法宫殿 (Fantasy Palace)
#  宏伟魔法宫殿大厅：极高穹顶，极华丽，巨型门洞，魔法彩窗
# ─────────────────────────────────────────────
FANTASY_PALACE = StyleDefinition(
    name="fantasy_palace",
    description="奇幻魔法宫殿大厅：极高穹顶，极致华丽，巨型门洞，魔法彩窗",
    feature_vector=[
        0.25,   # architectural_period
        0.85,   # wall_material_density (魔法石材)
        0.95,   # structural_complexity (极复杂)
        1.00,   # ornament_level (极致华丽)
        0.90,   # ceiling_type (穹顶)
        0.45,   # lighting_type (魔法光+彩窗)
        0.95,   # symmetry_level (极度对称)
        0.70,   # thermal_mass
        0.60,   # window_density (大彩窗)
        1.00,   # door_formality (最宏伟门洞)
        0.30,   # floor_material (魔法石板)
        0.90,   # roof_type (穹顶)
        0.20,   # interior_division (开阔大厅)
        0.45,   # climate_adaptation
        0.70,   # security_level
        0.95,   # vertical_emphasis (极高耸)
    ],
    base_params={
        "height_range_min": 6.0,
        "height_range_max": 18.0,
        "wall_thickness":   0.70,
        "floor_thickness":  0.35,
        "door_width":       2.50,
        "door_height":      4.50,
        "win_width":        1.50,
        "win_height":       3.00,
        "win_density":      0.60,
        "subdivision":      2.0,
    },
    param_bounds={
        "height_range_min": (4.5,  6.0),
        "height_range_max": (12.0, 20.0),  # 极高宫殿
        "wall_thickness":   (0.50, 1.00),
        "floor_thickness":  (0.25, 0.50),
        "door_width":       (1.80, 3.00),  # 宫殿级大门
        "door_height":      (3.50, 5.00),
        "win_width":        (1.10, 2.20),
        "win_height":       (2.00, 3.00),  # 高大彩窗
        "win_density":      (0.40, 0.75),
        "subdivision":      (1.0,  3.0),
    },
    param_noise_scale={**_default_noise(), "height_range_max": 2.5, "door_height": 0.5, "win_height": 0.30},
)

# ─────────────────────────────────────────────
#  风格 16: 废弃精神病院 (Horror Asylum)
#  制度性走廊，铁格窗，冰冷压抑，蜂巢格间
# ─────────────────────────────────────────────
HORROR_ASYLUM = StyleDefinition(
    name="horror_asylum",
    description="废弃精神病院：制度性走廊，铁格窗，冰冷压抑，蜂巢格间布局",
    feature_vector=[
        0.55,   # architectural_period (维多利亚/早现代)
        0.70,   # wall_material_density (砖/混凝土)
        0.45,   # structural_complexity
        0.10,   # ornament_level (制度冷漠)
        0.10,   # ceiling_type (低平顶)
        0.30,   # lighting_type (昏暗灯光)
        0.70,   # symmetry_level (对称走廊)
        0.65,   # thermal_mass
        0.30,   # window_density (铁格窗)
        0.25,   # door_formality (制度性门)
        0.75,   # floor_material (混凝土/瓷砖)
        0.15,   # roof_type
        0.95,   # interior_division (最多分间)
        0.80,   # climate_adaptation (阴冷)
        0.85,   # security_level (高度安保)
        0.20,   # vertical_emphasis (低矮压抑)
    ],
    base_params={
        "height_range_min": 2.8,
        "height_range_max": 4.5,
        "wall_thickness":   0.40,
        "floor_thickness":  0.28,
        "door_width":       0.90,
        "door_height":      2.20,
        "win_width":        0.60,
        "win_height":       0.80,
        "win_density":      0.30,
        "subdivision":      7.0,
    },
    param_bounds={
        "height_range_min": (2.4,  3.5),
        "height_range_max": (3.5,  6.0),
        "wall_thickness":   (0.28, 0.55),
        "floor_thickness":  (0.20, 0.38),
        "door_width":       (0.75, 1.05),
        "door_height":      (2.00, 2.50),
        "win_width":        (0.40, 0.80),  # 铁格窗
        "win_height":       (0.60, 1.10),
        "win_density":      (0.18, 0.42),
        "subdivision":      (5.0,  8.0),   # 极多小格间
    },
    param_noise_scale={**_default_noise(), "subdivision": 1.0},
)

# ─────────────────────────────────────────────
#  风格 17: 地下墓穴 (Horror Crypt)
#  地下墓室：极低矮，极厚石墙，几乎完全封闭，迷宫布局
# ─────────────────────────────────────────────
HORROR_CRYPT = StyleDefinition(
    name="horror_crypt",
    description="地下墓穴：极低矮压抑，极厚石墙，几乎完全封闭，迷宫式布局",
    feature_vector=[
        0.30,   # architectural_period
        1.00,   # wall_material_density (最厚重石材)
        0.55,   # structural_complexity
        0.20,   # ornament_level (墓葬纹饰)
        0.25,   # ceiling_type (极低拱顶)
        0.02,   # lighting_type (近乎无光)
        0.35,   # symmetry_level (不规则)
        1.00,   # thermal_mass (极厚重冰冷)
        0.02,   # window_density (近乎无窗)
        0.20,   # door_formality (沉重石门)
        0.20,   # floor_material (腐朽石板)
        0.10,   # roof_type
        1.00,   # interior_division (极多墓室)
        0.90,   # climate_adaptation (极阴冷)
        0.90,   # security_level
        0.05,   # vertical_emphasis (极低矮)
    ],
    base_params={
        "height_range_min": 2.0,
        "height_range_max": 3.0,
        "wall_thickness":   0.85,
        "floor_thickness":  0.30,
        "door_width":       0.70,
        "door_height":      1.90,
        "win_width":        0.30,
        "win_height":       0.40,
        "win_density":      0.02,
        "subdivision":      7.0,
    },
    param_bounds={
        "height_range_min": (2.0,  2.8),   # 极低矮
        "height_range_max": (2.8,  4.5),
        "wall_thickness":   (0.60, 1.10),
        "floor_thickness":  (0.20, 0.42),
        "door_width":       (0.60, 0.85),  # 勉强通过
        "door_height":      (1.80, 2.20),
        "win_width":        (0.30, 0.42),
        "win_height":       (0.40, 0.58),
        "win_density":      (0.00, 0.08),  # 几乎无窗
        "subdivision":      (4.0,  8.0),
    },
    param_noise_scale={**_default_noise(), "win_density": 0.02, "subdivision": 1.0},
)

# ─────────────────────────────────────────────
#  风格 18: 日式神社/佛寺 (Japanese Temple)
#  宏伟木构神社，高台翘屋脊，庄严对称
# ─────────────────────────────────────────────
JAPANESE_TEMPLE = StyleDefinition(
    name="japanese_temple",
    description="日式神社/佛寺：宏伟重木构，高台翘脊屋顶，庄严对称，宗教性空间",
    feature_vector=[
        0.38,   # architectural_period
        0.45,   # wall_material_density (重木构)
        0.75,   # structural_complexity (复杂斗拱)
        0.80,   # ornament_level (宗教装饰)
        0.15,   # ceiling_type (低斜顶)
        0.40,   # lighting_type (自然漫射光)
        1.00,   # symmetry_level (极度对称)
        0.35,   # thermal_mass (木构)
        0.30,   # window_density (少窗庄重)
        0.85,   # door_formality (宏伟鸟居/大门)
        0.50,   # floor_material (木/石)
        0.65,   # roof_type (翘脊坡屋顶)
        0.50,   # interior_division
        0.60,   # climate_adaptation
        0.45,   # security_level
        0.55,   # vertical_emphasis
    ],
    base_params={
        "height_range_min": 3.5,
        "height_range_max": 6.0,
        "wall_thickness":   0.25,
        "floor_thickness":  0.35,
        "door_width":       1.20,
        "door_height":      2.80,
        "win_width":        0.60,
        "win_height":       1.20,
        "win_density":      0.30,
        "subdivision":      3.0,
    },
    param_bounds={
        "height_range_min": (2.8,  4.5),
        "height_range_max": (4.5,  8.0),
        "wall_thickness":   (0.18, 0.35),
        "floor_thickness":  (0.25, 0.50),  # 高台基
        "door_width":       (0.90, 1.50),
        "door_height":      (2.30, 3.50),  # 宏伟门洞
        "win_width":        (0.40, 0.85),
        "win_height":       (0.80, 1.60),
        "win_density":      (0.18, 0.42),
        "subdivision":      (2.0,  5.0),
    },
    param_noise_scale={**_default_noise(), "floor_thickness": 0.04, "door_height": 0.25},
)

# ─────────────────────────────────────────────
#  风格 19: 京町家 (Japanese Machiya)
#  城市连排商住宅：窄面深进，多间布局，轻薄木壁
# ─────────────────────────────────────────────
JAPANESE_MACHIYA = StyleDefinition(
    name="japanese_machiya",
    description="京町家/城市连排：窄面深进，多间布局，极薄木壁，城市商住混合",
    feature_vector=[
        0.45,   # architectural_period
        0.30,   # wall_material_density (极薄木/纸)
        0.60,   # structural_complexity
        0.45,   # ornament_level (素雅城市风)
        0.10,   # ceiling_type
        0.45,   # lighting_type
        0.60,   # symmetry_level (有限对称)
        0.25,   # thermal_mass (轻木构)
        0.40,   # window_density (街道窗)
        0.40,   # door_formality (商铺推拉门)
        0.50,   # floor_material (木/榻榻米)
        0.55,   # roof_type (坡屋顶)
        0.70,   # interior_division (多小间深进)
        0.60,   # climate_adaptation
        0.35,   # security_level
        0.40,   # vertical_emphasis (水平延伸)
    ],
    base_params={
        "height_range_min": 2.5,
        "height_range_max": 4.5,
        "wall_thickness":   0.12,
        "floor_thickness":  0.20,
        "door_width":       0.80,
        "door_height":      2.00,
        "win_width":        0.60,
        "win_height":       0.90,
        "win_density":      0.40,
        "subdivision":      5.0,
    },
    param_bounds={
        "height_range_min": (2.2,  3.2),
        "height_range_max": (3.5,  5.5),
        "wall_thickness":   (0.10, 0.18),  # 极薄木壁
        "floor_thickness":  (0.15, 0.28),
        "door_width":       (0.65, 1.00),
        "door_height":      (1.80, 2.30),
        "win_width":        (0.40, 0.80),
        "win_height":       (0.65, 1.20),
        "win_density":      (0.28, 0.55),
        "subdivision":      (3.0,  7.0),   # 窄深多间
    },
    param_noise_scale={**_default_noise(), "wall_thickness": 0.02, "subdivision": 1.0},
)

# ─────────────────────────────────────────────
#  风格 20: 伊斯兰沙漠宫殿 (Desert Palace)
#  沙漠宫殿：厚墙隔热，装饰性伊斯兰尖拱窗，高大庭院空间
# ─────────────────────────────────────────────
DESERT_PALACE = StyleDefinition(
    name="desert_palace",
    description="伊斯兰沙漠宫殿：极厚隔热墙，装饰性尖拱窗，高大庭院空间，华丽装饰",
    feature_vector=[
        0.20,   # architectural_period (古代/中世纪)
        0.85,   # wall_material_density (厚重土坯/石灰岩)
        0.75,   # structural_complexity (伊斯兰几何装饰)
        0.85,   # ornament_level (华丽几何纹样)
        0.70,   # ceiling_type (穹顶/拱顶)
        0.30,   # lighting_type (小窗柔光)
        0.90,   # symmetry_level (伊斯兰轴对称)
        0.95,   # thermal_mass (极厚隔热)
        0.30,   # window_density (装饰性少量拱窗)
        0.90,   # door_formality (宏伟装饰拱门)
        0.20,   # floor_material (石/瓷砖)
        0.55,   # roof_type (穹顶)
        0.40,   # interior_division (庭院开放)
        0.05,   # climate_adaptation (极热沙漠)
        0.65,   # security_level
        0.70,   # vertical_emphasis (穹顶高耸)
    ],
    base_params={
        "height_range_min": 4.0,
        "height_range_max": 10.0,
        "wall_thickness":   0.80,
        "floor_thickness":  0.22,
        "door_width":       1.50,
        "door_height":      3.50,
        "win_width":        0.60,
        "win_height":       1.80,
        "win_density":      0.30,
        "subdivision":      3.0,
    },
    param_bounds={
        "height_range_min": (3.0,  5.5),
        "height_range_max": (7.0, 14.0),   # 高大庭院/大厅
        "wall_thickness":   (0.60, 1.10),  # 极厚隔热
        "floor_thickness":  (0.15, 0.32),
        "door_width":       (1.10, 2.00),
        "door_height":      (2.80, 4.50),  # 高大装饰拱门
        "win_width":        (0.40, 0.90),
        "win_height":       (1.20, 2.50),  # 细高伊斯兰拱窗
        "win_density":      (0.18, 0.45),
        "subdivision":      (2.0,  5.0),
    },
    param_noise_scale={**_default_noise(), "height_range_max": 1.5, "win_height": 0.25},
)

# 风格注册表
STYLE_REGISTRY: Dict[str, StyleDefinition] = {
    "medieval":              MEDIEVAL,
    "modern":                MODERN,
    "industrial":            INDUSTRIAL,
    "fantasy":               FANTASY,
    "horror":                HORROR,
    "japanese":              JAPANESE,
    "desert":                DESERT,
    # ── 中世纪衍生 ──
    "medieval_chapel":       MEDIEVAL_CHAPEL,
    "medieval_keep":         MEDIEVAL_KEEP,
    # ── 现代衍生 ──
    "modern_loft":           MODERN_LOFT,
    "modern_villa":          MODERN_VILLA,
    # ── 工业衍生 ──
    "industrial_workshop":   INDUSTRIAL_WORKSHOP,
    "industrial_powerplant": INDUSTRIAL_POWERPLANT,
    # ── 奇幻衍生 ──
    "fantasy_dungeon":       FANTASY_DUNGEON,
    "fantasy_palace":        FANTASY_PALACE,
    # ── 恐怖衍生 ──
    "horror_asylum":         HORROR_ASYLUM,
    "horror_crypt":          HORROR_CRYPT,
    # ── 和风衍生 ──
    "japanese_temple":       JAPANESE_TEMPLE,
    "japanese_machiya":      JAPANESE_MACHIYA,
    # ── 沙漠衍生 ──
    "desert_palace":         DESERT_PALACE,
}


def get_style_bounds_normalized(style_name: str) -> np.ndarray:
    """
    返回指定风格每个输出参数的归一化边界，shape: [OUTPUT_DIM, 2]
    每行 [lo_norm, hi_norm]，基于 param_bounds 中的物理值上下界换算。
    """
    style = STYLE_REGISTRY[style_name]
    bounds = np.zeros((OUTPUT_DIM, 2), dtype=np.float32)
    for i, key in enumerate(OUTPUT_KEYS):
        phys_lo, phys_hi = style.param_bounds[key]
        g_lo, g_hi = OUTPUT_PARAMS[key]["range"]
        bounds[i, 0] = (phys_lo - g_lo) / (g_hi - g_lo)
        bounds[i, 1] = (phys_hi - g_lo) / (g_hi - g_lo)
    return np.clip(bounds, 0.0, 1.0)


def get_feature_vector(style_name: str) -> np.ndarray:
    """返回指定风格的特征向量 (numpy array, shape: [16])"""
    return np.array(STYLE_REGISTRY[style_name].feature_vector, dtype=np.float32)


def get_param_vector(style_name: str) -> np.ndarray:
    """返回指定风格的参数向量 (numpy array, shape: [10])，已归一化到 [0,1]"""
    style = STYLE_REGISTRY[style_name]
    values = []
    for key in OUTPUT_KEYS:
        raw = style.base_params[key]
        lo, hi = OUTPUT_PARAMS[key]["range"]
        values.append((raw - lo) / (hi - lo))
    return np.clip(np.array(values, dtype=np.float32), 0.0, 1.0)


def denormalize_params(normalized: np.ndarray) -> Dict[str, float]:
    """将归一化参数向量还原为真实物理值"""
    result = {}
    for i, key in enumerate(OUTPUT_KEYS):
        lo, hi = OUTPUT_PARAMS[key]["range"]
        val = float(normalized[i]) * (hi - lo) + lo
        if key == "subdivision":
            val = round(max(1, min(8, val)))
        result[key] = round(val, 4)
    return result


if __name__ == "__main__":
    for name, style in STYLE_REGISTRY.items():
        fv = get_feature_vector(name)
        pv = get_param_vector(name)
        params = denormalize_params(pv)
        print(f"\n[{name.upper()}] {style.description}")
        print(f"  Feature vector: {fv.tolist()}")
        print(f"  Params: {params}")
