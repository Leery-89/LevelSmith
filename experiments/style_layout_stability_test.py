#!/usr/bin/env python3
"""
风格×布局稳定性验证测试
测试6个代表性风格在5种布局下的生成质量和稳定性
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any

# 添加training目录到路径
script_dir = Path(__file__).parent
training_dir = script_dir.parent / "training"
sys.path.insert(0, str(training_dir))

try:
    import trimesh
    import level_layout
    from style_base_profiles import STYLE_BASE_PROFILES
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the experiments/ directory")
    sys.exit(1)

# 测试配置
TEST_STYLES = ["japanese", "medieval_keep", "industrial", "fantasy_palace", "horror", "desert"]
TEST_LAYOUTS = ["street", "grid", "plaza", "random", "organic"]
TEST_COUNT = 8  # 每个测试生成8个建筑
RESULTS_DIR = script_dir / "stability_test_results"

def ensure_results_dir():
    """确保结果目录存在"""
    RESULTS_DIR.mkdir(exist_ok=True)

def analyze_mesh_complexity(glb_path: Path) -> Dict[str, Any]:
    """分析GLB文件的网格复杂度"""
    try:
        if not glb_path.exists():
            return {"error": "file_not_found"}
        
        scene = trimesh.load(str(glb_path))
        
        if hasattr(scene, 'geometry'):
            # Scene对象
            meshes = list(scene.geometry.values())
            total_faces = sum(len(mesh.faces) for mesh in meshes if hasattr(mesh, 'faces'))
            mesh_count = len(meshes)
        else:
            # 单个Mesh对象
            total_faces = len(scene.faces) if hasattr(scene, 'faces') else 0
            mesh_count = 1
        
        file_size = glb_path.stat().st_size
        
        return {
            "mesh_count": mesh_count,
            "total_faces": total_faces,
            "file_size_kb": round(file_size / 1024, 1),
            "complexity_score": round(total_faces / max(1, mesh_count), 1)  # 平均每mesh面数
        }
    except Exception as e:
        return {"error": str(e)}

def run_single_test(style: str, layout: str) -> Dict[str, Any]:
    """运行单个风格×布局组合测试"""
    print(f"  Testing {style} × {layout}...")
    
    result = {
        "style": style,
        "layout": layout,
        "timestamp": time.time(),
        "success": False,
        "error": None,
        "generation_time": 0,
        "mesh_analysis": {},
        "quality_assessment": "unknown"
    }
    
    output_filename = f"{style}_{layout}.glb"
    output_path = RESULTS_DIR / output_filename
    
    start_time = time.time()
    
    try:
        # 构建命令行参数（模拟level_layout.py的调用）
        args = [
            "--style", style,
            "--layout", layout,
            "--count", str(TEST_COUNT),
            "--out", str(output_path),
            "--area", "80",  # 中等大小区域
            "--seed", "42"   # 固定种子确保可重现
        ]
        
        # 调用level_layout的main函数
        # 需要临时修改sys.argv来传递参数
        original_argv = sys.argv[:]
        sys.argv = ["level_layout.py"] + args
        
        # 执行生成
        level_layout.main()
        
        # 恢复原始argv
        sys.argv = original_argv
        
        result["generation_time"] = time.time() - start_time
        result["success"] = True
        
        # 分析生成的文件
        result["mesh_analysis"] = analyze_mesh_complexity(output_path)
        
        # 简单的质量评估
        if result["mesh_analysis"].get("error"):
            result["quality_assessment"] = "broken"
        elif result["mesh_analysis"].get("total_faces", 0) < 100:
            result["quality_assessment"] = "weak"
        elif result["mesh_analysis"].get("total_faces", 0) > 10000:
            result["quality_assessment"] = "complex"
        else:
            result["quality_assessment"] = "acceptable"
            
    except Exception as e:
        result["error"] = str(e)
        result["error_traceback"] = traceback.format_exc()
        result["generation_time"] = time.time() - start_time
        
        # 尝试简单的错误分类
        error_str = str(e).lower()
        if "style" in error_str and "not found" in error_str:
            result["quality_assessment"] = "style_missing"
        elif "layout" in error_str:
            result["quality_assessment"] = "layout_error"
        else:
            result["quality_assessment"] = "broken"
    
    return result

def assess_combination_compatibility(style: str, layout: str) -> str:
    """评估风格×布局组合的理论兼容性"""
    
    # 基于风格特点的布局偏好
    style_layout_preferences = {
        "japanese": {"good": ["street", "organic"], "neutral": ["plaza"], "poor": ["grid", "random"]},
        "medieval_keep": {"good": ["plaza", "organic"], "neutral": ["street"], "poor": ["grid", "random"]},
        "industrial": {"good": ["grid", "street"], "neutral": ["random"], "poor": ["plaza", "organic"]},
        "fantasy_palace": {"good": ["plaza", "organic"], "neutral": ["street"], "poor": ["grid", "random"]},
        "horror": {"good": ["random", "organic"], "neutral": ["street"], "poor": ["grid", "plaza"]},
        "desert": {"good": ["plaza", "random"], "neutral": ["organic"], "poor": ["grid", "street"]}
    }
    
    prefs = style_layout_preferences.get(style, {"good": [], "neutral": [], "poor": []})
    
    if layout in prefs["good"]:
        return "theoretically_good"
    elif layout in prefs["neutral"]:
        return "theoretically_neutral"
    elif layout in prefs["poor"]:
        return "theoretically_poor"
    else:
        return "unknown"

def generate_stability_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成稳定性总报告"""
    
    # 统计成功率
    total_tests = len(results)
    successful_tests = len([r for r in results if r["success"]])
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    # 按布局分组统计
    layout_stats = {}
    for layout in TEST_LAYOUTS:
        layout_results = [r for r in results if r["layout"] == layout]
        layout_stats[layout] = {
            "total": len(layout_results),
            "successful": len([r for r in layout_results if r["success"]]),
            "success_rate": len([r for r in layout_results if r["success"]]) / len(layout_results) if layout_results else 0,
            "avg_generation_time": sum(r["generation_time"] for r in layout_results) / len(layout_results) if layout_results else 0
        }
    
    # 按风格分组统计
    style_stats = {}
    for style in TEST_STYLES:
        style_results = [r for r in results if r["style"] == style]
        style_stats[style] = {
            "total": len(style_results),
            "successful": len([r for r in style_results if r["success"]]),
            "success_rate": len([r for r in style_results if r["success"]]) / len(style_results) if style_results else 0,
            "avg_faces": sum(r["mesh_analysis"].get("total_faces", 0) for r in style_results) / len(style_results) if style_results else 0
        }
    
    # 识别问题组合
    failed_combinations = [r for r in results if not r["success"]]
    weak_combinations = [r for r in results if r["quality_assessment"] in ["weak", "broken"]]
    
    # 推荐展示组合
    good_combinations = []
    for r in results:
        if (r["success"] and 
            r["quality_assessment"] in ["acceptable", "complex"] and
            r["mesh_analysis"].get("total_faces", 0) > 500):  # 有足够复杂度
            
            theoretical_compatibility = assess_combination_compatibility(r["style"], r["layout"])
            good_combinations.append({
                "style": r["style"],
                "layout": r["layout"], 
                "faces": r["mesh_analysis"].get("total_faces", 0),
                "theoretical_compatibility": theoretical_compatibility,
                "generation_time": r["generation_time"]
            })
    
    # 按理论兼容性和面数排序
    good_combinations.sort(key=lambda x: (
        x["theoretical_compatibility"] == "theoretically_good",
        x["faces"]
    ), reverse=True)
    
    return {
        "test_summary": {
            "total_combinations": total_tests,
            "successful_combinations": successful_tests,
            "overall_success_rate": round(success_rate, 3),
            "test_timestamp": time.time()
        },
        "layout_performance": layout_stats,
        "style_performance": style_stats,
        "failed_combinations": [
            {"style": r["style"], "layout": r["layout"], "error": r["error"]} 
            for r in failed_combinations
        ],
        "weak_combinations": [
            {"style": r["style"], "layout": r["layout"], "assessment": r["quality_assessment"]}
            for r in weak_combinations
        ],
        "recommended_for_readme": good_combinations[:8],  # 推荐前8个组合
        "detailed_results": results
    }

def generate_markdown_report(report: Dict[str, Any]) -> str:
    """生成Markdown格式的报告"""
    
    md = ["# LevelSmith 风格×布局稳定性验证报告\n"]
    
    # 测试概况
    summary = report["test_summary"]
    md.append("## 测试概况\n")
    md.append(f"- **总测试组合数**: {summary['total_combinations']}")
    md.append(f"- **成功组合数**: {summary['successful_combinations']}")
    md.append(f"- **整体成功率**: {summary['overall_success_rate']:.1%}")
    md.append(f"- **测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(summary['test_timestamp']))}\n")
    
    # 布局性能分析
    md.append("## 布局性能分析\n")
    md.append("| 布局类型 | 成功率 | 平均生成时间(s) |")
    md.append("|----------|--------|----------------|")
    
    for layout, stats in report["layout_performance"].items():
        md.append(f"| {layout} | {stats['success_rate']:.1%} | {stats['avg_generation_time']:.1f} |")
    md.append("")
    
    # 风格性能分析
    md.append("## 风格性能分析\n")
    md.append("| 风格 | 成功率 | 平均面数 |")
    md.append("|------|--------|----------|")
    
    for style, stats in report["style_performance"].items():
        md.append(f"| {style} | {stats['success_rate']:.1%} | {stats['avg_faces']:.0f} |")
    md.append("")
    
    # 推荐组合
    md.append("## 推荐用于README展示的组合\n")
    md.append("| 风格 | 布局 | 面数 | 理论兼容性 | 生成时间(s) |")
    md.append("|------|------|------|------------|------------|")
    
    for combo in report["recommended_for_readme"]:
        md.append(f"| {combo['style']} | {combo['layout']} | {combo['faces']} | {combo['theoretical_compatibility']} | {combo['generation_time']:.1f} |")
    md.append("")
    
    # 问题组合
    if report["failed_combinations"]:
        md.append("## 失败组合\n")
        for combo in report["failed_combinations"]:
            md.append(f"- **{combo['style']} × {combo['layout']}**: {combo['error']}")
        md.append("")
    
    if report["weak_combinations"]:
        md.append("## 质量较弱组合\n")
        for combo in report["weak_combinations"]:
            md.append(f"- **{combo['style']} × {combo['layout']}**: {combo['assessment']}")
        md.append("")
    
    return "\n".join(md)

def main():
    """主测试流程"""
    print("=== LevelSmith 风格×布局稳定性验证测试 ===\n")
    
    ensure_results_dir()
    
    print(f"测试配置:")
    print(f"- 风格: {', '.join(TEST_STYLES)}")
    print(f"- 布局: {', '.join(TEST_LAYOUTS)}")
    print(f"- 每组合建筑数: {TEST_COUNT}")
    print(f"- 输出目录: {RESULTS_DIR}\n")
    
    all_results = []
    total_combinations = len(TEST_STYLES) * len(TEST_LAYOUTS)
    
    print(f"开始测试 {total_combinations} 个组合...\n")
    
    for i, (style, layout) in enumerate([(s, l) for s in TEST_STYLES for l in TEST_LAYOUTS], 1):
        print(f"[{i}/{total_combinations}] {style} × {layout}")
        result = run_single_test(style, layout)
        all_results.append(result)
        
        if result["success"]:
            faces = result["mesh_analysis"].get("total_faces", 0)
            print(f"  ✓ 成功 ({faces} 面, {result['generation_time']:.1f}s)")
        else:
            print(f"  ✗ 失败: {result['error']}")
        print()
    
    print("生成总报告...")
    
    # 生成报告
    stability_report = generate_stability_report(all_results)
    markdown_report = generate_markdown_report(stability_report)
    
    # 保存报告
    json_report_path = RESULTS_DIR / "stability_report.json"
    md_report_path = RESULTS_DIR / "stability_report.md"
    
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(stability_report, f, ensure_ascii=False, indent=2)
    
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"\n=== 测试完成 ===")
    print(f"总成功率: {stability_report['test_summary']['overall_success_rate']:.1%}")
    print(f"JSON报告: {json_report_path}")
    print(f"Markdown报告: {md_report_path}")
    print(f"生成文件目录: {RESULTS_DIR}")
    
    # 显示推荐组合
    print(f"\n推荐用于README展示的前5个组合:")
    for i, combo in enumerate(stability_report["recommended_for_readme"][:5], 1):
        print(f"  {i}. {combo['style']} × {combo['layout']} ({combo['faces']} 面)")

if __name__ == "__main__":
    main()