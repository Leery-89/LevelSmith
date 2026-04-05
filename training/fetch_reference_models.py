"""
fetch_reference_models.py
从 Sketchfab 按授权 + 分类批量抓取架构模型，
客户端按关键词打分后分配到对应风格目录并下载 GLB。

策略:
  - API 层: categories=architecture + license=by/cc0 + sort_by=-viewCount
  - 客户端: 按名称 / tags 关键词给每个模型打各风格的匹配分，
           最高分超阈值则归入该风格
  - 每风格最多 MAX_PER_STYLE 个模型
"""

import io
import json
import time
import zipfile
from pathlib import Path

import requests

# ─── 配置 ─────────────────────────────────────────────────────
SKETCHFAB_TOKEN = "9222967b2bb5448d97f1d8108439e77b"
BASE_URL  = "https://api.sketchfab.com/v3"
HEADERS   = {"Authorization": f"Token {SKETCHFAB_TOKEN}"}

OUTPUT_DIR    = Path(__file__).parent / "validation_data"
MAX_PER_STYLE = 5
PAGES_PER_LICENSE = 20  # 每种 license 抓取页数 (每页 24 个)

# API 层 license 参数（by-sa 覆盖 Cathedral 等 CC BY-SA 作品）
LICENSES = ["by", "by-sa", "cc0"]

# 每种风格的关键词（名称或 tags 中出现则得分 +1）
STYLE_KEYWORDS = {
    "modern": [
        "modern", "contemporary", "minimalist", "apartment", "office",
        "city", "urban", "skyscraper", "glass", "steel", "highrise",
        "low poly city", "city block", "night city",
    ],
    "industrial": [
        "warehouse", "industrial", "factory", "steampunk", "pipe",
        "metal", "rust", "machinery", "silo", "dock", "hangar",
    ],
    "fantasy": [
        "castle", "medieval", "cathedral", "fantasy", "tower", "fortress",
        "dungeon", "church", "chapel", "gothic", "knights", "palace",
        "keep", "battlement", "merlon",
    ],
    "horror": [
        "abandoned", "horror", "apocalyptic", "ruined", "destroyed",
        "post-apocalyptic", "haunted", "decay", "derelict", "wreckage",
        "scary", "creepy", "ghost", "backrooms",
    ],
    "japanese": [
        "japanese", "japan", "asia", "asian", "pagoda", "temple",
        "shrine", "torii", "tokyo", "osaka", "ninja", "samurai", "kyoto",
        "taiwan", "korea",
    ],
    "desert": [
        "desert", "adobe", "arid", "dune", "sandstone", "moroccan",
        "arabic", "arab", "egypt", "egyptian", "pueblo",
        "canyon", "oasis", "mud", "terracotta",
        "ancient", "ruin", "ruins", "mosque", "minaret",
        "kaaba", "mecca", "islam", "persia", "ottoman",
        "babylon", "mayan", "mesopotamia", "tatooine",
        "dubai", "middle east", "charminar",
    ],
}


def score_model(model: dict) -> dict[str, int]:
    """对每个风格计算关键词匹配分。"""
    name = (model.get("name") or "").lower()
    tags = " ".join(t.get("name", "") for t in (model.get("tags") or [])).lower()
    text = name + " " + tags

    scores = {}
    for style, keywords in STYLE_KEYWORDS.items():
        scores[style] = sum(1 for kw in keywords if kw in text)
    return scores


def best_style(scores: dict[str, int], threshold: int = 1) -> str | None:
    """返回得分最高且超过阈值的风格，否则 None。"""
    best = max(scores, key=scores.get)
    return best if scores[best] >= threshold else None


# ─── API 工具 ─────────────────────────────────────────────────

def fetch_page(license_slug: str, cursor: str | None) -> tuple[list, str | None]:
    """抓取一页结果，返回 (models, next_cursor)。"""
    params = {
        "categories":   "architecture",
        "license":      license_slug,
        "downloadable": "true",
        "count":        24,
        "sort_by":      "-viewCount",
    }
    if cursor:
        params["cursor"] = cursor

    try:
        resp = requests.get(f"{BASE_URL}/models", headers=HEADERS,
                            params=params, timeout=30)
    except requests.RequestException as e:
        print(f"  [网络错误] {e}")
        return [], None

    if resp.status_code != 200:
        print(f"  [抓取失败] HTTP {resp.status_code}")
        return [], None

    data = resp.json()
    next_url = data.get("next")
    next_cursor = None
    if next_url:
        import urllib.parse
        qs = urllib.parse.urlparse(next_url).query
        next_cursor = urllib.parse.parse_qs(qs).get("cursor", [None])[0]

    return data.get("results", []), next_cursor


def get_download_info(uid: str) -> dict:
    """返回 {"url":..., "format":...} 或 {}。"""
    try:
        resp = requests.get(f"{BASE_URL}/models/{uid}/download",
                            headers=HEADERS, timeout=30)
    except requests.RequestException as e:
        print(f"    [网络错误] {e}")
        return {}
    if resp.status_code != 200:
        return {}
    data = resp.json()
    for fmt in ("glb", "gltf", "source"):
        if fmt in data and data[fmt].get("url"):
            return {"url": data[fmt]["url"], "format": fmt}
    return {}


def download_glb(url: str, dest: Path) -> bool:
    """下载并保存（自动处理 ZIP）。"""
    try:
        resp = requests.get(url, stream=True, timeout=180)
    except requests.RequestException as e:
        print(f"    [下载错误] {e}")
        return False

    if resp.status_code != 200:
        print(f"    [下载失败] HTTP {resp.status_code}")
        return False

    raw = resp.content
    if "zip" in resp.headers.get("Content-Type", "") or raw[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                glb_names = [n for n in zf.namelist()
                             if n.lower().endswith((".glb", ".gltf"))]
                if not glb_names:
                    return False
                glb_names.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
                raw = zf.read(glb_names[0])
                print(f"    解压: {glb_names[0]}")
        except zipfile.BadZipFile:
            return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return True


# ─── 主逻辑 ───────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Sketchfab 参考模型下载器  (类别抓取 + 客户端关键词分配)")
    print("=" * 65)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 验证 Token
    try:
        resp = requests.get(f"{BASE_URL}/me", headers=HEADERS, timeout=15)
        user = resp.json().get("username", "?") if resp.status_code == 200 else "?"
        print(f"  账号: {user}\n")
    except Exception:
        print("  [警告] Token 验证失败，继续...\n")

    # 清理旧 GLB（保留 index.json）
    for style in STYLE_KEYWORDS:
        for f in (OUTPUT_DIR / style).glob("*.glb"):
            f.unlink()

    # ── 批量抓取候选模型 ────────────────────────────────────────
    print("[1] 抓取候选模型（architecture 分类，CC BY + CC0）...")
    all_candidates = []   # list of model dicts
    seen_uids = set()

    for lic in LICENSES:
        cursor = None
        for page in range(PAGES_PER_LICENSE):
            models, cursor = fetch_page(lic, cursor)
            new = [m for m in models if m.get("uid") not in seen_uids]
            for m in new:
                seen_uids.add(m["uid"])
            all_candidates.extend(new)
            print(f"  license={lic} page {page+1}: +{len(new)} 个 (合计 {len(all_candidates)})")
            time.sleep(0.5)
            if not cursor:
                break

    print(f"\n  共找到 {len(all_candidates)} 个候选模型")

    # ── 按关键词打分分配风格 ────────────────────────────────────
    print("\n[2] 客户端关键词打分，分配到各风格...")
    buckets: dict[str, list] = {s: [] for s in STYLE_KEYWORDS}

    for m in sorted(all_candidates, key=lambda x: x.get("viewCount", 0), reverse=True):
        scores = score_model(m)
        style  = best_style(scores)
        if style and len(buckets[style]) < MAX_PER_STYLE * 3:   # 预留备份
            buckets[style].append(m)

    for style, mods in buckets.items():
        print(f"  {style:<12}: {len(mods):3d} 候选")
        for m in mods[:MAX_PER_STYLE]:
            tags = [t.get("name","") for t in (m.get("tags") or [])][:4]
            print(f"    views={m.get('viewCount',0):7d} {m['name'][:50]:50s}  tags:{tags}")

    # ── 下载 ─────────────────────────────────────────────────
    print("\n[3] 下载 GLB 文件...")
    summary = {}

    for style, candidates in buckets.items():
        style_dir = OUTPUT_DIR / style
        style_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        meta_list  = []

        for model in candidates:
            if downloaded >= MAX_PER_STYLE:
                break

            uid  = model.get("uid", "")
            name = model.get("name", "unknown")[:60]
            lic  = (model.get("license") or {}).get("label", "?")

            print(f"\n  [{style}] {name}  [{lic}]")
            dl_info = get_download_info(uid)
            if not dl_info:
                print(f"    跳过（无下载链接）")
                time.sleep(0.5)
                continue

            dest = style_dir / f"{uid}.glb"
            print(f"    下载 [{dl_info['format']}] ...")
            ok = download_glb(dl_info["url"], dest)
            if ok:
                size_kb = dest.stat().st_size // 1024
                print(f"    已保存: {dest.name} ({size_kb} KB)")
                meta_list.append({"uid": uid, "name": name, "license": lic})
                downloaded += 1
            else:
                if dest.exists():
                    dest.unlink()

            time.sleep(1.0)

        # 写索引
        meta_path = style_dir / "index.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, indent=2, ensure_ascii=False)

        summary[style] = downloaded

    # ── 汇总 ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  下载汇总")
    print("=" * 65)
    total = 0
    for style, count in summary.items():
        status = "OK" if count >= 3 else ("部分" if count > 0 else "未找到")
        print(f"  {style:<15}: {count:>2} 个模型  [{status}]")
        total += count
    print(f"  {'─' * 35}")
    print(f"  {'总计':<15}: {total:>2} 个模型")
    print(f"\n  保存目录: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
