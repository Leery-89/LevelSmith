"""Per-game breakdown of VGLC dungeon graph statistics."""
import sys, glob, re, statistics
from collections import defaultdict, deque

sys.stdout.reconfigure(encoding="utf-8")

dot_files = glob.glob("training/VGLC/**/*.dot", recursive=True)

by_game = defaultdict(list)
for f in dot_files:
    if "Link to the Past" in f: game = "LttP"
    elif "Awakening" in f: game = "LA"
    elif "Legend of Zelda" in f: game = "LoZ"
    elif "Doom2" in f: game = "Doom2"
    elif "Doom" in f: game = "Doom"
    else: game = "other"

    text = open(f).read()
    nodes = re.findall(r'(\d+)\s*\[label="([^"]*)"\]', text)
    edges = re.findall(r'(\d+)\s*->\s*(\d+)\s*\[label="([^"]*)"\]', text)
    if not nodes:
        continue

    n = len(nodes)
    ue = set(tuple(sorted([s, d])) for s, d, _ in edges)
    e = len(ue)

    adj = defaultdict(set)
    for s, d, _ in edges:
        adj[s].add(d)
        adj[d].add(s)

    degree_vals = [len(adj[nid]) for nid, _ in nodes]
    avg_deg = sum(degree_vals) / max(1, len(degree_vals))
    dead = sum(1 for d in degree_vals if d == 1) / max(1, n)
    has_cycle = 1 if e > n - 1 else 0

    start = [nid for nid, nt in nodes if "s" in nt.split(",")]
    boss = [nid for nid, nt in nodes if "b" in nt.split(",")]
    sb_dist = None
    if start and boss:
        visited = {start[0]}
        queue = deque([(start[0], 0)])
        while queue:
            node, dist = queue.popleft()
            if node == boss[0]:
                sb_dist = dist
                break
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))

    by_game[game].append({
        "n": n, "e": e, "avg_deg": avg_deg,
        "dead_ratio": dead, "has_cycle": has_cycle, "sb_dist": sb_dist,
    })

header = "  Game   #  Nodes  Edges  Degree  Dead%  Cycle%  S->B"
print(header)
print("  " + "-" * (len(header) - 2))

for game in ["LoZ", "LttP", "LA", "Doom", "Doom2"]:
    data = by_game.get(game, [])
    if not data:
        continue
    ns = [d["n"] for d in data]
    es = [d["e"] for d in data]
    degs = [d["avg_deg"] for d in data]
    deads = [d["dead_ratio"] for d in data]
    cycles = sum(d["has_cycle"] for d in data)
    sbs = [d["sb_dist"] for d in data if d["sb_dist"] is not None]
    sb_str = f"{statistics.mean(sbs):.1f}" if sbs else "n/a"

    print(f"  {game:<5} {len(data):>2}  {statistics.mean(ns):>5.1f}  {statistics.mean(es):>5.1f}"
          f"  {statistics.mean(degs):>6.2f}  {statistics.mean(deads):>4.1%}"
          f"  {cycles/len(data):>5.0%}  {sb_str:>5}")

# Summary
print()
all_data = [d for dlist in by_game.values() for d in dlist]
all_degs = [d["avg_deg"] for d in all_data]
all_deads = [d["dead_ratio"] for d in all_data]
all_cycles = sum(d["has_cycle"] for d in all_data)
all_sbs = [d["sb_dist"] for d in all_data if d["sb_dist"] is not None]

print("=" * 50)
print("  Summary (all games)")
print("=" * 50)
print(f"  Total graphs:     {len(all_data)}")
print(f"  Avg degree:       {statistics.mean(all_degs):.2f}")
print(f"  Dead-end ratio:   {statistics.mean(all_deads):.1%}")
print(f"  Cycle ratio:      {all_cycles}/{len(all_data)} ({all_cycles/len(all_data)*100:.0f}%)")
if all_sbs:
    print(f"  Start->Boss:      min={min(all_sbs)} avg={statistics.mean(all_sbs):.1f} "
          f"max={max(all_sbs)} median={statistics.median(all_sbs):.0f}")
