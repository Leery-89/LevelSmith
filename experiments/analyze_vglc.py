"""Analyze VGLC dungeon graphs for connectivity patterns."""
import glob, re, sys, os
from collections import defaultdict, deque

sys.stdout.reconfigure(encoding="utf-8")

dot_files = glob.glob("training/VGLC/**/*.dot", recursive=True)
print(f"Total .dot files: {len(dot_files)}")

stats = {
    "node_counts": [],
    "edge_counts": [],
    "avg_degree": [],
    "dead_ends": [],
    "edge_types": defaultdict(int),
    "node_types": defaultdict(int),
    "start_to_boss": [],
}

has_cycle = 0
total_graphs = 0

for f in dot_files:
    text = open(f).read()
    nodes = re.findall(r'(\d+)\s*\[label="([^"]*)"\]', text)
    edges = re.findall(r'(\d+)\s*->\s*(\d+)\s*\[label="([^"]*)"\]', text)
    if not nodes:
        continue

    total_graphs += 1
    n = len(nodes)
    e = len(edges)

    degree = defaultdict(int)
    adj = defaultdict(set)
    for src, dst, lbl in edges:
        degree[src] += 1
        degree[dst] += 1
        adj[src].add(dst)
        adj[dst].add(src)
        stats["edge_types"][lbl or "normal"] += 1

    for nid, ntype in nodes:
        for t in ntype.split(","):
            t = t.strip()
            stats["node_types"][t if t else "empty"] += 1

    dead = sum(1 for d in degree.values() if d == 1)
    stats["node_counts"].append(n)
    stats["edge_counts"].append(e)
    stats["avg_degree"].append(2 * e / n if n > 0 else 0)
    stats["dead_ends"].append(dead / n if n > 0 else 0)

    # Cycle detection
    e_undirected = len(set(tuple(sorted([s, d])) for s, d, _ in edges))
    if e_undirected > n - 1:
        has_cycle += 1

    # BFS start -> boss
    start = [nid for nid, nt in nodes if "s" in nt.split(",")]
    boss = [nid for nid, nt in nodes if "b" in nt.split(",")]
    if start and boss:
        visited = {start[0]}
        queue = deque([(start[0], 0)])
        while queue:
            node, dist = queue.popleft()
            if node == boss[0]:
                stats["start_to_boss"].append(dist)
                break
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))

import statistics

print(f"\n{'='*50}")
print(f"  VGLC Dungeon Graph Statistics")
print(f"{'='*50}")
print(f"  Graphs analyzed:  {total_graphs}")
print(f"  Avg nodes/graph:  {statistics.mean(stats['node_counts']):.1f}")
print(f"  Avg edges/graph:  {statistics.mean(stats['edge_counts']):.1f}")
print(f"  Avg degree:       {statistics.mean(stats['avg_degree']):.2f}")
print(f"  Dead-end ratio:   {statistics.mean(stats['dead_ends']):.1%}")
print(f"  Graphs w/ cycles: {has_cycle}/{total_graphs} ({has_cycle/max(1,total_graphs)*100:.0f}%)")

if stats["start_to_boss"]:
    sb = stats["start_to_boss"]
    print(f"\n  Start-to-Boss distance:")
    print(f"    min={min(sb)}  avg={statistics.mean(sb):.1f}  max={max(sb)}  median={statistics.median(sb)}")

print(f"\n  Edge types:")
for k, v in sorted(stats["edge_types"].items(), key=lambda x: -x[1]):
    print(f"    {k:20s}: {v}")

print(f"\n  Node types:")
for k, v in sorted(stats["node_types"].items(), key=lambda x: -x[1]):
    print(f"    {k:20s}: {v}")
