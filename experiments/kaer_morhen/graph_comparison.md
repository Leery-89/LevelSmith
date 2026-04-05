# Graph Layout Family Comparison

Two fortified compound prototypes generated from the same graph pipeline,
proving graph mode is a reusable layout family.

## Cases

| | Kaer Morhen | Oxenfurt Redoubt |
|--|-------------|-----------------|
| **Graph file** | `kaer_morhen_layout_graph.json` | `oxenfurt_redoubt_layout_graph.json` |
| **Topology** | terraced_axial | gatehouse_axial |
| **Style** | medieval_keep | medieval_keep |
| **Seed** | 42 | 42 |
| **Area** | 100x100m | 100x100m |

## Metrics

| Metric | Kaer Morhen | Oxenfurt Redoubt |
|--------|-------------|-----------------|
| Block count | 7 | 7 |
| Role distribution | 1 primary, 3 secondary, 1 tertiary, 2 ambient | 1 primary, 3 secondary (1 gatehouse + 2 towers), 1 tertiary, 2 ambient |
| Gate count | 2 (south + north) | 1 (south only) |
| Tower count | 3 (SW, SE, NW corners) | 2 (flanking gate at SW, SE) |
| Coverage ratio | 6.3% | 6.6% |
| Overlap count | 0 | 0 |
| Too-close count | 0 | 0 |
| Keep position | (50, 70) rear-center | (50, 70) rear-center |
| Keep size | 15x12m | 15x12m |
| Gatehouse | None (barbican tower at NW) | Yes, south-center (50, 13), 10x8m |
| Courtyard | Yes | Yes |
| Entry axis | Gate-to-keep | Gatehouse-to-keep (stronger) |

## Spatial Differences

**Kaer Morhen** is a sprawling mountain fortress:
- 3 residential/defensive towers spread across corners
- 2 gates (south main + north garden) suggesting multiple access routes
- Barbican tower sits at NW, away from the main gate
- Towers serve residential and defensive purposes (Triss, Yennefer quarters)
- Damaged/ruined state (debris, rubble in graph data)

**Oxenfurt Redoubt** is a compact rectangular fortification:
- Dedicated gatehouse at south-center, directly on the entry axis
- Only 2 towers, both flanking the gate (east + west at south wall)
- Single gate, controlled entry
- Chapel as tertiary (religious, not military support)
- Stable as ambient (logistical, not training)
- Intact state (no damage distribution in graph)

## Why Both Are "Fortified Compound"

Both prototypes share the defining features of a fortified compound:

1. **Hierarchy**: Primary keep is largest, placed at rear-center (50, 70)
2. **Enclosure**: Perimeter curtain wall with battlements
3. **Controlled entry**: At least one gate with approach axis to keep
4. **Role differentiation**: Primary > secondary > tertiary > ambient
5. **Courtyard**: Open interior space (< 7% building coverage)
6. **Defensive perimeter**: Towers positioned along or near walls

The difference is in *composition*, not *family*:
- Kaer Morhen = distributed defense (many entrances, spread towers)
- Oxenfurt Redoubt = concentrated defense (single gate, flanking towers)

## Reproduce

```bash
# Generate both
python -c "
import sys; sys.path.insert(0, 'training')
import level_layout
for name in ['kaer_morhen', 'oxenfurt_redoubt']:
    scene = level_layout.generate_level(
        style='medieval_keep', layout_type='organic',
        building_count=7, area_size=100, seed=42, min_gap=5.0,
        output_path=f'experiments/kaer_morhen/{name}.glb',
        graph_name=name)
    print(f'{name}: {len(scene.geometry)} meshes')
"
```

## Conclusion

Graph mode is a **reusable layout family**. Adding a new fortified compound requires
only a new JSON graph file -- no code changes to the generation pipeline. The spatial
character emerges from the graph's node roles, edge relationships, and enclosure config.
