# LevelSmith 风格×布局稳定性验证报告

## 测试概况

- **总测试组合数**: 30
- **成功组合数**: 30
- **整体成功率**: 100.0%
- **测试时间**: 2026-04-03 21:05:45

## 布局性能分析

| 布局类型 | 成功率 | 平均生成时间(s) |
|----------|--------|----------------|
| street | 100.0% | 0.5 |
| grid | 100.0% | 0.6 |
| plaza | 100.0% | 0.5 |
| random | 100.0% | 0.2 |
| organic | 100.0% | 0.2 |

## 风格性能分析

| 风格 | 成功率 | 平均面数 |
|------|--------|----------|
| japanese | 100.0% | 4455 |
| medieval_keep | 100.0% | 7416 |
| industrial | 100.0% | 4720 |
| fantasy_palace | 100.0% | 8142 |
| horror | 100.0% | 3924 |
| desert | 100.0% | 3143 |

## 推荐用于README展示的组合

| 风格 | 布局 | 面数 | 理论兼容性 | 生成时间(s) |
|------|------|------|------------|------------|
| medieval_keep | plaza | 9668 | theoretically_good | 0.8 |
| fantasy_palace | plaza | 8696 | theoretically_good | 0.9 |
| industrial | grid | 7556 | theoretically_good | 0.6 |
| industrial | street | 7032 | theoretically_good | 0.5 |
| japanese | street | 5600 | theoretically_good | 0.2 |
| desert | plaza | 3944 | theoretically_good | 0.4 |
| medieval_keep | organic | 3876 | theoretically_good | 0.4 |
| horror | random | 2136 | theoretically_good | 0.2 |
