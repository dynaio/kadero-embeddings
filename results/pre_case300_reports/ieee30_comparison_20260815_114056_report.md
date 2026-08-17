# Verification Report 03 — IEEE 30-bus Comparison

**Experiment ID:** `ieee30_comparison_20260815_114056`  
**Date:** 20260815_114056  
**Network:** IEEE 30-bus

## Goal
Confirm whether the weak correlation observed on IEEE 14-bus between Kappa and classical Edge Betweenness persists on a larger network.

## Main Result
- **Spearman correlation (Kappa vs Edge Betweenness):** `0.0581`
- p-value: `7.1817e-01`
- Number of buses: 30
- Number of edges: 41

## Top 10 edges by Kappa

|   from_bus |   to_bus | type   |    kappa |   kappa_rank |   betweenness |   betweenness_rank |
|-----------:|---------:|:-------|---------:|-------------:|--------------:|-------------------:|
|          8 |       10 | line   | 1        |            1 |     0.0666667 |                 22 |
|         11 |       12 | line   | 1        |            2 |     0.0666667 |                 22 |
|         24 |       25 | line   | 1        |            3 |     0.0666667 |                 22 |
|         18 |       19 | line   | 0.919564 |            4 |     0.0722222 |                 18 |
|         20 |       21 | line   | 0.904049 |            5 |     0.0180077 |                 38 |
|          2 |        3 | line   | 0.893831 |            6 |     0.0712644 |                 20 |
|          9 |       16 | line   | 0.887151 |            7 |     0.0871648 |                 15 |
|          5 |        7 | line   | 0.867177 |            8 |     0.0509579 |                 29 |
|         17 |       18 | line   | 0.848778 |            9 |     0.0469349 |                 34 |
|          0 |        1 | line   | 0.837144 |           10 |     0.0574713 |                 27 |

## Interpretation
The correlation remains relatively low/moderate. This continues to support the idea that Kappa captures a different notion of criticality (more electrical / effective-resistance oriented) compared to pure topological betweenness.

## Files
- Edges table: `/home/abdelkader/Code/Lynxo/results/tables/ieee30_comparison_20260815_114056_edges.csv`
- Scatter plot: `/home/abdelkader/Code/Lynxo/results/figures/ieee30_comparison_20260815_114056_scatter.png`
- Rank plot: `/home/abdelkader/Code/Lynxo/results/figures/ieee30_comparison_20260815_114056_ranks.png`
- JSON summary: `/home/abdelkader/Code/Lynxo/results/logs/ieee30_comparison_20260815_114056_summary.json`
