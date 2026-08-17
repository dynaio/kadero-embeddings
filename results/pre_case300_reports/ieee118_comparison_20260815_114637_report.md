# Verification Report 04 — IEEE 118-bus Comparison

**Experiment ID:** `ieee118_comparison_20260815_114637`  
**Date:** 20260815_114637  
**Network:** IEEE 118-bus

## Goal
Final confirmation of the relationship between Kadero Kappa and classical Edge Betweenness on a significantly larger standard test case.

## Main Results
- **Spearman correlation:** `-0.1287`
- **p-value:** `8.5887e-02`
- **Buses:** 118
- **Edges:** 179
- **Edges with kappa > 0.99:** 9

## Top 10 edges ranked by Kappa

|   from_bus |   to_bus | type   |    kappa |   kappa_rank |   betweenness |   betweenness_rank |
|-----------:|---------:|:-------|---------:|-------------:|--------------:|-------------------:|
|          7 |        8 | line   | 1        |            1 |     0.0336086 |                 53 |
|         67 |      115 | trafo  | 1        |            2 |     0.0169492 |                100 |
|          8 |        9 | line   | 1        |            3 |     0.0169492 |                100 |
|         85 |       86 | trafo  | 1        |            4 |     0.0169492 |                100 |
|        109 |      111 | line   | 1        |            5 |     0.0169492 |                100 |
|         11 |      116 | line   | 1        |            6 |     0.0169492 |                100 |
|         84 |       85 | line   | 1        |            7 |     0.0336086 |                 53 |
|         70 |       72 | line   | 1        |            8 |     0.0169492 |                100 |
|        109 |      110 | line   | 1        |            9 |     0.0169492 |                100 |
|         24 |       25 | trafo  | 0.972803 |           10 |     0.0269212 |                 62 |

## Interpretation
If the correlation remains low and we continue to observe high-kappa edges with relatively low betweenness ranks, this strongly supports that Kappa provides a distinct criticality signal from pure topological betweenness.

## Files generated
- Edges table: `/home/abdelkader/Code/Lynxo/results/tables/ieee118_comparison_20260815_114637_edges.csv`
- Scatter plot: `/home/abdelkader/Code/Lynxo/results/figures/ieee118_comparison_20260815_114637_scatter.png`
- Rank plot: `/home/abdelkader/Code/Lynxo/results/figures/ieee118_comparison_20260815_114637_ranks.png`
- JSON summary: `/home/abdelkader/Code/Lynxo/results/logs/ieee118_comparison_20260815_114637_summary.json`
