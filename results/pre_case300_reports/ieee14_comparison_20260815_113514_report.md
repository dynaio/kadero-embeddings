# Verification Report 02 — IEEE 14-bus Metric Comparison

**Experiment ID:** `ieee14_comparison_20260815_113514`  
**Date:** 20260815_113514  
**Network:** IEEE 14-bus

## Goal
Compare the Kadero edge criticality (kappa) against classical graph metrics:
- Edge Betweenness Centrality
- Node Degree / Weighted Degree

## Main Quantitative Result
- **Spearman rank correlation (Kappa vs Edge Betweenness):** `0.2394`
- p-value: `3.0942e-01`

## Top 8 edges ranked by Kappa

|   from_bus |   to_bus | type   |    kappa |   kappa_rank |   betweenness |   betweenness_rank |
|-----------:|---------:|:-------|---------:|-------------:|--------------:|-------------------:|
|          6 |        7 | trafo  | 1        |            1 |     0.142857  |                  6 |
|          8 |        9 | line   | 0.931735 |            2 |     0.14652   |                  5 |
|          9 |       10 | line   | 0.842109 |            3 |     0.0989011 |                 12 |
|          0 |        1 | line   | 0.835259 |            4 |     0.0512821 |                 18 |
|          5 |       10 | line   | 0.833376 |            5 |     0.124542  |                  8 |
|          6 |        8 | trafo  | 0.789456 |            6 |     0.120879  |                  9 |
|          8 |       13 | line   | 0.774151 |            7 |     0.168498  |                  3 |
|          3 |        4 | line   | 0.732881 |            8 |     0.177656  |                  2 |

## Interpretation (preliminary)
- A correlation of 0.239 indicates that kappa and betweenness capture related but not identical notions of importance.
- Edges where the ranks differ significantly are the most interesting for further investigation (they show where the electrical/effective-resistance view diverges from pure topological betweenness).

## Generated Files
- Edges table: `/home/abdelkader/Code/Lynxo/results/tables/ieee14_comparison_20260815_113514_edges.csv`
- Nodes table: `/home/abdelkader/Code/Lynxo/results/tables/ieee14_comparison_20260815_113514_nodes.csv`
- Summary JSON: `/home/abdelkader/Code/Lynxo/results/logs/ieee14_comparison_20260815_113514_summary.json`
- Scatter plot: `/home/abdelkader/Code/Lynxo/results/figures/ieee14_comparison_20260815_113514_kappa_vs_betweenness.png`
- Rank comparison plot: `/home/abdelkader/Code/Lynxo/results/figures/ieee14_comparison_20260815_113514_rank_comparison.png`

## Next Steps
- Inspect edges where kappa rank and betweenness rank diverge the most.
- Move to a larger network (IEEE 30 or 118) once the pipeline is stable.
