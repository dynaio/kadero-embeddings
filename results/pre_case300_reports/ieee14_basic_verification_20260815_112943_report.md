# Verification Report 01 — IEEE 14-bus Basic Check

**Experiment ID:** `ieee14_basic_verification_20260815_112943`  
**Date:** 20260815_112943  
**Network:** IEEE 14-bus  
**Status:** theta = 0 (physics-only baseline)

## Goal
Perform the first mathematical verification of the core operators on a standard test case before introducing any learning.

## What was computed
- Weighted graph from lines + transformers
- Combinatorial Laplacian and its pseudoinverse
- Edge criticality \(\kappa_e = y_e \cdot R_{eff}(e)\)
- Simple node weighted degree

## Key Observations
- Number of buses: 14
- Number of edges: 20
- Highest kappa edges (most critical according to this baseline):

|   from_bus |   to_bus | edge_id   | type   |      weight_y |        R_eff |    kappa |
|-----------:|---------:|:----------|:-------|--------------:|-------------:|---------:|
|          6 |        7 | trafo_3   | trafo  |     0.0573432 | 17.4389      | 1        |
|          8 |        9 | line_10   | line   | 25593.3       |  3.64054e-05 | 0.931735 |
|          9 |       10 | line_12   | line   | 11065.4       |  7.61029e-05 | 0.842109 |
|          0 |        1 | line_0    | line   |     0.0881258 |  9.47804     | 0.835259 |
|          5 |       10 | line_7    | line   | 10485.5       |  7.94791e-05 | 0.833376 |
|          6 |        8 | trafo_4   | trafo  |     0.091819  |  8.59796     | 0.789456 |
|          8 |       13 | line_11   | line   |  7735.81      |  0.000100074 | 0.774151 |
|          3 |        4 | line_6    | line   |     0.124208  |  5.90041     | 0.732881 |

## Files generated
- Edges table: `/home/abdelkader/Code/Lynxo/results/tables/ieee14_basic_verification_20260815_112943_edges.csv`
- Nodes table: `/home/abdelkader/Code/Lynxo/results/tables/ieee14_basic_verification_20260815_112943_nodes.csv`
- Summary JSON: `/home/abdelkader/Code/Lynxo/results/logs/ieee14_basic_verification_20260815_112943_summary.json`
- Network plot: `/home/abdelkader/Code/Lynxo/results/figures/ieee14_basic_verification_20260815_112943_network_kappa.png`
- Kappa histogram: `/home/abdelkader/Code/Lynxo/results/figures/ieee14_basic_verification_20260815_112943_kappa_hist.png`

## Notes
This is a **baseline physics-only** verification.  
No geographic channel, no operating-point channel, and no learned parameters were used yet.
Next steps will progressively activate more channels of the Kadero Embedding.
