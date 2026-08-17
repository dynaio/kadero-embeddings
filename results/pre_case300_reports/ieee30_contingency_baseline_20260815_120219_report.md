# Learning Report 01 — IEEE 30 Contingency Baseline

**Experiment ID:** `ieee30_contingency_baseline_20260815_120219`  
**Date:** 20260815_120219

## Goal
Generate N-1 contingency severity labels and evaluate how well the **unlearned** (theta=0) Kappa ranks the true severity of outages.

## Method
- For every line and transformer: take it out of service and run power flow.
- Severity proxy = max(0, max_loading - 100) + 10 × (number of overloaded lines).
- Failed power flows receive very high severity.

## Baseline Result (No Learning)
- **Spearman correlation (Kappa vs Severity):** `-0.3202`
- p-value: `4.1249e-02`
- Number of contingencies simulated: 41

## Top 8 most severe contingencies

|   from_bus |   to_bus | element   |   severity |   severity_rank |    kappa |   kappa_rank |
|-----------:|---------:|:----------|-----------:|----------------:|---------:|-------------:|
|          5 |        7 | line      |    73.4162 |               1 | 0.867177 |            8 |
|          7 |       27 | line      |    48.6678 |               2 | 0.327344 |           40 |
|         14 |       22 | line      |    40.9598 |               3 | 0.767864 |           16 |
|         27 |       26 | line      |    37.9645 |               4 | 0.711129 |           24 |
|          9 |       21 | line      |    34.3235 |               5 | 0.343565 |           39 |
|         11 |       12 | line      |    33.3755 |               6 | 1        |            2 |
|         22 |       23 | line      |    32.6116 |               7 | 0.688903 |           27 |
|          1 |        5 | line      |    32.0803 |               8 | 0.37011  |           38 |

## Interpretation
This number is the **baseline performance** of Kappa before any parameter learning.  
In the next step we will try to improve this correlation by learning one or two parameters.

## Files
- Contingency table: `/home/abdelkader/Code/Lynxo/results/tables/ieee30_contingency_baseline_20260815_120219_contingency.csv`
- Scatter plot: `/home/abdelkader/Code/Lynxo/results/figures/ieee30_contingency_baseline_20260815_120219_scatter.png`
- JSON: `/home/abdelkader/Code/Lynxo/results/logs/ieee30_contingency_baseline_20260815_120219_summary.json`
