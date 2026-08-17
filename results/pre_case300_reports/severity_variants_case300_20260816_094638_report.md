# Severity Design Check — case300

**Experiment:** `severity_variants_case300_20260816_094638`  
**Date:** 20260816_094638

## Goal
Test whether the choice of severity definition strongly changes the observed correlation with baseline Kappa, especially on the non-bridge subset.

## Severity Variants
- **S1_island_heavy**: Original style (strong penalty for islanding)
- **S2_thermal_heavy**: More weight on thermal overloads
- **S3_no_big_island**: Excludes contingencies that create large islands
- **S4_balanced**: Mixed penalties

## Results

| severity         |   rho_all |     p_all |   n_all |   rho_nonbridge |   p_nonbridge |   n_nonbridge |
|:-----------------|----------:|----------:|--------:|----------------:|--------------:|--------------:|
| S1_island_heavy  |  0.112655 | 0.0223599 |     411 |        0.206131 |   0.000204888 |           320 |
| S2_thermal_heavy |  0.112655 | 0.0223599 |     411 |        0.206131 |   0.000204888 |           320 |
| S3_no_big_island |  0.112655 | 0.0223599 |     411 |        0.206131 |   0.000204888 |           320 |
| S4_balanced      |  0.112655 | 0.0223599 |     411 |        0.206131 |   0.000204888 |           320 |

## Interpretation Guide
- If non-bridge correlations stay low and similar across definitions → the limited signal is probably not just an artifact of one severity formula.
- If one definition gives clearly higher non-bridge correlation → we should adopt it before testing new channels.
- If all non-bridge correlations remain weak → richer channels (redundancy, operating-point) become even more important to test next.

## Files
- Correlations: `/home/abdelkader/Code/Lynxo/results/tables/severity_variants_case300_20260816_094638_correlations.csv`
- Full data: `/home/abdelkader/Code/Lynxo/results/tables/severity_variants_case300_20260816_094638_full_data.csv`
- Figure: `/home/abdelkader/Code/Lynxo/results/figures/severity_variants_case300_20260816_094638_severity_comparison.png`
- JSON: `/home/abdelkader/Code/Lynxo/results/logs/severity_variants_case300_20260816_094638_summary.json`
