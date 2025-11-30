# EDA Report

**Date:** 30 November 2025
**Dataset:** DrugComb (Cleaned)

## 1. Summary Statistics
## 1. Summary Statistics
- **Total Samples:** 57918
- **Unique Drugs:** 2520
- **Unique Cell Lines:** 120
- **Sparsity:** ~0.19% (57918 / (2520*120*2) possible drug-cell pairs)

## 2. Synergy Score Distributions
## 2. Synergy Score Distributions
- **ZIP:** Highly skewed, with extreme outliers (range: -216,626 to 1,781,336). After filtering, most scores are between -100 and 100, centered around 0.
- **Bliss:** Similar to ZIP, extreme outliers (range: -221,232 to 1,689,573), but most scores are near 0 after filtering.
- **Loewe:** Range: -9,663 to 2,696, less extreme, mostly centered near 0.
- **HSA:** Range: -11,718 to 1,075, less extreme, mostly centered near 0.

*Key Observation:* Most synergy scores are centered around 0, indicating no effect. Positive tails indicate synergy. Outliers were removed for analysis.

## 3. Data Coverage
## 3. Data Coverage
- **Drugs:** Top drugs are 5-FU, MK-4541, MRK-003. There is a long tail of infrequent drugs.
- **Cell Lines:** Top cell lines are KBM-7, DIPG25, TC-71, DD2, HB3. Distribution is imbalanced, with some cell lines much more frequent.

## 4. Missingness
## 4. Missingness
- No missing values in synergy scores or SMILES after cleaning.

## 5. Feature Correlations
## 5. Feature Correlations
- Simple descriptors (MolWt, LogP, TPSA) show weak correlation with synergy scores. No strong linear relationship observed in heatmap.

## 6. Preprocessing Decisions
## 6. Preprocessing Decisions
- **Outliers:** Extreme synergy scores (ZIP, Bliss) were removed (outside [-100, 100]).
- **Filtering:** Drugs with invalid SMILES were filtered out (0 missing after cleaning).
- **Splitting Strategy:** To be decided in Day 8.

---

## Advanced EDA Suggestions (No Code)
1. **Multivariate Analysis:** Use PCA/t-SNE/UMAP to visualize high-dimensional drug features and synergy scores.
2. **Clustering:** Cluster drugs/cell lines by feature similarity and synergy profiles (e.g., hierarchical, k-means, DBSCAN).
3. **Network Analysis:** Build drug-cell-synergy networks; analyze communities, centrality, and network motifs.
4. **Feature Importance:** Use tree-based models (Random Forest, XGBoost) to rank feature importance for synergy prediction.
5. **Interaction Effects:** Explore pairwise and higher-order interactions between drug features and cell line features.
6. **Subgroup Analysis:** Stratify EDA by cancer type, drug class, or cell line lineage to find context-specific patterns.
7. **Batch Effects:** Check for technical/experimental batch effects using metadata, visualize with boxplots/ANOVA.
8. **Rare Event Analysis:** Investigate rare drugs/cell lines and their synergy profiles; use oversampling or anomaly detection.
9. **Temporal Trends:** If time-series data exists, analyze changes in synergy over time or experimental batches.
10. **External Validation:** Compare synergy distributions and feature patterns to external datasets (e.g., CCLE, DeepSynergy).

These approaches will reveal deeper structure, confounders, and actionable insights for modeling and interpretation.
