# Dataset Splitting Strategy

**Date:** 30 November 2025

To ensure robust model evaluation, we employ three different splitting strategies. This allows us to test the model's generalization capabilities under different scenarios.

## 1. Stratified Random Split (Standard)
- **Method:** Randomly split drug-cell pairs into Train (80%), Validation (10%), and Test (10%).
- **Goal:** Test interpolation capability. The model has likely seen both the drugs and cell lines in the test set, but not these specific combinations.
- **Stratification:** (Optional) Stratified by cell line or synergy score bins to ensure balanced distributions.

## 2. Leave-One-Drug-Out (LODO)
- **Method:** Select 10% of drugs to be exclusively in the Test set. Any pair containing these drugs is removed from Train/Val.
- **Goal:** Test extrapolation to **unseen drugs**. This simulates the real-world scenario of screening a new chemical compound.
- **Implementation:**
    - Identify all unique drugs.
    - Randomly sample 10% of drugs as `test_drugs`.
    - Test Set = All pairs where Drug1 OR Drug2 is in `test_drugs`.
    - Train/Val = Remaining pairs.

## 3. Leave-One-Cell-Line-Out (LOCO)
- **Method:** Select 10% of cell lines to be exclusively in the Test set.
- **Goal:** Test extrapolation to **unseen biological contexts**. This simulates predicting drug efficacy on a new patient or tissue type.
- **Implementation:**
    - Identify all unique cell lines.
    - Randomly sample 10% as `test_cells`.
    - Test Set = All pairs where Cell Line is in `test_cells`.
    - Train/Val = Remaining pairs.

## Output Files
Splits are saved in `data/processed/splits/`:
- `random_train.csv`, `random_val.csv`, `random_test.csv`
- `lodo_train.csv`, `lodo_val.csv`, `lodo_test.csv`
- `loco_train.csv`, `loco_val.csv`, `loco_test.csv`
