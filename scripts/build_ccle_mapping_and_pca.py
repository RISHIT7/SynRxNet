#!/usr/bin/env python3
"""
Build cell line mapping between cleaned_drugcomb and CCLE expression columns,
then compute PCA(100) on log2(TPM+1) expression per CCLE cell line.

Outputs (written to data/processed/):
- cell_line_mapping.csv          : original_name,ccle_name,method
- unmatched_cell_lines.txt      : original names not matched automatically
- ccle_genes_selected_log2_tpm.csv : genes x selected CCLE columns (log2(TPM+1))
- ccle_cell_line_pca_100.csv    : PCA features (index = CCLE column names)

Usage: run from repo or directly: python scripts/build_ccle_mapping_and_pca.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Robustly locate the project dir by searching parent directories for the processed cleaned_drugcomb
curr = Path(__file__).resolve()
project_dir = None
for p in curr.parents:
    if (p / 'data' / 'processed' / 'cleaned_drugcomb.csv').exists():
        project_dir = p
        break
# fallback: use a reasonable parent
if project_dir is None:
    project_dir = Path(__file__).resolve().parents[2]
print('Project dir:', project_dir)

combos_path = project_dir / 'data' / 'processed' / 'cleaned_drugcomb.csv'
oneil_path = project_dir / 'data' / 'processed' / 'cleaned_oneil.csv'
expr_path_gz = project_dir / 'data' / 'raw' / 'CCLE' / 'CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz'
expr_path = project_dir / 'data' / 'raw' / 'CCLE' / 'CCLE_RNAseq_rsem_genes_tpm_20180929.txt'

out_dir = project_dir / 'data' / 'processed'
out_dir.mkdir(parents=True, exist_ok=True)
mapping_out = out_dir / 'cell_line_mapping.csv'
unmatched_out = out_dir / 'unmatched_cell_lines.txt'
ccle_genes_out = out_dir / 'ccle_genes_selected_log2_tpm.csv'
pca_out = out_dir / 'ccle_cell_line_pca_100.csv'

print('Loading cleaned combos from:', combos_path)
combos = pd.read_csv(combos_path)
# If an O'Neil cleaned file exists, append it to create a combined dataset
if oneil_path.exists():
    print('Found cleaned O\'Neil file at:', oneil_path)
    oneil = pd.read_csv(oneil_path)
    # attempt to align column names: prefer 'Cell line' column
    if 'Cell line' not in oneil.columns and 'Cell_line' in oneil.columns:
        oneil = oneil.rename(columns={'Cell_line': 'Cell line'})
    # concatenate rows that are not duplicates
    combos_combined = pd.concat([combos, oneil], ignore_index=True, sort=False)
    # optionally drop exact duplicate rows
    combos_combined = combos_combined.drop_duplicates()
    combined_out = out_dir / 'cleaned_combined.csv'
    combos_combined.to_csv(combined_out, index=False)
    print('Wrote combined cleaned dataset to:', combined_out)
    combos_for_cells = combos_combined
else:
    combos_for_cells = combos

my_cells = sorted(pd.Series(combos_for_cells['Cell line'].dropna().unique()))
print(f'Found {len(my_cells)} unique cleaned "Cell line" names (examples):', my_cells[:10])

# Load CCLE expression - prefer gz if exists
if expr_path_gz.exists():
    expr_path_to_load = expr_path_gz
elif expr_path.exists():
    expr_path_to_load = expr_path
else:
    raise FileNotFoundError('CCLE expression file not found in expected paths')

print('Loading CCLE expression (this may take a minute):', expr_path_to_load)
# Read with pandas; set index_col=0 so gene_id becomes index
expr = pd.read_csv(expr_path_to_load, sep='\t', index_col=0)
print('Expression shape:', expr.shape)
print('First columns:', list(expr.columns[:6]))

# Determine which initial columns are non-sample columns. Common CCLE file has a 'transcript_ids' column
cols = list(expr.columns)
non_sample_cols = []
for c in cols[:4]:
    if re.search(r'transcript', c, flags=re.I) or re.search(r'gene', c, flags=re.I) and c.lower() != 'gene_id':
        non_sample_cols.append(c)

# Heuristic: if first column name contains 'transcript' then skip first column
if len(cols) >= 2 and re.search(r'transcript', cols[0], flags=re.I):
    ccle_cells = cols[1:]
else:
    # fallback to skipping first two if obvious
    ccle_cells = cols[2:] if len(cols) > 2 else cols[1:]

print(f'Identified {len(ccle_cells)} CCLE sample columns (examples):', ccle_cells[:10])

# Build mapping with simple heuristics
mapping_rows = []
all_ccle_cols = expr.columns.tolist()
for c in my_cells:
    matched = ''
    method = ''
    # exact match
    if c in all_ccle_cols:
        matched = c
        method = 'exact'
    else:
        # case-insensitive exact
        ci_matches = [col for col in all_ccle_cols if col.lower() == c.lower()]
        if ci_matches:
            matched = ci_matches[0]
            method = 'ci_exact'
        else:
            # startswith c + underscore
            sw = [col for col in all_ccle_cols if col.startswith(c + '_')]
            if len(sw) == 1:
                matched = sw[0]
                method = 'starts_with_underscore'
            elif len(sw) > 1:
                # prefer one with common tissue suffix length small
                matched = sorted(sw, key=lambda x: len(x))[0]
                method = 'starts_with_underscore_multi_pick_shortest'
            else:
                # startswith c
                sw2 = [col for col in all_ccle_cols if col.startswith(c)]
                if len(sw2) == 1:
                    matched = sw2[0]
                    method = 'starts_with'
                elif len(sw2) > 1:
                    matched = sorted(sw2, key=lambda x: len(x))[0]
                    method = 'starts_with_multi_pick_shortest'
                else:
                    # contains
                    cont = [col for col in all_ccle_cols if c.lower() in col.lower()]
                    if len(cont) == 1:
                        matched = cont[0]
                        method = 'contains'
                    elif len(cont) > 1:
                        # try to prefer full token matches: e.g., 'ZR751' matches 'ZR751_BREAST'
                        token_matches = [col for col in cont if re.search(r'\b' + re.escape(c) + r'\b', col, flags=re.I)]
                        if len(token_matches) == 1:
                            matched = token_matches[0]
                            method = 'contains_token'
                        else:
                            matched = ''
                            method = 'multiple_candidates'
                    else:
                        matched = ''
                        method = 'no_match'

    mapping_rows.append({'original_name': c, 'ccle_name': matched, 'method': method})

mapping = pd.DataFrame(mapping_rows)
mapping.to_csv(mapping_out, index=False)
print('Wrote mapping to:', mapping_out)

# Save unmatched
unmatched = mapping[mapping['ccle_name'] == '']['original_name'].tolist()
with open(unmatched_out, 'w') as f:
    for u in unmatched:
        f.write(u + '\n')
print(f'{len(unmatched)} unmatched original names written to', unmatched_out)

# Select matched ccle columns that actually exist in expr
matched_ccle = [m for m in mapping['ccle_name'].tolist() if m]
available_cols = [c for c in matched_ccle if c in expr.columns]
missing_mapped = [c for c in matched_ccle if c not in expr.columns]
if missing_mapped:
    print('Warning: some mapped CCLE names are not present in expression columns:', missing_mapped)

if len(available_cols) == 0:
    print('No available CCLE columns to proceed. Exiting.')
    sys.exit(1)

print('Using', len(available_cols), 'CCLE columns for PCA (examples):', available_cols[:10])

# Subset expression to these columns
ccle_sub = expr[available_cols]
print('Subset expression shape (genes x cells):', ccle_sub.shape)

# Log2(TPM+1)
ccle_log = np.log2(ccle_sub + 1)
# Save genes x selected_cells log2 TPM
ccle_log.to_csv(ccle_genes_out)
print('Wrote log2 TPM subset to:', ccle_genes_out)

# Transpose to cells x genes for PCA
X = ccle_log.T
print('Transposed shape (cells x genes):', X.shape)

# Standardize then PCA
n_components = min(100, X.shape[0], X.shape[1])
print('Computing PCA with n_components =', n_components)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
pca = PCA(n_components=n_components, random_state=0)
Xp = pca.fit_transform(Xs)
cols_pca = [f'PC{i+1}' for i in range(Xp.shape[1])]

pca_df = pd.DataFrame(Xp, index=X.index, columns=cols_pca)
pca_df.to_csv(pca_out)
print('Wrote PCA to:', pca_out)

print('Done. Summary:')
print(' - mapping:', mapping_out)
print(' - unmatched list:', unmatched_out)
print(' - log2 TPM subset:', ccle_genes_out)
print(' - PCA:', pca_out)
