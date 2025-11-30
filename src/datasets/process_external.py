"""
Helpers to process external datasets (ONeil, CCLE) into the project's canonical format.

Functions:
 - process_oneil(raw_dir, out_path): produce cleaned_oneil.csv similar to cleaned_drugcomb.csv
 - process_ccle(raw_dir, out_path, n_components): produce CCLE cell-line features CSV

This module re-uses the project's data loader / validator utilities.
"""
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from typing import Optional

from .data_loader import validate_smiles

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def process_oneil(raw_dir: str, out_path: str, smiles_file: str = 'smiles.csv', labels_file: str = 'labels.csv') -> pd.DataFrame:
    """Process O'Neil dataset and return cleaned dataframe.

    Steps:
    - Read `labels.csv` which contains columns ['drug_a_name','drug_b_name','cell_line','synergy',...]
    - Read `smiles.csv` which maps drug name -> SMILES
    - Merge SMILES for drug A and B
    - Validate SMILES and drop invalid rows
    - Clip/filter extreme synergy values (same heuristic as DrugComb)
    - Save a cleaned CSV with the canonical columns used by the project

    Returns the cleaned DataFrame.
    """
    raw_dir = Path(raw_dir)
    labels_path = raw_dir / labels_file
    smiles_path = raw_dir / smiles_file

    logger.info(f"Reading O'Neil labels from {labels_path}")
    df_labels = pd.read_csv(labels_path, index_col=None)

    logger.info(f"Reading O'Neil smiles from {smiles_path}")
    # smiles.csv appears to be two columns without header: name,smiles
    df_smiles = pd.read_csv(smiles_path, header=None, names=['drugName', 'smilesString'])

    # Normalize column names and create canonical columns
    df = df_labels.rename(columns={
        'drug_a_name': 'Drug1',
        'drug_b_name': 'Drug2',
        'cell_line': 'Cell line',
        'synergy': 'ZIP'
    }).copy()

    # Merge smiles for Drug1 and Drug2
    df = df.merge(df_smiles, left_on='Drug1', right_on='drugName', how='left')
    df = df.rename(columns={'smilesString': 'smiles_drug1'})
    df = df.drop(columns=['drugName'], errors='ignore')

    df = df.merge(df_smiles, left_on='Drug2', right_on='drugName', how='left')
    df = df.rename(columns={'smilesString': 'smiles_drug2'})
    df = df.drop(columns=['drugName'], errors='ignore')

    # Basic cleaning: remove rows with missing SMILES
    missing_smiles = df['smiles_drug1'].isna() | df['smiles_drug2'].isna()
    if missing_smiles.any():
        logger.warning(f"Dropping {missing_smiles.sum()} rows with missing SMILES after merge")
        df = df[~missing_smiles].copy()

    # Validate SMILES using existing validator
    valid_mask = df['smiles_drug1'].apply(validate_smiles) & df['smiles_drug2'].apply(validate_smiles)
    n_invalid = (~valid_mask).sum()
    if n_invalid:
        logger.warning(f"Dropping {n_invalid} rows with invalid SMILES")
        df = df[valid_mask].copy()

    # Filter extreme ZIP / synergy values (same heuristic used in notebook)
    df = df[(df['ZIP'] > -1e2) & (df['ZIP'] < 100)].reset_index(drop=True)

    # Add missing canonical columns so downstream code can use this CSV
    out_cols = ['Drug1', 'Drug2', 'Cell line', 'ZIP', 'Bliss', 'Loewe', 'HSA', 'smiles_drug1', 'smiles_drug2']
    df_out = pd.DataFrame(columns=out_cols)
    df_out['Drug1'] = df['Drug1'].astype(str)
    df_out['Drug2'] = df['Drug2'].astype(str)
    df_out['Cell line'] = df['Cell line'].astype(str)
    df_out['ZIP'] = df['ZIP'].astype(float)
    # O'Neil doesn't provide Bliss/Loewe/HSA columns; keep as NaN
    df_out['Bliss'] = np.nan
    df_out['Loewe'] = np.nan
    df_out['HSA'] = np.nan
    df_out['smiles_drug1'] = df['smiles_drug1'].astype(str)
    df_out['smiles_drug2'] = df['smiles_drug2'].astype(str)

    df_out.insert(0, 'ID', range(1, len(df_out) + 1))

    # Save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info(f"Saved cleaned O'Neil dataset to {out_path} ({len(df_out)} rows)")

    return df_out


def process_ccle(raw_dir: str, out_path: str, n_components: int = 100, sample_cell_lines: Optional[list] = None) -> pd.DataFrame:
    """Process CCLE expression data to produce a per-cell-line feature matrix.

    Steps:
    - Load CCLE TPM (RSEM) file (gzip) into a DataFrame (genes x samples)
    - Transpose to samples x genes
    - Optionally restrict to `sample_cell_lines` (list of cell line names)
    - Log-transform (log1p), scale (StandardScaler), then reduce to `n_components` by PCA
    - Save a CSV with index=cell line name and columns PC1..PCn

    Returns the cell-line features DataFrame.
    """
    raw_dir = Path(raw_dir)
    tpm_path = None
    # Try to find a tpm file in the directory
    for p in raw_dir.iterdir():
        if 'tpm' in p.name.lower() and p.suffix in ['.gz', '.txt', '.tsv']:
            tpm_path = p
            break

    if tpm_path is None:
        raise FileNotFoundError(f"Could not find TPM file in {raw_dir}")

    logger.info(f"Loading CCLE TPM from {tpm_path}")
    # read with pandas (support gzip)
    df_tpm = pd.read_csv(tpm_path, sep='\t', compression='gzip' if tpm_path.suffix == '.gz' else None, index_col=0)

    # If TPM file has gene_id and gene_name columns, try to collapse to gene symbols
    # Many CCLE files have 'gene_id' as first column and sample columns afterwards.
    # After reading with index_col=0, columns should be sample names

    # Transpose: samples x genes
    df_cells = df_tpm.T

    # Normalize cell line names: upper-case, remove spaces
    df_cells.index = df_cells.index.str.upper().str.replace(' ', '').str.replace('-', '').str.strip()

    if sample_cell_lines is not None:
        target = [s.upper().replace(' ', '').replace('-', '').strip() for s in sample_cell_lines]
        available = set(df_cells.index)
        intersect = [s for s in target if s in available]
        if not intersect:
            logger.warning("No overlap between requested cell lines and CCLE TPM sample names. Saving full matrix.")
        else:
            df_cells = df_cells.loc[intersect]

    logger.info(f"Computing log1p + scaling and PCA (n_components={n_components}) on {len(df_cells)} cell lines")

    # log1p transform
    X = np.log1p(df_cells.values.astype(float))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, Xs.shape[1]))
    Xp = pca.fit_transform(Xs)

    cols = [f'PC{i+1}' for i in range(Xp.shape[1])]
    df_feats = pd.DataFrame(Xp, index=df_cells.index, columns=cols)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feats.to_csv(out_path)
    logger.info(f"Saved CCLE cell-line features to {out_path} ({df_feats.shape})")

    return df_feats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process ONeil and CCLE raw data into processed format')
    sub = parser.add_subparsers(dest='cmd')

    p1 = sub.add_parser('oneil')
    p1.add_argument('--raw_dir', default='data/raw/ONeil')
    p1.add_argument('--out', default='data/processed/cleaned_oneil.csv')

    p2 = sub.add_parser('ccle')
    p2.add_argument('--raw_dir', default='data/raw/CCLE')
    p2.add_argument('--out', default='data/processed/ccle_cellline_features.csv')
    p2.add_argument('--n_components', type=int, default=100)

    args = parser.parse_args()

    if args.cmd == 'oneil':
        process_oneil(args.raw_dir, args.out)
    elif args.cmd == 'ccle':
        process_ccle(args.raw_dir, args.out, n_components=args.n_components)
    else:
        parser.print_help()
