#!/usr/bin/env python3
"""Precompute ChemBERTa embeddings for unique SMILES in a CSV.

Writes per-SMILES cached torch files into the FeatureEngineer cache dir.
This script is resumable and idempotent: it skips SMILES already cached.
"""
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from src.datasets.feature_engineer import FeatureEngineer


def main(csv_path, cache_dir, batch_size=64):
    csv = Path(csv_path)
    if not csv.exists():
        raise FileNotFoundError(f"CSV not found: {csv}")

    df = pd.read_csv(csv)
    smiles_cols = []
    for c in ['smiles_drug1', 'smiles_drug2', 'smiles1', 'smiles2']:
        if c in df.columns:
            smiles_cols.append(c)
    if not smiles_cols:
        raise ValueError('No SMILES columns found in CSV')

    unique_smiles = pd.unique(df[smiles_cols].values.ravel())
    unique_smiles = [s for s in unique_smiles if isinstance(s, str) and s]
    print(f'Found {len(unique_smiles)} unique SMILES')

    fe = FeatureEngineer(cache_dir=cache_dir)

    for sm in tqdm(unique_smiles, desc='Encoding SMILES'):
        try:
            _ = fe.encode_smiles(sm)
        except Exception as e:
            print(f'Failed to encode SMILES {sm}: {e}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--cache-dir', default='data/chemberta_cache')
    args = p.parse_args()
    main(args.csv, args.cache_dir)
