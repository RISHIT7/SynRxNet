"""
Data loader module for drug synergy prediction.
Handles loading, validating, and cleaning drug combination datasets.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from typing import Optional, Tuple
import logging

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw(path: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.
    
    Parameters
    ----------
    path : str
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    logger.info(f"Loading dataset from {path}")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            logger.info(f"Successfully loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Could not load {path} with any of the attempted encodings")
    
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def validate_smiles(smiles: str) -> bool:
    """
    Validate a SMILES string using RDKit.
    
    Parameters
    ----------
    smiles : str
        SMILES string to validate
        
    Returns
    -------
    bool
        True if SMILES is valid, False otherwise
    """
    if pd.isna(smiles) or smiles == '' or not isinstance(smiles, str):
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def clean_dataset(df: pd.DataFrame, smiles_cols: list = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean dataset by validating SMILES strings and removing invalid entries.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to clean
    smiles_cols : list, optional
        List of column names containing SMILES strings.
        If None, will look for 'smilesString' column.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Cleaned dataset with valid SMILES
        - DataFrame with invalid entries
    """
    logger.info(f"Starting cleaning process. Initial rows: {len(df)}")
    
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Identify SMILES columns
    if smiles_cols is None:
        if 'smilesString' in df.columns:
            smiles_cols = ['smilesString']
        else:
            logger.warning("No SMILES columns specified and 'smilesString' not found")
            return df_clean, pd.DataFrame()
    
    # Track invalid entries
    invalid_mask = pd.Series([False] * len(df_clean), index=df_clean.index)
    invalid_reasons = []
    
    # Validate each SMILES column
    for col in smiles_cols:
        if col not in df_clean.columns:
            logger.warning(f"Column {col} not found in dataset")
            continue
            
        logger.info(f"Validating SMILES in column: {col}")
        
        # Check for missing values
        missing_mask = df_clean[col].isna() | (df_clean[col] == '')
        invalid_mask |= missing_mask
        
        # Validate non-missing SMILES
        valid_mask = df_clean[col].apply(validate_smiles)
        invalid_mask |= ~valid_mask
        
        logger.info(f"  - Missing/empty: {missing_mask.sum()}")
        logger.info(f"  - Invalid SMILES: {(~valid_mask & ~missing_mask).sum()}")
    
    # Separate valid and invalid entries
    df_valid = df_clean[~invalid_mask].copy()
    df_invalid = df_clean[invalid_mask].copy()
    
    logger.info(f"Cleaning complete:")
    logger.info(f"  - Valid entries: {len(df_valid)} ({100*len(df_valid)/len(df):.2f}%)")
    logger.info(f"  - Invalid entries: {len(df_invalid)} ({100*len(df_invalid)/len(df):.2f}%)")
    
    return df_valid, df_invalid


def merge_drug_info_with_combos(
    combos_df: pd.DataFrame,
    drug_info_df: pd.DataFrame,
    drug1_col: str = 'Drug1',
    drug2_col: str = 'Drug2'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge drug combination data with drug chemical information.
    
    Parameters
    ----------
    combos_df : pd.DataFrame
        Drug combinations dataset
    drug_info_df : pd.DataFrame
        Drug chemical information with SMILES
    drug1_col : str
        Column name for first drug
    drug2_col : str
        Column name for second drug
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Merged dataset with SMILES for both drugs
        - Entries that couldn't be merged (missing drug info)
    """
    logger.info("Merging drug combinations with chemical information")
    
    # Merge for Drug1
    merged = combos_df.merge(
        drug_info_df[['drugName', 'smilesString']],
        left_on=drug1_col,
        right_on='drugName',
        how='left',
        suffixes=('', '_drug1')
    )
    merged = merged.rename(columns={'smilesString': 'smiles_drug1'})
    merged = merged.drop(columns=['drugName'], errors='ignore')
    
    # Merge for Drug2
    merged = merged.merge(
        drug_info_df[['drugName', 'smilesString']],
        left_on=drug2_col,
        right_on='drugName',
        how='left',
        suffixes=('', '_drug2')
    )
    merged = merged.rename(columns={'smilesString': 'smiles_drug2'})
    merged = merged.drop(columns=['drugName'], errors='ignore')
    
    # Identify missing entries
    missing_mask = merged['smiles_drug1'].isna() | merged['smiles_drug2'].isna()
    df_complete = merged[~missing_mask].copy()
    df_missing = merged[missing_mask].copy()
    
    logger.info(f"Merge complete:")
    logger.info(f"  - Complete entries: {len(df_complete)}")
    logger.info(f"  - Missing drug info: {len(df_missing)}")
    
    return df_complete, df_missing


if __name__ == "__main__":
    # Example usage
    print("Data loader module loaded successfully")
    print("\nAvailable functions:")
    print("  - load_raw(path): Load CSV dataset")
    print("  - validate_smiles(smiles): Validate a SMILES string")
    print("  - clean_dataset(df, smiles_cols): Clean dataset by validating SMILES")
    print("  - merge_drug_info_with_combos(combos_df, drug_info_df): Merge drug combos with SMILES")
