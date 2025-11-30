"""
Feature engineering module for molecular descriptors and fingerprints.
Computes RDKit descriptors and Morgan fingerprints from SMILES strings.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import RDLogger
from typing import Optional, List, Dict
import logging

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rdkit_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    """
    Compute RDKit 2D molecular descriptors from a SMILES string.
    
    Parameters
    ----------
    smiles : str
        SMILES string
        
    Returns
    -------
    Optional[Dict[str, float]]
        Dictionary of descriptor names and values, or None if SMILES is invalid
    """
    if pd.isna(smiles) or smiles == '':
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = {
            # Basic molecular properties
            'MolWt': Descriptors.MolWt(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'TPSA': Descriptors.TPSA(mol),
            
            # Atom and bond counts
            'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
            'NumAtoms': mol.GetNumAtoms(),
            'NumBonds': mol.GetNumBonds(),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            
            # Ring information
            'RingCount': Descriptors.RingCount(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            
            # Electronic properties
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            
            # Complexity measures
            'BertzCT': Descriptors.BertzCT(mol),
            'HallKierAlpha': Descriptors.HallKierAlpha(mol),
            'Kappa1': Descriptors.Kappa1(mol),
            'Kappa2': Descriptors.Kappa2(mol),
            'Kappa3': Descriptors.Kappa3(mol),
            
            # Refractivity
            'MolMR': Descriptors.MolMR(mol),
            
            # Chi indices
            'Chi0': Descriptors.Chi0(mol),
            'Chi1': Descriptors.Chi1(mol),
            'Chi0v': Descriptors.Chi0v(mol),
            'Chi1v': Descriptors.Chi1v(mol),
            
            # Labute ASA
            'LabuteASA': Descriptors.LabuteASA(mol),
            
            # Fraction of SP3 carbons
            'FractionCsp3': Descriptors.FractionCSP3(mol),
        }
        
        return descriptors
        
    except Exception as e:
        logger.debug(f"Error computing descriptors for {smiles}: {e}")
        return None


def compute_morgan(
    smiles: str,
    radius: int = 2,
    nBits: int = 2048,
    useChirality: bool = False
) -> Optional[np.ndarray]:
    """
    Compute Morgan (circular) fingerprint from a SMILES string.
    
    Parameters
    ----------
    smiles : str
        SMILES string
    radius : int, default=2
        Radius of the Morgan fingerprint
    nBits : int, default=2048
        Number of bits in the fingerprint
    useChirality : bool, default=False
        Whether to include chirality information
        
    Returns
    -------
    Optional[np.ndarray]
        Binary fingerprint array, or None if SMILES is invalid
    """
    if pd.isna(smiles) or smiles == '':
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=nBits,
            useChirality=useChirality
        )
        
        # Convert to numpy array
        arr = np.zeros((nBits,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        
        return arr
        
    except Exception as e:
        logger.debug(f"Error computing Morgan fingerprint for {smiles}: {e}")
        return None


def compute_features_for_dataframe(
    df: pd.DataFrame,
    smiles_col: str = 'smilesString',
    include_descriptors: bool = True,
    include_morgan: bool = True,
    morgan_radius: int = 2,
    morgan_nbits: int = 2048
) -> pd.DataFrame:
    """
    Compute molecular features for all molecules in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing SMILES strings
    smiles_col : str
        Name of the column containing SMILES
    include_descriptors : bool
        Whether to include RDKit descriptors
    include_morgan : bool
        Whether to include Morgan fingerprints
    morgan_radius : int
        Radius for Morgan fingerprints
    morgan_nbits : int
        Number of bits for Morgan fingerprints
        
    Returns
    -------
    pd.DataFrame
        DataFrame with computed features
    """
    logger.info(f"Computing features for {len(df)} molecules")
    
    results = []
    
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        feature_dict = {'index': idx}
        
        # Compute descriptors
        if include_descriptors:
            descriptors = compute_rdkit_descriptors(smiles)
            if descriptors is not None:
                feature_dict.update(descriptors)
        
        # Compute Morgan fingerprint
        if include_morgan:
            morgan_fp = compute_morgan(smiles, radius=morgan_radius, nBits=morgan_nbits)
            if morgan_fp is not None:
                # Add fingerprint bits as separate columns
                for i, bit in enumerate(morgan_fp):
                    feature_dict[f'morgan_bit_{i}'] = bit
        
        results.append(feature_dict)
    
    features_df = pd.DataFrame(results)
    features_df = features_df.set_index('index')
    
    logger.info(f"Feature computation complete. Generated {len(features_df.columns)} features")
    
    return features_df


def compute_drug_pair_features(
    smiles1: str,
    smiles2: str,
    radius: int = 2,
    nBits: int = 2048
) -> Optional[Dict[str, any]]:
    """
    Compute features for a drug pair including individual and combined features.
    
    Parameters
    ----------
    smiles1 : str
        SMILES string for first drug
    smiles2 : str
        SMILES string for second drug
    radius : int
        Radius for Morgan fingerprints
    nBits : int
        Number of bits for Morgan fingerprints
        
    Returns
    -------
    Optional[Dict[str, any]]
        Dictionary with individual descriptors and concatenated fingerprints
    """
    # Compute descriptors for both drugs
    desc1 = compute_rdkit_descriptors(smiles1)
    desc2 = compute_rdkit_descriptors(smiles2)
    
    if desc1 is None or desc2 is None:
        return None
    
    # Compute Morgan fingerprints
    fp1 = compute_morgan(smiles1, radius=radius, nBits=nBits)
    fp2 = compute_morgan(smiles2, radius=radius, nBits=nBits)
    
    if fp1 is None or fp2 is None:
        return None
    
    # Combine features
    features = {}
    
    # Add descriptors with prefixes
    for key, val in desc1.items():
        features[f'drug1_{key}'] = val
    for key, val in desc2.items():
        features[f'drug2_{key}'] = val
    
    # Concatenate fingerprints
    features['morgan_concat'] = np.concatenate([fp1, fp2])
    
    return features


if __name__ == "__main__":
    # Example usage
    print("Feature engineering module loaded successfully")
    print("\nAvailable functions:")
    print("  - compute_rdkit_descriptors(smiles): Compute 2D molecular descriptors")
    print("  - compute_morgan(smiles, radius=2, nBits=2048): Compute Morgan fingerprint")
    print("  - compute_features_for_dataframe(df, smiles_col): Compute features for DataFrame")
    print("  - compute_drug_pair_features(smiles1, smiles2): Compute features for drug pair")
    
    # Test with example SMILES
    example_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    print(f"\n\nTesting with aspirin: {example_smiles}")
    
    descriptors = compute_rdkit_descriptors(example_smiles)
    if descriptors:
        print(f"\nComputed {len(descriptors)} descriptors:")
        for key, val in list(descriptors.items())[:5]:
            print(f"  {key}: {val:.2f}")
    
    morgan = compute_morgan(example_smiles)
    if morgan is not None:
        print(f"\nMorgan fingerprint: {len(morgan)} bits, {np.sum(morgan)} bits set")
