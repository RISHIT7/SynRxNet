import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    """
    Base dataset for drug synergy prediction.
    
    Handles loading, validation, and basic preprocessing.
    Subclasses must implement __getitem__ for specific modalities.
    
    Expected CSV columns: ID,Drug1,Drug2,Cell line,ZIP,Bliss,Loewe,HSA,smiles_drug1,smiles_drug2
    """
    
    REQUIRED_COLUMNS = {
        'ID', 'Drug1', 'Drug2', 'Cell line', 'ZIP', 
        'smiles_drug1', 'smiles_drug2'
    }

    def __init__(
            self,
            csv_path: str,
            splitter: 'Splitter',
            feature_engineer: 'FeatureEngineer',
            subset: str = 'train',
            cell_line_features_path: Optional[str] = None,
            n_cell_line_components: Optional[int] = 100
        ):
        """
        Initializes the dataset.

        Args:
            csv_path (str): Path to the CSV file containing the dataset.
            splitter (Splitter): Splitter object for train/val/test splits.
            feature_engineer (FeatureEngineer): Feature engineer for drug and cell line features.
            subset (str): Subset to load ('train', 'val', 'test').
            cell_line_features_path (Optional[str]): Path to precomputed cell line features.
            n_cell_line_components (Optional[int]): Number of components for dimensionality reduction.
        """

        self.csv_path = Path(csv_path)
        self.splitter = splitter
        self.feature_engineer = feature_engineer
        self.subset = subset

        self.df = self._load_and_validate()
        self.df = self._apply_split()

        self.cell_line_features = self._load_cell_line_features(
            cell_line_features_path,
            n_cell_line_components
        )

        logger.info(f"Loaded {len(self.df)} samples for subset '{self.subset}'")

    def _load_and_validate(self) -> pd.DataFrame:
        """
        Loads the dataset from CSV and validates required columns.

        Returns:
            pd.DataFrame: Loaded and validated DataFrame.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")
        

        df = pd.read_csv(self.csv_path)

        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        invalid_mask = (
            ~df['smiles_drug1'].apply(self.feature_engineer.validate_smiles) |
            ~df['smiles_drug2'].apply(self.feature_engineer.validate_smiles)
        )

        if invalid_mask.any():
            num_invalid = invalid_mask.sum()
            logger.warning(f"Found {num_invalid} invalid SMILES entries. These will be removed.")
            df = df[~invalid_mask].reset_index(drop=True)

        return df
    
    def _apply_split(self) -> pd.DataFrame:
        """
        Applies the specified split to the dataset.

        Returns:
            pd.DataFrame: Subset of the DataFrame corresponding to the specified split.
        """
        train_df, val_df, test_df = self.splitter.split(self.df)

        if self.subset == 'train':
            return train_df
        elif self.subset == 'val':
            return val_df
        elif self.subset == 'test':
            return test_df
        else:
            raise ValueError(f"Invalid subset: {self.subset}. Must be 'train', 'val', or 'test'.")
    
    def _load_cell_line_features(
            self,
            path: Optional[str],
            n_components: Optional[int]
        ) -> Dict[str, torch.Tensor]:
        """Load or compute cell line features."""

        if path is None or not Path(path).exists():
            logger.warning("No cell line features provided, using dummy features")
            unqiue_cells = self.df['Cell line'].unique()
            return {cell: torch.zeros(n_components) for cell in unqiue_cells}
        
        expr_df = pd.read_csv(path, index_col=0)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_df)

        pca = PCA(n_components=n_components)
        cell_emb = pca.fit_transform(expr_scaled)

        cell_names = expr_df.index.tolist()
        return {
            cell: torch.tensor(emb, dtype=torch.float32) 
            for cell, emb in zip(cell_names, cell_emb)
        }
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Subclasses must implement this method to return a data sample."""
        raise NotImplementedError("Subclasses must implement __getitem__ method.")
    
    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Returns metadata for the sample at the given index.
        """
        row = self.df.iloc[idx]
        metadata = {
            'ID': row['ID'],
            'Drug1': row['Drug1'],
            'Drug2': row['Drug2'],
            'Cell line': row['Cell line'],
            'ZIP': row['ZIP'],
            'Bliss': row.get('Bliss', None),
            'Loewe': row.get('Loewe', None),
            'HSA': row.get('HSA', None)
        }
        return metadata
    
    def get_targets(self, idx: int) -> torch.Tensor:
        """
        Returns the target synergy scores for the sample at the given index.
        """
        row = self.df.iloc[idx]
        targets = torch.tensor([
            row['ZIP'],
            row.get('Bliss', float('nan')),
            row.get('Loewe', float('nan')),
            row.get('HSA', float('nan'))
        ], dtype=torch.float32)
        return targets