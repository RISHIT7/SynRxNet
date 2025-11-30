import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class Splitter:
    """Handles dataset splitting with multiple strategies."""

    def __init__(self, strategy:str = 'random', seed: int = 42):
        """
        Initializes the splitter.

        Args:
            strategy (str): Splitting strategy ('random', 'stratified', etc.).
            seed (int): Random seed for reproducibility.
        """
        if strategy not in ['random', 'lodo', 'loco']:
            raise ValueError(f"Unsupported splitting strategy: {strategy}")
        
        self.strategy = strategy
        self.seed = seed
        self._splits_cache = {}

    @lru_cache(maxsize=None)
    def split(self, df: pd.DataFrame) -> Tuple[pd.Index, pd.Index, pd.Index]:
        """
        Splits the DataFrame into train, validation, and test sets.

        Args:
            df (pd.DataFrame): The full dataset.
        Returns:
            Tuple[pd.Index, pd.Index, pd.Index]: Indices for train, val,
            and test sets.
        """
        if self.strategy == 'random':
            return self._stratified_random_split(df)
        elif self.strategy == 'lodo':
            return self._leave_drug_out_split(df)
        elif self.strategy == 'loco':
            return self._leave_cell_out_split(df)
        
    def _stratified_random_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratify y synergy bins to preserve distribution."""

        df = df.copy()
        df['synergy_bin'] = pd.qcut(df['ZIP'], q=5, labels=False, duplicates='drop')

        train_val, test = train_test_split(
            df,
            test_size=0.1,
            stratify=df['synergy_bin'],
            random_state=self.seed
        )

        train, val = train_test_split(
            train_val,
            test_size=0.1/0.9,
            stratify=train_val['synergy_bin'],
            random_state=self.seed
        )

        for split_df in [train, val, test]:
            split_df.drop(columns=['synergy_bin'], inplace=True)
        
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
    
    def _leave_drug_out_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Strict LODO: test sets contain drugs never seen in train."""
        np.random.seed(self.seed)

        all_drugs = set(df['Drug1'].unique()) | set(df['Drug2'].unique())
        test_drugs = set(np.random.choice(list(all_drugs), size=int(0.1 * len(all_drugs)), replace=False))

        # Test: any pair containing a test drug
        test_mask = df['Drug1'].isin(test_drugs) | df['Drug2'].isin(test_drugs)
        
        # Train/val: only pairs with no test drugs
        train_val = df[~test_mask]

        # Split train/val
        train, val = train_test_split(
            train_val,
            test_size=0.1/0.9,
            random_state=self.seed
        )

        test = df[test_mask]

        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
    
    def _leave_cell_out_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Strict LOCO: test sets contain cell lines never seen in train."""
        np.random.seed(self.seed)

        all_cells = df['Cell line'].unique()
        test_cells = np.random.choice(all_cells, size=int(0.1 * len(all_cells)), replace=False)

        # Test: any pair containing a test cell line
        test_mask = df['Cell line'].isin(test_cells)
        
        # Train/val: only pairs with no test cell lines
        train_val = df[~test_mask]

        # Split train/val
        train, val = train_test_split(
            train_val,
            test_size=0.1/0.9,
            random_state=self.seed
        )

        test = df[test_mask]

        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)