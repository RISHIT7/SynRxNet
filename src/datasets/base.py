import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from src.datasets.splitter import Splitter  # type: ignore
from src.datasets.feature_engineer import FeatureEngineer  # type: ignore

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base dataset for drug synergy prediction.

    Order of operations (IMPORTANT):
      1. Load full CSV and validate.
      2. Attach / filter cell-line features (handle missing cell lines here).
      3. Apply splitter on the cleaned df with consistent coverage.
    """

    REQUIRED_COLUMNS = {
        "ID", "Drug1", "Drug2", "Cell line", "ZIP",
        "smiles_drug1", "smiles_drug2",
    }

    def __init__(
        self,
        csv_path: str,
        splitter: Splitter,
        feature_engineer: FeatureEngineer,
        subset: str = "train",
        cell_line_features_path: Optional[str] = None,
        n_cell_line_components: Optional[int] = 100,
        missing_cell_policy: str = "drop",
    ):
        """
        Args:
            csv_path: Path to cleaned synergy CSV.
            splitter: Splitter object for train/val/test splitting.
            feature_engineer: FeatureEngineer for SMILES validation.
            subset: 'train', 'val', or 'test'.
            cell_line_features_path: Path to cell-line feature matrix.
                - Index: cell line IDs (e.g. CCLE names or your names).
                - Columns: genes or precomputed embedding dims.
            n_cell_line_components: Target dim for PCA if raw expression is given.
            missing_cell_policy:
                - 'drop': drop rows whose cell lines lack features (recommended).
                - 'keep_zero': keep rows, use zero vector for missing cell lines.
                - 'keep_nan': keep rows, use NaN vector for missing cell lines.
        """
        assert missing_cell_policy in {"drop", "keep_zero", "keep_nan"}, \
            "missing_cell_policy must be one of {'drop','keep_zero','keep_nan'}"

        self.csv_path = Path(csv_path)
        self.splitter = splitter
        self.feature_engineer = feature_engineer
        self.subset = subset
        self.cell_line_features_path = (
            Path(cell_line_features_path) if cell_line_features_path else None
        )
        self.n_cell_line_components = n_cell_line_components
        self.missing_cell_policy = missing_cell_policy

        # 1) Load + validate full df
        df_full = self._load_and_validate()

        # 2) Attach cell-line features and (possibly) filter rows BEFORE split
        df_clean, self.cell_line_features = self._build_cell_line_features(df_full)

        # 3) Apply split on cleaned df
        self.df = self._apply_split(df_clean)

        logger.info(
            f"BaseDataset initialized: subset={self.subset}, "
            f"rows={len(self.df)}, cell_feature_dim={self._cell_feature_dim()}"
        )

    # ---------- Step 1: load & validate ----------

    def _load_and_validate(self) -> pd.DataFrame:
        """Load CSV and validate required columns and SMILES."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns {missing_cols} in {self.csv_path}")

        invalid_mask = (
            ~df["smiles_drug1"].apply(self.feature_engineer.validate_smiles)
            | ~df["smiles_drug2"].apply(self.feature_engineer.validate_smiles)
        )
        if invalid_mask.any():
            logger.warning(f"Dropping {invalid_mask.sum()} rows with invalid SMILES.")
            df = df[~invalid_mask].reset_index(drop=True)

        return df

    # ---------- Step 2: attach / filter cell-line features ----------
    def _build_cell_line_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor]]:
        """
        Build cell-line feature dict and possibly filter rows.

        Uses cell_line_mapping.csv to map original names -> CCLE IDs,
        then looks those up in ccle_cell_line_pca_100.csv (or other feature file).

        This runs BEFORE splitting.
        """
        unique_cells = sorted(df["Cell line"].dropna().unique())

        # No external features: zeros for all
        if self.cell_line_features_path is None or not self.cell_line_features_path.exists():
            logger.warning("No cell-line feature file provided; using zero vectors for all cell lines.")
            dim = int(self.n_cell_line_components or 1)
            features = {c: torch.zeros(dim, dtype=torch.float32) for c in unique_cells}
            return df.reset_index(drop=True), features

        # 1) Load PCA / feature matrix (index = CCLE IDs)
        feat_df = pd.read_csv(self.cell_line_features_path, index_col=0)
        dim = feat_df.shape[1]

        # 2) Load mapping file if present
        mapping_path = self.csv_path.parent / "cell_line_mapping.csv"
        if mapping_path.exists():
            mapping = pd.read_csv(mapping_path)
            # original_name -> ccle_name (may be empty string)
            map_dict = {
                str(row["original_name"]): (str(row["ccle_name"]) if isinstance(row["ccle_name"], str) else "")
                for _, row in mapping.iterrows()
            }
        else:
            logger.warning("cell_line_mapping.csv not found; assuming df['Cell line'] already uses CCLE IDs.")
            map_dict = {c: c for c in unique_cells}

        cell_line_features: Dict[str, torch.Tensor] = {}
        missing_cells = []

        for orig in unique_cells:
            ccle_id = map_dict.get(orig, "")
            if not ccle_id:
                # no mapped CCLE ID at all
                missing_cells.append(orig)
                continue
            if ccle_id not in feat_df.index:
                # mapped but not present in PCA file
                missing_cells.append(orig)
                continue

            vec = torch.as_tensor(feat_df.loc[ccle_id].values, dtype=torch.float32)
            cell_line_features[orig] = vec

        if missing_cells:
            logger.warning(
                f"{len(missing_cells)} cell lines in synergy data could not be matched to PCA features: {missing_cells}"
            )

        # Apply policy BEFORE splitting
        if self.missing_cell_policy == "drop":
            keep_mask = ~df["Cell line"].isin(missing_cells)
            dropped = (~keep_mask).sum()
            if dropped > 0:
                logger.warning(f"Dropping {dropped} rows due to missing cell-line features (after mapping).")
            df_clean = df[keep_mask].reset_index(drop=True)
        else:
            df_clean = df.reset_index(drop=True)
            default_val = 0.0 if self.missing_cell_policy == "keep_zero" else float("nan")
            default_vec = torch.full((dim,), default_val, dtype=torch.float32)
            for orig in missing_cells:
                cell_line_features[orig] = default_vec.clone()

        return df_clean, cell_line_features

    # ---------- Step 3: split ----------

    def _apply_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply splitter to the cleaned df."""
        train_df, val_df, test_df = self.splitter.split(df)

        if self.subset == "train":
            out = train_df
        elif self.subset == "val":
            out = val_df
        elif self.subset == "test":
            out = test_df
        else:
            raise ValueError("subset must be 'train', 'val', or 'test'")

        if len(out) == 0:
            logger.error(f"Subset '{self.subset}' is empty after split; check missing_cell_policy and coverage.")
            raise RuntimeError(f"Empty subset '{self.subset}' after split.")
        return out.reset_index(drop=True)

    # ---------- Helpers & Dataset API ----------

    def _cell_feature_dim(self) -> int:
        if not self.cell_line_features:
            return 0
        return len(next(iter(self.cell_line_features.values())))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement __getitem__.")

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "ID": int(row["ID"]),
            "Drug1": str(row["Drug1"]),
            "Drug2": str(row["Drug2"]),
            "cell_line": str(row["Cell line"]),
            "ZIP": float(row["ZIP"]),
            "Bliss": float(row.get("Bliss", float("nan"))),
            "Loewe": float(row.get("Loewe", float("nan"))),
            "HSA": float(row.get("HSA", float("nan"))),
            "smiles1": str(row["smiles_drug1"]),
            "smiles2": str(row["smiles_drug2"]),
        }

    def get_cell_line_feature(self, cell_name: str) -> torch.Tensor:
        return self.cell_line_features[cell_name]

    def get_targets(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        return torch.tensor(
            [
                row["ZIP"],
                row.get("Bliss", float("nan")),
                row.get("Loewe", float("nan")),
                row.get("HSA", float("nan")),
            ],
            dtype=torch.float32,
        )
