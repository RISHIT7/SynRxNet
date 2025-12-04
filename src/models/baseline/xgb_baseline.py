import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from abc import ABC
import logging
import joblib
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from src.models.base.base_model import BaseModel  # type: ignore

class XGBBaseline(BaseModel):
    """XGBoost baseline for drug synergy prediction.
    
    Handles PyTorch Lightning + scikit-learn/XGBoost integration.
    Features concatenated as: [drugA || drugB || cellline]
    """

    def __init__(self, input_dims: Tuple[int, int, int], **kwargs):
        super().__init__(inpute_dims=input_dims, **kwargs)

        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'n_estimators': kwargs.get('n_estimators', 1000),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 0
        }

        self.feature_scaler = StandardScaler()
        self.is_fitted = False

        self.train_scores_ = []
        self.val_scores_ = []

    def forward(self, drugA: np.ndarray, drugB: np.ndarray, 
                cellline: np.ndarray) -> np.ndarray:
        """XGBoost inference."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before inference.")
        # Normalize inputs and handle both torch tensors and numpy arrays
        if isinstance(drugA, torch.Tensor):
            device = drugA.device
            drugA_np = drugA.detach().cpu().numpy()
        else:
            device = torch.device('cpu')
            drugA_np = drugA

        if isinstance(drugB, torch.Tensor):
            drugB_np = drugB.detach().cpu().numpy()
        else:
            drugB_np = drugB

        if isinstance(cellline, torch.Tensor):
            cellline_np = cellline.detach().cpu().numpy()
        else:
            cellline_np = cellline

        X = np.concatenate([drugA_np, drugB_np, cellline_np], axis=1)
        X_scaled = self.feature_scaler.transform(X)
        dmatrix = xgb.DMatrix(X_scaled)

        # Use the trained xgb model stored on this object
        preds = self.xgb_model.predict(dmatrix)

        return torch.tensor(preds, dtype=torch.float32, device=device)
    
    def fit(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None):
        """Fit the XGBoost model."""
        X_train = np.concatenate([
            train_data['drugA'],
            train_data['drugB'],
            train_data['cellline']
        ], axis=1)
        y_train = train_data['synergy']

        self.feature_scaler.fit(X_train)
        X_train_scaled = self.feature_scaler.transform(X_train)
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)

        if val_data is not None:
            X_val = np.concatenate([
                val_data['drugA'],
                val_data['drugB'],
                val_data['cellline']
            ], axis=1)
            y_val = val_data['synergy']
            X_val_scaled = self.feature_scaler.transform(X_val)
            dval = xgb.DMatrix(X_val_scaled, label=y_val)
            self.xgb_model = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=self.xgb_params['n_estimators'],
                evals=[(dtrain, 'train'), (dval, 'validation')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
        else:
            self.xgb_model = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=self.xgb_params['n_estimators'],
                verbose_eval=False
            )

        self.is_fitted = True

        importances = self.xgb_model.get_score(importance_type='gain')
        logger.info(f"Top features: {sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]}")

    def setup(self, stage: str):
        """Lightning setup hook - fit on first epoch."""
        if stage == 'fit' and not self.is_fitted:
            # Access data from dataloader (your train_dl yields numpy arrays)
            # This assumes you have a collate_fn that converts to numpy for XGB
            logger.info("Fitting XGBoost baseline...")
            # fit() called externally in your training script
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """No gradients for XGBoost - use validation metrics only."""
        return torch.tensor(0.0, requires_grad=False)
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Compute metrics on validation set."""
        drugA, drugB, cellline, synergy = [b.cpu() for b in batch.values()]
        
        pred = self(drugA, drugB, cellline)
        loss = nn.functional.mse_loss(pred, synergy.float())
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_mae', nn.functional.l1_loss(pred, synergy.float()), 
                prog_bar=True, on_epoch=True, sync_dist=True)
        
        pearson = self._pearson_correlation(pred, synergy.float())
        self.log('val_pearson', pearson, prog_bar=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        """Standard predict interface."""
        drugA, drugB, cellline = batch['drugA'], batch['drugB'], batch['cellline']
        return self(drugA, drugB, cellline)
    
    def save_model(self, path: str):
        """Save XGBoost + scaler."""
        joblib.dump({
            'model': self.xgb_model,
            'scaler': self.feature_scaler,
            'params': self.xgb_params,
            'input_dims': self.input_dims
        }, path)
    
    @classmethod
    def load_model(cls, path: str):
        """Load saved model."""
        checkpoint = joblib.load(path)
        model = cls(checkpoint['input_dims'])
        model.xgb_model = checkpoint['model']
        model.feature_scaler = checkpoint['scaler']
        model.xgb_params = checkpoint['params']
        model.is_fitted = True
        return model
