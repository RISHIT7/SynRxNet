import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing import Any, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseModel(L.LightningModule, ABC):
    """
    Base class for all models in the SynRxNet framework.
    Provides common functionality and enforces structure for derived models.
    """

    def __init__(
            self, 
            inpute_dims: Tuple[int, int, int], 
            output_dims: int = 1,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            loss_fn: Optional[nn.Module] = None,
            dropout: float = 0.0,
            **kwargs
    ):
        """
        Initializes the base model with common parameters.

        Args:
            inpute_dims (Tuple[int, int, int]): Dimensions of the input data.
            output_dims (int): Number of output dimensions.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            loss_fn (Optional[nn.Module]): Loss function to be used.
            dropout (float): Dropout rate for regularization.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn'])
        # Keep both spellings for backward compatibility across the codebase
        self.inpute_dims = inpute_dims
        self.input_dims = inpute_dims
        self.output_dims = output_dims

        self.weight_decay = weight_decay
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.register_buffer('synergy_threshold', torch.tensor(30.0, dtype=torch.float32))

    @property
    def drug_dim(self) -> int:
        """
        Returns the dimension of the drug input.
        """
        return self.inpute_dims[0]
    
    @property
    def cellline_dim(self) -> int:
        """
        Returns the dimension of the cell line input.
        """
        return self.inpute_dims[2]

    @abstractmethod
    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the model. Must be implemented by derived classes.
        """
        raise NotImplementedError()

    def _common_head(self, drugA_emb: torch.Tensor, drugB_emb: torch.Tensor, cellline_emb: torch.Tensor) -> torch.Tensor:
        """
        Common head logic for combining drug and cell line embeddings.

        Args:
            drugA_emb (torch.Tensor): Embedding for drug A.
            drugB_emb (torch.Tensor): Embedding for drug B.
            cellline_emb (torch.Tensor): Embedding for the cell line.
        Returns:
            torch.Tensor: Combined representation.
        """
        drug_pair = (drugA_emb * drugB_emb) / 2
        fused = torch.cat([drug_pair, cellline_emb], dim=-1)

        # Use functional dropout to avoid creating a new module each call
        fused = F.dropout(fused, p=float(self.hparams.dropout), training=self.training)

        # Create and register a linear projection on first use so weights are persistent
        if not hasattr(self, '_common_head_linear'):
            linear = nn.Linear(fused.size(-1), int(self.output_dims))
            # register as attribute so Lightning / nn.Module can see parameters
            setattr(self, '_common_head_linear', linear)

        synergy = getattr(self, '_common_head_linear')(fused)
        return synergy.squeeze(-1)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        drugA, drugB, cellline, synergy = batch['drugA'], batch['drugB'], batch['cellline'], batch['synergy']
        
        pred = self(drugA, drugB, cellline)
        loss = self.loss_fn(pred, synergy.float())
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_mse', loss, prog_bar=False)
        
        pearson = self._pearson_correlation(pred, synergy)
        self.log('train_pearson', pearson, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        drugA, drugB, cellline, synergy = batch['drugA'], batch['drugB'], batch['cellline'], batch['synergy']
        
        pred = self(drugA, drugB, cellline)
        loss = self.loss_fn(pred, synergy.float())
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Additional metrics
        mae = nn.functional.l1_loss(pred, synergy.float())
        self.log('val_mae', mae, prog_bar=True, on_epoch=True, sync_dist=True)
        
        pearson = self._pearson_correlation(pred, synergy)
        self.log('val_pearson', pearson, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Classification metrics (synergy > threshold)
        binary_pred = (pred > self.synergy_threshold).float()
        binary_target = (synergy > self.synergy_threshold).float()
        auroc = self._binary_auroc(binary_pred, binary_target)
        self.log('val_auroc', auroc, prog_bar=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        # Reuse validation_step for test
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        drugA, drugB, cellline = batch['drugA'], batch['drugB'], batch['cellline']
        pred = self(drugA, drugB, cellline)
        return pred
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }
    
    def _pearson_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation coefficient."""
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        numerator = (pred_centered * target_centered).sum()
        denom_pred = (pred_centered ** 2).sum()
        denom_target = (target_centered ** 2).sum()
        pearson = numerator / torch.sqrt(denom_pred * denom_target + 1e-8)
        return pearson
    
    def _binary_auroc(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute binary AUROC."""
        from torchmetrics import AUROC
        auroc = AUROC(task="binary")
        return auroc(pred, target.int())
    
    @classmethod
    def from_dataset(cls, dataset, **kwargs) -> 'BaseModel':
        """
        Instantiate the model from a dataset object.

        Args:
            dataset: Dataset object containing input and output dimensions.
        Returns:
            BaseModel: An instance of the model.
        """
        drugA = dataset['drugA']
        drugB = dataset['drugB']
        cellline = dataset['cellline']
        inpute_dims = (drugA.shape[1], drugB.shape[1], cellline.shape[1])
        return cls(inpute_dims=inpute_dims, **kwargs)