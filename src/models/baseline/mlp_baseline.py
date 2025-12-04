import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

from src.models.base.base_model import BaseModel # type: ignore

# Example MLP Baseline (inherits from BaseModel)
class MLPBaseline(BaseModel):
    """Simple MLP baseline for drug synergy prediction."""
    
    def __init__(self, hidden_dims: list = [512, 256, 128], **kwargs):
        # Expect BaseModel required kwargs like `inpute_dims` or `input_dims` to be forwarded
        super().__init__(**kwargs)
        
        drug_dim = self.drug_dim
        cellline_dim = self.cellline_dim
        
        # Drug encoders (separate MLPs for each drug)
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(float(kwargs.get('dropout', 0.1))),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(float(kwargs.get('dropout', 0.1)))
        )
        
        # Cell line projector
        self.cellline_proj = nn.Sequential(
            nn.Linear(cellline_dim, hidden_dims[1]),
            nn.ReLU()
        )
        
    def forward(self, drugA: torch.Tensor, drugB: torch.Tensor, 
                cellline: torch.Tensor) -> torch.Tensor:
        drugA_emb = self.drug_encoder(drugA)
        drugB_emb = self.drug_encoder(drugB)  # Shared weights
        cellline_emb = self.cellline_proj(cellline)
        
        return self._common_head(drugA_emb, drugB_emb, cellline_emb)