import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Optional
from functools import lru_cache
import logging
from pathlib import Path
import joblib
import hashlib

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """A class for feature engineering of molecular data using RDKit and transformer models."""
    def __init__(self, cache_dir: Optional[str] = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use a known public ChemBERTa checkpoint
        self.model_name = "seyonec/ChemBERTa-zinc-base-v1"
        self.tokenizer = None
        self.model = None

        self.desc_names = [desc_name for desc_name, _ in Descriptors._descList]

    @lru_cache(maxsize=10_000)
    def validate_smiles(self, smiles: str) -> bool:
        """Validate a SMILES string."""
        if not smiles or not isinstance(smiles, str):
            return False
        
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    
    def get_chemberta(self):
        """Lazy load ChemBERTA model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            logger.info("Loading ChemBERTA model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
        return self.tokenizer, self.model
    
    @lru_cache(maxsize=10_000)
    def encode_smiles(self, smiles: str) -> torch.Tensor:
        """Encode a SMILES string using ChemBERTA."""
        if not self.validate_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")
        # Check on-disk cache first (cache key = sha1 of SMILES)
        key = hashlib.sha1(smiles.encode('utf-8')).hexdigest()
        cache_path = self.cache_dir / f"chemberta_{key}.pt"
        if cache_path.exists():
            try:
                emb = torch.load(cache_path)
                return emb
            except Exception:
                logger.warning(f"Failed loading cached embedding {cache_path}, recomputing")

        tokenizer, model = self.get_chemberta()
        inputs = tokenizer(smiles, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

        # Save to cache (best-effort)
        try:
            torch.save(embedding, cache_path)
        except Exception:
            logger.warning(f"Failed to save embedding cache to {cache_path}")

        return embedding
    
    @lru_cache(maxsize=10_000)
    def compute_descriptors(self, smiles: str) -> torch.Tensor:
        """Compute RDKit descriptors for a SMILES string."""
        if not self.validate_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        mol = Chem.MolFromSmiles(smiles)
        desc_values = [Descriptors.__dict__[name](mol) for name in self.desc_names]
        return torch.tensor(desc_values, dtype=torch.float32)
    
    def compute_3d_features(self, smiles: str) -> torch.Tensor:
        """Compute 3D features for a SMILES string."""
        if not self.validate_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        coords = torch.tensor(conf.GetPositions(), dtype=torch.float32)

        dists = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)

        i, j = torch.triu_indices(dists.size(0), dists.size(0), offset=1)
        pairwise_dists = dists[i, j]

        bin_edges = torch.linspace(0, pairwise_dists.max(), 21)
        hist, _ = torch.histogram(pairwise_dists, bins=bin_edges)

        features = torch.zeros(1000, dtype=torch.float32)
        features[:hist.numel()] = hist.float()

        return features
    
    def get_pair_features(self, smiles1: str, smiles2: str) -> Dict[str, torch.Tensor]:
        """Get combined features for a pair of SMILES strings."""
        emb1 = self.encode_smiles(smiles1)
        emb2 = self.encode_smiles(smiles2)

        diff = torch.abs(emb1 - emb2)
        prod = emb1 * emb2
        cosine_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).squeeze()

        return {
            "emb1": emb1,
            "emb2": emb2,
            "diff": diff,
            "prod": prod,
            "cosine_sim": cosine_sim
        }