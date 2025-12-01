import torch
from typing import Dict, Any
import logging

from src.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

class MultimodalSynergyDataset(BaseDataset):
    """Dataset combining graphs, ChemBERTa, RDKit, and 3D features."""
    
    def __init__(self, *args, use_3d: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_3d = use_3d
        # Create a GraphDataset instance once and reuse it to avoid
        # repeated re-initialization (and duplicate warnings).
        from .graph_dataset import GraphDataset
        # Pass along any cell line feature path if provided in kwargs
        cell_line_path = kwargs.get('cell_line_features_path', None)
        n_components = kwargs.get('n_cell_line_components', None)
        self.graph_ds = GraphDataset(
            csv_path=self.csv_path,
            splitter=self.splitter,
            feature_engineer=self.feature_engineer,
            subset=self.subset,
            cell_line_features_path=cell_line_path,
            n_cell_line_components=n_components
        )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return all feature modalities."""
        metadata = self.get_metadata(idx)
        smiles1, smiles2 = metadata['smiles1'], metadata['smiles2']
        cell = metadata['cell_line']
        
        # Graph features (use the GraphDataset implementation)
        graphs = self.graph_ds[idx]
        
        # ChemBERTa embeddings
        emb1 = self.feature_engineer.encode_smiles(smiles1)
        emb2 = self.feature_engineer.encode_smiles(smiles2)
        
        # RDKit descriptors
        desc1 = self.feature_engineer.compute_descriptors(smiles1)
        desc2 = self.feature_engineer.compute_descriptors(smiles2)
        
        # 3D features if requested
        coords1 = coords2 = None
        if self.use_3d:
            coords1 = self.feature_engineer.compute_3d_features(smiles1)
            coords2 = self.feature_engineer.compute_3d_features(smiles2)
        
        # Pair interaction features
        pair_features = self.feature_engineer.get_pair_features(smiles1, smiles2)
        
        # Cell line features
        cell_features = torch.as_tensor(self.cell_line_features[cell], dtype=torch.float32)
        
        return {
            # Graphs
            'drug1_graph': graphs['graph1'],
            'drug2_graph': graphs['graph2'],
            
            # Embeddings
            'drug1_chemberta': emb1,
            'drug2_chemberta': emb2,
            
            # Descriptors
            'drug1_rdkit': desc1,
            'drug2_rdkit': desc2,
            
            # 3D coordinates (optional)
            'drug1_coords': coords1,
            'drug2_coords': coords2,
            
            # Pair interactions
            **pair_features,
            
            # Cell line
            'cell_line': cell_features,
            
            # Targets
            'targets': self.get_targets(idx),
            
            # Metadata
            'metadata': metadata
        }
