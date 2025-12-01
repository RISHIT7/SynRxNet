import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from typing import Dict, Any
import logging

from src.datasets.base import BaseDataset # type: ignore

logger = logging.getLogger(__name__)

class GraphDataset(BaseDataset):
    """Dataset class for handling graph data in SynRxNet."""
    def __init__(self, *args, include_3d: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_3d = include_3d

    def _mol_to_graph(self, smiles: str) -> Data:
        """Convert a SMILES string to a PyTorch Geometric Data object."""
        mol = Chem.MolFromSmiles(smiles)
        
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization().real,
                atom.GetIsAromatic()
            ])
        x = torch.tensor(atom_features, dtype=torch.float32)

        edge_index = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append([start, end])
            edge_index.append([end, start])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        edge_attr = []
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            bond_feat = [
                int(bond_type == Chem.rdchem.BondType.SINGLE),
                int(bond_type == Chem.rdchem.BondType.DOUBLE),
                int(bond_type == Chem.rdchem.BondType.TRIPLE),
                int(bond_type == Chem.rdchem.BondType.AROMATIC)
            ]

            edge_attr.extend([bond_feat, bond_feat])
        
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        pos = None
        if self.include_3d:
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d)
            AllChem.UFFOptimizeMolecule(mol_3d)
            conf = mol_3d.GetConformer()
            pos = torch.Tensor(conf.GetPositions(), dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a data sample by index."""

        metadata = self.get_metadata(idx)

        graph1 = self._mol_to_graph(metadata['smiles1'])
        graph2 = self._mol_to_graph(metadata['smiles2'])

        cell_features = self.cell_line_features[metadata['cell_line']]

        return {
            'graph1': graph1,
            'graph2': graph2,
            'cell_line': torch.as_tensor(cell_features, dtype=torch.float32),
            'label': self.get_targets(idx),
            'metadata': metadata
        }
