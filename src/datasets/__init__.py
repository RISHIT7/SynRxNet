from .base import BaseDataset
from .graph_dataset import GraphDataset
from .multimodal_dataset import MultimodalSynergyDataset
from .splitter import Splitter
from .feature_engineer import FeatureEngineer

__all__ = [
    "BaseDataset",
    "GraphDataset",
    "MultimodalSynergyDataset",
    "Splitter",
    "FeatureEngineer",
    "create_dataset"
]

def create_dataset(
        csv_path: str,
        dataset_type: str,
        split_strategy: str,
        split_seed: int = 42,
        cache_dir: str = "./cache",
        **kwargs
) -> BaseDataset:
    """
    Factory function to create appropriate dataset.
    
    Args:
        csv_path: Path to cleaned CSV with columns: ID,Drug1,Drug2,Cell line,ZIP,smiles_drug1,smiles_drug2
        dataset_type: 'graph', 'smiles', or 'multimodal'
        split_strategy: 'random', 'lodo', 'loco'
        split_seed: Random seed for splitting
        cache_dir: Directory for feature caching
        
    Returns:
        Configured dataset instance
    """

    splitter = Splitter(strategy=split_strategy, seed=split_seed)
    feature_engineer = FeatureEngineer(cache_dir=cache_dir)

    if dataset_type == 'graph':
        dataset = GraphDataset(
            csv_path=csv_path,
            splitter=splitter,
            feature_engineer=feature_engineer,
            **kwargs
        )
    elif dataset_type == 'multimodal':
        dataset = MultimodalSynergyDataset(
            csv_path=csv_path,
            splitter=splitter,
            feature_engineer=feature_engineer,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return dataset