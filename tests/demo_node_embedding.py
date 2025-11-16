"""
Demo script to showcase the node embedding models.
Shows parameter counts, forward pass examples, and model capabilities.
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.transformer.node_embedding import DrugSynergyGNNEdgeAware, DrugSynergyGATEdgeAware # type: ignore
from torch_geometric.data import Data, Batch


def create_benzene_graph():
    """Create a benzene molecule graph (C6H6 ring)."""
    # 6 carbon atoms with features
    x = torch.randn(6, 9)  # 9 features per atom
    
    # Ring connectivity
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
    ], dtype=torch.long)
    
    # Bond features (aromatic, single/double alternating)
    edge_attr = torch.randn(12, 3)
    batch = torch.zeros(6, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


def print_model_info(model, name):
    """Print model architecture information."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print("\nArchitecture:")
    print(f"  - Graph convolution layers: {len(model.convs)}")
    print(f"  - Batch normalization layers: {len(model.batch_norms)}")
    if hasattr(model, 'edge_nns'):
        print(f"  - Edge neural networks: {len(model.edge_nns)}")
    
    return total_params


def test_forward_pass(model, name):
    """Test a forward pass through the model."""
    print(f"\n{'-'*60}")
    print(f"Testing Forward Pass: {name}")
    print(f"{'-'*60}")
    
    # Create sample data
    drug1 = create_benzene_graph()
    drug2 = create_benzene_graph()
    cell_line = torch.randn(1, 100)
    
    model.eval()
    with torch.no_grad():
        # Encode individual drugs
        drug1_embed = model.encode_drug(drug1.x, drug1.edge_index, drug1.edge_attr, drug1.batch)
        drug2_embed = model.encode_drug(drug2.x, drug2.edge_index, drug2.edge_attr, drug2.batch)
        
        # Full forward pass
        synergy_score = model(drug1, drug2, cell_line)
    
    print(f"✓ Drug 1 embedding shape: {drug1_embed.shape}")
    print(f"✓ Drug 2 embedding shape: {drug2_embed.shape}")
    print(f"✓ Synergy score shape: {synergy_score.shape}")
    print(f"✓ Predicted synergy: {synergy_score.item():.4f}")
    
    return synergy_score


def test_batch_processing(model, name):
    """Test batch processing capability."""
    print(f"\n{'-'*60}")
    print(f"Testing Batch Processing: {name}")
    print(f"{'-'*60}")
    
    # Create batch of 3 drug pairs
    drugs1 = [create_benzene_graph() for _ in range(3)]
    drugs2 = [create_benzene_graph() for _ in range(3)]
    
    batch1 = Batch.from_data_list(drugs1)
    batch2 = Batch.from_data_list(drugs2)
    cell_lines = torch.randn(3, 100)
    
    model.eval()
    with torch.no_grad():
        synergy_scores = model(batch1, batch2, cell_lines)
    
    print(f"✓ Batch size: 3 drug pairs")
    print(f"✓ Output shape: {synergy_scores.shape}")
    print(f"✓ Predictions: {synergy_scores.tolist()}")
    
    return synergy_scores


def compare_models():
    """Compare NNConv and GAT architectures."""
    print(f"{'#'*60}")
    print(f"  MODEL COMPARISON")
    print(f"{'#'*60}")
    
    # Initialize both models
    gnn_model = DrugSynergyGNNEdgeAware(
        node_feature_dim=9,
        edge_feature_dim=3,
        cell_line_feature_dim=100,
        hidden_dim=128,
        embedding_dim=64,
        num_gc_layers=3
    )
    
    gat_model = DrugSynergyGATEdgeAware(
        node_feature_dim=9,
        edge_feature_dim=3,
        cell_line_feature_dim=100,
        hidden_dim=128,
        embedding_dim=64,
        num_gc_layers=3,
        heads=4
    )
    
    # Test data
    drug1 = create_benzene_graph()
    drug2 = create_benzene_graph()
    cell_line = torch.randn(1, 100)
    
    gnn_model.eval()
    gat_model.eval()
    
    with torch.no_grad():
        gnn_pred = gnn_model(drug1, drug2, cell_line)
        gat_pred = gat_model(drug1, drug2, cell_line)
    
    print(f"\n{'='*60}")
    print(f"NNConv-based GNN:")
    print(f"  - Parameters: {sum(p.numel() for p in gnn_model.parameters()):,}")
    print(f"  - Prediction: {gnn_pred.item():.4f}")
    print(f"  - Uses edge-conditioned convolutions")
    
    print(f"\n{'='*60}")
    print(f"GAT (Graph Attention Network):")
    print(f"  - Parameters: {sum(p.numel() for p in gat_model.parameters()):,}")
    print(f"  - Prediction: {gat_pred.item():.4f}")
    print(f"  - Uses multi-head attention with edge features")
    
    print(f"\n{'='*60}")
    print("Key Differences:")
    print("  - NNConv: Learns edge-conditioned message functions")
    print("  - GAT: Learns attention weights for neighbor aggregation")
    print("  - Both incorporate bond/edge features (crucial for chemistry)")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("  DRUG SYNERGY NODE EMBEDDING - MODEL DEMO")
    print("="*60)
    
    # Initialize models
    print("\nInitializing models...")
    gnn_model = DrugSynergyGNNEdgeAware(
        node_feature_dim=9,
        edge_feature_dim=3,
        cell_line_feature_dim=100,
        hidden_dim=128,
        embedding_dim=64,
        num_gc_layers=3
    )
    
    gat_model = DrugSynergyGATEdgeAware(
        node_feature_dim=9,
        edge_feature_dim=3,
        cell_line_feature_dim=100,
        hidden_dim=128,
        embedding_dim=64,
        num_gc_layers=3,
        heads=4
    )
    
    # Show model info
    gnn_params = print_model_info(gnn_model, "NNConv-based Edge-Aware GNN")
    gat_params = print_model_info(gat_model, "GAT Edge-Aware Model")
    
    # Test forward passes
    test_forward_pass(gnn_model, "NNConv GNN")
    test_forward_pass(gat_model, "GAT")
    
    # Test batch processing
    test_batch_processing(gnn_model, "NNConv GNN")
    test_batch_processing(gat_model, "GAT")
    
    # Compare models
    compare_models()
    
    print("\n" + "="*60)
    print("  ✓ ALL DEMOS COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
