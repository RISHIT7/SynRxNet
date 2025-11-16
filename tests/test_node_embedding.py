"""
Unit tests for node embedding models.

Tests cover:
- Model initialization
- Forward pass shapes
- Edge feature handling
- Batch processing
- Gradient flow
- Device compatibility (CPU/MPS)
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.transformer.node_embedding import DrugSynergyGNNEdgeAware, DrugSynergyGATEdgeAware # type: ignore


class TestDrugSynergyGNNEdgeAware:
    """Test suite for edge-aware GNN with NNConv."""
    
    @pytest.fixture
    def model_params(self):
        """Standard model parameters for testing."""
        return {
            'node_feature_dim': 9,      # Common atom features: atom type, charge, etc.
            'edge_feature_dim': 3,      # Bond type, aromatic, conjugated
            'cell_line_feature_dim': 100,  # Gene expression features
            'hidden_dim': 64,
            'embedding_dim': 32,
            'num_gc_layers': 3
        }
    
    @pytest.fixture
    def model(self, model_params):
        """Initialize model for testing."""
        return DrugSynergyGNNEdgeAware(**model_params)
    
    @pytest.fixture
    def sample_graph_data(self):
        """Create a sample molecular graph (e.g., benzene-like structure)."""
        # 6 atoms in a ring
        x = torch.randn(6, 9)  # 6 nodes, 9 features each
        
        # Create ring connectivity (0-1-2-3-4-5-0)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],  # source nodes
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]   # target nodes
        ], dtype=torch.long)
        
        # 12 edges (bidirectional), 3 features each
        edge_attr = torch.randn(12, 3)
        
        batch = torch.zeros(6, dtype=torch.long)  # Single graph
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    @pytest.fixture
    def batch_graph_data(self):
        """Create a batch of two molecular graphs."""
        # Graph 1: 6 nodes
        x1 = torch.randn(6, 9)
        edge_index1 = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
        ], dtype=torch.long)
        edge_attr1 = torch.randn(12, 3)
        data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1)
        
        # Graph 2: 4 nodes (smaller molecule)
        x2 = torch.randn(4, 9)
        edge_index2 = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3]
        ], dtype=torch.long)
        edge_attr2 = torch.randn(8, 3)
        data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)
        
        return Batch.from_data_list([data1, data2])
    
    def test_model_initialization(self, model, model_params):
        """Test that model initializes correctly."""
        assert isinstance(model, nn.Module)
        assert len(model.convs) == model_params['num_gc_layers']
        assert len(model.batch_norms) == model_params['num_gc_layers']
        assert len(model.edge_nns) == model_params['num_gc_layers']
    
    def test_encode_drug_single_graph(self, model, sample_graph_data):
        """Test drug encoding on a single graph."""
        model.eval()
        with torch.no_grad():
            embedding = model.encode_drug(
                sample_graph_data.x,
                sample_graph_data.edge_index,
                sample_graph_data.edge_attr,
                sample_graph_data.batch
            )
        
        # Should output [batch_size, embedding_dim]
        assert embedding.shape == (1, 32)
        assert not torch.isnan(embedding).any()
        assert not torch.isinf(embedding).any()
    
    def test_encode_drug_batch(self, model, batch_graph_data):
        """Test drug encoding on a batch of graphs."""
        model.eval()
        with torch.no_grad():
            embedding = model.encode_drug(
                batch_graph_data.x,
                batch_graph_data.edge_index,
                batch_graph_data.edge_attr,
                batch_graph_data.batch
            )
        
        # Batch of 2 graphs
        assert embedding.shape == (2, 32)
        assert not torch.isnan(embedding).any()
    
    def test_forward_pass_shape(self, model, sample_graph_data):
        """Test full forward pass produces correct output shape."""
        model.eval()
        batch_size = 1
        cell_line_features = torch.randn(batch_size, 100)
        
        with torch.no_grad():
            output = model(sample_graph_data, sample_graph_data, cell_line_features)
        
        # Should output synergy scores [batch_size]
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()
    
    def test_forward_pass_batch(self, model, batch_graph_data):
        """Test forward pass with batched graphs."""
        model.eval()
        batch_size = 2
        cell_line_features = torch.randn(batch_size, 100)
        
        with torch.no_grad():
            output = model(batch_graph_data, batch_graph_data, cell_line_features)
        
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()
    
    def test_gradient_flow(self, model, sample_graph_data):
        """Test that gradients flow through the model."""
        model.train()
        batch_size = 1
        cell_line_features = torch.randn(batch_size, 100)
        
        output = model(sample_graph_data, sample_graph_data, cell_line_features)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    def test_different_sized_molecules(self, model):
        """Test model handles molecules of different sizes."""
        model.eval()
        
        # Small molecule (3 atoms)
        small_data = Data(
            x=torch.randn(3, 9),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long),
            edge_attr=torch.randn(6, 3),
            batch=torch.zeros(3, dtype=torch.long)
        )
        
        # Large molecule (20 atoms)
        large_data = Data(
            x=torch.randn(20, 9),
            edge_index=torch.randint(0, 20, (2, 50)),
            edge_attr=torch.randn(50, 3),
            batch=torch.zeros(20, dtype=torch.long)
        )
        
        cell_line_features = torch.randn(1, 100)
        
        with torch.no_grad():
            output_small = model(small_data, small_data, cell_line_features)
            output_large = model(large_data, large_data, cell_line_features)
        
        assert output_small.shape == (1,)
        assert output_large.shape == (1,)
    
    def test_deterministic_output(self, model, sample_graph_data):
        """Test that model produces same output for same input in eval mode."""
        model.eval()
        cell_line_features = torch.randn(1, 100)
        
        with torch.no_grad():
            output1 = model(sample_graph_data, sample_graph_data, cell_line_features)
            output2 = model(sample_graph_data, sample_graph_data, cell_line_features)
        
        assert torch.allclose(output1, output2)
    
    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_compatibility(self, model, sample_graph_data):
        """Test model works on Apple Silicon MPS device."""
        device = torch.device("mps")
        model = model.to(device)
        
        # Move data to device
        sample_graph_data = sample_graph_data.to(device)
        cell_line_features = torch.randn(1, 100, device=device)
        
        model.eval()
        with torch.no_grad():
            output = model(sample_graph_data, sample_graph_data, cell_line_features)
        
        assert output.device.type == "mps"
        assert output.shape == (1,)
    
    def test_edge_feature_importance(self, model, sample_graph_data):
        """Test that edge features affect the output."""
        model.eval()
        cell_line_features = torch.randn(1, 100)
        
        with torch.no_grad():
            # Original edge features
            output1 = model(sample_graph_data, sample_graph_data, cell_line_features)
            
            # Modified edge features
            modified_data = sample_graph_data.clone()
            modified_data.edge_attr = torch.randn_like(sample_graph_data.edge_attr)
            output2 = model(modified_data, modified_data, cell_line_features)
        
        # Outputs should be different when edge features change
        assert not torch.allclose(output1, output2, atol=1e-5)


class TestDrugSynergyGATEdgeAware:
    """Test suite for edge-aware GAT model."""
    
    @pytest.fixture
    def model_params(self):
        """Standard model parameters for testing."""
        return {
            'node_feature_dim': 9,
            'edge_feature_dim': 3,
            'cell_line_feature_dim': 100,
            'hidden_dim': 64,
            'embedding_dim': 32,
            'num_gc_layers': 3,
            'heads': 4
        }
    
    @pytest.fixture
    def model(self, model_params):
        """Initialize GAT model for testing."""
        return DrugSynergyGATEdgeAware(**model_params)
    
    @pytest.fixture
    def sample_graph_data(self):
        """Create a sample molecular graph."""
        x = torch.randn(6, 9)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
        ], dtype=torch.long)
        edge_attr = torch.randn(12, 3)
        batch = torch.zeros(6, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    def test_model_initialization(self, model, model_params):
        """Test that GAT model initializes correctly."""
        assert isinstance(model, nn.Module)
        assert len(model.convs) == model_params['num_gc_layers']
        assert len(model.batch_norms) == model_params['num_gc_layers']
    
    def test_forward_pass_shape(self, model, sample_graph_data):
        """Test GAT forward pass produces correct output shape."""
        model.eval()
        batch_size = 1
        cell_line_features = torch.randn(batch_size, 100)
        
        with torch.no_grad():
            output = model(sample_graph_data, sample_graph_data, cell_line_features)
        
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()
    
    def test_attention_mechanism(self, model, sample_graph_data):
        """Test that attention mechanism produces different outputs for different inputs."""
        model.eval()
        cell_line_features = torch.randn(1, 100)
        
        # Create two different node feature sets
        data1 = sample_graph_data.clone()
        data2 = sample_graph_data.clone()
        data2.x = torch.randn_like(data1.x)
        
        with torch.no_grad():
            output1 = model(data1, data1, cell_line_features)
            output2 = model(data2, data2, cell_line_features)
        
        # Different inputs should produce different outputs
        assert not torch.allclose(output1, output2, atol=1e-5)
    
    def test_gradient_flow(self, model, sample_graph_data):
        """Test that gradients flow through GAT layers."""
        model.train()
        cell_line_features = torch.randn(1, 100)
        
        output = model(sample_graph_data, sample_graph_data, cell_line_features)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestModelComparison:
    """Compare NNConv and GAT implementations."""
    
    @pytest.fixture
    def common_params(self):
        """Shared parameters for both models."""
        return {
            'node_feature_dim': 9,
            'edge_feature_dim': 3,
            'cell_line_feature_dim': 100,
            'hidden_dim': 64,
            'embedding_dim': 32,
            'num_gc_layers': 3
        }
    
    @pytest.fixture
    def sample_data(self):
        """Sample graph data."""
        x = torch.randn(6, 9)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]
        ], dtype=torch.long)
        edge_attr = torch.randn(12, 3)
        batch = torch.zeros(6, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    def test_both_models_work(self, common_params, sample_data):
        """Test that both models produce valid outputs."""
        gnn_model = DrugSynergyGNNEdgeAware(**common_params)
        gat_model = DrugSynergyGATEdgeAware(**common_params, heads=4)
        
        cell_line_features = torch.randn(1, 100)
        
        gnn_model.eval()
        gat_model.eval()
        
        with torch.no_grad():
            output_gnn = gnn_model(sample_data, sample_data, cell_line_features)
            output_gat = gat_model(sample_data, sample_data, cell_line_features)
        
        # Both should produce valid outputs
        assert output_gnn.shape == (1,)
        assert output_gat.shape == (1,)
        assert not torch.isnan(output_gnn).any()
        assert not torch.isnan(output_gat).any()
    
    def test_different_architectures_different_outputs(self, common_params, sample_data):
        """Test that GNN and GAT produce different predictions (as expected)."""
        torch.manual_seed(42)
        gnn_model = DrugSynergyGNNEdgeAware(**common_params)
        
        torch.manual_seed(42)
        gat_model = DrugSynergyGATEdgeAware(**common_params, heads=4)
        
        cell_line_features = torch.randn(1, 100)
        
        gnn_model.eval()
        gat_model.eval()
        
        with torch.no_grad():
            output_gnn = gnn_model(sample_data, sample_data, cell_line_features)
            output_gat = gat_model(sample_data, sample_data, cell_line_features)
        
        # Even with same seed, different architectures produce different outputs
        # (due to different layer structures)
        assert not torch.allclose(output_gnn, output_gat, atol=1e-3)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def model(self):
        """Initialize a model for edge case testing."""
        return DrugSynergyGNNEdgeAware(
            node_feature_dim=9,
            edge_feature_dim=3,
            cell_line_feature_dim=100,
            hidden_dim=64,
            embedding_dim=32,
            num_gc_layers=3
        )
    
    def test_single_node_graph(self, model):
        """Test handling of graph with single node (unusual but valid)."""
        # Single atom (e.g., noble gas or radical)
        single_node_data = Data(
            x=torch.randn(1, 9),
            edge_index=torch.empty((2, 0), dtype=torch.long),  # No edges
            edge_attr=torch.empty((0, 3)),
            batch=torch.zeros(1, dtype=torch.long)
        )
        
        cell_line_features = torch.randn(1, 100)
        model.eval()
        
        with torch.no_grad():
            output = model(single_node_data, single_node_data, cell_line_features)
        
        assert output.shape == (1,)
        # Should handle gracefully even without edges
    
    def test_parameter_count(self, model):
        """Test that model has reasonable number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params
        # Sanity check: should be in reasonable range (not too small, not huge)
        assert 10_000 < total_params < 10_000_000


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
