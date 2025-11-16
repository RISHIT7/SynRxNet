# node_embedding.py (with edge features)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import GATConv

class DrugSynergyGNNEdgeAware(nn.Module):
    """
    Edge-Aware Graph Neural Network for Node Embedding in Drug Synergy Prediction.
    
    This model incorporates both node (atom) and edge (bond) features from drug 
    molecular graphs, combined with cell line genomic features to predict drug synergy.
    
    Args:
        node_feature_dim (int): Dimension of node features (atom features).
        edge_feature_dim (int): Dimension of edge features (bond features).
        cell_line_feature_dim (int): Dimension of cancer cell line features.
        hidden_dim (int): Hidden dimension size for graph convolutions.
        embedding_dim (int): Output embedding dimension for drugs.
        num_gc_layers (int): Number of graph convolution layers.
    """
    def __init__(self, node_feature_dim, edge_feature_dim, cell_line_feature_dim,
                 hidden_dim=128, embedding_dim=64, num_gc_layers=3):
        super(DrugSynergyGNNEdgeAware, self).__init__()
        
        # Edge-conditioned convolution layers (NNConv)
        self.convs = nn.ModuleList()
        self.edge_nns = nn.ModuleList()
        
        # First layer
        edge_nn_1 = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim * node_feature_dim),
            nn.ReLU()
        )
        self.edge_nns.append(edge_nn_1)
        self.convs.append(NNConv(node_feature_dim, hidden_dim, edge_nn_1, aggr='mean'))
        
        # Intermediate layers
        for _ in range(num_gc_layers - 2):
            edge_nn = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim * hidden_dim),
                nn.ReLU()
            )
            self.edge_nns.append(edge_nn)
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))
        
        # Final layer
        edge_nn_final = nn.Sequential(
            nn.Linear(edge_feature_dim, embedding_dim * hidden_dim),
            nn.ReLU()
        )
        self.edge_nns.append(edge_nn_final)
        self.convs.append(NNConv(hidden_dim, embedding_dim, edge_nn_final, aggr='mean'))
        
        # Batch normalization for stabilizing training
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_gc_layers - 1)])
        self.batch_norms.append(nn.BatchNorm1d(embedding_dim))
        
        # Cell line feature projection
        self.cell_proj = nn.Sequential(
            nn.Linear(cell_line_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),  # two drugs + cell line
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def encode_drug(self, x, edge_index, edge_attr, batch):
        """
        Encode a drug molecular graph with edge features.
        
        Args:
            x (Tensor): Node features [num_nodes, node_feature_dim].
            edge_index (LongTensor): Edge connectivity [2, num_edges].
            edge_attr (Tensor): Edge features [num_edges, edge_feature_dim].
            batch (LongTensor): Batch assignment vector [num_nodes].
        
        Returns:
            Tensor: Drug embedding [batch_size, embedding_dim].
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            if i < len(self.convs) - 1:  # ReLU on all but last layer
                x = F.relu(x)
        
        # Global pooling to get graph-level representation
        drug_embed = global_mean_pool(x, batch)
        return drug_embed

    def forward(self, data1, data2, cell_line_features):
        """
        Forward pass for two drug molecular graphs (with edge features) and cell line features.
        
        Args:
            data1 (Data): PyG Data object for drug 1 with x, edge_index, edge_attr, batch.
            data2 (Data): PyG Data object for drug 2 with x, edge_index, edge_attr, batch.
            cell_line_features (Tensor): [batch_size, cell_line_feature_dim].
        
        Returns:
            Tensor: Predicted synergy scores [batch_size].
        """
        # Encode both drugs with edge-aware GNN
        drug1_embed = self.encode_drug(data1.x, data1.edge_index, data1.edge_attr, data1.batch)
        drug2_embed = self.encode_drug(data2.x, data2.edge_index, data2.edge_attr, data2.batch)
        
        # Encode cell line features
        cell_embed = self.cell_proj(cell_line_features)
        
        # Combine embeddings
        combined = torch.cat([drug1_embed, drug2_embed, cell_embed], dim=1)
        
        # Predict synergy score
        synergy_score = self.predictor(combined)
        
        return synergy_score.squeeze(-1)


# Alternative: Using GAT with edge features (GATv2Conv or custom implementation)
class DrugSynergyGATEdgeAware(nn.Module):
    """
    Graph Attention Network with Edge Features for Drug Synergy Prediction.
    
    Uses attention mechanisms conditioned on edge features to learn 
    importance weights for neighboring atoms in drug molecular graphs.
    """
    def __init__(self, node_feature_dim, edge_feature_dim, cell_line_feature_dim,
                 hidden_dim=128, embedding_dim=64, num_gc_layers=3, heads=4):
        super(DrugSynergyGATEdgeAware, self).__init__()
        
        # GAT layers with edge features
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(node_feature_dim, hidden_dim // heads, 
                                  heads=heads, edge_dim=edge_feature_dim, concat=True))
        
        # Intermediate layers
        for _ in range(num_gc_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, 
                                      heads=heads, edge_dim=edge_feature_dim, concat=True))
        
        # Final layer
        self.convs.append(GATConv(hidden_dim, embedding_dim, 
                                  heads=1, edge_dim=edge_feature_dim, concat=False))
        
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_gc_layers - 1)])
        self.batch_norms.append(nn.BatchNorm1d(embedding_dim))
        
        # Cell line encoder
        self.cell_proj = nn.Sequential(
            nn.Linear(cell_line_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def encode_drug(self, x, edge_index, edge_attr, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.batch_norms[i](x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        
        drug_embed = global_mean_pool(x, batch)
        return drug_embed

    def forward(self, data1, data2, cell_line_features):
        drug1_embed = self.encode_drug(data1.x, data1.edge_index, data1.edge_attr, data1.batch)
        drug2_embed = self.encode_drug(data2.x, data2.edge_index, data2.edge_attr, data2.batch)
        cell_embed = self.cell_proj(cell_line_features)
        
        combined = torch.cat([drug1_embed, drug2_embed, cell_embed], dim=1)
        synergy_score = self.predictor(combined)
        
        return synergy_score.squeeze(-1)
