"""
gnn.py

This script defines the `InverseFoldGVP` model, which serves as a wrapper
around the core `ResidueGNN`. Its primary role is to instantiate the GNN and
provide a simplified forward pass that is compatible with the main
`DFMNodeClassifier` model.
"""

import torch
from torch import nn
import sys
sys.path.append('..')
from models.residue_gnn import ResidueGNN

class InverseFoldGVP(nn.Module):
    """
    A wrapper class for the ResidueGNN.

    This class simplifies the interface to the GNN. It takes the graph `Data`
    object, extracts the necessary tensors, passes them to the `ResidueGNN`,
    and returns only the final scalar node embeddings, as required by the
    downstream `DFMNodeClassifier`.
    """
    def __init__(self, **kwargs):
        """
        Initializes the underlying ResidueGNN.

        Args:
            **kwargs: Keyword arguments that are passed directly to the
                      `ResidueGNN` constructor.
        """
        super().__init__()
        self.gnn = ResidueGNN(**kwargs)

    def forward(self, data, predictions=None, time_emb=None):
        """
        Performs a forward pass through the GNN.

        Args:
            data (torch_geometric.data.Data): The input graph object, which
                contains all node and edge features.
            predictions (torch.Tensor, optional): Current prediction logits for each node,
                shape [num_nodes, K]. These will be converted to probabilities internally.
            time_emb (torch.Tensor, optional): Time embeddings for each node,
                shape [num_nodes, time_dim].

        Returns:
            hs (torch.Tensor): The final combined node embeddings from the GNN,
                               with shape [num_nodes, hidden_dim_s + vector_features].
        """
        # Unpack the features from the data object and pass them to the GNN.
        hs, hv = self.gnn(data.x_s, data.x_v,
                          data.edge_index, data.edge_s, data.edge_v,
                          predictions=predictions, time_emb=time_emb)
        
        # Convert vector features to scalar form using enhanced rotation-invariant operations
        # hv has shape [num_nodes, v_dim, 3] - after proper message passing aggregation!
        # 
        # Extract richer geometric invariants while preserving equivariance
        # The vector features now contain aggregated neighborhood information
        
        # Safety check for vector feature dimensions
        if hv.dim() != 3 or hv.shape[-1] != 3:
            raise ValueError(f"Expected vector features shape [N, v_dim, 3], got {hv.shape}")
        
        num_nodes, v_dim, _ = hv.shape
        geometric_features = []
        
        # 1. Vector magnitudes (contains neighborhood information now)
        hv_norms = torch.norm(hv, dim=-1)  # [num_nodes, v_dim]
        geometric_features.append(hv_norms)
        
        # 2. Pairwise dot products between vector channels (enhanced by message passing)
        if v_dim > 1:
            dot_products = []
            for i in range(v_dim):
                for j in range(i + 1, v_dim):
                    dot_ij = torch.sum(hv[:, i, :] * hv[:, j, :], dim=-1)  # [num_nodes]
                    dot_products.append(dot_ij.unsqueeze(-1))
            
            if dot_products:
                hv_dots = torch.cat(dot_products, dim=-1)
                geometric_features.append(hv_dots)
        
        # 3. Vector variance (measures directional consistency after aggregation)
        hv_var = torch.var(hv, dim=-1)  # [num_nodes, v_dim]
        geometric_features.append(hv_var)
        
        # 4. NEW: Vector coherence (how aligned are vectors after neighborhood aggregation?)
        if v_dim > 1:
            # Compute mean direction across all vector channels
            mean_vec = torch.mean(hv, dim=1)  # [num_nodes, 3]
            mean_norm = torch.norm(mean_vec, dim=-1, keepdim=True)  # [num_nodes, 1]
            
            # Coherence: how much do individual vectors align with mean direction?
            coherences = []
            for i in range(v_dim):
                coherence = torch.sum(hv[:, i, :] * mean_vec, dim=-1) / (hv_norms[:, i] * mean_norm.squeeze(-1) + 1e-8)
                coherences.append(coherence.unsqueeze(-1))
            
            hv_coherence = torch.cat(coherences, dim=-1)  # [num_nodes, v_dim]
            geometric_features.append(hv_coherence)
        
        # 5. NEW: Cross product features (captures geometric relationships)
        if v_dim >= 2:
            cross_norms = []
            for i in range(v_dim):
                for j in range(i + 1, v_dim):
                    cross = torch.cross(hv[:, i, :], hv[:, j, :], dim=-1)
                    cross_norm = torch.norm(cross, dim=-1)  # [num_nodes]
                    cross_norms.append(cross_norm.unsqueeze(-1))
            
            if cross_norms:
                hv_cross = torch.cat(cross_norms, dim=-1)
                geometric_features.append(hv_cross)
        
        # Combine all features
        combined_features = torch.cat([hs] + geometric_features, dim=-1)
        
        # Safety check for NaN/Inf values in geometric features
        if torch.isnan(combined_features).any() or torch.isinf(combined_features).any():
            print("WARNING: NaN/Inf detected in combined geometric features")
        
        return combined_features


