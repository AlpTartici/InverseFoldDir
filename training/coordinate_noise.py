"""
coordinate_noise.py

Functions for adding Gaussian noise to protein coordinates during training.
This module handles coordinate noise application in the training loop where
time values are available.
"""

import torch
from typing import Optional, List, Union
from torch_geometric.data import Data, Batch


def apply_coordinate_noise_to_batch(data: Union[Data, Batch], t: torch.Tensor, args) -> Union[Data, Batch]:
    """
    Apply coordinate noise to a batch of protein structures.
    
    This function adds Gaussian noise to coordinates based on the training configuration
    and current time values. It works with both single Data objects and batched Data.
    
    Args:
        data: PyTorch Geometric Data or Batch object
        t: Time values [B] for each sample in the batch  
        args: Training arguments containing noise parameters
        
    Returns:
        Data/Batch object with noisy coordinates (modifies in-place)
    """
    # Skip if noise is disabled
    if args.structure_noise_mag_std <= 0:
        return data
    
    # Handle both single Data and Batch objects
    if isinstance(data, Batch):
        # For batched data, we need to apply noise per-sample
        batch_size = data.num_graphs
        if len(t) != batch_size:
            raise ValueError(f"Time tensor size {len(t)} doesn't match batch size {batch_size}")
        
        # Get batch information
        batch_indices = data.batch  # [N] - which batch each node belongs to
        ptr = data.ptr  # [B+1] - start/end indices for each graph
        
        for i in range(batch_size):
            start_idx = ptr[i].item()
            end_idx = ptr[i+1].item()
            t_val = t[i].item()
            
            # Apply noise to this sample's coordinates
            apply_noise_to_sample(data, start_idx, end_idx, t_val, args)
    else:
        # Single data object
        if len(t) != 1:
            raise ValueError(f"Expected single time value for single Data object, got {len(t)}")
        
        # Get sample information
        num_nodes = data.num_nodes
        has_virtual = getattr(data, 'use_virtual_node', False)
        L = num_nodes - 1 if has_virtual else num_nodes
        
        apply_noise_to_sample(data, 0, L, t[0].item(), args)
    
    return data


def apply_noise_to_sample(data: Data, start_idx: int, end_idx: int, t: float, args):
    """
    Apply coordinate noise to a single sample within a batch.
    
    Args:
        data: Data object containing the coordinates
        start_idx: Start index for this sample's nodes
        end_idx: End index for this sample's nodes  
        t: Time value for this sample
        args: Training arguments
    """
    device = data.x_s.device
    L = end_idx - start_idx
    
    # Check if we have stored original coordinates
    if not hasattr(data, '_original_coords'):
        # Skip if original coordinates aren't available
        # This is expected for the current implementation
        return
    
    # Extract sample-specific data
    sample_coords = data._original_coords[start_idx:end_idx]  # [L, 4, 3]
    geom_missing = data.geom_missing[start_idx:end_idx] if hasattr(data, 'geom_missing') else torch.zeros(L, dtype=torch.bool, device=device)
    
    # Calculate time coefficient
    if args.time_based_struct_noise == 'increasing':
        time_coeff = torch.sqrt(torch.tensor(t, device=device)) + 0.1
    elif args.time_based_struct_noise == 'decreasing':
        time_coeff = torch.sqrt(torch.clamp(8.0 - t, min=0.0)) + 0.1
    else:  # 'fixed'
        time_coeff = 1.0
    
    # Calculate uncertainty coefficient  
    if args.uncertainty_struct_noise_scaling:
        # Extract normalized uncertainty from node features
        # Node scalar features: [dihedrals(6), scores_norm(1), source_indicator(2), geom_missing(1)]
        scores_norm = data.x_s[start_idx:end_idx, 6]  # Extract normalized uncertainty
        uncert_coeff = 1.0 - scores_norm  # More flexible parts get more noise
        uncert_coeff = uncert_coeff.unsqueeze(1).unsqueeze(2)  # [L, 1, 1] for broadcasting
    else:
        uncert_coeff = 1.0
    
    # Generate and apply noise
    noise = torch.randn_like(sample_coords) * args.structure_noise_mag_std
    scaled_noise = time_coeff * uncert_coeff * noise
    
    # Don't add noise to residues with missing geometry
    scaled_noise[geom_missing] = 0.0
    
    # Apply noise to coordinates
    data._original_coords[start_idx:end_idx] = sample_coords + scaled_noise


def rebuild_geometric_features_after_noise(data: Union[Data, Batch], graph_builder) -> Union[Data, Batch]:
    """
    Rebuild geometric node and edge features after applying coordinate noise.
    
    This is a more comprehensive approach that recomputes all geometric features
    with the noisy coordinates. Currently simplified - could be expanded to
    recompute all geometric features.
    
    Args:
        data: Data/Batch object with noisy coordinates
        graph_builder: GraphBuilder instance for feature computation
        
    Returns:
        Data/Batch object with updated geometric features
    """
    # For now, this is a placeholder for future enhancement
    # A full implementation would recompute:
    # - Distance matrices
    # - Edge features (3D distances, direction vectors)
    # - Node features (dihedrals, orientations, sidechains)
    
    # This is complex because it requires rebuilding the entire graph
    # For the initial implementation, we'll use the simpler noise-only approach
    return data
