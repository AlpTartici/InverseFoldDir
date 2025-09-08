#!/usr/bin/env python3
"""
Protein sequence inpainting using trained DFM models.

Predicts masked positions while conditioning on known positions in protein sequences.
"""

import os
import sys
import torch
import argparse
import json
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from torch.distributions import Dirichlet
from tqdm import tqdm

# Add the parent directory to the Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import utilities from the sample_utils module (existing infrastructure)
from training.sample_utils import (
    SamplingCoordinator, compute_sampling_metrics, simplex_proj, create_inpainting_mask_with_alignment,
    THREE_TO_ONE, IDX_TO_AA, AA_TO_IDX, SINGLE_TO_TRIPLE, load_model_distributed,
    create_structural_ensemble, compute_ensemble_consensus, sample_with_ensemble_consensus
)


def format_probability_distribution(probabilities, K=21, top_n=5):
    """
    Format probability distribution with amino acid names.
    
    Args:
        probabilities: Tensor of shape [K] with probabilities
        K: Number of amino acid classes (default 21)
        top_n: Number of top probabilities to show first (default 5)
        
    Returns:
        Formatted string with probability distribution
    """
    # Convert to numpy if needed
    if hasattr(probabilities, 'cpu'):
        probs = probabilities.cpu().numpy()
    else:
        probs = probabilities
    
    # Create list of (amino_acid, probability) pairs
    aa_probs = []
    for i in range(min(K, len(probs))):
        if i < len(IDX_TO_AA):
            aa_name = THREE_TO_ONE.get(IDX_TO_AA[i], 'X')
            aa_probs.append((aa_name, probs[i]))
        else:
            aa_probs.append(('?', probs[i]))
    
    # Sort by probability (descending)
    aa_probs_sorted = sorted(aa_probs, key=lambda x: x[1], reverse=True)
    
    # Format output
    lines = []

    lines.append("    Probability distribution:")
    
    # Show top N first
    lines.append(f"    Top {top_n}:")
    for i, (aa, prob) in enumerate(aa_probs_sorted[:top_n]):
        lines.append(f"      {aa}: {prob:.7f}")
    
    # Show all 20 standard amino acids in alphabetical order
    lines.append("    All amino acids (alphabetical):")
    aa_probs_alpha = sorted(aa_probs[:20], key=lambda x: x[0])  # Only first 20 (standard AAs)
    
    # Format in 4 columns for better readability
    for i in range(0, len(aa_probs_alpha), 5):
        row_items = aa_probs_alpha[i:i+5]
        row_str = "      " + "  ".join([f"{aa}: {prob:.7f}" for aa, prob in row_items])
        lines.append(row_str)
    
    return "\n".join(lines)


def fetch_uniprot_sequence(uniprot_id: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Fetch protein sequence from UniProt API.
    
    Args:
        uniprot_id: UniProt accession ID (e.g., 'P69905')
        verbose: Print fetching details
        
    Returns:
        Dictionary with sequence information
    """
    if verbose:
        print(f"Fetching sequence for UniProt ID: {uniprot_id}")
    
    try:
        # UniProt REST API for FASTA format
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse FASTA format
        fasta_text = response.text.strip()
        lines = [line.strip() for line in fasta_text.split('\n') if line.strip()]
        
        if not lines or not lines[0].startswith('>'):
            raise ValueError(f"Invalid FASTA format for {uniprot_id}")
        
        header = lines[0] if lines else ""
        sequence = ''.join(lines[1:]) if len(lines) > 1 else ""
        
        if not sequence:
            raise ValueError(f"No sequence found for {uniprot_id}")
        
        # Extract organism and protein name from header
        organism = "Unknown"
        protein_name = uniprot_id
        
        if "|" in header and "OS=" in header:
            parts = header.split("OS=")
            if len(parts) > 1:
                organism = parts[1].split()[0] + " " + parts[1].split()[1] if len(parts[1].split()) > 1 else parts[1].split()[0]
            
            name_part = parts[0].split("|")
            if len(name_part) >= 3:
                protein_name = name_part[2].strip()
        
        result = {
            'uniprot_id': uniprot_id,
            'sequence': sequence,
            'length': len(sequence),
            'organism': organism,
            'protein_name': protein_name,
            'header': header
        }
        
        if verbose:
            print(f"  Found sequence: {len(sequence)} amino acids")
            print(f"  Organism: {organism}")
            print(f"  Protein: {protein_name}")
        
        return result
        
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to fetch UniProt sequence for {uniprot_id}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to parse UniProt response for {uniprot_id}: {e}")


def find_structure_by_uniprot(uniprot_id: str, dataset, pdb_uniprot_mapping: Optional[Dict] = None, verbose: bool = False):
    """
    Find structure in dataset by UniProt ID using mapping or sequence similarity.
    
    Args:
        uniprot_id: UniProt accession ID
        dataset: Dataset to search
        pdb_uniprot_mapping: Optional PDB->UniProt mapping dict
        verbose: Print search details
        
    Returns:
        Protein data object or None if not found
    """
    if verbose:
        print(f"Searching for structures matching UniProt {uniprot_id}")
    
    # Try mapping-based search first if available
    if pdb_uniprot_mapping:
        # Create reverse mapping: UniProt -> PDB IDs
        uniprot_to_pdb = {}
        for pdb_id, mapped_uniprot in pdb_uniprot_mapping.items():
            if mapped_uniprot:
                if mapped_uniprot not in uniprot_to_pdb:
                    uniprot_to_pdb[mapped_uniprot] = []
                uniprot_to_pdb[mapped_uniprot].append(pdb_id.lower())
        
        if uniprot_id in uniprot_to_pdb:
            pdb_ids = uniprot_to_pdb[uniprot_id]
            if verbose:
                print(f"  Found {len(pdb_ids)} mapped PDB IDs: {pdb_ids}")
            
            # Try to find the first available PDB structure
            for pdb_id in pdb_ids:
                protein_data = find_structure_by_pdb_id(pdb_id, dataset, verbose=False)
                if protein_data:
                    if verbose:
                        print(f"  Using PDB {pdb_id} for UniProt {uniprot_id}")
                    return protein_data
    
    if verbose:
        print(f"  No structures found for UniProt {uniprot_id}")
    
    return None


def find_structure_by_pdb_id(pdb_id: str, dataset, verbose: bool = False):
    """
    Find structure by PDB ID in the dataset.
    
    Args:
        pdb_id: PDB ID (e.g., '1abc')
        dataset: Dataset to search
        verbose: Print search details
        
    Returns:
        Protein data object or None if not found
    """
    pdb_id = pdb_id.lower()
    
    if verbose:
        print(f"Searching dataset for PDB ID: {pdb_id}")
    
    # Search through dataset
    for i, data in enumerate(dataset):
        # Check various possible ID fields
        data_pdb_id = None
        
        # Handle tuple format (PyTorch dataset)
        if isinstance(data, tuple) and len(data) > 0:
            first_item = data[0]
            if hasattr(first_item, 'name'):
                data_pdb_id = first_item.name
            elif hasattr(first_item, 'keys') and 'name' in first_item:
                data_pdb_id = first_item['name']
        # Handle dictionary format
        elif isinstance(data, dict):
            # Dictionary format - check common keys
            data_pdb_id = data.get('name') or data.get('pdb_id') or data.get('id')
        # Handle object format
        else:
            # Object format - check common attributes
            if hasattr(data, 'pdb_id'):
                data_pdb_id = data.pdb_id
            elif hasattr(data, 'name'):
                data_pdb_id = data.name
            elif hasattr(data, 'id'):
                data_pdb_id = data.id
        
        if data_pdb_id and pdb_id in str(data_pdb_id).lower():
            if verbose:
                print(f"  Found {pdb_id} at index {i}")
                # Handle sequence length for different formats
                seq_len = 'unknown'
                if isinstance(data, tuple) and len(data) > 0:
                    first_item = data[0]
                    if hasattr(first_item, 'filtered_seq'):
                        seq_len = len(first_item.filtered_seq)
                    elif hasattr(first_item, 'x_s') and hasattr(first_item.x_s, 'shape'):
                        seq_len = first_item.x_s.shape[0]
                elif isinstance(data, dict):
                    if 'seq' in data:
                        seq_len = len(data['seq'])
                    elif 'filtered_seq' in data:
                        seq_len = len(data['filtered_seq'])
                else:
                    if hasattr(data, 'filtered_seq'):
                        seq_len = len(data.filtered_seq)
                    elif hasattr(data, 'seq'):
                        seq_len = len(data.seq)
                print(f"  Sequence length: {seq_len}")
            return data
    
    if verbose:
        print(f"  PDB ID {pdb_id} not found in dataset")
    
    return None


def generate_inpainting_trajectory_json(results, structure_names, output_dir, output_prefix, K=21):
    """
    Generate detailed JSON output with time-step information for inpainting (masked positions only).
    
    Args:
        results: List of result dictionaries containing trajectory data
        structure_names: List of structure names/PDB IDs
        output_dir: Output directory
        output_prefix: Prefix for output filename
        K: Number of amino acid classes
        
    Returns:
        str: Path to the generated JSON file
    """
    from datetime import datetime
    import json
    
    # Build output data structure
    detailed_output = {}
    
    for i, (result, structure_name) in enumerate(zip(results, structure_names)):
        if 'error' in result:
            # Handle error cases
            detailed_output[structure_name] = {
                'error': result['error'],
                'structure_idx': result.get('structure_idx', i)
            }
            continue
        
        trajectory = result.get('trajectory_data')
        if not trajectory:
            detailed_output[structure_name] = {
                'error': 'No trajectory data available',
                'structure_idx': result.get('structure_idx', i)
            }
            continue
        
        # Extract PDB ID for compatibility
        pdb_id = structure_name.split('_')[0] if '_' in structure_name else structure_name
        
        # Basic structure info
        structure_data = {
            'structure_idx': result.get('structure_idx', i),
            'pdb_id': pdb_id,
            'structure_name': structure_name,
            'sequence_length': result.get('length', 0),
            'masked_positions_only': trajectory.get('masked_positions_only', True),
            'total_positions': trajectory.get('total_positions', 0),
            'masked_count': trajectory.get('masked_count', 0),
            'inpainting_info': {
                'mask_ratio': result.get('mask_ratio', 'unknown'),
                'alignment_score': result.get('alignment_info', {}).get('alignment_score', 'unknown'),
                'masked_accuracy': result.get('evaluation_metrics', {}).get('masked_accuracy', 'unknown'),
                'overall_accuracy': result.get('evaluation_metrics', {}).get('accuracy', 'unknown')
            }
        }
        
        # Add sequences
        if 'predicted_sequence' in result:
            if isinstance(result['predicted_sequence'], list):
                # Handle case where list elements might be tensors
                sequence_indices = []
                for idx in result['predicted_sequence']:
                    if hasattr(idx, 'item'):
                        sequence_indices.append(int(idx.item()))
                    else:
                        sequence_indices.append(int(idx))
                structure_data['predicted_sequence'] = ''.join([THREE_TO_ONE[IDX_TO_AA[idx]] if 0 <= idx < len(IDX_TO_AA) else 'X' for idx in sequence_indices])
            else:
                structure_data['predicted_sequence'] = result['predicted_sequence']
        
        if 'true_sequence' in result and result['true_sequence']:
            structure_data['true_sequence'] = result['true_sequence']
        
        # Extract model predictions at each timestep for masked positions only
        time_points = trajectory['time_points']
        positions = trajectory['positions']
        
        if not positions:
            structure_data['error'] = 'No position data in trajectory'
            detailed_output[structure_name] = structure_data
            continue
        
        # Get masked positions (keys in trajectory positions dict)
        masked_positions = sorted([int(pos) for pos in positions.keys()])
        
        # Initialize arrays for this protein's masked positions only
        num_timesteps = len(time_points)
        
        model_predictions = []  # List of lists: [timestep][masked_position_index]
        current_states = []     # List of lists: [timestep][masked_position_index]
        
        # Extract data for each timestep, but only for masked positions
        for timestep in range(num_timesteps):
            timestep_predictions = []
            timestep_states = []
            
            for pos in masked_positions:
                pos_data = positions[str(pos)]
                
                # Get prediction for this timestep
                if timestep < len(pos_data['most_likely_aa']):
                    pred_aa_idx = pos_data['most_likely_aa'][timestep]
                    prob = pos_data['probabilities'][timestep]
                    
                    timestep_predictions.append(pred_aa_idx)
                    timestep_states.append(prob)
                else:
                    timestep_predictions.append(20)  # Unknown
                    timestep_states.append(0.0)
            
            model_predictions.append(timestep_predictions)
            current_states.append(timestep_states)
        
        # Store in structure data
        structure_data['masked_positions'] = masked_positions
        structure_data['model_predictions'] = model_predictions  # [timestep][masked_pos_idx] -> aa_index
        structure_data['current_states'] = current_states        # [timestep][masked_pos_idx] -> probability
        structure_data['time_points'] = time_points
        
        # Store ground truth for masked positions only
        if 'true_indices' in result and result['true_indices'] is not None:
            true_indices = result['true_indices']
            masked_true_indices = [true_indices[pos] if pos < len(true_indices) else 20 for pos in masked_positions]
            structure_data['true_indices_masked'] = masked_true_indices
        
        # Add detailed amino acid breakdown for masked positions
        structure_data['detailed_trajectory'] = {}
        for pos in masked_positions:
            pos_data = positions[str(pos)]
            detailed_breakdown = pos_data.get('detailed_breakdown', [])
            
            structure_data['detailed_trajectory'][f'position_{pos}'] = {
                'position_index': pos,
                'time_points': pos_data['time_points'],
                'most_likely_aa': pos_data['most_likely_aa'],
                'probabilities': pos_data['probabilities'],
                'amino_acid_evolution': detailed_breakdown  # Full AA breakdown with velocities
            }
        
        detailed_output[structure_name] = structure_data
    
    # Generate output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filename = f"{timestamp}_{output_prefix}_inpainting_trajectory.json"
    json_filepath = os.path.join(output_dir, json_filename)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON with pretty formatting
    with open(json_filepath, 'w') as f:
        json.dump(detailed_output, f, indent=2, separators=(',', ': '))
    
    print(f"Inpainting trajectory JSON saved to: {json_filepath}")
    
    # Generate trajectory analysis NPZ file for get_prediction_accuracy compatibility
    trajectory_analysis_data = {}
    
    for i, (result, structure_name) in enumerate(zip(results, structure_names)):
        if 'error' in result or 'trajectory_data' not in result:
            continue
            
        trajectory = result['trajectory_data']
        positions = trajectory['positions']
        
        if not positions:
            continue
            
        # Extract PDB ID
        pdb_id = structure_name.split('_')[0] if '_' in structure_name else structure_name
        
        # Get masked positions
        masked_positions = sorted([int(pos) for pos in positions.keys()])
        time_points = trajectory['time_points']
        num_timesteps = len(time_points)
        
        # Build data arrays for masked positions only
        model_predictions = []
        current_states = []
        
        for timestep in range(num_timesteps):
            timestep_predictions = []
            timestep_states = []
            
            for pos in masked_positions:
                pos_data = positions[str(pos)]
                
                if timestep < len(pos_data['most_likely_aa']):
                    pred_aa_idx = pos_data['most_likely_aa'][timestep]
                    prob = pos_data['probabilities'][timestep]
                    
                    timestep_predictions.append(pred_aa_idx)
                    timestep_states.append(prob)
                else:
                    timestep_predictions.append(20)
                    timestep_states.append(0.0)
            
            model_predictions.append(timestep_predictions)
            current_states.append(timestep_states)
        
        # Store predictions and states for this protein (masked positions only)
        trajectory_analysis_data[f'{pdb_id}_model_predictions'] = model_predictions
        trajectory_analysis_data[f'{pdb_id}_current_states'] = current_states
        trajectory_analysis_data[f'{pdb_id}_masked_positions'] = masked_positions  # Store which positions were masked
        
        # Store ground truth for masked positions
        if 'true_indices' in result and result['true_indices'] is not None:
            true_indices = result['true_indices']
            masked_true_indices = [true_indices[pos] if pos < len(true_indices) else 20 for pos in masked_positions]
            trajectory_analysis_data[f'{pdb_id}_true_indices_masked'] = masked_true_indices
    
    # Save trajectory analysis file
    if trajectory_analysis_data:
        trajectory_filename = f"{timestamp}_{output_prefix}_inpainting_trajectory_analysis.npz"
        trajectory_filepath = os.path.join(output_dir, trajectory_filename)
        
        import numpy as np
        np.savez_compressed(trajectory_filepath, **trajectory_analysis_data)
        print(f"Inpainting trajectory analysis data saved to: {trajectory_filepath}")
        print(f"Use get_prediction_accuracy('{trajectory_filename}', timestep, pdb_id) to analyze masked position results")
        print(f"Note: This file contains trajectory data for MASKED POSITIONS ONLY")
    
    return json_filepath


def parse_and_validate_mask_positions(mask_positions_str: str, ground_truth_sequence: str, verbose: bool = False, strict_validation: bool = True) -> List[int]:
    """
    Parse mask positions with ground truth validation for variant effect prediction.
    
    Supports two formats:
    1. "28,89,104" - positions only (existing behavior)
    2. "D28,Y89,K104" - amino acid + position with validation
    
    For format 2, validates that the specified amino acid matches the ground truth
    at that position before proceeding. This provides a safety mechanism for
    variant effect prediction.
    
    Args:
        mask_positions_str: Comma-separated position specifications
        ground_truth_sequence: Single-letter amino acid sequence for validation
        verbose: Print validation details
        strict_validation: If False, skip amino acid validation for format "A123"
        
    Returns:
        List of positions to mask (0-indexed)
        
    Raises:
        ValueError: If ground truth validation fails at any position
    """
    if not mask_positions_str or not ground_truth_sequence:
        raise ValueError("Both mask_positions_str and ground_truth_sequence are required")
    
    positions = []
    validation_results = []
    
    for item in mask_positions_str.split(','):
        item = item.strip()
        
        if not item:
            continue
            
        # Check if format is "A123" (amino acid + position)
        if item[0].isalpha():
            expected_aa = item[0].upper()
            try:
                pos_1indexed = int(item[1:])
            except ValueError:
                raise ValueError(f"Invalid position format: '{item}'. Expected format: 'A123' or '123'")
                
            pos_0indexed = pos_1indexed - 1
            
            # Validate position bounds
            if pos_0indexed < 0 or pos_0indexed >= len(ground_truth_sequence):
                raise ValueError(f"Position {pos_1indexed} is out of bounds (sequence length: {len(ground_truth_sequence)})")
            
            # Validate ground truth amino acid (only if strict_validation is enabled)
            actual_aa = ground_truth_sequence[pos_0indexed].upper()
            if strict_validation and actual_aa != expected_aa:
                raise ValueError(f"❌ Ground truth validation failed at position {pos_1indexed}: "
                               f"expected {expected_aa}, but found {actual_aa}")
            
            positions.append(pos_0indexed)
            
            if strict_validation:
                validation_results.append((pos_1indexed, expected_aa, "OK"))
                if verbose:
                    print(f"  Position {pos_1indexed}: Expected {expected_aa}, Found {actual_aa} OK")
            else:
                validation_results.append((pos_1indexed, expected_aa, f"WARNING (found {actual_aa})"))
                if verbose:
                    print(f"  Position {pos_1indexed}: Expected {expected_aa}, Found {actual_aa} WARNING (non-strict mode)")
                
        else:
            # Position-only format "123" (existing behavior)
            try:
                pos_1indexed = int(item)
            except ValueError:
                raise ValueError(f"Invalid position format: '{item}'. Expected format: 'A123' or '123'")
                
            pos_0indexed = pos_1indexed - 1
            
            # Validate position bounds
            if pos_0indexed < 0 or pos_0indexed >= len(ground_truth_sequence):
                raise ValueError(f"Position {pos_1indexed} is out of bounds (sequence length: {len(ground_truth_sequence)})")
            
            positions.append(pos_0indexed)
            
            if verbose:
                actual_aa = ground_truth_sequence[pos_0indexed].upper()
                print(f"  Position {pos_1indexed}: No validation required, found {actual_aa}")
    
    if not positions:
        raise ValueError("No valid positions found in mask specification")
    
    # Print validation summary for amino acid + position format
    if validation_results and verbose:
        print(f"✅ Ground truth validation passed for {len(validation_results)} positions")
    
    return positions


def extract_sequence_from_data(data, use_virtual_node=False) -> str:
    """
    Extract single-letter amino acid sequence from protein data object.
    
    Args:
        data: Protein structure graph data
        use_virtual_node: Whether virtual nodes are used
        
    Returns:
        Single-letter amino acid sequence string
        
    Raises:
        ValueError: If no sequence can be extracted
    """
    # Handle virtual nodes
    total_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x_s.size(0)
    if use_virtual_node:
        N = total_nodes - 1  # Exclude virtual node
    else:
        N = total_nodes

    # Try different sequence sources
    if hasattr(data, 'filtered_seq') and data.filtered_seq is not None:
        sequence = data.filtered_seq[:N]
        debug_seq = ''.join(sequence)
        return debug_seq
    elif hasattr(data, 'y') and data.y is not None:
        # Convert one-hot to sequence
        sequence_indices = data.y[:N].argmax(-1).tolist()
        sequence = ''.join([THREE_TO_ONE.get(IDX_TO_AA[idx], 'X') if 0 <= idx < len(IDX_TO_AA) else 'X' 
                           for idx in sequence_indices])
        return sequence
    elif hasattr(data, 'seq') and data.seq is not None:
        # Handle different sequence formats
        if isinstance(data.seq, str):
            sequence = data.seq[:N]
            return sequence
        elif isinstance(data.seq, list):
            sequence = ''.join(data.seq[:N])
            return sequence
    
    raise ValueError("No sequence data found in protein data object")
def load_protein_from_map_pkl(protein_id: str, map_pkl_path: str, dataset_params: dict, verbose: bool = False, graph_builder=None):
    """
    Load a protein directly from map_pkl file without requiring it to be in splits.
    
    This function is used as a fallback when the protein isn't found in the regular dataset splits.
    
    Args:
        protein_id: Protein ID to load (UniProt ID or PDB ID)
        map_pkl_path: Path to the map_pkl file
        dataset_params: Dataset parameters from the model checkpoint
        verbose: Print loading details
        graph_builder: Optional pre-initialized GraphBuilder to reuse (avoids RBF Manager reinitialization)
        
    Returns:
        Graph data object or None if not found
    """
    try:
        import pickle
        from data.graph_builder import GraphBuilder
        
        if verbose:
            print(f"  Loading {protein_id} directly from {map_pkl_path}")
        
        # Load the map_pkl file
        with open(map_pkl_path, 'rb') as f:
            protein_data = pickle.load(f)
        
        # Try different possible keys for the protein
        possible_keys = [
            protein_id,
            protein_id.upper(),
            protein_id.lower(),
        ]
        
        found_key = None
        for key in possible_keys:
            if key in protein_data:
                found_key = key
                break
        
        if found_key is None:
            if verbose:
                print(f"    {protein_id} not found in map_pkl file")
                print(f"    Available keys (first 10): {list(protein_data.keys())[:10]}")
            return None
        
        if verbose:
            print(f"    Found {protein_id} as key '{found_key}'")
        
        # Get the protein entry
        protein_entry = protein_data[found_key]
        
        # Debug: check what sequence data is in the original entry
        if verbose:
            if 'seq' in protein_entry:
                original_seq = protein_entry['seq']
                print(f"    Original seq length: {len(original_seq)}")
                if len(original_seq) >= 146:
                    print(f"    Position 146 in original seq: {original_seq[145]}")
                    print(f"    Original seq around 146: {original_seq[140:151]}")
                else:
                    print(f"    WARNING: Original seq too short for position 146")
        
        # Use provided graph_builder if available, otherwise create new one
        if graph_builder is not None:
            if verbose:
                print(f"    Using shared GraphBuilder (avoiding RBF Manager reinitialization)")
        else:
            # Create graph builder with dataset parameters (fallback)
            if verbose:
                print(f"    WARNING: Creating new GraphBuilder (this will reinitialize RBF Manager)")
            
            from data.graph_builder import GraphBuilder
            
            graph_builder_kwargs = {
                'k': dataset_params.get('k_neighbors'),
                'k_farthest': dataset_params.get('k_farthest'), 
                'k_random': dataset_params.get('k_random'),
                'max_edge_dist': dataset_params.get('max_edge_dist'),
                'num_rbf_3d': dataset_params.get('num_rbf_3d'),
                'num_rbf_seq': dataset_params.get('num_rbf_seq'),
                'no_source_indicator': dataset_params.get('no_source_indicator'),
                'rbf_3d_min': dataset_params.get('rbf_3d_min'),
                'rbf_3d_max': dataset_params.get('rbf_3d_max'),
                'rbf_3d_spacing': dataset_params.get('rbf_3d_spacing'),
            }
            
            # Handle distance vs neighbor based modes
            if dataset_params.get('max_edge_dist') is not None:
                # Distance-based mode: ensure k parameters are explicitly None
                graph_builder_kwargs.update({
                    'k': None,
                    'k_farthest': None,
                    'k_random': None,
                })
                # Keep None values for k parameters in distance-based mode
                graph_builder_kwargs = {k: v for k, v in graph_builder_kwargs.items() 
                                      if v is not None or k in ['k', 'k_farthest', 'k_random']}
            else:
                # Remove None values in neighbor-based mode
                graph_builder_kwargs = {k: v for k, v in graph_builder_kwargs.items() if v is not None}
            
            # Create graph builder
            graph_builder = GraphBuilder(**graph_builder_kwargs)
        
        # Build graph from protein entry (need to add time parameter)
        time_param = 0.0  # Use default time for structure loading
        
        # Debug: check if filtering is happening
        if verbose and 'seq' in protein_entry:
            original_seq_len = len(protein_entry['seq'])
            print(f"     About to build graph from protein entry with {original_seq_len} residues")
        
        graph_data = graph_builder.build_from_dict(protein_entry, time_param, af_filter_mode=False)
        
        # Add use_virtual_node flag
        graph_data.use_virtual_node = dataset_params.get('use_virtual_node', False)
        
        if verbose:
            seq_len = len(graph_data.filtered_seq) if hasattr(graph_data, 'filtered_seq') else 'unknown'
            print(f"     Successfully loaded graph with sequence length: {seq_len}")
            
            # Debug: compare original vs filtered sequence
            if hasattr(graph_data, 'filtered_seq') and 'seq' in protein_entry:
                filtered_seq = graph_data.filtered_seq
                original_seq = protein_entry['seq']
                print(f"     Sequence comparison:")
                print(f"       Original: {len(original_seq)} residues")
                print(f"       Filtered: {len(filtered_seq)} residues")
                print(f"       Difference: {len(original_seq) - len(filtered_seq)} residues removed")
                
                if len(filtered_seq) >= 146:
                    print(f"       Filtered seq around pos 146: {filtered_seq[140:151]}")
                    
                # Show which residues were removed
                if len(original_seq) != len(filtered_seq):
                    print(f"     WARNING: FILTERING DETECTED - this should not happen for ProteinGym!")
                    print(f"       This suggests missing coordinates or quality issues")
        
        return graph_data
        
    except Exception as e:
        if verbose:
            print(f"     Error loading {protein_id} from map_pkl: {e}")
        return None


def load_pdb_uniprot_mapping(mapping_path: str, verbose: bool = False) -> Dict[str, str]:
    """Load PDB-UniProt mapping from JSON file."""
    if not mapping_path or not Path(mapping_path).exists():
        if verbose:
            print("No PDB-UniProt mapping file provided or found")
        return {}
    
    try:
        if verbose:
            print(f"Loading PDB-UniProt mapping from {mapping_path}")
        
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        
        if verbose:
            print(f"  Loaded {len(mapping)} PDB-UniProt mappings")
        
        return mapping
        
    except Exception as e:
        if verbose:
            print(f"  WARNING: Failed to load mapping file: {e}")
        return {}


def sample_chain_inpainting_with_trajectory(model, data, T=8.0, t_min=0.0, steps=20, K=21,
                                           full_sequence=None, structure_sequence=None,
                                           mask_positions=None, known_sequence=None, 
                                           mask_ratio=0.3, verbose=False, args=None):
    """
    Generates amino acid sequence with inpainting while tracking trajectories for masked positions only.
    
    Args:
        model: Trained DFM model
        data: Protein structure graph
        T: Starting time (max noise)
        t_min: Minimum time (initial noise level)
        steps: Number of denoising steps
        K: Number of amino acid classes
        full_sequence: Full sequence string (for alignment)
        structure_sequence: Structure sequence string (from dataset)
        mask_positions: List of positions in FULL sequence to predict
        known_sequence: Template string with 'X' for positions to predict
        mask_ratio: Fraction to randomly mask if no specific positions given
        verbose: Print detailed information
        args: Arguments object with sampling parameters
        
    Returns:
        tuple: (final_probabilities, predicted_sequence, inpainting_mask, alignment_info, evaluation_metrics, trajectory_data)
    """
    from training.collate import collate_fn
    
    device = next(model.parameters()).device
    model.eval()
    
    # Validate inputs
    if full_sequence is None and structure_sequence is None:
        raise ValueError("Either full_sequence or structure_sequence must be provided")
    
    if mask_positions is None and known_sequence is None and mask_ratio <= 0:
        raise ValueError("Must specify either mask_positions, known_sequence, or positive mask_ratio")
    
    # Setup batched data - follow exact pattern from sample.py
    dummy_y = torch.zeros(1, K)
    dummy_mask = torch.ones(1, dtype=torch.bool)
    dummy_time = torch.tensor(0.0)  # Add dummy time for collate_fn compatibility
    
    try:
        batched_data, y_pad, mask_pad, time_batch = collate_fn([(data, dummy_y, dummy_mask, dummy_time)])
        batched_data = batched_data.to(device)
    except Exception as e:
        raise ValueError(f"Error preparing batched data: {e}")
    
    # Handle virtual nodes - follow exact pattern from sample.py
    total_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x_s.size(0)
    use_virtual_node = getattr(data, 'use_virtual_node', False)
    
    if use_virtual_node:
        N = total_nodes - 1  # Exclude virtual node
        if verbose:
            print(f"Using virtual nodes: {total_nodes} total nodes, {N} real residues")
    else:
        N = total_nodes
        if verbose:
            print(f"No virtual nodes: {N} residues")
    
    if N <= 0:
        raise ValueError(f"No real nodes found. Total nodes: {total_nodes}, use_virtual_node: {use_virtual_node}")
    
    # Get the ground truth sequence from the data for conditioning
    ground_truth_onehot = None
    true_sequence_indices = None
    
    # Try to get sequence from different sources - follow pattern from sample.py
    if hasattr(data, 'filtered_seq') and data.filtered_seq is not None:
        filtered_seq = data.filtered_seq[:N]  # Exclude virtual node if present
        ground_truth_onehot = torch.zeros(N, K, device=device)
        for i, aa in enumerate(filtered_seq):
            if aa in SINGLE_TO_TRIPLE:
                aa3 = SINGLE_TO_TRIPLE[aa]
                if aa3 in AA_TO_IDX:
                    ground_truth_onehot[i, AA_TO_IDX[aa3]] = 1.0
                else:
                    ground_truth_onehot[i, 20] = 1.0  # Unknown
            else:
                ground_truth_onehot[i, 20] = 1.0  # Unknown
    elif hasattr(data, 'y') and data.y is not None:
        ground_truth_onehot = data.y[:N].to(device)
    else:
        raise ValueError("No ground truth sequence found in data object")
    
    # Create inpainting mask - pass device parameter for PyTorch tensors
    try:
        mask_info = create_inpainting_mask_with_alignment(
            full_sequence, structure_sequence, mask_positions, known_sequence, 
            mask_ratio, verbose, device=device
        )
    except Exception as e:
        raise ValueError(f"Error creating inpainting mask: {e}")
    
    inpainting_mask = mask_info['mask']  # Already a PyTorch tensor
    alignment_info = mask_info['alignment_info']
    
    if inpainting_mask.sum().item() == 0:
        raise ValueError("No positions are masked for inpainting")
    
    if verbose:
        print(f"Inpainting mask: {inpainting_mask.sum().item()}/{len(inpainting_mask)} positions masked")
        masked_positions = torch.where(inpainting_mask)[0].tolist()
        print(f"Masked positions: {masked_positions}")
    
    # Override ground truth with custom known sequence if provided
    if known_sequence is not None and ground_truth_onehot is not None:
        for i, char in enumerate(known_sequence):
            if i < N and char != 'X' and char.upper() in SINGLE_TO_TRIPLE:
                aa3 = SINGLE_TO_TRIPLE[char.upper()]
                if aa3 in AA_TO_IDX:
                    ground_truth_onehot[i] = 0.0  # Clear existing
                    ground_truth_onehot[i, AA_TO_IDX[aa3]] = 1.0
    
    # Initialize sequence: known positions get ground truth, masked positions get noise
    # Extract dirichlet_concentration and use_c_factor from args with proper defaults
    dirichlet_concentration = getattr(args, 'dirichlet_concentration', 20.0) if args else 20.0
    use_c_factor = getattr(args, 'use_c_factor', False) if args else False
    
    try:
        dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))
        x = dirichlet_dist.sample((1, N))  # Shape: [1, N, K]
    except Exception as e:
        raise ValueError(f"Error initializing sequence distribution: {e}")

    # Set known positions to ground truth if available
    if ground_truth_onehot is not None:
        # Create inverted mask for known positions (not masked = known)
        known_mask = ~inpainting_mask  # Known positions
        x[0, known_mask] = ground_truth_onehot[known_mask]
    
    # Time steps - go from low t (noisy) to high t (clean) - follow exact pattern from sample.py
    times = torch.linspace(t_min, T, steps, device=device)
    dt = (T - t_min) / (steps - 1)
    
    if verbose:
        print(f"Time integration: t={t_min:.1f} -> {T:.1f}, dt={dt:.4f}")
        print(f"Starting inpainting with trajectory tracking for {inpainting_mask.sum().item()} masked positions")
    
    # Initialize trajectory tracking - ONLY FOR MASKED POSITIONS
    masked_positions = torch.where(inpainting_mask)[0].tolist()
    trajectory_data = {
        'time_points': [],
        'positions': {},
        'masked_positions_only': True,  # Flag to indicate this is masked-only trajectory
        'total_positions': N,
        'masked_count': len(masked_positions)
    }
    
    # Initialize position data with detailed amino acid breakdown - ONLY FOR MASKED POSITIONS
    for pos in masked_positions:
        trajectory_data['positions'][str(pos)] = {
            'time_points': [],
            'most_likely_aa': [],
            'probabilities': [],
            'detailed_breakdown': []
        }
    
    with torch.no_grad():
        time_steps = tqdm(enumerate(times), total=len(times), desc="Inpainting with trajectory", disable=not verbose)
        for i, t_val in time_steps:
            t = torch.full((1,), t_val, device=device)
            
            # Store trajectory data for masked positions only
            current_probs = x.squeeze(0).cpu().numpy()  # [N, K]
            current_most_likely = x.argmax(-1).squeeze(0).cpu().tolist()  # [N]
            current_max_probs = x.max(-1)[0].squeeze(0).cpu().tolist()  # [N]
            
            # Update progress bar with current step info
            current_entropy = -(x * torch.log(x + 1e-10)).sum(-1).mean().item()
            current_avg_max_prob = x.max(-1)[0].mean().item()
            masked_count = inpainting_mask.sum().item()
            time_steps.set_postfix({
                't': f'{t_val:.3f}',
                'entropy': f'{current_entropy:.4f}',
                'avg_max_prob': f'{current_avg_max_prob:.4f}',
                'masked': f'{masked_count}'
            })
            
            # Skip velocity prediction on last step
            if i == len(times) - 1:
                break
                
            # Get model prediction
            model_output = model(batched_data, t, x)
            
            # Handle DSSP multitask output
            if isinstance(model_output, tuple):
                position_logits = model_output[0]  # Use only sequence logits for sampling
                if verbose and i == 0:
                    print("  Model returns tuple (sequence + DSSP), using sequence logits for sampling")
            else:
                position_logits = model_output
            
            # Convert to predicted target distribution
            # Apply time-dependent temperature if requested
            if args and getattr(args, 'time_as_temperature', False):
                # Temperature starts high (at t_min) and decreases as we approach t_max
                flow_temp = T - t_val + 0.1
            else:
                flow_temp = args.flow_temp if args else 1.0
            
            predicted_target = torch.softmax(position_logits/flow_temp, dim=-1)
            
            # Extract only real node predictions and ensure same shape as x
            if use_virtual_node:
                predicted_target_real = predicted_target[:N, :].unsqueeze(0)  # [N, K] -> [1, N, K]
            else:
                predicted_target_real = predicted_target.unsqueeze(0)  # [N, K] -> [1, N, K]
            
            # Compute analytical velocity using conditional flow
            cond_flow = model.cond_flow
            v_analytical = cond_flow.velocity(
                x,
                predicted_target_real,
                t,
                use_virtual_node=use_virtual_node,
                use_smoothed_targets=getattr(args, 'use_smoothed_targets', False),
                use_c_factor=use_c_factor
            )
            
            # Record detailed trajectory data for MASKED POSITIONS ONLY
            predicted_probs = predicted_target_real.squeeze(0).cpu().numpy()  # [N, K]
            velocities = v_analytical.squeeze(0).cpu().numpy()  # [N, K]
            
            # Update trajectory data with detailed breakdown - ONLY FOR MASKED POSITIONS
            trajectory_data['time_points'].append(float(t_val))
            for pos in masked_positions:
                trajectory_data['positions'][str(pos)]['time_points'].append(float(t_val))
                trajectory_data['positions'][str(pos)]['most_likely_aa'].append(current_most_likely[pos])
                trajectory_data['positions'][str(pos)]['probabilities'].append(current_max_probs[pos])
                
                # Store detailed amino acid breakdown with velocity components
                aa_breakdown = {}
                aa_names = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
                for k, aa_name in enumerate(aa_names):
                    aa_breakdown[aa_name] = {
                        'current_prob': round(float(current_probs[pos, k]), 6),
                        'predicted_prob': round(float(predicted_probs[pos, k]), 6) if k < predicted_probs.shape[1] else 0.0,
                        'velocity_component': round(float(velocities[pos, k]), 6) if k < velocities.shape[1] else 0.0,
                        'c_factor_component': 0.0  # Placeholder
                    }
                trajectory_data['positions'][str(pos)]['detailed_breakdown'].append(aa_breakdown)
            
            if verbose and (i + 1) % 10 == 0:
                current_entropy = -(x * torch.log(x + 1e-10)).sum(-1).mean().item()
                current_avg_max_prob = x.max(-1)[0].mean().item()
                print(f"  Step {i+1}/{steps}: t={t_val:.3f}, entropy={current_entropy:.4f}, avg_max_prob={current_avg_max_prob:.4f}")
                
                # Show prediction progress for a few masked positions
                if len(masked_positions) > 0:
                    sample_pos = masked_positions[0]
                    pred_aa = current_most_likely[sample_pos]
                    pred_prob = current_max_probs[sample_pos]
                    print(f"    Sample masked pos {sample_pos}: {IDX_TO_AA[pred_aa]} ({pred_prob:.3f})")
            
            # Apply velocity only to masked positions, keep known positions fixed
            x_new = x + dt * v_analytical
            x_new = simplex_proj(x_new)
            
            # Restore known positions to ground truth
            if ground_truth_onehot is not None:
                known_mask = ~inpainting_mask  # Known positions
                x_new[0, known_mask] = ground_truth_onehot[known_mask]
            
            x = x_new
    
    final_probabilities = x.squeeze(0)
    predicted_sequence = final_probabilities.argmax(-1).tolist()
    
    # Compute evaluation metrics if ground truth is available - enhanced error handling
    evaluation_metrics = {}
    try:
        if ground_truth_onehot is not None:
            evaluation_metrics = compute_sampling_metrics(
                final_probabilities, ground_truth_onehot, data, model, args, device, use_virtual_node, K
            )
            
            # Add true sequence for ground truth comparison
            true_classes = ground_truth_onehot.argmax(-1).tolist()
            true_sequence_single = ''.join([THREE_TO_ONE.get(IDX_TO_AA[idx], 'X') if 0 <= idx < len(IDX_TO_AA) else 'X' for idx in true_classes])
            evaluation_metrics['true_sequence'] = true_sequence_single
            
            # Add inpainting-specific metrics
            if inpainting_mask is not None:
                masked_positions_list = torch.where(inpainting_mask)[0].tolist()
                masked_predictions = [predicted_sequence[i] for i in masked_positions_list]
                masked_ground_truth = ground_truth_onehot[inpainting_mask].argmax(-1).tolist()
                
                masked_correct = sum(p == t for p, t in zip(masked_predictions, masked_ground_truth))
                masked_accuracy = masked_correct / len(masked_predictions) * 100 if masked_predictions else 0.0
                
                evaluation_metrics['masked_accuracy'] = masked_accuracy
                evaluation_metrics['masked_positions_count'] = len(masked_positions_list)
                evaluation_metrics['total_positions'] = N
    except Exception as e:
        print(f"Warning: Could not compute evaluation metrics: {e}")
        evaluation_metrics = {'accuracy': 0.0, 'masked_accuracy': 0.0}
    
    # Enhanced final results reporting
    if verbose:
        print(f"\nINPAINTING RESULTS:")
        print(f"  Total positions: {N}")
        print(f"  Masked positions: {inpainting_mask.sum().item()}")
        print(f"  Known positions: {(~inpainting_mask).sum().item()}")
        print(f"  Overall accuracy: {evaluation_metrics.get('accuracy', 0):.2f}%")
        print(f"  Masked-only accuracy: {evaluation_metrics.get('masked_accuracy', 0):.2f}%")
        
        # Show some example predictions for masked positions with probability distributions
        masked_positions_list = torch.where(inpainting_mask)[0].tolist()
        print(f"  Sample predictions (first 5 masked positions):")
        for i, pos in enumerate(masked_positions_list[:5]):
            pred_idx = predicted_sequence[pos]
            pred_aa = IDX_TO_AA[pred_idx] if 0 <= pred_idx < len(IDX_TO_AA) else 'XXX'
            pred_single = THREE_TO_ONE.get(pred_aa, 'X')
            pred_prob = final_probabilities[pos, pred_idx].item()
            
            # Show basic prediction
            print(f"    Position {pos+1}: {pred_single} ({pred_aa}) - {pred_prob:.3f}")
            
            # Show full probability distribution for this position
            prob_dist_str = format_probability_distribution(final_probabilities[pos], K=K, top_n=3)
            print(prob_dist_str)
    
    return final_probabilities, predicted_sequence, inpainting_mask, alignment_info, evaluation_metrics, trajectory_data


def sample_chain_inpainting(model, data, T=8.0, t_min=0.0, steps=20, K=21,
                           full_sequence=None, structure_sequence=None,
                           mask_positions=None, known_sequence=None, 
                           mask_ratio=0.3, verbose=False, args=None):
    """
    Generates amino acid sequence with inpainting: predicts masked positions
    while conditioning on known positions.
    
    Args:
        model: Trained DFM model
        data: Protein structure graph
        T: Starting time (max noise)
        t_min: Minimum time (initial noise level)
        steps: Number of denoising steps
        K: Number of amino acid classes
        full_sequence: Full sequence string (for alignment)
        structure_sequence: Structure sequence string (from dataset)
        mask_positions: List of positions in FULL sequence to predict
        known_sequence: Template string with 'X' for positions to predict
        mask_ratio: Fraction to randomly mask if no specific positions given
        verbose: Print detailed information
        args: Arguments object with sampling parameters
        
    Returns:
        tuple: (final_probabilities, predicted_sequence, inpainting_mask, alignment_info, evaluation_metrics)
    """
    from training.collate import collate_fn
    
    device = next(model.parameters()).device
    model.eval()
    
    # Validate inputs
    if full_sequence is None and structure_sequence is None:
        raise ValueError("Either full_sequence or structure_sequence must be provided")
    
    if mask_positions is None and known_sequence is None and mask_ratio <= 0:
        raise ValueError("Must specify either mask_positions, known_sequence, or positive mask_ratio")
    
    # Setup batched data - follow exact pattern from sample.py
    dummy_y = torch.zeros(1, K)
    dummy_mask = torch.ones(1, dtype=torch.bool)
    dummy_time = torch.tensor(0.0)  # Add dummy time for collate_fn compatibility
    
    try:
        batched_data, y_pad, mask_pad, time_batch = collate_fn([(data, dummy_y, dummy_mask, dummy_time)])
        batched_data = batched_data.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare batched data: {e}")
    
    # Handle virtual nodes - follow exact pattern from sample.py
    total_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x_s.size(0)
    use_virtual_node = getattr(data, 'use_virtual_node', False)
    
    if use_virtual_node:
        N = total_nodes - 1  # Exclude virtual node
        if verbose:
            print(f"Using virtual node: {total_nodes} total nodes, {N} real nodes")
    else:
        N = total_nodes
        if verbose:
            print(f"No virtual node: {N} real nodes")
    
    if N <= 0:
        raise ValueError(f"No real nodes found. Total nodes: {total_nodes}, use_virtual_node: {use_virtual_node}")
    # Get the ground truth sequence from the data for conditioning
    ground_truth_onehot = None
    true_sequence_indices = None
    
    # Try to get sequence from different sources - follow pattern from sample.py
    if hasattr(data, 'filtered_seq') and data.filtered_seq is not None:
        # Use filtered sequence from graph builder (preferred)
        filtered_seq = data.filtered_seq[:N]  # Exclude virtual node if present
        
        try:
            true_sequence_indices = []
            for aa in filtered_seq:
                if aa in SINGLE_TO_TRIPLE:
                    aa3 = SINGLE_TO_TRIPLE[aa]
                    if aa3 in AA_TO_IDX:
                        true_sequence_indices.append(AA_TO_IDX[aa3])
                    else:
                        true_sequence_indices.append(20)  # Unknown
                else:
                    true_sequence_indices.append(20)  # Unknown
            
            true_sequence_indices = torch.tensor(true_sequence_indices, device=device)
            
            # Create ground truth one-hot tensor for conditioning
            ground_truth_onehot = torch.zeros(N, K, device=device)
            for i, seq_idx in enumerate(true_sequence_indices):
                ground_truth_onehot[i, seq_idx] = 1.0
            
            if verbose:
                print(f"Using filtered_seq from data: {len(filtered_seq)} amino acids")
        except Exception as e:
            raise RuntimeError(f"Failed to process filtered_seq: {e}")
    elif hasattr(data, 'y') and data.y is not None:
        # Fallback to y if available
        try:
            ground_truth_onehot = data.y[:N].to(device)
            if verbose:
                print(f"Using y from data: {ground_truth_onehot.shape[0]} amino acids")
        except Exception as e:
            raise RuntimeError(f"Failed to process data.y: {e}")
    else:
        raise ValueError("Ground truth sequence not available for inpainting. Need data.filtered_seq or data.y")
    
    # Create inpainting mask - pass device parameter for PyTorch tensors
    try:
        mask_info = create_inpainting_mask_with_alignment(
            full_sequence, structure_sequence if structure_sequence else ''.join(filtered_seq),
            mask_positions, known_sequence, mask_ratio, verbose, device=device
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create inpainting mask: {e}")
    
    inpainting_mask = mask_info['mask']  # Already a PyTorch tensor
    alignment_info = mask_info['alignment_info']
    
    if inpainting_mask.sum().item() == 0:
        raise ValueError("No positions to inpaint. Check mask_positions, known_sequence, or mask_ratio.")
    
    if verbose:
        print(f"Inpainting setup: {inpainting_mask.sum().item()}/{N} positions will be predicted")
    
    # Override ground truth with custom known sequence if provided
    if known_sequence is not None and ground_truth_onehot is not None:
        try:
            for i, aa in enumerate(known_sequence):
                if i < N and aa.upper() != 'X':
                    if aa.upper() in SINGLE_TO_TRIPLE:
                        aa3 = SINGLE_TO_TRIPLE[aa.upper()]
                        if aa3 in AA_TO_IDX:
                            ground_truth_onehot[i, :] = 0
                            ground_truth_onehot[i, AA_TO_IDX[aa3]] = 1.0
            if verbose:
                print(f"Applied custom known sequence template")
        except Exception as e:
            raise RuntimeError(f"Failed to apply known sequence: {e}")
    
    # Initialize sequence: known positions get ground truth, masked positions get noise
    # Extract dirichlet_concentration and use_c_factor from args with proper defaults
    dirichlet_concentration = getattr(args, 'dirichlet_concentration', 20.0) if args else 20.0
    use_c_factor = getattr(args, 'use_c_factor', False) if args else False
    
    try:
        dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))
        x = dirichlet_dist.sample((1, N))  # [1, N, K] - start with noise everywhere
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Dirichlet distribution: {e}")

    # Set known positions to ground truth if available
    if ground_truth_onehot is not None:
        x[0, ~inpainting_mask] = ground_truth_onehot[~inpainting_mask]
        
        if verbose:
            masked_positions = inpainting_mask.nonzero().squeeze().tolist()
            if not isinstance(masked_positions, list):
                masked_positions = [masked_positions]
            print(f"Initialized: {len(masked_positions)} masked positions will be predicted, others frozen to ground truth")
    
    # Time steps - go from low t (noisy) to high t (clean) - follow exact pattern from sample.py
    times = torch.linspace(t_min, T, steps, device=device)
    dt = (T - t_min) / (steps - 1)
    
    if verbose:
        print(f"Starting reverse sampling from t={t_min:.1f} to t={T} in {steps} steps")
        print(f"Inpainting {inpainting_mask.sum().item()}/{N} positions")
    
    with torch.no_grad():
        time_steps = tqdm(enumerate(times), total=len(times), desc="Inpainting sampling", disable=not verbose)
        for i, t_val in time_steps:
            t = torch.full((1,), t_val, device=device)
            
            # Update progress bar with current step info
            current_entropy = -(x * torch.log(x + 1e-10)).sum(-1).mean().item()
            masked_count = inpainting_mask.sum().item()
            time_steps.set_postfix({
                't': f'{t_val:.3f}',
                'entropy': f'{current_entropy:.4f}',
                'masked': f'{masked_count}'
            })
            
            try:
                # Get model predictions - follow exact pattern from sample.py
                model_output = model(batched_data, t, x)
                
                # Handle both single output and tuple output (for DSSP models)
                if isinstance(model_output, tuple):
                    position_logits = model_output[0]  # First element is sequence logits
                else:
                    position_logits = model_output
                
                # Convert to predicted target distribution
                # Apply time-dependent temperature if requested
                if args and getattr(args, 'time_as_temperature', False):
                    # Temperature starts high (at t_min) and decreases as we approach t_max
                    flow_temp = T - t_val + 0.1
                else:
                    flow_temp = getattr(args, 'flow_temp', 1.0) if args else 1.0
                predicted_target = torch.softmax(position_logits / flow_temp, dim=-1)
                
                # Extract only real node predictions and ensure same shape as x - follow exact pattern from sample.py
                if use_virtual_node:
                    # position_logits has shape [total_nodes, K], need to slice out real nodes only
                    # For single structure: real nodes are indices 0 to N-1, virtual node is at index N
                    predicted_target_real = predicted_target[:N, :].unsqueeze(0)  # [N, K] -> [1, N, K]
                else:
                    # If no virtual node, still need to add batch dimension to match x shape
                    predicted_target_real = predicted_target.unsqueeze(0)  # [N, K] -> [1, N, K]
                
                # Compute analytical velocity using conditional flow - follow exact pattern from sample.py
                cond_flow = model.cond_flow
                v_analytical = cond_flow.velocity(
                    x,
                    predicted_target_real,
                    t,
                    use_virtual_node=use_virtual_node,
                    use_smoothed_targets=getattr(args, 'use_smoothed_targets', False),
                    use_c_factor=use_c_factor
                )
                
                # Use analytical velocity for integration
                v_processed = v_analytical
                
                # Update sequence using analytical velocity
                x_new = x + dt * v_processed
                x_new = simplex_proj(x_new)  # Use proper simplex projection
                
                # For inpainting: only update masked positions, keep known positions fixed
                # Always freeze known positions if ground truth is available
                if ground_truth_onehot is not None:
                    x_new[0, ~inpainting_mask] = ground_truth_onehot[~inpainting_mask]
                
                x = x_new
                
            except Exception as e:
                raise RuntimeError(f"Failed at sampling step {i+1}/{steps} (t={t_val:.3f}): {e}")
            
            if verbose and (i + 1) % 5 == 0:
                current_entropy = -(x * torch.log(x + 1e-10)).sum(-1).mean().item()
                print(f"  Step {i+1}/{steps}: t={t_val:.3f}, entropy={current_entropy:.4f}")
    
    final_probabilities = x.squeeze(0)
    predicted_sequence = final_probabilities.argmax(-1).tolist()
    
    # Compute evaluation metrics if ground truth is available - enhanced error handling
    evaluation_metrics = {}
    try:
        if ground_truth_onehot is not None and ground_truth_onehot.sum().item() > 0:
            # Calculate overall accuracy using existing function
            evaluation_metrics = compute_sampling_metrics(
                final_probabilities, ground_truth_onehot, data, model, args, device, use_virtual_node, K
            )
            
            # Add true sequence for ground truth comparison
            true_classes = ground_truth_onehot.argmax(-1).tolist()
            true_sequence_single = ''.join([THREE_TO_ONE.get(IDX_TO_AA[idx], 'X') if 0 <= idx < len(IDX_TO_AA) else 'X' for idx in true_classes])
            evaluation_metrics['true_sequence'] = true_sequence_single
            
            # Calculate masked position accuracy specifically
            true_sequence = ground_truth_onehot.argmax(-1)
            predicted_sequence_tensor = final_probabilities.argmax(-1)
            
            # Get masked positions
            masked_positions = inpainting_mask.nonzero().squeeze()
            if masked_positions.numel() > 0:
                if masked_positions.dim() == 0:  # Single position
                    masked_positions = masked_positions.unsqueeze(0)
                
                # Calculate accuracy only for masked positions
                masked_true = true_sequence[masked_positions]
                masked_pred = predicted_sequence_tensor[masked_positions]
                masked_correct = (masked_true == masked_pred).sum().item()
                masked_total = len(masked_positions)
                masked_accuracy = masked_correct / masked_total if masked_total > 0 else 0.0
                
                # Add masked position metrics
                evaluation_metrics['masked_accuracy'] = masked_accuracy
                evaluation_metrics['masked_correct'] = masked_correct
                evaluation_metrics['masked_total'] = masked_total
                evaluation_metrics['masked_positions'] = masked_positions.tolist()
                
                # Add true sequence for plotting compatibility
                evaluation_metrics['true_sequence'] = true_sequence.tolist()
            else:
                evaluation_metrics['masked_accuracy'] = 1.0  # No masked positions means perfect
                evaluation_metrics['masked_correct'] = 0
                evaluation_metrics['masked_total'] = 0
                evaluation_metrics['masked_positions'] = []
                
                # Add true sequence for plotting compatibility if available
                if ground_truth_onehot is not None:
                    true_sequence = ground_truth_onehot.argmax(-1)
                    evaluation_metrics['true_sequence'] = true_sequence.tolist()
            
            if verbose:
                print(f"Evaluation metrics: overall_accuracy={evaluation_metrics.get('accuracy', 0.0):.4f}, masked_accuracy={evaluation_metrics.get('masked_accuracy', 0.0):.4f} ({evaluation_metrics.get('masked_correct', 0)}/{evaluation_metrics.get('masked_total', 0)})")
    except Exception as e:
        if verbose:
            print(f"WARNING: Could not compute evaluation metrics: {e}")
        # Provide safe defaults
        evaluation_metrics = {
            'accuracy': 0.0, 
            'masked_accuracy': 0.0,
            'masked_correct': 0,
            'masked_total': 0,
            'masked_positions': [],
            'cce_loss_hard': float('inf'), 
            'cce_loss_smooth': float('inf')
        }
    
    # Enhanced final results reporting
    if verbose:
        print(f"\nFinal inpainting results:")
        try:
            masked_positions = inpainting_mask.nonzero().squeeze().tolist()
            if not isinstance(masked_positions, list):
                masked_positions = [masked_positions]
            print(f"  Masked positions: {masked_positions}")
            
            # Convert indices to amino acids for display
            # Ensure we have proper integers, not tensors
            sequence_for_display = []
            for idx in predicted_sequence:
                if hasattr(idx, 'item'):
                    sequence_for_display.append(int(idx.item()))
                else:
                    sequence_for_display.append(int(idx))
            predicted_aa_single = ''.join([THREE_TO_ONE.get(IDX_TO_AA[idx], 'X') if 0 <= idx < len(IDX_TO_AA) else 'X' for idx in sequence_for_display])
            print(f"  Predicted sequence (single): {predicted_aa_single}")
            
            if ground_truth_onehot is not None:
                true_sequence_indices_list = ground_truth_onehot.argmax(-1).cpu().tolist()
                true_aa_single = ''.join([THREE_TO_ONE.get(IDX_TO_AA[idx], 'X') if 0 <= idx < len(IDX_TO_AA) else 'X' for idx in true_sequence_indices_list])
                print(f"  True sequence (single):      {true_aa_single}")
                
                # Show detailed comparison for masked positions with probability distributions
                print(f"\nMasked position details:")
                for pos in masked_positions:
                    if 0 <= pos < len(predicted_sequence):
                        pred_idx = predicted_sequence[pos]
                        true_idx = true_sequence_indices_list[pos] if pos < len(true_sequence_indices_list) else -1
                        pred_aa = THREE_TO_ONE.get(IDX_TO_AA[pred_idx], 'X') if 0 <= pred_idx < len(IDX_TO_AA) else 'X'
                        true_aa = THREE_TO_ONE.get(IDX_TO_AA[true_idx], 'X') if 0 <= true_idx < len(IDX_TO_AA) else 'X'
                        match = "MATCH" if pred_aa == true_aa else "MISMATCH"
                        
                        print(f"    Position {pos+1}: {pred_aa} vs {true_aa} {match}")
                        
                        # Show probability distribution for this position
                        try:
                            prob_dist_str = format_probability_distribution(final_probabilities[pos], K=21, top_n=3)
                            print(prob_dist_str)
                            print()  # Add blank line for readability
                        except Exception as e:
                            print(f"    Warning: Could not display probability distribution for position {pos+1}: {e}")
                            print()
                
                # Show accuracy summary
                if 'masked_accuracy' in evaluation_metrics:
                    masked_acc = evaluation_metrics['masked_accuracy']
                    masked_correct = evaluation_metrics.get('masked_correct', 0)
                    masked_total = evaluation_metrics.get('masked_total', 0)
                    overall_acc = evaluation_metrics.get('accuracy', 0.0)
                    print(f"\nAccuracy Summary:")
                    print(f"  Masked positions: {masked_acc:.4f} ({masked_correct}/{masked_total})")
                    print(f"  Overall sequence: {overall_acc:.4f}")
        except Exception as e:
            print(f"  Warning: Could not display detailed results: {e}")
    
    return final_probabilities, predicted_sequence, inpainting_mask.cpu(), alignment_info, evaluation_metrics


def sample_chain_inpainting_with_ensemble(model, data, T=8.0, t_min=0.0, steps=20, K=21,
                                          full_sequence=None, structure_sequence=None,
                                          mask_positions=None, known_sequence=None, 
                                          mask_ratio=0.3, verbose=False, args=None,
                                          ensemble_size=1, ensemble_consensus_strength=0.7, 
                                          ensemble_method='arithmetic', structure_noise_mag_std=0.0,
                                          uncertainty_struct_noise_scaling=False):
    """
    Generates amino acid sequence with inpainting using ensemble consensus: predicts masked positions
    while conditioning on known positions, optionally using structural noise and consensus averaging.
    
    Args:
        model: Trained DFM model
        data: Protein structure graph
        T: Starting time (max noise)
        t_min: Minimum time (initial noise level)
        steps: Number of denoising steps
        K: Number of amino acid classes
        full_sequence: Full sequence string (for alignment)
        structure_sequence: Structure sequence string (from dataset)
        mask_positions: List of positions in FULL sequence to predict
        known_sequence: Template string with 'X' for positions to predict
        mask_ratio: Fraction to randomly mask if no specific positions given
        verbose: Print detailed information
        args: Arguments object with sampling parameters
        ensemble_size: Number of ensemble members (1 = no ensemble)
        ensemble_consensus_strength: Consensus strength parameter
        ensemble_method: 'arithmetic' or 'geometric' ensemble averaging
        structure_noise_mag_std: Standard deviation for structural noise
        uncertainty_struct_noise_scaling: Enable uncertainty-based noise scaling
        
    Returns:
        tuple: (final_probabilities, predicted_sequence, inpainting_mask, alignment_info, evaluation_metrics)
    """
    if ensemble_size <= 1:
        # No ensemble, use regular inpainting
        return sample_chain_inpainting(
            model, data, T, t_min, steps, K, full_sequence, structure_sequence,
            mask_positions, known_sequence, mask_ratio, verbose, args
        )
    
    if verbose:
        print(f"Starting ensemble inpainting with {ensemble_size} members")
        print(f"Ensemble method: {ensemble_method}, consensus strength: {ensemble_consensus_strength}")
        if structure_noise_mag_std > 0:
            print(f"Structural noise std: {structure_noise_mag_std}")
    
    # Create structural ensemble if noise is requested
    if structure_noise_mag_std > 0.0:
        try:
            ensemble_data = create_structural_ensemble(
                data, ensemble_size, structure_noise_mag_std, 
                uncertainty_struct_noise_scaling, verbose
            )
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to create structural ensemble: {e}")
                print("Falling back to regular ensemble without structural noise")
            ensemble_data = [data] * ensemble_size
    else:
        # No structural noise, just replicate the data
        ensemble_data = [data] * ensemble_size
    
    # Run inpainting for each ensemble member
    ensemble_results = []
    ensemble_probabilities = []
    ensemble_sequences = []
    
    for i, member_data in enumerate(ensemble_data):
        if verbose:
            print(f"\nProcessing ensemble member {i+1}/{ensemble_size}")
        
        try:
            # Run inpainting on this ensemble member
            member_probs, member_seq, member_mask, member_alignment, member_metrics = sample_chain_inpainting(
                model, member_data, T, t_min, steps, K, full_sequence, structure_sequence,
                mask_positions, known_sequence, mask_ratio, verbose=False, args=args
            )
            
            ensemble_results.append({
                'probabilities': member_probs,
                'sequence': member_seq,
                'mask': member_mask,
                'alignment_info': member_alignment,
                'evaluation_metrics': member_metrics
            })
            
            ensemble_probabilities.append(member_probs.unsqueeze(0))  # Add batch dimension
            ensemble_sequences.append(member_seq)
            
        except Exception as e:
            if verbose:
                print(f"Warning: Ensemble member {i+1} failed: {e}")
            # Skip this member and continue
            continue
    
    if not ensemble_probabilities:
        raise RuntimeError("All ensemble members failed")
    
    if verbose:
        print(f"\nComputing ensemble consensus from {len(ensemble_probabilities)} successful members")
    
    # Stack probabilities for consensus computation
    ensemble_probs_tensor = torch.stack(ensemble_probabilities, dim=0)  # [ensemble_size, N, K]
    
    # Compute ensemble consensus
    try:
        consensus_probabilities = compute_ensemble_consensus(
            ensemble_probs_tensor, ensemble_consensus_strength, ensemble_method
        )
    except Exception as e:
        if verbose:
            print(f"Warning: Ensemble consensus failed: {e}")
            print("Falling back to arithmetic mean")
        consensus_probabilities = ensemble_probs_tensor.mean(dim=0)
    
    # Get final predictions from consensus
    final_probabilities = consensus_probabilities.squeeze(0) if consensus_probabilities.dim() == 3 else consensus_probabilities
    predicted_sequence = final_probabilities.argmax(-1).tolist()
    
    # Use the mask and alignment info from the first successful member
    inpainting_mask = ensemble_results[0]['mask']
    alignment_info = ensemble_results[0]['alignment_info']
    
    # Aggregate evaluation metrics from ensemble members
    evaluation_metrics = {}
    if ensemble_results and 'evaluation_metrics' in ensemble_results[0]:
        # Use the first member's metrics as base and add ensemble info
        evaluation_metrics = ensemble_results[0]['evaluation_metrics'].copy()
        
        # Add ensemble-specific metrics
        evaluation_metrics['ensemble_size'] = len(ensemble_results)
        evaluation_metrics['ensemble_method'] = ensemble_method
        evaluation_metrics['ensemble_consensus_strength'] = ensemble_consensus_strength
        
        # Compute individual member accuracies if available
        member_accuracies = []
        member_masked_accuracies = []
        for result in ensemble_results:
            if 'accuracy' in result['evaluation_metrics']:
                member_accuracies.append(result['evaluation_metrics']['accuracy'])
            if 'masked_accuracy' in result['evaluation_metrics']:
                member_masked_accuracies.append(result['evaluation_metrics']['masked_accuracy'])
        
        if member_accuracies:
            evaluation_metrics['ensemble_member_accuracies'] = member_accuracies
            evaluation_metrics['ensemble_accuracy_mean'] = np.mean(member_accuracies)
            evaluation_metrics['ensemble_accuracy_std'] = np.std(member_accuracies)
        
        if member_masked_accuracies:
            evaluation_metrics['ensemble_masked_accuracies'] = member_masked_accuracies
            evaluation_metrics['ensemble_masked_accuracy_mean'] = np.mean(member_masked_accuracies)
            evaluation_metrics['ensemble_masked_accuracy_std'] = np.std(member_masked_accuracies)
    
    if verbose:
        print(f"\nEnsemble inpainting completed:")
        print(f"  Successfully used {len(ensemble_results)} ensemble members")
        if 'ensemble_accuracy_mean' in evaluation_metrics:
            acc_mean = evaluation_metrics['ensemble_accuracy_mean']
            acc_std = evaluation_metrics['ensemble_accuracy_std']
            print(f"  Member accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        if 'ensemble_masked_accuracy_mean' in evaluation_metrics:
            masked_acc_mean = evaluation_metrics['ensemble_masked_accuracy_mean']
            masked_acc_std = evaluation_metrics['ensemble_masked_accuracy_std']
            print(f"  Member masked accuracy: {masked_acc_mean:.4f} ± {masked_acc_std:.4f}")
    
    return final_probabilities, predicted_sequence, inpainting_mask, alignment_info, evaluation_metrics


def load_model_for_inpainting(model_path, device='auto', verbose=False):
    """
    Load a trained DFM model for inpainting with proper parameter extraction.
    
    Uses the same utilities as sample.py to ensure compatibility.
    
    Args:
        model_path: Path to model checkpoint
        device: Device specification ('auto', 'cuda', 'cpu', or specific device)
        verbose: Print loading information
        
    Returns:
        Tuple of (model, extracted_args) where extracted_args contains
        checkpoint parameters for consistent sampling
    """
    # Import here to avoid circular imports
    from training.sample_utils import load_model_distributed
    
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    if verbose:
        print(f"Loading model for inpainting from: {model_path}")
        print(f"Using device: {device}")
    
    try:
        # Create a minimal args object for load_model_distributed
        class MinimalArgs:
            def __init__(self):
                self.verbose = verbose
        
        minimal_args = MinimalArgs()
        model, dataset_params = load_model_distributed(model_path, device, minimal_args)
        
        # Extract important parameters for inpainting
        extracted_args = type('Args', (), {
            'dirichlet_concentration': dataset_params.get('dirichlet_concentration', 20.0),
            'use_c_factor': dataset_params.get('use_c_factor', False),
            'flow_temp': dataset_params.get('flow_temp', 1.0),
            'verbose': verbose
        })()
        
        if verbose:
            print(f"Extracted parameters:")
            print(f"  dirichlet_concentration: {extracted_args.dirichlet_concentration}")
            print(f"  use_c_factor: {extracted_args.use_c_factor}")
            print(f"  flow_temp: {extracted_args.flow_temp}")
        
        return model, extracted_args
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model for inpainting: {e}")


def run_inpainting_inference(model_path, data=None, uniprot_id=None, pdb_id=None, 
                            full_sequence=None, structure_sequence=None,
                            mask_positions=None, known_sequence=None, mask_ratio=0.3,
                            T=8.0, t_min=0.0, steps=20, K=21, device='auto', verbose=False,
                            pdb_uniprot_mapping_path=None, args=None, enable_trajectory=False,
                            split_json_override=None, map_pkl_override=None):
    """
    High-level function to run inpainting inference using existing SamplingDataLoader infrastructure.
    
    Args:
        model_path: Path to trained model checkpoint
        data: Protein structure graph data (optional if using uniprot_id/pdb_id)
        uniprot_id: UniProt ID to fetch sequence and find structure
        pdb_id: PDB ID to find structure in dataset
        full_sequence: Full sequence string (for alignment)
        structure_sequence: Structure sequence string (from dataset)
        mask_positions: List of positions in FULL sequence to predict
        known_sequence: Template string with 'X' for positions to predict
        mask_ratio: Fraction to randomly mask if no specific positions given
        T: Starting time (max noise)
        t_min: Minimum time (initial noise level)
        steps: Number of denoising steps
        K: Number of amino acid classes
        device: Device specification
        verbose: Print detailed information
        pdb_uniprot_mapping_path: Path to PDB-UniProt mapping JSON file
        enable_trajectory: Whether to enable trajectory tracking for detailed analysis
        split_json_override: Override split_json path from command line
        map_pkl_override: Override map_pkl path from command line
        
    Returns:
        Dict containing results: predicted_sequence, probabilities, etc.
    """
    try:
        # Use existing SamplingCoordinator for proper model/dataset loading
        if verbose:
            print("Setting up SamplingCoordinator for proper model and dataset loading...")
        
        # Create sampling coordinator (will be overridden by checkpoint parameters)
        sampling_coordinator = SamplingCoordinator(
            model_path=model_path,
            dataset_path="",  # Will be extracted from checkpoint
            split='test'  # Changed to 'test' where 1kll.A is located
        )
        
        # Determine device
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Load model and dataset properly (uses existing infrastructure)
        # Create args object with missing RBF parameters for backwards compatibility
        class Args:
            def __init__(self):
                self.rbf_3d_min = 2.0
                self.rbf_3d_max = 350.0
                self.rbf_3d_spacing = 'exponential'
                # Use override parameters if provided, otherwise None (will be extracted from checkpoint)
                self.split_json = split_json_override
                self.map_pkl = map_pkl_override
        
        args_obj = Args()
        sampling_coordinator.load_model_and_dataset(device, args_obj)
        
        # Load PDB-UniProt mapping if provided
        pdb_uniprot_mapping = None
        if pdb_uniprot_mapping_path:
            pdb_uniprot_mapping = load_pdb_uniprot_mapping(pdb_uniprot_mapping_path, verbose)
        
        # Handle different input modes - with flexible data loading
        data_found = False
        if data is None:
            if uniprot_id:
                # Fetch UniProt sequence and find structure
                if verbose:
                    print(f"Mode: UniProt-based inpainting ({uniprot_id})")
                
                uniprot_info = fetch_uniprot_sequence(uniprot_id, verbose)
                full_sequence = uniprot_info['sequence']
                
                # Try to find matching structure using mapping first
                data = find_structure_by_uniprot(uniprot_id, sampling_coordinator.dataset, pdb_uniprot_mapping, verbose)
                
                if data is None and map_pkl_override:
                    # If not found in splits, try loading directly from map_pkl
                    if verbose:
                        print(f"  WARNING: Not found in dataset splits, trying direct map_pkl loading...")
                    data = load_protein_from_map_pkl(uniprot_id, map_pkl_override, sampling_coordinator.dataset_params, verbose, graph_builder=sampling_coordinator.dataset.graph_builder)
                
                if data is None:
                    raise ValueError(f"No structure found for UniProt {uniprot_id}")
                
                data_found = True
                
            elif pdb_id:
                # Find structure by PDB ID
                if verbose:
                    print(f"Mode: PDB-based inpainting ({pdb_id})")
                
                # Try to find in dataset splits first
                data = find_structure_by_pdb_id(pdb_id, sampling_coordinator.dataset, verbose)
                
                if data is None and map_pkl_override:
                    # If not found in splits, try loading directly from map_pkl
                    if verbose:
                        print(f"  WARNING: Not found in dataset splits, trying direct map_pkl loading...")
                    data = load_protein_from_map_pkl(pdb_id, map_pkl_override, sampling_coordinator.dataset_params, verbose, graph_builder=sampling_coordinator.dataset.graph_builder)
                
                if data is None:
                    raise ValueError(f"PDB {pdb_id} not found in dataset")
                
                data_found = True
                
            else:
                raise ValueError("Must provide either data, uniprot_id, or pdb_id")
        
        # Handle data format and extract sequences
        if data_found or data is not None:
            # Handle tuple format from dataset
            if isinstance(data, tuple) and len(data) > 0:
                graph_data = data[0]  # First element contains the graph data
                structure_sequence = graph_data.filtered_seq
                data = graph_data  # Use graph data for subsequent processing
            else:
                structure_sequence = data.filtered_seq
            
            # For PDB mode, use structure sequence as full sequence if not already set
            if pdb_id and not full_sequence:
                full_sequence = structure_sequence
        
        # Extract args from the loaded model for compatibility
        extracted_args = getattr(sampling_coordinator, 'model_args', None)
        
        # Override extracted args with command-line arguments when provided
        if args:
            # Create a new args object with overrides
            override_args = type('Args', (), {})()
            
            # Set defaults from checkpoint or fallback defaults
            checkpoint_params = getattr(sampling_coordinator, 'dataset_params', {})
            override_args.dirichlet_concentration = checkpoint_params.get('dirichlet_concentration', 20.0)
            override_args.use_c_factor = checkpoint_params.get('use_c_factor', False)
            override_args.flow_temp = checkpoint_params.get('flow_temp', 1.0)
            override_args.use_smoothed_targets = getattr(args, 'use_smoothed_targets', False)
            override_args.verbose = getattr(args, 'verbose', False)
            
            # Override with command-line arguments if provided
            if hasattr(args, 'flow_temp') and args.flow_temp is not None:
                override_args.flow_temp = args.flow_temp
                if verbose:
                    print(f"  Overriding flow_temp from command line: {args.flow_temp}")
                    
            if hasattr(args, 'use_c_factor') and args.use_c_factor is not None:
                override_args.use_c_factor = args.use_c_factor
                if verbose:
                    print(f"  Overriding use_c_factor from command line: {args.use_c_factor}")
                    
            if hasattr(args, 'dirichlet_concentration') and args.dirichlet_concentration is not None:
                override_args.dirichlet_concentration = args.dirichlet_concentration
                if verbose:
                    print(f"  Overriding dirichlet_concentration from command line: {args.dirichlet_concentration}")
            
            extracted_args = override_args
        
        # Get use_virtual_node setting
        use_virtual_node = getattr(sampling_coordinator, 'dataset_params', {}).get('use_virtual_node', False)
        
        # Extract and validate ground truth sequence for safety checking
        validated_mask_positions = None
        if mask_positions:
            try:
                # Extract ground truth sequence from data
                ground_truth_sequence = extract_sequence_from_data(data, use_virtual_node)
                
                if verbose:
                    print(f"Ground truth sequence length: {len(ground_truth_sequence)}")
                    print("Validating mask positions against ground truth...")
                
                # Parse and validate mask positions
                validated_mask_positions = parse_and_validate_mask_positions(mask_positions, ground_truth_sequence, verbose)
                
                if verbose:
                    print(f" Successfully validated {len(validated_mask_positions)} mask positions")
                    
            except ValueError as e:
                raise ValueError(f" Safety validation failed: {e}")
            except Exception as e:
                raise ValueError(f" Error during ground truth validation: {e}")
        
        # Get checkpoint time parameters for reporting
        checkpoint_t_max = getattr(sampling_coordinator, 'dataset_params', {}).get('t_max', 8.0)
        checkpoint_t_min = getattr(sampling_coordinator, 'dataset_params', {}).get('t_min', 0.0)
        
        if verbose:
            print(f"Running inpainting with:")
            print(f"  Full sequence length: {len(full_sequence) if full_sequence else 'N/A'}")
            print(f"  Structure sequence length: {len(structure_sequence) if structure_sequence else 'N/A'}")
            print(f"  Device: {device}")
            print(f"  Enable trajectory: {enable_trajectory}")
            
            # Time parameter reporting
            t_max_source = "command line" if T != checkpoint_t_max else "checkpoint" if checkpoint_t_max != 8.0 else "default"
            t_min_source = "command line" if t_min != checkpoint_t_min else "checkpoint" if checkpoint_t_min != 0.0 else "default"
            print(f"  Time parameters:")
            print(f"    t_min: {t_min} (source: {t_min_source})")
            print(f"    t_max: {T} (source: {t_max_source})")
            
            if split_json_override:
                print(f"  Split JSON override: {split_json_override}")
            if map_pkl_override:
                print(f"  Map PKL override: {map_pkl_override}")
        
        # Check if ensemble is requested
        ensemble_size = getattr(args, 'ensemble_size', 1) if args else 1
        use_ensemble = ensemble_size > 1
        
        if verbose and use_ensemble:
            print(f"  Ensemble parameters:")
            print(f"    ensemble_size: {getattr(args, 'ensemble_size', 1)}")
            print(f"    ensemble_method: {getattr(args, 'ensemble_method', 'arithmetic')}")
            print(f"    ensemble_consensus_strength: {getattr(args, 'ensemble_consensus_strength', 0.7)}")
            print(f"    structure_noise_mag_std: {getattr(args, 'structure_noise_mag_std', 0.0)}")
        
        # Run inpainting with or without trajectory tracking
        if enable_trajectory:
            results = sample_chain_inpainting_with_trajectory(
                sampling_coordinator.model, data, T=T, t_min=t_min, steps=steps, K=K,
                full_sequence=full_sequence, structure_sequence=structure_sequence,
                mask_positions=validated_mask_positions, known_sequence=known_sequence,
                mask_ratio=mask_ratio, verbose=verbose, args=extracted_args
            )
            final_probabilities, predicted_sequence, inpainting_mask, alignment_info, evaluation_metrics, trajectory_data = results
            
            return {
                'predicted_sequence': predicted_sequence,
                'final_probabilities': final_probabilities,
                'inpainting_mask': inpainting_mask,
                'alignment_info': alignment_info,
                'evaluation_metrics': evaluation_metrics,
                'trajectory_data': trajectory_data,
                'full_sequence': full_sequence,
                'structure_sequence': structure_sequence,
                'input_data': {
                    'uniprot_id': uniprot_id,
                    'pdb_id': pdb_id,
                    'mask_positions': mask_positions,  # Keep original string for display
                    'validated_positions': validated_mask_positions,  # Add validated positions
                    'known_sequence': known_sequence,
                    'steps': steps,
                    'temperature': T
                }
            }
        else:
            # Choose regular or ensemble inpainting based on parameters
            if use_ensemble:
                # Use ensemble inpainting
                results = sample_chain_inpainting_with_ensemble(
                    sampling_coordinator.model, data, T=T, t_min=t_min, steps=steps, K=K,
                    full_sequence=full_sequence, structure_sequence=structure_sequence,
                    mask_positions=validated_mask_positions, known_sequence=known_sequence,
                    mask_ratio=mask_ratio, verbose=verbose, args=extracted_args,
                    ensemble_size=getattr(args, 'ensemble_size', 1),
                    ensemble_consensus_strength=getattr(args, 'ensemble_consensus_strength', 0.7),
                    ensemble_method=getattr(args, 'ensemble_method', 'arithmetic'),
                    structure_noise_mag_std=getattr(args, 'structure_noise_mag_std', 0.0),
                    uncertainty_struct_noise_scaling=getattr(args, 'uncertainty_struct_noise_scaling', False)
                )
            else:
                # Use existing function without trajectory or ensemble
                results = sample_chain_inpainting(
                    sampling_coordinator.model, data, T=T, t_min=t_min, steps=steps, K=K,
                    full_sequence=full_sequence, structure_sequence=structure_sequence,
                    mask_positions=validated_mask_positions, known_sequence=known_sequence,
                    mask_ratio=mask_ratio, verbose=verbose, args=extracted_args
                )
            final_probabilities, predicted_sequence, inpainting_mask, alignment_info, evaluation_metrics = results
            
            return {
                'predicted_sequence': predicted_sequence,
                'final_probabilities': final_probabilities,
                'inpainting_mask': inpainting_mask,
                'alignment_info': alignment_info,
                'evaluation_metrics': evaluation_metrics,
                'full_sequence': full_sequence,
                'structure_sequence': structure_sequence,
                'input_data': {
                    'uniprot_id': uniprot_id,
                    'pdb_id': pdb_id,
                    'mask_positions': mask_positions,
                    'known_sequence': known_sequence,
                    'steps': steps,
                    'temperature': T
                }
            }
        
    except Exception as e:
        raise RuntimeError(f"Inpainting inference failed: {e}")


def process_csv_batch(csv_path: str, args, verbose: bool = False) -> List[Dict]:
    """
    Process a CSV file for batch inpainting.
    
    Required columns: mask-positions, protein
    
    Args:
        csv_path: Path to CSV file
        args: Command line arguments
        verbose: Print processing details
        
    Returns:
        List of dictionaries containing batch processing data
    """
    try:
        import pandas as pd
        
        if verbose:
            print(f"Loading CSV file: {csv_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['mask-positions', 'protein']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        if verbose:
            print(f"Found {len(df)} entries in CSV file")
        
        # Process each row
        batch_data = []
        for idx, row in df.iterrows():
            try:
                # Extract only essential data
                mask_positions_str = str(row['mask-positions']).strip()
                protein_id = str(row['protein']).strip()
                
                # Validate required fields
                if not mask_positions_str or mask_positions_str == 'nan':
                    if verbose:
                        print(f"Skipping row {idx}: Empty mask-positions")
                    continue
                
                if not protein_id or protein_id == 'nan':
                    if verbose:
                        print(f"Skipping row {idx}: Empty protein")
                    continue
                
                entry = {
                    'mask_positions': mask_positions_str,
                    'protein': protein_id,
                    'row_index': idx
                }
                
                batch_data.append(entry)
                
            except Exception as e:
                if verbose:
                    print(f"Error processing row {idx}: {e}")
                continue
        
        if verbose:
            print(f"Successfully processed {len(batch_data)} entries")
            
            # Show protein grouping info
            protein_groups = {}
            for entry in batch_data:
                protein = entry['protein']
                if protein not in protein_groups:
                    protein_groups[protein] = 0
                protein_groups[protein] += 1
            
            print(f"Entries grouped by protein:")
            for protein, count in protein_groups.items():
                print(f"  {protein}: {count} entries")
        
        return batch_data
        
    except Exception as e:
        raise RuntimeError(f"Failed to process CSV file {csv_path}: {e}")


def run_batch_inpainting(batch_data: List[Dict], args) -> List[Dict]:
    """
    Run inpainting on a batch of entries from CSV.
    
    Args:
        batch_data: List of entries from CSV processing
        args: Command line arguments
        
    Returns:
        List of results for each entry
    """
    results = []
    
    # Group entries by protein to enable batched processing
    protein_groups = {}
    for i, entry in enumerate(batch_data):
        protein_id = entry['protein']
        if protein_id not in protein_groups:
            protein_groups[protein_id] = []
        protein_groups[protein_id].append((i, entry))
    
    # Determine if trajectory tracking should be enabled
    # Only enable trajectory if explicitly requested via detailed_json
    # Don't auto-enable for small batches to allow testing batched processing
    enable_trajectory = args.detailed_json or False  # Simplified logic
    
    if enable_trajectory and len(batch_data) >= 4:
        print(f"Warning: Trajectory tracking disabled for batch size {len(batch_data)} (>=4) to save memory")
        enable_trajectory = False
    
    print(f"\nProcessing {len(batch_data)} entries from CSV...")
    print(f"Found {len(protein_groups)} unique proteins")
    
    # Check if batch_size is specified
    batch_size = getattr(args, 'batch_size', None)
    
    if batch_size is not None and batch_size > 1:
        # Process entries in batches of protein-mask combinations
        print(f"Using batch processing with batch size: {batch_size} protein-mask combinations")
        results = process_entries_in_batches(batch_data, protein_groups, args, enable_trajectory, batch_size)
    else:
        # Original per-protein processing
        print("Using per-protein sequential processing")
        results = process_entries_by_protein(protein_groups, args, enable_trajectory)
    
    # Sort results by original batch index to maintain order
    results.sort(key=lambda x: x.get('batch_index', 0))
    
    # Summary
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful
    
    print(f"\nBatch processing completed:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(results)}")
    
    return results


def process_entries_in_batches(batch_data: List[Dict], protein_groups: Dict, args, 
                              enable_trajectory: bool, batch_size: int) -> List[Dict]:
    """
    Process entries in batches of protein-mask combinations.
    
    Args:
        batch_data: Original list of entries
        protein_groups: Dictionary of protein_id -> list of entries  
        args: Command line arguments
        enable_trajectory: Whether to track trajectories
        batch_size: Number of protein-mask combinations per batch
        
    Returns:
        List of results
    """
    results = []
    
    # Flatten all entries into a single list with protein context
    all_entries = []
    for protein_id, protein_entries in protein_groups.items():
        for batch_index, entry in protein_entries:
            all_entries.append({
                'batch_index': batch_index,
                'entry': entry,
                'protein_id': protein_id
            })
    
    total_entries = len(all_entries)
    total_batches = (total_entries + batch_size - 1) // batch_size
    
    print(f"Processing {total_entries} protein-mask combinations in {total_batches} batches")
    
    # Load model and coordinator once (shared across all batches)
    print("Loading model and dataset coordinator (shared across all batches)...")
    
    # Load model once
    model, extracted_args = load_model_for_inpainting(
        args.model, device='auto', verbose=args.verbose
    )
    
    # Load dataset coordinator once
    coordinator = SamplingCoordinator(
        model_path=args.model,
        dataset_path="",  # Will be extracted from checkpoint
        split='test'
    )
    
    # Create args object for loading model and dataset
    class Args:
        def __init__(self):
            self.rbf_D_min = 0.0
            self.rbf_D_max = 20.0
            self.rbf_num_freqs = 16
            # Add missing attributes expected by SamplingCoordinator
            self.split_json = args.split_json
            self.map_pkl = args.map_pkl
            
    args_obj = Args()
    
    # Determine device
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and dataset properly once
    coordinator.load_model_and_dataset(device, args_obj)
    
    # Load PDB-UniProt mapping once
    pdb_uniprot_mapping = load_pdb_uniprot_mapping(
        args.pdb_uniprot_mapping, verbose=args.verbose
    )
    
    if args.verbose:
        print(f"Model and dataset loaded. Ready for batch processing.")
    
    # Process entries in batches with progress bar
    from tqdm import tqdm
    batch_progress = tqdm(range(total_batches), desc="Processing batches", unit="batch")
    
    for batch_num in batch_progress:
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_entries)
        
        current_batch = all_entries[start_idx:end_idx]
        
        # Show which proteins are in this batch
        batch_proteins = set(item['protein_id'] for item in current_batch)
        batch_progress.set_postfix({
            'entries': f'{start_idx + 1}-{end_idx}',
            'proteins': len(batch_proteins)
        })
        
        if args.verbose:
            print(f"\n=== Processing Batch {batch_num + 1}/{total_batches} ===")
            print(f"Entries {start_idx + 1}-{end_idx} ({len(current_batch)} combinations)")
            print(f"Proteins in this batch: {sorted(batch_proteins)}")
        
        # Group entries in current batch by protein for batched processing
        protein_groups = {}
        for item in current_batch:
            protein_id = item['protein_id']
            if protein_id not in protein_groups:
                protein_groups[protein_id] = []
            protein_groups[protein_id].append(item)
        
        # Process each entry in the current batch
        batch_results = []
        
        # Check if we can use batched processing
        if len(protein_groups) == 1 and not enable_trajectory:
            # All entries are for the same protein - use batched processing
            protein_id = list(protein_groups.keys())[0]
            protein_entries = protein_groups[protein_id]
            
            if args.verbose:
                print(f"Using BATCHED processing for {len(protein_entries)} {protein_id} entries")
            
            try:
                # Call our fixed batched processing function
                batched_results = run_batched_protein_inpainting_shared(
                    protein_id, protein_entries, args, enable_trajectory,
                    model, extracted_args, coordinator, pdb_uniprot_mapping
                )
                batch_results.extend(batched_results)
                
                if args.verbose:
                    print(f"✅ Batched processing completed for {len(protein_entries)} entries")
                    
            except Exception as e:
                if args.verbose:
                    print(f"❌ Batched processing failed for {protein_id}, falling back to sequential: {e}")
                
                # Fallback to sequential processing
                entry_progress = tqdm(current_batch, desc=f"Batch {batch_num + 1} (sequential fallback)", 
                                    leave=False, disable=not args.verbose)
                
                for item in entry_progress:
                    batch_index = item['batch_index']
                    entry = item['entry']
                    protein_id = item['protein_id']
                    
                    try:
                        result = run_single_inpainting_shared(
                            entry, batch_index, args, enable_trajectory, 
                            model, extracted_args, coordinator, pdb_uniprot_mapping
                        )
                        batch_results.append(result)
                        
                    except Exception as e:
                        error_result = {
                            'csv_data': entry,
                            'batch_index': batch_index,
                            'error': f"Processing failed: {e}",
                            'success': False
                        }
                        batch_results.append(error_result)
        else:
            # Multiple proteins or trajectory enabled - use sequential processing
            if args.verbose:
                if len(protein_groups) > 1:
                    print(f"Using sequential processing: {len(protein_groups)} different proteins in batch")
                if enable_trajectory:
                    print(f"Using sequential processing: trajectory tracking enabled")
            
            entry_progress = tqdm(current_batch, desc=f"Batch {batch_num + 1}", 
                                leave=False, disable=not args.verbose)
            
            for item in entry_progress:
                batch_index = item['batch_index']
                entry = item['entry']
                protein_id = item['protein_id']
                
                try:
                    # Use shared model and coordinator - no reloading!
                    result = run_single_inpainting_shared(
                        entry, batch_index, args, enable_trajectory, 
                        model, extracted_args, coordinator, pdb_uniprot_mapping
                    )
                    batch_results.append(result)
                    
                except Exception as e:
                    error_result = {
                        'csv_data': entry,
                        'batch_index': batch_index,
                        'error': f"Processing failed: {e}",
                        'success': False
                    }
                    batch_results.append(error_result)
        
        results.extend(batch_results)
        
        # Batch summary
        batch_successful = sum(1 for r in batch_results if 'error' not in r)
        batch_failed = len(batch_results) - batch_successful
        
        if args.verbose:
            print(f"Batch {batch_num + 1} completed: {batch_successful} successful, {batch_failed} failed")
        
        # Update main progress bar
        successful_so_far = sum(1 for r in results if 'error' not in r)
        batch_progress.set_description(
            f"Processing batches ({successful_so_far}/{len(results)} entries)"
        )
        
        # Optional: Clear GPU cache between batches to prevent memory accumulation
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    return results


def process_entries_by_protein(protein_groups: Dict, args, enable_trajectory: bool) -> List[Dict]:
    """
    Original per-protein processing logic.
    
    Args:
        protein_groups: Dictionary of protein_id -> list of entries
        args: Command line arguments
        enable_trajectory: Whether to track trajectories
        
    Returns:
        List of results
    """
    results = []
    
    # Process each protein group
    for protein_id, protein_entries in protein_groups.items():
        print(f"\nProcessing protein {protein_id} with {len(protein_entries)} variants...")
        
        if len(protein_entries) > 1 and not enable_trajectory:
            # Use batched processing for multiple entries of the same protein
            try:
                batched_results = run_batched_protein_inpainting(
                    protein_id, protein_entries, args, enable_trajectory
                )
                results.extend(batched_results)
            except Exception as e:
                print(f"Batched processing failed for {protein_id}, falling back to sequential: {e}")
                # Fallback to sequential processing
                for batch_index, entry in protein_entries:
                    result = run_single_inpainting(entry, batch_index, args, enable_trajectory)
                    results.append(result)
        else:
            # Sequential processing for single entries or when trajectory tracking is enabled
            for batch_index, entry in protein_entries:
                result = run_single_inpainting(entry, batch_index, args, enable_trajectory)
                results.append(result)
    
    return results


def run_batched_protein_inpainting(protein_id: str, protein_entries: List[Tuple[int, Dict]], 
                                  args, enable_trajectory: bool = False) -> List[Dict]:
    """
    Run batched inpainting for multiple mask patterns on the same protein.
    
    Args:
        protein_id: UniProt ID of the protein
        protein_entries: List of (batch_index, entry) tuples
        args: Command line arguments
        enable_trajectory: Whether to track trajectories
        
    Returns:
        List of results for each entry
    """
    try:
        # Load model and setup (same as single processing)
        model, extracted_args = load_model_for_inpainting(
            args.model, device='auto', verbose=args.verbose
        )
        
        # Load dataset and find protein structure - use the same pattern as the working version
        coordinator = SamplingCoordinator(
            model_path=args.model,
            dataset_path="",  # Will be extracted from checkpoint
            split='test'
        )
        
        # Create args object for loading model and dataset
        class Args:
            def __init__(self):
                self.rbf_3d_min = 2.0
                self.rbf_3d_max = 350.0
                self.rbf_3d_spacing = 'exponential'
                self.split_json = args.split_json
                self.map_pkl = args.map_pkl
        
        args_obj = Args()
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and dataset properly
        coordinator.load_model_and_dataset(device, args_obj)
        
        # Load PDB-UniProt mapping
        pdb_uniprot_mapping = load_pdb_uniprot_mapping(
            args.pdb_uniprot_mapping, verbose=args.verbose
        )
        
        # Find structure for this protein
        # For ProteinGym datasets, prioritize map_pkl since it contains the specific structures/domains used
        dataset = coordinator.dataset
        protein_data = None
        
        # Try map_pkl first (contains ProteinGym-specific structures)
        if args.map_pkl:
            if args.verbose:
                print(f"Trying to load {protein_id} from map_pkl first...")
            protein_data = load_protein_from_map_pkl(
                protein_id, args.map_pkl, getattr(coordinator, 'dataset_params', {}), 
                verbose=args.verbose, graph_builder=coordinator.dataset.graph_builder
            )
            if args.verbose and protein_data:
                print(f"✅ Found {protein_id} in map_pkl file")
            elif args.verbose:
                print(f"❌ Not found in map_pkl file")
        
        # Fallback to dataset search if not found in map_pkl
        if protein_data is None:
            if args.verbose:
                print(f"Fallback: searching dataset splits for {protein_id}...")
            protein_data = find_structure_by_uniprot(
                protein_id, dataset, pdb_uniprot_mapping, verbose=args.verbose
            )
        
        if protein_data is None:
            raise ValueError(f"Could not find protein structure for {protein_id}")
        
        # Extract structure sequence
        use_virtual_node = getattr(coordinator, 'dataset_params', {}).get('use_virtual_node', False)
        structure_sequence = extract_sequence_from_data(
            protein_data, use_virtual_node=use_virtual_node
        )
        
        if args.verbose:
            print(f"Found structure sequence for {protein_id}: {len(structure_sequence)} residues")
            # Debug: show residue at position 146 specifically
            if len(structure_sequence) >= 146:
                print(f"  Position 146 (0-indexed 145): {structure_sequence[145]}")
                print(f"  Sequence around position 146: {structure_sequence[140:151]}")
            else:
                print(f"  WARNING: Sequence too short for position 146 (length: {len(structure_sequence)})")
        
        # Prepare all mask patterns for batched processing
        batch_masks = []
        batch_entries = []
        
        for batch_index, entry in protein_entries:
            try:
                # Parse mask positions for this entry
                mask_positions_str = str(entry['mask_positions']).strip()
                validated_positions = parse_and_validate_mask_positions(
                    mask_positions_str, structure_sequence, verbose=False
                )
                
                batch_masks.append(validated_positions)
                batch_entries.append((batch_index, entry))
                
            except Exception as e:
                if args.verbose:
                    print(f"Skipping entry {batch_index} due to validation error: {e}")
                # Add error result for this entry
                error_result = {
                    'csv_data': entry,
                    'batch_index': batch_index,
                    'error': f"Mask validation failed: {e}",
                    'success': False
                }
                continue
        
        if not batch_masks:
            raise ValueError("No valid mask patterns found")
        
        # Create proper args object with extracted parameters from model
        batched_args = type('Args', (), {
            'dirichlet_concentration': extracted_args.dirichlet_concentration,
            'use_c_factor': extracted_args.use_c_factor,
            'flow_temp': extracted_args.flow_temp,
            'verbose': args.verbose
        })()
        
        # Run batched inpainting
        batched_results = sample_chain_inpainting_batched(
            model=model,
            data=protein_data,
            batch_mask_positions=batch_masks,
            structure_sequence=structure_sequence,
            T=args.t_max,
            t_min=args.t_min,
            steps=args.steps,
            verbose=args.verbose,
            args=batched_args,
            enable_trajectory=enable_trajectory
        )
        
        # Process results
        final_results = []
        for i, ((batch_index, entry), result_data) in enumerate(zip(batch_entries, batched_results)):
            if 'error' in result_data:
                final_result = {
                    'csv_data': entry,
                    'batch_index': batch_index,
                    'error': result_data['error'],
                    'success': False
                }
            else:
                # Add CSV metadata to successful results
                final_result = result_data.copy()
                final_result['csv_data'] = entry
                final_result['batch_index'] = batch_index
                final_result['success'] = True
                
                # Add input data for consistency
                final_result['input_data'] = {
                    'protein_id': protein_id,
                    'mask_positions_str': str(entry['mask_positions']),
                    'validated_positions': batch_masks[i],
                    'structure_sequence': structure_sequence
                }
            
            final_results.append(final_result)
        
        return final_results
        
    except Exception as e:
        # Return error results for all entries in this protein group
        error_results = []
        for batch_index, entry in protein_entries:
            error_results.append({
                'csv_data': entry,
                'batch_index': batch_index,
                'error': f"Batched processing failed for protein {protein_id}: {e}",
                'success': False
            })
        return error_results


def run_batched_protein_inpainting_shared(protein_id: str, protein_entries: List[Dict], 
                                        args, enable_trajectory: bool,
                                        model, extracted_args, coordinator, pdb_uniprot_mapping) -> List[Dict]:
    """
    Run batched inpainting for multiple mask patterns on the same protein using shared resources.
    
    Args:
        protein_id: UniProt ID of the protein
        protein_entries: List of entry dictionaries (not tuples)
        args: Command line arguments
        enable_trajectory: Whether to track trajectories
        model: Pre-loaded model
        extracted_args: Pre-extracted model arguments
        coordinator: Pre-loaded SamplingCoordinator
        pdb_uniprot_mapping: Pre-loaded PDB-UniProt mapping
        
    Returns:
        List of results for each entry
    """
    try:
        if args.verbose:
            print(f"Starting batched processing for {protein_id} with {len(protein_entries)} entries")
        
        # Find structure for this protein using shared coordinator
        # For ProteinGym datasets, prioritize map_pkl since it contains the specific structures/domains used
        dataset = coordinator.dataset
        protein_data = None
        
        # Try map_pkl first (contains ProteinGym-specific structures)
        if args.map_pkl:
            protein_data = load_protein_from_map_pkl(
                protein_id, args.map_pkl, getattr(coordinator, 'dataset_params', {}), 
                verbose=False, graph_builder=coordinator.dataset.graph_builder
            )
        
        # Fallback to dataset search if not found in map_pkl
        if protein_data is None:
            protein_data = find_structure_by_uniprot(
                protein_id, dataset, pdb_uniprot_mapping, verbose=False
            )
        
        if protein_data is None:
            raise ValueError(f"Could not find protein structure for {protein_id}")
        
        # Extract structure sequence
        use_virtual_node = getattr(coordinator, 'dataset_params', {}).get('use_virtual_node', False)
        structure_sequence = extract_sequence_from_data(
            protein_data, use_virtual_node=use_virtual_node
        )
        
        # For ProteinGym datasets, get original sequence for validation
        original_sequence = structure_sequence  # Default fallback
        if args.map_pkl:
            try:
                import pickle
                with open(args.map_pkl, 'rb') as f:
                    map_data = pickle.load(f)
                
                if protein_id in map_data and 'seq' in map_data[protein_id]:
                    original_sequence = map_data[protein_id]['seq']
            except Exception as e:
                if args.verbose:
                    print(f"Failed to load original sequence from map_pkl: {e}")
        
        # Prepare all mask patterns for batched processing
        batch_masks = []
        valid_entries = []
        
        for item in protein_entries:
            entry = item['entry']
            batch_index = item['batch_index']
            
            try:
                # Parse mask positions for this entry
                mask_positions_str = str(entry['mask_positions']).strip()
                
                # Validate against original sequence when available
                validated_positions = parse_and_validate_mask_positions(
                    mask_positions_str, original_sequence, verbose=False
                )
                
                batch_masks.append(validated_positions)
                valid_entries.append(item)
                
            except Exception as e:
                if args.verbose:
                    print(f"Skipping entry {batch_index} due to validation error: {e}")
        
        if not batch_masks:
            raise ValueError("No valid mask patterns found")
        
        if args.verbose:
            print(f"Processing {len(batch_masks)} valid mask patterns in parallel")
        
        # Create proper args object with extracted parameters from model
        batched_args = type('Args', (), {
            'dirichlet_concentration': extracted_args.dirichlet_concentration,
            'use_c_factor': extracted_args.use_c_factor,
            'flow_temp': extracted_args.flow_temp,
            'time_as_temperature': getattr(extracted_args, 'time_as_temperature', False),
            'use_smoothed_targets': getattr(extracted_args, 'use_smoothed_targets', False),
            'verbose': False  # Reduce verbosity for batched processing
        })()
        
        # Run batched inpainting - this calls our fixed parallel processing function!
        batched_results = sample_chain_inpainting_batched(
            model=model,
            data=protein_data,
            batch_mask_positions=batch_masks,
            structure_sequence=structure_sequence,
            T=args.t_max,
            t_min=args.t_min,
            steps=args.steps,
            verbose=args.verbose,
            args=batched_args,
            enable_trajectory=enable_trajectory
        )
        
        # Process results
        final_results = []
        for i, (item, result_data) in enumerate(zip(valid_entries, batched_results)):
            entry = item['entry']
            batch_index = item['batch_index']
            
            if 'error' in result_data:
                final_result = {
                    'csv_data': entry,
                    'batch_index': batch_index,
                    'error': result_data['error'],
                    'success': False
                }
            else:
                # Add CSV metadata to successful results
                final_result = result_data.copy()
                final_result['csv_data'] = entry
                final_result['batch_index'] = batch_index
                final_result['success'] = True
                
                # Add input data for consistency
                final_result['input_data'] = {
                    'uniprot_id': protein_id,
                    'pdb_id': None,
                    'mask_positions': str(entry['mask_positions']),
                    'known_sequence': None,
                    'steps': args.steps,
                    'temperature': args.t_max,
                    'validated_positions': batch_masks[i]
                }
            
            final_results.append(final_result)
        
        if args.verbose:
            successful = sum(1 for r in final_results if 'error' not in r)
            print(f"✅ Batched processing completed: {successful}/{len(final_results)} successful")
        
        return final_results
        
    except Exception as e:
        if args.verbose:
            print(f"❌ Batched processing error: {e}")
        
        # Return error results for all entries in this protein group
        error_results = []
        for item in protein_entries:
            entry = item['entry']
            batch_index = item['batch_index']
            error_results.append({
                'csv_data': entry,
                'batch_index': batch_index,
                'error': f"Batched processing failed for protein {protein_id}: {e}",
                'success': False
            })
        return error_results


def sample_chain_inpainting_batched(model, data, batch_mask_positions, structure_sequence=None,
                                   T=8.0, t_min=0.0, steps=20, K=21, verbose=False, 
                                   args=None, enable_trajectory=False):
    """
    Batched inpainting for multiple mask patterns on the same protein structure.
    
    Args:
        model: Trained DFM model
        data: Protein structure graph (single protein)
        batch_mask_positions: List of mask position lists (each is list of 0-indexed positions)
        structure_sequence: Structure sequence string
        T: Starting time (max noise)
        t_min: Minimum time
        steps: Number of denoising steps
        K: Number of amino acid classes
        verbose: Print detailed information
        args: Arguments object
        enable_trajectory: Whether to track trajectories
        
    Returns:
        List of result dictionaries, one per mask pattern
    """
    from training.collate import collate_fn
    
    device = next(model.parameters()).device
    model.eval()
    
    batch_size = len(batch_mask_positions)
    
    if verbose:
        print(f"Running batched inpainting for {batch_size} mask patterns on same protein")
    
    # Setup batched data (same structure replicated)
    dummy_y = torch.zeros(1, K)
    dummy_mask = torch.ones(1, dtype=torch.bool)
    dummy_time = torch.tensor(0.0)
    
    try:
        batched_data, y_pad, mask_pad, time_batch = collate_fn([(data, dummy_y, dummy_mask, dummy_time)])
        batched_data = batched_data.to(device)
    except Exception as e:
        return [{'error': f"Failed to prepare batched data: {e}"} for _ in range(batch_size)]
    
    # Handle virtual nodes
    total_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x_s.size(0)
    use_virtual_node = getattr(data, 'use_virtual_node', False)
    
    if use_virtual_node:
        N = total_nodes - 1
    else:
        N = total_nodes
    
    if N <= 0:
        error_msg = f"No real nodes found. Total: {total_nodes}, virtual: {use_virtual_node}"
        return [{'error': error_msg} for _ in range(batch_size)]
    
    # Get ground truth sequence
    try:
        ground_truth_onehot = None
        if hasattr(data, 'filtered_seq') and data.filtered_seq is not None:
            filtered_seq = data.filtered_seq[:N]
            ground_truth_onehot = torch.zeros(N, K, device=device)
            for i, aa in enumerate(filtered_seq):
                if aa in SINGLE_TO_TRIPLE:
                    aa3 = SINGLE_TO_TRIPLE[aa]
                    if aa3 in AA_TO_IDX:
                        ground_truth_onehot[i, AA_TO_IDX[aa3]] = 1.0
                    else:
                        ground_truth_onehot[i, 20] = 1.0
                else:
                    ground_truth_onehot[i, 20] = 1.0
        elif hasattr(data, 'y') and data.y is not None:
            ground_truth_onehot = data.y[:N].to(device)
        else:
            raise ValueError("No ground truth sequence found")
    except Exception as e:
        error_msg = f"Failed to extract ground truth sequence: {e}"
        return [{'error': error_msg} for _ in range(batch_size)]
    
    # Create masks for each pattern
    batch_masks = []
    batch_valid = []
    
    for mask_positions in batch_mask_positions:
        try:
            # Create mask for this pattern
            mask_tensor = torch.zeros(N, dtype=torch.bool, device=device)
            for pos in mask_positions:
                if 0 <= pos < N:
                    mask_tensor[pos] = True
            
            if mask_tensor.sum().item() == 0:
                batch_valid.append(False)
                batch_masks.append(None)
            else:
                batch_valid.append(True)
                batch_masks.append(mask_tensor)
                
        except Exception as e:
            batch_valid.append(False)
            batch_masks.append(None)
    
    # Check if any masks are valid
    if not any(batch_valid):
        return [{'error': "No valid masks created"} for _ in range(batch_size)]
    
    # Initialize batch of sequences [batch_size, N, K]
    dirichlet_concentration = getattr(args, 'dirichlet_concentration', 20.0) if args else 20.0
    use_c_factor = getattr(args, 'use_c_factor', False) if args else False
    
    try:
        dirichlet_dist = Dirichlet(dirichlet_concentration * torch.ones(K, device=device))
        x_batch = dirichlet_dist.sample((batch_size, N))  # [batch_size, N, K]
    except Exception as e:
        error_msg = f"Failed to initialize sequences: {e}"
        return [{'error': error_msg} for _ in range(batch_size)]
    
    # Apply ground truth to known positions for each mask pattern
    for i, (mask_tensor, is_valid) in enumerate(zip(batch_masks, batch_valid)):
        if is_valid and ground_truth_onehot is not None:
            # Set known positions (not masked) to ground truth
            known_positions = ~mask_tensor
            x_batch[i, known_positions] = ground_truth_onehot[known_positions]
    
    # Time steps
    times = torch.linspace(t_min, T, steps, device=device)
    dt = (T - t_min) / (steps - 1)
    
    if verbose:
        print(f"Processing {sum(batch_valid)}/{batch_size} valid mask patterns")
        print(f"Time integration: t={t_min:.1f} -> {T:.1f}, steps={steps}")
    
    # Initialize velocity storage for batch processing
    v_batch = torch.zeros_like(x_batch)  # [batch_size, N, K]
    
    # Batched denoising loop
    with torch.no_grad():
        time_steps = tqdm(enumerate(times), total=len(times), desc="Batched inpainting", disable=not verbose)
        for i, t_val in time_steps:
            if i == len(times) - 1:  # Skip velocity on last step
                break
            
            t = torch.full((1,), t_val, device=device)
            
            # Update progress bar with batch info
            valid_count = sum(batch_valid)
            time_steps.set_postfix({
                't': f'{t_val:.3f}',
                'valid_batches': f'{valid_count}/{batch_size}'
            })
            
            # TRUE PARALLEL PROCESSING: Process each pattern individually but batch operations where possible  
            # The model expects one sequence per graph, so we process patterns individually
            # but accumulate results for parallel velocity computation
            try:
                    # Filter out invalid patterns
                    valid_indices = [i for i, valid in enumerate(batch_valid) if valid]
                    if not valid_indices:
                        continue
                        
                    valid_batch_size = len(valid_indices)
                    
                    if verbose:
                        print(f"PARALLEL: Processing {valid_batch_size} mask patterns individually then batching velocity")
                    
                    # Process each mask pattern individually (like sequential)
                    # but collect results for batched velocity computation
                    model_outputs = []
                    predicted_targets = []
                    
                    for batch_idx in valid_indices:
                        x_single = x_batch[batch_idx:batch_idx+1]  # [1, N, K]
                        
                        # Get model prediction for this pattern
                        model_output = model(batched_data, t, x_single)
                        
                        # Handle DSSP output
                        if isinstance(model_output, tuple):
                            position_logits = model_output[0]
                        else:
                            position_logits = model_output
                        
                        # Apply temperature
                        if args and getattr(args, 'time_as_temperature', False):
                            flow_temp = T - t_val + 0.1
                        else:
                            flow_temp = args.flow_temp if args else 1.0
                        
                        predicted_target = torch.softmax(position_logits / flow_temp, dim=-1)
                        
                        # Handle virtual nodes
                        if use_virtual_node:
                            predicted_target_real = predicted_target[:N, :].unsqueeze(0)  # [N, K] -> [1, N, K]
                        else:
                            predicted_target_real = predicted_target.unsqueeze(0)  # [N, K] -> [1, N, K]
                        
                        model_outputs.append(position_logits)
                        predicted_targets.append(predicted_target_real)
                    
                    # Now batch the velocity computation
                    x_batch_valid = torch.stack([x_batch[i] for i in valid_indices])  # [valid_batch_size, N, K]
                    predicted_batch = torch.stack(predicted_targets).squeeze(1)  # [valid_batch_size, N, K]
                    t_batch_expanded = t.expand(valid_batch_size)  # [valid_batch_size]
                    
                    # Compute velocity for all patterns in parallel  
                    cond_flow = model.cond_flow
                    v_parallel = cond_flow.velocity(
                        x_batch_valid,  # [valid_batch_size, N, K]
                        predicted_batch,  # [valid_batch_size, N, K] 
                        t_batch_expanded,  # [valid_batch_size]
                        use_virtual_node=use_virtual_node,
                        use_smoothed_targets=getattr(args, 'use_smoothed_targets', False),
                        use_c_factor=use_c_factor
                    )
                    
                    # Apply masks and update velocities for valid patterns
                    for i, batch_idx in enumerate(valid_indices):
                        mask_tensor = batch_masks[batch_idx]  # [N]
                        # Zero velocity for non-masked positions
                        v_parallel[i, ~mask_tensor] = 0
                        # Store back to main batch
                        v_batch[batch_idx] = v_parallel[i]
                    
                    if verbose:
                        print(f"✅ PARALLEL: Successfully processed {valid_batch_size} patterns")
                        
            except Exception as e:
                print(f"❌ PARALLEL FAILED: {e}, falling back to sequential")
                if verbose:
                    import traceback
                    print("Full error traceback:")
                    traceback.print_exc()
                
                # Fallback to sequential processing if parallel fails
                for batch_idx in range(batch_size):
                    if not batch_valid[batch_idx]:
                        continue
                        
                    try:
                        x_single = x_batch[batch_idx:batch_idx+1]  # [1, N, K]
                        
                        # Get model prediction
                        model_output = model(batched_data, t, x_single)
                        
                        # Handle DSSP output
                        if isinstance(model_output, tuple):
                            position_logits = model_output[0]
                        else:
                            position_logits = model_output
                        
                        # Apply temperature
                        if args and getattr(args, 'time_as_temperature', False):
                            flow_temp = T - t_val + 0.1
                        else:
                            flow_temp = args.flow_temp if args else 1.0
                        
                        predicted_target = torch.softmax(position_logits / flow_temp, dim=-1)
                        
                        # Handle virtual nodes
                        if use_virtual_node:
                            predicted_target_real = predicted_target[:N, :].unsqueeze(0)
                        else:
                            predicted_target_real = predicted_target.unsqueeze(0)
                        
                        # Compute velocity
                        cond_flow = model.cond_flow
                        v_analytical = cond_flow.velocity(
                            x_single,
                            predicted_target_real,
                            t,
                            use_virtual_node=use_virtual_node,
                            use_smoothed_targets=getattr(args, 'use_smoothed_targets', False),
                            use_c_factor=use_c_factor
                        )
                        
                        # Apply velocity only to masked positions
                        mask_tensor = batch_masks[batch_idx]  # [N]
                        v_analytical[:, ~mask_tensor] = 0  # Zero velocity for non-masked positions
                        
                        v_batch[batch_idx] = v_analytical[0]
                        
                    except Exception as e:
                        print(f"Error processing batch item {batch_idx}: {e}")
                        batch_valid[batch_idx] = False
        
        # Update x_batch using computed velocities (Euler integration)
        for batch_idx in range(batch_size):
            if batch_valid[batch_idx]:
                try:
                    # Euler step: x_new = x_old + dt * v
                    x_old = x_batch[batch_idx:batch_idx+1]  # [1, N, K]
                    v = v_batch[batch_idx:batch_idx+1]      # [1, N, K] 
                    x_new = x_old + dt * v
                    
                    # Project to simplex
                    x_new = simplex_proj(x_new)
                    
                    # Restore known positions (non-masked)
                    mask_tensor = batch_masks[batch_idx]
                    if ground_truth_onehot is not None:
                        known_positions = ~mask_tensor
                        x_new[0, known_positions] = ground_truth_onehot[known_positions]
                    
                    x_batch[batch_idx] = x_new[0]
                    
                except Exception as e:
                    if verbose:
                        print(f"Error updating batch {batch_idx} at step {i+1}: {e}")
                    batch_valid[batch_idx] = False
            
            if verbose and (i + 1) % 10 == 0:
                valid_count = sum(batch_valid)
                print(f"  Step {i+1}/{steps}: t={t_val:.3f}, processing {valid_count} patterns")
    
    # Prepare results
    results = []
    for batch_idx in range(batch_size):
        if not batch_valid[batch_idx]:
            results.append({'error': f"Processing failed for mask pattern {batch_idx}"})
            continue
        
        try:
            final_probabilities = x_batch[batch_idx]  # [N, K]
            predicted_sequence = final_probabilities.argmax(-1).tolist()
            mask_tensor = batch_masks[batch_idx]
            
            # Compute evaluation metrics
            evaluation_metrics = {}
            if ground_truth_onehot is not None:
                try:
                    evaluation_metrics = compute_sampling_metrics(
                        final_probabilities, ground_truth_onehot, data, model, args, device, use_virtual_node, K
                    )
                    
                    # Add masked-specific metrics
                    true_classes = ground_truth_onehot.argmax(-1).tolist()
                    true_sequence_single = ''.join([THREE_TO_ONE.get(IDX_TO_AA[idx], 'X') if 0 <= idx < len(IDX_TO_AA) else 'X' for idx in true_classes])
                    evaluation_metrics['true_sequence'] = true_sequence_single
                    
                    # Masked accuracy
                    masked_positions_list = torch.where(mask_tensor)[0].tolist()
                    if masked_positions_list:
                        masked_predictions = [predicted_sequence[i] for i in masked_positions_list]
                        masked_ground_truth = ground_truth_onehot[mask_tensor].argmax(-1).tolist()
                        
                        masked_correct = sum(p == t for p, t in zip(masked_predictions, masked_ground_truth))
                        masked_accuracy = masked_correct / len(masked_predictions) * 100
                        
                        evaluation_metrics['masked_accuracy'] = masked_accuracy
                        evaluation_metrics['masked_correct'] = masked_correct
                        evaluation_metrics['masked_total'] = len(masked_predictions)
                    
                except Exception as e:
                    evaluation_metrics = {'accuracy': 0.0, 'masked_accuracy': 0.0}
            
            result = {
                'predicted_sequence': predicted_sequence,
                'final_probabilities': final_probabilities,
                'inpainting_mask': mask_tensor.cpu(),
                'evaluation_metrics': evaluation_metrics,
                'alignment_info': {'score': 1.0}  # Perfect alignment since same protein
            }
            
            results.append(result)
            
        except Exception as e:
            results.append({'error': f"Failed to prepare result for batch {batch_idx}: {e}"})
    
    return results


def run_single_inpainting_shared(entry: Dict, batch_index: int, args, enable_trajectory: bool,
                               model, extracted_args, coordinator, pdb_uniprot_mapping) -> Dict:
    """
    Run single inpainting using shared model and coordinator (no reloading).
    
    Args:
        entry: CSV entry data
        batch_index: Index in batch
        args: Command line arguments  
        enable_trajectory: Whether to track trajectories
        model: Pre-loaded model
        extracted_args: Pre-extracted model arguments
        coordinator: Pre-loaded SamplingCoordinator
        pdb_uniprot_mapping: Pre-loaded PDB-UniProt mapping
        
    Returns:
        Result dictionary
    """
    try:
        if args.verbose:
            mask_positions = str(entry.get('mask_positions', ''))
            protein = entry.get('protein', 'unknown')
            print(f"Processing entry {batch_index + 1}: {protein} with mask {mask_positions}")
        
        # Use the shared coordinator and model
        protein_id = entry['protein']
        
        # Find structure for this protein using shared coordinator
        # For ProteinGym datasets, prioritize map_pkl since it contains the specific structures/domains used
        dataset = coordinator.dataset
        protein_data = None
        
        # Try map_pkl first (contains ProteinGym-specific structures)
        if args.map_pkl:
            protein_data = load_protein_from_map_pkl(
                protein_id, args.map_pkl, getattr(coordinator, 'dataset_params', {}), 
                verbose=False, graph_builder=coordinator.dataset.graph_builder
            )
        
        # Fallback to dataset search if not found in map_pkl
        if protein_data is None:
            protein_data = find_structure_by_uniprot(
                protein_id, dataset, pdb_uniprot_mapping, verbose=False
            )
        
        if protein_data is None:
            raise ValueError(f"Could not find protein structure for {protein_id}")
        
        # Extract structure sequence
        use_virtual_node = getattr(coordinator, 'dataset_params', {}).get('use_virtual_node', False)
        structure_sequence = extract_sequence_from_data(
            protein_data, use_virtual_node=use_virtual_node
        )
        
        # Parse and validate mask positions
        mask_positions_str = str(entry['mask_positions']).strip()
        
        # For ProteinGym datasets, we need to validate against the original sequence
        # from map_pkl, not the filtered sequence from GraphBuilder
        if args.map_pkl:
            try:
                import pickle
                with open(args.map_pkl, 'rb') as f:
                    map_data = pickle.load(f)
                
                if protein_id in map_data and 'seq' in map_data[protein_id]:
                    original_sequence = map_data[protein_id]['seq']
                    if args.verbose:
                        print(f"Using original sequence from map_pkl for validation (length: {len(original_sequence)})")
                    validated_mask_positions = parse_and_validate_mask_positions(
                        mask_positions_str, original_sequence, verbose=False
                    )
                else:
                    # Fallback to structure sequence
                    validated_mask_positions = parse_and_validate_mask_positions(
                        mask_positions_str, structure_sequence, verbose=False
                    )
            except Exception as e:
                if args.verbose:
                    print(f"Failed to load original sequence from map_pkl: {e}")
                # Fallback to structure sequence
                validated_mask_positions = parse_and_validate_mask_positions(
                    mask_positions_str, structure_sequence, verbose=False
                )
        else:
            validated_mask_positions = parse_and_validate_mask_positions(
                mask_positions_str, structure_sequence, verbose=False
            )
        
        # Get full sequence for alignment (use structure sequence as fallback)
        full_sequence = structure_sequence  # For CSV mode, structure sequence serves as full sequence
        
        # Create proper args object with extracted parameters from model
        inpainting_args = type('Args', (), {
            'dirichlet_concentration': extracted_args.dirichlet_concentration,
            'use_c_factor': extracted_args.use_c_factor,
            'flow_temp': extracted_args.flow_temp,
            'verbose': False  # Reduce verbosity for individual entries
        })()
        
        # Run inpainting with or without trajectory tracking
        if enable_trajectory:
            predicted_sequence, final_probabilities, inpainting_mask, alignment_info, evaluation_metrics, trajectory_data = sample_chain_inpainting_with_trajectory(
                model=model,
                data=protein_data,
                full_sequence=full_sequence,
                structure_sequence=structure_sequence,
                mask_positions=validated_mask_positions,
                T=args.t_max,
                t_min=args.t_min,
                steps=args.steps,
                verbose=False,
                args=inpainting_args
            )
            
            result = {
                'predicted_sequence': predicted_sequence,
                'final_probabilities': final_probabilities,
                'inpainting_mask': inpainting_mask,
                'alignment_info': alignment_info,
                'evaluation_metrics': evaluation_metrics,
                'full_sequence': None,  # Not used in CSV mode
                'structure_sequence': structure_sequence,
                'trajectory_data': trajectory_data,
                'input_data': {
                    'uniprot_id': protein_id,
                    'pdb_id': None,
                    'mask_positions': mask_positions_str,
                    'known_sequence': None,
                    'steps': args.steps,
                    'temperature': args.t_max,
                    'validated_positions': validated_mask_positions
                }
            }
        else:
            predicted_sequence, final_probabilities, inpainting_mask, alignment_info, evaluation_metrics = sample_chain_inpainting(
                model=model,
                data=protein_data,
                full_sequence=full_sequence,
                structure_sequence=structure_sequence,
                mask_positions=validated_mask_positions,
                T=args.t_max,
                t_min=args.t_min,
                steps=args.steps,
                verbose=False,
                args=inpainting_args
            )
            
            result = {
                'predicted_sequence': predicted_sequence,
                'final_probabilities': final_probabilities,
                'inpainting_mask': inpainting_mask,
                'alignment_info': alignment_info,
                'evaluation_metrics': evaluation_metrics,
                'full_sequence': None,  # Not used in CSV mode
                'structure_sequence': structure_sequence,
                'input_data': {
                    'uniprot_id': protein_id,
                    'pdb_id': None,
                    'mask_positions': mask_positions_str,
                    'known_sequence': None,
                    'steps': args.steps,
                    'temperature': args.t_max,
                    'validated_positions': validated_mask_positions
                }
            }
        
        # Add metadata
        result['csv_data'] = entry
        result['batch_index'] = batch_index
        result['success'] = True
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        if args.verbose:
            print(f"  Error processing entry {batch_index + 1}: {error_msg}")
        return {
            'csv_data': entry,
            'batch_index': batch_index,
            'error': error_msg,
            'success': False
        }


def run_single_inpainting(entry: Dict, batch_index: int, args, enable_trajectory: bool) -> Dict:
    """
    Run single inpainting (fallback for sequential processing).
    
    Args:
        entry: CSV entry data
        batch_index: Index in batch
        args: Command line arguments  
        enable_trajectory: Whether to track trajectories
        
    Returns:
        Result dictionary
    """
    try:
        if args.verbose:
            mask_positions = str(entry.get('mask_positions', ''))
            protein = entry.get('protein', 'unknown')
            print(f"Processing entry {batch_index + 1}: {protein} with mask {mask_positions}")
        
        # Run inpainting for this entry (existing logic)
        result = run_inpainting_inference(
            model_path=args.model,
            uniprot_id=entry['protein'],
            pdb_id=None,
            mask_positions=str(entry['mask_positions']),
            known_sequence=None,
            steps=args.steps,
            T=args.t_max,
            t_min=args.t_min,
            verbose=args.verbose,
            pdb_uniprot_mapping_path=args.pdb_uniprot_mapping,
            args=args,
            enable_trajectory=enable_trajectory,
            split_json_override=args.split_json,
            map_pkl_override=args.map_pkl
        )
        
        # Add metadata
        result['csv_data'] = entry
        result['batch_index'] = batch_index
        result['success'] = True
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        if args.verbose:
            print(f"  Error processing entry {batch_index + 1}: {error_msg}")
        return {
            'csv_data': entry,
            'batch_index': batch_index,
            'error': error_msg,
            'success': False
        }


def main():
    """Simple command-line interface for inpainting."""
    parser = argparse.ArgumentParser(
        description="Protein sequence inpainting using existing infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Position-only inpainting (positions 45, 67, 89)
  python inpainting.py --uniprot P69905 --mask-positions 45,67,89 --model model.pt
  
  # Variant effect prediction with ground truth validation (safety check)
  python inpainting.py --uniprot P69905 --mask-positions D45,Y67,K89 --model model.pt
  
  # PDB-based inpainting with validation
  python inpainting.py --pdb-id 1abc --mask-positions A10,V20,L30 --model model.pt
  
  # Template-based inpainting (existing method)
  python inpainting.py --uniprot P69905 --template "ACDEFGHXKLMNPQRX" --model model.pt
  
  # Batch processing from CSV file
  python inpainting.py --list_csv mutations.csv --model model.pt --output-dir batch_results
  
  # Batch processing with custom batch size (process 10 protein-mask combinations at a time)
  python inpainting.py --list_csv mutations.csv --model model.pt --batch_size 10
  
  # Override dataset paths from command line
  python inpainting.py --pdb-id 1abc --mask-positions D28,G32 --model model.pt \\
    --split_json ../datasets/cath-4.2/chain_set_splits.json \\
    --map_pkl ../datasets/cath-4.2/chain_set_map_with_b_factors_dssp.pkl
    
Mask Position Formats:
  - Position only: "45,67,89" - mask these positions
  - With validation: "D45,Y67,K89" - mask positions 45,67,89 but first verify
    that position 45 has amino acid D, position 67 has Y, position 89 has K.
    If validation fails, the program will error out as a safety mechanism.
    Perfect for variant effect prediction studies.

CSV File Format (for --list_csv):
  Required columns: mutant, mutated_sequence, DMS_score, DMS_score_bin, mask-positions, protein
  - mutant: identifier for the mutation
  - mutated_sequence: the sequence with mutations (optional for validation)
  - DMS_score: Deep Mutational Scanning score
  - DMS_score_bin: binned score category
  - mask-positions: positions to mask (same format as --mask-positions)
  - protein: UniProt ID (corresponds to --uniprot argument)
  
  Example CSV content:
    mutant,mutated_sequence,DMS_score,DMS_score_bin,mask-positions,protein
    A1V,MVQPQVQHPIQ...,-2.1,low,1,PIN1_HUMAN
    L2P,MPQPQVQHPIQ...,0.5,medium,2,PIN1_HUMAN
    G3A,MVQAQVQHPIQ...,1.2,high,3,PIN1_HUMAN
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--uniprot', type=str, help='UniProt accession ID')
    input_group.add_argument('--pdb-id', type=str, help='PDB ID from dataset')
    input_group.add_argument('--list_csv', type=str, help='CSV file with columns: mutant, mutated_sequence, DMS_score, DMS_score_bin, mask-positions, protein')
    
    # Masking options
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument('--mask-positions', type=str, 
                           help='Positions to mask. Formats: "45,67,89" (positions only) or "D45,Y67,K89" (with ground truth validation)')
    mask_group.add_argument('--template', type=str, help='Template sequence with X for positions to predict')
    mask_group.add_argument('--mask-ratio', type=float, default=0.0, help='Random mask ratio (default: 0.0)')

    # Required arguments
    parser.add_argument('--model', type=str, default = "../ckpts/model_316.pt", help='Path to trained model checkpoint')
    
    # Ensemble arguments
    parser.add_argument('--ensemble_size', type=int, default=1,
                       help='Number of ensemble members to use (default: 1, no ensemble)')
    parser.add_argument('--ensemble_consensus_strength', type=float, default=0.7,
                       help='Consensus strength parameter for ensemble averaging (default: 0.7)')
    parser.add_argument('--ensemble_method', type=str, choices=['arithmetic', 'geometric'], default='arithmetic',
                       help='Ensemble consensus method: arithmetic or geometric averaging (default: arithmetic)')
    parser.add_argument('--structure_noise_mag_std', type=float, default=0.0,
                       help='Standard deviation for structural noise injection (default: 0.0, no noise)')
    parser.add_argument('--uncertainty_struct_noise_scaling', action='store_true', default=False,
                       help='Enable uncertainty-based structural noise scaling')
    
    # Optional arguments
    parser.add_argument('--pdb-uniprot-mapping', type=str, help='Path to PDB-UniProt mapping JSON file')
    parser.add_argument('--output-dir', type=str, default='../output/inpainting_results', help='Output directory')
    parser.add_argument('--steps', type=int, default=20, help='Number of diffusion steps')
    parser.add_argument('--t_max', type=float, default=8.0, help='Maximum time for diffusion process')
    parser.add_argument('--t_min', type=float, default=0.0, help='Minimum time for diffusion process')
    parser.add_argument('--flow_temp', type=float, default=None, help='Flow temperature for sampling (default: use checkpoint value or 1.0)')
    parser.add_argument('--time_as_temperature', action='store_true',
                       help="Use time-dependent temperature: flow_temp = t_max - current_time + 0.1 (starts high, cools down)")
    parser.add_argument('--use_c_factor', action='store_true', default=None, help='Use C factor in velocity computation (default: use checkpoint value)')
    parser.add_argument('--dirichlet_concentration', type=float, default=None, help='Dirichlet concentration for initialization (default: use checkpoint value or 20.0)')
    parser.add_argument('--use_smoothed_targets', action='store_true', default=False,
                       help="Use smoothed targets in velocity computation (default: False). Set to True if --use_smoothed_labels is present.")
    parser.add_argument('--use_smoothed_labels', action='store_true', default=False,
                       help="Enable smoothed labels (sets use_smoothed_targets=True automatically)")
    parser.add_argument('--detailed_json', action='store_true', help='Generate detailed trajectory JSON for masked positions')
    
    # Batch processing options
    parser.add_argument('--batch_size', type=int, default=None, 
                       help='Number of protein-mask combinations to process in each batch (default: process all combinations sequentially)')
    
    # Dataset override arguments
    parser.add_argument('--split_json', type=str, help='Override split JSON path from model checkpoint')
    parser.add_argument('--map_pkl', type=str, help='Override map PKL path from model checkpoint')
    
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    
    args = parser.parse_args()
    
    # Process arguments to handle interdependent logic
    from training.sample_utils import process_sampling_args
    args = process_sampling_args(args)
    
    try:
        # Check if we're doing batch processing from CSV
        if args.list_csv:
            # CSV batch processing mode
            print(f"Running batch inpainting from CSV: {args.list_csv}")
            
            # Process the CSV file
            batch_data = process_csv_batch(args.list_csv, args, verbose=args.verbose)
            
            if not batch_data:
                raise ValueError("No valid entries found in CSV file")
            
            # Run batch inpainting
            batch_results = run_batch_inpainting(batch_data, args)
            
            # Save batch results
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create summary CSV
            summary_data = []
            for result in batch_results:
                if 'error' not in result and 'predicted_sequence' in result:
                    # Extract CSV metadata
                    csv_data = result.get('csv_data', {})
                    
                    # Convert predicted sequence to amino acids
                    predicted_indices = result['predicted_sequence']
                    # Ensure we have a list/array of integers, not tensors
                    if hasattr(predicted_indices, 'cpu'):
                        predicted_indices = predicted_indices.cpu().numpy().tolist()
                    elif hasattr(predicted_indices, 'tolist'):
                        predicted_indices = predicted_indices.tolist()
                    
                    # Handle nested lists or complex structures
                    def flatten_and_convert_to_int(obj):
                        """Recursively flatten and convert to integers."""
                        if isinstance(obj, (list, tuple)):
                            result = []
                            for item in obj:
                                if isinstance(item, (list, tuple)):
                                    result.extend(flatten_and_convert_to_int(item))
                                else:
                                    try:
                                        result.append(int(item))
                                    except (ValueError, TypeError):
                                        print(f"Warning: Could not convert {item} to int")
                                        result.append(0)  # Default to 0 for unknown
                            return result
                        else:
                            try:
                                return [int(obj)]
                            except (ValueError, TypeError):
                                print(f"Warning: Could not convert {obj} to int")
                                return [0]
                    
                    # Flatten and convert predicted_indices
                    predicted_indices = flatten_and_convert_to_int(predicted_indices)
                    
                    predicted_aa_single = ''.join([
                        THREE_TO_ONE.get(IDX_TO_AA[idx], 'X') if 0 <= idx < len(IDX_TO_AA) else 'X' 
                        for idx in predicted_indices
                    ])
                    
                    # Get accuracy metrics if available
                    eval_metrics = result.get('evaluation_metrics', {})
                    
                    summary_entry = {
                        'protein': csv_data.get('protein', ''),
                        'mask_positions': csv_data.get('mask_positions', ''),
                        'predicted_sequence': predicted_aa_single,
                        'sequence_length': len(predicted_indices),
                        'masked_accuracy': eval_metrics.get('masked_accuracy', 0.0),
                        'overall_accuracy': eval_metrics.get('accuracy', 0.0),
                        'masked_correct': eval_metrics.get('masked_correct', 0),
                        'masked_total': eval_metrics.get('masked_total', 0),
                        'success': True
                    }
                else:
                    # Error case
                    csv_data = result.get('csv_data', {})
                    summary_entry = {
                        'protein': csv_data.get('protein', ''),
                        'mask_positions': csv_data.get('mask_positions', ''),
                        'predicted_sequence': '',
                        'sequence_length': 0,
                        'masked_accuracy': 0.0,
                        'overall_accuracy': 0.0,
                        'masked_correct': 0,
                        'masked_total': 0,
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    }
                
                summary_data.append(summary_entry)
            
            # Save summary CSV
            import pandas as pd
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = output_path / "batch_inpainting_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            
            # Save comprehensive NPZ file with all probabilities
            batch_npz_path = output_path / "batch_inpainting_results.npz"
            npz_data = {}
            
            # Add metadata
            import datetime
            npz_data['timestamp'] = np.array([datetime.datetime.now().isoformat()])
            npz_data['model_path'] = np.array([str(args.model)])
            npz_data['batch_size'] = np.array([len(batch_results)])
            
            # Store data for each successful entry
            entry_count = 0
            for i, result in enumerate(batch_results):
                if 'error' not in result and 'predicted_sequence' in result:
                    csv_data = result.get('csv_data', {})
                    
                    # Create entry identifier using protein and mask positions
                    protein = csv_data.get('protein', 'unknown')
                    mask_positions = csv_data.get('mask_positions', 'unknown')
                    # Clean mask positions for use in key names (replace commas and spaces with underscores)
                    clean_mask_positions = str(mask_positions).replace(',', '_').replace(' ', '_')
                    entry_key = f"entry_{protein}_{clean_mask_positions}"
                    
                    # Helper function to convert tensors/arrays to numpy arrays
                    def to_numpy_array(obj):
                        """Convert tensor, list, or array to numpy array."""
                        if hasattr(obj, 'cpu'):  # PyTorch tensor
                            return obj.cpu().numpy()
                        elif hasattr(obj, 'numpy'):  # NumPy array on GPU or similar
                            return obj.numpy()
                        elif isinstance(obj, (list, tuple)):  # Python list/tuple
                            return np.array(obj)
                        else:
                            return obj  # Already numpy array or other format
                    
                    # Store probabilities and predictions for this entry
                    if 'final_probabilities' in result:
                        npz_data[f'{entry_key}_probabilities'] = to_numpy_array(result['final_probabilities'])
                    
                    # Handle predicted_sequence (already processed above for display)
                    npz_data[f'{entry_key}_predicted_indices'] = np.array(predicted_indices, dtype=int)
                    
                    if 'inpainting_mask' in result:
                        npz_data[f'{entry_key}_mask'] = to_numpy_array(result['inpainting_mask']).astype(bool)
                    
                    # Store metadata for this entry
                    npz_data[f'{entry_key}_protein'] = np.array([protein])
                    npz_data[f'{entry_key}_mask_positions'] = np.array([mask_positions])
                    
                    # Store evaluation metrics
                    if 'evaluation_metrics' in result:
                        eval_metrics = result['evaluation_metrics']
                        npz_data[f'{entry_key}_masked_accuracy'] = np.array([eval_metrics.get('masked_accuracy', 0.0)])
                        npz_data[f'{entry_key}_overall_accuracy'] = np.array([eval_metrics.get('accuracy', 0.0)])
                        
                        # Store true sequence if available
                        if 'true_sequence' in eval_metrics:
                            true_sequence_str = eval_metrics['true_sequence']
                            true_indices = []
                            for aa_char in true_sequence_str:
                                aa_three = SINGLE_TO_TRIPLE.get(aa_char.upper(), 'XXX')
                                idx = AA_TO_IDX.get(aa_three, 20)
                                true_indices.append(idx)
                            npz_data[f'{entry_key}_true_indices'] = np.array(true_indices, dtype=int)
                    
                    entry_count += 1
            
            # Add global information
            npz_data['successful_entries'] = np.array([entry_count])
            npz_data['aa_index_to_name'] = np.array(IDX_TO_AA)  # Reference for amino acid indices
            
            # Save NPZ file
            np.savez_compressed(batch_npz_path, **npz_data)
            
            # Print final summary
            successful = sum(1 for entry in summary_data if entry['success'])
            failed = len(summary_data) - successful
            
            if successful > 0:
                avg_masked_acc = sum(entry['masked_accuracy'] for entry in summary_data if entry['success']) / successful / 100
                avg_overall_acc = sum(entry['overall_accuracy'] for entry in summary_data if entry['success']) / successful
                
                print(f"\nBATCH INPAINTING COMPLETED!")
                print(f"Successful entries: {successful}/{len(summary_data)}")
                print(f"Average masked accuracy: {avg_masked_acc:.1%}")
                print(f"Average overall accuracy: {avg_overall_acc:.1%}")
            else:
                print(f"\nBATCH INPAINTING COMPLETED!")
                print(f"No successful entries: {failed} failed")
            
            print(f"\nResults saved to:")
            print(f"  Summary CSV: {summary_csv_path}")
            print(f"  Complete NPZ: {batch_npz_path}  (ALL PROBABILITIES)")
            print(f"\nNPZ file contains {entry_count} entries with full probability matrices!")
            
        else:
            # Single protein inpainting mode (existing logic)
            # Determine if trajectory tracking should be enabled
            num_proteins = 1  # Single protein inpainting for CLI
            enable_trajectory = args.detailed_json or num_proteins < 4
            
            if enable_trajectory:
                print("Enabling trajectory tracking for masked positions (detailed JSON output)")
            
            # Run inpainting - pass mask_positions as string for validation
            results = run_inpainting_inference(
                model_path=args.model,
                uniprot_id=args.uniprot,
                pdb_id=args.pdb_id,
                mask_positions=args.mask_positions,  # Pass as string for validation
                known_sequence=args.template,
                steps=args.steps,
                T=args.t_max,
                t_min=args.t_min,
                verbose=args.verbose,
                pdb_uniprot_mapping_path=args.pdb_uniprot_mapping,
                args=args,
                enable_trajectory=enable_trajectory,
                split_json_override=args.split_json,
                map_pkl_override=args.map_pkl
            )
            
            # Save results
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Convert predicted sequence indices to amino acid names
            predicted_aa_names = []
            predicted_aa_single = []
            for idx in results['predicted_sequence']:
                if 0 <= idx < len(IDX_TO_AA):
                    aa_three = IDX_TO_AA[idx]
                    aa_single = THREE_TO_ONE.get(aa_three, 'X')
                    predicted_aa_names.append(aa_three)
                    predicted_aa_single.append(aa_single)
                else:
                    predicted_aa_names.append('XXX')
                    predicted_aa_single.append('X')
            
            # Convert true sequence if available
            true_aa_names = []
            true_aa_single = []
            if 'evaluation_metrics' in results and 'true_sequence' in results['evaluation_metrics']:
                true_sequence_str = results['evaluation_metrics']['true_sequence']
                # true_sequence is already a string of single letter amino acids
                for aa_char in true_sequence_str:
                    aa_three = SINGLE_TO_TRIPLE.get(aa_char.upper(), 'XXX')
                    true_aa_names.append(aa_three)
                    true_aa_single.append(aa_char.upper())
            
            # Save JSON results (with amino acid names)
            json_path = output_path / "inpainting_results.json"
            with open(json_path, 'w') as f:
                # Convert tensors to lists for JSON serialization
                json_results = {}
                for key, value in results.items():
                    if hasattr(value, 'tolist'):  # PyTorch tensor
                        json_results[key] = value.tolist()
                    elif isinstance(value, dict):
                        json_results[key] = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in value.items()}
                    else:
                        json_results[key] = value
                
                # Add amino acid name information
                json_results['predicted_aa_names'] = predicted_aa_names
                json_results['predicted_aa_single'] = ''.join(predicted_aa_single)
                if true_aa_names:
                    json_results['true_aa_names'] = true_aa_names  
                    json_results['true_aa_single'] = ''.join(true_aa_single)
                
                json.dump(json_results, f, indent=2)
            
            # Save NPZ results (following sample_utils.py format for compatibility with plotting functions)
            npz_path = output_path / "inpainting_results.npz"
            npz_data = {
                # Original inpainting format
                'predicted_indices': np.array(results['predicted_sequence'], dtype=int),
                'predicted_aa_names': np.array(predicted_aa_names),
                'predicted_aa_single': np.array(list(''.join(predicted_aa_single))),
                'final_probabilities': results['final_probabilities'].cpu().numpy(),
                'inpainting_mask': results['inpainting_mask'].cpu().numpy().astype(bool),
                'aa_index_to_name': np.array(IDX_TO_AA),  # Index mapping for reference
                'sequence_length': len(results['predicted_sequence']),
                'masked_positions': np.array(results.get('input_data', {}).get('validated_positions', [])),
                
                # Compatible format for existing plotting functions (struct_0 format)
                'struct_0_probabilities': results['final_probabilities'].cpu().numpy(),
                'struct_0_predicted_indices': np.array(results['predicted_sequence'], dtype=int),
            }
            
            # Add true sequence data if available
            if 'evaluation_metrics' in results and 'true_sequence' in results['evaluation_metrics']:
                true_sequence_str = results['evaluation_metrics']['true_sequence']
                # Convert true sequence string to indices
                true_indices = []
                for aa_char in true_sequence_str:
                    aa_three = SINGLE_TO_TRIPLE.get(aa_char.upper(), 'XXX')
                    idx = AA_TO_IDX.get(aa_three, 20)  # Default to 'XXX' (index 20)
                    true_indices.append(idx)
                
                npz_data['true_indices'] = np.array(true_indices, dtype=int)
                npz_data['true_aa_names'] = np.array(true_aa_names)
                npz_data['true_aa_single'] = np.array(list(''.join(true_aa_single)))
                npz_data['overall_accuracy'] = results['evaluation_metrics'].get('accuracy', 0.0)
                npz_data['masked_accuracy'] = results['evaluation_metrics'].get('masked_accuracy', 0.0)
                npz_data['masked_correct'] = results['evaluation_metrics'].get('masked_correct', 0)
                npz_data['masked_total'] = results['evaluation_metrics'].get('masked_total', 0)
                
                # Add compatible format for plotting functions
                npz_data['struct_0_true_indices'] = np.array(true_indices, dtype=int)
            
            # Add metadata
            import datetime
            npz_data['timestamp'] = np.array([datetime.datetime.now().isoformat()])
            npz_data['model_path'] = np.array([str(args.model)])
            if args.uniprot:
                npz_data['uniprot_id'] = np.array([args.uniprot])
            if args.pdb_id:
                npz_data['pdb_id'] = np.array([args.pdb_id])
                
            np.savez_compressed(npz_path, **npz_data)
            
            # Generate detailed trajectory JSON if trajectory data is available
            if enable_trajectory and 'trajectory_data' in results:
                print(f"Generating detailed trajectory JSON for masked positions...")
                
                # Create a single result entry for JSON generation
                structure_name = args.pdb_id if args.pdb_id else (args.uniprot if args.uniprot else "inpainting_result")
                
                json_results = [{
                    'structure_name': structure_name,
                    'predicted_sequence': results['predicted_sequence'],
                    'true_indices': results.get('evaluation_metrics', {}).get('true_sequence'),
                    'trajectory_data': results['trajectory_data'],
                    'evaluation_metrics': results.get('evaluation_metrics', {}),
                    'alignment_info': results.get('alignment_info', {}),
                    'length': len(results['predicted_sequence'])
                }]
                
                # Generate detailed trajectory JSON
                trajectory_json_path = generate_inpainting_trajectory_json(
                    json_results, [structure_name], str(output_path), "inpainting", K=21
                )
            
            # Print summary
            print("\n" + "="*60)
            print("INPAINTING COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            if 'predicted_sequence' in results:
                indices = results['predicted_sequence']
                # Convert to amino acid sequence
                aa_sequence = ''.join([THREE_TO_ONE.get(IDX_TO_AA[idx], 'X') if 0 <= idx < len(IDX_TO_AA) else 'X' for idx in indices])
                aa_names = [IDX_TO_AA[idx] if 0 <= idx < len(IDX_TO_AA) else 'XXX' for idx in indices]
                
                print(f"Predicted sequence (single letter): {aa_sequence}")
                print(f"Predicted sequence (three letter): {'-'.join(aa_names)}")
                print(f"Length: {len(indices)} amino acids")
                
                # Show masked positions and their predictions
                validated_positions = results.get('input_data', {}).get('validated_positions', [])
                if validated_positions:
                    print(f"\nMasked positions and predictions:")
                    
                    # Get ground truth sequence if available
                    true_indices = None
                    if 'evaluation_metrics' in results and 'true_sequence' in results['evaluation_metrics']:
                        true_indices = results['evaluation_metrics']['true_sequence']
                    
                    for pos in validated_positions:
                        if 0 <= pos < len(indices):
                            # Predicted amino acid
                            pred_idx = indices[pos]
                            pred_single = THREE_TO_ONE.get(IDX_TO_AA[pred_idx], 'X') if 0 <= pred_idx < len(IDX_TO_AA) else 'X'
                            pred_three = IDX_TO_AA[pred_idx] if 0 <= pred_idx < len(IDX_TO_AA) else 'XXX'
                            
                            # Ground truth amino acid if available  
                            if (results.get('evaluation_metrics') and 
                                'true_sequence' in results['evaluation_metrics'] and 
                                0 <= pos < len(results['evaluation_metrics']['true_sequence'])):
                                
                                true_single = results['evaluation_metrics']['true_sequence'][pos].upper()
                                true_three = SINGLE_TO_TRIPLE.get(true_single, 'XXX')
                                
                                # Check if prediction is correct
                                is_correct = pred_single.upper() == true_single.upper()
                                correct_marker = "" if is_correct else ""
                                
                                print(f"  Position {pos+1}: {pred_single} ({pred_three}) | True: {true_single} ({true_three}) {correct_marker}")
                            else:
                                print(f"  Position {pos+1}: {pred_single} ({pred_three}) | True: N/A")
                            
                            # Show probability distribution for this position if available
                            if 'final_probabilities' in results:
                                try:
                                    # Convert to torch tensor if needed
                                    final_probs = results['final_probabilities']
                                    if hasattr(final_probs, 'shape') and len(final_probs.shape) == 2:
                                        if pos < final_probs.shape[0]:
                                            prob_dist_str = format_probability_distribution(final_probs[pos], K=21, top_n=3)
                                            print(prob_dist_str)
                                            print()  # Add blank line for readability
                                except Exception as e:
                                    if args.verbose:
                                        print(f"    Warning: Could not display probability distribution for position {pos+1}: {e}")
                                    print()  # Add blank line anyway
            
            # Show accuracy metrics if available
            if 'evaluation_metrics' in results:
                metrics = results['evaluation_metrics']
                if 'masked_accuracy' in metrics:
                    masked_acc = metrics['masked_accuracy']/100
                    masked_correct = metrics.get('masked_correct', 0)
                    masked_total = metrics.get('masked_total', 0)
                    overall_acc = metrics.get('accuracy', 0.0)
                    
                    print(f"\nPERFORMANCE METRICS:")
                    print(f"  Masked Position Accuracy: {masked_acc:.1%}")
                    if overall_acc > 0:
                        print(f"   Overall Sequence Accuracy: {overall_acc:.1%}")
                        total_positions = len(results.get('predicted_sequence', []))
                        correct_positions = int(overall_acc * total_positions)
                        print(f"   Total Correct Positions: {correct_positions}/{total_positions}")
            
            print(f"\nResults saved to:")
            print(f"  JSON: {json_path}")
            print(f"  NPZ:  {npz_path}")
            print("="*60)
        
    except KeyboardInterrupt:
        print("\n Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
