#!/usr/bin/env python3
"""
sample_cif.py

Direct CIF file sampling script for protein inverse folding.
This allows sampling sequences from any CIF file (PDB or AlphaFold2) 
with optional inpainting functionality.

Usage examples:
# Basic sampling from CIF file
python sample_cif.py --cif_file ../data/examples/1abc.cif

# Inpainting with known sequence
python sample_cif.py --cif_file ../data/examples/1abc.cif \
    --inpainting_mode --full_sequence "MKTAYIAKQRQISFVKSHFSRQLE" \
    --mask_ratio 0.3

# High quality sampling
python sample_cif.py --cif_file ../data/examples/1abc.cif \
    --steps 100 --T 8.0 --verbose
"""

import torch
import sys
import os
import argparse
from datetime import datetime
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dfm_model import DFMNodeClassifier
from data.graph_builder import GraphBuilder
from flow.sampler import sample_forward
from training.config import setup_reproducibility

# Import sampling utilities from the main sample.py
from sample import (
    sample_multiple_proteins, sample_single_protein, 
    save_results_to_files, AMINO_ACIDS
)


def load_cif_as_graph(cif_path: str, device: torch.device) -> torch.geometric.data.Data:
    """
    Load a CIF file and convert it to a graph representation.
    
    Args:
        cif_path: Path to the CIF file
        device: PyTorch device
        
    Returns:
        Graph data object ready for model input
    """
    print(f"Loading structure from: {cif_path}")
    
    # Initialize graph builder
    builder = GraphBuilder()
    
    try:
        # Build graph from CIF file
        graph_data = builder.build(cif_path)
        
        # Move to device
        graph_data = graph_data.to(device)
        
        print(f"Graph created: {graph_data.x_s.shape[0]} nodes, {graph_data.edge_index.shape[1]} edges")
        return graph_data
        
    except Exception as e:
        print(f"Error loading CIF file: {e}")
        raise


def create_fake_dataset_entry(graph_data, cif_path: str):
    """
    Create a fake dataset entry compatible with existing sampling functions.
    This allows us to reuse the sampling code designed for the CathDataset.
    """
    class FakeCIFEntry:
        def __init__(self, graph_data, cif_path):
            self.graph_data = graph_data
            self.cif_path = cif_path
            
        def __getitem__(self, idx):
            if idx != 0:
                raise IndexError("Fake dataset only has one entry")
            return self.graph_data, None, None  # data, y_true, mask
            
        def __len__(self):
            return 1
            
        def get_sequence_length(self, idx=0):
            # Subtract 1 for virtual node
            return self.graph_data.x_s.shape[0] - 1
            
        def get_structure_info(self, idx=0):
            return {
                'source': 'cif_file',
                'path': self.cif_path,
                'length': self.get_sequence_length()
            }
    
    return FakeCIFEntry(graph_data, cif_path)


def sample_from_cif(model, cif_path: str, args):
    """
    Sample protein sequences from a CIF file.
    
    Args:
        model: Trained DFM model
        cif_path: Path to CIF file
        args: Command line arguments
        
    Returns:
        Dictionary with sampling results
    """
    device = next(model.parameters()).device
    
    # Load CIF file as graph
    graph_data = load_cif_as_graph(cif_path, device)
    
    # Create fake dataset entry for compatibility
    fake_dataset = create_fake_dataset_entry(graph_data, cif_path)
    
    # Handle inpainting mode
    if args.inpainting_mode:
        if not args.full_sequence:
            print("Warning: Inpainting mode requires --full_sequence. Proceeding without inpainting.")
            args.inpainting_mode = False
    
    print(f"Sampling with {args.steps} steps, T={args.T}")
    
    try:
        # Use existing sampling function
        if args.inpainting_mode:
            print("Running inpainting sampling...")
            # For inpainting, we need to modify the sampling function call
            # This is a simplified version - you may need to adapt based on your inpainting implementation
            results = sample_single_protein(
                model, fake_dataset, 0, 
                steps=args.steps, T=args.T,
                save_probabilities=args.save_probabilities,
                full_sequence=args.full_sequence,
                known_sequence=args.known_sequence,
                mask_positions=args.mask_positions,
                mask_ratio=args.mask_ratio,
                verbose=args.verbose
            )
        else:
            print("Running standard sampling...")
            results = sample_single_protein(
                model, fake_dataset, 0,
                steps=args.steps, T=args.T, 
                save_probabilities=args.save_probabilities
            )
        
        # Add CIF-specific metadata
        results['cif_file'] = cif_path
        results['sampling_mode'] = 'inpainting' if args.inpainting_mode else 'standard'
        
        return results
        
    except Exception as e:
        print(f"Error during sampling: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Sample protein sequences from CIF files")
    
    # Required arguments
    parser.add_argument('--cif_file', type=str, required=True,
                       help="Path to the CIF file to sample from")
    parser.add_argument('--model_path', type=str, 
                       default='../output/saved_models/best_model.pt',
                       help="Path to the trained model checkpoint")
    
    # Sampling parameters
    parser.add_argument('--steps', type=int, default=50,
                       help="Number of sampling steps")
    parser.add_argument('--T', type=float, default=8.0,
                       help="Maximum time (noise level)")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Output options
    parser.add_argument('--output_prefix', type=str, default='cif_sampling',
                       help="Prefix for output files")
    parser.add_argument('--save_probabilities', action='store_true', default=True,
                       help="Save raw probability distributions")
    parser.add_argument('--no_probabilities', action='store_true',
                       help="Don't save raw probabilities")
    parser.add_argument('--verbose', action='store_true',
                       help="Print detailed information")
    
    # Inpainting arguments
    parser.add_argument('--inpainting_mode', action='store_true',
                       help="Enable inpainting mode")
    parser.add_argument('--full_sequence', type=str, default=None,
                       help="Full sequence for inpainting alignment")
    parser.add_argument('--known_sequence', type=str, default=None,
                       help="Known sequence template with 'X' for unknown positions")
    parser.add_argument('--mask_positions', type=str, default=None,
                       help="Comma-separated positions to mask (0-indexed)")
    parser.add_argument('--mask_ratio', type=float, default=0.3,
                       help="Fraction of positions to randomly mask")
    
    args = parser.parse_args()
    
    # Setup
    setup_reproducibility(args.seed)
    args.save_probabilities = args.save_probabilities and not args.no_probabilities
    
    print("="*60)
    print("CIF FILE SAMPLING")
    print("="*60)
    print(f"CIF file: {args.cif_file}")
    print(f"Model: {args.model_path}")
    print(f"Sampling steps: {args.steps}")
    print(f"Max time T: {args.T}")
    if args.inpainting_mode:
        print("Mode: Inpainting")
        if args.full_sequence:
            print(f"Full sequence: {args.full_sequence}")
        if args.mask_ratio:
            print(f"Mask ratio: {args.mask_ratio}")
    else:
        print("Mode: Standard sampling")
    print("="*60)
    
    # Validate inputs
    if not os.path.exists(args.cif_file):
        print(f"Error: CIF file not found: {args.cif_file}")
        return 1
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = torch.load(args.model_path, map_location=device, weights_only=False)
        model.eval()
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Perform sampling
    results = sample_from_cif(model, args.cif_file, args)
    
    if results is None:
        print("Sampling failed")
        return 1
    
    # Save results
    print("\nSaving results...")
    try:
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cif_name = os.path.splitext(os.path.basename(args.cif_file))[0]
        output_prefix = f"{args.output_prefix}_{cif_name}_{timestamp}"
        
        # Format results for saving
        formatted_results = [{
            'structure_idx': 0,
            'cif_file': args.cif_file,
            'length': len(results.get('predicted_indices', [])),
            'predicted_indices': results.get('predicted_indices'),
            'predicted_aa': results.get('predicted_aa'),
            'predicted_sequence': results.get('predicted_sequence'),
            'final_probabilities': results.get('final_probabilities'),
            'sampling_mode': results.get('sampling_mode', 'standard')
        }]
        
        # Save files
        file_info = save_results_to_files(
            formatted_results,
            output_prefix,
            model_name=os.path.basename(args.model_path).replace('.pt', ''),
            split='cif_file',
            steps=args.steps,
            T=args.T
        )
        
        print(f"\nResults saved:")
        print(f"  Sequences: {file_info['sequences_file']}")
        print(f"  Probabilities: {file_info['probabilities_file']}")  
        print(f"  Metadata: {file_info['metadata_file']}")
        
        # Print summary
        print(f"\nSampling Summary:")
        print(f"  Structure: {cif_name}")
        print(f"  Length: {len(results.get('predicted_indices', []))}")
        print(f"  Predicted sequence: {results.get('predicted_sequence', 'N/A')}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nCIF sampling completed successfully!")
    return 0


if __name__ == '__main__':
    exit(main())
