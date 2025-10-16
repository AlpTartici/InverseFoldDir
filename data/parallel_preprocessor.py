"""
parallel_preprocessor.py

Parallel preprocessing script for large-scale CIF file datasets.
This will be used when you have your 2.5M protein dataset.
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cif_parser import parse_cif_backbone_auto
from data.graph_builder import GraphBuilder

# Amino acid mapping
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def process_single_cif(cif_path):
    """
    Process a single CIF file into the required format for CathDataset.

    Args:
        cif_path: Path to CIF file

    Returns:
        tuple: (cif_path, entry_dict or None)
    """
    try:
        # Parse CIF file
        coords, scores, residue_types, source = parse_cif_backbone_auto(cif_path)

        # Convert residue types to sequence
        sequence = ''.join([THREE_TO_ONE.get(rt, 'X') for rt in residue_types])

        # Convert coordinates to numpy format expected by CathDataset
        if isinstance(coords, torch.Tensor):
            coords_np = coords.numpy()
        else:
            coords_np = coords

        # Create entry in CATH format
        entry = {
            'name': Path(cif_path).stem,
            'seq': sequence,
            'coords': {
                'N': coords_np[:, 0],
                'CA': coords_np[:, 1],
                'C': coords_np[:, 2],
                'O': coords_np[:, 3] if coords_np.shape[1] > 3 else coords_np[:, 1]
            },
            'source': source,
            'num_chains': 1,
            'CATH': ['unknown']  # Placeholder for compatibility
        }

        return str(cif_path), entry

    except Exception as e:
        print(f"Error processing {cif_path}: {e}")
        return str(cif_path), None

def parallel_preprocess_cifs(cif_file_list, output_pkl, output_json=None,
                           num_processes=None, chunk_size=100000):
    """
    Preprocess CIF files in parallel and save in CATH-compatible format.

    Args:
        cif_file_list: List of CIF file paths
        output_pkl: Output pickle file for processed data
        output_json: Optional output JSON file for splits
        num_processes: Number of parallel processes
        chunk_size: Process files in chunks to manage memory
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    print(f"Processing {len(cif_file_list)} CIF files with {num_processes} processes")

    processed_data = {}
    failed_files = []

    # Process in chunks to manage memory
    for i in range(0, len(cif_file_list), chunk_size):
        chunk = cif_file_list[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(cif_file_list)-1)//chunk_size + 1}")

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(tqdm(
                executor.map(process_single_cif, chunk),
                total=len(chunk),
                desc=f"Chunk {i//chunk_size + 1}"
            ))

            for cif_path, entry in results:
                if entry is not None:
                    processed_data[Path(cif_path).stem] = entry
                else:
                    failed_files.append(cif_path)

    # Save processed data
    print(f"Saving processed data to {output_pkl}")
    with open(output_pkl, 'wb') as f:
        pickle.dump(processed_data, f)

    # Create splits file if requested
    if output_json:
        protein_ids = list(processed_data.keys())
        np.random.shuffle(protein_ids)

        n_train = int(0.8 * len(protein_ids))
        n_val = int(0.1 * len(protein_ids))

        splits = {
            'train': protein_ids[:n_train],
            'validation': protein_ids[n_train:n_train+n_val],
            'test': protein_ids[n_train+n_val:]
        }

        with open(output_json, 'w') as f:
            json.dump(splits, f, indent=2)

        print(f"Splits saved to {output_json}")
        print(f"  Train: {len(splits['train'])}")
        print(f"  Validation: {len(splits['validation'])}")
        print(f"  Test: {len(splits['test'])}")

    print(f"Successfully processed {len(processed_data)} files")
    print(f"Failed to process {len(failed_files)} files")

    # Save failed files list
    if failed_files:
        failed_file = output_pkl.replace('.pkl', '_failed.txt')
        with open(failed_file, 'w') as f:
            for file_path in failed_files:
                f.write(f"{file_path}\n")
        print(f"Failed files list saved to {failed_file}")

    return processed_data, failed_files

def main():
    parser = argparse.ArgumentParser(description="Preprocess CIF files for inverse folding training")
    parser.add_argument('--input_dir', type=str, required=True,
                       help="Directory containing CIF files")
    parser.add_argument('--output_pkl', type=str, required=True,
                       help="Output pickle file for processed data")
    parser.add_argument('--output_json', type=str, default=None,
                       help="Output JSON file for dataset splits")
    parser.add_argument('--num_processes', type=int, default=None,
                       help="Number of parallel processes (default: all CPUs)")
    parser.add_argument('--chunk_size', type=int, default=100000,
                       help="Process files in chunks of this size")
    parser.add_argument('--pattern', type=str, default='*.cif',
                       help="File pattern to match (default: *.cif)")

    args = parser.parse_args()

    # Find all CIF files
    input_path = Path(args.input_dir)
    cif_files = list(input_path.glob(args.pattern))

    if not cif_files:
        print(f"No files found matching pattern {args.pattern} in {args.input_dir}")
        return

    print(f"Found {len(cif_files)} CIF files")

    # Process files
    processed_data, failed_files = parallel_preprocess_cifs(
        cif_files, args.output_pkl, args.output_json,
        args.num_processes, args.chunk_size
    )

    print("Preprocessing completed!")

if __name__ == '__main__':
    main()
