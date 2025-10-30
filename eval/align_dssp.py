#!/usr/bin/env python3
"""
align_dssp.py
Efficient DSSP secondary-structure alignment using Parasail for comparing predictions vs references.

This script takes two JSON files (output from generate_dssp_from_pdbs.py) and performs
pairwise Needleman-Wunsch alignment for all matching chains.

Supports:
    - 3-state DSSP (H/E/C)
    - 8-state DSSP (H,G,I,E,B,T,S,C,-)
    - Handles missing residues (X) by excluding them from alignment
    - Parallel processing for efficiency
    - CSV output with alignment scores and percent identities

Usage:
    # Basic usage (uses all CPU cores)
    python align_dssp.py --ref_json reference.json --pred_json predictions.json --output results.csv
    
    # Sequential processing (no parallelization)
    python align_dssp.py --ref_json ref.json --pred_json pred.json --output results.csv --num_workers 1
    
    # Use 8 parallel workers
    python align_dssp.py --ref_json ref.json --pred_json pred.json --output results.csv --num_workers 8

Output CSV Format:
    chain_name,percent_id_3,score_3,percent_id_8,score_8,aligned_ref_3,aligned_pred_3,aligned_ref_8,aligned_pred_8
    
    - percent_id_3: Percent identity for 3-state DSSP alignment
    - score_3: Alignment score for 3-state DSSP
    - percent_id_8: Percent identity for 8-state DSSP alignment
    - score_8: Alignment score for 8-state DSSP
    - aligned_ref_3/8: Aligned reference sequence (3-state or 8-state)
    - aligned_pred_3/8: Aligned predicted sequence (3-state or 8-state)
"""

import argparse
import csv
import json
import sys
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Tuple, Optional

import parasail
from tqdm import tqdm


# -------------------------------------------------------------------------
# Internal helpers for alignment
# -------------------------------------------------------------------------
def _build_matrix(mode: str, scale: int = 10):
    """
    Build a Parasail scoring matrix for 3-state or 8-state DSSP alignment.
    Scales float scores to integers for Parasail compatibility.
    
    Args:
        mode: '3' for 3-state or '8' for 8-state DSSP
        scale: Integer scale factor for converting floats to ints
        
    Returns:
        tuple: (alphabet, matrix) where alphabet is the character set and matrix is the scoring matrix
    """
    if mode == "3":
        alphabet = "HEC"
        matrix = parasail.matrix_create(alphabet, 0, 0)
        scores = {
            ('H','H'):  1.0, ('H','E'): -1.0, ('H','C'): -0.5,
            ('E','H'): -1.0, ('E','E'):  1.0, ('E','C'): -0.5,
            ('C','H'): -0.5, ('C','E'): -0.5, ('C','C'):  1.0,
        }
        for (a,b), val in scores.items():
            i, j = alphabet.index(a), alphabet.index(b)
            matrix[i,j] = int(val * scale)

    elif mode == "8":
        # Include "-" character which is standard DSSP for loop/irregular
        # 'P' is BioPython's notation for Pi helix (standard DSSP uses 'I')
        alphabet = "HGPIEBTSC-"
        matrix = parasail.matrix_create(alphabet, 0, 0)
        helix = {'H','G','I','P'}  # P is pi helix in BioPython
        sheet = {'E','B'}
        coil  = {'T','S','C','-'}  # "-" is grouped with coil
        
        for a in alphabet:
            for b in alphabet:
                if a == b:
                    val = 1.0
                elif (a in helix and b in helix) or (a in sheet and b in sheet) or (a in coil and b in coil):
                    val = 0.5
                else:
                    val = -0.5
                matrix[alphabet.index(a), alphabet.index(b)] = int(val * scale)
    else:
        raise ValueError("mode must be '3' or '8'")

    return alphabet, matrix


def _filter_missing_residues(seq1: str, seq2: str) -> Tuple[str, str]:
    """
    Filter out positions where either sequence has 'X' (unknown residue).
    This ensures missing residues don't contribute to alignment score.
    
    Args:
        seq1: First DSSP sequence
        seq2: Second DSSP sequence
        
    Returns:
        tuple: (filtered_seq1, filtered_seq2) with X positions removed
    """
    if len(seq1) != len(seq2):
        # If sequences are different lengths, just return as-is
        # The alignment will handle this
        return seq1, seq2
    
    # Filter out positions with X in either sequence
    filtered_seq1 = []
    filtered_seq2 = []
    
    for c1, c2 in zip(seq1, seq2):
        if c1 != 'X' and c2 != 'X':
            filtered_seq1.append(c1)
            filtered_seq2.append(c2)
    
    return ''.join(filtered_seq1), ''.join(filtered_seq2)


# -------------------------------------------------------------------------
# Main alignment function
# -------------------------------------------------------------------------
def align_dssp(seq1: str, seq2: str, mode: str = "3", gap_open: float = 0.4,
               gap_extend: float = 0.0, scale: int = 10, show: bool = False):
    """
    Perform Needleman–Wunsch alignment between two DSSP sequences.

    Args:
        seq1, seq2: DSSP sequences (3-state or 8-state)
        mode: '3' or '8' for DSSP type
        gap_open: gap open penalty (float)
        gap_extend: gap extension penalty (float)
        scale: integer scale factor to preserve fractional weights
        show: if True, prints alignment side-by-side

    Returns:
        (score, percent_identity, aligned_seq1, aligned_seq2)
    """
    # Filter out X (unknown) residues to avoid penalizing missing data
    seq1_filtered, seq2_filtered = _filter_missing_residues(seq1, seq2)
    
    if not seq1_filtered or not seq2_filtered:
        # If filtering removed everything, return zeros
        return 0.0, 0.0, "", ""
    
    _, matrix = _build_matrix(mode, scale)
    result = parasail.nw_trace_striped_16(
        seq1_filtered,
        seq2_filtered,
        int(gap_open * scale),
        int(gap_extend * scale),
        matrix
    )

    aligned1, aligned2 = result.traceback.query, result.traceback.ref
    matches = sum(a == b for a, b in zip(aligned1, aligned2)
                  if a != '-' and b != '-')
    aligned_positions = sum(1 for a, b in zip(aligned1, aligned2)
                            if a != '-' and b != '-')
    percent_id = 100 * matches / aligned_positions if aligned_positions else 0.0
    score = result.score / scale

    if show:
        print(f"\nAlignment ({mode}-state DSSP):")
        print(aligned1)
        print(aligned2)
        print(f"Score: {score:.2f}")
        print(f"Percent identity: {percent_id:.2f}%\n")

    return score, percent_id, aligned1, aligned2


# -------------------------------------------------------------------------
# JSON loading and processing
# -------------------------------------------------------------------------
def load_dssp_json(json_path: str) -> Dict:
    """
    Load DSSP JSON file generated by generate_dssp_from_pdbs.py
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        dict: Dictionary with chain names as keys and DSSP data as values
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def process_single_chain_pair(args_tuple):
    """
    Process a single chain pair for alignment.
    Designed for multiprocessing.
    
    Args:
        args_tuple: (chain_name, ref_data, pred_data) where:
            - chain_name: Name of the chain
            - ref_data: Reference DSSP data dict or None
            - pred_data: Predicted DSSP data dict or None
            
    Returns:
        dict: Results dictionary with alignment metrics
    """
    chain_name, ref_data, pred_data = args_tuple
    
    result = {
        'chain_name': chain_name,
        'percent_id_3': None,
        'score_3': None,
        'percent_id_8': None,
        'score_8': None,
        'aligned_ref_3': '',
        'aligned_pred_3': '',
        'aligned_ref_8': '',
        'aligned_pred_8': '',
        'status': 'missing'
    }
    
    try:
        # Check if both ref and pred exist
        if ref_data is None or pred_data is None:
            if ref_data is None and pred_data is None:
                result['status'] = 'missing_both'
            elif ref_data is None:
                result['status'] = 'missing_reference'
            else:
                result['status'] = 'missing_prediction'
            return result
        
        # Extract DSSP sequences
        ref_dssp_3 = ''.join(ref_data['dssp_3'])
        pred_dssp_3 = ''.join(pred_data['dssp_3'])
        ref_dssp_8 = ''.join(ref_data['dssp_8'])
        pred_dssp_8 = ''.join(pred_data['dssp_8'])
        
        # Align 3-state DSSP
        score_3, percent_id_3, aligned_ref_3, aligned_pred_3 = align_dssp(
            ref_dssp_3, pred_dssp_3, mode="3"
        )
        
        # Align 8-state DSSP
        score_8, percent_id_8, aligned_ref_8, aligned_pred_8 = align_dssp(
            ref_dssp_8, pred_dssp_8, mode="8"
        )
        
        # Update result
        result.update({
            'percent_id_3': percent_id_3,
            'score_3': score_3,
            'percent_id_8': percent_id_8,
            'score_8': score_8,
            'aligned_ref_3': aligned_ref_3,
            'aligned_pred_3': aligned_pred_3,
            'aligned_ref_8': aligned_ref_8,
            'aligned_pred_8': aligned_pred_8,
            'status': 'success'
        })
        
    except Exception as e:
        result['status'] = f'error: {str(e)}'
        print(f"\nError processing chain {chain_name}: {e}")
        traceback.print_exc()
    
    return result


def align_dssp_pairs(ref_json_path: str, pred_json_path: str, num_workers: Optional[int] = None):
    """
    Align DSSP pairs from two JSON files.
    
    Args:
        ref_json_path: Path to reference DSSP JSON
        pred_json_path: Path to predicted DSSP JSON
        num_workers: Number of parallel workers (None = all CPUs, 1 = sequential)
        
    Returns:
        list: List of result dictionaries with alignment metrics
    """
    # Load JSON files
    print(f"Loading reference JSON: {ref_json_path}")
    ref_data = load_dssp_json(ref_json_path)
    print(f"  Found {len(ref_data)} chains in reference")
    
    print(f"Loading prediction JSON: {pred_json_path}")
    pred_data = load_dssp_json(pred_json_path)
    print(f"  Found {len(pred_data)} chains in prediction")
    
    # Get all unique chain names from both datasets
    all_chains = sorted(set(ref_data.keys()) | set(pred_data.keys()))
    print(f"\nTotal unique chains to process: {len(all_chains)}")
    
    # Report missing chains
    ref_only = set(ref_data.keys()) - set(pred_data.keys())
    pred_only = set(pred_data.keys()) - set(ref_data.keys())
    
    if ref_only:
        print(f"  Chains only in reference: {len(ref_only)}")
    if pred_only:
        print(f"  Chains only in prediction: {len(pred_only)}")
    
    # Prepare arguments for processing
    process_args = []
    for chain_name in all_chains:
        ref_chain_data = ref_data.get(chain_name, None)
        pred_chain_data = pred_data.get(chain_name, None)
        process_args.append((chain_name, ref_chain_data, pred_chain_data))
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    elif num_workers <= 0:
        num_workers = cpu_count()
    
    # Process chains
    results = []
    
    if num_workers == 1:
        # Sequential processing
        print("\nProcessing chains sequentially (num_workers=1)")
        for args in tqdm(process_args, desc="Aligning chains"):
            results.append(process_single_chain_pair(args))
    else:
        # Parallel processing
        print(f"\nProcessing chains in parallel using {num_workers} workers")
        chunksize = max(1, len(process_args) // (num_workers * 4))
        
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_single_chain_pair, process_args, chunksize=chunksize),
                    total=len(process_args),
                    desc="Aligning chains"
                )
            )
    
    return results


def save_results_to_csv(results: list, output_path: str):
    """
    Save alignment results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
    """
    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Define CSV columns
    fieldnames = [
        'chain_name',
        'percent_id_3',
        'score_3', 
        'percent_id_8',
        'score_8',
        'status',
        'aligned_ref_3',
        'aligned_pred_3',
        'aligned_ref_8',
        'aligned_pred_8'
    ]
    
    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Create row dict with only the fields we want
            row = {field: result.get(field, None) for field in fieldnames}
            writer.writerow(row)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    missing = sum(1 for r in results if 'missing' in r['status'])
    errors = sum(1 for r in results if 'error' in r['status'])
    
    print("\n" + "=" * 60)
    print("Alignment Summary:")
    print(f"  Total chains: {len(results)}")
    print(f"  Successfully aligned: {successful}")
    print(f"  Missing data: {missing}")
    print(f"  Errors: {errors}")
    
    if successful > 0:
        # Calculate average percent identities for successful alignments
        valid_results = [r for r in results if r['status'] == 'success']
        avg_percent_3 = sum(r['percent_id_3'] for r in valid_results) / len(valid_results)
        avg_percent_8 = sum(r['percent_id_8'] for r in valid_results) / len(valid_results)
        
        print(f"\n  Average 3-state percent identity: {avg_percent_3:.2f}%")
        print(f"  Average 8-state percent identity: {avg_percent_8:.2f}%")
    
    print("=" * 60)


# -------------------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------------------
def main():
    """
    Main function to orchestrate DSSP alignment from command line.
    """
    parser = argparse.ArgumentParser(
        description="Align DSSP secondary structure predictions using Needleman-Wunsch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses all CPU cores)
  python align_dssp.py --ref_json reference.json --pred_json predictions.json --output results.csv
  
  # Sequential processing (no parallelization)
  python align_dssp.py --ref_json ref.json --pred_json pred.json --output results.csv --num_workers 1
  
  # Use 8 parallel workers
  python align_dssp.py --ref_json ref.json --pred_json pred.json --output results.csv --num_workers 8

CSV Output:
  The output CSV contains the following columns (in order of importance):
    - chain_name: Identifier for the protein chain
    - percent_id_3: Percent identity for 3-state DSSP (H/E/C) - PRIMARY METRIC
    - percent_id_8: Percent identity for 8-state DSSP - PRIMARY METRIC
    - score_3: Alignment score for 3-state DSSP
    - score_8: Alignment score for 8-state DSSP
    - status: Processing status (success, missing_reference, missing_prediction, error)
    - aligned_ref_3: Aligned reference sequence (3-state) - for sanity checking
    - aligned_pred_3: Aligned predicted sequence (3-state) - for sanity checking
    - aligned_ref_8: Aligned reference sequence (8-state) - for sanity checking
    - aligned_pred_8: Aligned predicted sequence (8-state) - for sanity checking
        """
    )
    
    parser.add_argument(
        '--ref_json',
        type=str,
        required=True,
        help='Path to reference DSSP JSON file (from generate_dssp_from_pdbs.py)'
    )
    
    parser.add_argument(
        '--pred_json',
        type=str,
        required=True,
        help='Path to predicted DSSP JSON file (from generate_dssp_from_pdbs.py)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers for processing. Default: cpu_count(). Use 1 for sequential processing.'
    )
    
    args = parser.parse_args()
    
    # Determine number of workers for display
    display_workers = args.num_workers if args.num_workers is not None else cpu_count()
    
    print("DSSP Alignment Tool")
    print("=" * 60)
    print(f"Reference JSON: {args.ref_json}")
    print(f"Prediction JSON: {args.pred_json}")
    print(f"Output CSV: {args.output}")
    print(f"Number of workers: {display_workers}" + (" (sequential)" if display_workers == 1 else " (parallel)"))
    print("=" * 60 + "\n")
    
    try:
        # Perform alignments
        results = align_dssp_pairs(args.ref_json, args.pred_json, num_workers=args.num_workers)
        
        if not results:
            print("No results to save. Exiting.")
            return 1
        
        # Save to CSV
        save_results_to_csv(results, args.output)
        
        print("\nProcessing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 130
    except FileNotFoundError as e:
        print(f"\n\nError: {e}")
        return 1
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        traceback.print_exc()
        return 1


# -------------------------------------------------------------------------
# Legacy test harness (for backwards compatibility)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Check if running in command-line mode or legacy test mode
    if len(sys.argv) > 1:
        # Command-line mode
        sys.exit(main())
    else:
        # Legacy test mode (for quick testing)
        print("Running in legacy test mode (no command-line args provided)")
        print("For production use, run with --help to see command-line options\n")
        
        print("=== DSSP 3-State Alignment ===")
        align_dssp("HHHEECCC", "HHEECCCH", mode="3", show=True)

        print("=== DSSP 8-State Alignment ===")
        align_dssp("HHHTTSSCC", "HGHTTCC", mode="8", show=True)
