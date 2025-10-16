
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Structure Comparison Module

This module handles structural comparisons between predicted and reference structures
using TM-score calculations via tmtools.
"""

import logging
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tmtools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StructureComparator:
    """
    A class to handle structure comparisons using TM-score and other metrics.
    """

    def __init__(self):
        """Initialize the structure comparator."""
        logger.info("Initialized StructureComparator with tmtools")

    def parse_pdb_structure(self, pdb_path: str) -> Optional[Tuple[np.ndarray, str]]:
        """
        Parse PDB file to extract CA coordinates and sequence.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Tuple of (coordinates array, sequence string) or None if parsing fails
        """
        try:
            coords = []
            sequence = []

            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and ' CA ' in line:
                        # Extract CA coordinates
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])

                        # Extract residue name and convert to single letter
                        res_name = line[17:20].strip()
                        aa_map = {
                            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
                            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
                            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
                            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
                        }
                        sequence.append(aa_map.get(res_name, 'X'))

            if len(coords) == 0:
                logger.error(f"No CA atoms found in {pdb_path}")
                return None

            coords_array = np.array(coords, dtype=np.float64)
            seq_string = ''.join(sequence)

            logger.debug(f"Parsed {len(coords)} CA atoms from {pdb_path}")
            return coords_array, seq_string

        except Exception as e:
            logger.error(f"Failed to parse PDB structure {pdb_path}: {e}")
            return None

    def calculate_tm_score(self, structure1_path: str, structure2_path: str) -> Optional[Dict[str, float]]:
        """
        Calculate TM-score between two protein structures.

        Args:
            structure1_path: Path to first structure (PDB format)
            structure2_path: Path to second structure (PDB format)

        Returns:
            Dict with TM-score metrics or None if calculation fails
        """
        try:
            # Validate file existence early (fail-fast)
            if not os.path.exists(structure1_path):
                logger.error(f"Structure file not found: {structure1_path}")
                return None
            if not os.path.exists(structure2_path):
                logger.error(f"Structure file not found: {structure2_path}")
                return None

            # Parse both structures
            struct1_data = self.parse_pdb_structure(structure1_path)
            struct2_data = self.parse_pdb_structure(structure2_path)

            if struct1_data is None or struct2_data is None:
                return None

            coords1, seq1 = struct1_data
            coords2, seq2 = struct2_data

            # Calculate TM-score using tmtools
            result = tmtools.tm_align(coords1, coords2, seq1, seq2)

            # Calculate aligned length from alignment string (colons indicate matched positions)
            aligned_length = len(result.seqM)

            # Calculate sequence identity from alignment
            seq_identity = result.seqM.count(':') / aligned_length if aligned_length > 0 else 0.0

            metrics = {
                'tm_score': result.tm_norm_chain1,  # TM-score normalized by first structure
                'tm_score_ref': result.tm_norm_chain2,  # TM-score normalized by second structure
                'rmsd': result.rmsd,  # RMSD of aligned residues
                'aligned_length': aligned_length,  # Number of aligned residues
                'seq_identity': seq_identity,  # Sequence identity of aligned region
                'aligned_seq1': result.seqxA,  # Aligned sequence 1
                'aligned_seq2': result.seqyA   # Aligned sequence 2
            }

            logger.debug(f"TM-score calculated: {metrics['tm_score']:.4f}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate TM-score between {structure1_path} and {structure2_path}: {e}")
            return None

    def parse_pdb_sequence_length(self, pdb_path: str) -> Optional[int]:
        """
        Extract sequence length from a PDB file.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Sequence length or None if parsing fails
        """
        try:
            ca_atoms = 0
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and ' CA ' in line:
                        ca_atoms += 1
            return ca_atoms

        except Exception as e:
            logger.error(f"Failed to parse sequence length from {pdb_path}: {e}")
            return None


class BatchStructureComparator:
    """
    Handles batch structure comparisons from prediction results.
    """

    def __init__(self, predicted_dir: str, reference_dir: str, csv_path: str, verbose: bool = False,
                 protein_subset: List[str] = None, protein_timeout_minutes: float = 3.0):
        """
        Initialize batch comparator.

        Args:
            predicted_dir: Directory containing predicted structures
            reference_dir: Directory containing reference structures
            csv_path: Path to CSV file with sequence predictions
            verbose: Enable verbose logging
            protein_subset: List of specific proteins to compare (optional)
            protein_timeout_minutes: Timeout per protein in minutes
        """
        self.predicted_dir = Path(predicted_dir)
        self.reference_dir = Path(reference_dir)
        try:
            self.csv_path = Path(csv_path)
        except:
            self.csv_path = None
            print("Failed to parse CSV path")
        self.comparator = StructureComparator()
        self.verbose = verbose
        self.protein_subset = protein_subset
        self.protein_timeout_minutes = protein_timeout_minutes

        # Validate inputs
        if not self.predicted_dir.exists():
            raise FileNotFoundError(f"Predicted structures directory not found: {predicted_dir}")
        if not self.reference_dir.exists():
            raise FileNotFoundError(f"Reference structures directory not found: {reference_dir}")
        if not self.csv_path is None:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Initialized batch comparator:")
        logger.info(f"  Predicted: {self.predicted_dir}")
        logger.info(f"  Reference: {self.reference_dir}")
        logger.info(f"  CSV: {self.csv_path}")

        if verbose:
            pred_count = len(list(self.predicted_dir.glob("*.pdb")))
            ref_count = len(list(self.reference_dir.glob("*.pdb")))
            logger.debug(f"Structure file counts:")
            logger.debug(f"  Predicted structures: {pred_count}")
            logger.debug(f"  Reference structures: {ref_count}")

    def load_sequence_data(self) -> pd.DataFrame:
        """Load sequence prediction data from CSV."""
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(df)} sequence predictions")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")


    def find_structure_pairs(self) -> List[Tuple[str, str, str]]:
        """
        Find pairs of predicted and reference structures.

        Returns:
            List of tuples: (structure_name, predicted_path, reference_path)
        """
        pairs = []

        # List all predicted files
        pred_files = list(self.predicted_dir.glob("pred_for_*.pdb"))
        logger.info(f"Found {len(pred_files)} predicted structure files")

        if self.verbose:
            logger.debug(f"Predicted files found: {[f.name for f in pred_files[:5]]}")
            if len(pred_files) > 5:
                logger.debug(f"... and {len(pred_files) - 5} more")

        # Filter by protein subset if provided
        if self.protein_subset:
            initial_count = len(pred_files)
            pred_files = [f for f in pred_files if f.stem.replace("pred_for_", "") in self.protein_subset]
            logger.info(f"Filtered predicted files: {len(pred_files)} from subset (was {initial_count} total)")

        # List all reference files (they are named directly as structure_name.pdb, not pred_for_structure_name.pdb)
        ref_files = list(self.reference_dir.glob("*.pdb"))
        logger.info(f"Found {len(ref_files)} reference structure files")

        if self.verbose:
            logger.debug(f"Reference files found: {[f.name for f in ref_files[:5]]}")
            if len(ref_files) > 5:
                logger.debug(f"... and {len(ref_files) - 5} more")

        if len(ref_files) == 0:
            logger.error(f"No reference files found in {self.reference_dir}")
            logger.error("Check if the reference directory path is correct")
            return pairs

        for pred_file in pred_files:
            structure_name = pred_file.stem.replace("pred_for_", "")
            if 'all_chain_pdbs' in self.reference_dir.name:
                ref_file = self.reference_dir / f"{structure_name.split('.')[0]}_chain{structure_name.split('.')[1]}.pdb"
            else:
                ref_file = self.reference_dir / f"{structure_name}.pdb"  # Reference files are named directly as structure_name.pdb

            if ref_file.exists():
                pairs.append((structure_name, str(pred_file), str(ref_file)))
                if self.verbose:
                    logger.debug(f"✓ Found pair for {structure_name}")
            else:
                logger.warning(f"✗ Reference structure not found for {structure_name}")
                if self.verbose:
                    logger.debug(f"  Expected: {ref_file}")

        logger.info(f"Found {len(pairs)} structure pairs for comparison")

        if len(pairs) == 0:
            logger.error("No structure pairs found for comparison!")
            logger.error("This means no predicted structures have corresponding reference structures.")
            logger.error(f"Predicted dir: {self.predicted_dir}")
            logger.error(f"Reference dir: {self.reference_dir}")

        return pairs

    def compare_structures(self, output_path: str) -> pd.DataFrame:
        """
        Compare all structure pairs and save results.

        Args:
            output_path: Path to save comparison results CSV

        Returns:
            DataFrame with comparison results
        """
        # Load sequence data
        if self.verbose:
            logger.debug("Loading sequence prediction data...")

        if self.csv_path is not None:
            seq_df = self.load_sequence_data()
            seq_data = {row['structure_name']: row for _, row in seq_df.iterrows()}
        else:
            seq_data = {}

        # Find structure pairs
        if self.verbose:
            logger.debug("Finding structure pairs for comparison...")
        pairs = self.find_structure_pairs()

        if self.verbose:
            logger.debug(f"Found {len(pairs)} pairs to compare")
            logger.debug(f"Sequence data available for {len(seq_data)} structures")

        results = []
        timeout_count = 0
        logger.info(f"Starting structural comparison for {len(pairs)} pairs...")

        for i, (structure_name, pred_path, ref_path) in enumerate(pairs):
            if self.verbose:
                logger.debug(f"Comparing {structure_name} ({i+1}/{len(pairs)})")
                logger.debug(f"  Predicted: {pred_path}")
                logger.debug(f"  Reference: {ref_path}")
            else:
                logger.info(f"Comparing {structure_name} ({i+1}/{len(pairs)})")

            # Get sequence data
            seq_info = seq_data.get(structure_name, {})

            if self.verbose and structure_name not in seq_data:
                logger.debug(f"  Warning: No sequence data found for {structure_name}")

            # Calculate structural metrics with timeout
            try:
                metrics = self._calculate_metrics_with_timeout(pred_path, ref_path, structure_name)

                if metrics is None:
                    logger.error(f"Failed to calculate metrics for {structure_name}")
                    continue

                # Get sequence lengths from PDB files
                if self.verbose:
                    logger.debug("  Parsing structure lengths...")
                pred_length = self.comparator.parse_pdb_sequence_length(pred_path)
                ref_length = self.comparator.parse_pdb_sequence_length(ref_path)

                logger.info(f"✓ {structure_name}: TM={metrics['tm_score']:.4f}, RMSD={metrics['rmsd']:.2f}Å")
                if self.verbose:
                    logger.debug(f"  Lengths: pred={pred_length}, ref={ref_length}")

                # Compile results
                result = {
                    'structure_name': structure_name,
                    'sequence_length': seq_info.get('length', None),
                    'predicted_sequence': seq_info.get('predicted_sequence', ''),
                    'true_sequence': seq_info.get('true_sequence', ''),
                    'sequence_accuracy': seq_info.get('accuracy', None),
                    'pred_structure_length': pred_length,
                    'ref_structure_length': ref_length,
                    'tm_score': metrics['tm_score'],
                    'tm_score_ref': metrics['tm_score_ref'],
                    'rmsd': metrics['rmsd'],
                    'aligned_length': metrics['aligned_length'],
                    'seq_id_aligned': metrics['seq_identity'],
                    'predicted_structure_path': pred_path,
                    'reference_structure_path': ref_path
                }

                results.append(result)

            except TimeoutError:
                timeout_count += 1
                logger.warning(f"⏰ Timeout ({self.protein_timeout_minutes} min) for {structure_name}")
                continue

        # Create DataFrame and save results
        if self.verbose:
            logger.debug("Creating results DataFrame...")
        results_df = pd.DataFrame(results)

        if timeout_count > 0:
            logger.warning(f"Total timeouts: {timeout_count}")

        if len(results_df) > 0:
            # Sort by TM-score descending
            results_df = results_df.sort_values('tm_score', ascending=False)

            # Save to CSV
            if self.verbose:
                logger.debug(f"Saving results to {output_path}")
            results_df.to_csv(output_path, index=False)
            logger.info(f"Comparison results saved to {output_path}")

            # Print summary statistics
            self._print_summary_stats(results_df)
        else:
            logger.warning("No successful comparisons were performed")
            # Create empty DataFrame with correct columns for consistency
            empty_df = pd.DataFrame(columns=[
                'structure_name', 'sequence_length', 'predicted_sequence', 'true_sequence',
                'sequence_accuracy', 'pred_structure_length', 'ref_structure_length',
                'tm_score', 'tm_score_ref', 'rmsd', 'aligned_length', 'seq_id_aligned',
                'predicted_structure_path', 'reference_structure_path'
            ])
            # Save empty CSV to prevent FileNotFoundError downstream
            empty_df.to_csv(output_path, index=False)
            logger.info(f"Empty comparison results saved to {output_path}")

            # Also save a summary of what went wrong
            error_summary_path = Path(output_path).parent / "comparison_errors.txt"
            with open(error_summary_path, 'w') as f:
                f.write("Structure Comparison Error Summary\n")
                f.write("=" * 40 + "\n")
                f.write(f"Found {len(pairs)} structure pairs to compare\n")
                f.write("Issues encountered:\n")
                f.write("- All TM-score calculations failed\n")
                f.write("- Check PDB file formats and tmtools installation\n")
                f.write("- Verify structure file integrity\n")

            return empty_df

        return results_df

    def _print_summary_stats(self, df: pd.DataFrame):
        """Print summary statistics of comparison results."""
        print("\n" + "="*60)
        print("STRUCTURAL COMPARISON SUMMARY")
        print("="*60)

        print(f"Total comparisons: {len(df)}")
        print(f"Average TM-score: {df['tm_score'].mean():.4f} ± {df['tm_score'].std():.4f}")
        print(f"Average RMSD: {df['rmsd'].mean():.2f} ± {df['rmsd'].std():.2f} Å")
        print(f"Average sequence accuracy: {df['sequence_accuracy'].mean():.2f}%")

        print("\nTM-score distribution:")
        print(f"  > 0.5 (good): {(df['tm_score'] > 0.5).sum()} ({(df['tm_score'] > 0.5).mean()*100:.1f}%)")
        print(f"  > 0.4: {(df['tm_score'] > 0.4).sum()} ({(df['tm_score'] > 0.4).mean()*100:.1f}%)")
        print(f"  > 0.3: {(df['tm_score'] > 0.3).sum()} ({(df['tm_score'] > 0.3).mean()*100:.1f}%)")

        print("\nTop 5 structures by TM-score:")
        top5 = df.nlargest(5, 'tm_score')[['structure_name', 'tm_score', 'rmsd', 'sequence_accuracy']]
        for _, row in top5.iterrows():
            print(f"  {row['structure_name']}: TM={row['tm_score']:.4f}, RMSD={row['rmsd']:.2f}Å, SeqAcc={row['sequence_accuracy']:.1f}%")

        print("\nWorst 5 structures by TM-score:")
        worst5 = df.nsmallest(5, 'tm_score')[['structure_name', 'tm_score', 'rmsd', 'sequence_accuracy']]
        for _, row in worst5.iterrows():
            print(f"  {row['structure_name']}: TM={row['tm_score']:.4f}, RMSD={row['rmsd']:.2f}Å, SeqAcc={row['sequence_accuracy']:.1f}%")

        print("="*60)

    def _calculate_metrics_with_timeout(self, pred_path: str, ref_path: str, structure_name: str) -> Optional[Dict[str, float]]:
        """
        Calculate structural metrics with timeout handling.

        Args:
            pred_path: Path to predicted structure
            ref_path: Path to reference structure
            structure_name: Name of the structure (for logging)

        Returns:
            Dict with metrics or None if failed

        Raises:
            TimeoutError: If calculation exceeds timeout
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Comparison timeout for {structure_name}")

        # Set up timeout
        timeout_seconds = int(self.protein_timeout_minutes * 60)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            if self.verbose:
                logger.debug(f"  Calculating TM-score and RMSD with {timeout_seconds}s timeout...")

            metrics = self.comparator.calculate_tm_score(pred_path, ref_path)
            signal.alarm(0)  # Cancel timeout
            return metrics

        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            logger.warning(f"Structure comparison timed out for {structure_name} after {self.protein_timeout_minutes} minutes")
            raise

        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            logger.error(f"Structure comparison failed for {structure_name}: {e}")
            return None

        finally:
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler


def main():
    """Example usage of the structure comparator."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare predicted and reference protein structures")
    parser.add_argument("predicted_dir", help="Directory containing predicted structures")
    parser.add_argument("reference_dir", help="Directory containing reference structures")
    parser.add_argument("--csv_path", default=None, help="Path to CSV file with sequence predictions")
    parser.add_argument("--output_path", default='./datasets/pdb_esmfold_comparison_train_set.csv', help="Path to save comparison results CSV")

    args = parser.parse_args()

    try:
        comparator = BatchStructureComparator(
            args.predicted_dir,
            args.reference_dir,
            args.csv_path
        )

        results_df = comparator.compare_structures(args.output_path)

        if len(results_df) > 0:
            print(f"\nComparison completed successfully!")
            print(f"Results saved to: {args.output_path}")
        else:
            print("No comparisons were successful.")
            return 1

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
