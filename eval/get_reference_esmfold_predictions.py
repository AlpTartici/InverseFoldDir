# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import os
import signal
from contextlib import contextmanager

import biotite.structure.io as bsio
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

# print if cuda is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU for inference.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU for inference.")
    device = torch.device("cpu")

model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
model = model.to(device)
model.eval()  # Set to evaluation mode
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

# read the json file in ../datasets/dict_pdb_to_clean_seq_for_esmfold.json
import json

print("Script started...")


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timeout functionality"""

    def signal_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")

    # Set the signal handler and a alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the old signal handler and cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def predict_structure(
    sequence,
    corresponding_pdb_id,
    output_dir="../datasets/esmfold_predictions/esmfold_predictions_on_ref_test/",
    timeout_seconds=15,
):
    """
    Predict structure with timeout and return success status
    """
    output_path = f"{output_dir}/pred_for_{corresponding_pdb_id}.pdb"

    try:
        with timeout(timeout_seconds):
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(output_path, "w") as f:
                f.write(output)

            struct = bsio.load_structure(output_path, extra_fields=["b_factor"])
            return struct, True, None

    except TimeoutError as e:
        print(f"  ⚠️  Timeout ({timeout_seconds}s) for {corresponding_pdb_id}")
        return None, False, str(e)
    except Exception as e:
        print(f"  ❌ Error for {corresponding_pdb_id}: {str(e)}")
        return None, False, str(e)


def get_reference_esmfold_predictions(
    input_json_path="../datasets/dict_chain_to_seq_test.json",
    output_dir="../datasets/esmfold_predictions/esmfold_predictions_on_ref_test/",
    timeout_seconds=15,
):
    """
    This function iterates over the dictionary of PDB IDs and their corresponding sequences,
    predicts the structure using ESMFold, and saves the predictions in the specified output directory.
    Skips proteins that already have output files and adds timeout functionality.
    """
    # Load the input JSON file
    with open(input_json_path, "r") as f:
        dict_pdb_to_clean_seq = json.load(f)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Track statistics
    total_proteins = len(dict_pdb_to_clean_seq)
    skipped_existing = 0
    successful_predictions = 0
    failed_predictions = 0
    timed_out = 0

    print(f"Input JSON file: {input_json_path}")
    print(f"Total proteins to process: {total_proteins}")
    print(f"Using timeout: {timeout_seconds} seconds per protein")
    print(f"Output directory: {output_dir}")

    for pdb_id, sequence in tqdm(
        dict_pdb_to_clean_seq.items(), desc="Predicting structures"
    ):
        output_path = f"{output_dir}/pred_for_{pdb_id}.pdb"

        # Skip if output file already exists
        if os.path.exists(output_path):
            print(f"  ⏭️  Skipping {pdb_id} (output file already exists)")
            skipped_existing += 1
            continue

        print(f"  🧬 Predicting structure for {pdb_id} (seq len: {len(sequence)})...")
        struct, success, error = predict_structure(
            sequence, pdb_id, output_dir, timeout_seconds
        )

        if success:
            print(f"  ✅ Structure for {pdb_id} saved successfully")
            successful_predictions += 1
        else:
            failed_predictions += 1
            if "Timed out" in str(error):
                timed_out += 1

    # Print summary statistics
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Total proteins: {total_proteins}")
    print(f"Skipped (existing files): {skipped_existing}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"  - Timed out: {timed_out}")
    print(f"  - Other errors: {failed_predictions - timed_out}")
    print("=" * 50)

    print("All predictions completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ESMFold predictions for reference test sequences."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="../datasets/dict_chain_to_seq_test.json",
        help="Input JSON file containing chain IDs and sequences (default: ../datasets/dict_chain_to_seq_test.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../datasets/esmfold_predictions/esmfold_predictions_on_ref_test/",
        help="Output directory for saving predictions (default: ../datasets/esmfold_predictions/esmfold_predictions_on_ref_test/)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=25,
        help="Timeout duration in seconds for each prediction (default: 25)",
    )
    
    args = parser.parse_args()
    
    get_reference_esmfold_predictions(
        input_json_path=args.input_json,
        output_dir=args.output_dir, 
        timeout_seconds=args.timeoutkol
    )
    print("All ESMFold predictions have been saved.")

# write usage guide from command line to save stdout and stderr to a file with tee command 2>&1 | tee ../datasets/esmfold_predictions_on_ref/log.txt
# Usage: python get_reference_esmfold_predictions.py --input-json ../datasets/missing_chains.json --output-dir ../datasets/esmfold_predictions/ --timeout 40 2>&1 | tee ../output/run_logs/20250714_120000_esmfold_on_reference.txt
# This will save both stdout and stderr to log.txt in the specified directory.
