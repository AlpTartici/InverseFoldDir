# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import subprocess
from glob import glob

import pandas as pd
from cif_parser import (  # Assuming this can provide sequence length
    parse_cif_backbone_auto,
)


def run_tmalign(file1, file2):
    try:
        result = subprocess.run(
            ["TM-align", file1, file2], capture_output=True, text=True
        )
        tm_score = None
        for line in result.stdout.split("\n"):
            if line.startswith("TM-score="):
                tm_score = float(line.split("=")[1].split()[0])
                break
        return tm_score
    except Exception as e:
        print(f"Error running TM-align: {e}")
        return None


def process_files():
    # Gather all ground truth files ending with '.cif'
    ground_truth_files = glob("*.cif")  # Modify as per your needs for the file types
    data = []

    for gt_file in ground_truth_files:
        pdb_id = os.path.basename(gt_file).split(".")[0]
        pred_file = f"pred_for_{pdb_id}.pdb"  # Corresponding predicted file name

        if not os.path.exists(pred_file):
            pred_file = f"{pdb_id}.pdb"

        if os.path.exists(pred_file):
            # Get the sequence length from the ground truth file (using assumed parsing function)
            seq_length = parse_cif_backbone_auto(
                gt_file
            )  # This function will provide the length

            # Run TM-align and capture the score if the binary works
            tm_score = run_tmalign(gt_file, pred_file)

            # Add results to our data list
            data.append([pdb_id, seq_length, tm_score])

    # Convert data list into a pandas DataFrame
    df = pd.DataFrame(data, columns=["PDB ID", "Sequence Length", "TM-score"])

    # Save the data to a CSV file with timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"{timestamp}_tm_align_scores.csv", index=False)


if __name__ == "__main__":
    process_files()
