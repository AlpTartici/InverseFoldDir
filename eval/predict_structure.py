# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# conda activate esmfold # this is the environment where esmfold is installed

import biotite.structure.io as bsio
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")


sequence = "MKTAYIAKQRQISFVKSHFSRQDILD"  # This needs to come from our predictions

corresponding_pdb_id = (
    "1A2B"  # This is a placeholder; replace with the actual PDB ID if needed
)

with torch.no_grad():
    output = model.infer_pdb(sequence)

with open(f"pred_for_{corresponding_pdb_id}.pdb", "w") as f:
    f.write(output)


struct = bsio.load_structure(
    f"pred_for_{corresponding_pdb_id}.pdb", extra_fields=["b_factor"]
)
