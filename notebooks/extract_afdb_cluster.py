# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import json
from tqdm import tqdm

print("Starting extraction of AFDB50 cluster representative IDs...", flush=True)
# load the dictionary from the file
with open("../datasets/test_val_pdb_uniprot_mapping.json", "r") as f:
    dict_pdb_uniprot = json.load(f)

print(
    f"Loaded {len(dict_pdb_uniprot)} UniProt entries from the dictionary. Proceeding with AFDB50 cluster extraction..."
)
df_af_clusters_mem = pd.read_csv(
    "../datasets/af_clusters/7-AFDB50-repId_memId.tsv.gz",
    sep="\t",
    compression="gzip",
    header=0,
)

df_af_clusters_mem.columns = ["repID", "memID", "idk"]

print(
    "AFDB50 cluster data loaded. Extracting representative IDs for each UniProt entry..."
)
# write code to extract the repID from the repID column for each uniprot entry in the dictionary and save it like a dictionary
# write code to extract the repID from the repID column for each uniprot entry in the dictionary and save it like a dictionary
dict_repID = {}
for pdb_id in tqdm(dict_pdb_uniprot.keys()):
    uniprot_id = dict_pdb_uniprot[pdb_id]
    repID = df_af_clusters_mem.loc[df_af_clusters_mem["memID"] == uniprot_id, "repID"]
    if not repID.empty:
        dict_repID[pdb_id] = {
            "cluster_repID": repID.values[0],
            "uniprot_id": uniprot_id,
        }
    else:
        dict_repID[pdb_id] = None
# save the list to a file
with open(
    "../datasets/af_clusters/afdb_clusters_repid_for_test_and_val.json", "w"
) as f:
    json.dump(dict_repID, f, indent=4)
