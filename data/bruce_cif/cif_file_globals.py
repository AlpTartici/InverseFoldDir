"""Contains the global variables used by CIF file parsers and related code.
"""

# Atoms used by proteinMPNN in the correct order
MPNN_TARGET_ATOMS = ("N", "CA", "C", "O")

# Get a set of the known types of polymers found in the PDB
KNOWN_POLYMER_TYPES = {
    "cyclic-pseudo-peptide",
    "other",
    "peptide nucleic acid",
    "polydeoxyribonucleotide",
    "polydeoxyribonucleotide/polyribonucleotide hybrid",
    "polypeptide(D)",
    "polypeptide(L)",
    "polyribonucleotide",
}

# A dictionary that converts the PDBx/mmCIF keys to the PDB keys
MMCIF_TO_PDB = {
    "_entity.pdbx_description": "MOLECULE",
    "_entity.pdbx_ec": "EC",
    "_entity.pdb_details": "OTHER_DETAILS",
    "_entity_src_nat.pdbx_organism_scientific": "ORGANISM_SCIENTIFIC",
    "_entity_src_gen.pdbx_gene_src_scientific_name": "ORGANISM_SCIENTIFIC",
    "_pdbx_entity_src_syn.organism_scientific": "ORGANISM_SCIENTIFIC",
    "_entity_src_nat.common_name": "ORGANISM_COMMON",
    "_entity_src_gen.gene_src_common_name": "ORGANISM_COMMON",
    "_pdbx_entity_src_syn.organism_common_name": "ORGANISM_COMMON",
    "_entity_src_nat.pdbx_ncbi_taxonomy_id": "ORGANISM_TAXID",
    "_entity_src_gen.gene_src_ncbi_taxonomy_id": "ORGANISM_TAXID",
    "_pdbx_entity_src_syn.ncbi_taxonomy_id": "ORGANISM_TAXID",
    "_entity_src_nat.pdbx_cell_line": "CELL_LINE",
    "_entity_src_gen.pdbx_gene_src_cell_line": "CELL_LINE",
    "_entity_src_nat.pdbx_organ": "ORGAN",
    "_entity_src_gen.pdbx_gene_src_organ": "ORGAN",
    "_entity_src_nat.tissue": "TISSUE",
    "_entity_src_gen.gene_src_tissue": "TISSUE",
    "_entity_src_nat.pdbx_cell": "CELL",
    "_entity_src_gen.pdbx_gene_src_cell": "CELL",
    "_entity_src_nat.pdbx_organelle": "ORGANELLE",
    "_entity_src_gen.pdbx_gene_src_organelle": "ORGANELLE",
    "_entity_src_nat.pdbx_cellular_location": "CELLULAR_LOCATION",
    "_entity_src_gen.pdbx_gene_src_cellular_location": "CELLULAR_LOCATION",
    "_entity_src_gen.pdbx_host_org_scientific_name": "EXPRESSION_SYSTEM",
    "_entity_src_gen.host_org_common_name": "EXPRESSION_SYSTEM_COMMON",
    "_entity_src_gen.pdbx_host_org_ncbi_taxonomy_id": "EXPRESSION_SYSTEM_TAXID",
    "_entity_src_gen.pdbx_host_org_organ": "EXPRESSION_SYSTEM_ORGAN",
    "_entity_src_gen.host_org_tissue": "EXPRESSION_SYSTEM_TISSUE",
    "_entity_src_gen.pdbx_host_org_cell": "EXPRESSION_SYSTEM_CELL",
    "_entity_src_gen.pdbx_host_org_organelle": "EXPRESSION_SYSTEM_ORGANELLE",
    "_entity_src_gen.pdbx_host_org_cellular_location": "EXPRESSION_SYSTEM_CELLULAR_LOCATION",
    "_entity_src_nat.details": "OTHER_DETAILS",
    "_entity_src_gen.pdbx_description": "OTHER_DETAILS",
    "_pdbx_entity_src_syn.details": "OTHER_DETAILS",
    "_citation.title": "TITL",
    "_citation.pdbx_database_id_pubmed": "PMID",
    "_citation.pdbx_database_id_doi": "DOI",
    "_citation.journal_id_astm": "REFN ASTN",
    "_citation.journal_id_issn": "REFN ISSN",
    "_citation.book_id_isbn": "REFN ISBN",
    "_struct_site.id": "SITE_IDENTIFIER",
    "_struct_site.details": "SITE_DESCRIPTION",
    "_struct_site.pdbx_evidence_code": "EVIDENCE_CODE",
}

# Dictionary linking DSSP mmCIF secondary structure codes to an index and more
# human-readable format
DSSP_SS_TO_HUMAN = {
    "HELX_RH_AL_P": ("Alpha Helix", 0),
    "STRN": ("Strand", 1),
    "HELX_RH_3T_P": ("Helix 3-10", 2),
    "HELX_RH_PI_P": ("Pi Helix", 3),
    "HELX_LH_PP_P": ("Polyproline Helix", 4),
    "TURN_TY1_P": ("Turn", 5),
    "BEND": ("Bend", 6),
    "OTHER": ("Loop", 7),
}