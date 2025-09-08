"""Defines the tables that will be pulled from the PDBx/mmCIF file. Each entry in
the below "ENTRIES" tuples is a tuple of the form (key, flag), where the key gives
the field name of the table we wish to pull and the flag indicates whether the
field is required for the program to run (2), required for the associated table to
be loaded (1), or optional (0).
"""

# Keys of the citation information we wish to gather
CITATION_ID = "_citation.id"
CITATION_ENTRIES = (
    (CITATION_ID, 1),
    ("_citation.abstract", 0),
    ("_citation.title", 1),
    ("_citation.pdbx_database_id_pubmed", 0),
    ("_citation.pdbx_database_id_doi", 0),
    ("_citation.journal_id_astm", 0),
    ("_citation.journal_id_issn", 0),
    ("_citation.book_id_isbn", 0),
)

# Keys of the entity information we wish to gather
ENTITY_ID = "_entity.id"
ENTITY_ENTRIES = (
    (ENTITY_ID, 2),
    ("_entity.type", 2),
    ("_entity.pdbx_number_of_molecules", 2),
    ("_entity.pdbx_description", 0),
    ("_entity.pdbx_ec", 0),
    ("_entity.details", 0),
)

# Keys of the entity_poly information we wish to gather
ENTITY_POLY_ID = "_entity_poly.entity_id"
ENTITY_POLY_ENTRIES = (
    (ENTITY_POLY_ID, 1),
    ("_entity_poly.type", 1),
    ("_entity_poly.nstd_linkage", 0),
    ("_entity_poly.nstd_monomer", 0),
    ("_entity_poly.pdbx_seq_one_letter_code", 1),
    ("_entity_poly.pdbx_seq_one_letter_code_can", 1),
    ("_entity_poly.pdbx_strand_id", 1),
)

# Keys of the entity_poly_seq information we wish to gather
ENTITY_POLY_SEQ_ID = "_entity_poly_seq.entity_id"
ENTITY_POLY_SEQ_ENTRIES = (
    (ENTITY_POLY_SEQ_ID, 1),
    ("_entity_poly_seq.num", 1),
    ("_entity_poly_seq.mon_id", 1),
    ("_entity_poly_seq.hetero", 0),
)

# Keys of the entity_src_gen information we wish to gather
ENTITY_SRC_GEN_ID = "_entity_src_gen.entity_id"
ENTITY_SRC_GEN_ENTRIES = (
    (ENTITY_SRC_GEN_ID, 1),
    ("_entity_src_gen.pdbx_src_id", 1),
    ("_entity_src_gen.pdbx_gene_src_ncbi_taxonomy_id", 0),
    ("_entity_src_gen.pdbx_gene_src_scientific_name", 0),
    ("_entity_src_gen.pdbx_gene_src_common_name", 0),
    ("_entity_src_gen.pdbx_gene_src_organ", 0),
    ("_entity_src_gen.gene_src_tissue", 0),
    ("_entity_src_gen.pdbx_gene_src_organelle", 0),
    ("_entity_src_gen.pdbx_gene_src_cellular_location", 0),
    ("_entity_src_gen.pdbx_gene_src_cell", 0),
    ("_entity_src_gen.pdbx_host_org_scientific_name", 0),
    ("_entity_src_gen.pdbx_host_org_common_name", 0),
    ("_entity_src_gen.pdbx_host_org_organ", 0),
    ("_entity_src_gen.pdbx_host_org_tissue", 0),
    ("_entity_src_gen.pdbx_host_org_organelle", 0),
    ("_entity_src_gen.pdbx_host_org_cellular_location", 0),
    ("_entity_src_gen.pdbx_host_org_cell", 0),
    ("_entity_src_gen.pdbx_description", 0),
)

# Keys of the entity_src_nat information we wish to gather
ENTITY_SRC_NAT_ID = "_entity_src_nat.entity_id"
ENTITY_SRC_NAT_ENTRIES = (
    (ENTITY_SRC_NAT_ID, 1),
    ("_entity_src_nat.pdbx_src_id", 1),
    ("_entity_src_nat.pdbx_organism_scientific", 0),
    ("_entity_src_nat.common_name", 0),
    ("_entity_src_nat.pdbx_ncbi_taxonomy_id", 0),
    ("_entity_src_nat.pdbx_organ", 0),
    ("_entity_src_nat.tissue", 0),
    ("_entity_src_nat.pdbx_cell", 0),
    ("_entity_src_nat.pdbx_organelle", 0),
    ("_entity_src_nat.pdbx_cellular_location", 0),
    ("_entity_src_nat.details", 0),
)

# Keys of the entity_src_syn information we wish to gather
ENTITY_SRC_SYN_ID = "_pdbx_entity_src_syn.entity_id"
ENTITY_SRC_SYN_ENTRIES = (
    (ENTITY_SRC_SYN_ID, 1),
    ("_pdbx_entity_src_syn.organism_scientific", 0),
    ("_pdbx_entity_src_syn.organism_common_name", 0),
    ("_pdbx_entity_src_syn.ncbi_taxonomy_id", 0),
    ("_pdbx_entity_src_syn.details", 0),
)

# Keys of the _struct_site information we wish to gather
STRUCT_SITE_ID = "_struct_site.id"
STRUCT_SITE_ENTRIES = (
    (STRUCT_SITE_ID, 1),
    ("_struct_site.pdbx_evidence_code", 0),
    ("_struct_site.details", 0),
)

# Keys of the _struct_site_gen information we wish to gather
STRUCT_SITE_GEN_ID = "_struct_site_gen.id"
STRUCT_SITE_GEN_ENTRIES = (
    (STRUCT_SITE_GEN_ID, 1),
    ("_struct_site_gen.site_id", 1),
    ("_struct_site_gen.label_comp_id", 1),
    ("_struct_site_gen.label_asym_id", 1),
    ("_struct_site_gen.label_seq_id", 1),
    ("_struct_site_gen.label_alt_id", 0),
    ("_struct_site_gen.details", 0),
)

# Keys of information on _struct_asysm information we wish to gather
STRUCT_ASYMM_ID = "_struct_asym.id"
STRUCT_ASYMM_ENTRIES = (
    (STRUCT_ASYMM_ID, 2),
    ("_struct_asym.entity_id", 2),
    ("_struct_asym.details", 0),
)

# Keys of information on _atom_site information we wish to gather
ATOM_SITE_ID = "_atom_site.id"
ATOM_SITE_ENTRIES = (
    (ATOM_SITE_ID, 2),
    ("_atom_site.group_pdb", 2),
    ("_atom_site.type_symbol", 2),
    ("_atom_site.label_atom_id", 2),
    ("_atom_site.label_alt_id", 2),
    ("_atom_site.label_comp_id", 2),
    ("_atom_site.label_asym_id", 2),
    ("_atom_site.label_entity_id", 2),
    ("_atom_site.label_seq_id", 2),
    ("_atom_site.pdbx_pdb_ins_code", 2),
    ("_atom_site.cartn_x", 2),
    ("_atom_site.cartn_y", 2),
    ("_atom_site.cartn_z", 2),
    ("_atom_site.occupancy", 2),
    ("_atom_site.b_iso_or_equiv", 2),
    ("_atom_site.pdbx_formal_charge", 2),
    ("_atom_site.auth_seq_id", 2),
    ("_atom_site.auth_comp_id", 2),
    ("_atom_site.auth_asym_id", 2),
    ("_atom_site.auth_atom_id", 2),
    ("_atom_site.pdbx_pdb_model_num", 2),
)

# Keys of information on _pdbx_poly_seq_scheme information we wish to gather
PDBX_POLY_SEQ_SCHEME_ENTRIES = (
    ("_pdbx_poly_seq_scheme.asym_id", 1),
    ("_pdbx_poly_seq_scheme.entity_id", 1),
    ("_pdbx_poly_seq_scheme.seq_id", 1),
    ("_pdbx_poly_seq_scheme.mon_id", 1),
    ("_pdbx_poly_seq_scheme.pdb_seq_num", 1),
    ("_pdbx_poly_seq_scheme.pdb_strand_id", 1),
    ("_pdbx_poly_seq_scheme.pdb_ins_code", 1),
    ("_pdbx_poly_seq_scheme.auth_seq_num", 1),
)

# Keys of information on _atom_site_anisotrop that we wish to gather
ATOM_SITE_ANISOTROP_ENTRIES = (
    ("_atom_site_anisotrop.id", 1),
    ("_atom_site_anisotrop.pdbx_label_comp_id", 1),
    ("_atom_site_anisotrop.u[1][1]", 1),
    ("_atom_site_anisotrop.u[2][2]", 1),
    ("_atom_site_anisotrop.u[3][3]", 1),
    ("_atom_site_anisotrop.u[1][2]", 1),
    ("_atom_site_anisotrop.u[1][3]", 1),
    ("_atom_site_anisotrop.u[2][3]", 1),
)

# Keys of information on _chem_comp that we wish to gather
CHEM_COMP_ID = "_chem_comp.id"
CHEM_COMP_ENTRIES = (
    (CHEM_COMP_ID, 2),
    ("_chem_comp.formula", 2),
    ("_chem_comp.formula_weight", 2),
    ("_chem_comp.name", 2),
    ("_chem_comp.pdbx_synonyms", 0),
)

# Keys of information on the _struct_conn that we wish to gather
STRUCT_CONN_ENTRIES = (
    ("_struct_conn.conn_type_id", 1),
    ("_struct_conn.ptnr1_label_asym_id", 1),
    ("_struct_conn.ptnr1_label_comp_id", 1),
    ("_struct_conn.ptnr1_label_seq_id", 1),
    ("_struct_conn.ptnr2_label_asym_id", 0),
    ("_struct_conn.ptnr2_label_comp_id", 0),
    ("_struct_conn.ptnr2_label_seq_id", 0),
    ("_struct_conn.details", 0),
)

# Keys of information on the _struct_ref that we wish to gather
STRUCT_REF_ID = "_struct_ref.id"
STRUCT_REF_ENTRIES = (
    (STRUCT_REF_ID, 1),
    ("_struct_ref.db_name", 1),
    ("_struct_ref.db_code", 0),
    ("_struct_ref.pdbx_db_accession", 0),
    ("_struct_ref.pdbx_db_isoform", 0),
    ("_struct_ref.entity_id", 1),
    ("_struct_ref.pdbx_seq_one_letter_code", 1),
)

# Keys of information on _modres that we wish to gather
MODRES_ID = "_pdbx_struct_mod_residue.id"
MODRES_ENTRIES = (
    (MODRES_ID, 1),
    ("_pdbx_struct_mod_residue.label_asym_id", 1),
    ("_pdbx_struct_mod_residue.label_comp_id", 1),
    ("_pdbx_struct_mod_residue.label_seq_id", 1),
    ("_pdbx_struct_mod_residue.details", 1),
)

DSSP_STRUCT_LADDER_ENTRIES = (
    ("_dssp_struct_ladder.id", 1),
    ("_dssp_struct_ladder.sheet_id", 1),
    ("_dssp_struct_ladder.type", 1),
    ("_dssp_struct_ladder.beg_1_label_comp_id", 1),
    ("_dssp_struct_ladder.beg_1_label_asym_id", 1),
    ("_dssp_struct_ladder.beg_1_label_seq_id", 1),
    ("_dssp_struct_ladder.end_1_label_comp_id", 1),
    ("_dssp_struct_ladder.end_1_label_asym_id", 1),
    ("_dssp_struct_ladder.end_1_label_seq_id", 1),
    ("_dssp_struct_ladder.beg_2_label_comp_id", 1),
    ("_dssp_struct_ladder.beg_2_label_asym_id", 1),
    ("_dssp_struct_ladder.beg_2_label_seq_id", 1),
    ("_dssp_struct_ladder.end_2_label_comp_id", 1),
    ("_dssp_struct_ladder.end_2_label_asym_id", 1),
    ("_dssp_struct_ladder.end_2_label_seq_id", 1),
)

DSSP_BRIDGE_PAIR_ENTRIES = (
    ("_dssp_struct_bridge_pairs.id", 1),
    ("_dssp_struct_bridge_pairs.label_comp_id", 1),
    ("_dssp_struct_bridge_pairs.label_seq_id", 1),
    ("_dssp_struct_bridge_pairs.label_asym_id", 1),
    ("_dssp_struct_bridge_pairs.acceptor_1_label_comp_id", 1),
    ("_dssp_struct_bridge_pairs.acceptor_1_label_seq_id", 1),
    ("_dssp_struct_bridge_pairs.acceptor_1_label_asym_id", 1),
    ("_dssp_struct_bridge_pairs.acceptor_1_energy", 1),
    ("_dssp_struct_bridge_pairs.acceptor_2_label_comp_id", 1),
    ("_dssp_struct_bridge_pairs.acceptor_2_label_seq_id", 1),
    ("_dssp_struct_bridge_pairs.acceptor_2_label_asym_id", 1),
    ("_dssp_struct_bridge_pairs.acceptor_2_energy", 1),
    ("_dssp_struct_bridge_pairs.donor_1_label_comp_id", 1),
    ("_dssp_struct_bridge_pairs.donor_1_label_seq_id", 1),
    ("_dssp_struct_bridge_pairs.donor_1_label_asym_id", 1),
    ("_dssp_struct_bridge_pairs.donor_1_energy", 1),
    ("_dssp_struct_bridge_pairs.donor_2_label_comp_id", 1),
    ("_dssp_struct_bridge_pairs.donor_2_label_seq_id", 1),
    ("_dssp_struct_bridge_pairs.donor_2_label_asym_id", 1),
    ("_dssp_struct_bridge_pairs.donor_2_energy", 1),
)

DSSP_SS_ENTRIES = (
    ("_struct_conf.conf_type_id", 1),
    ("_struct_conf.id", 1),
    ("_struct_conf.beg_label_comp_id", 1),
    ("_struct_conf.beg_label_asym_id", 1),
    ("_struct_conf.beg_label_seq_id", 1),
    ("_struct_conf.end_label_comp_id", 1),
    ("_struct_conf.end_label_asym_id", 1),
    ("_struct_conf.end_label_seq_id", 1),
    ("_struct_conf.details", 1),
)