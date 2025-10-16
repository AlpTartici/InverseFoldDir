"""
Holds code for parsing DSSP-annotated mmCIF files and extracting information
from them.
"""

import itertools
import logging
import os.path
from typing import Dict, Generator, Optional, Sequence, Set, Tuple

import Levenshtein as ls
import numpy as np
import numpy.typing as npt
import pandas as pd
from Bio.PDB import MMCIF2Dict, MMCIFParser
from Bio.PDB.Residue import DisorderedResidue
from cif_file_errors import CATHError, NoPolypeptidesError
from cif_file_globals import MPNN_TARGET_ATOMS
from cif_file_utilities import _missing_to_empty_str, _process_citation_row, _select_opt
from custom_types import (
    ChainMetadata,
    CitationInfoOutput,
    CompoundInfo,
    ConnectionInfo,
    DSSPLadderInfo,
    DSSPSecondaryStructure,
    DSSPSheetInfo,
    ImportantSiteInfo,
    ModresOutput,
    ResidueData,
    ScrapedChainData,
    StructureData,
    StructureMetadata,
)
from tables import (
    ATOM_SITE_ANISOTROP_ENTRIES,
    ATOM_SITE_ENTRIES,
    CHEM_COMP_ENTRIES,
    CITATION_ENTRIES,
    DSSP_BRIDGE_PAIR_ENTRIES,
    DSSP_SS_ENTRIES,
    DSSP_STRUCT_LADDER_ENTRIES,
    ENTITY_ENTRIES,
    ENTITY_POLY_ENTRIES,
    ENTITY_SRC_GEN_ENTRIES,
    ENTITY_SRC_NAT_ENTRIES,
    ENTITY_SRC_SYN_ENTRIES,
    MODRES_ENTRIES,
    PDBX_POLY_SEQ_SCHEME_ENTRIES,
    STRUCT_CONN_ENTRIES,
    STRUCT_REF_ENTRIES,
    STRUCT_SITE_ENTRIES,
    STRUCT_SITE_GEN_ENTRIES,
)

from data.bruce_cif.aa_info import AA3_TO_1_CANONICAL, AA3_TO_1_FULL
from data.bruce_cif.cath_record import CATHRecord
from data.bruce_cif.chain_filters import _filter_all

# pylint: disable=too-many-lines


class CIFFile:
    """The primary class for processing CIF files that have not been annotated
    with DSSP.
    """

    def __init__(self, cif_file: str, target_atoms: Sequence[str] = MPNN_TARGET_ATOMS):
        """Loads the cif file and parses it.

        Args:
            cif_file (str): The path to the cif file.
            target_atoms (Sequence[str], optional): The atoms whose data we wish
                to extract when scraping. Defaults to MPNN_TARGET_ATOMS.
        """
        # Note the target atoms
        self.target_atoms = target_atoms

        # Instance variable noting whether the CIF file contains multiletter amino
        # acid codes
        self.found_multiletter = False

        # Parse the CIF file
        self.cif_file = cif_file
        self.mmcif_dict = MMCIF2Dict.MMCIF2Dict(cif_file)
        self.structure = MMCIFParser(QUIET=True).get_structure(
            os.path.splitext(os.path.basename(cif_file))[0], cif_file
        )

        # All keys in the mmcif_dict are converted to lowercase
        self.mmcif_dict = {key.lower(): value for key, value in self.mmcif_dict.items()}

        # Assign information about the file
        self.pdb_id = self._assign_singleton("_struct.entry_id").lower()
        self.title = self._assign_singleton("_struct.title")
        self.exptl_method = ";".join(
            el.replace("\n", "") for el in self.mmcif_dict["_exptl.method"]
        )
        self.description = self._assign_singleton("_struct.pdbx_descriptor", "")
        self.keywords = self._assign_singleton("_struct_keywords.text", "")

        # Get dataframes for each section of the CIF file
        self.citation_data = self._compile_data(CITATION_ENTRIES)
        self.entity_data = self._compile_data(ENTITY_ENTRIES)
        self.struct_site_data = self._compile_data(STRUCT_SITE_ENTRIES)
        self.struct_site_gen_data = self._compile_data(
            STRUCT_SITE_GEN_ENTRIES
        ).drop_duplicates()
        self.atom_data = self._compile_data(ATOM_SITE_ENTRIES)
        self.entity_src_gen = self._compile_data(ENTITY_SRC_GEN_ENTRIES)
        self.entity_src_nat = self._compile_data(ENTITY_SRC_NAT_ENTRIES)
        self.entity_src_syn = self._compile_data(ENTITY_SRC_SYN_ENTRIES)
        self.atom_site_anisotrop_data = self._compile_data(ATOM_SITE_ANISOTROP_ENTRIES)
        self.chem_comp_data = self._compile_data(CHEM_COMP_ENTRIES)
        self.struct_conn_data = self._compile_data(
            STRUCT_CONN_ENTRIES
        ).drop_duplicates()
        self.entity_poly_data = self._compile_data(ENTITY_POLY_ENTRIES)
        self.pdbx_poly_seq_scheme_data = self._compile_data(
            PDBX_POLY_SEQ_SCHEME_ENTRIES
        )
        self.struct_ref_data = self._compile_data(STRUCT_REF_ENTRIES)
        self.modres_data = self._compile_data(MODRES_ENTRIES)

        # Extract dssp tables
        self.dssp_struct_ladder_data = self._compile_data(DSSP_STRUCT_LADDER_ENTRIES)
        self.dssp_bridge_pair_data = self._compile_data(DSSP_BRIDGE_PAIR_ENTRIES)
        self.dssp_secondary_structure_data = self._compile_data(
            DSSP_SS_ENTRIES
        ).drop_duplicates()

        # Merge the atom and anisotrop data
        self._merge_atom_and_anisotrop()

        # Build a dictionary that maps from model number to an index
        self.model_num_to_ind = self._build_model_num_to_ind()

        # Filter the poly seq scheme data to only include polypeptides and note
        # the entity IDs that correspond to polypeptides
        self.polymeric_entity_ids = self._filter_poly_seq_scheme_atom_data()

        # Take the first disordered residue only from the poly seq scheme data
        self.all_kept_ids = self._remove_disordered_residues()

        # Remove disordered residues from other dataframes
        self._process_struct_site_gen_data()
        self._process_struct_conn_data()
        self._process_modres_data()
        self._process_bridge_pairs1()
        self._process_secondary_structure1()
        self._process_sheet_ladder1()

        # Assign the one letter code to `self.pdbx_poly_seq_scheme_data`
        self._assign_one_letter_code()

        # Build a dictionary that maps asym ID to PDB chain ID
        (
            self.asym_id_to_chain_id,
            self.chain_id_to_asym_id,
        ) = self._build_asym_id_to_chain_id()

        # Build a dictionary that maps from asym ID to an integer
        self.asym_id_to_int = self._build_asym_to_ind()

        # Build a dictionary that maps from entity ID to a tuple of integers
        # representing the asym IDs. Build a dictionary that maps from chain ind
        # to entity ID as well
        (
            self.entity_to_asym_int,
            self.asym_int_to_entity,
        ) = self._build_entity_to_asym_ind()

        # Add sequence indices, chain indices, and CATH indices to the poly seq
        # scheme data
        self._index_poly_seq_scheme()

        # Get a dataframe with both the atom and peptide data
        self.peptides_with_atoms = self._build_peptides_with_atoms()

        # We need a dictionary that maps from chemical compound to chemical compound
        # information
        self.chem_comp_dict = self._build_chem_comp_dict()

        # We need a dictionary that maps from asym and seq IDs to indices in the
        # output sequence data
        self.asym_seq_to_seq_ind = self._build_asym_seq_to_seq_ind()

        # We need a dictionary that maps from asym and seq IDs to indices in the
        # output sequence data
        self.asym_seq_to_comp = self._build_asym_seq_to_resi()

        # Remove data from DSSP for chains that are not polypeptides
        self._process_bridge_pairs2()
        self._process_secondary_structure2()
        self._process_sheet_ladder2()

    def _assign_singleton(self, key: str, default: Optional[str] = None) -> str:
        """Makes sure that the key (1) is in the instance variable `self.mmcif_dict`
        and (2) has an associated value with length 1. If the key is not in the
        dictionary and `default` is not provided, then a KeyError is raised. If
        the key is not in the dictionary and `default` IS provided, then the value
        of `default` is returned. If the key is in the dictionary, then the single
        value in the sequence associated with that key is returned. This function
        is used in parsing the CIF file when we expect a single value for a given
        key.

        Args:
            key (str): The key to look up in the dictionary.
            default (Optional[str], optional): The default value to return if the
                key is not in the dictionary. Defaults to None, which means that
                a KeyError will be raised if the key is not in the dictionary.

        Returns:
            str: The value associated with the key or, if the key is not in the
                self.mmcif_dict and default is provided, the value of default.

        """
        # If default is not provided, then the key must be present
        if key not in self.mmcif_dict:
            if default is None:
                raise KeyError(f"Missing required entry: {key}")
            val = default
        else:
            val = self.mmcif_dict[key]
            assert len(val) == 1
            val = val[0]

        return val.replace("\n", "")

    def _compile_data(
        self, initial_entries: Tuple[Tuple[str, int], ...]
    ) -> pd.DataFrame:
        """Gathers data from a single CIF file table and compiles it into a dataframe.

        Args:
            initial_entries (Tuple[Tuple[str, int], ...]): Defines the tables that
                will be pulled from the PDBx/mmCIF file. Each entry in `initial_entries`
                is a tuple of the form (key, flag), where the key gives the field
                name in the table we wish to pull and the flag indicates whether
                the field is required for the program to run (2), required for the
                associated table to be loaded (1), or optional (0). See the structure
                of the variables defined in `tables.py` for examples of this. Note
                that if a field is required for the program to run, then a KeyError
                will be raised if that field is not present in the CIF file. If
                a field is required for the associated table to be loaded, then
                an empty dataframe will be returned if that field is not present.
                If a field is optional, then that field will still be present in
                the returned dataframe, but the values will be NaN.

        Returns:
            pd.DataFrame: A dataframe that contains the requested fields from the
                CIF file.
        """
        # Make sure required entries are present in the dictionary. If they are
        # missing, we cannot record the data and we return an empty dataframe for
        # low priority data and raise an error for high priority data
        missing_entries = [
            (entry, required)
            for entry, required in initial_entries
            if required > 0 and entry not in self.mmcif_dict
        ]
        if len(missing_entries) > 0:
            missing_entries, missing_required = zip(*missing_entries)
            if any(req == 2 for req in missing_required):
                raise KeyError(
                    f"Missing required entries: {', '.join(missing_entries)}"
                )
            else:
                return pd.DataFrame(columns=[entry for entry, _ in initial_entries])

        # Separate out entries and whether they are required
        entries, _ = zip(*initial_entries)

        # Pull entries that are in the dictionary and replace newlines with empty
        # strings
        data = {
            entry: [el.replace("\n", "") for el in self.mmcif_dict[entry]]
            for entry in entries
            if entry in self.mmcif_dict
        }

        # Make sure all entries are the same length
        entry_len = len(list(data.values())[0])
        assert all(len(entry_data) == entry_len for entry_data in data.values())

        # Add data for entries that are not in the dictionary
        data.update(
            {
                entry: [None] * entry_len
                for entry in entries
                if entry not in self.mmcif_dict
            }
        )

        # Build a dataframe from the data
        return pd.DataFrame(data)

    def _merge_atom_and_anisotrop(self) -> None:
        """Data on atomic coordinates (`_atom_site`) and anisotropic B factors
        (`_atom_site_anisotrop`) are kept in separate tables in the CIF file. This
        function merges the two tables together so that all data on a given atom
        can be accessed from a single row in the atom dataframe. The merge keys
        are `_atom_site.id` and `_atom_site_anisotrop.id`. `_atom_site_anisotrop.id`
        points directly to `_atom_site.id`, so the merge should be one to one.
        """
        # If there is no anisotrop data, then we add a series of NaNs to the atom
        # dataframe and return
        if len(self.atom_site_anisotrop_data) == 0:
            for colname, _ in ATOM_SITE_ANISOTROP_ENTRIES:
                self.atom_data[colname] = np.nan
            return

        # Otherwise, we merge the anisotropy dataframe onto the atom dataframe.
        # To begin, both dataframes must have nonzero length
        assert len(self.atom_site_anisotrop_data) > 0
        assert len(self.atom_data) > 0

        # All atom IDs should be unique in both dataframes
        assert len(
            self.atom_site_anisotrop_data["_atom_site_anisotrop.id"].unique()
        ) == len(self.atom_site_anisotrop_data)
        assert len(self.atom_data["_atom_site.id"].unique()) == len(self.atom_data)

        # Merge the dataframes, making sure that order is preserved
        original_id_order = self.atom_data["_atom_site.id"].tolist()
        self.atom_data = self.atom_data.merge(
            self.atom_site_anisotrop_data,
            how="left",
            left_on="_atom_site.id",
            right_on="_atom_site_anisotrop.id",
            validate="1:1",
        )
        assert original_id_order == self.atom_data["_atom_site.id"].tolist()

        # Wherever the anisotropy data is not NaN, we make sure that the amino acids
        # match
        nonna_df = self.atom_data[self.atom_data["_atom_site_anisotrop.id"].notna()]
        assert len(nonna_df) > 0
        assert (
            nonna_df["_atom_site_anisotrop.pdbx_label_comp_id"].tolist()
            == nonna_df["_atom_site.label_comp_id"].tolist()
        )

    def _build_model_num_to_ind(self) -> Dict[str, int]:
        """Builds a mapping from `_atom_site.pdbx_pdb_model_num` to an index. This
        is useful for converting between model numbers in mmCIF files and the indexing
        scheme used to distinguish models in the BioPython structure object.

        Returns:
            Dict[str, int]: A dictionary linking model numbers to indices.
        """

        return {
            model_num: ind
            for ind, model_num in enumerate(
                self.atom_data["_atom_site.pdbx_pdb_model_num"].unique()
            )
        }

    def _filter_poly_seq_scheme_atom_data(self) -> Set[str]:
        """Removes any rows from `self.pdbx_poly_seq_scheme_data` and `self.atom_data`
        that do not correspond to polypeptides.

        Returns:
            Set[str]: The set of entity IDs that correspond to polypeptides.

        Notes:
            This function identifies polypeptides by looking for "_entity_poly.entity_id"
            fields that are associated with an "_entity_poly.type" field that has
            the value "polypeptide(L)". "_entity_poly.entity_id" points to "_entity.id".
            The returned entity IDs are then used to filter out rows from "self.atom_data"
            and "self.pdbx_poly_seq_scheme_data" that do not correspond to polypeptides.
            This is done by taking only those rows in "self.atom_data" that have
            an "_atom_site.label_entity_id" that is in the set of entity IDs that
            correspond to polypeptides. Similarly, this is done by taking only those
            rows in "self.pdbx_poly_seq_scheme_data" that have a "_pdbx_poly_seq_scheme.entity_id"
            that is in the set of entity IDs that correspond to polypeptides. The
            fields "_pdbx_poly_seq_scheme.entity_id" and "_atom_site.label_entity_id"
            both point to "_entity.id", just like "_entity_poly.entity_id".
        """

        # Identify entities containing polypeptides
        poly_entities = set(
            self.entity_poly_data.loc[
                self.entity_poly_data["_entity_poly.type"] == "polypeptide(L)",
                "_entity_poly.entity_id",
            ].tolist()
        )

        # Filter out entities that are not polypeptides
        self.pdbx_poly_seq_scheme_data = self.pdbx_poly_seq_scheme_data[
            self.pdbx_poly_seq_scheme_data["_pdbx_poly_seq_scheme.entity_id"].isin(
                poly_entities
            )
        ]
        self.atom_data = self.atom_data[
            self.atom_data["_atom_site.label_entity_id"].isin(poly_entities)
        ]

        # If there are no polypeptides, then we raise an error
        if not self.contains_polypeptides:
            raise NoPolypeptidesError(f"No polypeptides found in {self.pdb_id}")

        return poly_entities

    def _remove_disordered_residues(self) -> pd.DataFrame:
        """Takes the first occurrence of each seqind in the peptide data for each
        chain. This is the simplest way of handling disordered residues. It also
        appears to be how DSSP handles disordered residues.

        Returns
            pd.DataFrame: A dataframe with the indices of the kept residues.
        """
        # Remove duplicate asym ids and sequence indices
        self.pdbx_poly_seq_scheme_data.drop_duplicates(
            subset=["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.seq_id"],
            inplace=True,
        )

        return (
            self.pdbx_poly_seq_scheme_data[
                [
                    "_pdbx_poly_seq_scheme.asym_id",
                    "_pdbx_poly_seq_scheme.seq_id",
                    "_pdbx_poly_seq_scheme.mon_id",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def _process_struct_site_gen_data(self) -> None:
        """Removes any rows from `self.struct_site_gen_data` that do not correspond
        to elements kept after removing disordered residues
        """
        # Note that _struct_site_gen.label_comp_id maps to _atom_site.label_comp_id,
        # which points to _chem_comp.id; _pdbx_poly_seq_scheme.mon_id points to
        # _entity_poly_seq.mon_id which in turn points to `_chem_comp.id`;
        # _struct_site_gen.label_asym_id maps to _atom_site.label_asym_id, which
        # _pdbx_poly_seq_scheme.asym_id points to as well; and _struct_site_gen.label_seq_id
        # maps to _atom_site.label_seq_id, which points to `_entity_poly_seq.num`
        # along with `_pdbx_poly_seq_scheme.seq_id`.
        merge_test = (
            self.struct_site_gen_data.merge(
                self.all_kept_ids,
                how="left",
                left_on=[
                    "_struct_site_gen.label_asym_id",
                    "_struct_site_gen.label_seq_id",
                    "_struct_site_gen.label_comp_id",
                ],
                right_on=[
                    "_pdbx_poly_seq_scheme.asym_id",
                    "_pdbx_poly_seq_scheme.seq_id",
                    "_pdbx_poly_seq_scheme.mon_id",
                ],
                validate="m:1",
            )["_pdbx_poly_seq_scheme.asym_id"]
            .notna()
            .to_numpy()
        )

        # Filter out the rows that don't have a corresponding entry in _pdbx_poly_seq_scheme
        self.struct_site_gen_data = self.struct_site_gen_data.loc[merge_test]

    def _process_struct_conn_data(self) -> None:
        """Removes any rows from `self.struct_conn_data` that do not correspond
        to elements kept after removing disordered residues
        """
        # Note that `_struct_conn.ptnr#_label_comp_id` points to `_atom_site.label_comp_id`
        # which points to _chem_comp.id; _pdbx_poly_seq_scheme.mon_id points to
        # _entity_poly_seq.mon_id which in turn points to `_chem_comp.id`;
        # `_struct_conn.ptnr#_label_asym_id` points to `_atom_site.label_asym_id`,
        # to which `_pdbx_poly_seq_scheme.asym_id` also points; and
        # `_struct_conn.ptnr#_label_seq_id` points to `_atom_site.label_seq_id`,
        # which points to `_entity_poly_seq.num` along with
        # `_pdbx_poly_seq_scheme.seq_id`
        merge_test = self.struct_conn_data.merge(
            self.all_kept_ids,
            how="left",
            left_on=[
                "_struct_conn.ptnr1_label_asym_id",
                "_struct_conn.ptnr1_label_seq_id",
                "_struct_conn.ptnr1_label_comp_id",
            ],
            right_on=[
                "_pdbx_poly_seq_scheme.asym_id",
                "_pdbx_poly_seq_scheme.seq_id",
                "_pdbx_poly_seq_scheme.mon_id",
            ],
            validate="m:1",
        ).merge(
            self.all_kept_ids,
            how="left",
            left_on=[
                "_struct_conn.ptnr2_label_asym_id",
                "_struct_conn.ptnr2_label_seq_id",
                "_struct_conn.ptnr2_label_comp_id",
            ],
            right_on=[
                "_pdbx_poly_seq_scheme.asym_id",
                "_pdbx_poly_seq_scheme.seq_id",
                "_pdbx_poly_seq_scheme.mon_id",
            ],
            validate="m:1",
        )
        merge_test = (
            merge_test["_pdbx_poly_seq_scheme.asym_id_x"].notna()
            & merge_test["_pdbx_poly_seq_scheme.asym_id_y"].notna()
        ).to_numpy()

        # Filter out the rows that don't have a match
        self.struct_conn_data = self.struct_conn_data.loc[merge_test]

    def _process_modres_data(self) -> None:
        """Removes rows from `self.modres_data` that do not correspond to elements
        kept after removing disordered residues
        """
        # Note that _pdbx_struct_mod_residue.label_asym_id points to _atom_site.label_asym_id
        # to which `_pdbx_poly_seq_scheme.asym_id` also points;
        # _pdbx_struct_mod_residue.label_seq_id points to _atom_site.label_seq_id
        # which points to `_entity_poly_seq.num` along with`_pdbx_poly_seq_scheme.seq_id`;
        # _pdbx_struct_mod_residue.label_comp_id points to _atom_site.label_comp_id
        # which points to _chem_comp.id; _pdbx_poly_seq_scheme.mon_id points to
        # _entity_poly_seq.mon_id which in turn points to `_chem_comp.id`
        merge_test = (
            self.modres_data.merge(
                self.all_kept_ids,
                how="left",
                left_on=[
                    "_pdbx_struct_mod_residue.label_asym_id",
                    "_pdbx_struct_mod_residue.label_seq_id",
                    "_pdbx_struct_mod_residue.label_comp_id",
                ],
                right_on=[
                    "_pdbx_poly_seq_scheme.asym_id",
                    "_pdbx_poly_seq_scheme.seq_id",
                    "_pdbx_poly_seq_scheme.mon_id",
                ],
                validate="m:1",
            )["_pdbx_poly_seq_scheme.asym_id"]
            .notna()
            .to_numpy()
        )

        # Filter out the rows that don't have a match
        self.modres_data = self.modres_data.loc[merge_test]

    def _process_bridge_pairs1(self) -> None:
        """Removes rows from `self.dssp_bridge_pair_data` that do not correspond
        to elements kept after removing disordered residues
        """
        # We eliminate rows where the reference is no longer present after removing
        # disordered residues
        merge_test = (
            self.dssp_bridge_pair_data.merge(
                self.all_kept_ids,
                how="left",
                left_on=[
                    "_dssp_struct_bridge_pairs.label_asym_id",
                    "_dssp_struct_bridge_pairs.label_seq_id",
                    "_dssp_struct_bridge_pairs.label_comp_id",
                ],
                right_on=[
                    "_pdbx_poly_seq_scheme.asym_id",
                    "_pdbx_poly_seq_scheme.seq_id",
                    "_pdbx_poly_seq_scheme.mon_id",
                ],
                validate="m:1",
            )["_pdbx_poly_seq_scheme.asym_id"]
            .notna()
            .to_numpy()
        )
        self.dssp_bridge_pair_data = self.dssp_bridge_pair_data.loc[merge_test]

        # For each chain, convert the values to "?" if the partner is no longer
        # present after removing disordered residues
        masks = []
        for pair_ind in (1, 2):
            for key_type in ("acceptor", "donor"):
                # Get the names of the asym id, seq id and comp id columns
                comp_id_col = (
                    f"_dssp_struct_bridge_pairs.{key_type}_{pair_ind}_label_comp_id"
                )
                seq_id_col = (
                    f"_dssp_struct_bridge_pairs.{key_type}_{pair_ind}_label_seq_id"
                )
                asym_id_col = (
                    f"_dssp_struct_bridge_pairs.{key_type}_{pair_ind}_label_asym_id"
                )

                # Determine when partner chains are still present
                merge_test = (
                    self.dssp_bridge_pair_data.merge(
                        self.all_kept_ids,
                        how="left",
                        left_on=[asym_id_col, seq_id_col, comp_id_col],
                        right_on=[
                            "_pdbx_poly_seq_scheme.asym_id",
                            "_pdbx_poly_seq_scheme.seq_id",
                            "_pdbx_poly_seq_scheme.mon_id",
                        ],
                        validate="m:1",
                    )["_pdbx_poly_seq_scheme.asym_id"]
                    .notna()
                    .to_numpy()
                )

                # Convert the values to "?" if the partner is no longer present
                self.dssp_bridge_pair_data.loc[
                    ~merge_test, [comp_id_col, seq_id_col, asym_id_col]
                ] = "?"

                # Record the merge test
                masks.append(merge_test)

        # Only keep rows where at least one partner chain is still present
        self.dssp_bridge_pair_data = self.dssp_bridge_pair_data.loc[
            np.stack(masks).any(axis=0)
        ]

    def _process_secondary_structure1(self) -> None:
        """Eliminates rows from `self.dssp_secondary_structure_data` that do not
        correspond to elements kept after removing disordered residues.
        """

        # Both beginning and end comp ids must still be here after removing disordered
        # residues
        merge_test = self.dssp_secondary_structure_data.merge(
            self.all_kept_ids,
            how="left",
            left_on=[
                "_struct_conf.beg_label_asym_id",
                "_struct_conf.beg_label_seq_id",
                "_struct_conf.beg_label_comp_id",
            ],
            right_on=[
                "_pdbx_poly_seq_scheme.asym_id",
                "_pdbx_poly_seq_scheme.seq_id",
                "_pdbx_poly_seq_scheme.mon_id",
            ],
            validate="m:1",
        ).merge(
            self.all_kept_ids,
            how="left",
            left_on=[
                "_struct_conf.end_label_asym_id",
                "_struct_conf.end_label_seq_id",
                "_struct_conf.end_label_comp_id",
            ],
            right_on=[
                "_pdbx_poly_seq_scheme.asym_id",
                "_pdbx_poly_seq_scheme.seq_id",
                "_pdbx_poly_seq_scheme.mon_id",
            ],
            validate="m:1",
        )
        merge_test = (
            merge_test["_pdbx_poly_seq_scheme.asym_id_x"].notna()
            & merge_test["_pdbx_poly_seq_scheme.asym_id_y"].notna()
        ).to_numpy()

        # Filter out the rows that don't have a match
        self.dssp_secondary_structure_data = self.dssp_secondary_structure_data.loc[
            merge_test
        ]

    def _process_sheet_ladder1(self) -> None:
        """Removes rows from `self.dssp_struct_ladder_data` that do not correspond
        to elements kept after removing disordered residues
        """
        # Beginning and end comp ids must still be here for both strands to keep
        # the ladder
        merge_test = (
            self.dssp_struct_ladder_data.merge(
                self.all_kept_ids,
                how="left",
                left_on=[
                    "_dssp_struct_ladder.beg_1_label_asym_id",
                    "_dssp_struct_ladder.beg_1_label_seq_id",
                    "_dssp_struct_ladder.beg_1_label_comp_id",
                ],
                right_on=[
                    "_pdbx_poly_seq_scheme.asym_id",
                    "_pdbx_poly_seq_scheme.seq_id",
                    "_pdbx_poly_seq_scheme.mon_id",
                ],
                validate="m:1",
            )
            .merge(
                self.all_kept_ids,
                how="left",
                left_on=[
                    "_dssp_struct_ladder.end_1_label_asym_id",
                    "_dssp_struct_ladder.end_1_label_seq_id",
                    "_dssp_struct_ladder.end_1_label_comp_id",
                ],
                right_on=[
                    "_pdbx_poly_seq_scheme.asym_id",
                    "_pdbx_poly_seq_scheme.seq_id",
                    "_pdbx_poly_seq_scheme.mon_id",
                ],
                validate="m:1",
                suffixes=("_1", "_2"),
            )
            .merge(
                self.all_kept_ids,
                how="left",
                left_on=[
                    "_dssp_struct_ladder.beg_2_label_asym_id",
                    "_dssp_struct_ladder.beg_2_label_seq_id",
                    "_dssp_struct_ladder.beg_2_label_comp_id",
                ],
                right_on=[
                    "_pdbx_poly_seq_scheme.asym_id",
                    "_pdbx_poly_seq_scheme.seq_id",
                    "_pdbx_poly_seq_scheme.mon_id",
                ],
                validate="m:1",
            )
            .merge(
                self.all_kept_ids,
                how="left",
                left_on=[
                    "_dssp_struct_ladder.end_2_label_asym_id",
                    "_dssp_struct_ladder.end_2_label_seq_id",
                    "_dssp_struct_ladder.end_2_label_comp_id",
                ],
                right_on=[
                    "_pdbx_poly_seq_scheme.asym_id",
                    "_pdbx_poly_seq_scheme.seq_id",
                    "_pdbx_poly_seq_scheme.mon_id",
                ],
                validate="m:1",
                suffixes=("_3", "_4"),
            )
        )
        merge_test = (
            merge_test["_pdbx_poly_seq_scheme.asym_id_1"].notna()
            & merge_test["_pdbx_poly_seq_scheme.asym_id_2"].notna()
            & merge_test["_pdbx_poly_seq_scheme.asym_id_3"].notna()
            & merge_test["_pdbx_poly_seq_scheme.asym_id_4"].notna()
        ).to_numpy()

        # Filter out the rows that don't have a match
        self.dssp_struct_ladder_data = self.dssp_struct_ladder_data.loc[merge_test]

    def _process_bridge_pairs2(self) -> None:
        """Eliminates asym IDs that do not correspond to polypeptides from the
        bridge pair data.
        """
        # Only keep rows where the reference is a polypeptide
        self.dssp_bridge_pair_data = self.dssp_bridge_pair_data[
            self.dssp_bridge_pair_data["_dssp_struct_bridge_pairs.label_asym_id"].isin(
                self.asym_id_to_int
            )
        ]

        # For each partner chain, convert the values to "?" if the chain is not a polypeptide
        masks = []
        for pair_ind in (1, 2):
            for key_type in ("acceptor", "donor"):
                # Get the names of the asym id, seq id, comp id, and energy columns
                comp_id_col = (
                    f"_dssp_struct_bridge_pairs.{key_type}_{pair_ind}_label_comp_id"
                )
                seq_id_col = (
                    f"_dssp_struct_bridge_pairs.{key_type}_{pair_ind}_label_seq_id"
                )
                asym_id_col = (
                    f"_dssp_struct_bridge_pairs.{key_type}_{pair_ind}_label_asym_id"
                )
                energy_col = f"_dssp_struct_bridge_pairs.{key_type}_{pair_ind}_energy"

                # Determine when partner chains are polypeptides
                mask = self.dssp_bridge_pair_data[asym_id_col].isin(self.asym_id_to_int)
                masks.append(mask.to_numpy())

                # Convert the values to "?" if the chain is not a polypeptide
                self.dssp_bridge_pair_data.loc[
                    ~mask, [comp_id_col, seq_id_col, asym_id_col, energy_col]
                ] = "?"

        # Only keep rows where at least one partner chain is a polypeptides
        self.dssp_bridge_pair_data = self.dssp_bridge_pair_data.loc[
            np.stack(masks).any(axis=0)
        ]

    def _process_secondary_structure2(self) -> None:
        """Eliminates asym IDs that do not correspond to polypeptides from the
        secondary structure data. Also eliminates rows where the beginning or
        end element are no longer present after removing disordered residues.
        """
        # Both beginning and end comp ids must be polypeptides
        self.dssp_secondary_structure_data = self.dssp_secondary_structure_data[
            (
                self.dssp_secondary_structure_data[
                    "_struct_conf.beg_label_asym_id"
                ].isin(self.asym_id_to_chain_id)
            )
            & (
                self.dssp_secondary_structure_data[
                    "_struct_conf.end_label_asym_id"
                ].isin(self.asym_id_to_chain_id)
            )
        ]

    def _process_sheet_ladder2(self) -> None:
        """Eliminates asym IDs that do not correspond to polypeptides from the
        sheet ladder data.
        """
        # Beginning and end comp ids must be polypeptides for strands in a ladder
        self.dssp_struct_ladder_data = self.dssp_struct_ladder_data[
            (
                self.dssp_struct_ladder_data[
                    "_dssp_struct_ladder.beg_1_label_asym_id"
                ].isin(self.asym_id_to_chain_id)
            )
            & (
                self.dssp_struct_ladder_data[
                    "_dssp_struct_ladder.end_1_label_asym_id"
                ].isin(self.asym_id_to_chain_id)
            )
            & (
                self.dssp_struct_ladder_data[
                    "_dssp_struct_ladder.beg_2_label_asym_id"
                ].isin(self.asym_id_to_chain_id)
            )
            & (
                self.dssp_struct_ladder_data[
                    "_dssp_struct_ladder.end_2_label_asym_id"
                ].isin(self.asym_id_to_chain_id)
            )
        ]

    def _assign_one_letter_code(self) -> None:
        """Assigns the canonical and noncanonical one letter codes to
        `self.pdbx_poly_seq_scheme_data`.
        """
        # The sequence IDs must be unique at this point
        assert len(self.pdbx_poly_seq_scheme_data) == len(
            self.pdbx_poly_seq_scheme_data[
                ["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.seq_id"]
            ].drop_duplicates()
        )

        # Process each 3 letter code into a single letter code
        one_letter_dfs = []
        self.found_multiletter = False
        for asym_id, seqid, three_letter in self.pdbx_poly_seq_scheme_data[
            [
                "_pdbx_poly_seq_scheme.asym_id",
                "_pdbx_poly_seq_scheme.seq_id",
                "_pdbx_poly_seq_scheme.mon_id",
            ]
        ].itertuples(index=False):
            # Get the canonical and noncanonical one letter code for the three
            # letter code
            canon_one_letter = AA3_TO_1_CANONICAL.get(three_letter, "X")
            noncanon_one_letter = AA3_TO_1_FULL[three_letter]

            # If the one letter code is actually more than one letters, then we need to
            # create multiple sets of target atom entries. Otherwise, we just need one.
            if (aa_len := len(noncanon_one_letter)) > 1:
                assert canon_one_letter == "X"
                self.found_multiletter = True
                target_atom_suffixes = [str(i) for i in range(1, aa_len + 1)]
            else:
                target_atom_suffixes = [""]

            # Create an entry for each target atom in the residue
            assert len(noncanon_one_letter) == len(target_atom_suffixes)
            target_atom_results = [
                (
                    asym_id,
                    seqid,
                    true_one_letter,
                    canon_one_letter,
                    target_atom_suffix,
                )
                for true_one_letter, target_atom_suffix in zip(
                    noncanon_one_letter, target_atom_suffixes
                )
            ]

            # Convert that entry to a dataframe and append it to the list of dataframes
            one_letter_dfs.append(
                pd.DataFrame(
                    target_atom_results,
                    columns=[
                        "_pdbx_poly_seq_scheme.asym_id",
                        "_pdbx_poly_seq_scheme.seq_id",
                        "noncanon_one_letter",
                        "canon_one_letter",
                        "target_atom_suffix",
                    ],
                )
            )

        # Concatenate dataframes into one large one
        one_letter_df = pd.concat(one_letter_dfs, ignore_index=True)

        # Merge the new data onto `self.pdbx_poly_seq_scheme_data`. This
        # should be one to many if we had any multiletter single letter codes or
        # if this is a multi-model file, one to one otherwise.
        validation = (
            "one_to_many"
            if self.found_multiletter or self.multi_model_file
            else "one_to_one"
        )
        self.pdbx_poly_seq_scheme_data = self.pdbx_poly_seq_scheme_data.merge(
            one_letter_df,
            on=["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.seq_id"],
            how="left",
            validate=validation,
        )

    def _build_asym_id_to_chain_id(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Builds a dictionary that links asym IDs to PDB chain IDs. This is useful
        for converting between chain identities in PDB files and mmCIF files.

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: The dictionaries that maps asym
                IDs to PDB chain IDs and vice versa, respectively.
        """

        # Map asym IDs to PDB chain IDs
        asym_id_to_chain_id = dict(
            self.pdbx_poly_seq_scheme_data[
                ["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.pdb_strand_id"]
            ].to_numpy()
        )

        # Make sure that each asym ID should correspond to a single pdb strand ID
        assert (
            len(asym_id_to_chain_id)
            == len(
                self.pdbx_poly_seq_scheme_data["_pdbx_poly_seq_scheme.asym_id"].unique()
            )
            == len(
                self.pdbx_poly_seq_scheme_data[
                    "_pdbx_poly_seq_scheme.pdb_strand_id"
                ].unique()
            )
        )

        # Build the reverse mapping
        chain_id_to_asym_id = {
            chain_id: asym_id for asym_id, chain_id in asym_id_to_chain_id.items()
        }
        assert len(chain_id_to_asym_id) == len(asym_id_to_chain_id)

        return asym_id_to_chain_id, chain_id_to_asym_id

    def _build_asym_to_ind(self) -> Dict[str, int]:
        """Builds a dictionary that maps from `_pdbx_poly_seq_scheme.asym_id` to
        an integer.

        Returns:
            Dict[str, int]: The mapping.
        """

        # Map from asym ID to an integer
        return {
            asym_id: i
            for i, asym_id in enumerate(
                self.pdbx_poly_seq_scheme_data["_pdbx_poly_seq_scheme.asym_id"].unique()
            )
        }

    def _build_entity_to_asym_ind(
        self,
    ) -> Tuple[Dict[str, Tuple[int, ...]], Dict[int, str]]:
        """Builds two dictionaries. One that maps from an entity ID to a tuple of
        integers reflecting the asym IDs captured by that entity ID (strictly, maps
        from `_pdbx_poly_seq_scheme.entity_id` to a tuple of `_pdbx_poly_seq_scheme.asym_id`
        values) and second that maps from those integers to the entity ID.

        Returns:
            Tuple[Dict[str, Tuple[int, ...]], Dict[int, str]]: The mappings.
        """
        # Populate the dictionary
        entity_to_asym = {}
        asym_to_entity = {}
        for entity_id, asym_id in (
            self.pdbx_poly_seq_scheme_data[
                ["_pdbx_poly_seq_scheme.entity_id", "_pdbx_poly_seq_scheme.asym_id"]
            ]
            .drop_duplicates()
            .itertuples(index=False)
        ):
            # Get the integer representing the asym ID
            asym_int = self.asym_id_to_int[asym_id]

            # Populate `entity_to_asym`
            if entity_id not in entity_to_asym:
                entity_to_asym[entity_id] = []
            entity_to_asym[entity_id].append(asym_int)

            # Populate `asym_to_entity`
            assert asym_int not in asym_to_entity
            asym_to_entity[asym_int] = entity_id

        # Convert lists to tuples
        entity_to_asym = {
            entity_id: tuple(asym_list)
            for entity_id, asym_list in entity_to_asym.items()
        }

        return entity_to_asym, asym_to_entity

    def _index_poly_seq_scheme(self) -> None:
        """Updates `self.pdbx_poly_seq_scheme_data` to include sequence indices
        for each chain and the full sequence, CATH indices for each chain, and
        chain indices that map each chain to an integer.
        """

        # Add sequence and CATH indices to the poly seq scheme data
        updated_dfs = []
        for asym_id in self.asym_id_to_int:
            # Get the data for this asym ID
            asym_df = self.pdbx_poly_seq_scheme_data[
                self.pdbx_poly_seq_scheme_data["_pdbx_poly_seq_scheme.asym_id"]
                == asym_id
            ].copy()

            # Get the cath IDs for this asym ID
            asym_df["CATH_ids"] = [
                str(seq_num) if ins_code == "." else f"{seq_num}({ins_code})"
                for seq_num, ins_code in asym_df[
                    [
                        "_pdbx_poly_seq_scheme.pdb_seq_num",
                        "_pdbx_poly_seq_scheme.pdb_ins_code",
                    ]
                ].itertuples(index=False)
            ]

            # Get the sequence inds for this asym ID
            asym_df["chain_seq_inds"] = np.arange(len(asym_df))

            # Add the updated dataframe to the list
            updated_dfs.append(asym_df)

        # Concatenate the updated dataframes and make sure that the original ordering is
        # preserved
        updated_df = pd.concat(updated_dfs)
        assert updated_df[
            ["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.seq_id"]
        ].equals(
            self.pdbx_poly_seq_scheme_data[
                ["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.seq_id"]
            ]
        )

        # Add full PDB indices to the sequence data
        updated_df["pdb_seq_inds"] = np.arange(len(updated_df))

        # Add chain indices to the sequence data
        updated_df["chain_inds"] = updated_df["_pdbx_poly_seq_scheme.asym_id"].map(
            self.asym_id_to_int
        )

        # Update the poly seq scheme data
        self.pdbx_poly_seq_scheme_data = updated_df

    def _build_peptides_with_atoms(self) -> pd.DataFrame:
        """Expands the `self.pdbx_poly_seq_scheme_data` dataframe to include
        the target atoms for each position. For instance, if `self.pdbx_poly_seq_scheme_data`
        is L long and contains only three letter codes that correspond to one letter
        codes, and if there are 4 target atoms, then the resulting dataframe will
        be 4L long and will contain the target atoms for each position (assuming
        a single model mmCIF file).

        Returns:
            pd.DataFrame: The expanded dataframe.
        """
        # The sequence IDs must be unique at this point
        assert len(self.pdbx_poly_seq_scheme_data) == len(
            self.pdbx_poly_seq_scheme_data.pdb_seq_inds.unique()
        )

        # Build a dataframe that maps from the sequence ID to the target atoms
        seq_id_to_target_atoms = pd.DataFrame(
            [
                (seq_id, target_atom + target_atom_suffix)
                for seq_id, target_atom_suffix in self.pdbx_poly_seq_scheme_data[
                    ["pdb_seq_inds", "target_atom_suffix"]
                ].itertuples(index=False)
                for target_atom in self.target_atoms
            ],
            columns=["pdb_seq_inds", "target_atom"],
        )

        # Map the target atoms onto `self.pdbx_poly_seq_scheme_data` to get the target
        # atoms for each sequence ID
        peptides_with_atoms = self.pdbx_poly_seq_scheme_data.merge(
            seq_id_to_target_atoms,
            how="left",
            on="pdb_seq_inds",
            validate="one_to_many",
        )
        assert len(peptides_with_atoms) == self.n_target_atoms * len(
            self.pdbx_poly_seq_scheme_data
        )

        return peptides_with_atoms

    def _build_chem_comp_dict(self) -> Dict[str, CompoundInfo]:
        """Builds a dictionary that maps from all chemical compounds in
        `self.chem_comp_data` to their properties.

        Returns:
            Dict[str, CompoundInfo]: A dictionary that maps from compound IDs to
                their properties.
        """
        # Build a dictionary that maps from the name of a compound to its properties
        return {
            row[0]: {
                "formula": row[1],
                "weight": row[2],
                "name": row[3],
                "synonyms": "" if row[4] in {".", "?"} else row[4],
            }
            for row in self.chem_comp_data.itertuples(index=False)
        }

    def _build_asym_seq_to_seq_ind(
        self,
    ) -> Dict[
        Tuple[str, str],
        Tuple[
            Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]],
            ...,
        ],
    ]:
        """Builds a dictionary that maps from asym and seq id to pdb sequence
        index, chain sequence index, and chain index

        Returns:
             Dict[
                Tuple[str, str],
                Tuple[
                Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]],
                ...,
            ]
                ]: The mapping.
        """
        # Build a dictionary that maps from asym and seq id to indices
        base_dict = {}
        for (
            asym_id,
            seq_id,
            pdb_seq_inds,
            chain_seq_inds,
            chain_inds,
        ) in self.pdbx_poly_seq_scheme_data[
            [
                "_pdbx_poly_seq_scheme.asym_id",
                "_pdbx_poly_seq_scheme.seq_id",
                "pdb_seq_inds",
                "chain_seq_inds",
                "chain_inds",
            ]
        ].itertuples(
            index=False
        ):
            # Build the key
            key = (asym_id, seq_id)

            # If the key is not in the dictionary, then start a list
            if key not in base_dict:
                base_dict[key] = []

            # Add the indices to the list
            base_dict[key].append((pdb_seq_inds, chain_seq_inds, chain_inds))

        # Conver the lists to tuples
        return {k: tuple(v) for k, v in base_dict.items()}

    def _build_asym_seq_to_resi(self) -> Dict[Tuple[str, str], str]:
        """Builds a dictionary that maps from asym and seq id to residue ID.

        Returns:
            Dict[Tuple[str, str], str]: The mapping.
        """
        # Build a dictionary that maps from asym and seq id to compound ID
        return {
            (asym_id, seq_id): compound_id
            for asym_id, seq_id, compound_id in self.pdbx_poly_seq_scheme_data[
                [
                    "_pdbx_poly_seq_scheme.asym_id",
                    "_pdbx_poly_seq_scheme.seq_id",
                    "_pdbx_poly_seq_scheme.mon_id",
                ]
            ].itertuples(index=False)
        }

    def _scrape_structure_data(self) -> StructureData:
        """Scrapes data at the structure level for every model in the structure.
        The data is returned as a dictionary, one entry for each kind of data. The
        fields returned are below:

            (1) "site_data": See the docstring for the `_gather_important_sites`
                method for details.
            (2) "connections_data": See the docstring for the `_gather_connections`
                method for details.
            (3) "chain_metadata": See the docstring for the `_gather_chain_metadata`
                method for details.
            (4) "structure_metadata": See the docstring for the `_gather_structure_metadata`
                method for details.
            (5) "citations": See the docstring for the `_gather_citation_data` method
                for details.
            (6) "modified_residue_data": See the docstring for the `_gather_modres_info`
                method for details.
            (7) "secondary_structure_data": See the docstring for the `_gather_secondary_structure`
                method for details.
            (8) "bridge_pair_data": See the docstring for the `_gather_bridge_pairs`
                method for details.
            (9) "ladder_sheet_data": See the docstring for the `_gather_ladders_sheets`
                method for details.
            (10) "residue_data": See the docstring for the `_gather_residue_data`
                method for details.
        """
        return {
            "site_data": self._gather_important_sites(),
            "connections_data": self._gather_connections(),
            "chain_metadata": self._gather_chain_metadata(),
            "structure_metadata": self._gather_structure_metadata(),
            "citations": self._gather_citation_data(),
            "modified_residue_data": self._gather_modres_info(),
            "secondary_structure_data": self._gather_secondary_structure(),
            "bridge_pair_data": self._gather_bridge_pairs(),
            "ladder_sheet_data": self._gather_ladders_sheets(),
            "residue_data": self._gather_residue_data(),
        }

    def _scrape_chain_data(self, structure_data: StructureData) -> ScrapedChainData:
        # Now get the chain-specific data for each chain
        chain_data = {}
        for asym_id, chain_ind in self.asym_id_to_int.items():
            # Identify the PDB chain ID
            pdb_chain_id = self.asym_id_to_chain_id[asym_id]

            # Get the chain-specific data
            filtered_data = _filter_all(
                target_chain_ind=chain_ind, structure_data=structure_data
            )

            # Get the expected sequence for this chain. Note that
            # `self.asym_int_to_entity` maps from the chain index to
            # `_pdbx_poly_seq_scheme.entity_id`, which itself then maps to `_entity.id`.
            # We can pull the expected sequence from `self.entity_poly_data` as
            # it has the field `_entity_poly.entity_id` which also maps to `_entity.id`.
            expected_seq = self.entity_poly_data.loc[
                self.entity_poly_data["_entity_poly.entity_id"]
                == self.asym_int_to_entity[chain_ind],
                "_entity_poly.pdbx_seq_one_letter_code_can",
            ].item()

            # Now make sure that the chain sequence is close to the expected
            # sequence for all models
            for model, residue_data in filtered_data["residue_data"].items():
                # Calculate the Levenshtein distance between the expected sequence
                # and the chain sequence
                seqdiff = ls.distance(expected_seq, residue_data["noncanonical_seq"])

                # The distance should be 0 if the sequence is the same. We allow
                # a certain number of differences as it is known that some PDB files
                # have mismatches between the sequence and the structure. Above
                # that threshold, however, we want to investigate further.
                if 1 <= seqdiff <= 10:
                    logging.warning(
                        "Model %s, chain %s has a Levenshtein distance of %d between "
                        "the expected sequence and the chain sequence for %s.",
                        model,
                        pdb_chain_id,
                        seqdiff,
                        self.pdb_id,
                    )
                elif seqdiff > 10:
                    raise ValueError(
                        f"Model {model}, chain {pdb_chain_id} has a Levenshtein "
                        f"distance of {seqdiff} between the expected sequence and "
                        f"the chain sequence for {self.pdb_id}."
                    )

            # Record data for the chain
            assert pdb_chain_id not in chain_data
            chain_data[pdb_chain_id] = filtered_data

        return chain_data

    def _get_cath_data(
        self, chain_data: ScrapedChainData, cath_records: Sequence[CATHRecord]
    ) -> ScrapedChainData:
        # Process the CATH records
        cath_data = {}
        for cath_record in cath_records:
            # We cannot have recorded a CATH domain with this ID already
            assert cath_record.domain_name not in cath_data

            # If the chain id is not in the chain id to asym id mapping, then
            # we cannot get the sequence data for this chain. Raise a warning
            # and continue.
            if cath_record.chain_id not in self.chain_id_to_asym_id:
                logging.warning(
                    "Could not find chain %s in the chain ID to asym ID mapping "
                    "for domain %s. Abandoning scraping for this domain.",
                    cath_record.chain_id,
                    cath_record.domain_name,
                )
                continue

            # Get the sequence data for the target chain
            seq_data = self.pdbx_poly_seq_scheme_data[
                self.pdbx_poly_seq_scheme_data["_pdbx_poly_seq_scheme.asym_id"]
                == self.chain_id_to_asym_id[cath_record.chain_id]
            ]

            # Scrape the CATH data. If we get a CATH exception, then raise a warning
            # and continue.
            try:
                cath_data[cath_record.domain_name] = (
                    cath_record._scrape_from_chain_data(  # pylint: disable=protected-access
                        seq_data=seq_data, chain_data=chain_data
                    )
                )
            except CATHError as error:
                logging.warning(
                    "Encountered an error when scraping CATH data for domain %s: %s."
                    "Abandoning scraping for this domain.",
                    cath_record.domain_name,
                    error,
                )

        # Return the data
        return cath_data

    def scrape(
        self, cath_records: Optional[Sequence[CATHRecord]] = None
    ) -> Tuple[StructureData, ScrapedChainData, ScrapedChainData]:
        """Scrapes data at the chain and structure levels for every chain and every
        model in the structure.

        Returns:
            Tuple[StructureData, ScrapedChainData, ScrapedChainData]: A tuple of
                dictionaries. The first dictionary contains data at the structure
                level, while the second and third contain data at the chain and
                CATH levels, respectively. The chain and CATH data contain the same
                information as the structure data (information on which can be found
                in the docstring for the `_scrape_structure_data` method), only
                for each chain and CATH domain, respectively.
        """
        # Get the structure data
        structure_data = self._scrape_structure_data()

        # Get the chain data
        chain_data = self._scrape_chain_data(structure_data)

        # Get the CATH data
        cath_data = self._get_cath_data(
            chain_data, [] if cath_records is None else cath_records
        )

        return structure_data, chain_data, cath_data

    def _gather_important_sites(self) -> ImportantSiteInfo:
        """Extracts information on important sites in the structure. Primarily,
        this includes binding sites

        Returns:
            ImportantSiteInfo: A dictionary containing information on the sites of
                interest. The keys of this dictionary are the site ids (i.e., the
                "_struct_site_gen.site_id" field in the cif file). The values are
                dictionaries that contain the following information:

                (1) "pdb_seq_inds": A numpy array of sequence indices that correspond
                    to the residues that make up the site in the full PDB sequence.
                (2) "chain_seq_inds": A numpy array of sequence indices that correspond
                    to the residues that make up the site in the corresponding chain
                    sequence.
                (3) "chain_inds": A numpy array of chain indices that correspond
                    to the chain of the residue that makes up the site.
                (4) "desc": A description of the site. This is the "_struct_site.details"
                    field in the cif file.
                (5) "bound_molecules": A dictionary that maps from the name of a
                    compound (i.e., the "_chem_comp.name" field in the cif file)
                    to properties of that compound. These properties include the
                    below:

                    (a) "formula": The chemical formula of the compound. This is
                        the "_chem_comp.formula" field in the cif file.
                    (b) "weight": The molecular weight of the compound. This is
                        the "_chem_comp.formula_weight" field in the cif file.
                    (c) "name": The name of the compound. This is the "_chem_comp.name"
                        field in the cif file.
                    (d) "synonyms": A list of synonyms for the compound. This is
                        the "_chem_comp.pdbx_synonyms" field in the cif file.
        """
        # Separate amino acids from ligands in the important sites
        non_aa_mask = (
            self.struct_site_gen_data["_struct_site_gen.label_seq_id"] == "."
        ) | (
            ~self.struct_site_gen_data["_struct_site_gen.label_asym_id"].isin(
                self.polypeptide_asym_ids
            )
        )
        aa_data = self.struct_site_gen_data[~non_aa_mask]
        ligand_data = self.struct_site_gen_data[non_aa_mask]

        # Attach sequence index data to the amino acid data. Note that
        # `_struct_site_gen.label_asym_id` points to `_atom_site.label_asym_id`,
        # which `_pdbx_poly_seq_scheme.asym_id` points to as well.
        # `_struct_site_gen.label_seq_id` points to `_atom_site.label_seq_id`,
        # which points to `_entity_poly_seq.num` along with `_pdbx_poly_seq_scheme.seq_id`.
        aa_data = aa_data.merge(
            self.pdbx_poly_seq_scheme_data,
            how="left",
            left_on=["_struct_site_gen.label_asym_id", "_struct_site_gen.label_seq_id"],
            right_on=["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.seq_id"],
            validate="many_to_many" if self.found_multiletter else "many_to_one",
        )
        assert (aa_data["_pdbx_poly_seq_scheme.mon_id"].notna()).all()
        assert (
            aa_data["_pdbx_poly_seq_scheme.mon_id"]
            == aa_data["_struct_site_gen.label_comp_id"]
        ).all()

        # Get a description of each site
        site_to_desc = {
            site: "" if desc == "?" or desc == "." else desc
            for site, desc in self.struct_site_data[
                ["_struct_site.id", "_struct_site.details"]
            ].itertuples(index=False)
        }

        # Gather information on each site of interest
        site_info = {}
        for site_id in aa_data["_struct_site_gen.site_id"].unique():
            # Get data on the ligand and amino acid residues in the site
            site_aa_data = aa_data[aa_data["_struct_site_gen.site_id"] == site_id]
            ligands = (
                ligand_data.loc[
                    ligand_data["_struct_site_gen.site_id"] == site_id,
                    "_struct_site_gen.label_comp_id",
                ]
                .unique()
                .tolist()
            )

            # Get index data for the amino acid residues in the site
            chain_seq_inds = site_aa_data.chain_seq_inds.to_numpy()
            pdb_seq_inds = site_aa_data.pdb_seq_inds.to_numpy()
            chain_inds = site_aa_data.chain_inds.to_numpy()

            # Record information on the site
            site_info[site_id] = {
                "pdb_seq_inds": pdb_seq_inds,
                "chain_seq_inds": chain_seq_inds,
                "chain_inds": chain_inds,
                "desc": site_to_desc[site_id],
                "bound_molecules": tuple(
                    self.chem_comp_dict[ligand] for ligand in ligands
                ),
            }

        return site_info

    def _gather_connections(self) -> ConnectionInfo:
        """Identifies connections between residues in a model.

        Returns:
            ConnectionInfo: Information on the identified connections. Each entry
                in this output is a dictionary corresponding to a single connection.
                The key of the dictionary is the connection index. The values contain
                the below fields:

                    (1) "type": The connection type (e.g., "disulfide bond")
                    (2) "pdb_seq_inds": An array of sequence indices that take part
                        in the connection. These indices point to the full PDB sequence.
                    (3) "chain_seq_inds": An array of sequence indices that take part
                        in the connection. These indices point to the chain sequence.
                    (4) "chain_inds": An array of chain indices that take part in the
                        connection. These identify the chain of the residues that
                        take part in the connection.
                    (5) "compound": A dictionary containing information on the
                        non-polymeric compound that takes part in the connection.
                        This is a dictionary that contains the below entries:

                        (a) "name": The name of the compound. This is the
                            "_chem_comp.name" field in the cif file.
                        (b) "formula": The chemical formula of the compound. This is
                            the "_chem_comp.formula" field in the cif file.
                        (c) "weight": The molecular weight of the compound. This is
                            the "_chem_comp.formula_weight" field in the cif file.
                        (d) "synonyms": A list of synonyms for the compound. This is
                            the "_chem_comp.pdbx_synonyms" field in the cif file.

                        Note that if there is no non-polymeric compound involved
                        in the connection, then the "compound" field will be None.
        """
        # Now we process all connections
        connections = {}
        for conn_ind, (_, row) in enumerate(self.struct_conn_data.iterrows()):
            # Get the connection type
            conn_type = row["_struct_conn.conn_type_id"]

            # Get the indices of the atoms involved in the connection. Note that
            # `_struct_conn.ptnr#_label_asym_id` points to `_atom_site.label_asym_id`,
            # to which `_pdbx_poly_seq_scheme.asym_id` also points, and
            # `_struct_conn.ptnr#_label_seq_id` points to `_atom_site.label_seq_id`,
            # which points to `_entity_poly_seq.num` along with `_pdbx_poly_seq_scheme.seq_id`
            ptnr1_key = (
                row["_struct_conn.ptnr1_label_asym_id"],
                row["_struct_conn.ptnr1_label_seq_id"],
            )
            ptnr2_key = (
                row["_struct_conn.ptnr2_label_asym_id"],
                row["_struct_conn.ptnr2_label_seq_id"],
            )
            ptnr1_inds = self.asym_seq_to_seq_ind.get(ptnr1_key, None)
            ptnr2_inds = self.asym_seq_to_seq_ind.get(ptnr2_key, None)

            # If neither partner is part of the polymer, skip
            if ptnr1_inds is None and ptnr2_inds is None:
                continue

            # Variables for storing information on the connection
            connected_residues = []
            connected_compound = None

            # If ptnr1 is part of the polymer, add its residue information
            if ptnr1_inds is not None:
                assert (
                    self.asym_seq_to_comp[ptnr1_key]
                    == row["_struct_conn.ptnr1_label_comp_id"]
                )
                connected_residues.extend(ptnr1_inds)
            else:
                connected_compound = self.chem_comp_dict[
                    row["_struct_conn.ptnr1_label_comp_id"]
                ]

            # If ptnr2 is part of the polymer, add its residue information
            if ptnr2_inds is not None:
                assert (
                    self.asym_seq_to_comp[ptnr2_key]
                    == row["_struct_conn.ptnr2_label_comp_id"]
                )
                connected_residues.extend(ptnr2_inds)
            else:
                assert connected_compound is None
                connected_compound = self.chem_comp_dict[
                    row["_struct_conn.ptnr2_label_comp_id"]
                ]

            # Convert connected residues to numpy arrays
            pdb_seq_inds, chain_seq_inds, chain_inds = [
                np.array(inds) for inds in zip(*connected_residues)
            ]

            # Record the connection
            connections[conn_ind] = {
                "type": conn_type,
                "pdb_seq_inds": pdb_seq_inds,
                "chain_seq_inds": chain_seq_inds,
                "chain_inds": chain_inds,
                "compound": connected_compound,
            }

        return connections

    def _gather_chain_metadata(self) -> ChainMetadata:
        """Gathers metadata at the chain level.

        Returns:
            ChainMetadata: Collected metadata. This is a dictionary that maps from
                chain int to a tuple of dictionaries of information about that chain.
                Note that the chain int values are the values in `self.asym_id_to_int`.
                Each dictionary making up the values of the output dictionary contains
                the following information:

                (1) "src_host": The organism from which this chain was originally
                    derived. This is one of the following fields from the cif file:
                    (a) "_pdbx_entity_src_syn.organism_scientific"
                    (b) "_entity_src_nat.pdbx_organism_scientific"
                    (c) "_entity_src_gen.pdbx_gene_src_scientific_name"
                (2) "src_tax_id": The taxonomic ID of the organism from which this
                    chain was originally derived. This is one of the following fields
                    from the cif file:
                    (a) "_pdbx_entity_src_syn.ncbi_taxonomy_id"
                    (b) "_entity_src_nat.pdbx_ncbi_taxonomy_id"
                    (c) "_entity_src_gen.pdbx_gene_src_ncbi_taxonomy_id"
                (3) "src_organ": The organ from which this chain was originally
                    derived. This is one of the following fields from the cif file:
                    (a) "_entity_src_nat.pdbx_organ"
                    (b) "_entity_src_gen.pdbx_gene_src_organ"
                (4) "src_tissue": The tissue from which this chain was originally
                    derived. This is one of the following fields from the cif file:
                    (a) "_entity_src_nat.tissue"
                    (b) "_entity_src_gen.gene_src_tissue"
                (5) "src_cell": The cell from which this chain was originally
                    derived. This is one of the following fields from the cif file:
                    (a) "_entity_src_nat.pdbx_cell"
                    (b) "_entity_src_gen.pdbx_gene_src_cell"
                (6) "src_organelle": The organelle from which this chain was originally
                    derived. This is one of the following fields from the cif file:
                    (a) "_entity_src_nat.pdbx_organelle"
                    (b) "_entity_src_gen.pdbx_gene_src_organelle"
                (7) "src_cellular_location": The cellular location from which this
                    chain was originally derived. This is one of the following fields
                    from the cif file:
                    (a) "_entity_src_nat.pdbx_cellular_location"
                    (b) "_entity_src_gen.pdbx_gene_src_cellular_location"
                (8) "exp_name": The name of the expression system used to produce
                    this chain. This is the "_entity_src_gen.pdbx_host_org_scientific_name"
                    field from the cif file.
                (9) "exp_organ": The organ of the expression system used to produce
                    this chain. This is the "_entity_src_gen.pdbx_host_org_organ"
                    field from the cif file.
                (10) "exp_tissue": The tissue of the expression system used to produce
                    this chain. This is the "_entity_src_gen.pdbx_host_org_tissue"
                    field from the cif file.
                (11) "exp_cell": The cell of the expression system used to produce
                    this chain. This is the "_entity_src_gen.pdbx_host_org_cell"
                    field from the cif file.
                (12) "exp_organelle": The organelle of the expression system used to
                    produce this chain. This is the "_entity_src_gen.pdbx_host_org_organelle"
                    field from the cif file.
                (13) "exp_cellular_location": The cellular location of the expression
                    system used to produce this chain. This is the
                    "_entity_src_gen.pdbx_host_org_cellular_location" field from the
                    cif file.
                (14) "ec_number": The EC number of the enzyme that this chain encodes.
                    This is the "_entity.pdbx_ec" field from the cif file.
                (15) "description": A description of the chain. This is the
                    "_entity.pdbx_description" field from the cif file.
                (16) "details": Details on the chain. This is the "_entity.details"
                    field from the cif file.
        """
        # Merge source information onto the entity_poly table and replace NaNs with
        # empty strings. Note that `_entity_src_gen.entity_id`, `_entity_src_nat.entity_id`,
        # `_pdbx_entity_src_syn.entity_id`, and `_entity_poly.entity_id` all point
        # to `_entity.id`
        entity_poly_data = (
            self.entity_poly_data.merge(
                self.entity_src_gen,
                how="left",
                left_on="_entity_poly.entity_id",
                right_on="_entity_src_gen.entity_id",
            )
            .merge(
                self.entity_src_nat,
                how="left",
                left_on="_entity_poly.entity_id",
                right_on="_entity_src_nat.entity_id",
            )
            .merge(
                self.entity_src_syn,
                how="left",
                left_on="_entity_poly.entity_id",
                right_on="_pdbx_entity_src_syn.entity_id",
            )
            .merge(
                self.entity_data,
                how="left",
                left_on="_entity_poly.entity_id",
                right_on="_entity.id",
            )
            .fillna("")
        )

        # We are only taking polypeptides, so we filter out any non-polypeptide
        # entities
        entity_poly_data = entity_poly_data[
            entity_poly_data["_entity_poly.type"] == "polypeptide(L)"
        ]

        # For each entity, pull the relevant information.
        chain_data = {}
        for _, entity in entity_poly_data.iterrows():
            # Build a dictionary for storing the information on this entity
            entity_data = {}

            # Grab the host organism
            entity_data["src_host"] = _select_opt(
                [
                    entity["_pdbx_entity_src_syn.organism_scientific"],
                    entity["_entity_src_nat.pdbx_organism_scientific"],
                    entity["_entity_src_gen.pdbx_gene_src_scientific_name"],
                ]
            )

            # Grab the taxonomic id
            entity_data["src_tax_id"] = _select_opt(
                [
                    entity["_pdbx_entity_src_syn.ncbi_taxonomy_id"],
                    entity["_entity_src_nat.pdbx_ncbi_taxonomy_id"],
                    entity["_entity_src_gen.pdbx_gene_src_ncbi_taxonomy_id"],
                ]
            )

            # Grab the organ, tissue, cell, organell, and cellular location from
            # the source
            entity_data["src_organ"] = _select_opt(
                [
                    entity["_entity_src_nat.pdbx_organ"],
                    entity["_entity_src_gen.pdbx_gene_src_organ"],
                ]
            )
            entity_data["src_tissue"] = _select_opt(
                [
                    entity["_entity_src_nat.tissue"],
                    entity["_entity_src_gen.gene_src_tissue"],
                ]
            )
            entity_data["src_cell"] = _select_opt(
                [
                    entity["_entity_src_nat.pdbx_cell"],
                    entity["_entity_src_gen.pdbx_gene_src_cell"],
                ]
            )
            entity_data["src_organelle"] = _select_opt(
                [
                    entity["_entity_src_nat.pdbx_organelle"],
                    entity["_entity_src_gen.pdbx_gene_src_organelle"],
                ]
            )
            entity_data["src_cellular_location"] = _select_opt(
                [
                    entity["_entity_src_nat.pdbx_cellular_location"],
                    entity["_entity_src_gen.pdbx_gene_src_cellular_location"],
                ]
            )

            # Grab the scientific name, taxonomic id, organ, tissue, cell, organell,
            # and cellular location from the expression system
            entity_data["exp_name"] = _missing_to_empty_str(
                entity["_entity_src_gen.pdbx_host_org_scientific_name"]
            )
            entity_data["exp_organ"] = _missing_to_empty_str(
                entity["_entity_src_gen.pdbx_host_org_organ"]
            )
            entity_data["exp_tissue"] = _missing_to_empty_str(
                entity["_entity_src_gen.pdbx_host_org_tissue"]
            )
            entity_data["exp_cell"] = _missing_to_empty_str(
                entity["_entity_src_gen.pdbx_host_org_cell"]
            )
            entity_data["exp_organelle"] = _missing_to_empty_str(
                entity["_entity_src_gen.pdbx_host_org_organelle"]
            )
            entity_data["exp_cellular_location"] = _missing_to_empty_str(
                entity["_entity_src_gen.pdbx_host_org_cellular_location"]
            )

            # Grab the commision number and a description for the entity
            entity_data["ec_number"] = _missing_to_empty_str(entity["_entity.pdbx_ec"])
            entity_data["description"] = _missing_to_empty_str(
                entity["_entity.pdbx_description"]
            )
            entity_data["details"] = _missing_to_empty_str(entity["_entity.details"])

            # Use `self.entity_to_asym_int` to assign the entity to the correct chains.
            # Note that this is a dictionary mapping entity ids given by
            # `_pdbx_poly_seq_scheme.entity_id` to the indices of the asym IDs given
            # in `self.asym_id_to_int`. `_pdbx_poly_seq_scheme.entity_id` points
            # directly to `_entity.id`.
            for asym_int in self.entity_to_asym_int[entity["_entity.id"]]:
                # Record the data for this entity
                if asym_int not in chain_data:
                    chain_data[asym_int] = []
                chain_data[asym_int].append(entity_data)

        # All values should be tuples
        return {k: tuple(v) for k, v in chain_data.items()}

    def _gather_structure_metadata(self) -> StructureMetadata:
        """Gathers metadata at the structure level. These data apply to all chains
        within a structure.

        Returns:
            StructureMetadata: A dictionary containing the following information:

                (1) "title": The title of the structure. This is the "_struct.title"
                    field from the cif file.
                (2) "resolution": The resolution of the structure. This is derived
                    from the PDB CIF parser held within the CIFFile object.
                (3) "experimental_method": The experimental method used to determine
                    the structure. This is the "_exptl.method" field from the cif file.
                (4) "description": A description of the structure. This is the
                    "_struct.pdbx_descriptor" field from the cif file.
                (5) "keywords": A list of keywords associated with the structure.
                    This is the "_struct_keywords.text" field from the cif file.
        """
        # Gather meta data
        return {
            "title": self.title,
            "resolution": self.structure.header["resolution"],
            "experimental_method": self.exptl_method,
            "description": self.description,
            "keywords": self.keywords,
        }

    def _gather_citation_data(self) -> CitationInfoOutput:
        """Extracts information on the citations associated with the structure
        from the cif file.

        Returns:
            CitationInfoOutput: A dictionary that maps from from citation ID to
                a dictionary containing information on that citation. One key of
                this dictionary will be "primary", and accounts for the primary
                citation associated with the structure. All others are numbered
                starting at 2 and account for additional citations associated with
                the structure. See `_process_citation_row` for more information
                on the information contained in the dictionary.
        """
        # A dictionary for storing information on citations
        citation_data = {}

        # Gather data on the primary citation. There should only be one primary
        # citation
        primary_data = self.citation_data[
            self.citation_data["_citation.id"] == "primary"
        ]
        if len(primary_data) == 1:
            citation_data["primary"] = _process_citation_row(
                primary_data.iloc[0].fillna("")
            )
        else:
            assert len(primary_data) == 0

        # Now gather data on the other citations
        other_data = self.citation_data[self.citation_data["_citation.id"] != "primary"]
        for citation_ind, (_, row) in enumerate(other_data.iterrows(), 2):
            citation_data[str(citation_ind)] = _process_citation_row(row.fillna(""))

        return citation_data

    def _gather_modres_info(self) -> ModresOutput:
        """Gathers information on any modified residues in the structure.

        Returns:
            ModresOutput: A tuple of dictionaries, where each dictionary provides
                information on a single modified residue. The keys of each dictionary
                are the following:

                    (1) "pdb_seq_ind": The sequence index of the modified residue
                        in the full PDB sequence.
                    (2) "chain_seq_ind": The sequence index of the modified residue
                        in the chain sequence.
                    (3) "chain_ind": The index of the chain that the modified residue
                        is part of.
                    (4) "residue_name": The name of the modified residue. This is
                        the "_chem_comp.name" field from the cif file for this
                        modified residue.
                    (5) "formula": The chemical formula of the modified residue.
                        This is the "_chem_comp.formula" field from the cif file
                        for this modified residue.
                    (6) "weight": The molecular weight of the modified residue.
                        This is the "_chem_comp.formula_weight" field from the cif
                        file for this modified residue.
                    (7) "synonyms": A list of synonyms for the modified residue.
                        This is the "_chem_comp.pdbx_synonyms" field from the cif
                        file for this modified residue.
                    (8) "details": Details on the modified residue. This is the
                        "_pdbx_struct_mod_residue.details" field from the cif file
                        for this modified residue.
        """
        # Loop over the modres data. Note that we can pull the sequence indices using the
        # self.asym_seq_to_seq_ind as it's keys are derived from `_pdbx_poly_seq_scheme.asym_id`
        # and `_pdbx_poly_seq_scheme.seq_id` columns. `_pdbx_poly_seq_scheme.asym_id`
        # and `_pdbx_struct_mod_residue.label_asym_id` both point to `_atom_site.label_asym_id`.
        # `_pdbx_struct_mod_residue.label_seq_id` points to `_atom_site.label_seq_id`, and
        # both `_atom_site.label_seq_id` and `_pdbx_poly_seq_scheme.seq_id` point to
        # `_entity_poly_seq.num`.
        modified_residues = []
        for _, asym_id, comp_id, seq_id, details in self.modres_data.itertuples(
            index=False
        ):
            # Get the sequence indices for the modified residue.
            out = self.asym_seq_to_seq_ind.get((asym_id, seq_id), None)

            # If the sequence indices are not found, then skip this modified residue. It
            # is likely that the modified residue is not part of a polypeptide.
            if out is None:
                continue

            # Make sure the compound matches what we expect
            assert self.asym_seq_to_comp[(asym_id, seq_id)] == comp_id

            # Get information on the modified residue. Note that
            # `_pdb_struct_mod_residue.label_comp_id` points to `_atom_site.label_comp_id`
            # which in turn points to `_chem_comp.id`, which defines the keys of
            # `self.chem_comp_dict`.
            chemical_data = self.chem_comp_dict[comp_id]

            # Record all modified residues
            for pdb_seq_ind, chain_seq_ind, chain_ind in out:
                # Record the modified residue.
                modified_residues.append(
                    {
                        "pdb_seq_ind": pdb_seq_ind,
                        "chain_seq_ind": chain_seq_ind,
                        "chain_ind": chain_ind,
                        "residue_name": chemical_data["name"],
                        "formula": chemical_data["formula"],
                        "weight": chemical_data["weight"],
                        "synonyms": chemical_data["synonyms"],
                        "details": details,
                    }
                )

        return tuple(modified_residues)

    def _gather_secondary_structure(self) -> DSSPSecondaryStructure:
        """Gathers information on secondary structure elements.

        Returns:
            DSSPSecondaryStructure: A dictionary that links secondary structure
                IDs (the "_struct_conf.id" field from the cif file) to a dictionary
                with the below fields:

                    (1) "pdb_seq_inds": An array of sequence indices that take part
                        in the secondary structure element. These indices are for
                        the full PDB sequence.
                    (2) "chain_seq_inds": An array of sequence indices that take
                        part in the secondary structure element. These indices are
                        for the chain sequence.
                    (3) "chain_inds": An array of chain indices that take part in
                        the secondary structure element. These indices identify the
                        chain of the residues that take part in the secondary
                        structure element.
                    (4) "ss_type": The type of secondary structure element. This
                        is the "_struct_conf.conf_type_id" field from the cif file.
                    (5) "details": Details on the secondary structure element. This
                        is the "_struct_conf.details" field from the cif file.
        """
        # Filter out rows that have starts and ends that are not in the same chain.
        # There shouldn't be many. Raise a warning when we encounter some.
        ss_data = self.dssp_secondary_structure_data[
            self.dssp_secondary_structure_data["_struct_conf.beg_label_asym_id"]
            == self.dssp_secondary_structure_data["_struct_conf.end_label_asym_id"]
        ]
        if (lendiff := len(ss_data) - len(self.dssp_secondary_structure_data)) != 0:
            logging.warning(
                "Found and dropped %d secondary structure elements with different "
                "start and end chains for %s",
                lendiff,
                self.pdb_id,
            )

        # Check the identities of the residues at the start and end of each secondary
        # structure element. Note that `struct_conf.beg_/end_label_asym_id` and
        # to `_pdbx_poly_seq_scheme.asym_id` both point to `_atom_site.label_asym_id`.
        # `struct_conf.beg/end_label_seq_id` point to `_atom_site.label_seq_id`, which
        # itself points to `_entity_poly_seq.num` along with `_pdbx_poly_seq_scheme.seq_id`.
        validate = "many_to_many" if self.found_multiletter else "many_to_one"
        merge_on_beg = ss_data.merge(
            self.pdbx_poly_seq_scheme_data,
            how="left",
            left_on=["_struct_conf.beg_label_asym_id", "_struct_conf.beg_label_seq_id"],
            right_on=["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.seq_id"],
            validate=validate,
        )
        merge_on_end = ss_data.merge(
            self.pdbx_poly_seq_scheme_data,
            how="left",
            left_on=["_struct_conf.end_label_asym_id", "_struct_conf.end_label_seq_id"],
            right_on=["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.seq_id"],
            validate=validate,
        )
        assert (
            merge_on_beg["_struct_conf.beg_label_comp_id"]
            == merge_on_beg["_pdbx_poly_seq_scheme.mon_id"]
        ).all()
        assert (
            merge_on_end["_struct_conf.end_label_comp_id"]
            == merge_on_end["_pdbx_poly_seq_scheme.mon_id"]
        ).all()

        # Expand the secondary structure data to include all intermediate sequence IDs.
        ss_info = {}
        for _, row in ss_data.iterrows():
            # Get the range of sequence IDs for this secondary structure element.
            beg_seq_id = int(row["_struct_conf.beg_label_seq_id"])
            end_seq_id = int(row["_struct_conf.end_label_seq_id"])
            start_id = min(beg_seq_id, end_seq_id)
            end_id = max(beg_seq_id, end_seq_id) + 1

            # Get the asym IDs for the residues belonging to this secondary structure
            # element.
            beg_asym_id = row["_struct_conf.beg_label_asym_id"]
            end_asym_id = row["_struct_conf.end_label_asym_id"]
            assert beg_asym_id == end_asym_id

            # Get the sequence IDs for the residues belonging to this secondary structure
            # element.
            pdb_inds, chain_seq_inds, chain_inds = [], [], []
            for seqid in range(start_id, end_id):
                temp_pdb_inds, temp_chain_seq_inds, temp_chain_inds = zip(
                    *self.asym_seq_to_seq_ind[(beg_asym_id, str(seqid))]
                )
                pdb_inds.extend(temp_pdb_inds)
                chain_seq_inds.extend(temp_chain_seq_inds)
                chain_inds.extend(temp_chain_inds)

            # Record the secondary structure information
            ss_info[row["_struct_conf.id"]] = {
                "pdb_seq_inds": np.array(pdb_inds),
                "chain_seq_inds": np.array(chain_seq_inds),
                "chain_inds": np.array(chain_inds),
                "ss_type": row["_struct_conf.conf_type_id"],
                "details": row["_struct_conf.details"],
            }

        # The data and info should be the same length
        assert len(ss_data) == len(ss_info)

        return ss_info

    def _gather_bridge_pairs(
        self,
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        """Gathers information on bridge pairs present in the structure.

        Returns:
            Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]: The first array
                is a 2D array that gives the sequence indices of the reference and
                partner residues in the bridge pair. The first column gives the
                pdb sequence index of the reference residue, the second column gives
                chain sequence index of the reference residue, the third column gives
                the chain index of the reference residue, the fourth column gives
                the pdb sequence index of the partner residue, the fifth column gives
                the chain sequence index of the partner residue, the sixth column
                gives the chain index of the partner residue, and the seventh column
                indicates whether the partner is a donor or acceptor. A "1" in the
                last column indicates a donor and a "-1" indicates an acceptor.
                The second array is a 1D array that gives the energy of each bridge
                pair.
        """
        # Process the bridge pair data
        bridge_pair_inds = []
        bridge_pair_energies = []
        for _, row in self.dssp_bridge_pair_data.iterrows():
            # Get the reference sequence index
            ref_key = (
                row["_dssp_struct_bridge_pairs.label_asym_id"],
                row["_dssp_struct_bridge_pairs.label_seq_id"],
            )
            ref_inds = self.asym_seq_to_seq_ind[ref_key]

            # Make sure the compound matches
            assert (
                self.asym_seq_to_comp[ref_key]
                == row["_dssp_struct_bridge_pairs.label_comp_id"]
            )

            # We have two donors and two acceptors. Process each.
            for pair_ind in (1, 2):
                # We have a donor and an acceptor. Process each
                for key_type in ("acceptor", "donor"):
                    # Get the key and energy for the pair
                    key = (
                        row[
                            f"_dssp_struct_bridge_pairs.{key_type}_{pair_ind}_label_asym_id"
                        ],
                        row[
                            f"_dssp_struct_bridge_pairs.{key_type}_{pair_ind}_label_seq_id"
                        ],
                    )
                    energy = row[
                        f"_dssp_struct_bridge_pairs.{key_type}_{pair_ind}_energy"
                    ]

                    # If the key is "?", then we have no partner
                    if key == ("?", "?"):
                        continue

                    # If the key is not in the asym id to index dictionary, then we
                    # have a bridge pair to a non-polypeptide. We skip this bridge pair
                    if key not in self.asym_seq_to_seq_ind:
                        continue

                    # Make sure the compound matches
                    assert (
                        self.asym_seq_to_comp[key]
                        == row[
                            f"_dssp_struct_bridge_pairs.{key_type}_{pair_ind}_label_comp_id"
                        ]
                    )

                    # Get the sequence indices and convert energy to a float
                    pdb_inds = self.asym_seq_to_seq_ind[key]
                    energy = float(energy)

                    # Record information on the pair. Encode a donor as "1" and an acceptor
                    # as "-1"
                    type_code = 1 if key_type == "donor" else -1
                    for ref_pdb_seq_ind, ref_chain_seq_ind, ref_chain_ind in ref_inds:
                        for pdb_seq_ind, chain_seq_ind, chain_ind in pdb_inds:
                            bridge_pair_inds.append(
                                (
                                    ref_pdb_seq_ind,
                                    ref_chain_seq_ind,
                                    ref_chain_ind,
                                    pdb_seq_ind,
                                    chain_seq_ind,
                                    chain_ind,
                                    type_code,
                                )
                            )
                            bridge_pair_energies.append(energy)

        # Convert the bridge pair data to numpy arrays
        bridge_pair_inds = np.array(bridge_pair_inds, dtype=np.int64)
        bridge_pair_energies = np.array(bridge_pair_energies, dtype=np.float32)

        return bridge_pair_inds, bridge_pair_energies

    def _gather_ladders_sheets(self) -> Tuple[DSSPLadderInfo, DSSPSheetInfo]:
        """Gathers information on the ladders and sheets present in the structure.

        Returns:
            Tuple[DSSPLadderInfo, DSSPSheetInfo]: The first output gives information
                on the ladders present in the structure while the second output
                gives information on the sheets present in the structure. Each entry
                in the first tuple gives information on a single ladder. The first
                entry in the tuple is an integer encoding whether the two strands
                in the ladder are oriented parallel (1) or antiparallel (-1). The
                second entry gives the sequence indices of the first strand in the
                ladder and the third entry gives the sequence indices of the second
                strand in the ladder. In both the second and third entries, the
                first column in the output array gives the pdb sequence indices,
                the second gives the chain sequence indices, and the third identifies
                the chain. The second output maps from sheet names to a tuple of
                indices. The tuple of indices points to ladders in the first output
                that make up that sheet.
        """
        # Start and end asym IDs must be the same. Filter out any where they are not.
        ladder_data = self.dssp_struct_ladder_data[
            (
                self.dssp_struct_ladder_data["_dssp_struct_ladder.beg_1_label_asym_id"]
                == self.dssp_struct_ladder_data[
                    "_dssp_struct_ladder.end_1_label_asym_id"
                ]
            )
            & (
                self.dssp_struct_ladder_data["_dssp_struct_ladder.beg_2_label_asym_id"]
                == self.dssp_struct_ladder_data[
                    "_dssp_struct_ladder.end_2_label_asym_id"
                ]
            )
        ]
        if (difflen := len(self.dssp_struct_ladder_data) - len(ladder_data)) > 0:
            logging.warning(
                "Filtered out %d ladder entries where the start and end asym IDs were "
                "not the same for %s",
                difflen,
                self.pdb_id,
            )

        # Gather the sequence indices of all ladders
        ladder_inds = []
        for _, ladder in ladder_data.iterrows():
            # Determine whether this is parallel or antiparallel. Encode as 1 for parallel
            # and -1 for antiparallel
            par_anti_par = ladder["_dssp_struct_ladder.type"]
            if par_anti_par == "parallel":
                parallel_enc = 1
            elif par_anti_par == "anti-parallel":
                parallel_enc = -1
            else:
                raise ValueError(
                    f"Unexpected parallel/antiparallel name: {par_anti_par}"
                )

            # Get the beta sheet that this ladder is a part of
            sheet_id = ladder["_dssp_struct_ladder.sheet_id"]

            # Get the indices of the first and second ranges
            range_inds = [sheet_id, parallel_enc]
            for range_ in (1, 2):
                # Get the asym ids and seq inds for the start and end of the range
                asym_id = ladder[f"_dssp_struct_ladder.beg_{range_}_label_asym_id"]
                range_seq_ids = (
                    int(ladder[f"_dssp_struct_ladder.beg_{range_}_label_seq_id"]),
                    int(ladder[f"_dssp_struct_ladder.end_{range_}_label_seq_id"]),
                )

                # Make sure beginning and end compounds match
                assert (
                    self.asym_seq_to_comp[(asym_id, str(range_seq_ids[0]))]
                    == ladder[f"_dssp_struct_ladder.beg_{range_}_label_comp_id"]
                )
                assert (
                    self.asym_seq_to_comp[(asym_id, str(range_seq_ids[1]))]
                    == ladder[f"_dssp_struct_ladder.end_{range_}_label_comp_id"]
                )

                # Get the sequence indices for the range. Skip entries that are not
                # part of polypeptides
                expanded_inds = []
                for seqind in range(min(range_seq_ids), max(range_seq_ids) + 1):
                    # Get the key for the sequence index
                    seqind_key = (asym_id, str(seqind))

                    # Skip if this is not a polypeptide. Otherwise, add the relevant
                    # indices to the list
                    if seqind_key in self.asym_seq_to_seq_ind:
                        expanded_inds.extend(self.asym_seq_to_seq_ind[seqind_key])

                # If we have expanded indices, add them to the list of range indices
                if len(expanded_inds) > 0:
                    stacked_expanded = np.array(expanded_inds)
                    assert len(np.unique(stacked_expanded[:, -1])) == 1
                    range_inds.append(stacked_expanded)

            # Add the ladder indices if we have both ranges
            assert len(range_inds) <= 4
            if len(range_inds) == 4:
                ladder_inds.append(range_inds)

        # Create groups of indices by sheet
        sheet_inds = {}
        for infolist_ind, infolist in enumerate(ladder_inds):
            sheet_id = infolist[0]
            if sheet_id not in sheet_inds:
                sheet_inds[sheet_id] = []
            sheet_inds[sheet_id].append(infolist_ind)

        # Convert the ladder indices to a tuple and remove info on the sheet
        ladder_inds = tuple(tuple(item[1:]) for item in ladder_inds)

        # Convert sheet indices to a tuple
        sheet_inds = {key: tuple(value) for key, value in sheet_inds.items()}

        return ladder_inds, sheet_inds

    def _gather_residue_data(self) -> ResidueData:
        """Isolates information on each residue in `self.pdbx_poly_seq_scheme_data`.

        Returns:
            ResidueData: A dictionary where keys are model numbers and values are
                dictionaries with the below fields:

                    (1) "coordinates": A 3D array of shape (n_residues, n_atoms, 3).
                        This gives the coordinates of each `self.target_atoms`
                        atom in the residue.
                    (2) "seqinds": A 2D array of shape (n_residues, 3). This gives
                        the sequence indices of each residue in the full PDB sequence
                        (column 1), the chain sequence (column 2), and the chain
                        (column 3).
                    (3) "canonical_seq": The one letter code for the chains in the
                        full PDB sequence. Any noncanonical residues are replaced
                        with "X".
                    (4) "noncanonical_seq": The one letter code for the chains in
                        the full PDB sequence. Any noncanonical residues are mapped
                        to their parent canonical residue.
                    (5) "iso_b_factors": A 2D array of shape (n_residues, n_atoms).
                        This gives the isotropic B factors for each atom in
                        `self.target_atoms` in the residue.
                    (6) "aniso_u_factors": A 3D array of shape (n_residues, n_atoms, 6).
                        This gives the anisotropic B factors for each atom in
                        `self.target_atoms` in the residue. The last six dimensions
                        give U[11], U[22], U[33], U[12], U[13], and U[23], respectively.

                Note that for "coordinates", "iso_b_factors", and "aniso_u_factors",
                missing atoms are indicated by a value of np.nan.
        """
        # Get the expected length of the sequence and the maximum atom index
        seqlen = len(self.pdbx_poly_seq_scheme_data)
        max_atom_ind = self.n_target_atoms - 1

        # Create a cycling iterator over the target atoms
        target_atom_iter = itertools.cycle(enumerate(self.target_atoms))

        # We need to process every model and every chain in the file
        residue_data = {}
        for model_num, model_atom_data in self._generate_model_atomic_data():
            # Set up arrays for storing results
            coordinates = np.full(
                (seqlen, self.n_target_atoms, 3), np.inf, dtype=np.float32
            )
            seqinds = np.full((seqlen, 3), -1, dtype=np.int64)
            canonical_seq = [None] * seqlen
            noncanonical_seq = [None] * seqlen
            iso_b_factors = np.full(
                (seqlen, self.n_target_atoms), np.inf, dtype=np.float32
            )
            aniso_u_factors = np.full(
                (seqlen, self.n_target_atoms, 6), np.inf, dtype=np.float32
            )

            # Get the ID of the model according to BioPython
            biopython_model_id = self.model_num_to_ind[model_num]

            # Loop over the atom data
            residue_ind = 0
            previous_info = None
            for _, row in model_atom_data.iterrows():
                # Within a residue, all non-atom data should be the same
                resi_constant_info = (
                    row["_pdbx_poly_seq_scheme.asym_id"],
                    row["_pdbx_poly_seq_scheme.entity_id"],
                    row["_pdbx_poly_seq_scheme.seq_id"],
                    row["_pdbx_poly_seq_scheme.mon_id"],
                    row["_pdbx_poly_seq_scheme.pdb_seq_num"],
                    row["_pdbx_poly_seq_scheme.pdb_strand_id"],
                    row["_pdbx_poly_seq_scheme.pdb_ins_code"],
                    row["_pdbx_poly_seq_scheme.auth_seq_num"],
                    row["noncanon_one_letter"],
                    row["canon_one_letter"],
                    row["CATH_ids"],
                    row["chain_seq_inds"],
                    row["pdb_seq_inds"],
                    row["chain_inds"],
                )
                if previous_info is None:
                    previous_info = resi_constant_info
                else:
                    assert previous_info == resi_constant_info

                # Make sure the atom matches what we expect
                atom_ind, expected_atom = next(target_atom_iter)
                expected_atom = expected_atom + row.target_atom_suffix
                assert expected_atom == row["_atom_site.label_atom_id"] or np.isnan(
                    row["_atom_site.label_atom_id"]
                )

                # Get the ID of the chain and residue according to BioPython. The
                # `expected_atom` is the atom ID for this residue.
                biopython_chain_id = row["_pdbx_poly_seq_scheme.pdb_strand_id"]
                if row["AtomResiType"] == "HETATM":
                    hetero_flag = "H_" + row["AtomCompID"]
                else:
                    hetero_flag = " "
                biopython_resi_id = (
                    hetero_flag,
                    int(row["_pdbx_poly_seq_scheme.pdb_seq_num"]),
                    row["_pdbx_poly_seq_scheme.pdb_ins_code"].replace(".", " "),
                )

                # Get the BioPython chain. If the chain is missing, then there were
                # no atoms in the chain
                if biopython_chain_id not in self.structure[biopython_model_id]:
                    biopython_chain = None
                else:
                    biopython_chain = self.structure[biopython_model_id][
                        biopython_chain_id
                    ]

                # Get the BioPython residue. If it is a missing residue, then we know
                # that the atom and all its results should be missing
                if biopython_chain is None or biopython_resi_id not in biopython_chain:
                    biopython_resi = None
                else:
                    biopython_resi = biopython_chain[biopython_resi_id]

                    # If disordered, set to the appropriate residue
                    if isinstance(biopython_resi, DisorderedResidue):
                        biopython_resi.disordered_select(
                            row["_pdbx_poly_seq_scheme.mon_id"]
                        )

                # Get the biopython atom. If it is a missing atom, then we know that
                # the atom and all its results should be missing
                if biopython_resi is None or expected_atom not in biopython_resi:
                    biopython_atom = None
                else:
                    biopython_atom = biopython_resi[expected_atom]

                    # If disordered, set to the appropriate atom
                    if (
                        biopython_atom.is_disordered()
                        and row["_atom_site.label_alt_id"] != "."
                    ):
                        biopython_atom.disordered_select(row["_atom_site.label_alt_id"])

                # Get the coordinates and b factors for the atom
                atom_coords = row[
                    [f"_atom_site.cartn_{coord}" for coord in "xyz"]
                ].to_numpy(dtype=np.float32)
                atom_iso_b_factor = float(row["_atom_site.b_iso_or_equiv"])
                atom_aniso_u_factors = row[
                    [
                        "_atom_site_anisotrop.u[1][1]",
                        "_atom_site_anisotrop.u[2][2]",
                        "_atom_site_anisotrop.u[3][3]",
                        "_atom_site_anisotrop.u[1][2]",
                        "_atom_site_anisotrop.u[1][3]",
                        "_atom_site_anisotrop.u[2][3]",
                    ]
                ].to_numpy(dtype=np.float32)

                # If there is no BioPython residue or the atom is missing, then
                # we know that the atom and all b factors should be missing
                if biopython_resi is None or biopython_atom is None:
                    assert np.all(np.isnan(atom_coords))
                    assert np.isnan(atom_iso_b_factor)
                    assert np.all(np.isnan(atom_aniso_u_factors))

                # If there is a BioPython residue and the atom is not missing, then
                # we make sure that the coordinates and isotropic b factors match
                # what we expect. The anisotropic b factors are incorrectly handled
                # by BioPython, so we ignore them.
                else:
                    assert np.array_equal(atom_coords, biopython_atom.get_coord())
                    assert atom_iso_b_factor == biopython_atom.get_bfactor()

                # If there is a BioPython residue, then we make sure that the residue
                # type matches what we expect
                if biopython_resi is not None:
                    assert (
                        biopython_resi.get_resname()
                        == row["_pdbx_poly_seq_scheme.mon_id"]
                    )

                # Store atom-level results
                coordinates[residue_ind, atom_ind] = atom_coords
                iso_b_factors[residue_ind, atom_ind] = atom_iso_b_factor
                aniso_u_factors[residue_ind, atom_ind] = atom_aniso_u_factors

                # If we are at the last atom in the residue, then we also store the residue-level
                # results
                if atom_ind == max_atom_ind:
                    # Get the sequence indices for the residue
                    seqinds[residue_ind] = (
                        row.pdb_seq_inds,
                        row.chain_seq_inds,
                        row.chain_inds,
                    )

                    # Get the canonical and noncanonical sequences
                    noncanonical_seq[residue_ind] = row.noncanon_one_letter
                    canonical_seq[residue_ind] = row.canon_one_letter

                    # Increment the residue index and reset the previous info
                    residue_ind += 1
                    previous_info = None

            # Make sure that everything was processed
            assert residue_ind == seqlen
            assert not np.any(np.isinf(coordinates))
            assert not np.any(np.isinf(iso_b_factors))
            assert not np.any(np.isinf(aniso_u_factors))
            assert np.all(seqinds >= 0)
            assert None not in canonical_seq
            assert None not in noncanonical_seq

            # No repeated pdb sequence indices
            assert len(np.unique(seqinds[:, 0])) == len(seqinds)

            # PDB sequence indices should be in order
            assert np.all(np.diff(seqinds[:, 0]) == 1)

            # Everything should be the same length
            assert (
                len(coordinates)
                == len(seqinds)
                == len(canonical_seq)
                == len(noncanonical_seq)
                == len(iso_b_factors)
                == len(aniso_u_factors)
                == seqlen
            )

            # Record data for the model
            assert model_num not in residue_data
            residue_data[model_num] = {
                "coordinates": coordinates,
                "seqinds": seqinds,
                "canonical_seq": "".join(canonical_seq),
                "noncanonical_seq": "".join(noncanonical_seq),
                "iso_b_factors": iso_b_factors,
                "aniso_u_factors": aniso_u_factors,
            }

        return residue_data

    def _generate_model_atomic_data(
        self,
    ) -> Generator[Tuple[str, pd.DataFrame], None, None]:
        """Maps atom data for a single model onto the full PDB sequence.

        Yields:
            Generator[Tuple[str, pd.DataFrame], None, None]: A copy of
                `self.peptides_with_atoms` with the atom data for the given model
                mapped onto the full PDB sequence.

        Notes:
            This function will merge a `self.atom_data` onto a derivative of
            `self.pdbx_poly_seq_scheme_data`. The merge keys for `self.atom_data`
            are `_atom_site.label_asym_id`, `_atom_site.label_seq_id`, `_atom_site.label_comp_id`,
            and `_atom_site.label_atom_id`. The merge keys for the derivative of
            `self.pdbx_poly_seq_scheme_data` are `_pdbx_poly_seq_scheme.asym_id`,
            `_pdbx_poly_seq_scheme.seq_id`, and `target_atom`. `_pdbx_poly_seq_scheme.asym_id`
            points directly to `_atom_site.label_asym_id`. `_pdbx_poly_seq_scheme.seq_id`
            and `_atom_site.label_seq_id` both point to `_entity_poly_seq.num`.
            `_pdbx_poly_seq_scheme.mon_id` points to `_entity_poly_seq.mon_id` which
            in turn points to `_chem_comp.id` and `_atom_site.label_comp_id` points
            directly to `_chem_comp.id`. `target_atom` is a column that is added
            and points directly to `_atom_site.label_atom_id`. The merge should be
            one to one if there are no disordered atoms and this is a single model
            mmCIF file. Otherwise, it will be one to many.
        """
        # Loop over all models
        for model_num in self.model_num_to_ind:
            # Pull out the atom data for this model
            sliced_atom_data = self.atom_data[
                self.atom_data["_atom_site.pdbx_pdb_model_num"] == model_num
            ]

            # There must be a single model in the atom data
            assert len(sliced_atom_data["_atom_site.pdbx_pdb_model_num"].unique()) == 1

            # Now map the atom data for the target atoms onto `peptides_with_atoms`.
            # If we have any disordered atoms, then this is going to be a one to many
            # mapping. Otherwise, it will be a one to one mapping.
            validate = (
                "one_to_many"
                if (sliced_atom_data["_atom_site.label_alt_id"] != ".").any()
                else "one_to_one"
            )
            coordinates_merged = self.peptides_with_atoms.merge(
                sliced_atom_data,
                how="left",
                left_on=[
                    "_pdbx_poly_seq_scheme.asym_id",
                    "_pdbx_poly_seq_scheme.seq_id",
                    "_pdbx_poly_seq_scheme.mon_id",
                    "target_atom",
                ],
                right_on=[
                    "_atom_site.label_asym_id",
                    "_atom_site.label_seq_id",
                    "_atom_site.label_comp_id",
                    "_atom_site.label_atom_id",
                ],
                validate=validate,
            )

            # Sometimes the atoms will not match even though all other fields do.
            # This happens for nonstandard polypeptide elements that are not amino
            # acids. We need some information about these in downstream processing,
            # so we add it here.
            old_len = len(coordinates_merged)
            required_atom_data = (
                sliced_atom_data[
                    [
                        "_atom_site.label_asym_id",
                        "_atom_site.label_seq_id",
                        "_atom_site.label_comp_id",
                        "_atom_site.group_pdb",
                    ]
                ]
                .rename(
                    mapper={
                        "_atom_site.label_asym_id": "AtomLabelAsymID",
                        "_atom_site.label_seq_id": "AtomLabelSeqID",
                        "_atom_site.label_comp_id": "AtomCompID",
                        "_atom_site.group_pdb": "AtomResiType",
                    },
                    axis=1,
                )
                .drop_duplicates()
            )
            coordinates_merged = coordinates_merged.merge(
                required_atom_data,
                how="left",
                left_on=[
                    "_pdbx_poly_seq_scheme.asym_id",
                    "_pdbx_poly_seq_scheme.seq_id",
                    "_pdbx_poly_seq_scheme.mon_id",
                ],
                right_on=[
                    "AtomLabelAsymID",
                    "AtomLabelSeqID",
                    "AtomCompID",
                ],
                validate="many_to_one",
            )
            assert len(coordinates_merged) == old_len

            # The length of the new array should be greater than or equal to the length
            # of the peptides with targets. Greater than occurs if we have disordered
            # atoms.
            assert len(coordinates_merged) >= len(self.peptides_with_atoms)

            # Now that we have the merged coordinates, select the best alt locs
            yield model_num, self._select_best_alt_locs(coordinates_merged)

    def _select_best_alt_locs(self, coordinates_merged: pd.DataFrame) -> pd.DataFrame:
        """Identifies the best alt locs for disordered atoms and eliminates the
        rest. The "best" is defined as the atom with the highest occupancy.

        Args:
            coordinates_merged (pd.DataFrame): `self.peptides_with_atoms` merged
                with a slice of `self.atom_data` for a single model.

        Returns:
            pd.DataFrame: Identical to `coordinates_merged` except that wherever
                there were disordered atoms, only the best alt loc is kept.
        """
        # There should only be one model in the merged coordinates, two if one
        # of the models is nan
        unique_models = coordinates_merged["_atom_site.pdbx_pdb_model_num"].unique()
        unique_notna_models = unique_models[~np.isnan(unique_models.astype(float))]
        if len(unique_models) == 2:
            assert len(unique_notna_models) == 1
        else:
            assert len(unique_models) == 1

        # Identify disordered atoms that weren't handled as part of disordered residues.
        # These are indicated by duplicated sequence inds and target atoms
        atom_counts = coordinates_merged.groupby(
            ["pdb_seq_inds", "target_atom"],
            as_index=False,
        ).size()
        disordered_atoms = atom_counts[atom_counts["size"] > 1]

        # By default, we are taking all atoms in the peptide. We will then remove atoms
        # from this list when selecting the best alt locs.
        coordinates_merged["to_keep"] = True

        # Make sure the coordinates_merged dataframe has a unique index
        coordinates_merged.reset_index(drop=True, inplace=True)

        # Process each disordered atom.
        deselected_indices = []
        for (
            disordered_seqind,
            disordered_atom,
            expected_size,
        ) in disordered_atoms.itertuples(index=False):
            # Get the atoms for this disordered atom.
            limited_occupancy = coordinates_merged.loc[
                (coordinates_merged["pdb_seq_inds"] == disordered_seqind)
                & (coordinates_merged["target_atom"] == disordered_atom),
                "_atom_site.occupancy",
            ].astype(float)

            # Make sure we have the expected number of atoms.
            assert len(limited_occupancy) == expected_size

            # Get the indices of the atoms to discard
            best_idx = limited_occupancy.idxmax()
            to_discard = [
                ind for ind in limited_occupancy.index.to_list() if ind != best_idx
            ]
            assert len(to_discard) == expected_size - 1

            # Update the growing list of indices to discard
            deselected_indices.extend(to_discard)

        # The list of deselected indices should be unique
        assert len(deselected_indices) == len(set(deselected_indices))

        # Update the to_keep column
        coordinates_merged.loc[deselected_indices, "to_keep"] = False

        # Filter the dataset down to the selected atoms
        coordinates_merged = coordinates_merged[
            coordinates_merged["to_keep"]
        ].reset_index(drop=True)

        # The dataset should be as many times the size of `self.pdbx_poly_seq_scheme_data`
        # as there are atoms in `self.target_atoms`
        assert (
            len(coordinates_merged)
            == len(self.pdbx_poly_seq_scheme_data) * self.n_target_atoms
        )

        # There should be exactly self.n_target_atoms for each sequence index
        assert np.all(
            np.unique(coordinates_merged.pdb_seq_inds.to_numpy(), return_counts=True)[1]
            == self.n_target_atoms
        )

        return coordinates_merged

    @property
    def contains_polypeptides(self) -> bool:
        """True if there are polypeptides in the file, False otherwise."""
        return len(self.pdbx_poly_seq_scheme_data) != 0

    @property
    def n_target_atoms(self) -> int:
        """Gives the number of target atoms specified."""
        return len(self.target_atoms)

    @property
    def multi_model_file(self) -> bool:
        """True if there are multiple models in the file, False otherwise."""
        return len(self.model_num_to_ind) > 1

    @property
    def n_models(self) -> int:
        """The number of models in the file."""
        return len(self.model_num_to_ind)

    @property
    def polypeptide_asym_ids(self) -> Set[str]:
        """The asym ids belonging to polypeptides."""
        return set(self.asym_id_to_chain_id.keys())
