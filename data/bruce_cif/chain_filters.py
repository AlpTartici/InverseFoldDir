"""Contains functions for filtering outputs from the helper functions of
`.cif_file.CIFFile`"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
from custom_types import (
    ChainData,
    ChainMetadata,
    ChainMetadataValues,
    ConnectionInfo,
    DSSPLadderInfo,
    DSSPSecondaryStructure,
    DSSPSheetInfo,
    ImportantSiteInfo,
    ModresOutput,
    ResidueData,
    StructureData,
    UpdatedConnectionInfo,
    UpdatedDSSPLadderInfo,
    UpdatedDSSPSecondaryStructure,
    UpdatedImportantSiteInfo,
    UpdatedModresOutput,
)


def _filter_important_sites(
    target_chain_ind: int, site_info: ImportantSiteInfo
) -> UpdatedImportantSiteInfo:
    """Filters the sites of interest to only those that are in the target chain.

    Args:
        target_chain_ind (int): The index of the target chain.
        site_info (ImportantSiteInfo): The output from
            `.cif_file.CIFFile._gather_important_sites`.

    Returns:
        UpdatedImportantSiteInfo: `site_info`, but only with sites that are
            in the target chain. Note that even if a site is made up from multiple
            chains, it will still be included here so long as at least one of
            those chains is the target chain. The site will not, however, include
            indices that point to residues in chains other than the target chain.
            The fields `pdb_seq_inds` and `chain_inds` have also been removed
            as they are no longer relevant.
    """
    # Extract important sites in the target chain
    updated_info = {}
    for site_id, site_data in site_info.items():
        # Note which positions pertain to the target chain
        chain_mask = site_data["chain_inds"] == target_chain_ind

        # If there are no positions in the target chain, skip this site
        if not np.any(chain_mask):
            continue

        # Otherwise, extract the relevant data
        updated_info[site_id] = {
            "chain_seq_inds": site_data["chain_seq_inds"][chain_mask],
            "desc": site_data["desc"],
            "bound_molecules": site_data["bound_molecules"],
        }

    return updated_info


def _filter_connections(
    target_chain_ind: int, connection_info: ConnectionInfo
) -> UpdatedConnectionInfo:
    """Updates the output of `.cif_file.CIFFile._gather_connections` to only
        include connections that involve the target chain.

    Args:
        target_chain_ind (int): The index of the target chain.
        connection_info (ConnectionInfo): The output from
            `.cif_file.CIFFile._gather_connections`.

    Returns:
        UpdatedConnectionInfo: `connection_info` but only with connections that
            involve the target chain. The fields `pdb_seq_inds` and `chain_inds`
            have also been removed as they are no longer relevant.
    """
    # Process each connection
    updated_connections = {}
    for connection_ind, connection_values in connection_info.items():
        # Connections must be between members of the same chain to count
        if not np.all(connection_values["chain_inds"] == target_chain_ind):
            continue

        # Update the connection info
        updated_connections[connection_ind] = {
            "type": connection_values["type"],
            "chain_seq_inds": connection_values["chain_seq_inds"],
            "compound": connection_values["compound"],
        }

    return updated_connections


def _filter_chain_metadata(
    target_chain_ind: int, chain_metadata: ChainMetadata
) -> ChainMetadataValues:
    """Returns the chain metadata for the target chain.

    Args:
        target_chain_ind (int): The index of the target chain.
        chain_metadata (ChainMetadata): The output from
            `.cif_file.CIFFile._gather_chain_metadata`.

    Returns:
        ChainMetadataValues: Identical to the single value of `chain_metadata`
            pertaining to the target chain.
    """
    return chain_metadata[target_chain_ind]


def _filter_modres_data(
    target_chain_ind: int, modres_data: ModresOutput
) -> UpdatedModresOutput:
    """Extracts modified residue data for the target chain.

    Args:
        target_chain_ind (int): The index of the target chain.
        modres_data (ModresOutput): The output from
            `.cif_file.CIFFile._gather_modres_info`.

    Returns:
        UpdatedModresOutput: Identical to the output from
            `.cif_file.CIFFile._gather_modres_info` but only containing modified
            residues that are part of the target chain. The fields `pdb_seq_ind`
            and `chain_ind` have also been removed as they are no longer relevant.
    """
    return tuple(
        {
            "chain_seq_ind": old_val["chain_seq_ind"],
            "residue_name": old_val["residue_name"],
            "formula": old_val["formula"],
            "weight": old_val["weight"],
            "synonyms": old_val["synonyms"],
            "details": old_val["details"],
        }
        for old_val in modres_data
        if old_val["chain_ind"] == target_chain_ind
    )


def _filter_ss_data(
    target_chain_ind: int, ss_data: DSSPSecondaryStructure
) -> UpdatedDSSPSecondaryStructure:
    """Returns the secondary structure data for the target chain.

    Args:
        target_chain_ind (int): The index of the target chain.
        ss_data (DSSPSecondaryStructure): The output from
            `.cif_file.CIFFile._gather_secondary_structure`.

    Returns:
        UpdatedDSSPSecondaryStructure: Identical to the output from
            `.cif_file.CIFFile._gather_secondary_structure` but only containing
            secondary structure elements that are part of the target chain.
            The fields `pdb_seq_inds` and `chain_inds` have also been removed
            as they are no longer relevant.
    """
    # Filter out secondary structure elements that are not in the target chain
    filtered_ss_data = {}
    for ss_ind, ss_values in ss_data.items():
        # The full element must belong to one chain
        assert len(chain_ind := np.unique(ss_values["chain_inds"])) == 1
        chain_ind = chain_ind.item()

        # If the element is not in the target chain, continue
        if chain_ind != target_chain_ind:
            continue

        # Otherwise, keep the element
        filtered_ss_data[ss_ind] = {
            "chain_seq_inds": ss_values["chain_seq_inds"],
            "ss_type": ss_values["ss_type"],
            "details": ss_values["details"],
        }

    return filtered_ss_data


def _filter_bridge_pairs(
    target_chain_ind: int,
    bp_data: Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]],
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Returns the bridge pair data for the target chain.

    Args:
        target_chain_ind (int): The index of the target chain.
        bp_data (Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]): The
            output from `.cif_file.CIFFile._gather_bridge_pairs`.

    Returns:
        Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]: Similar to
            the output from `.cif_file.CIFFile._gather_bridge_pairs` but only
            containing bridge pairs that are part of the target chain. The
            first array is now a 2D array that gives just the sequence indices
            of the reference and partner residues in the bridge pair followed by
            the type code indicating whether the partner residue is a donor or
            acceptor. The second array is a 1D array that gives the energy of
            each bridge pair.
    """
    # Split apart data.
    bp_inds, bp_energies = bp_data
    assert len(bp_inds) == len(bp_energies)

    # Just return empty arrays if there are no bridge pairs
    if len(bp_inds) == 0:
        return bp_inds, bp_energies

    # Both the reference and partner must be from the target chain
    bp_mask = (bp_inds[:, 2] == target_chain_ind) & (bp_inds[:, 5] == target_chain_ind)

    # Return the filtered data
    return bp_inds[bp_mask][:, [1, 4, 6]], bp_energies[bp_mask]


def _filter_ladder_sheet_data(
    target_chain_ind: int, ls_data: Tuple[DSSPLadderInfo, DSSPSheetInfo]
) -> Tuple[UpdatedDSSPLadderInfo, DSSPSheetInfo]:
    """Filters the ladder and sheet data to just include those ladders and sheets
    that are part of the target chain. Note that so long as one strand remains
    in a ladder, the ladder will be kept; so long as one ladder remains in a
    sheet, the sheet will be kept.

    Args:
        target_chain_ind (int): The index of the target chain.
        ls_data (Tuple[DSSPLadderInfo, DSSPSheetInfo]): The output from
            `.cif_file.CIFFile._gather_ladders_sheets`.

    Returns:
        Tuple[UpdatedDSSPLadderInfo, DSSPSheetInfo]: Identical to the output from
            `.cif_file.CIFFile._gather_ladders_sheets` but only containing ladders
            and sheets that are part of the target chain. Note also that, if
            any ladders were eliminated during filtering, the indices of the
            sheet output have been updated to map to the indices of the ladders
            in the first output.
    """
    # Split apart ladder and sheet info
    ladder_data, sheet_data = ls_data

    # Create containers for storing which ladders are still around and for storing the
    # filtered ladder and sheet data
    remaining_ladders = []
    filtered_ladder_data = []
    filtered_sheet_data = {}

    # Process each ladder
    for ladder_ind, (directionality, strand1_inds, strand2_inds) in enumerate(
        ladder_data
    ):
        # A list for just this ladder
        temp_ladder_data = [directionality]

        # There should be just one chain per strand
        assert len(strand1_chain_id := np.unique(strand1_inds[:, -1])) == 1
        assert len(strand2_chain_id := np.unique(strand2_inds[:, -1])) == 1
        strand1_chain_id = strand1_chain_id.item()
        strand2_chain_id = strand2_chain_id.item()

        # If the strand is for the target chain, record
        if strand1_chain_id == target_chain_ind:
            temp_ladder_data.append(strand1_inds[:, 1])
        if strand2_chain_id == target_chain_ind:
            temp_ladder_data.append(strand2_inds[:, 1])

        # If we found at least one strand for the target chain, record the ladder
        if len(temp_ladder_data) > 1:
            remaining_ladders.append(ladder_ind)
            filtered_ladder_data.append(tuple(temp_ladder_data))

    # Map from old ladder ind to new ladder ind
    ladder_ind_map = {
        old_ind: new_ind for new_ind, old_ind in enumerate(remaining_ladders)
    }

    # Filter out sheets that are no longer connected to a ladder
    for sheet_id, ladder_inds in sheet_data.items():
        # Map the ladder inds to the new ladder inds
        new_ladder_inds = tuple(
            ladder_ind_map[ladder_ind]
            for ladder_ind in ladder_inds
            if ladder_ind in ladder_ind_map
        )

        # If there are still ladders connected to this sheet, record
        if len(new_ladder_inds) > 0:
            filtered_sheet_data[sheet_id] = new_ladder_inds

    return tuple(filtered_ladder_data), filtered_sheet_data


def _filter_residue_data(
    target_chain_ind: int, residue_data: ResidueData
) -> ResidueData:
    """Takes the output of `.cif_file.CIFFile._gather_residue_data` and filters
    it to only include residues that are part of the target chain.

    Args:
        target_chain_ind (int): The index of the target chain.
        residue_data (ResidueData): The output of `.cif_file.CIFFile._gather_residue_data`.

    Returns:
        ResidueData: Identical to `residue_data` but only containing residues that
            are part of the target chain. Note that the "seqinds" field has been
            reduced to a 1D array of chain sequence indices--the pdb indices and
            chain indices are no longer relevant and have been removed.
    """
    # Process all models and isolate the target chains
    filtered_residue_data = {}
    for model_num, model_data in residue_data.items():
        # Get a mask that is True for all residues in the target chain
        chain_mask = model_data["seqinds"][:, 2] == target_chain_ind

        # Make sure everything has the same length
        expected_len = len(chain_mask)
        assert all(len(val) == expected_len for val in model_data.values())

        # Get the updated sequences and CATH indices
        canonical_seq = "".join(
            aa for mask, aa in zip(chain_mask, model_data["canonical_seq"]) if mask
        )
        noncanonical_seq = "".join(
            aa for mask, aa in zip(chain_mask, model_data["noncanonical_seq"]) if mask
        )

        # Filter the residue data to only include the target chain
        filtered_residue_data[model_num] = {
            "coordinates": model_data["coordinates"][chain_mask],
            "seqinds": model_data["seqinds"][chain_mask, 1],
            "canonical_seq": canonical_seq,
            "noncanonical_seq": noncanonical_seq,
            "iso_b_factors": model_data["iso_b_factors"][chain_mask],
            "aniso_u_factors": model_data["aniso_u_factors"][chain_mask],
        }

    return filtered_residue_data


def _filter_null(
    target_chain_ind: int, data: object  # pylint: disable=W0613
) -> object:
    """A function that does nothing to the data but matches the signature of the
    other filtering functions.

    Args:
        target_chain_ind (int): The index of the target chain.
        data (object): The data to be "filtered".

    Returns:
        object: `data` unchanged.
    """
    return data


# Define a dictionary that maps from the name of a data field to one of the
# filtering functions defined above
_FIELD_TO_FILTER = {
    "site_data": _filter_important_sites,
    "connections_data": _filter_connections,
    "chain_metadata": _filter_chain_metadata,
    "structure_metadata": _filter_null,
    "citations": _filter_null,
    "modified_residue_data": _filter_modres_data,
    "secondary_structure_data": _filter_ss_data,
    "bridge_pair_data": _filter_bridge_pairs,
    "ladder_sheet_data": _filter_ladder_sheet_data,
    "residue_data": _filter_residue_data,
}


def _filter_all(target_chain_ind: int, structure_data: StructureData) -> ChainData:
    """Applies all of the filtering functions defined above to the output of
    `.cif_file.CIFFile.scrape_structure_data`.

    Args:
        target_chain_ind (int): The index of the target chain.
        structure_data (StructureData): The output of
            `.cif_file.CIFFile.scrape_structure_data`.

    Returns:
        ChainData: `structure_data` but with all of the filtering functions
            applied. See the documentation for each filtering function for
            details.
    """
    return {
        field: _FIELD_TO_FILTER[field](target_chain_ind, data)
        for field, data in structure_data.items()
    }


def _get_cath_site_data(
    in_domain_inds: npt.NDArray[np.int64], site_data: UpdatedImportantSiteInfo
) -> UpdatedImportantSiteInfo:
    """Isolates those sites that are in a CATH domain.

    Args:
        in_domain_inds (npt.NDArray[np.int64]): The inds in the original chain
            data that are still present in the CATH domain.
        site_data (UpdatedImportantSiteInfo): The output from
            `_filter_important_sites`.

    Returns:
        UpdatedImportantSiteInfo: Identical to `site_data`, only with all sites
            that no longer have at least one residue remaining removed.
    """
    # Process the site data
    cath_site_data = {}
    for site_id, site_data_entry in site_data.items():
        # Determine which indices, if any, are still present in the CATH domain
        remaining_inds = site_data_entry["chain_seq_inds"][
            np.isin(site_data_entry["chain_seq_inds"], in_domain_inds)
        ]

        # Record if there are any remaining indices
        if len(remaining_inds) > 0:
            cath_site_data[site_id] = {
                "chain_seq_inds": remaining_inds,
                "desc": site_data_entry["desc"],
                "bound_molecules": site_data_entry["bound_molecules"],
            }

    return cath_site_data


def _get_cath_connections_data(
    in_domain_inds: npt.NDArray[np.int64], connections_data: UpdatedConnectionInfo
) -> UpdatedConnectionInfo:
    """Isolates connection data to those sites that are in a CATH domain.

    Args:
        in_domain_inds (npt.NDArray[np.int64]): The inds in the original chain
            data that are still present in the CATH domain.
        connections_data (UpdatedConnectionInfo): The output from
            `_filter_connections`.

    Returns:
        UpdatedConnectionInfo: Identical to `connection_data`, only with all
            connections that no longer have at least one residue remaining
            removed.
    """
    # Process the connections data
    cath_connections_data = {}
    for connection_id, connection_entry in connections_data.items():
        # Determine which indices, if any, are still present in the CATH domain
        remaining_inds = connection_entry["chain_seq_inds"][
            np.isin(connection_entry["chain_seq_inds"], in_domain_inds)
        ]

        # Record if there are any remaining indices
        if len(remaining_inds) > 0:
            cath_connections_data[connection_id] = {
                "type": connection_entry["type"],
                "chain_seq_inds": remaining_inds,
                "compound": connection_entry["compound"],
            }

    return cath_connections_data


def _get_cath_secondary_structure(
    in_domain_inds: npt.NDArray[np.int64], ss_data: UpdatedDSSPSecondaryStructure
) -> UpdatedDSSPSecondaryStructure:
    """Isolates secondary structure elements specific to the CATH domain of interest.

    Args:
        in_domain_inds (npt.NDArray[np.int64]): The inds in the original chain
            data that are still present in the CATH domain.
        ss_data (UpdatedDSSPSecondaryStructure): The output from `_filter_ss_data`.

    Returns:
        UpdatedDSSPSecondaryStructure: Identical to `ss_data`, only with all
            secondary structure elements that no longer have at least one residue
            remaining removed. Note that the `chain_seq_inds` field has been
            updated to only include indices that are still present in the CATH
            domain.
    """
    # Process the secondary structure data
    cath_ss_data = {}
    for data_key, data_entry in ss_data.items():
        # Determine the remaining indices
        remaining_inds = data_entry["chain_seq_inds"][
            np.isin(data_entry["chain_seq_inds"], in_domain_inds)
        ]

        # Record if there are any remaining indices
        if len(remaining_inds) > 0:
            cath_ss_data[data_key] = {
                "chain_seq_inds": remaining_inds,
                "ss_type": data_entry["ss_type"],
                "details": data_entry["details"],
            }

    return cath_ss_data


def _get_cath_bridge_pairs(
    in_domain_inds: npt.NDArray[np.int64],
    bp_data: Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]],
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Filters the output of `_filter_bridge_pairs` to only include bridge pairs
    that are part of the CATH domain.

    Args:
        in_domain_inds (npt.NDArray[np.int64]): The inds in the original chain
            data that are still present in the CATH domain.
        bp_data (Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]): The output
            from `_filter_bridge_pairs`.

    Returns:
        Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]: Identical to `bp_data`
            but only containing bridge pairs that are part of the CATH domain. This
            means that only those bridge pairs that have both residues in the CATH
            domain are kept.
    """
    # If there are no bridge pairs, just return empty arrays
    if len(bp_data[0]) == 0:
        assert len(bp_data[1]) == 0
        return bp_data

    # Identify the bridge pairs that have both indices in the domain
    both_in_domain_mask = np.all(np.isin(bp_data[0][:, [0, 1]], in_domain_inds), axis=1)

    # Return filtered data
    return bp_data[0][both_in_domain_mask], bp_data[1][both_in_domain_mask]


def _get_cath_ladder_sheet_data(
    in_domain_inds: npt.NDArray[np.int64],
    ls_data: Tuple[UpdatedDSSPLadderInfo, DSSPSheetInfo],
) -> Tuple[UpdatedDSSPLadderInfo, DSSPSheetInfo]:
    """Filters the output of `_filter_ladder_sheet_data` to only include ladders
    and sheets that are part of the CATH domain.

    Args:
        in_domain_inds (npt.NDArray[np.int64]): The inds in the original chain
            data that are still present in the CATH domain.
        ls_data (Tuple[UpdatedDSSPLadderInfo, DSSPSheetInfo]): The output from
            `_filter_ladder_sheet_data`.

    Returns:
        Tuple[UpdatedDSSPLadderInfo, DSSPSheetInfo]: Identical to `ls_data`, but
            only ladders that have at least one strand remaining and only sheets
            that have at least one ladder remaining are kept.
    """
    # Split apart the two data types
    ladder_data, sheet_data = ls_data

    # Determine which ladder elements are in the target domain
    cath_ladder_data = []
    old_inds_to_new_inds = {}
    ladder_counter = 0
    for old_ladder_ind, ladder_data in enumerate(ladder_data):
        # Determine which indices are still in the target domain
        remaining_inds = [
            indarray[np.isin(indarray, in_domain_inds)] for indarray in ladder_data[1:]
        ]
        remaining_inds = [indarray for indarray in remaining_inds if len(indarray) > 0]

        # Keep only the ladder elements that still have at least one residue in the
        # target domain
        if len(remaining_inds) > 0:
            # Record data
            cath_ladder_data.append(tuple([ladder_data[0]] + remaining_inds))

            # Note the conversion between old and new inds and update the new ind
            # counter
            old_inds_to_new_inds[old_ladder_ind] = ladder_counter
            ladder_counter += 1

    # Determine which sheets are being kept
    cath_sheet_data = {}
    for sheet_id, ladder_inds in sheet_data.items():
        # Determine which sheets still have at least one ladder element in the target
        # domain
        new_ladder_inds = tuple(
            old_inds_to_new_inds[ind]
            for ind in ladder_inds
            if ind in old_inds_to_new_inds
        )

        # If we have at least one ladder element in the target domain, keep the
        # sheet
        if len(new_ladder_inds) > 0:
            cath_sheet_data[sheet_id] = new_ladder_inds

    return tuple(cath_ladder_data), cath_sheet_data


def _get_cath_residue_data(
    in_domain_mask: npt.NDArray[np.bool_], residue_data: ResidueData
) -> ResidueData:
    """Converts "coordinates", "iso_b_factors", and "aniso_u_factors" to np.nan
    wherever the residue is not in the CATH domain for the output from
    `_filter_residue_data`. All other entries are left unchanged.

    Args:
        in_domain_mask (npt.NDArray[np.bool_]): A boolean mask that is True for
            positions that are in the CATH domain and False for positions that
            are not.
        residue_data (ResidueData): The output from `_filter_residue_data`.

    Returns:
        ResidueData: The same as `residue_data`, but with the "coordinates",
            "iso_b_factors", and "aniso_u_factors" fields converted to np.nan
            wherever the residue is not in the CATH domain.
    """
    # Get the out of domain mask
    out_of_domain_mask = np.logical_not(in_domain_mask)

    # Process all models and all residue data
    cath_residue_data = {}
    for model_id, model_res_data in residue_data.items():
        # Build a new dictionary for the model
        cath_residue_data_model = {}

        # Get the expected length
        expected_length = len(model_res_data["seqinds"])

        # Wherever we have a missing residue, we fill it with np.nan
        for field in ("coordinates", "iso_b_factors", "aniso_u_factors"):
            # Get a copy of the data
            data = model_res_data[field].copy()

            # Fill the missing data with np.nan
            data[out_of_domain_mask] = np.nan

            # Save the data
            cath_residue_data_model[field] = data

            # Make sure the length is correct
            assert len(data) == expected_length

        # Add the other data to the dictionary
        for field in ("seqinds", "canonical_seq", "noncanonical_seq"):
            cath_residue_data_model[field] = model_res_data[field]
            assert len(model_res_data[field]) == expected_length

        # Sequence indices should be sorted
        assert np.all(np.diff(cath_residue_data_model["seqinds"]) == 1)

        # No repeated indices
        assert len(np.unique(cath_residue_data_model["seqinds"])) == expected_length

        # Save the data
        cath_residue_data[model_id] = cath_residue_data_model

    return cath_residue_data
