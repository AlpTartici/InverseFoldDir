"""Holds the `CATHRecord` class, which contains information on a single entry in
`cath-domain-seqs.fa`."""

import re

from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from Bio.SeqRecord import SeqRecord

from data.bruce_cif.chain_filters import (
    _get_cath_bridge_pairs,
    _get_cath_connections_data,
    _get_cath_ladder_sheet_data,
    _get_cath_residue_data,
    _get_cath_secondary_structure,
    _get_cath_site_data,
)
from cif_file_errors import (
    EndBeforeStartError,
    MissingEndsError,
    MissingStartsError,
)
from custom_types import (
    ChainData,
    ScrapedChainData,
)

# Parses the CATH domain name into its component parts
DOMAIN_NAME_REGEX = re.compile("^([a-z0-9]{4})([a-zA-Z0-9]{1})([0-9]{2})$")

# Parses a range of cath indices into its component parts
RANGE_SPLITTER_REGEX = re.compile(r"^(-?[0-9A-Za-z\(\)]+)-(-?[0-9A-Za-z\(\)]+)$")

# Regexes for splitting a cath index into its component parts
NO_INS_BOUNDARY_SPLITTER_REGEX = re.compile("^(-?[0-9]+)$")
INS_BOUNDARY_SPLITTER_REGEX = re.compile(r"^(-?[0-9]+)\(([A-Za-z0-9]+)\)$")


class CATHRecord:
    """Contains information on a single entry in `cath-domain-seqs.fa`."""

    def __init__(self, cath_seq: SeqRecord):
        """Extracts information on the target CATH domain from the sequence record.

        Args:
            cath_seq (SeqRecord): A single entry from the file "cath-domain-seqs.fa".
        """
        # Get the name of the domain and the sequence ranges included in that domain
        _, _, domain_range_info = cath_seq.id.split("|")
        domain_name, domain_ranges = domain_range_info.split("/")

        # Split the domain name into its component parts
        self.domain_name = domain_name
        self.parent_pdb, self.chain_id, self.fragment_id = DOMAIN_NAME_REGEX.match(
            domain_name
        ).groups()

        # Lowercase parent pdb
        self.parent_pdb = self.parent_pdb.lower()

        # Convert the string ranges into a list of tuples giving the start and end
        # positions of the fragments in the domain
        domain_ranges = tuple(
            RANGE_SPLITTER_REGEX.match(range_).groups()
            for range_ in domain_ranges.split("_")
        )
        assert all(len(range_) == 2 for range_ in domain_ranges)

        # Make sure that the ranges are a valid format
        assert all(
            NO_INS_BOUNDARY_SPLITTER_REGEX.match(boundary)
            or INS_BOUNDARY_SPLITTER_REGEX.match(boundary)
            for range_ in domain_ranges
            for boundary in range_
        )

        # Get all start and all end positions. They must be unique.
        self.domain_starts, self.domain_ends = zip(*domain_ranges)
        assert len(self.domain_starts) == len(set(self.domain_starts))
        assert len(self.domain_ends) == len(set(self.domain_ends))

        # Get the expected sequence for the domain
        self.domain_seq = str(cath_seq.seq)

    def _build_chain_mask(
        self, seq_data: pd.DataFrame
    ) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
        """Creates a mask that is True for all residues that are in the target
        CATH domain and False otherwise. Also returns the sequence indices of the
        residues that are in the target domain.

        Args:
            seq_data (pd.DataFrame): A slice from the `pdbx_poly_seq_scheme` table
                of `.cif_file.CIFFile` that pertains specifically to data for
                `self.chain_id`.

        Returns:
            Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]: The first element
                is a mask that is True for all residues that are in the target
                CATH domain and False otherwise. The second element is the sequence
                indices of the residues that are in the target domain.
        """
        # Get a list of the domain starts and ends. They cannot be empty.
        domain_starts = list(self.domain_starts)
        domain_ends = list(self.domain_ends)
        assert len(domain_starts) > 0, "No domain starts"
        assert len(domain_ends) > 0, "No domain ends"
        assert len(domain_starts) == len(
            domain_ends
        ), "Different number of starts and ends"

        # Get a mask that indicates which CATH indices are included in the domain
        in_domain_mask = np.zeros(len(seq_data), dtype=bool)
        recording = False
        stop_rec_ind = None
        for mask_ind, cath_ind in enumerate(seq_data.CATH_ids.tolist()):
            # If we have reached an end, stop recording if this cath_ind is different
            # than the one that we stopped recording at.
            if stop_rec_ind is not None and stop_rec_ind != cath_ind:
                assert recording
                recording = False
                stop_rec_ind = None

            # If we see the start of the domain, make sure that we are not recording,
            # remove the element from the list of starts, and then start recording
            if cath_ind in domain_starts:
                assert (
                    not recording
                ), "Found another domain range start before the first one ended"
                domain_starts.remove(cath_ind)
                recording = True

            # If we are recording, make sure that the mask is currently False, then set
            # it to True. This makes sure we don't have overlapping ranges.
            if recording:
                assert not in_domain_mask[mask_ind], "Already in domain"
                in_domain_mask[mask_ind] = True

            # If we see the end of the domain, make sure that we are recording, remove the
            # element from the list of ends, and then note that we need to stop recording
            # the next time we see a different cath ind than the end here.
            if cath_ind in domain_ends:
                if not recording:
                    raise EndBeforeStartError(
                        "The end of a domain was found before the start!"
                    )
                domain_ends.remove(cath_ind)
                stop_rec_ind = cath_ind

        # We should have exhausted the lists of starts and ends
        if len(domain_starts) != 0:
            raise MissingStartsError("Not all domain starts were found")
        if len(domain_ends) != 0:
            raise MissingEndsError("Not all domain ends were found")

        # Get the indices that are in domain
        in_domain_inds = seq_data.chain_seq_inds.to_numpy()[in_domain_mask]
        assert np.array_equal(in_domain_inds, np.nonzero(in_domain_mask)[0])

        return in_domain_mask, in_domain_inds

    def _scrape_from_chain_data(
        self,
        seq_data: pd.DataFrame,
        chain_data: ScrapedChainData,
    ) -> ChainData:
        """Given

        Args:
            seq_data (pd.DataFrame): A slice from the `pdbx_poly_seq_scheme` table
                of `.cif_file.CIFFile` that pertains specifically to data for
                `self.chain_id`.
            chain_data (ScrapedChainData): The second output of
                `.cif_file.CIFFile.scrape`. This contains the data for ALL chains
                in the PDB file.

        Returns:
            ChainData: Identical to the entry of `chain_data` pertaining to this
                CATH domain's chain, but with the data filtered to only include
                the data for the CATH domain. See the documentation of
                `.chain_filters._get_cath_bridge_pairs`,
                `.chain_filters._get_cath_site_data`,
                `.chain_filters._get_cath_connections_data`,
                `.chain_filters._get_cath_secondary_structure`,
                `.chain_filters._get_cath_ladder_sheet_data`, and
                `.chain_filters._get_cath_residue_data`
                for more details on the changed fields. The other fields are unchanged.
        """
        # Get the chain data specific to this domain
        domain_chain_data = chain_data[self.chain_id]

        # Build a mask that is True for all residues that are in the target domain
        in_domain_mask, in_domain_inds = self._build_chain_mask(seq_data)

        # Update the domain chain data to only include the data for the target domain
        return {
            "site_data": _get_cath_site_data(
                in_domain_inds, domain_chain_data["site_data"]
            ),
            "connections_data": _get_cath_connections_data(
                in_domain_inds, domain_chain_data["connections_data"]
            ),
            "chain_metadata": domain_chain_data["chain_metadata"],
            "structure_metadata": domain_chain_data["structure_metadata"],
            "citations": domain_chain_data["citations"],
            "modified_residue_data": domain_chain_data["modified_residue_data"],
            "secondary_structure_data": _get_cath_secondary_structure(
                in_domain_inds, domain_chain_data["secondary_structure_data"]
            ),
            "bridge_pair_data": _get_cath_bridge_pairs(
                in_domain_inds, domain_chain_data["bridge_pair_data"]
            ),
            "ladder_sheet_data": _get_cath_ladder_sheet_data(
                in_domain_inds, domain_chain_data["ladder_sheet_data"]
            ),
            "residue_data": _get_cath_residue_data(
                in_domain_mask, domain_chain_data["residue_data"]
            ),
        }