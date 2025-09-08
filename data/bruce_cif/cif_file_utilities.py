"""Holds utility functions for the `.cif_file` module"""

import logging

from typing import Dict, List, Sequence

import pandas as pd

from Bio import SeqIO

from data.bruce_cif.cath_record import CATHRecord
from custom_types import CitationInfo


def _missing_to_empty_str(val: str) -> str:
    """Converts a missing value to an empty string and eliminates newlines.

    Args:
        val (str): The value to convert.

    Returns:
        str: The converted value.
    """
    if val == "?" or val == ".":
        return ""
    else:
        return val.replace("\n", "")


def _select_opt(options: Sequence[str]) -> str:
    """Given a list of options from a series of keys in a cif file, returns the
    non-empty option. If there are multiple non-empty options, then they must
    all be the same. Otherwise, we raise a warning and return an empty string, as
    we cannot determine which option is correct.

    Args:
        options (Sequence[str]): The options to select from.

    Returns:
        str: The selected option.
    """
    # Get options that are not empty strings or placeholders
    options = [opt for opt in options if opt not in {"", "?", "."}]

    # If there are more than one non-empty options, then we make sure they are
    # all the same. Otherwise, we raise a warning and return an empty string
    if len(options) > 1:
        if not all(opt == options[0] for opt in options):
            logging.warning("Multiple non-empty options found: %s", options)
            return ""

    # If there are no options, then we return an empty string
    elif len(options) == 0:
        return ""

    # Return the singular option
    return _missing_to_empty_str(options[0])


def _process_citation_row(row: pd.Series) -> CitationInfo:
    """Converts a row of a citation table to a dictionary of citation information.
    The main purpose of this function is to (1) convert missing values to empty
    strings, (2) eliminate newlines, and (3) convert mmcif keys to more human-readable
    keys.

    Args:
        row (pd.Series): A row from the pandas DataFrame defined using the `CITATION_ENTRIES`
            set of keys.

    Returns:
        CitationInfo: The citation information.
    """
    return {
        "title": _missing_to_empty_str(row["_citation.title"]),
        "abstract": _missing_to_empty_str(row["_citation.abstract"]),
        "pubmed_id": _missing_to_empty_str(row["_citation.pdbx_database_id_pubmed"]),
        "doi": _missing_to_empty_str(row["_citation.pdbx_database_id_doi"]),
        "journal_astm": _missing_to_empty_str(row["_citation.journal_id_astm"]),
        "journal_issn": _missing_to_empty_str(row["_citation.journal_id_issn"]),
        "book_isbn": _missing_to_empty_str(row["_citation.book_id_isbn"]),
    }


def build_pdb_to_cath(
    domain_seq_fasta: str,
) -> Dict[str, List[CATHRecord]]:
    """Builds a dictionary that links a PDB ID to a list of SeqRecord instances
    containing information about the CATH domains associated with that PDB structure.

    Args:
        domain_seq_fasta (str): The path to the fasta file that contains the CATH
            domain sequences. This is "cath-domain-seqs.fa" in the CATH download.

    Returns:
        Dict[str, List[CATHRecord]]: A dictionary that links a PDB ID to a list
            of CATHRecord instances containing information about the CATH domains
            associated with that PDB structure.
    """
    # Set up the dictionary
    pdb_to_cath = {}
    for cath_seq in SeqIO.parse(domain_seq_fasta, "fasta"):
        # Build the CATH record
        cath_record = CATHRecord(cath_seq)

        # Add the record to the dictionary
        if cath_record.parent_pdb not in pdb_to_cath:
            pdb_to_cath[cath_record.parent_pdb] = []
        pdb_to_cath[cath_record.parent_pdb].append(cath_record)

    return pdb_to_cath