"""Defines custom types for the PDB data scraping module.
"""

from typing import Dict, Tuple, TypedDict, Union

import numpy as np
import numpy.typing as npt


# Base type definitions
class PdbAndChainInds(TypedDict):
    """Type definition for the pdb and chain indices."""

    pdb_seq_inds: npt.NDArray[np.int64]
    chain_inds: npt.NDArray[np.int64]


# Citations
class CitationInfo(TypedDict):
    """Type definition for the citation information."""

    title: str
    abstract: str
    pubmed_id: str
    doi: str
    journal_astm: str
    journal_issn: str
    book_isbn: str


CitationInfoOutput = Dict[
    str, CitationInfo
]  # output of `CIFFile._gather_citation_info`


# Compounds
class CompoundInfo(TypedDict):
    """Type definition for compound information"""

    name: str
    formula: str
    weight: str
    synonyms: str


# Connections
class UpdatedConnectionValues(TypedDict):
    """Type definition for the output of `CIFFile._filter_connections`"""

    type: str
    chain_seq_inds: npt.NDArray[np.int64]
    compound: Union[None, CompoundInfo]


class ConnectionValues(UpdatedConnectionValues, PdbAndChainInds):
    """Type definition for the values in the output of `CIFFile._gather_connections`"""


ConnectionInfo = Dict[int, ConnectionValues]
UpdatedConnectionInfo = Dict[int, UpdatedConnectionValues]


# Important sites
class ChainImportantSiteInfoValues(TypedDict):
    """Type definition for the values of the dictionary output by
    `CIFFIle._filter_important_sites`."""

    chain_seq_inds: npt.NDArray[np.int64]
    desc: str
    bound_molecules: Tuple[CompoundInfo, ...]


class ImportantSiteInfoValues(ChainImportantSiteInfoValues, PdbAndChainInds):
    """Type definition for the values of the dictionary output by
    `CIFFIle._gather_important_sites`."""


ImportantSiteInfo = Dict[str, ImportantSiteInfoValues]
UpdatedImportantSiteInfo = Dict[str, ChainImportantSiteInfoValues]

# Ladder and sheet
DSSPLadderInfo = Tuple[Tuple[int, npt.NDArray[np.int64], npt.NDArray[np.int64]], ...]
DSSPSheetInfo = Dict[str, Tuple[int, ...]]
UpdatedDSSPLadderInfo = Tuple[
    Union[
        Tuple[int, npt.NDArray[np.int64], npt.NDArray[np.int64]],
        Tuple[int, npt.NDArray[np.int64]],
    ],
    ...,
]


# Metadata
class ChainMetadataEntry(TypedDict):
    """Type definition for the output of `CIFFile._gather_chain_metadata`"""

    src_host: str
    src_tax_id: str
    src_organ: str
    src_tissue: str
    src_cell: str
    src_organelle: str
    src_cellular_location: str
    exp_name: str
    exp_organ: str
    exp_tissue: str
    exp_cell: str
    exp_organelle: str
    exp_cellular_location: str
    ec_number: str
    description: str
    details: str


ChainMetadataValues = Tuple[ChainMetadataEntry, ...]
ChainMetadata = Dict[int, ChainMetadataValues]


class StructureMetadata(TypedDict):
    """Type definition for the values output by `CIFFile._gather_structure_metadata`"""

    title: str
    resolution: Union[float, None]
    experimental_method: str
    description: str
    keywords: str


# Modified residues
class UpdatedModresInfo(TypedDict):
    """Type definition for the values output by `CIFFile._filter_modres_info`"""

    chain_seq_ind: int
    residue_name: str
    formula: str
    weight: str
    synonyms: str
    details: str


class ModresInfo(UpdatedModresInfo):
    """Type definition for the output of `CIFFile._gather_modres_info`"""

    pdb_seq_ind: int
    chain_ind: int


ModresOutput = Union[Tuple[()], Tuple[ModresInfo, ...]]
UpdatedModresOutput = Union[Tuple[()], Tuple[UpdatedModresInfo, ...]]


# Residue
class ResidueDataValues(TypedDict):
    """Type definition for the values of the dictionary output by
    `CIFFile._gather_residue_data`."""

    coordinates: npt.NDArray[np.float32]
    seqinds: npt.NDArray[np.int64]
    canonical_seq: str
    noncanonical_seq: str
    iso_b_factors: npt.NDArray[np.float32]
    aniso_u_factors: npt.NDArray[np.float32]


ResidueData = Dict[str, ResidueDataValues]


# Secondary structure
class UpdatedSecondaryStructureValues(TypedDict):
    """Type definition for the output of `CIFFile._filter_secondary_structure`"""

    chain_seq_inds: npt.NDArray[np.int64]
    ss_type: str
    details: str


class DSSPSecondaryStructureValues(UpdatedSecondaryStructureValues, PdbAndChainInds):
    """Type definition for the values of the dictionary output by
    `CIFFile._gather_secondary_structure`."""


DSSPSecondaryStructure = Dict[str, DSSPSecondaryStructureValues]
UpdatedDSSPSecondaryStructure = Dict[str, UpdatedSecondaryStructureValues]


# Composites
class StructureData(TypedDict):
    """Type definition for the output of `CIFFile.scrape_structure_data`"""

    site_data: ImportantSiteInfo
    connections_data: ConnectionInfo
    chain_metadata: ChainMetadata
    structure_metadata: StructureMetadata
    citations: CitationInfoOutput
    modified_residue_data: ModresOutput
    secondary_structure_data: DSSPSecondaryStructure
    bridge_pair_data: Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]
    ladder_sheet_data: Tuple[DSSPLadderInfo, DSSPSheetInfo]
    residue_data: ResidueData


class ChainData(TypedDict):
    """Type definition for the output of `chain_filters._filter_all`"""

    site_data: UpdatedImportantSiteInfo
    connections_data: UpdatedConnectionInfo
    chain_metadata: ChainMetadataValues
    structure_metadata: StructureMetadata
    citations: CitationInfoOutput
    modified_residue_data: UpdatedModresOutput
    secondary_structure_data: UpdatedDSSPSecondaryStructure
    bridge_pair_data: Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]
    ladder_sheet_data: Tuple[UpdatedDSSPLadderInfo, DSSPSheetInfo]
    residue_data: ResidueData


ScrapedChainData = Dict[str, ChainData]
