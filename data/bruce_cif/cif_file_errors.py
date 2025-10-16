"""Contains the error classes for the CIF file parser.
"""


class NoPolypeptidesError(Exception):
    """Raised when there are no polypeptides in the DSSPCIF file."""


class CATHError(Exception):
    """Base class for exceptions to do with CATH domains"""


class MissingStartsError(CATHError):
    """Raised when at least one CATH domain start is missing from the scraped data."""


class MissingEndsError(CATHError):
    """Raised when at least one CATH domain end is missing from the scraped data."""


class EndBeforeStartError(CATHError):
    """Raised when a CATH domain end is before the start."""
