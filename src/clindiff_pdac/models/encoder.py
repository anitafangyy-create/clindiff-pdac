"""
Compatibility wrapper for temporal encoders.

The main implementation lives in ``clindiff_pdac.models``. This module keeps
older import paths working for tests and downstream callers.
"""

from . import PositionalEncoding, TemporalConvNet, TemporalEncoder, Time2Vec

__all__ = [
    "PositionalEncoding",
    "TemporalConvNet",
    "TemporalEncoder",
    "Time2Vec",
]
