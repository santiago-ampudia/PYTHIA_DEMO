"""
submodule_metadata_embedding/__init__.py

This module implements Step 1 of the paper search pipeline: embedding of metadata.
"""

from .metadata_embedding import run_metadata_embedding

__all__ = ['run_metadata_embedding']
