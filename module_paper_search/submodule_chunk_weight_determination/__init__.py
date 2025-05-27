"""
submodule_chunk_weight_determination/__init__.py

This module implements Step 6 of the paper search pipeline: Chunk Weight Determination.
"""

from .chunk_weight_determination import run_chunk_weight_determination

__all__ = ['run_chunk_weight_determination']
