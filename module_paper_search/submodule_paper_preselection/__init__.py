"""
submodule_paper_preselection/__init__.py

This module implements Step 3 and Step 4 of the paper search pipeline:
- Step 3: FAISS Semantic Search (original implementation)
- Step 4: Paper Preselection by Category (new implementation)

It exposes the main functions to run both paper preselection processes.
"""

from .category_paper_preselection import run_category_paper_preselection
from .category_paper_preselection_recommendation import run_category_paper_preselection_recommendation

__all__ = ['run_category_paper_preselection', 'run_category_paper_preselection_recommendation']
