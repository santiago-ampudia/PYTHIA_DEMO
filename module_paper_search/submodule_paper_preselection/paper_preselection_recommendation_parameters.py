"""
submodule_paper_preselection/paper_preselection_recommendation_parameters.py

Parameters for the paper preselection recommendation module.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASES_DIR = os.path.join(BASE_DIR, "databases")

# Path to the SQLite database containing arXiv metadata
DB_PATH = os.path.join(DATABASES_DIR, "arxiv_metadata.db")

# Path to save the preselected papers database for recommendation mode
PRESELECTION_RECOMMENDATION_DB_PATH = os.path.join(DATABASES_DIR, "preselected_papers_recommendation.db")

# Path to save the preselected papers as JSON for recommendation mode
PRESELECTION_RECOMMENDATION_JSON_PATH = os.path.join(DATABASES_DIR, "preselected_papers_recommendation.json")

# Maximum number of papers to preselect for recommendation mode
# Using a higher number for recommendation mode to ensure broader coverage
MAX_PAPERS = 1500000
