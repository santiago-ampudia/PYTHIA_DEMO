"""
submodule_paper_preselection/paper_preselection_parameters.py

Parameters for the paper preselection module.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASES_DIR = os.path.join(BASE_DIR, "databases")

# Ensure databases directory exists
os.makedirs(DATABASES_DIR, exist_ok=True)

# Path to the SQLite database containing arXiv metadata
DB_PATH = os.path.join(DATABASES_DIR, "arxiv_metadata.db")

# Path to save the preselected papers database
PRESELECTION_DB_PATH = os.path.join(DATABASES_DIR, "preselected_papers.db")

# Path to save the preselected papers as JSON
PRESELECTION_JSON_PATH = os.path.join(DATABASES_DIR, "preselected_papers.json")

# Maximum number of papers to preselect
MAX_PAPERS = 1000000
