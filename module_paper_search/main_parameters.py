"""
Main parameters for the paper search module.

This module contains configuration parameters that are used across
multiple submodules in the paper search pipeline.
"""

import os
from pathlib import Path

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory for results
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ANSWER_MODE_DIR = os.path.join(RESULTS_DIR, "answer_mode")
RECOMMENDATION_MODE_DIR = os.path.join(RESULTS_DIR, "recommendation_mode")

# Ensure results directories exist
os.makedirs(ANSWER_MODE_DIR, exist_ok=True)
os.makedirs(RECOMMENDATION_MODE_DIR, exist_ok=True)

# Database directories
DATABASE_DIR = os.path.join(BASE_DIR, "databases")
os.makedirs(DATABASE_DIR, exist_ok=True)

# KGB agent parameters
MAX_LOOPS = 4 # Maximum number of feedback loops for the KGB agent
DEFAULT_QUERIES_ORDER = ["subtopic", "enhanced", "topic"]  # Default order of queries

# Output file paths
ANSWER_KGB_JSON_PATH = os.path.join(ANSWER_MODE_DIR, "answer_kgb.json")
ANSWER_KGB_TXT_PATH = os.path.join(ANSWER_MODE_DIR, "answer_kgb.txt")

# Chunk selection parameters
TOP_N_CHUNKS_PER_QUERY = 10  # Number of top chunks to select per query
