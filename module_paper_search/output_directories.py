"""
module_paper_search/output_directories.py

This module contains parameters for output directories used by the paper search pipeline.
"""

import os

# Base output directory
BASE_OUTPUT_DIR = "results"

# Answer mode output directory
ANSWER_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "answer_mode")

# Recommendation mode output directory
RECOMMENDATION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "recommendation_mode")

# Create output directories if they don't exist
os.makedirs(ANSWER_OUTPUT_DIR, exist_ok=True)
os.makedirs(RECOMMENDATION_OUTPUT_DIR, exist_ok=True)
