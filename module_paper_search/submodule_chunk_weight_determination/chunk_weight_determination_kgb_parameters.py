"""
submodule_chunk_weight_determination/chunk_weight_determination_kgb_parameters.py

This file contains all parameters for the KGB chunk weight determination submodule.
KGB (Keyword-Guided Batch) version supports processing multiple queries in a batch.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASES_DIR = os.path.join(BASE_DIR, "databases")

# Input database (from KGB chunk similarity selection)
INPUT_DB_PATH = os.path.join(DATABASES_DIR, "selected_chunks_kgb.db")

# Output database
WEIGHTED_CHUNKS_DB_PATH = os.path.join(DATABASES_DIR, "weighted_chunks_kgb.db")

# Output JSON file for weighted chunks
WEIGHTED_CHUNKS_JSON_PATH = os.path.join(DATABASES_DIR, "weighted_chunks_kgb.json")

# Ensure databases directory exists
os.makedirs(DATABASES_DIR, exist_ok=True)

# Query weight parameters
# Weight for the current query (i)
CURRENT_QUERY_WEIGHT = 0.6

# Weight for the previous query (i-1)
PREVIOUS_QUERY_WEIGHT = 0.2

# Weight for the next query (i+1)
NEXT_QUERY_WEIGHT = 0.2

# Adjusted weights for edge cases
# When there's no previous query (i=0)
FIRST_QUERY_WEIGHT = 0.6
FIRST_QUERY_NEXT_WEIGHT = 0.4

# When there's no next query (i=N)
LAST_QUERY_WEIGHT = 0.6
LAST_QUERY_PREVIOUS_WEIGHT = 0.4

# Answer mode parameters
SIMILARITY_PRESELECTION_THRESHOLD = 0.5  # Minimum similarity score to consider chunks

# No top chunk preservation - only use similarity threshold

# Metadata similarity boost parameters
# Weight for metadata similarity boost (conservative value)
METADATA_BOOST_WEIGHT = 0.15

# Minimum metadata similarity threshold to apply boost
METADATA_BOOST_THRESHOLD = 0.8

# Which queries to consider for metadata boost (edge queries)
METADATA_BOOST_FIRST_QUERY = True  # Boost based on first query metadata similarity
METADATA_BOOST_LAST_QUERY = True   # Boost based on last query metadata similarity
