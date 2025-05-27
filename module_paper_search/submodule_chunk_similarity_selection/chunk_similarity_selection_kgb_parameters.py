"""
submodule_chunk_similarity_selection/chunk_similarity_selection_kgb_parameters.py

This file contains all parameters for the KGB chunk similarity selection submodule.
KGB (Keyword-Guided Batch) version supports processing multiple queries in a batch.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASES_DIR = os.path.join(BASE_DIR, "databases")
DB_PATH = os.path.join(DATABASES_DIR, "arxiv_metadata.db")
CHUNK_INDEX_DB_PATH = os.path.join(DATABASES_DIR, "chunk_index.db")
METADATA_INDEX_DB_PATH = os.path.join(DATABASES_DIR, "metadata_index.db")
METADATA_FAISS_INDEX_PATH = os.path.join(DATABASES_DIR, "metadata_embeddings.faiss")
CHUNK_FAISS_INDEX_PATH = os.path.join(DATABASES_DIR, "content_embeddings.faiss")
DETAILED_RESULTS_PATH = os.path.join(DATABASES_DIR, "detailed_chunk_results_kgb.txt")
SIMILARITY_RESULTS_PATH = os.path.join(DATABASES_DIR, "similarity_results_kgb.json")
SELECTED_CHUNKS_DB_PATH = os.path.join(DATABASES_DIR, "selected_chunks_kgb.db")

# Ensure databases directory exists
os.makedirs(DATABASES_DIR, exist_ok=True)

# Embedding model
EMBEDDING_MODEL = "intfloat/e5-small-v2"

# Similarity thresholds
THRESHOLD_METADATA_SCORE = 0.6  # Minimum metadata similarity score to consider chunks

# Mode-specific parameters
# Answer mode parameters (default)
TOP_K_CHUNKS_ANSWER = 10  # Number of top chunks to select per query for answer mode

# Recommendation mode parameters
TOP_K_CHUNKS_RECOMMENDATION = 20  # Number of top chunks to select per query for recommendation mode

# Default values (for backward compatibility)
TOP_K_CHUNKS = TOP_K_CHUNKS_ANSWER

# Query type names (for logging and database)
QUERY_TYPE_NAMES = ["subtopic", "enhanced", "topic"]  # Default names for the first three queries
