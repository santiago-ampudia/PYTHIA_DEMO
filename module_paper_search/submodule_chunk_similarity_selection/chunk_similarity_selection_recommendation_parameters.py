"""
submodule_chunk_similarity_selection/chunk_similarity_selection_recommendation_parameters.py

This file contains all parameters for the chunk similarity selection recommendation submodule.
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
DETAILED_RESULTS_RECOMMENDATION_PATH = os.path.join(DATABASES_DIR, "detailed_chunk_results_recommendation.txt")
SIMILARITY_RESULTS_RECOMMENDATION_PATH = os.path.join(DATABASES_DIR, "similarity_results_recommendation.json")
SELECTED_CHUNKS_RECOMMENDATION_DB_PATH = os.path.join(DATABASES_DIR, "selected_chunks_recommendation.db")

# Ensure databases directory exists
os.makedirs(DATABASES_DIR, exist_ok=True)

# Embedding model
EMBEDDING_MODEL = "intfloat/e5-small-v2"

# Similarity thresholds for recommendation mode (all set to 0.65 as requested)
THRESHOLD_METADATA_SCORE_ARCHITECTURE = 0.65  # Minimum metadata similarity score for architecture query
THRESHOLD_METADATA_SCORE_TECHNICAL = 0.65  # Minimum metadata similarity score for technical implementation query
THRESHOLD_METADATA_SCORE_ALGORITHMIC = 0.65  # Minimum metadata similarity score for algorithmic approach query
THRESHOLD_METADATA_SCORE_DOMAIN = 0.65  # Minimum metadata similarity score for domain-specific query
THRESHOLD_METADATA_SCORE_INTEGRATION = 0.65  # Minimum metadata similarity score for integration pipeline query

# Number of top chunks to select for each query (all set to 10 as requested)
TOP_K_ARCHITECTURE = 10  # Number of top chunks to select by architecture query
TOP_K_TECHNICAL = 10  # Number of top chunks to select by technical implementation query
TOP_K_ALGORITHMIC = 10  # Number of top chunks to select by algorithmic approach query
TOP_K_DOMAIN = 10  # Number of top chunks to select by domain-specific query
TOP_K_INTEGRATION = 10  # Number of top chunks to select by integration pipeline query
