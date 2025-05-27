"""
submodule_chunk_similarity_selection/chunk_similarity_selection_parameters.py

This file contains all parameters for the chunk similarity selection submodule.
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
DETAILED_RESULTS_PATH = os.path.join(DATABASES_DIR, "detailed_chunk_results.txt")
SIMILARITY_RESULTS_PATH = os.path.join(DATABASES_DIR, "similarity_results.json")
SELECTED_CHUNKS_DB_PATH = os.path.join(DATABASES_DIR, "selected_chunks.db")

# Ensure databases directory exists
os.makedirs(DATABASES_DIR, exist_ok=True)

# Embedding model
EMBEDDING_MODEL = "intfloat/e5-small-v2"

# Similarity thresholds and limits
THRESHOLD_METADATA_SCORE_TOPIC = 0.75  # Minimum metadata similarity score to consider chunks
THRESHOLD_METADATA_SCORE_SUBTOPIC = 0.75  # Minimum metadata similarity score to consider chunks
THRESHOLD_METADATA_SCORE_ENHANCED = 0.6  # Minimum metadata similarity score to consider chunks

# Mode-specific parameters
# Answer mode parameters (default)
TOP_K_TOPIC_ANSWER = 5  # Number of top chunks to select by topic for answer mode
TOP_M_SUBTOPIC_ANSWER = 5  # Number of top chunks to select by subtopic for answer mode
TOP_N_ENHANCED_ANSWER = 5  # Number of top chunks to select by enhanced query for answer mode

# Recommendation mode parameters
TOP_K_TOPIC_RECOMMENDATION = 20  # Number of top chunks to select by topic for recommendation mode
TOP_M_SUBTOPIC_RECOMMENDATION = 20  # Number of top chunks to select by subtopic for recommendation mode
TOP_N_ENHANCED_RECOMMENDATION = 20  # Number of top chunks to select by enhanced query for recommendation mode

# Default values (for backward compatibility)
TOP_K_TOPIC = TOP_K_TOPIC_ANSWER
TOP_M_SUBTOPIC = TOP_M_SUBTOPIC_ANSWER
TOP_N_ENHANCED = TOP_N_ENHANCED_ANSWER
