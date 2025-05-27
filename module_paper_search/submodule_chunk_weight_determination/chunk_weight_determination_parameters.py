"""
submodule_chunk_weight_determination/chunk_weight_determination_parameters.py

This file contains all parameters for the chunk weight determination submodule.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASES_DIR = os.path.join(BASE_DIR, "databases")

# Input database (from chunk similarity selection)
INPUT_DB_PATH = os.path.join(DATABASES_DIR, "selected_chunks.db")

# Output database
WEIGHTED_CHUNKS_DB_PATH = os.path.join(DATABASES_DIR, "weighted_chunks.db")

# Output JSON file for weighted chunks
WEIGHTED_CHUNKS_JSON_PATH = os.path.join(DATABASES_DIR, "weighted_chunks.json")

# Ensure databases directory exists
os.makedirs(DATABASES_DIR, exist_ok=True)

# Mode-specific weight determination parameters
# Answer mode - focused on depth and precision
ENHANCED_QUERY_WEIGHT_ANSWER = 0.5  # αₑ - weight for enhanced query similarity
SUBTOPIC_QUERY_WEIGHT_ANSWER = 0.3  # αₛ - weight for subtopic query similarity
TOPIC_QUERY_WEIGHT_ANSWER = 0.2     # αₜ - weight for topic query similarity

# Recommendation mode - focused on breadth and diversity
ENHANCED_QUERY_WEIGHT_RECOMMENDATION = 0.4  # αₑ - weight for enhanced query similarity
SUBTOPIC_QUERY_WEIGHT_RECOMMENDATION = 0.3  # αₛ - weight for subtopic query similarity
TOPIC_QUERY_WEIGHT_RECOMMENDATION = 0.3     # αₜ - weight for topic query similarity

# Recommendation mode retry parameters (used when not enough high-quality tweets are found)
ENHANCED_QUERY_WEIGHT_RECOMMENDATION_RETRY = 0.1  # Weight for enhanced query similarity on retry
SUBTOPIC_QUERY_WEIGHT_RECOMMENDATION_RETRY = 0.45  # Weight for subtopic query similarity on retry
TOPIC_QUERY_WEIGHT_RECOMMENDATION_RETRY = 0.45  # Weight for topic query similarity on retry

# Default values (for backward compatibility)
ENHANCED_QUERY_WEIGHT = ENHANCED_QUERY_WEIGHT_ANSWER
SUBTOPIC_QUERY_WEIGHT = SUBTOPIC_QUERY_WEIGHT_ANSWER
TOPIC_QUERY_WEIGHT = TOPIC_QUERY_WEIGHT_ANSWER

# Mode-specific metadata boost parameters
# Answer mode - higher thresholds for precision
METADATA_BOOST_FACTOR_ANSWER = 0.5   # γ - boost factor for metadata relevance
METADATA_THRESHOLD_TOPIC_ANSWER = 0.65  # τ - threshold for applying metadata boost
METADATA_THRESHOLD_SUBTOPIC_ANSWER = 0.8  # τ - threshold for applying metadata boost
METADATA_THRESHOLD_ENHANCED_ANSWER = 0.8  # τ - threshold for applying metadata boost

# Recommendation mode - lower thresholds for diversity
METADATA_BOOST_FACTOR_RECOMMENDATION = 0.3   # γ - boost factor for metadata relevance
METADATA_THRESHOLD_TOPIC_RECOMMENDATION = 0.65  # τ - threshold for applying metadata boost
METADATA_THRESHOLD_SUBTOPIC_RECOMMENDATION = 0.8  # τ - threshold for applying metadata boost
METADATA_THRESHOLD_ENHANCED_RECOMMENDATION = 0.8  # τ - threshold for applying metadata boost

# Default values (for backward compatibility)
METADATA_BOOST_FACTOR = METADATA_BOOST_FACTOR_ANSWER
METADATA_THRESHOLD_TOPIC = METADATA_THRESHOLD_TOPIC_ANSWER
METADATA_THRESHOLD_SUBTOPIC = METADATA_THRESHOLD_SUBTOPIC_ANSWER
METADATA_THRESHOLD_ENHANCED = METADATA_THRESHOLD_ENHANCED_ANSWER

# Mode-specific preselection cut threshold
SIMILARITY_PRESELECTION_THRESHOLD_ANSWER = 0.7  # Minimum similarity score for answer mode
SIMILARITY_PRESELECTION_THRESHOLD_RECOMMENDATION = 0.65  # Lower threshold for recommendation mode

# Default value (for backward compatibility)
SIMILARITY_PRESELECTION_THRESHOLD = SIMILARITY_PRESELECTION_THRESHOLD_ANSWER

# Flag to ensure top chunks from each query are always included
ENSURE_TOP_CHUNKS_PER_QUERY = True  # Always include top 1 chunk from each query type
