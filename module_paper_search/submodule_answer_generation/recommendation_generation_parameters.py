"""
module_paper_search/submodule_answer_generation/recommendation_generation_parameters.py

This module contains parameters for the recommendation generation process.
"""

# Import os
import os

# Number of recommendations to generate
N_RECOMMENDATIONS = 10

# Minimum quality score for recommendations
RECOMMENDATION_QUALITY_THRESHOLD = 0.5

# Import output directory from the central module
from module_paper_search.output_directories import RECOMMENDATION_OUTPUT_DIR

# Output directory for saving recommendations
OUTPUT_DIR = RECOMMENDATION_OUTPUT_DIR

# Retry parameters
MAX_RETRY_ATTEMPTS = 1  # Maximum number of retry attempts if not enough high-quality recommendations

# Default OpenAI API key (should be overridden by environment variable)
# Always use the API key from environment variables
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Recommendation formatting
TWEET_TEXT_KEY = "tweet_text"  # Key in chunk dictionary containing the tweet text
PAPER_IDS_KEY = "paper_ids"    # Key in chunk dictionary containing the paper IDs
BATCH_ID_KEY = "batch_id"      # Key in chunk dictionary containing the batch ID

# Recommendation output format
RECOMMENDATION_OUTPUT_FORMAT = {
    "query": "",                # The query that was used
    "timestamp": "",            # Timestamp of generation
    "recommendations": [        # List of recommendations
        {
            "tweet_text": "",   # The tweet text
            "score": 0.0,       # The final weight adjusted score
            "paper_ids": [],    # List of paper IDs used in the tweet
            "batch_id": "",     # Batch ID for the tweet
            "component_scores": {
                "tweet_relevance_score": 0.0,  # LLM relevance score for the tweet
                "avg_normalized_weight": 0.0,  # Average normalized weight of chunks
                "lambda_weight": 0.0           # Lambda weight used for combining scores
            }
        }
    ]
}
