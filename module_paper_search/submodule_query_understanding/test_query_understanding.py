"""
Test script for the query understanding module.

This script tests the query understanding module by running it on a sample query
and printing the results.
"""

import os
import sys
import logging

# Add the project root to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the query understanding module
from module_paper_search.submodule_query_understanding import run_query_understanding
from module_query_obtention.query_input import get_user_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_query_understanding')

def main():
    """
    Main function to test the query understanding module.
    """
    logger.info("Testing query understanding module...")
    
    # Get the user query
    user_query = get_user_query()
    logger.info(f"User query: {user_query}")
    
    # Run the query understanding
    topic_query, subtopic_query, original_query = run_query_understanding(user_query)
    
    # Print the results
    print("\n===== Query Understanding Results =====")
    print(f"Original Query: {original_query}")
    print(f"Topic Query: {topic_query}")
    print(f"Subtopic Query: {subtopic_query}")
    print("======================================\n")
    
    logger.info("Test completed.")

if __name__ == "__main__":
    main()
