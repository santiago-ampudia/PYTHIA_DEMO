# module_query_obtention/search_mode.py

"""
Module 1: Query Obtention - Search Mode
--------------------------------------
This module is responsible for determining the search mode for the paper search pipeline.
Supports two modes:
- "answer": Provides a concise answer to the query (used for Smart Answer)
- "recommendation": Recommends relevant papers in a Twitter-style format (used for Smart Search and GitHub)

The mode is determined dynamically based on the source of the query:
- If the query is from the Smart Answer feature, use answer mode
- If the query is from Smart Search or GitHub features, use recommendation mode
"""

import os
import inspect
import traceback
import logging

# Configure logging
logger = logging.getLogger("search_mode")

def get_search_mode():
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    # Hardcoded to recommendation mode for GitHub integration
    mode = "recommendation"
    """
    Determines the search mode to be used based on the source of the query.
    
    This function examines the call stack and environment variables to determine
    which endpoint or feature initiated the query, and sets the appropriate mode.
    
    Returns:
        str: The search mode, either "answer" or "recommendation".
    """
    try:
        # Check if a mode was explicitly set in environment variables
        explicit_mode = os.environ.get("SEARCH_MODE")
        if explicit_mode in ["answer", "recommendation"]:
            logger.info(f"Using explicitly set search mode from environment: {explicit_mode}")
            return explicit_mode
        
        # Check the call stack to determine which endpoint called this function
        stack = inspect.stack()
        calling_functions = [frame.function for frame in stack]
        calling_files = [frame.filename for frame in stack]
        
        # Log the call stack for debugging
        logger.debug(f"Call stack functions: {calling_functions}")
        logger.debug(f"Call stack files: {calling_files}")
        
        # Check for GitHub-related endpoints (always use recommendation mode)
        github_indicators = [
            "github_recommendations_endpoint",
            "github_recommendations_recent_work_endpoint",
            "/github/",
            "github_page"
        ]
        
        for func in calling_functions:
            if any(indicator in func for indicator in github_indicators):
                logger.info("Detected GitHub-related endpoint, using recommendation mode")
                return "recommendation"
                
        for file in calling_files:
            if any(indicator in file for indicator in github_indicators):
                logger.info("Detected GitHub-related file, using recommendation mode")
                return "recommendation"
        
        # Check for Smart Search indicators (recommendation mode)
        search_indicators = ["smart_search", "search_page"]
        for func in calling_functions:
            if any(indicator in func for indicator in search_indicators):
                logger.info("Detected Smart Search feature, using recommendation mode")
                return "recommendation"
                
        for file in calling_files:
            if any(indicator in file for indicator in search_indicators):
                logger.info("Detected Smart Search file, using recommendation mode")
                return "recommendation"
        
        # If no specific indicators are found, check if it's coming from the research_query_endpoint
        # This is the default endpoint for Smart Answer
        if "research_query_endpoint" in calling_functions:
            logger.info("Detected research_query_endpoint (Smart Answer), using answer mode")
            return "answer"
            
        # Default to answer mode if we can't determine the source
        # This ensures that if there's any doubt, we provide a direct answer rather than recommendations
        logger.warning("Could not determine query source, defaulting to answer mode")
        return "answer"
        
    except Exception as e:
        # If anything goes wrong in determining the mode, log the error and default to answer mode
        logger.error(f"Error determining search mode: {str(e)}")
        logger.error(traceback.format_exc())
        return "answer"

if __name__ == "__main__":
    # If this script is run directly, obtain the search mode and print it to the console.
    # Set up console logging for direct execution
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    mode = get_search_mode()
    print("Search Mode:", mode)  # Output the mode for verification
