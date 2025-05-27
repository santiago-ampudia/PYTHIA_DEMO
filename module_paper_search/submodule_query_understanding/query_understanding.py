"""
submodule_query_understanding/query_understanding.py

This module implements Step 2 of the paper search pipeline: Query Understanding & Decomposition.
It uses OpenAI's GPT-4 to analyze a user query and break it down into:
1. A topic query (high-level research theme)
2. A subtopic query (specific focus, method, or technique)
3. An enhanced query (rewritten version optimized for semantic search)

This decomposition helps with more targeted semantic and structural matching in later steps.
"""

import os
import json
import logging
from openai import OpenAI
from typing import Dict, Tuple

# Import parameters from the parameters file
from module_paper_search.submodule_query_understanding.query_understanding_parameters import (
    DEFAULT_OPENAI_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    SYSTEM_PROMPT,
    USER_PROMPT,
    get_system_prompt
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('query_understanding')

def get_openai_api_key() -> str:
    """
    Get the OpenAI API key from environment variables.
    
    Returns:
        str: The OpenAI API key
    """
    # Get from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
        
    return api_key

def split_query(query: str) -> Dict[str, str]:
    """
    Split a user query into topic, subtopic, and enhanced query components using LLM.
    
    Args:
        query (str): The original user query
        
    Returns:
        Dict[str, str]: A dictionary containing:
            - 'topic_query': The high-level research theme
            - 'subtopic_query': The specific focus, method, or technique
            - 'enhanced_query': A rewritten version optimized for semantic search
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_openai_api_key())
        
        # Log the query being processed
        logger.info(f"Splitting query: {query}")
        
        # Format the user prompt with the query
        formatted_user_prompt = USER_PROMPT.format(query=query)
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted_user_prompt}
            ],
            temperature=TEMPERATURE
        )
        
        # Extract the response content
        content = response.choices[0].message.content
        
        # Parse the response to extract the three components
        result = {}
        
        # Try to parse the response line by line
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('topic_query:'):
                result['topic_query'] = line[len('topic_query:'):].strip()
            elif line.startswith('subtopic_query:'):
                result['subtopic_query'] = line[len('subtopic_query:'):].strip()
            elif line.startswith('enhanced_query:'):
                result['enhanced_query'] = line[len('enhanced_query:'):].strip()
        
        # Check if all components were extracted
        if not all(k in result for k in ['topic_query', 'subtopic_query', 'enhanced_query']):
            logger.warning(f"Not all query components were extracted from response: {content}")
            # Fill in any missing components with the original query
            for component in ['topic_query', 'subtopic_query', 'enhanced_query']:
                if component not in result:
                    result[component] = query
        
        # Log the decomposition result
        logger.info(f"Query split successfully: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error splitting query: {str(e)}")
        # Return a basic result with just the original query if there's an error
        return {
            "topic_query": query,
            "subtopic_query": query,
            "enhanced_query": query
        }

def run_query_understanding(query: str) -> Tuple[str, str, str]:
    """
    Main function to run the query understanding and decomposition step.
    
    Args:
        query (str): The user's original query
        
    Returns:
        Tuple[str, str, str]: A tuple containing (topic_query, subtopic_query, enhanced_query)
    """
    logger.info("Running query understanding and decomposition...")
    
    # Split the query
    result = split_query(query)
    
    # Extract the components
    topic_query = result.get("topic_query", query)
    subtopic_query = result.get("subtopic_query", query)
    enhanced_query = result.get("enhanced_query", query)
    
    logger.info("Query understanding completed.")
    return topic_query, subtopic_query, enhanced_query


if __name__ == "__main__":
    # For testing purposes
    import sys
    import os
    
    # Add the project root to the Python path to allow imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    
    from module_query_obtention.query_input import get_user_query
    
    # Get the user query
    user_query = get_user_query()
    
    # Run the query understanding
    topic, subtopic, enhanced = run_query_understanding(user_query)
    
    # Print the results
    print("\nQuery Understanding Results:")
    print(f"Topic Query: {topic}")
    print(f"Subtopic Query: {subtopic}")
    print(f"Enhanced Query: {enhanced}")
