"""
KGB-based midpoint finding module.

This module identifies intermediate nodes (midpoints) between pairs of queries
to build a more comprehensive knowledge graph path for RAG-based research.
"""

import os
import json
import logging
from typing import List, Dict, Any
from openai import OpenAI

from .midpoint_finding_kgb_parameters import (
    RESULTS_DIR,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
    MIDPOINTS_JSON_PATH
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def find_midpoint_between_queries(query1: str, query2: str) -> str:
    """
    Find a midpoint between two queries using GPT-4 Turbo.
    
    Args:
        query1: The first query
        query2: The second query
        
    Returns:
        A midpoint query that bridges the conceptual gap between query1 and query2
    """
    # Get the OpenAI API key
    api_key = get_openai_api_key()
    
    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=api_key)
    
    # Prepare the user prompt
    user_prompt = f"""
I need to find a conceptual midpoint between two research queries that will serve as an intermediate node in a knowledge graph.

QUERY A: {query1}
QUERY B: {query2}

Please identify a midpoint query that:
1. Represents a logical stepping stone between Query A and Query B
2. Is specific enough to guide retrieval of relevant academic literature
3. Helps form a coherent path from A to B
4. Is formulated as a clear, searchable query similar in format to the input queries

The midpoint should help bridge the conceptual gap between these queries when used with a RAG system to retrieve academic papers.

IMPORTANT INSTRUCTIONS:
- Return ONLY the midpoint query text without any additional explanation or formatting
- DO NOT prefix your response with phrases like "MIDPOINT QUERY:" or similar labels
- Just provide the query text directly, as if it were a standalone search query
"""
    
    try:
        logger.info(f"Calling LLM API to find midpoint between queries")
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        # Extract the midpoint query from the response
        if response.choices and len(response.choices) > 0:
            midpoint_query = response.choices[0].message.content.strip()
        else:
            logger.warning("No choices returned from LLM API.")
            midpoint_query = ""
        
        logger.info(f"Generated midpoint query: {midpoint_query}")
        return midpoint_query
        
    except Exception as e:
        logger.error(f"API request failed: {e}")
        raise ValueError(f"API request failed: {e}")

def find_all_midpoints(queries_list: List[str]) -> List[str]:
    """
    Find midpoints between all consecutive pairs of queries in the list.
    
    Args:
        queries_list: List of queries
        
    Returns:
        A new list of queries with midpoints inserted between original queries
    """
    if not queries_list or len(queries_list) < 2:
        logger.warning("Need at least 2 queries to find midpoints")
        return queries_list
    
    # Create a new list to store original queries and midpoints
    new_queries_list = [queries_list[0]]  # Start with the first query
    
    # Find midpoints between each consecutive pair of queries
    for i in range(len(queries_list) - 1):
        query1 = queries_list[i]
        query2 = queries_list[i + 1]
        
        logger.info(f"Finding midpoint between query {i} and query {i+1}")
        
        # Find the midpoint between the current pair of queries
        midpoint = find_midpoint_between_queries(query1, query2)
        
        # Add the midpoint and the next query to the new list
        new_queries_list.append(midpoint)
        new_queries_list.append(query2)
    
    # Save the results to a JSON file
    save_midpoints(queries_list, new_queries_list)
    
    return new_queries_list

def save_midpoints(original_queries: List[str], new_queries: List[str]) -> None:
    """
    Save the original and new queries with midpoints to a JSON file.
    
    Args:
        original_queries: The original list of queries
        new_queries: The new list of queries with midpoints inserted
    """
    # Create a dictionary to store the results
    results = {
        "original_queries": original_queries,
        "queries_with_midpoints": new_queries
    }
    
    # Save the results to a JSON file
    with open(MIDPOINTS_JSON_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved midpoints to {MIDPOINTS_JSON_PATH}")

def run_midpoint_finding_kgb(queries_list: List[str]) -> List[str]:
    """
    Run the KGB-based midpoint finding process.
    
    Args:
        queries_list: List of queries to find midpoints between
        
    Returns:
        A new list of queries with midpoints inserted between original queries
    """
    logger.info("Starting KGB-based midpoint finding...")
    
    # Find midpoints between all consecutive pairs of queries
    new_queries_list = find_all_midpoints(queries_list)
    
    logger.info(f"Midpoint finding completed. Original query count: {len(queries_list)}, New query count: {len(new_queries_list)}")
    
    return new_queries_list
