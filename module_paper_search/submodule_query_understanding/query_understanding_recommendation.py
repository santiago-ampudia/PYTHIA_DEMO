"""
submodule_query_understanding/query_understanding_recommendation.py

This module implements a specialized version of Query Understanding & Decomposition for recommendation mode.
It uses OpenAI's GPT-4 to analyze a repository description and break it down into five components:
1. Architecture Query: Focus on the system architecture, design patterns, and overall structure
2. Technical Implementation Query: Capture the specific technologies, libraries, and frameworks used
3. Algorithmic Approach Query: Focus on the algorithms, mathematical models, and computational techniques
4. Domain-Specific Query: Target the specific academic domain and research methodologies
5. Integration & Pipeline Query: Capture how components interact in the pipeline

This decomposition helps with more targeted semantic and structural matching for repository-based recommendations.
"""

import os
import json
import logging
from openai import OpenAI
from typing import Dict, Tuple

# Import parameters from the parameters file
from module_paper_search.submodule_query_understanding.query_understanding_recommendation_parameters import (
    DEFAULT_OPENAI_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    SYSTEM_PROMPT,
    USER_PROMPT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('query_understanding_recommendation')

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

def split_query_recommendation(query: str) -> Dict[str, str]:
    """
    Split a repository description into five specialized components using LLM.
    
    Args:
        query (str): The repository description or README content
        
    Returns:
        Dict[str, str]: A dictionary containing:
            - 'architecture_query': System architecture and design patterns
            - 'technical_implementation_query': Technologies, libraries, and frameworks
            - 'algorithmic_approach_query': Algorithms and computational techniques
            - 'domain_specific_query': Academic domain and research methodologies
            - 'integration_pipeline_query': Component interactions and pipeline structure
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_openai_api_key())
        
        # Log the query being processed
        logger.info(f"Splitting repository description for recommendation mode")
        
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
        
        # Parse the response to extract the five components
        result = {}
        
        # Try to parse the response line by line
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('architecture_query:'):
                result['architecture_query'] = line[len('architecture_query:'):].strip()
            elif line.startswith('technical_implementation_query:'):
                result['technical_implementation_query'] = line[len('technical_implementation_query:'):].strip()
            elif line.startswith('algorithmic_approach_query:'):
                result['algorithmic_approach_query'] = line[len('algorithmic_approach_query:'):].strip()
            elif line.startswith('domain_specific_query:'):
                result['domain_specific_query'] = line[len('domain_specific_query:'):].strip()
            elif line.startswith('integration_pipeline_query:'):
                result['integration_pipeline_query'] = line[len('integration_pipeline_query:'):].strip()
        
        # Check if all components were extracted
        expected_components = [
            'architecture_query', 
            'technical_implementation_query', 
            'algorithmic_approach_query', 
            'domain_specific_query', 
            'integration_pipeline_query'
        ]
        
        # Check if all components were extracted
        missing_components = [comp for comp in expected_components if comp not in result]
        if missing_components:
            logger.warning(f"Not all query components were extracted from response: {missing_components}")
            # Fill in any missing components with the original query
            for component in missing_components:
                result[component] = query
        
        # Log the decomposition result
        logger.info(f"Repository description split successfully into 5 components")
        
        return result
        
    except Exception as e:
        logger.error(f"Error splitting repository description: {str(e)}")
        # Return a basic result with just the original query if there's an error
        return {
            "architecture_query": query,
            "technical_implementation_query": query,
            "algorithmic_approach_query": query,
            "domain_specific_query": query,
            "integration_pipeline_query": query
        }

def run_query_understanding_recommendation(query: str) -> Tuple[str, str, str, str, str]:
    """
    Main function to run the query understanding and decomposition step for recommendation mode.
    
    Args:
        query (str): The repository description or README content
        
    Returns:
        Tuple[str, str, str, str, str]: A tuple containing 
            (architecture_query, technical_implementation_query, algorithmic_approach_query, 
             domain_specific_query, integration_pipeline_query)
    """
    logger.info("Running query understanding and decomposition for recommendation mode...")
    
    # Split the query
    result = split_query_recommendation(query)
    
    # Extract the components
    architecture_query = result.get("architecture_query", query)
    technical_implementation_query = result.get("technical_implementation_query", query)
    algorithmic_approach_query = result.get("algorithmic_approach_query", query)
    domain_specific_query = result.get("domain_specific_query", query)
    integration_pipeline_query = result.get("integration_pipeline_query", query)
    
    logger.info("Query understanding for recommendation mode completed.")
    return (
        architecture_query, 
        technical_implementation_query, 
        algorithmic_approach_query, 
        domain_specific_query, 
        integration_pipeline_query
    )


if __name__ == "__main__":
    # For testing purposes
    import sys
    import os
    
    # Add the project root to the Python path to allow imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    
    from module_query_obtention.query_input import get_user_query
    
    # Get the user query (in this case, a repository description)
    repo_description = get_user_query()
    
    # Run the query understanding for recommendation mode
    (architecture, technical, algorithmic, 
     domain_specific, integration) = run_query_understanding_recommendation(repo_description)
    
    # Print the results
    print("\nQuery Understanding Recommendation Results:")
    print(f"Architecture Query: {architecture}")
    print(f"Technical Implementation Query: {technical}")
    print(f"Algorithmic Approach Query: {algorithmic}")
    print(f"Domain-Specific Query: {domain_specific}")
    print(f"Integration & Pipeline Query: {integration}")
