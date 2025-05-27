"""
submodule_arxiv_category_prediction/arxiv_category_prediction.py

This module implements Step 3 of the paper search pipeline: arXiv Category Prediction.
It uses a large language model to predict the most relevant arXiv categories for a given research question,
based on the topic_query, subtopic_query, and enhanced_query from the query understanding step.

This helps narrow the paper search space by focusing on the most relevant arXiv categories.
The implementation is deliberately conservative to avoid losing relevant papers.
"""

import os
import json
import logging
import re
from openai import OpenAI
from typing import Dict, List, Tuple, Set

# Import parameters from the parameters file
from module_paper_search.submodule_arxiv_category_prediction.arxiv_category_prediction_parameters import (
    DEFAULT_OPENAI_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    SYSTEM_PROMPT,
    USER_PROMPT,
    ARXIV_CATEGORIES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('arxiv_category_prediction')

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

def get_related_categories(categories: List[str], max_categories: int = 7) -> List[str]:
    """
    Get related categories for the given categories based on the category hierarchy.
    This helps ensure we don't miss relevant papers by being too restrictive.
    Avoids including overly broad parent categories to prevent pulling in too many papers.
    
    Args:
        categories (List[str]): List of arXiv category codes
        max_categories (int): Maximum number of categories to return
        
    Returns:
        List[str]: List of related category codes
    """
    # Start with the original predicted categories
    related_categories = set(categories)
    
    # Category relationships - mapping from specific to related specific categories
    # Avoid parent categories like "physics" or "cs" which are too broad
    category_relationships = {
        # Physics categories
        "hep-ex": ["physics.ins-det", "nucl-ex"],
        "hep-th": ["hep-ph", "math-ph", "gr-qc"],
        "hep-ph": ["hep-ex", "nucl-th"],
        "hep-lat": ["hep-th", "hep-ph"],
        "nucl-ex": ["hep-ex", "nucl-th"],
        "nucl-th": ["hep-ph", "nucl-ex"],
        "gr-qc": ["astro-ph.CO", "hep-th"],
        "quant-ph": ["cond-mat.mes-hall", "physics.atom-ph"],
        "physics.ins-det": ["hep-ex", "physics.acc-ph"],
        "physics.acc-ph": ["physics.ins-det", "hep-ex"],
        "physics.data-an": ["stat.ML", "cs.LG"],
        
        # Computer Science categories
        "cs.AI": ["cs.LG", "cs.CL", "cs.CV"],
        "cs.LG": ["stat.ML", "cs.AI", "cs.CV"],
        "cs.CV": ["cs.LG", "cs.AI"],
        "cs.CL": ["cs.LG", "cs.AI"],
        "cs.IR": ["cs.LG", "cs.AI"],
        
        # Math categories
        "math.PR": ["stat.TH", "math.ST"],
        "math.ST": ["stat.TH", "stat.ME"],
        
        # Statistics categories
        "stat.ML": ["cs.LG", "stat.ME"],
        
        # Quantitative Biology categories
        "q-bio.QM": ["q-bio.GN", "q-bio.MN"],
        
        # Quantitative Finance categories
        "q-fin.ST": ["q-fin.PR", "stat.AP"],
    }
    
    # Add related categories based on the relationships map
    # but only if we haven't reached the maximum
    for category in categories:
        if category in category_relationships and len(related_categories) < max_categories:
            # Sort related categories by relevance (we assume the order in the list reflects relevance)
            for related in category_relationships[category]:
                if related not in related_categories and len(related_categories) < max_categories:
                    related_categories.add(related)
    
    # Convert set back to list, maintaining the original order of predicted categories first
    result = list(categories)
    for cat in list(related_categories - set(categories)):
        result.append(cat)
    
    return result[:max_categories]

def predict_categories(topic_query: str, subtopic_query: str, enhanced_query: str) -> List[str]:
    """
    Predict the most relevant arXiv categories for a given research question.
    Takes a conservative approach to avoid missing relevant papers, but remains focused
    on specific categories rather than broad parent categories.
    
    Args:
        topic_query (str): The high-level research theme
        subtopic_query (str): The specific focus, method, or technique
        enhanced_query (str): A rewritten version optimized for semantic search
        
    Returns:
        List[str]: A list of ranked arXiv category codes (e.g., ["hep-ph", "hep-ex"])
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_openai_api_key())
        
        # Log the queries being processed
        logger.info(f"Predicting categories for:")
        logger.info(f"Topic: {topic_query}")
        logger.info(f"Subtopic: {subtopic_query}")
        logger.info(f"Enhanced query: {enhanced_query}")
        
        # Format the user prompt with the queries
        formatted_user_prompt = USER_PROMPT.format(
            topic_query=topic_query,
            subtopic_query=subtopic_query,
            enhanced_query=enhanced_query
        )
        
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
        
        # Parse the response to extract the predicted categories
        # Looking for a pattern like: predicted_categories: ["hep-ph", "hep-ex", "physics.acc-ph"]
        match = re.search(r'predicted_categories:\s*\[(.*?)\]', content, re.DOTALL)
        
        if match:
            # Extract the categories from the match
            categories_str = match.group(1)
            
            # Split by commas and clean up each category
            categories = [
                cat.strip().strip('"\'') 
                for cat in categories_str.split(',')
            ]
            
            # Filter out any empty strings
            categories = [cat for cat in categories if cat]
            
            # Validate that each category exists in our list of arXiv categories
            validated_categories = []
            for cat in categories:
                if cat in ARXIV_CATEGORIES:
                    validated_categories.append(cat)
                else:
                    logger.warning(f"Category '{cat}' not found in arXiv categories list. Skipping.")
            
            # Ensure we have at least 2 categories
            if len(validated_categories) < 2:
                logger.warning(f"Less than 2 valid categories predicted. Adding default categories.")
                # Add default categories (hep-ex, hep-th) if we don't have enough
                default_categories = ["hep-ex", "hep-ph"]
                for cat in default_categories:
                    if cat not in validated_categories:
                        validated_categories.append(cat)
                        if len(validated_categories) >= 2:
                            break
            
            # Get related categories to be more conservative, but limit to a reasonable number
            # and avoid overly broad parent categories
            all_categories = get_related_categories(validated_categories, max_categories=7)
            
            # Log the original and expanded categories
            logger.info(f"Original predicted categories: {validated_categories}")
            logger.info(f"Expanded categories (including related): {all_categories}")
            
            return all_categories
        else:
            logger.error(f"Failed to parse categories from response: {content}")
            # Return default categories if parsing fails
            default_categories = ["hep-ex", "hep-ph", "physics.ins-det", "physics.acc-ph", "nucl-ex"]
            logger.info(f"Using default categories: {default_categories}")
            return default_categories
        
    except Exception as e:
        logger.error(f"Error predicting categories: {str(e)}")
        # Return default categories if there's an error
        default_categories = ["hep-ex", "hep-ph", "physics.ins-det", "physics.acc-ph", "nucl-ex"]
        logger.info(f"Using default categories: {default_categories}")
        return default_categories

def run_arxiv_category_prediction(topic_query: str, subtopic_query: str, enhanced_query: str) -> List[str]:
    """
    Main function to run the arXiv category prediction step.
    
    Args:
        topic_query (str): The high-level research theme
        subtopic_query (str): The specific focus, method, or technique
        enhanced_query (str): A rewritten version optimized for semantic search
        
    Returns:
        List[str]: A list of ranked arXiv category codes
    """
    logger.info("Running arXiv category prediction...")
    
    # Predict the categories
    predicted_categories = predict_categories(topic_query, subtopic_query, enhanced_query)
    
    logger.info("arXiv category prediction completed.")
    return predicted_categories


if __name__ == "__main__":
    # For testing purposes
    import sys
    import os
    
    # Add the project root to the Python path to allow imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    
    from module_query_obtention.query_input import get_user_query
    from module_paper_search.submodule_query_understanding.query_understanding import run_query_understanding
    
    # Get the user query
    user_query = get_user_query()
    
    # Run the query understanding
    topic, subtopic, enhanced = run_query_understanding(user_query)
    
    # Run the arXiv category prediction
    categories = run_arxiv_category_prediction(topic, subtopic, enhanced)
    
    # Print the results
    print("\narXiv Category Prediction Results:")
    print(f"Predicted Categories: {categories}")
    print("\nFull Category Names:")
    for cat in categories:
        print(f"- {cat}: {ARXIV_CATEGORIES.get(cat, 'Unknown category')}")
