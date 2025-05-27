"""
module_query_obtention/submodule_github_query/subsubmodule_general_understanding/generate_github_general_query.py

This module is responsible for generating the final query for the search engine using GPT-4.
It takes the repository analysis (README content, code snippets, etc.) and generates a comprehensive
query that captures the essence of the repository for academic literature search.
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
import requests
from dotenv import load_dotenv

from . import parameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def call_openai_api(prompt: str, model: str = parameters.GPT_MODEL, 
                   max_tokens: int = parameters.MAX_FINAL_QUERY_LENGTH) -> Optional[str]:
    """
    Call the OpenAI API to generate a response.
    
    Args:
        prompt: The prompt to send to the API
        model: The model to use (default: from parameters)
        max_tokens: Maximum number of tokens in the response (default: from parameters)
        
    Returns:
        The generated response or None if there was an error
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found in environment variables")
        return None
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": parameters.GPT_QUERY_INSTRUCTIONS},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,  # Balanced between creativity and precision
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                                headers=headers, 
                                json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"Unexpected API response format: {result}")
            return None
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        return None

def generate_final_query(repo_analysis: str) -> str:
    """
    Generate the final query for the search engine using GPT-4.
    
    Args:
        repo_analysis: The repository analysis containing README content, code snippets, etc.
        
    Returns:
        The generated query for the search engine
    """
    logger.info("Generating final query using GPT-4...")
    
    # Create a prompt for GPT-4
    prompt = f"""
Here is the detailed analysis of a GitHub repository:

{repo_analysis}

Based on this repository analysis, generate a comprehensive query for academic literature search.
"""
    
    # Save the prompt to a file
    prompts_dir = Path("prompts")
    prompts_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a timestamp for the filename
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_file = prompts_dir / f"general_understanding_prompt_{timestamp}.txt"
    
    # Write the prompt to the file
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(f"SYSTEM PROMPT:\n{parameters.GPT_QUERY_INSTRUCTIONS}\n\nUSER PROMPT:\n{prompt}")
    
    logger.info(f"Saved prompt to file: {prompt_file}")
    
    # Call the OpenAI API
    response = call_openai_api(prompt)
    
    if not response:
        logger.warning("Failed to generate query using GPT-4. Using repository analysis as fallback.")
        # Truncate the repository analysis if it's too long
        if len(repo_analysis) > parameters.MAX_FINAL_QUERY_LENGTH * 4:  # Rough estimate of tokens to characters
            logger.warning(f"Repository analysis too long ({len(repo_analysis)} characters). Truncating...")
            repo_analysis = repo_analysis[:parameters.MAX_FINAL_QUERY_LENGTH * 4]
        return repo_analysis
    
    # Check if the response meets the minimum length requirement
    if len(response.split()) < parameters.MIN_FINAL_QUERY_LENGTH / 5:  # Rough estimate of tokens to words
        logger.warning(f"Generated query too short. Using repository analysis as fallback.")
        return repo_analysis
    
    logger.info("Successfully generated final query using GPT-4")
    logger.info(f"Query length: approximately {len(response.split())} words")
    
    return response
