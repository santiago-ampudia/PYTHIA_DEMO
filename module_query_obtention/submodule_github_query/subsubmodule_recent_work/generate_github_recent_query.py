"""
module_query_obtention/submodule_github_query/subsubmodule_recent_work/generate_github_recent_query.py

This module is responsible for generating the final query for recent work analysis
using GPT-4 Turbo to interpret the repository activity and hot files information.
"""

import os
import logging
import json
from typing import Dict, Any, Optional

import openai

# Import parameters
from module_query_obtention.submodule_github_query.subsubmodule_recent_work.parameters import (
    GPT_MODEL,
    MAX_FINAL_QUERY_LENGTH,
    MIN_FINAL_QUERY_LENGTH,
    GPT_RECENT_WORK_INSTRUCTIONS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_final_query(repository_analysis: str) -> str:
    """
    Generate a final query using GPT-4 Turbo based on the repository analysis.
    
    Args:
        repository_analysis: String containing the repository analysis (commits and hot files)
        
    Returns:
        A refined query string for academic paper search
    """
    # Check if OpenAI API key is available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        error_msg = "OpenAI API key is required but not found in environment variables"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Set up the OpenAI client
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Prepare the prompt
    system_prompt = GPT_RECENT_WORK_INSTRUCTIONS
    user_prompt = f"Here is the repository analysis:\n\n{repository_analysis}"
    
    logger.info("Generating final query using GPT-4 Turbo")
    logger.info(f"System prompt length: {len(system_prompt)} characters")
    logger.info(f"User prompt length: {len(user_prompt)} characters")
    
    # Save the prompt to a file
    from pathlib import Path
    import datetime
    
    prompts_dir = Path("prompts")
    prompts_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_file = prompts_dir / f"recent_work_prompt_{timestamp}.txt"
    
    # Write the prompt to the file
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(f"SYSTEM PROMPT:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}")
    
    logger.info(f"Saved prompt to file: {prompt_file}")
    
    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=MAX_FINAL_QUERY_LENGTH,
            n=1,
            stop=None
        )
        
        # Extract the generated query
        generated_query = response.choices[0].message.content.strip()
        
        # Check if the query meets the minimum length requirement
        if len(generated_query.split()) < MIN_FINAL_QUERY_LENGTH:
            logger.warning(f"Generated query is too short: {len(generated_query.split())} tokens")
            logger.warning("Using the repository analysis as the query instead")
            return repository_analysis
        
        logger.info(f"Generated query: {generated_query}")
        return generated_query
        
    except Exception as e:
        logger.error(f"Error generating query with GPT-4 Turbo: {str(e)}")
        logger.warning("Using the repository analysis as the query instead")
        return repository_analysis
