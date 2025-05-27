"""
module_query_obtention/submodule_github_query/subsubmodule_recent_work/main.py

This module is responsible for generating queries based on recent work in GitHub repositories.
It analyzes the repository to create a more relevant query for paper recommendations.
"""

import os
import sys
import logging
from typing import Optional
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import parameters
from module_query_obtention.submodule_github_query.subsubmodule_recent_work.parameters import (
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH
)

# Import the repository handling functions
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.clone_repo import clone_repository
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.clean_repo import clean_repository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('github_recent_work_query_generator')

def generate_recent_work_query(repo_name: str, github_token: Optional[str] = None) -> str:
    """
    Generate a query based on recent work in a GitHub repository.
    
    Args:
        repo_name: Name of the GitHub repository
        github_token: GitHub API token for authentication (REQUIRED)
        
    Returns:
        A query string based on the repository's recent work or raises an exception if unable to access the repository
    """
    logger.info(f"Generating recent work query for repository: {repo_name}")
    repo_path = None
    
    try:
        # STRICT REQUIREMENT: We must have a GitHub token
        if not github_token:
            error_msg = "GitHub token is required for repository access. No token provided."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Step 1: Clone the repository
        logger.info(f"Cloning repository: {repo_name}")
        clone_success, clone_result = clone_repository(repo_name, github_token)
        
        if not clone_success:
            error_msg = f"Failed to clone repository: {clone_result}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        repo_path = clone_result
        logger.info(f"Repository cloned to: {repo_path}")
        
        # Step 2: Fetch information about recent commits
        logger.info(f"Fetching recent commits information from repository: {repo_name}")
        from module_query_obtention.submodule_github_query.subsubmodule_recent_work.fetch_recent_commits import fetch_recent_commits_info
        
        commits_query, commits_processed = fetch_recent_commits_info(repo_path)
        
        # Step 3: Identify hot files
        logger.info(f"Identifying hot files in repository: {repo_name}")
        from module_query_obtention.submodule_github_query.subsubmodule_recent_work.fetch_hot_files import fetch_hot_files
        
        hot_files = fetch_hot_files(repo_path)
        
        # Log summary of hot files
        logger.info(f"Hot files summary: {len(hot_files)} files identified")
        if hot_files:
            paths = [f['path'] for f in hot_files]
            logger.info(f"Hot file paths: {paths}")
        
        # Step 4: Extract snippets from hot files
        logger.info(f"Extracting snippets from hot files")
        from module_query_obtention.submodule_github_query.subsubmodule_recent_work.extract_hot_file_snippets import generate_hot_files_snippets_query
        
        hot_files_query, snippets_processed = generate_hot_files_snippets_query(repo_path, hot_files)
        
        # Combine the queries
        if commits_processed == 0 and snippets_processed == 0:
            logger.warning(f"No recent commits or hot file snippets found in repository: {repo_name}")
            query = f"Recent developments in software projects similar to {repo_name}"
        else:
            combined_query = ""
            
            if commits_processed > 0:
                logger.info(f"Successfully processed {commits_processed} recent commits")
                combined_query += commits_query
            
            if snippets_processed > 0:
                logger.info(f"Successfully processed {snippets_processed} snippets from hot files")
                if combined_query:
                    combined_query += "\n\n" + "#" * 100 + "\n\n"
                combined_query += hot_files_query
            
            # Step 5: Generate the final query using GPT-4 Turbo
            logger.info(f"Generating final query using GPT-4 Turbo")
            try:
                from module_query_obtention.submodule_github_query.subsubmodule_recent_work.generate_github_recent_query import generate_final_query
                
                # Generate the final query
                raw_query = combined_query
                final_query = generate_final_query(raw_query)
                
                # Use the generated query
                logger.info(f"Successfully generated final query using GPT-4 Turbo")
                query = final_query
            except Exception as e:
                logger.error(f"Error generating final query: {str(e)}")
                logger.warning("Using the raw query instead")
                query = combined_query
        
        # Ensure the query is not too long
        if len(query) > MAX_QUERY_LENGTH:
            logger.info(f"Query too long ({len(query)} chars), truncating to {MAX_QUERY_LENGTH} chars")
            query = query[:MAX_QUERY_LENGTH]
        
        # Ensure the query is not too short
        if len(query) < MIN_QUERY_LENGTH:
            logger.info(f"Query too short ({len(query)} chars), using with repository name")
            query = f"Recent developments in software projects similar to {repo_name}"
        
        logger.info(f"Generated recent work query of length {len(query)} characters")
        return query
    
    finally:
        # Clean up the repository directory
        if repo_path:
            logger.info(f"Cleaning up repository directory: {repo_path}")
            clean_success, clean_message = clean_repository(repo_path)
            if not clean_success:
                logger.warning(f"Failed to clean up repository: {clean_message}")
            else:
                logger.info(f"Repository cleanup successful: {clean_message}")
