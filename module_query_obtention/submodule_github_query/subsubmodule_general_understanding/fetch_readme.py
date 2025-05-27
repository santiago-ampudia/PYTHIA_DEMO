"""
module_query_obtention/submodule_github_query/subsubmodule_general_understanding/fetch_readme.py

This module is responsible for fetching the README content from an already cloned GitHub repository.
It extracts the README.md file from the repository directory.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict

from . import parameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('github_readme_fetcher')

def fetch_readme_content(repo_path: str) -> Tuple[bool, str]:
    """
    Fetch README content from an already cloned GitHub repository.
    Searches level-by-level for README files and concatenates their content.
    
    Args:
        repo_path: Path to the cloned repository directory
        
    Returns:
        A tuple of (success, readme_content or error_message)
    """
    if not repo_path:
        error_msg = "No repository path provided"
        logger.error(error_msg)
        return False, error_msg
    
    logger.info(f"Fetching README files from repository at: {repo_path}")
    
    # Ensure the repository directory exists
    repo_dir = Path(repo_path)
    if not repo_dir.exists():
        error_msg = f"Repository directory does not exist: {repo_dir}"
        logger.error(error_msg)
        return False, error_msg
    
    # Common README filenames (case-insensitive)
    readme_patterns = ["readme.md", "readme.txt", "readme", "readme.rst"]
    
    # Get all directories in the repo, organized by level
    all_dirs_by_level = {}
    root_level = len(repo_dir.parts)
    
    # First, organize all directories by their level
    for root, dirs, _ in os.walk(repo_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Calculate the directory level
        path = Path(root)
        level = len(path.parts) - root_level
        
        if level not in all_dirs_by_level:
            all_dirs_by_level[level] = []
        
        all_dirs_by_level[level].append(path)
    
    # Concatenated README content
    all_readme_content = ""
    found_readmes = []
    
    # Search each level for README files
    max_level = max(all_dirs_by_level.keys()) if all_dirs_by_level else 0
    
    for level in range(max_level + 1):
        # If this level doesn't exist, skip it
        if level not in all_dirs_by_level:
            continue
            
        logger.info(f"Searching level {level} for README files")
        level_readmes = []
        
        # Search all directories at this level
        for directory in all_dirs_by_level[level]:
            # Get all files in this directory (not recursive)
            try:
                for file_path in directory.iterdir():
                    # Skip directories and hidden files
                    if file_path.is_dir() or file_path.name.startswith('.'):
                        continue
                        
                    # Check if the file is a README file
                    if file_path.name.lower() in readme_patterns:
                        level_readmes.append(file_path)
                        logger.info(f"Found README file: {file_path}")
            except Exception as e:
                logger.error(f"Error accessing directory {directory}: {str(e)}")
        
        # Process README files found at this level
        for readme_path in level_readmes:
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Add a header for each README file
                relative_path = readme_path.relative_to(repo_dir)
                header = f"\n\n--- README: {relative_path} ---\n\n"
                
                # Check if adding this content would exceed the maximum size
                if len(all_readme_content) + len(header) + len(content) > parameters.MAX_README_SIZE:
                    # If we already have some content, we can stop here
                    if all_readme_content:
                        logger.info(f"Reached maximum README size ({parameters.MAX_README_SIZE}), stopping search")
                        break
                    # If this is the first README, truncate it to fit
                    else:
                        available_space = parameters.MAX_README_SIZE - len(header)
                        content = content[:available_space]
                        logger.info(f"Truncated README content to fit maximum size")
                
                # Add the content
                all_readme_content += header + content
                found_readmes.append(readme_path)
                logger.info(f"Added content from README file: {readme_path}")
                
                # Check if we've reached the maximum size
                if len(all_readme_content) >= parameters.MAX_README_SIZE:
                    logger.info(f"Reached maximum README size ({parameters.MAX_README_SIZE}), stopping search")
                    break
                    
            except Exception as e:
                logger.error(f"Error reading README file {readme_path}: {str(e)}")
        
        # If we've reached the maximum size, stop searching deeper levels
        if len(all_readme_content) >= parameters.MAX_README_SIZE:
            break
    
    # Check if we found any README files
    if not found_readmes:
        error_msg = f"No README files found in repository: {repo_dir}"
        logger.warning(error_msg)
        return False, error_msg
    
    logger.info(f"Successfully fetched content from {len(found_readmes)} README files")
    return True, all_readme_content
