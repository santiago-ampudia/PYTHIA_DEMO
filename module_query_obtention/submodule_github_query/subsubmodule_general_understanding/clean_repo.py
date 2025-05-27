#!/usr/bin/env python3
"""
Module for cleaning up cloned GitHub repositories.
This module handles the deletion of temporary repository directories.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('github_repo_cleaner')

def clean_repository(repo_path: str) -> Tuple[bool, str]:
    """
    Clean up a cloned repository by removing its directory.
    
    Args:
        repo_path: Path to the cloned repository directory
        
    Returns:
        A tuple of (success, message)
    """
    if not repo_path:
        error_msg = "No repository path provided for cleanup"
        logger.error(error_msg)
        return False, error_msg
    
    repo_dir = Path(repo_path)
    
    # Check if the directory exists
    if not repo_dir.exists():
        logger.warning(f"Repository directory does not exist: {repo_dir}")
        return True, f"Repository directory does not exist: {repo_dir}"
    
    try:
        logger.info(f"Cleaning up repository directory: {repo_dir}")
        shutil.rmtree(repo_dir)
        logger.info(f"Successfully removed repository directory: {repo_dir}")
        return True, f"Successfully removed repository directory: {repo_dir}"
    except Exception as e:
        error_msg = f"Error cleaning up repository directory {repo_dir}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
