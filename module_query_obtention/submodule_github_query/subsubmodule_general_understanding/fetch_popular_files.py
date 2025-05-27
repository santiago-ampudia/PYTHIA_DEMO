"""
module_query_obtention/submodule_github_query/subsubmodule_general_understanding/fetch_popular_files.py

This module is responsible for finding the "popular" files in a repository - files with the most commits.
It returns a list of file paths for further processing by other modules.
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
import re

from . import parameters
from .fetch_main_files import is_binary_file, is_readme_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_popular_files(repo_dir: str or Path) -> List[Path]:
    """
    Find the files with the most commits in a repository.
    
    Args:
        repo_dir: Path to the cloned repository directory (can be string or Path object)
        
    Returns:
        List of Path objects pointing to the files with the most commits
    """
    # Convert string path to Path object if needed
    if isinstance(repo_dir, str):
        repo_dir = Path(repo_dir)
    
    # Ensure the repository directory exists
    if not repo_dir.exists() or not repo_dir.is_dir():
        logger.error(f"Repository directory does not exist: {repo_dir}")
        return []
    
    logger.info(f"Finding popular files in repository: {repo_dir}")
    
    # Dictionary to store file paths, their commit counts, and directory level
    file_info = {}
    root_level = len(repo_dir.parts)
    
    try:
        # Change to the repository directory
        original_dir = os.getcwd()
        os.chdir(repo_dir)
        
        # Run git command to get file commit counts
        # This command lists all files with their commit counts, sorted by commit count (descending)
        cmd = ["git", "log", "--pretty=format:", "--name-only"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Process the output
        files = result.stdout.strip().split('\n')
        
        # Count occurrences of each file
        for file in files:
            if not file:  # Skip empty lines
                continue
                
            file_path = repo_dir / file
            
            # Skip files that don't exist anymore
            if not file_path.exists() or not file_path.is_file():
                continue
                
            # Skip hidden files
            if file_path.name.startswith('.'):
                continue
                
            # Skip binary files
            if is_binary_file(file_path):
                continue
                
            # Skip README files (they're processed separately)
            if is_readme_file(file_path):
                continue
                
            # Calculate directory level (depth from repo root)
            level = len(Path(file).parent.parts)
            
            # Get file size (for tie-breaking)
            try:
                file_size = file_path.stat().st_size
            except Exception:
                file_size = 0
            
            # Store or update commit count, level, and size for this file
            if file in file_info:
                file_info[file]['commits'] += 1
            else:
                file_info[file] = {
                    'commits': 1,
                    'level': level,
                    'size': file_size
                }
        
        # Change back to the original directory
        os.chdir(original_dir)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running git command: {e}")
        # Make sure we change back to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return []
    except Exception as e:
        logger.error(f"Error finding popular files: {e}")
        # Make sure we change back to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return []
    
    # Sort files by multiple criteria:
    # 1. Commit count (descending) - primary criterion
    # 2. Directory level (ascending) - secondary criterion for ties in commit count
    # 3. File size (descending) - tertiary criterion for ties in both commit count and level
    sorted_files = sorted(file_info.items(), key=lambda x: (-x[1]['commits'], x[1]['level'], -x[1]['size']))
    
    # Return the top N popular files
    top_files = []
    for file_path_str, info in sorted_files[:parameters.MAX_POPULAR_FILES]:
        file_path = repo_dir / file_path_str
        top_files.append(file_path)
        logger.info(f"Popular file: {file_path.name} with {info['commits']} commits, level {info['level']}, size {info['size']} bytes")
    # Log information about tie-breaking if applicable
    if len(sorted_files) > 1:
        # Check for ties in commit counts
        commit_counts = [info['commits'] for _, info in sorted_files[:parameters.MAX_POPULAR_FILES]]
        has_commit_ties = len(set(commit_counts)) < len(commit_counts)
        
        # Check for ties in both commit counts and levels
        level_counts = [(info['commits'], info['level']) for _, info in sorted_files[:parameters.MAX_POPULAR_FILES]]
        has_level_ties = len(set(level_counts)) < len(level_counts)
        
        if has_commit_ties and has_level_ties:
            logger.info("Some files had the same commit count and directory level. Used file size as tiebreaker (larger size = higher priority).")
        elif has_commit_ties:
            logger.info("Some files had the same commit count. Used directory level as tiebreaker (lower level = higher priority).")
            
        # If all files have the same commit count, explicitly mention we're using level and size
        if len(set(commit_counts)) == 1:
            if len(set([info['level'] for _, info in sorted_files[:parameters.MAX_POPULAR_FILES]])) == 1:
                logger.warning(f"All files have the same commit count ({commit_counts[0]}) and directory level. Using file size as tiebreaker.")
            else:
                logger.warning(f"All files have the same commit count ({commit_counts[0]}). Using directory level as tiebreaker.")
    
    logger.info(f"Found {len(top_files)} popular files")
    return top_files

def get_popular_files(repo_dir: str or Path) -> List[Path]:
    """
    Find the files with the most commits in a repository and return their paths.
    
    Args:
        repo_dir: Path to the cloned repository directory (can be string or Path object)
        
    Returns:
        List of Path objects pointing to the files with the most commits
    """
    return find_popular_files(repo_dir)
