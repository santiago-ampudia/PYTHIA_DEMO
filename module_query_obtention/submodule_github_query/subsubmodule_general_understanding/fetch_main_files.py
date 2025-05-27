"""
module_query_obtention/submodule_github_query/subsubmodule_general_understanding/fetch_main_files.py

This module is responsible for finding the most important files in a repository
using regex patterns and extracting their content for query generation.
"""

import os
import re
import logging
import mimetypes
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from . import parameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_binary_file(file_path: Path) -> bool:
    """
    Check if a file is binary based on its extension.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file has a binary extension, False otherwise
    """
    # Only check by extension
    binary_extensions = (
        # Binary files
        '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
        '.zip', '.tar', '.gz', '.rar', '.7z',
        '.exe', '.dll', '.so', '.dylib',
        '.pyc', '.pyd', '.pyo',
        '.class', '.jar',
        '.mp3', '.mp4', '.avi', '.mov', '.flv', '.wav',
        '.db', '.sqlite', '.sqlite3',
        '.log'
    )
    
    # Only consider actual binary files as binary, not text files like .md or .txt
    return file_path.suffix.lower() in binary_extensions


def is_readme_file(file_path: Path) -> bool:
    """
    Check if a file is a README file.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file is a README file, False otherwise
    """
    # Check if the filename (case-insensitive) starts with 'readme'
    return file_path.name.lower().startswith('readme')

def find_main_files(repo_dir: str or Path) -> List[Path]:
    """
    Find the most important files in a repository based on predefined patterns.
    Searches level-by-level for each pattern before moving to the next pattern.
    
    Args:
        repo_dir: Path to the cloned repository directory (can be string or Path object)
        
    Returns:
        List of Path objects pointing to the most important files
    """
    # Convert string path to Path object if needed
    if isinstance(repo_dir, str):
        repo_dir = Path(repo_dir)
    
    # Ensure the repository directory exists
    if not repo_dir.exists() or not repo_dir.is_dir():
        logger.error(f"Repository directory does not exist: {repo_dir}")
        return []
    
    found_files = []
    
    # For each pattern in the list
    for pattern in parameters.MAIN_FILE_PATTERNS:
        # Skip if we've already found enough files
        if len(found_files) >= parameters.MAX_MAIN_FILES:
            logger.info(f"Reached maximum number of files ({parameters.MAX_MAIN_FILES}), stopping search")
            break
            
        logger.info(f"Searching for files matching pattern: '{pattern}'")
        pattern_lower = pattern.lower()
        pattern_files = []
        
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
        
        # Now search each level for the current pattern
        max_level = max(all_dirs_by_level.keys()) if all_dirs_by_level else 0
        
        for level in range(max_level + 1):
            # If this level doesn't exist, skip it
            if level not in all_dirs_by_level:
                continue
                
            logger.info(f"Searching level {level} for pattern '{pattern}'")
            
            # Search all directories at this level
            for directory in all_dirs_by_level[level]:
                # Get all files in this directory (not recursive)
                try:
                    for file_path in directory.iterdir():
                        # Skip directories and hidden files
                        if file_path.is_dir() or file_path.name.startswith('.'):
                            continue
                            
                        # Skip binary files
                        if is_binary_file(file_path):
                            continue
                            
                        # Skip README files (they're processed separately)
                        if is_readme_file(file_path):
                            logger.info(f"Skipping README file: {file_path}")
                            continue
                            
                        # Check if the filename contains the pattern
                        if pattern_lower in file_path.name.lower():
                            pattern_files.append(file_path)
                            logger.info(f"Found file matching pattern '{pattern}': {file_path}")
                except Exception as e:
                    logger.error(f"Error accessing directory {directory}: {str(e)}")
            
            # If we've found files at this level, we can stop searching deeper levels for this pattern
            if pattern_files and len(pattern_files) >= parameters.MAX_MAIN_FILES:
                logger.info(f"Found enough files at level {level}, stopping search for pattern '{pattern}'")
                break
        
        # Log the results for this pattern
        if pattern_files:
            logger.info(f"Found {len(pattern_files)} files matching pattern '{pattern}'")
            
            # Add the files found for this pattern to the overall results
            num_to_add = min(len(pattern_files), parameters.MAX_MAIN_FILES - len(found_files))
            found_files.extend(pattern_files[:num_to_add])
            logger.info(f"Found {len(pattern_files)} files matching pattern '{pattern}', added {num_to_add} to results")
        else:
            logger.info(f"No files found matching pattern '{pattern}'")
            
        # If we've reached the maximum number of files, stop searching
        if len(found_files) >= parameters.MAX_MAIN_FILES:
            logger.info(f"Reached maximum number of files ({parameters.MAX_MAIN_FILES}), stopping search")
            break
    
    # If we didn't find any files, just log it
    if not found_files:
        logger.info("No files found matching any of the patterns")
    
    logger.info(f"Found {len(found_files)} main files: {[f.name for f in found_files]}")
    return found_files

def get_main_files(repo_dir: str or Path) -> List[Path]:
    """
    Find the most important files in a repository and return their paths.
    
    Args:
        repo_dir: Path to the cloned repository directory (can be string or Path object)
        
    Returns:
        List of Path objects pointing to the most important files
    """
    return find_main_files(repo_dir)
