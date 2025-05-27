"""
module_query_obtention/submodule_github_query/subsubmodule_recent_work/extract_hot_file_snippets.py

This module is responsible for extracting code snippets from hot files.
It uses the extract_snippets module from the general_understanding submodule
but adapts it to work with the hot files list.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Import the extract_snippets module from general_understanding
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.extract_snippets import (
    extract_snippets_from_file,
    Snippet,
    format_snippets
)

# Import parameters from general_understanding
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.parameters import (
    SNIPPET_PATTERNS,
    MAX_MAIN_SNIPPETS_SIZE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_snippets_from_hot_files(repo_path: str or Path, hot_files: List[Dict[str, Any]]) -> Tuple[List[Snippet], int]:
    """
    Extract code snippets from hot files based on predefined patterns.
    
    Args:
        repo_path: Path to the repository directory
        hot_files: List of dictionaries containing information about hot files
        
    Returns:
        Tuple of (snippets, total size in characters)
    """
    # Convert repo_path to Path object if it's a string
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)
    
    # Log the input files
    logger.info(f"Extracting snippets from {len(hot_files)} hot files")
    
    # Create snippet list
    snippets = []
    total_size = 0
    
    # Maximum size for snippets (using the same limit as main files in general_understanding)
    max_size = MAX_MAIN_SNIPPETS_SIZE
    
    # Process each hot file
    for file_info in hot_files:
        file_path = file_info.get('path', '')
        if not file_path:
            continue
        
        # Create full path to the file
        full_path = repo_path / file_path
        
        # Skip if the file doesn't exist
        if not full_path.exists() or not full_path.is_file():
            logger.warning(f"File does not exist: {full_path}")
            continue
        
        # Extract snippets from the file
        logger.info(f"Extracting snippets from hot file: {file_path}")
        file_snippets = extract_snippets_from_file(full_path, SNIPPET_PATTERNS)
        
        # Set the relative path for each snippet
        for snippet in file_snippets:
            snippet.relative_path = file_path
        
        # Add snippets to the list, respecting the size limit
        for snippet in file_snippets:
            snippet_size = len(snippet.get_formatted_content())
            
            # Check if adding this snippet would exceed the size limit
            if total_size + snippet_size > max_size:
                logger.info(f"Reached maximum snippet size ({max_size} characters)")
                break
            
            snippets.append(snippet)
            total_size += snippet_size
    
    logger.info(f"Extracted {len(snippets)} snippets from hot files (total size: {total_size} characters)")
    return snippets, total_size

def generate_hot_files_snippets_query(repo_path: str or Path, hot_files: List[Dict[str, Any]]) -> Tuple[str, int]:
    """
    Generate a query based on snippets extracted from hot files.
    
    Args:
        repo_path: Path to the repository directory
        hot_files: List of dictionaries containing information about hot files
        
    Returns:
        Tuple of (formatted query string, number of snippets processed)
    """
    # Extract snippets from hot files
    snippets, total_size = extract_snippets_from_hot_files(repo_path, hot_files)
    
    if not snippets:
        logger.warning("No snippets extracted from hot files")
        return "", 0
    
    # Format the snippets into a query string with a detailed description
    query = """##############################################################################
#                          HOT FILES CODE ANALYSIS                           #
##############################################################################

This section contains key code snippets from the repository's "hot files" - 
the files that have been modified most frequently in recent commits.

Hot files are particularly significant for understanding the project because:

1. CORE FUNCTIONALITY: Files that are modified frequently often contain the most
   critical functionality of the project. They represent the "beating heart" of
   the codebase that developers continually refine and extend.

2. ACTIVE DEVELOPMENT AREAS: These files highlight the specific components that
   are under active development, indicating where the project's focus and
   priorities currently lie.

3. TECHNICAL COMPLEXITY: Files requiring frequent changes often represent the most
   complex or challenging aspects of the system that need ongoing attention and
   refinement.

4. ARCHITECTURAL SIGNIFICANCE: Hot files frequently represent key architectural
   components or interfaces that connect multiple parts of the system.

5. EVOLVING REQUIREMENTS: Files with high modification frequency often reflect
   areas where requirements are still evolving or being refined based on new
   insights or user feedback.

The following code snippets are extracted from these hot files, focusing on the
most important patterns and structures that define the project's core functionality.
These snippets have been selected to provide a representative view of the key
technical components and patterns used in the most actively developed parts of the
codebase.

"""
    query += format_snippets(snippets)
    
    logger.info(f"Generated hot files snippets query of length {len(query)} characters")
    return query, len(snippets)
