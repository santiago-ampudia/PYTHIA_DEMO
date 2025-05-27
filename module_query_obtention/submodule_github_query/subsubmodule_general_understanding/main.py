"""
module_query_obtention/submodule_github_query/subsubmodule_general_understanding/main.py

This module is responsible for generating a query based on the content of a GitHub repository.
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
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.parameters import (
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
    MAX_MAIN_FILES,
    MAIN_FILES_LINES
)

# Import the repository handling functions
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.clone_repo import clone_repository
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.fetch_readme import fetch_readme_content
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.clean_repo import clean_repository
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.fetch_main_files import get_main_files
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.fetch_popular_files import get_popular_files
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.extract_snippets import get_code_snippets
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.generate_github_general_query import generate_final_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('github_query_generator')

def generate_repo_based_query(repo_name: str, github_token: Optional[str] = None) -> str:
    """
    Generate a query based on the repository name and README content.
    
    Args:
        repo_name: Name of the GitHub repository
        github_token: GitHub API token for authentication (REQUIRED)
        
    Returns:
        A query string based on the repository content or raises an exception if unable to access the repository
    """
    logger.info(f"Generating query for repository: {repo_name}")
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
        
        # Step 2: Fetch the README content from the cloned repository
        logger.info(f"Fetching README content from cloned repository")
        readme_success, readme_content = fetch_readme_content(repo_path)
        
        if not readme_success:
            error_msg = f"Failed to fetch README content: {readme_content}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Step 3: Find main files and return their paths
        logger.info("Finding main files in the repository")
        main_files = get_main_files(repo_path)
        
        # Step 4: Find popular files (files with most commits) and return their paths
        logger.info("Finding popular files in the repository")
        popular_files = get_popular_files(repo_path)
        
        # Step 5: Extract code snippets from main files and popular files
        logger.info("Extracting code snippets from main and popular files")
        main_snippets, popular_snippets, total_snippet_size = get_code_snippets(repo_path, main_files, popular_files)
        
        # Log detailed information about the snippets
        logger.info(f"Extracted {len(main_snippets)} snippets from main files")
        for i, snippet in enumerate(main_snippets):
            logger.info(f"Main snippet {i+1}: {snippet.pattern_description} from {snippet.relative_path} (lines {snippet.line_start}-{snippet.line_end})")
            
        logger.info(f"Extracted {len(popular_snippets)} snippets from popular files")
        for i, snippet in enumerate(popular_snippets):
            logger.info(f"Popular snippet {i+1}: {snippet.pattern_description} from {snippet.relative_path} (lines {snippet.line_start}-{snippet.line_end})")
        
        # Format snippets into strings
        from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.extract_snippets import format_snippets
        main_snippets_content = format_snippets(main_snippets)
        popular_snippets_content = format_snippets(popular_snippets)
        
        logger.info(f"Main snippets content length: {len(main_snippets_content)}")
        logger.info(f"Popular snippets content length: {len(popular_snippets_content)}")
        
        # Step 6: Generate the query from the README content and code snippets
        logger.info("Using README content and code snippets for query generation")
        query = ""
        
        # 1. README Content Section
        if readme_content:
            readme_header = """
=================================================================
                       REPOSITORY DOCUMENTATION
=================================================================

The following section contains the documentation from README files found in the repository. 
These files provide an overview of the project, its purpose, installation instructions, 
usage examples, and other important information directly from the project maintainers. 
The README files are presented in order of their location in the repository structure, 
starting from the root directory and proceeding to deeper levels.
"""
            query = readme_header + "\n\n" + readme_content
            
        # 2. Main Files Snippets Section
        if main_snippets:
            main_snippets_header = """
=================================================================
                 CODE SNIPPETS FROM MAIN FILES
=================================================================

The following section contains important code snippets extracted from the main files 
of the repository. These files were identified based on their names matching common 
patterns for entry points, main scripts, or core functionality (such as 'main', 'app', 
'train', etc.). The snippets include main execution blocks, function definitions, 
class declarations, and other significant code patterns that illustrate the core 
functionality of the project. Each snippet is labeled with its source file, line 
numbers, and the type of code pattern it represents.
"""
            query += "\n\n" + main_snippets_header + "\n\n" + main_snippets_content
            
        # 3. Popular Files Snippets Section
        if popular_snippets:
            popular_snippets_header = """
=================================================================
           CODE SNIPPETS FROM MOST FREQUENTLY MODIFIED FILES
=================================================================

The following section contains important code snippets extracted from the most 
frequently modified files in the repository. These files were identified based on 
the number of commits they have received, indicating they are actively developed 
and likely contain important functionality. The snippets include main execution blocks, 
function definitions, class declarations, and other significant code patterns that 
illustrate how the project evolves over time. Each snippet is labeled with its source 
file, line numbers, and the type of code pattern it represents.
"""
            query += "\n\n" + popular_snippets_header + "\n\n" + popular_snippets_content
        
        # Log the main files found
        if main_files:
            main_file_names = [f.name for f in main_files]
            logger.info(f"Found main files: {main_file_names}")
        
        # Log the popular files found
        if popular_files:
            popular_file_names = [f.name for f in popular_files]
            logger.info(f"Found popular files: {popular_file_names}")
            
        # Log the snippets found
        total_snippets = len(main_snippets) + len(popular_snippets)
        if total_snippets > 0:
            logger.info(f"Extracted {total_snippets} code snippets in total")
            logger.info(f"  - {len(main_snippets)} snippets from main files")
            logger.info(f"  - {len(popular_snippets)} snippets from popular files")
            
            # Store snippet metadata for future reference
            main_snippet_metadata = [{
                'file': str(s.relative_path),
                'pattern': s.pattern_description,
                'lines': f"{s.line_start}-{s.line_end}",
                'type': 'main_file'
            } for s in main_snippets]
            
            popular_snippet_metadata = [{
                'file': str(s.relative_path),
                'pattern': s.pattern_description,
                'lines': f"{s.line_start}-{s.line_end}",
                'type': 'popular_file'
            } for s in popular_snippets]
            
            all_snippet_metadata = main_snippet_metadata + popular_snippet_metadata
            logger.debug(f"Snippet metadata: {all_snippet_metadata}")
            
        # 3. File Information Section
        files_header = """
=================================================================
                       REPOSITORY FILE STRUCTURE
=================================================================

This section provides information about the key files in the repository.
"""
        query += "\n\n" + files_header
        
        # Add main files information
        if main_files:
            main_files_description = """
-----------------------------------------------------------------
MAIN FILES:
The following files were identified as main files in the repository based on 
their names matching common patterns for entry points, main scripts, or core 
functionality (such as 'main', 'app', 'train', etc.). These files are likely 
to contain the primary functionality and entry points of the project.
-----------------------------------------------------------------
"""
            query += "\n" + main_files_description
            for file in main_files:
                query += f"\n- {file.name} ({file.relative_to(repo_path)})" 
        
        # Add popular files information
        if popular_files:
            popular_files_description = """
-----------------------------------------------------------------
MOST FREQUENTLY MODIFIED FILES:
The following files have been identified as the most frequently modified files 
in the repository based on the number of commits. These files represent the most 
actively developed parts of the codebase and are likely to contain core functionality 
that is frequently updated or improved.
-----------------------------------------------------------------
"""
            query += "\n\n" + popular_files_description
            for file in popular_files:
                query += f"\n- {file.name} ({file.relative_to(repo_path)})"
        
        # Ensure the query is not too long
        if len(query) > MAX_QUERY_LENGTH:
            logger.info(f"README content too long ({len(query)} chars), truncating to {MAX_QUERY_LENGTH} chars")
            query = query[:MAX_QUERY_LENGTH]
        
        # Ensure the query is not too short
        if len(query) < MIN_QUERY_LENGTH:
            logger.info(f"README content too short ({len(query)} chars), using with repository name")
            query = f"{query}\n\nLatest research papers related to {repo_name} technologies and methodologies"
        
        # Step 7: Generate the final query using GPT-4
        logger.info("Generating final query using GPT-4...")
        final_query = generate_final_query(query)
        
        # Step 8: Return the final query
        logger.info("Returning the final generated query")
        return final_query
    
    finally:
        # Step 5: Clean up the repository directory
        if repo_path:
            logger.info(f"Cleaning up repository directory: {repo_path}")
            clean_success, clean_message = clean_repository(repo_path)
            if not clean_success:
                logger.warning(f"Failed to clean up repository: {clean_message}")
            else:
                logger.info(f"Repository cleanup successful: {clean_message}")
