"""
module_query_obtention/submodule_github_query/subsubmodule_recent_work/fetch_recent_commits.py

This module is responsible for fetching recent commits from a GitHub repository
and extracting relevant information for query generation.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Import parameters
from module_query_obtention.submodule_github_query.subsubmodule_recent_work.parameters import (
    MAX_COMMITS,
    MAX_COMMITS_QUERY_LENGTH,
    CONTEXT_LINES,
    MAX_FILES_PER_COMMIT,
    MAX_DIFF_SIZE_PER_FILE,
    COMMIT_INFO_TO_INCLUDE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_recent_commits(repo_path: str or Path) -> List[str]:
    """
    Get the hashes of the most recent commits in the repository.
    
    Args:
        repo_path: Path to the cloned repository
        
    Returns:
        List of commit hashes
    """
    # Convert to Path object if string
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)
    
    # Ensure the repository exists
    if not repo_path.exists() or not repo_path.is_dir():
        logger.error(f"Repository path does not exist: {repo_path}")
        return []
    
    try:
        # Change to the repository directory
        original_dir = os.getcwd()
        os.chdir(repo_path)
        
        # Get the most recent commits
        cmd = ["git", "log", f"-{MAX_COMMITS}", "--format=%H"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Change back to the original directory
        os.chdir(original_dir)
        
        # Parse the output
        commit_hashes = result.stdout.strip().split('\n')
        logger.info(f"Found {len(commit_hashes)} recent commits")
        return commit_hashes
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running git command: {e}")
        # Make sure we change back to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return []
    except Exception as e:
        logger.error(f"Error getting recent commits: {e}")
        # Make sure we change back to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return []

def get_commit_info(repo_path: str or Path, commit_hash: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific commit.
    
    Args:
        repo_path: Path to the cloned repository
        commit_hash: Hash of the commit to analyze
        
    Returns:
        Dictionary with commit information
    """
    # Convert to Path object if string
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)
    
    # Ensure the repository exists
    if not repo_path.exists() or not repo_path.is_dir():
        logger.error(f"Repository path does not exist: {repo_path}")
        return {}
    
    try:
        # Change to the repository directory
        original_dir = os.getcwd()
        os.chdir(repo_path)
        
        commit_info = {'hash': commit_hash}
        
        # Get commit message if needed
        if 'message' in COMMIT_INFO_TO_INCLUDE:
            cmd_message = ["git", "log", "-1", "--pretty=format:%s%n%n%b", commit_hash]
            result_message = subprocess.run(cmd_message, capture_output=True, text=True, check=True)
            commit_info['message'] = result_message.stdout.strip()
        
        # Get changed files if needed
        if 'files' in COMMIT_INFO_TO_INCLUDE or 'diff' in COMMIT_INFO_TO_INCLUDE or 'context' in COMMIT_INFO_TO_INCLUDE:
            cmd_files = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash]
            result_files = subprocess.run(cmd_files, capture_output=True, text=True, check=True)
            files = result_files.stdout.strip().split('\n')
            
            # Limit the number of files
            if len(files) > MAX_FILES_PER_COMMIT:
                logger.info(f"Limiting files from {len(files)} to {MAX_FILES_PER_COMMIT}")
                files = files[:MAX_FILES_PER_COMMIT]
            
            commit_info['files'] = files
        
        # Change back to the original directory
        os.chdir(original_dir)
        
        return commit_info
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running git command: {e}")
        # Make sure we change back to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return {}
    except Exception as e:
        logger.error(f"Error getting commit info: {e}")
        # Make sure we change back to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return {}

def get_file_diff(repo_path: str or Path, commit_hash: str, file_path: str) -> str:
    """
    Get the diff for a specific file in a commit.
    
    Args:
        repo_path: Path to the cloned repository
        commit_hash: Hash of the commit to analyze
        file_path: Path to the file to get diff for
        
    Returns:
        String containing the diff with context
    """
    # Convert to Path object if string
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)
    
    # Ensure the repository exists
    if not repo_path.exists() or not repo_path.is_dir():
        logger.error(f"Repository path does not exist: {repo_path}")
        return ""
    
    try:
        # Change to the repository directory
        original_dir = os.getcwd()
        os.chdir(repo_path)
        
        # Get the diff with context
        # Use git show with the correct syntax to show the diff for a specific file
        cmd = ["git", "show", f"-U{CONTEXT_LINES}", f"{commit_hash}" , "--", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Change back to the original directory
        os.chdir(original_dir)
        
        # Check if command was successful
        if result.returncode != 0:
            logger.warning(f"Error getting diff for {file_path}: {result.stderr}")
            return ""
        
        # Limit the diff size
        diff = result.stdout
        if len(diff) > MAX_DIFF_SIZE_PER_FILE:
            logger.info(f"Truncating diff from {len(diff)} to {MAX_DIFF_SIZE_PER_FILE} characters")
            diff = diff[:MAX_DIFF_SIZE_PER_FILE] + "\n[... truncated ...]"
        
        return diff
        
    except Exception as e:
        logger.error(f"Error getting file diff: {e}")
        # Make sure we change back to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return ""

def is_binary_file(file_path: str) -> bool:
    """
    Check if a file is binary based on its extension.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file is likely binary, False otherwise
    """
    binary_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
        '.pdf', '.zip', '.tar', '.gz', '.rar', '.7z',
        '.exe', '.dll', '.so', '.dylib',
        '.pyc', '.pyd', '.pyo',
        '.class', '.jar',
        '.mp3', '.mp4', '.avi', '.mov', '.flv', '.wav',
        '.db', '.sqlite', '.sqlite3'
    }
    
    _, ext = os.path.splitext(file_path.lower())
    return ext in binary_extensions

def fetch_recent_commits_info(repo_path: str or Path) -> Tuple[str, int]:
    """
    Fetch information about recent commits and format it as a query string.
    
    Args:
        repo_path: Path to the cloned repository
        
    Returns:
        Tuple of (formatted query string, number of commits processed)
    """
    # Convert to Path object if string
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)
    
    # Ensure the repository exists
    if not repo_path.exists() or not repo_path.is_dir():
        logger.error(f"Repository path does not exist: {repo_path}")
        return "", 0
    
    logger.info(f"Fetching recent commits from repository: {repo_path}")
    
    # Get recent commit hashes
    commit_hashes = get_recent_commits(repo_path)
    if not commit_hashes:
        logger.warning("No recent commits found")
        return "", 0
    
    # Build the query string with a detailed description
    query = """##############################################################################
#                     RECENT REPOSITORY ACTIVITY ANALYSIS                      #
##############################################################################

This section contains an analysis of the most recent commits in the repository.
Recent commits provide critical insight into the active development focus and
current priorities of the project. They reveal:

1. CURRENT DEVELOPMENT PRIORITIES: The files and features that developers are
   actively working on right now, indicating the most important aspects of the
   project from the developers' perspective.

2. LATEST TECHNICAL CHALLENGES: The specific problems being solved and
   implementation details being refined, showing the technical hurdles the
   project is currently addressing.

3. EVOLUTION OF THE CODEBASE: How the project is changing over time, including
   new features, bug fixes, refactoring, and architectural changes.

4. DEVELOPER INTENT: Through commit messages and code changes, you can see the
   reasoning behind modifications and the direction the project is heading.

The following commits represent the most recent development activity, with
detailed information about what files were changed and the specific code
modifications made.

"""
    commits_processed = 0
    
    for commit_hash in commit_hashes:
        # Check if we've reached the maximum query length
        if len(query) >= MAX_COMMITS_QUERY_LENGTH:
            logger.info(f"Reached maximum query length ({MAX_COMMITS_QUERY_LENGTH} characters)")
            break
        
        # Get commit information
        commit_info = get_commit_info(repo_path, commit_hash)
        if not commit_info:
            logger.warning(f"Could not get information for commit {commit_hash}")
            continue
        
        # Start the commit section with a clear separator
        commit_section = "\n" + "=" * 80 + "\n"
        commit_section += "NEW COMMIT\n"
        commit_section += "=" * 80 + "\n\n"
        
        # Add commit message if requested
        if 'message' in COMMIT_INFO_TO_INCLUDE and 'message' in commit_info:
            commit_section += f"COMMIT MESSAGE:\n{commit_info['message']}\n\n"
        
        # Add file changes if requested
        if 'files' in COMMIT_INFO_TO_INCLUDE and 'files' in commit_info:
            commit_section += "CHANGED FILES:\n"
            
            for file_path in commit_info['files']:
                # Skip binary files
                if is_binary_file(file_path):
                    commit_section += f"- {file_path} [binary file - diff skipped]\n"
                    continue
                
                commit_section += f"- {file_path}\n"
                
                # Get file diff if requested
                if ('diff' in COMMIT_INFO_TO_INCLUDE or 'context' in COMMIT_INFO_TO_INCLUDE):
                    diff = get_file_diff(repo_path, commit_hash, file_path)
                    if diff and diff.strip():
                        # Extract only the actual diff content, removing git metadata
                        diff_lines = diff.split('\n')
                        cleaned_diff_lines = []
                        capture = False
                        
                        for line in diff_lines:
                            # Skip git metadata lines
                            if line.startswith('commit ') or \
                               line.startswith('Author: ') or \
                               line.startswith('Date: ') or \
                               line.startswith('index ') or \
                               line.startswith('diff --git '):
                                continue
                                
                            # Start capturing when we see the actual diff
                            if line.startswith('+++') or line.startswith('---') or \
                               line.startswith('+') or line.startswith('-') or \
                               line.startswith(' '):
                                capture = True
                                
                            if capture:
                                cleaned_diff_lines.append(line)
                        
                        if cleaned_diff_lines:
                            commit_section += "```diff\n"
                            commit_section += '\n'.join(cleaned_diff_lines)
                            commit_section += "\n```\n\n"
        
        # Add to query if it doesn't exceed the maximum length
        if len(query) + len(commit_section) <= MAX_COMMITS_QUERY_LENGTH:
            query += commit_section
            commits_processed += 1
        else:
            # Truncate the commit section to fit within the maximum length
            available_space = MAX_COMMITS_QUERY_LENGTH - len(query)
            if available_space > 100:  # Only add if we have reasonable space
                truncated_section = commit_section[:available_space - 20] + "\n[... truncated ...]"
                query += truncated_section
                commits_processed += 1
            break
    
    logger.info(f"Processed {commits_processed} of {len(commit_hashes)} recent commits")
    logger.info(f"Generated query of length {len(query)} characters")
    
    return query, commits_processed
