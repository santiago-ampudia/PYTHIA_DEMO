"""
module_query_obtention/submodule_github_query/subsubmodule_recent_work/fetch_hot_files.py

This module is responsible for identifying and analyzing the "hot files" in a repository,
which are files that have been modified most frequently in recent commits.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set
from collections import Counter, defaultdict

# Import parameters
from module_query_obtention.submodule_github_query.subsubmodule_recent_work.parameters import (
    MAX_COMMITS,
    MAX_HOT_FILES,
    HOT_FILES_QUERY_LENGTH,
    HOT_FILES_CONTEXT_LINES,
    MAX_HOT_FILE_SIZE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_recent_commit_hashes(repo_path: str or Path) -> List[str]:
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

def get_files_changed_in_commit(repo_path: str or Path, commit_hash: str) -> List[str]:
    """
    Get the list of files changed in a specific commit.
    
    Args:
        repo_path: Path to the cloned repository
        commit_hash: Hash of the commit to analyze
        
    Returns:
        List of file paths that were changed
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
        
        # Get changed files
        cmd = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Change back to the original directory
        os.chdir(original_dir)
        
        # Parse the output
        files = [f for f in result.stdout.strip().split('\n') if f]
        return files
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running git command: {e}")
        # Make sure we change back to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return []
    except Exception as e:
        logger.error(f"Error getting files changed in commit: {e}")
        # Make sure we change back to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return []

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

def get_file_content(repo_path: str or Path, file_path: str) -> str:
    """
    Get the content of a file from the repository.
    
    Args:
        repo_path: Path to the cloned repository
        file_path: Path to the file
        
    Returns:
        String containing the file content
    """
    # Convert to Path object if string
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)
    
    # Ensure the repository exists
    if not repo_path.exists() or not repo_path.is_dir():
        logger.error(f"Repository path does not exist: {repo_path}")
        return ""
    
    # Check if the file exists
    full_path = repo_path / file_path
    if not full_path.exists() or not full_path.is_file():
        logger.error(f"File does not exist: {full_path}")
        return ""
    
    # Check if it's a binary file
    if is_binary_file(file_path):
        logger.info(f"Skipping binary file: {file_path}")
        return ""
    
    try:
        # Read the file content
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Limit the content size
        if len(content) > MAX_HOT_FILE_SIZE:
            logger.info(f"Truncating file content from {len(content)} to {MAX_HOT_FILE_SIZE} characters")
            content = content[:MAX_HOT_FILE_SIZE] + "\n[... truncated ...]"
        
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return ""

def get_commit_timestamp(repo_path: str or Path, commit_hash: str) -> int:
    """
    Get the timestamp of a commit.
    
    Args:
        repo_path: Path to the cloned repository
        commit_hash: Hash of the commit
        
    Returns:
        Unix timestamp of the commit
    """
    # Convert to Path object if string
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)
    
    try:
        # Change to the repository directory
        original_dir = os.getcwd()
        os.chdir(repo_path)
        
        # Get the commit timestamp
        cmd = ["git", "show", "-s", "--format=%ct", commit_hash]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Change back to the original directory
        os.chdir(original_dir)
        
        # Parse the output
        timestamp = int(result.stdout.strip())
        return timestamp
        
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Error getting commit timestamp: {e}")
        # Make sure we change back to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return 0
    except Exception as e:
        logger.error(f"Error getting commit timestamp: {e}")
        # Make sure we change back to the original directory
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return 0

def fetch_hot_files(repo_path: str or Path) -> List[Dict[str, Any]]:
    """
    Identify the "hot files" in a repository - files that have been modified most frequently in recent commits.
    
    Args:
        repo_path: Path to the cloned repository
        
    Returns:
        List of dictionaries containing information about hot files, each with keys:
        - 'path': Path to the file
        - 'changes': Number of commits the file appears in
        - 'last_modified': Timestamp of the most recent commit that modified the file
        - 'content': Content of the file (if available and not binary)
    """
    # Convert to Path object if string
    if isinstance(repo_path, str):
        repo_path = Path(repo_path)
    
    # Ensure the repository exists
    if not repo_path.exists() or not repo_path.is_dir():
        logger.error(f"Repository path does not exist: {repo_path}")
        return []
    
    logger.info(f"Identifying hot files in repository: {repo_path}")
    
    # Get recent commit hashes
    commit_hashes = get_recent_commit_hashes(repo_path)
    if not commit_hashes:
        logger.warning("No recent commits found")
        return []
    
    # Track files changed in each commit
    file_changes = Counter()
    file_last_commit = {}
    
    # Process each commit
    for commit_hash in commit_hashes:
        # Get files changed in this commit
        files = get_files_changed_in_commit(repo_path, commit_hash)
        
        # Update the counter and last commit timestamp for each file
        for file_path in files:
            if not is_binary_file(file_path):
                file_changes[file_path] += 1
                if file_path not in file_last_commit:
                    timestamp = get_commit_timestamp(repo_path, commit_hash)
                    file_last_commit[file_path] = timestamp
    
    # Get the most frequently changed files
    most_common_files = file_changes.most_common()
    
    # If there are ties, sort by most recent commit
    hot_file_paths = []
    current_count = -1
    tied_files = []
    
    for file_path, count in most_common_files:
        if count != current_count:
            # Process any tied files from the previous count
            if tied_files:
                # Sort tied files by recency (highest timestamp first)
                tied_files.sort(key=lambda f: file_last_commit.get(f, 0), reverse=True)
                hot_file_paths.extend(tied_files)
                tied_files = []
            
            current_count = count
        
        tied_files.append(file_path)
        
        # If we have enough files, stop
        if len(hot_file_paths) + len(tied_files) >= MAX_HOT_FILES:
            # Sort remaining tied files by recency
            tied_files.sort(key=lambda f: file_last_commit.get(f, 0), reverse=True)
            # Add only as many as needed to reach MAX_HOT_FILES
            hot_file_paths.extend(tied_files[:MAX_HOT_FILES - len(hot_file_paths)])
            break
    
    # Process any remaining tied files if we haven't reached the limit
    if tied_files and len(hot_file_paths) < MAX_HOT_FILES:
        tied_files.sort(key=lambda f: file_last_commit.get(f, 0), reverse=True)
        hot_file_paths.extend(tied_files[:MAX_HOT_FILES - len(hot_file_paths)])
    
    # Limit to MAX_HOT_FILES
    hot_file_paths = hot_file_paths[:MAX_HOT_FILES]
    
    # Create the result list with detailed information about each hot file
    hot_files = []
    
    for file_path in hot_file_paths:
        # Get the file content
        content = get_file_content(repo_path, file_path)
        
        # Create a dictionary with file information
        file_info = {
            'path': file_path,
            'changes': file_changes[file_path],
            'last_modified': file_last_commit.get(file_path, 0),
            'content': content if content else ""
        }
        
        hot_files.append(file_info)
    
    # Log the hot files list (without the full content to keep logs readable)
    logger.info(f"Identified {len(hot_files)} hot files")
    for i, file_info in enumerate(hot_files):
        content_preview = file_info.get('content', '')[:100] + '...' if file_info.get('content') else 'No content'
        logger.info(f"Hot file #{i+1}: {file_info.get('path')} - Modified in {file_info.get('changes')} commits - Last modified: {file_info.get('last_modified')} - Content preview: {content_preview}")
    
    return hot_files
