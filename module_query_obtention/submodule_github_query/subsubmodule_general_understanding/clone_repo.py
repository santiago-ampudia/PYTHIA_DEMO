#!/usr/bin/env python3
"""
Module for cloning GitHub repositories.
This module handles the authentication and cloning of GitHub repositories.
"""

import os
import logging
import requests
import subprocess
import urllib.parse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union

# Import parameters
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.parameters import (
    GITHUB_API_URL,
    GITHUB_API_ACCEPT_HEADER,
    CLONE_DEPTH
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('github_repo_cloner')

def get_authenticated_user_info(github_token: str) -> Tuple[bool, Union[Dict[str, Any], str]]:
    """
    Get information about the authenticated user using the GitHub API.
    
    Args:
        github_token: GitHub API token
        
    Returns:
        A tuple of (success, user_info or error_message)
    """
    if not github_token:
        return False, "No GitHub token provided"
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": GITHUB_API_ACCEPT_HEADER
    }
    
    try:
        response = requests.get(f"{GITHUB_API_URL}/user", headers=headers)
        response.raise_for_status()
        user_info = response.json()
        logger.info(f"Successfully retrieved user info for {user_info.get('login')}")
        return True, user_info
    except requests.RequestException as e:
        error_msg = f"Failed to get user info: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def clone_repository(repo_name: str, github_token: Optional[str] = None) -> Tuple[bool, str]:
    """
    Clone a GitHub repository to a temporary directory.
    
    Args:
        repo_name: Name of the GitHub repository (can be just 'repo' or 'username/repo')
        github_token: GitHub API token for authentication (REQUIRED)
        
    Returns:
        A tuple of (success, repo_path or error_message)
    """
    # STRICT REQUIREMENT: We must have a GitHub token
    if not github_token:
        error_msg = "GitHub token is required for repository access. No token provided."
        logger.error(error_msg)
        return False, error_msg
    
    # Create a clean temporary directory for the repo
    safe_repo_name = repo_name.replace('/', '_')
    repo_dir = Path(f"/tmp/{safe_repo_name}")
    
    # Ensure the directory is properly cleaned up
    try:
        if repo_dir.exists():
            logger.info(f"Removing existing repository directory: {repo_dir}")
            import shutil
            shutil.rmtree(repo_dir)
    except Exception as e:
        logger.error(f"Error cleaning up directory {repo_dir}: {str(e)}")
        # Try a different directory name if cleanup fails
        import time
        safe_repo_name = f"{safe_repo_name}_{int(time.time())}"
        repo_dir = Path(f"/tmp/{safe_repo_name}")
        logger.info(f"Using alternative directory: {repo_dir}")
    
    # Make sure parent directory exists
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # If the repository name already includes the owner (username/repo format)
    if '/' in repo_name:
        full_repo_name = repo_name
        logger.info(f"Repository name already includes owner: {full_repo_name}")
    else:
        # STRICT REQUIREMENT: We must get the user's username from GitHub
        logger.info("Repository name doesn't include owner. Getting authenticated user info...")
        success, user_info = get_authenticated_user_info(github_token)
        
        if not success or not isinstance(user_info, dict) or 'login' not in user_info:
            error_msg = f"Failed to get authenticated user info from GitHub. Cannot proceed without username."
            if isinstance(user_info, str):
                error_msg += f" Error: {user_info}"
            logger.error(error_msg)
            return False, error_msg
        
        # We have the username, now construct the full repository name
        username = user_info['login']
        logger.info(f"Successfully retrieved GitHub username: {username}")
        full_repo_name = f"{username}/{repo_name}"
        logger.info(f"Constructed full repository name: {full_repo_name}")
    
    # Clone the repository using the authenticated URL
    clone_url = f"https://{github_token}@github.com/{full_repo_name}.git"
    logger.info(f"Attempting to clone from: {clone_url.replace(github_token, '***')}")
    
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", str(CLONE_DEPTH), clone_url, str(repo_dir)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60  # Increased timeout for deeper clones
        )
        logger.info(f"Successfully cloned repository: {full_repo_name}")
        return True, str(repo_dir)
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to clone repository {full_repo_name}: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg
    except subprocess.TimeoutExpired:
        error_msg = f"Timeout while cloning repository {full_repo_name}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Error cloning repository {full_repo_name}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
