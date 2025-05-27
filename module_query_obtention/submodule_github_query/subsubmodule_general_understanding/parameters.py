"""
module_query_obtention/submodule_github_query/subsubmodule_general_understanding/parameters.py

This file contains parameters and configuration values for the GitHub repository query generation.
"""

# Default query template if no specific analysis is available
DEFAULT_QUERY_TEMPLATE = "Latest research papers related to {repo_name} technologies and methodologies"

# GitHub API configuration
GITHUB_API_URL = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
GITHUB_API_ACCEPT_HEADER = "application/vnd.github+json"

# Repository analysis parameters
MAX_README_SIZE = 10000  # Maximum size of README to analyze (in characters, ~20 pages)
MAX_FILES_TO_ANALYZE = 10  # Maximum number of files to analyze for content
MAX_FILE_SIZE = 2000000  # Maximum size of a file to analyze (in bytes, ~10000 chars or 20 pages)

# Git clone parameters
CLONE_DEPTH = 1000  # Depth of git history to clone (number of commits)

# Query generation parameters
# ----------------------------------------
MAX_QUERY_LENGTH = 10000000  # Maximum length of the generated query (characters)
MIN_QUERY_LENGTH = 20  # Minimum length of the generated query (characters)

# ----------------------------------------
# GPT-4 query generation parameters
# ----------------------------------------
GPT_MODEL = "gpt-4-turbo"  # Model to use for generating the final query
MAX_FINAL_QUERY_LENGTH = 1000  # Maximum length of the final query (tokens)
MIN_FINAL_QUERY_LENGTH = 200  # Minimum length of the final query (tokens)

# Instructions for GPT-4 to generate the final query
GPT_QUERY_INSTRUCTIONS = """
You are an expert in software engineering, computer science, and technical analysis. Your task is to deeply analyze the provided code snippets and repository information to extrapolate and articulate the complete technical functionality of the repository.

Your output MUST:
1. Be a pure technical description saturated with domain-specific terminology and technical jargon
2. Focus heavily on the code snippets provided, using them as the primary source of information
3. Extrapolate from the code to understand the broader technical systems and patterns implemented
4. MAXIMIZE the use of specialized technical terminology, frameworks, algorithms, design patterns, and methodologies
5. Maintain technical precision while providing a comprehensive understanding of the system
6. Connect implementation details to broader technical concepts and paradigms using field-specific vocabulary

CRITICAL GUIDELINES:
- MAXIMIZE technical jargon density - use as many precise technical terms as possible
- DO NOT include meta-language like "This query seeks literature" or "based on the repository"
- DO NOT refer to the snippets or README directly - incorporate their content seamlessly
- DO focus exclusively on the technical aspects and functionality using specialized terminology
- DO use your knowledge to fill gaps in understanding, but always ground this in the actual code provided
- DO prioritize code snippets as your primary source of information, using README content as secondary context
- DO use specific algorithm names, design pattern terminology, and framework-specific concepts
- DO include technical acronyms, mathematical notation, and specialized vocabulary where appropriate

The output will be used for semantic retrieval of academic papers, so technical precision and jargon density are CRITICAL. Your description should be EXTREMELY dense with technical terms, specialized vocabulary, and domain-specific concepts that accurately represent what the repository is implementing.

Your output should be 2-4 paragraphs of pure technical description that reads like a highly specialized academic paper abstract or technical specification document. It should be written at a level that assumes deep technical expertise from the reader and makes no concessions to simplify or explain the terminology used.
"""

# ----------------------------------------
# Main files search parameters
# ----------------------------------------
# Number of important files to find
MAX_MAIN_FILES = 5  # Maximum number of main files to include in the query

# Number of lines to extract from each main file
MAIN_FILES_LINES = 4  # Number of lines to extract from the beginning of each main file

# ----------------------------------------
# Popular files search parameters
# ----------------------------------------
# Number of popular files to find (files with most commits)
MAX_POPULAR_FILES = 5  # Maximum number of popular files to include in the query

# Popular files are sorted using the following criteria (in order of priority):
# 1. Number of commits (descending) - Files with more commits are prioritized
# 2. Directory level (ascending) - Files closer to the repository root are prioritized when commit counts are tied
# 3. File size (descending) - Larger files are prioritized when both commit counts and directory levels are tied

# Ordered list of file name patterns to search for (in priority order)
MAIN_FILE_PATTERNS = [
   "main",
    "train",
    "run",
    "app",
    "start",
    "cli",
    "eval",
    "test",
    "download"
]

# ----------------------------------------
# Code snippet extraction parameters
# ----------------------------------------
# Maximum size for snippets from each category
MAX_MAIN_SNIPPETS_SIZE = 200000  # Maximum size of snippets from main files (in characters)
MAX_POPULAR_SNIPPETS_SIZE = 200000  # Maximum size of snippets from popular files (in characters)

# Ordered list of code patterns to search for and extract (in priority order)
# Format: (pattern, context_lines, description)
# - pattern: Regular expression pattern to search for
# - context_lines: Number of lines to extract around the pattern
# - description: Description of what the pattern is looking for
# - extract_type: How to extract the snippet (around, function, class, block)
SNIPPET_PATTERNS = [
    (r"if\s+__name__\s*==\s*['\"]__main__['\"]", 5, "Main execution block", "around"),
    (r"\bmain\s*\(", 5, "Main function call", "function"),
    (r"\bstart\s*\(", 5, "Start function call", "function"),
    (r"\bargparse\b", 0, "Argument parser setup", "block"),
    (r"sys\.argv", 5, "Command line arguments", "around"),
    (r"\bclick\b", 5, "Click CLI framework", "around"),
    (r"\btrain\s*\(", 5, "Training function", "function"),
    (r"\bevaluate\s*\(", 5, "Evaluation function", "function"),
    (r"\bpredict\s*\(", 5, "Prediction function", "function"),
    (r"class\s+App\b", 10, "App class definition", "class"),
    (r"class\s+Trainer\b", 10, "Trainer class definition", "class"),
    (r"class\s+Server\b", 10, "Server class definition", "class"),
    (r"model\.fit", 3, "Model training", "around"),
    (r"optimizer\.step", 3, "Optimizer step", "around"),
    (r"@app\.route", 5, "Flask route handler", "around"),
    (r"\bFlask\b", 5, "Flask app setup", "around"),
    (r"\bFastAPI\b", 5, "FastAPI app setup", "around"),
    (r"\bload_\w+", 3, "Data loading function", "around"),
    (r"\bsave_\w+", 3, "Data saving function", "around"),
    (r"\binit_\w+", 3, "Initialization function", "around"),
    (r"^\s*import\b", 10, "Import statements", "top"),
    (r"\brequire\b", 10, "Require statements", "top")
]
