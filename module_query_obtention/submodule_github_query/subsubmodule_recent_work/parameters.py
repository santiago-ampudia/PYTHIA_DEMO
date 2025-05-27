"""
module_query_obtention/submodule_github_query/subsubmodule_recent_work/parameters.py

This file contains parameters and configuration values for the GitHub repository
recent work query generation.
"""

# ----------------------------------------
# Repository analysis parameters
# ----------------------------------------
MAX_README_SIZE = 10000  # Maximum size of README to analyze (in characters, ~20 pages)
MAX_FILES_TO_ANALYZE = 10  # Maximum number of files to analyze for content
MAX_FILE_SIZE = 2000000  # Maximum size of a file to analyze (in bytes)

# Git clone parameters
CLONE_DEPTH = 1000  # Depth of git history to clone (number of commits)

# ----------------------------------------
# Query generation parameters
# ----------------------------------------
MAX_QUERY_LENGTH = 10000000  # Maximum length of the generated query (characters)
MIN_QUERY_LENGTH = 20  # Minimum length of the generated query (characters)

# ----------------------------------------
# Recent commits parameters
# ----------------------------------------
MAX_COMMITS = 10  # Maximum number of recent commits to fetch
MAX_COMMITS_QUERY_LENGTH = 50000  # Maximum length of the commits query string
CONTEXT_LINES = 3  # Number of context lines to include before and after each change
MAX_FILES_PER_COMMIT = 5  # Maximum number of files to include per commit
MAX_DIFF_SIZE_PER_FILE = 2000  # Maximum size of diff to include per file (in characters)

# List of commit information to include in the query
# Possible values: 'message', 'files', 'diff', 'context'
COMMIT_INFO_TO_INCLUDE = [
    'message',  # Commit message
    'files',    # Names of files that changed
    'diff',     # Patch/diff of the changes
    'context'   # Code context around changes
]

# ----------------------------------------
# Hot files parameters
# ----------------------------------------
MAX_HOT_FILES = 5  # Maximum number of hot files to identify
HOT_FILES_QUERY_LENGTH = 30000  # Maximum length of the hot files query string
HOT_FILES_CONTEXT_LINES = 10  # Number of context lines to include for hot files
MAX_HOT_FILE_SIZE = 5000  # Maximum size of content to include per hot file (in characters)

# ----------------------------------------
# GPT-4 query generation parameters for recent work
# ----------------------------------------
GPT_MODEL = "gpt-4-turbo"  # Model to use for generating the final query
MAX_FINAL_QUERY_LENGTH = 1000  # Maximum length of the final query (tokens)
MIN_FINAL_QUERY_LENGTH = 200  # Minimum length of the final query (tokens)

# Instructions for GPT-4 to generate the final query based on recent work
GPT_RECENT_WORK_INSTRUCTIONS = """
You are an expert in software engineering, computer science, and technical analysis. Your task is to analyze the provided recent commit information and hot file code snippets to create a precise technical description of the CURRENT DEVELOPMENT FOCUS and RECENT CHANGES in the repository.

Your output MUST:
1. Be a pure technical description focused specifically on the RECENT CHANGES and ACTIVE DEVELOPMENT areas
2. Concentrate on the actual code changes in the commits and the key functionality in the hot files
3. Identify specific technical problems being solved and implementation details being refined
4. Use precise technical terminology related to the frameworks, algorithms, and design patterns evident in the code
5. Highlight the technical evolution occurring in the codebase based on the commit history
6. Connect the recent changes to their technical implications and purpose

CRITICAL GUIDELINES:
- FOCUS ON RECENCY - emphasize what is happening NOW in the codebase, not general functionality
- BE SPECIFIC - reference the actual technical changes shown in the commits and hot files
- USE TECHNICAL PRECISION - employ domain-specific terminology and technical jargon
- DO NOT include meta-language like "This query seeks literature" or "based on the repository"
- DO NOT refer to the commits or hot files directly - incorporate their content seamlessly
- DO focus on the technical significance of the recent changes and active development areas
- DO identify technical patterns and architectural decisions evident in the recent work
- DO highlight specific algorithms, data structures, or techniques being implemented or modified

The output will be used for semantic retrieval of academic papers related to the CURRENT technical challenges and focus areas of the project. Your description should be dense with technical terms and specialized vocabulary that accurately represent what the developers are CURRENTLY working on.

Your output should be 2-4 paragraphs of precise technical description that reads like a highly specialized academic paper abstract focused on cutting-edge developments in this technical domain. It should clearly convey the specific technical problems being addressed and solutions being implemented in the recent development work.
"""

