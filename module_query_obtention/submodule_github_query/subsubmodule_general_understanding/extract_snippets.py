"""
module_query_obtention/submodule_github_query/subsubmodule_general_understanding/extract_snippets.py

This module is responsible for extracting code snippets from files based on predefined patterns.
It processes files from main_files and popular_files lists and extracts relevant code sections.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import ast

from . import parameters
from .fetch_main_files import is_binary_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Snippet:
    """Class to represent a code snippet extracted from a file."""
    
    def __init__(self, file_path: Path, pattern: str, pattern_description: str, 
                 content: str, line_start: int, line_end: int):
        """
        Initialize a Snippet object.
        
        Args:
            file_path: Path to the file the snippet was extracted from
            pattern: The pattern that matched this snippet
            pattern_description: Description of the pattern
            content: The extracted code content
            line_start: Starting line number in the original file
            line_end: Ending line number in the original file
        """
        self.file_path = file_path
        self.pattern = pattern
        self.pattern_description = pattern_description
        self.content = content
        self.line_start = line_start
        self.line_end = line_end
        self.relative_path = None  # Will be set later
    
    def __str__(self) -> str:
        """String representation of the snippet."""
        return f"Snippet from {self.file_path.name} (lines {self.line_start}-{self.line_end}): {self.pattern_description}"
    
    def get_formatted_content(self) -> str:
        """Get the snippet content with a header showing its source."""
        if self.relative_path is None:
            file_info = self.file_path.name
        else:
            file_info = str(self.relative_path)
            
        header = f"-----------------------------------------------------------------\nCODE SNIPPET: {self.pattern_description}\nFILE: {file_info}\nLINES: {self.line_start}-{self.line_end}\n-----------------------------------------------------------------"
        return f"{header}\n{self.content}\n"


def extract_around_pattern(file_lines: List[str], match_line: int, context_lines: int) -> Tuple[int, int, str]:
    """
    Extract lines around a pattern match.
    
    Args:
        file_lines: All lines from the file
        match_line: Line number where the pattern was found (0-indexed)
        context_lines: Number of lines to extract before and after the match
        
    Returns:
        Tuple of (start_line, end_line, extracted_content)
    """
    start_line = max(0, match_line - context_lines)
    end_line = min(len(file_lines) - 1, match_line + context_lines)
    
    extracted_lines = file_lines[start_line:end_line + 1]
    return start_line, end_line, ''.join(extracted_lines)


def extract_function_def(file_lines: List[str], match_line: int, context_lines: int) -> Tuple[int, int, str]:
    """
    Extract a function definition and context lines.
    
    Args:
        file_lines: All lines from the file
        match_line: Line number where the pattern was found (0-indexed)
        context_lines: Number of lines to extract after the function definition
        
    Returns:
        Tuple of (start_line, end_line, extracted_content)
    """
    # First, find the function definition start
    start_line = match_line
    while start_line > 0 and not re.match(r'^\s*def\s+', file_lines[start_line]):
        start_line -= 1
    
    # If we couldn't find the function definition, fall back to context extraction
    if start_line == 0 and not re.match(r'^\s*def\s+', file_lines[start_line]):
        return extract_around_pattern(file_lines, match_line, context_lines)
    
    # Find the function end (looking for dedent or end of file)
    indent_level = len(re.match(r'^(\s*)', file_lines[start_line]).group(1))
    end_line = start_line
    
    while end_line < len(file_lines) - 1:
        end_line += 1
        # Skip empty lines or comments when determining indentation
        if not file_lines[end_line].strip() or file_lines[end_line].strip().startswith('#'):
            continue
            
        current_indent = len(re.match(r'^(\s*)', file_lines[end_line]).group(1))
        if current_indent <= indent_level and file_lines[end_line].strip():
            end_line -= 1
            break
    
    # Add context lines after function
    end_line = min(len(file_lines) - 1, end_line + context_lines)
    
    extracted_lines = file_lines[start_line:end_line + 1]
    return start_line, end_line, ''.join(extracted_lines)


def extract_class_def(file_lines: List[str], match_line: int, context_lines: int) -> Tuple[int, int, str]:
    """
    Extract a class definition including __init__ and context lines.
    
    Args:
        file_lines: All lines from the file
        match_line: Line number where the pattern was found (0-indexed)
        context_lines: Number of lines to extract after the class definition
        
    Returns:
        Tuple of (start_line, end_line, extracted_content)
    """
    # First, find the class definition start
    start_line = match_line
    while start_line > 0 and not re.match(r'^\s*class\s+', file_lines[start_line]):
        start_line -= 1
    
    # If we couldn't find the class definition, fall back to context extraction
    if start_line == 0 and not re.match(r'^\s*class\s+', file_lines[start_line]):
        return extract_around_pattern(file_lines, match_line, context_lines)
    
    # Find the class end (looking for dedent or end of file)
    indent_level = len(re.match(r'^(\s*)', file_lines[start_line]).group(1))
    end_line = start_line
    
    # Find the end of the class definition
    while end_line < len(file_lines) - 1:
        end_line += 1
        # Skip empty lines or comments when determining indentation
        if not file_lines[end_line].strip() or file_lines[end_line].strip().startswith('#'):
            continue
            
        current_indent = len(re.match(r'^(\s*)', file_lines[end_line]).group(1))
        if current_indent <= indent_level and file_lines[end_line].strip():
            end_line -= 1
            break
    
    # Add context lines after class
    end_line = min(len(file_lines) - 1, end_line + context_lines)
    
    extracted_lines = file_lines[start_line:end_line + 1]
    return start_line, end_line, ''.join(extracted_lines)


def extract_block(file_lines: List[str], match_line: int) -> Tuple[int, int, str]:
    """
    Extract a logical code block (e.g., argparse setup).
    
    Args:
        file_lines: All lines from the file
        match_line: Line number where the pattern was found (0-indexed)
        
    Returns:
        Tuple of (start_line, end_line, extracted_content)
    """
    # Find the start of the block (looking for a line that's not indented more than the match line)
    indent_level = len(re.match(r'^(\s*)', file_lines[match_line]).group(1))
    start_line = match_line
    
    while start_line > 0:
        prev_indent = len(re.match(r'^(\s*)', file_lines[start_line - 1]).group(1))
        if prev_indent < indent_level and file_lines[start_line - 1].strip():
            break
        start_line -= 1
    
    # Find the end of the block (looking for a line that's not indented more than the start line)
    end_line = match_line
    start_indent = len(re.match(r'^(\s*)', file_lines[start_line]).group(1))
    
    while end_line < len(file_lines) - 1:
        end_line += 1
        # Skip empty lines or comments when determining indentation
        if not file_lines[end_line].strip() or file_lines[end_line].strip().startswith('#'):
            continue
            
        current_indent = len(re.match(r'^(\s*)', file_lines[end_line]).group(1))
        if current_indent <= start_indent and file_lines[end_line].strip():
            end_line -= 1
            break
    
    extracted_lines = file_lines[start_line:end_line + 1]
    return start_line, end_line, ''.join(extracted_lines)


def extract_top_lines(file_lines: List[str], match_line: int, context_lines: int) -> Tuple[int, int, str]:
    """
    Extract the top N lines from a file, starting from the match.
    
    Args:
        file_lines: All lines from the file
        match_line: Line number where the pattern was found (0-indexed)
        context_lines: Number of lines to extract
        
    Returns:
        Tuple of (start_line, end_line, extracted_content)
    """
    end_line = min(len(file_lines) - 1, match_line + context_lines)
    extracted_lines = file_lines[match_line:end_line + 1]
    return match_line, end_line, ''.join(extracted_lines)


def extract_snippets_from_file(file_path: Path, patterns: List[Tuple[str, int, str, str]]) -> List[Snippet]:
    """
    Extract code snippets from a file based on the given patterns.
    
    Args:
        file_path: Path to the file to extract snippets from
        patterns: List of (pattern, context_lines, description, extract_type) tuples
        
    Returns:
        List of Snippet objects
    """
    # Check if file exists
    if not file_path.exists() or not file_path.is_file():
        logger.warning(f"File does not exist or is not a file: {file_path}")
        return []
        
    # Skip binary files
    if is_binary_file(file_path):
        logger.debug(f"Skipping binary file: {file_path}")
        return []
    
    logger.info(f"Attempting to extract snippets from file: {file_path}")
        
    # Read the file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            file_lines = file_content.splitlines(keepends=True)
        logger.info(f"Successfully read file {file_path} with {len(file_lines)} lines")
    except Exception as e:
        logger.warning(f"Error reading file {file_path}: {e}")
        return []
    
    snippets = []
    
    # Process each pattern
    for pattern, context_lines, description, extract_type in patterns:
        logger.info(f"Searching for pattern '{pattern}' in file {file_path.name}")
        
        # Find matches for this pattern
        matches = []
        for i, line in enumerate(file_lines):
            if re.search(pattern, line):
                logger.info(f"Found match for pattern '{pattern}' in file {file_path.name} at line {i+1}: {line.strip()}")
                matches.append(i)
        
        # Process each match
        for match_idx in matches:
            try:
                # Extract the snippet based on the extraction type
                if extract_type == "around":
                    start_line, end_line, snippet_content = extract_around_pattern(
                        file_lines, match_idx, context_lines)
                elif extract_type == "function":
                    start_line, end_line, snippet_content = extract_function_def(
                        file_lines, match_idx, context_lines)
                elif extract_type == "class":
                    start_line, end_line, snippet_content = extract_class_def(
                        file_lines, match_idx, context_lines)
                elif extract_type == "block":
                    start_line, end_line, snippet_content = extract_block(
                        file_lines, match_idx)
                elif extract_type == "top":
                    start_line, end_line, snippet_content = extract_top_lines(
                        file_lines, match_idx, context_lines)
                else:
                    # Default to around extraction
                    start_line, end_line, snippet_content = extract_around_pattern(
                        file_lines, match_idx, context_lines)
                
                # Create a snippet object
                snippet = Snippet(
                    file_path=file_path,
                    pattern=pattern,
                    pattern_description=description,
                    content=snippet_content,
                    line_start=start_line + 1,  # Convert to 1-indexed
                    line_end=end_line + 1       # Convert to 1-indexed
                )
                
                snippets.append(snippet)
                
            except Exception as e:
                logger.error(f"Error extracting snippet for pattern '{pattern}' in file {file_path}: {str(e)}")
    
    return snippets


def extract_code_snippets(repo_dir: Path, main_files: List[Path], popular_files: List[Path]) -> Tuple[List[Snippet], List[Snippet], int]:
    """
    Extract code snippets from main_files and popular_files based on predefined patterns.
    Keep snippets from main files and popular files separate.
    
    Args:
        repo_dir: Path to the repository directory
        main_files: List of paths to main files
        popular_files: List of paths to popular files
        
    Returns:
        Tuple of (main file snippets, popular file snippets, total size in characters)
    """
    # Log the input files
    logger.info(f"Extracting snippets from {len(main_files)} main files and {len(popular_files)} popular files")
    
    # Convert to sets for faster lookup and to remove duplicates within each list
    main_file_set = set(str(f) for f in main_files)
    popular_file_set = set(str(f) for f in popular_files)
    
    # Find overlapping files (files that are both main and popular)
    overlap = main_file_set.intersection(popular_file_set)
    
    # We prioritize main files over popular files for overlapping files
    # Main files are typically more representative of the core functionality
    # and entry points of the project, which is more valuable for the LLM
    if overlap:
        logger.info(f"Found {len(overlap)} files that are both main and popular files")
        logger.info(f"Prioritizing these files as main files for snippet extraction")
        
        # List the overlapping files for reference
        for file_path in overlap:
            logger.info(f"Overlapping file: {file_path}")
    
    # Ensure files are only in one list (prioritize main files if in both)
    popular_files_unique = [f for f in popular_files if str(f) not in main_file_set]
    logger.info(f"After removing duplicates, we have {len(popular_files_unique)} unique popular files")
    
    # Create separate snippet lists
    main_file_snippets = []
    popular_file_snippets = []
    main_snippets_size = 0
    popular_snippets_size = 0
    
    # Use separate size limits for each category
    max_main_size = parameters.MAX_MAIN_SNIPPETS_SIZE
    max_popular_size = parameters.MAX_POPULAR_SNIPPETS_SIZE
    
    # Helper function to process files for a specific pattern
    def process_files_for_pattern(files, snippet_list, pattern_info, size_limit, size_tracker):
        pattern, context_lines, description, extract_type = pattern_info
        
        # Get the current size of snippets in this category
        current_size = sum(len(s.get_formatted_content()) for s in snippet_list)
        
        for file_path in files:
            # Skip if we've reached the size limit for this category
            if current_size >= size_limit:
                break
                
            # Extract snippets from the file for this pattern
            file_snippets = extract_snippets_from_file(file_path, [(pattern, context_lines, description, extract_type)])
            
            # Set relative paths for the snippets
            for snippet in file_snippets:
                snippet.relative_path = file_path.relative_to(repo_dir)
                
            # Add snippets to the result (up to the maximum size)
            for snippet in file_snippets:
                snippet_size = len(snippet.get_formatted_content())
                
                if current_size + snippet_size <= size_limit:
                    snippet_list.append(snippet)
                    current_size += snippet_size
                    size_tracker['size'] += snippet_size
                    logger.info(f"Found snippet in {snippet.relative_path}: {description} (lines {snippet.line_start}-{snippet.line_end})")
                else:
                    # If this snippet would exceed the limit, stop
                    break
            
            # Stop processing files if we've reached the size limit
            if current_size >= size_limit:
                break
                
        return current_size >= size_limit
    
    # Create dictionaries to track sizes for each category
    main_size_tracker = {'size': 0}
    popular_size_tracker = {'size': 0}
    
    # Process each pattern in order
    for pattern_info in parameters.SNIPPET_PATTERNS:
        pattern, _, description, _ = pattern_info
        logger.info(f"Searching for pattern: '{pattern}' ({description})")
        
        # Process main files first
        main_limit_reached = process_files_for_pattern(
            main_files, main_file_snippets, pattern_info, max_main_size, main_size_tracker)
            
        # Then process popular files
        logger.info(f"Processing {len(popular_files_unique)} unique popular files for pattern '{pattern}'")
        if not popular_files_unique:
            logger.warning("No unique popular files to process! All popular files are also in main files.")
        
        popular_limit_reached = process_files_for_pattern(
            popular_files_unique, popular_file_snippets, pattern_info, max_popular_size, popular_size_tracker)
        
        # If both categories have reached their limits, stop processing patterns
        if main_limit_reached and popular_limit_reached:
            logger.info(f"Both main files and popular files have reached their snippet size limits")
            break
    
    # Calculate total size
    total_size = main_size_tracker['size'] + popular_size_tracker['size']
    
    logger.info(f"Extracted {len(main_file_snippets)} snippets from main files (size: {main_size_tracker['size']} characters)")
    logger.info(f"Extracted {len(popular_file_snippets)} snippets from popular files (size: {popular_size_tracker['size']} characters)")
    logger.info(f"Total size of all snippets: {total_size} characters")
    
    return main_file_snippets, popular_file_snippets, total_size


def format_snippets(snippets: List[Snippet]) -> str:
    """
    Format a list of snippets into a single string.
    
    Args:
        snippets: List of Snippet objects
        
    Returns:
        Formatted string containing all snippets
    """
    if not snippets:
        return "No relevant code snippets found in the repository."
        
    # Sort snippets by file path and line number for a consistent order
    sorted_snippets = sorted(snippets, key=lambda s: (str(s.relative_path), s.line_start))
    
    # Concatenate all snippet contents with double newlines between them
    formatted_content = "\n\n".join(snippet.get_formatted_content() for snippet in sorted_snippets)
    
    return formatted_content


def get_code_snippets(repo_dir: str or Path, main_files: List[Path], popular_files: List[Path]) -> Tuple[List[Snippet], List[Snippet], int]:
    """
    Get code snippets from main_files and popular_files, keeping them separate.
    
    Args:
        repo_dir: Path to the repository directory
        main_files: List of paths to main files
        popular_files: List of paths to popular files
        
    Returns:
        Tuple of (main file snippets, popular file snippets, total size in characters)
    """
    # Convert string path to Path object if needed
    if isinstance(repo_dir, str):
        repo_dir = Path(repo_dir)
    
    return extract_code_snippets(repo_dir, main_files, popular_files)
