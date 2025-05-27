"""
KGB-based answer generation module.

This module selects top chunks from KGB LLM-processed chunks and generates
answers using a reasoning model that follows the knowledge graph path.
"""

import os
import json
import logging
import sqlite3
import requests
import re
import unicodedata
from typing import List, Dict, Any, Tuple
from pathlib import Path
from openai import OpenAI

from .answer_generation_kgb_parameters import (
    INPUT_DB_PATH,
    RESULTS_DIR,
    TOP_N_GLOBAL_CHUNKS,
    TOP_M_PER_QUERY_CHUNKS,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
    ANSWER_JSON_PATH,
    ANSWER_TXT_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_text_formatting(text: str) -> str:
    """
    Clean up text formatting to ensure plain text output without Unicode escape sequences.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text with proper formatting
    """
    if not text:
        return ""
        
    # First, try to decode any escaped Unicode sequences
    try:
        text = text.encode('utf-8').decode('unicode_escape')
    except Exception:
        # If decoding fails, continue with the original text
        pass
    
    # Normalize Unicode characters to their closest ASCII representation
    text = unicodedata.normalize('NFKD', text)
    # Convert to ASCII, ignoring non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Comprehensive dictionary of Unicode replacements
    replacements = {
        # Spaces and formatting
        '\u2009': ' ',  # Thin space
        '\u200a': ' ',  # Hair space
        '\u200b': '',   # Zero width space
        '\u200c': '',   # Zero width non-joiner
        '\u200d': '',   # Zero width joiner
        '\u2028': '\n', # Line separator
        '\u2029': '\n', # Paragraph separator
        '\u202f': ' ',  # Narrow no-break space
        '\u205f': ' ',  # Medium mathematical space
        '\u3000': ' ',  # Ideographic space
        
        # Dashes and hyphens
        '\u2010': '-',  # Hyphen
        '\u2011': '-',  # Non-breaking hyphen
        '\u2012': '-',  # Figure dash
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2015': '--', # Horizontal bar
        '\u2212': '-',  # Minus sign
        
        # Quotes
        '\u2018': "'", # Left single quote
        '\u2019': "'", # Right single quote
        '\u201a': "'", # Single low-9 quote
        '\u201b': "'", # Single high-reversed-9 quote
        '\u201c': '"', # Left double quote
        '\u201d': '"', # Right double quote
        '\u201e': '"', # Double low-9 quote
        '\u201f': '"', # Double high-reversed-9 quote
        '\u2039': '<', # Single left-pointing angle quote
        '\u203a': '>', # Single right-pointing angle quote
        '\u00ab': '"', # Left-pointing double angle quote
        '\u00bb': '"', # Right-pointing double angle quote
        
        # Math symbols
        '\u00b1': '+/-',     # Plus-minus sign
        '\u00d7': 'x',       # Multiplication sign
        '\u00f7': '/',       # Division sign
        '\u221a': 'sqrt',    # Square root
        '\u221b': 'cbrt',    # Cube root
        '\u221c': '4thrt',   # Fourth root
        '\u221d': 'prop to', # Proportional to
        '\u221e': 'infinity',# Infinity
        '\u221f': 'right angle', # Right angle
        '\u2220': 'angle',   # Angle
        '\u2221': 'measured angle', # Measured angle
        '\u2222': 'spherical angle', # Spherical angle
        '\u2223': '|',       # Divides
        '\u2224': '|/',      # Does not divide
        '\u2225': '||',      # Parallel to
        '\u2226': '||/',     # Not parallel to
        '\u2227': 'and',     # Logical and
        '\u2228': 'or',      # Logical or
        '\u2229': 'intersection', # Intersection
        '\u222a': 'union',   # Union
        '\u222b': 'integral', # Integral
        '\u222c': 'double integral', # Double integral
        '\u222d': 'triple integral', # Triple integral
        '\u222e': 'contour integral', # Contour integral
        '\u2234': 'therefore', # Therefore
        '\u2235': 'because',  # Because
        '\u2237': '::',       # Proportion
        '\u2243': '~=',       # Asymptotically equal to
        '\u2245': '~=',       # Approximately equal to
        '\u2248': '~=',       # Almost equal to
        '\u224d': '~~',       # Equivalent to
        '\u2260': '!=',       # Not equal to
        '\u2261': '==',       # Identical to
        '\u2264': '<=',       # Less than or equal to
        '\u2265': '>=',       # Greater than or equal to
        '\u2282': 'subset of', # Subset of
        '\u2283': 'superset of', # Superset of
        '\u2284': 'not subset of', # Not a subset of
        '\u2286': 'subset or equal', # Subset of or equal to
        '\u2287': 'superset or equal', # Superset of or equal to
        '\u2295': 'circled plus', # Circled plus
        '\u2297': 'circled times', # Circled times
        '\u22c5': '·',        # Dot operator
        
        # Greek letters
        '\u0391': 'Alpha',   # Greek capital Alpha
        '\u0392': 'Beta',    # Greek capital Beta
        '\u0393': 'Gamma',   # Greek capital Gamma
        '\u0394': 'Delta',   # Greek capital Delta
        '\u0395': 'Epsilon', # Greek capital Epsilon
        '\u0396': 'Zeta',    # Greek capital Zeta
        '\u0397': 'Eta',     # Greek capital Eta
        '\u0398': 'Theta',   # Greek capital Theta
        '\u0399': 'Iota',    # Greek capital Iota
        '\u039a': 'Kappa',   # Greek capital Kappa
        '\u039b': 'Lambda',  # Greek capital Lambda
        '\u039c': 'Mu',      # Greek capital Mu
        '\u039d': 'Nu',      # Greek capital Nu
        '\u039e': 'Xi',      # Greek capital Xi
        '\u039f': 'Omicron', # Greek capital Omicron
        '\u03a0': 'Pi',      # Greek capital Pi
        '\u03a1': 'Rho',     # Greek capital Rho
        '\u03a3': 'Sigma',   # Greek capital Sigma
        '\u03a4': 'Tau',     # Greek capital Tau
        '\u03a5': 'Upsilon', # Greek capital Upsilon
        '\u03a6': 'Phi',     # Greek capital Phi
        '\u03a7': 'Chi',     # Greek capital Chi
        '\u03a8': 'Psi',     # Greek capital Psi
        '\u03a9': 'Omega',   # Greek capital Omega
        '\u03b1': 'alpha',   # Greek small alpha
        '\u03b2': 'beta',    # Greek small beta
        '\u03b3': 'gamma',   # Greek small gamma
        '\u03b4': 'delta',   # Greek small delta
        '\u03b5': 'epsilon', # Greek small epsilon
        '\u03b6': 'zeta',    # Greek small zeta
        '\u03b7': 'eta',     # Greek small eta
        '\u03b8': 'theta',   # Greek small theta
        '\u03b9': 'iota',    # Greek small iota
        '\u03ba': 'kappa',   # Greek small kappa
        '\u03bb': 'lambda',  # Greek small lambda
        '\u03bc': 'mu',      # Greek small mu
        '\u03bd': 'nu',      # Greek small nu
        '\u03be': 'xi',      # Greek small xi
        '\u03bf': 'omicron', # Greek small omicron
        '\u03c0': 'pi',      # Greek small pi
        '\u03c1': 'rho',     # Greek small rho
        '\u03c2': 'final sigma', # Greek small final sigma
        '\u03c3': 'sigma',   # Greek small sigma
        '\u03c4': 'tau',     # Greek small tau
        '\u03c5': 'upsilon', # Greek small upsilon
        '\u03c6': 'phi',     # Greek small phi
        '\u03c7': 'chi',     # Greek small chi
        '\u03c8': 'psi',     # Greek small psi
        '\u03c9': 'omega',   # Greek small omega
        
        # Superscripts and subscripts
        '\u00b2': '^2',      # Superscript two
        '\u00b3': '^3',      # Superscript three
        '\u00b9': '^1',      # Superscript one
        '\u2070': '^0',      # Superscript zero
        '\u2071': '^i',      # Superscript latin small letter i
        '\u2074': '^4',      # Superscript four
        '\u2075': '^5',      # Superscript five
        '\u2076': '^6',      # Superscript six
        '\u2077': '^7',      # Superscript seven
        '\u2078': '^8',      # Superscript eight
        '\u2079': '^9',      # Superscript nine
        '\u207a': '^+',      # Superscript plus sign
        '\u207b': '^-',      # Superscript minus
        '\u207c': '^=',      # Superscript equals sign
        '\u207d': '^(',      # Superscript left parenthesis
        '\u207e': '^)',      # Superscript right parenthesis
        '\u207f': '^n',      # Superscript latin small letter n
        '\u2080': '_0',      # Subscript zero
        '\u2081': '_1',      # Subscript one
        '\u2082': '_2',      # Subscript two
        '\u2083': '_3',      # Subscript three
        '\u2084': '_4',      # Subscript four
        '\u2085': '_5',      # Subscript five
        '\u2086': '_6',      # Subscript six
        '\u2087': '_7',      # Subscript seven
        '\u2088': '_8',      # Subscript eight
        '\u2089': '_9',      # Subscript nine
        '\u208a': '_+',      # Subscript plus sign
        '\u208b': '_-',      # Subscript minus
        '\u208c': '_=',      # Subscript equals sign
        '\u208d': '_(',      # Subscript left parenthesis
        '\u208e': '_)',      # Subscript right parenthesis
        
        # Other symbols
        '\u00a9': '(c)',     # Copyright sign
        '\u00ae': '(r)',     # Registered sign
        '\u2026': '...',     # Ellipsis
        '\u2122': '(tm)',    # Trade mark sign
        '\u2190': '<-',      # Leftwards arrow
        '\u2191': '^',       # Upwards arrow
        '\u2192': '->',      # Rightwards arrow
        '\u2193': 'v',       # Downwards arrow
        '\u2194': '<->',     # Left right arrow
        '\u21d2': '=>',      # Rightwards double arrow
        '\u21d4': '<=>',     # Left right double arrow
        '\u2200': 'for all', # For all
        '\u2203': 'exists',  # There exists
        '\u2205': 'empty set', # Empty set
        '\u2208': 'in',      # Element of
        '\u2209': 'not in',  # Not an element of
        '\u2211': 'sum',     # N-ary summation
        '\u2217': '*',       # Asterisk operator
        '\u2219': '·',       # Bullet operator
        '\u221a': 'sqrt',    # Square root
        '\u221e': 'infinity', # Infinity
        '\u222b': 'integral', # Integral
        '\u2248': '~=',      # Almost equal to
        '\u2260': '!=',      # Not equal to
        '\u2264': '<=',      # Less-than or equal to
        '\u2265': '>=',      # Greater-than or equal to
        '\u2282': 'subset of', # Subset of
        '\u2283': 'superset of', # Superset of
        '\u2284': 'not subset of', # Not a subset of
        '\u2286': 'subset or equal', # Subset of or equal to
        '\u2287': 'superset or equal', # Superset of or equal to
        '\u2295': 'circled plus', # Circled plus
        '\u2297': 'circled times', # Circled times
        '\u22c5': '·',       # Dot operator
        '\u2713': 'check mark', # Check mark
        '\u2717': 'ballot x', # Ballot x
        
        # Special characters
        '\u00a0': ' ',       # Non-breaking space
        '\u00ad': '-',       # Soft hyphen
        '\u00b7': '·',       # Middle dot
        '\u00ce': 'I',       # Latin capital letter I with circumflex
        '\u00cf': 'I',       # Latin capital letter I with diaeresis
        '\u00e2': 'a',       # Latin small letter a with circumflex
        '\u00e3': 'a',       # Latin small letter a with tilde
        '\u00c3': 'A',       # Latin capital letter A with tilde
        '\u00e9': 'e',       # Latin small letter e with acute
        '\u00e8': 'e',       # Latin small letter e with grave
        '\u00ea': 'e',       # Latin small letter e with circumflex
        '\u00eb': 'e',       # Latin small letter e with diaeresis
        '\u00f3': 'o',       # Latin small letter o with acute
        '\u00f2': 'o',       # Latin small letter o with grave
        '\u00f4': 'o',       # Latin small letter o with circumflex
        '\u00f6': 'o',       # Latin small letter o with diaeresis
        '\u00fa': 'u',       # Latin small letter u with acute
        '\u00fb': 'u',       # Latin small letter u with circumflex
        '\u00fc': 'u',       # Latin small letter u with diaeresis
        '\u00df': 'ss',      # Latin small letter sharp s
        '\u00c7': 'C',       # Latin capital letter C with cedilla
        '\u00e7': 'c',       # Latin small letter c with cedilla
        '\u00d1': 'N',       # Latin capital letter N with tilde
        '\u00f1': 'n',       # Latin small letter n with tilde
        '\u00c6': 'AE',      # Latin capital letter AE
        '\u00e6': 'ae',      # Latin small letter ae
        '\u0152': 'OE',      # Latin capital ligature OE
        '\u0153': 'oe',      # Latin small ligature oe
        '\u00a1': '!',       # Inverted exclamation mark
        '\u00bf': '?',       # Inverted question mark
    }
    
    # Apply all replacements
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    
    # Replace any remaining Unicode escape sequences
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    
    # Replace any Unicode character references like \u00ce\u00bb with their ASCII equivalents
    text = re.sub(r'\\u[0-9a-fA-F]{4}\\u[0-9a-fA-F]{4}', '', text)
    
    # Replace common LaTeX-style formatting
    latex_replacements = {
        '\\alpha': 'alpha',
        '\\beta': 'beta',
        '\\gamma': 'gamma',
        '\\delta': 'delta',
        '\\epsilon': 'epsilon',
        '\\zeta': 'zeta',
        '\\eta': 'eta',
        '\\theta': 'theta',
        '\\iota': 'iota',
        '\\kappa': 'kappa',
        '\\lambda': 'lambda',
        '\\mu': 'mu',
        '\\nu': 'nu',
        '\\xi': 'xi',
        '\\pi': 'pi',
        '\\rho': 'rho',
        '\\sigma': 'sigma',
        '\\tau': 'tau',
        '\\upsilon': 'upsilon',
        '\\phi': 'phi',
        '\\chi': 'chi',
        '\\psi': 'psi',
        '\\omega': 'omega',
        '\\Gamma': 'Gamma',
        '\\Delta': 'Delta',
        '\\Theta': 'Theta',
        '\\Lambda': 'Lambda',
        '\\Xi': 'Xi',
        '\\Pi': 'Pi',
        '\\Sigma': 'Sigma',
        '\\Phi': 'Phi',
        '\\Psi': 'Psi',
        '\\Omega': 'Omega',
        '\\sqrt': 'sqrt',
        '\\int': 'integral',
        '\\sum': 'sum',
        '\\infty': 'infinity',
        '\\approx': '~=',
        '\\neq': '!=',
        '\\leq': '<=',
        '\\geq': '>=',
        '\\times': 'x',
        '\\div': '/',
        '\\pm': '+/-',
        '\\cdot': '·',
        '\\ldots': '...',
        '\\rightarrow': '->',
        '\\leftarrow': '<-',
        '\\Rightarrow': '=>',
        '\\Leftarrow': '<=',
        '\\leftrightarrow': '<->',
        '\\Leftrightarrow': '<=>',
        '\\forall': 'for all',
        '\\exists': 'exists',
        '\\emptyset': 'empty set',
        '\\in': 'in',
        '\\notin': 'not in',
        '\\subset': 'subset of',
        '\\supset': 'superset of',
        '\\subseteq': 'subset or equal',
        '\\supseteq': 'superset or equal',
        '\\cup': 'union',
        '\\cap': 'intersection',
        '\\oplus': 'circled plus',
        '\\otimes': 'circled times',
    }
    
    for latex_cmd, replacement in latex_replacements.items():
        text = text.replace(latex_cmd, replacement)
    
    # Clean up any remaining escape sequences
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    return text

def get_reweighted_chunks_from_db(db_path: str = INPUT_DB_PATH) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """
    Get reweighted chunks from the KGB LLM database.
    
    Args:
        db_path: Path to the database
    
    Returns:
        Tuple of (reweighted_chunks_by_query, queries_list)
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get queries
        cursor.execute("""
            SELECT id, query_text, query_type, query_index
            FROM queries
            ORDER BY query_index
        """)
        queries = cursor.fetchall()
        
        if not queries:
            logger.error("No queries found in the database")
            raise ValueError("No queries found in the database")
        
        queries_list = [query['query_text'] for query in queries]
        query_ids = [query['id'] for query in queries]
        
        # Get reweighted chunks for each query
        reweighted_chunks_by_query = []
        
        for query_id in query_ids:
            cursor.execute("""
                SELECT rc.*, q.query_text, q.query_type, q.query_index
                FROM reweighted_chunks rc
                JOIN queries q ON rc.query_id = q.id
                WHERE rc.query_id = ?
                ORDER BY rc.query_rank
            """, (query_id,))
            
            query_chunks = []
            for row in cursor.fetchall():
                chunk = dict(row)
                query_chunks.append(chunk)
            
            reweighted_chunks_by_query.append(query_chunks)
        
        conn.close()
        
        logger.info(f"Retrieved {sum(len(chunks) for chunks in reweighted_chunks_by_query)} reweighted chunks from KGB LLM database")
        return reweighted_chunks_by_query, queries_list
        
    except sqlite3.Error as e:
        logger.error(f"Database error when retrieving reweighted chunks: {e}")
        raise ValueError(f"Database error: {e}")

def select_top_chunks(
    reweighted_chunks_by_query: List[List[Dict[str, Any]]],
    top_n_global: int = TOP_N_GLOBAL_CHUNKS,
    top_m_per_query: int = TOP_M_PER_QUERY_CHUNKS
) -> List[Dict[str, Any]]:
    """
    Select top chunks for answer generation, ensuring representation from each query.
    
    Args:
        reweighted_chunks_by_query: List of reweighted chunks for each query
        top_n_global: Number of top chunks to select from global ranking
        top_m_per_query: Number of top chunks to select from each query
    
    Returns:
        List of selected chunks ordered by query index and then by rank
    """
    # First, ensure representation from each query
    selected_chunks = []
    selected_chunk_ids = set()
    
    for i, query_chunks in enumerate(reweighted_chunks_by_query):
        # Sort by final weight (descending)
        sorted_chunks = sorted(query_chunks, key=lambda x: x.get('final_weight', 0), reverse=True)
        
        # Select top M chunks from this query
        count = 0
        for chunk in sorted_chunks:
            chunk_id = chunk['chunk_id']
            if count < top_m_per_query and chunk_id not in selected_chunk_ids:
                selected_chunks.append(chunk)
                selected_chunk_ids.add(chunk_id)
                count += 1
            
            if count >= top_m_per_query:
                break
        
        logger.info(f"Selected {count} top chunks from query {i}")
    
    # Then, get top N chunks globally that weren't already selected
    all_chunks = []
    for query_chunks in reweighted_chunks_by_query:
        all_chunks.extend(query_chunks)
    
    # Sort by global rank
    all_chunks.sort(key=lambda x: x.get('global_rank', float('inf')))
    
    # Add top global chunks until we reach exactly top_n_global total chunks
    # If we already have more than top_n_global from the queries, don't add more
    remaining_slots = max(0, top_n_global - len(selected_chunks))
    global_count = 0
    for chunk in all_chunks:
        chunk_id = chunk['chunk_id']
        if chunk_id not in selected_chunk_ids and global_count < remaining_slots:
            selected_chunks.append(chunk)
            selected_chunk_ids.add(chunk_id)
            global_count += 1
    
    logger.info(f"Added {global_count} additional top chunks from global ranking")
    logger.info(f"Selected {len(selected_chunks)} chunks in total for answer generation")
    
    # Sort selected chunks by query index and then by rank within query
    selected_chunks.sort(key=lambda x: (x.get('query_index', 0), x.get('query_rank', 0)))
    
    return selected_chunks

def prepare_llm_prompt(
    selected_chunks: List[Dict[str, Any]],
    queries_list: List[str],
    enhanced_query: str
) -> str:
    """
    Prepare the prompt for the LLM.
    
    Args:
        selected_chunks: List of selected chunks
        queries_list: List of queries
        enhanced_query: Enhanced query
    
    Returns:
        Prompt for the LLM
    """
    # Group chunks by query
    chunks_by_query = {}
    for chunk in selected_chunks:
        query_idx = chunk.get('query_index', 0)
        if query_idx not in chunks_by_query:
            chunks_by_query[query_idx] = []
        chunks_by_query[query_idx].append(chunk)
    
    # Build the prompt
    prompt = f"Research Question: {enhanced_query}\n\n"
    prompt += "Knowledge Graph Path:\n"
    
    for i, query in enumerate(queries_list):
        prompt += f"Node {i+1}: {query}\n"
    
    prompt += "\nInformation from the Knowledge Graph Path (ordered by the path):\n\n"
    
    # Add chunks following the path order
    for i in range(len(queries_list)):
        if i in chunks_by_query:
            prompt += f"--- Node {i+1}: {queries_list[i]} ---\n\n"
            
            for j, chunk in enumerate(chunks_by_query[i]):
                arxiv_id = chunk.get('arxiv_id', 'unknown')
                chunk_idx = chunk.get('chunk_idx', 'unknown')
                llm_summary = chunk.get('llm_summary', 'No summary available')
                
                prompt += f"Chunk {j+1} [Source: arXiv:{arxiv_id}, Chunk:{chunk_idx}]:\n"
                prompt += f"Summary: {llm_summary}\n"
                prompt += f"Original Text: {chunk.get('chunk_text', 'No text available')}\n\n"
    
    prompt += "Please generate a comprehensive answer to the research question by following the knowledge graph path. IMPORTANT: For citations, use ONLY the basic arXiv ID in square brackets (e.g., [2306.10057]). DO NOT include any suffixes, chunk numbers, or underscores after the arXiv ID. DO NOT include 'arXiv:' or 'ID:' inside the brackets."
    
    return prompt

def get_openai_api_key() -> str:
    """
    Get the OpenAI API key from environment variables.
    
    Returns:
        str: The OpenAI API key
    """
    # Get from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    
    return api_key

def call_llm_api(
    prompt: str,
    model: str = LLM_MODEL,
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMPERATURE
) -> str:
    """
    Call the LLM API to generate an answer using the OpenAI client.
    
    Args:
        prompt: Prompt for the LLM
        model: LLM model to use
        max_tokens: Maximum tokens for the LLM response
        temperature: Temperature for the LLM response
    
    Returns:
        Generated answer
    """
    try:
        # Get the OpenAI API key
        api_key = get_openai_api_key()
        
        # Initialize the OpenAI client with the API key
        client = OpenAI(api_key=api_key)
        
        # Call the OpenAI API with appropriate parameters
        logger.info(f"Calling LLM API with model {model}")
        
        # Create the API request
        # Prepare kwargs based on model type
        kwargs = {}
        if model.lower() != "o3":
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
            
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )
        
        # Extract the answer from the response
        if response.choices and len(response.choices) > 0:
            answer = response.choices[0].message.content
        else:
            logger.warning("No choices returned from LLM API.")
            answer = ""
        
        # Clean up the text to ensure proper formatting
        answer = clean_text_formatting(answer)
        
        logger.info(f"Generated answer with {len(answer)} characters")
        return answer
        
    except Exception as e:
        logger.error(f"API request failed: {e}")
        raise ValueError(f"API request failed: {e}")

def save_answer(answer: str, enhanced_query: str, selected_chunks: List[Dict[str, Any]]) -> None:
    """
    Save the generated answer to JSON and TXT files.
    
    Args:
        answer: Generated answer
        enhanced_query: Enhanced query
        selected_chunks: List of selected chunks
    """
    # Prepare data for JSON
    answer_data = {
        "query": enhanced_query,
        "answer": answer,
        "chunks": [
            {
                "chunk_id": chunk.get('chunk_id', ''),
                "arxiv_id": chunk.get('arxiv_id', ''),
                "chunk_idx": chunk.get('chunk_idx', ''),
                "summary": chunk.get('llm_summary', ''),
                "query_index": chunk.get('query_index', 0),
                "query_rank": chunk.get('query_rank', 0),
                "global_rank": chunk.get('global_rank', 0),
                "final_weight": chunk.get('final_weight', 0.0)
            }
            for chunk in selected_chunks
        ]
    }
    
    # Save to JSON
    try:
        with open(ANSWER_JSON_PATH, 'w') as f:
            json.dump(answer_data, f, indent=2)
        logger.info(f"Saved answer to JSON: {ANSWER_JSON_PATH}")
    except Exception as e:
        logger.error(f"Error saving answer to JSON: {e}")
    
    # Save to TXT
    try:
        with open(ANSWER_TXT_PATH, 'w') as f:
            f.write(answer)
        logger.info(f"Saved answer to TXT: {ANSWER_TXT_PATH}")
    except Exception as e:
        logger.error(f"Error saving answer to TXT: {e}")

def run_answer_generation_kgb(enhanced_query: str = None) -> str:
    """
    Run the KGB-based answer generation process.
    
    Args:
        enhanced_query: Enhanced query to use for answer generation
        
    Returns:
        Generated answer
    """
    logger.info("Starting KGB-based answer generation...")
    
    # Get reweighted chunks from database
    reweighted_chunks_by_query, queries_list = get_reweighted_chunks_from_db()
    
    # If enhanced_query is not provided, use the first query
    if enhanced_query is None:
        enhanced_query = queries_list[1] if len(queries_list) > 1 else queries_list[0]
        logger.info(f"Using query as enhanced query: {enhanced_query}")
    
    # Select top chunks
    selected_chunks = select_top_chunks(reweighted_chunks_by_query)
    
    # Prepare prompt
    prompt = prepare_llm_prompt(selected_chunks, queries_list, enhanced_query)
    
    # Generate answer
    answer = call_llm_api(prompt)
    
    # Save answer
    save_answer(answer, enhanced_query, selected_chunks)
    
    logger.info("KGB-based answer generation completed")
    return answer
