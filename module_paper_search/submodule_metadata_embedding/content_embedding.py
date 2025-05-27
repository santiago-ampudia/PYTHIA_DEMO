"""
submodule_metadata_embedding/content_embedding.py

This module implements Step 1 of the paper search pipeline: embedding of full text and metadata.

Purpose: Embed full text and metadata of unprocessed arXiv papers, using hybrid chunking, 
         local storage, and append-only logic for FAISS and SQLite.

Tool: S2ORC (local JSONL), ar5iv (HTML), GROBID (TEI XML), PyMuPDF (PDF fallback), 
      BeautifulSoup (HTML parsing), HuggingFace intfloat/e5-small-v2 (CPU embedding), 
      FAISS (vector store), SQLite (chunk_index.db), progress_tracker_embedding.json
"""

import os
import re
import json
import time
import hashlib
import logging
import sqlite3
import requests
import numpy as np
import tarfile
import io
from bs4 import BeautifulSoup
from tqdm import tqdm
import fitz  # PyMuPDF
from datetime import datetime
from threading import Lock

# Set environment variable to disable tokenizers parallelism
# This prevents the warning: "huggingface/tokenizers: The current process just got forked..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModel
import torch
import faiss

from module_paper_search.submodule_metadata_embedding.content_embedding_parameters import (
    CHUNK_INDEX_DB,
    METADATA_INDEX_DB,
    FAISS_INDEX_PATH,
    METADATA_FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION as EMBEDDING_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    BATCH_SIZE,
    MAX_PAPERS_PER_BATCH,
    ARXIV_METADATA_DB as METADATA_DB,
    AR5IV_DIR,
    AR5IV_URL,
    CONTENT_SOURCES,
    PROGRESS_TRACKER_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Set device (CPU, CUDA, or MPS)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

def setup_database():
    """
    Set up the database for storing chunk data
    
    Returns:
        sqlite3.Connection: Database connection
    """
    os.makedirs(os.path.dirname(CHUNK_INDEX_DB), exist_ok=True)
    
    conn = sqlite3.connect(CHUNK_INDEX_DB)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        arxiv_id TEXT NOT NULL,
        chunk_hash TEXT NOT NULL,
        embedding_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        section_title TEXT,
        chunk_position INTEGER,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create index on arxiv_id for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_arxiv_id ON chunks (arxiv_id)')
    
    conn.commit()
    return conn

def setup_metadata_database():
    """
    Set up the database for storing metadata embeddings
    
    Returns:
        sqlite3.Connection: Database connection
    """
    os.makedirs(os.path.dirname(METADATA_INDEX_DB), exist_ok=True)
    
    conn = sqlite3.connect(METADATA_INDEX_DB)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metadata_embeddings (
        arxiv_id TEXT PRIMARY KEY,
        embedding_index INTEGER NOT NULL,
        metadata_text TEXT NOT NULL,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    return conn

def load_embedding_model():
    """
    Load the embedding model and tokenizer
    
    Returns:
        tuple: (tokenizer, model)
    """
    logger.info(f"Loading embedding model and tokenizer from {EMBEDDING_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model

def get_unprocessed_papers(force_update=False, specific_arxiv_id=None):
    """
    Get list of papers that need to be processed
    
    Args:
        force_update (bool): Whether to force update all papers
        specific_arxiv_id (str): Specific arXiv ID to process
        
    Returns:
        list: List of arXiv IDs to process
    """
    # If specific_arxiv_id is provided, only process that paper
    if specific_arxiv_id:
        return [specific_arxiv_id]
    
    # Make sure the database is set up
    setup_database()
    setup_metadata_database()
    
    # Connect to databases
    with sqlite3.connect(METADATA_DB) as conn_metadata, \
         sqlite3.connect(CHUNK_INDEX_DB) as conn_chunks, \
         sqlite3.connect(METADATA_INDEX_DB) as conn_metadata_chunks:
        
        # Get all papers from metadata database
        cursor_metadata = conn_metadata.execute("SELECT arxiv_id FROM papers")
        all_papers = [row[0] for row in cursor_metadata.fetchall()]
        logger.info(f"DEBUG: Total papers in metadata database: {len(all_papers)}")
        logger.info(f"DEBUG: Paper IDs in metadata database: {all_papers}")
        
        if force_update:
            return all_papers
        
        # Get processed papers from chunks database
        cursor_chunks = conn_chunks.execute("SELECT DISTINCT arxiv_id FROM chunks")
        processed_content_papers = {row[0] for row in cursor_chunks.fetchall()}
        logger.info(f"DEBUG: Papers with content chunks: {len(processed_content_papers)}")
        logger.info(f"DEBUG: Paper IDs with content chunks: {processed_content_papers}")
        
        # Get processed papers from metadata chunks database
        cursor_metadata_chunks = conn_metadata_chunks.execute("SELECT DISTINCT arxiv_id FROM metadata_embeddings")
        processed_metadata_papers = {row[0] for row in cursor_metadata_chunks.fetchall()}
        logger.info(f"DEBUG: Papers with metadata embeddings: {len(processed_metadata_papers)}")
        logger.info(f"DEBUG: Paper IDs with metadata embeddings: {processed_metadata_papers}")
        
        # Find papers that haven't been fully processed
        # A paper needs processing if it's missing from EITHER the content database OR the metadata database
        # This ensures both content and metadata are processed for each paper
        unprocessed_papers = []
        missing_content = []
        missing_metadata = []
        for paper in all_papers:
            missing_from_content = paper not in processed_content_papers
            missing_from_metadata = paper not in processed_metadata_papers
            if missing_from_content or missing_from_metadata:
                unprocessed_papers.append(paper)
                if missing_from_content:
                    missing_content.append(paper)
                if missing_from_metadata:
                    missing_metadata.append(paper)
        
        logger.info(f"DEBUG: Papers missing content: {len(missing_content)} - {missing_content}")
        logger.info(f"DEBUG: Papers missing metadata: {len(missing_metadata)} - {missing_metadata}")
        
        logger.info(f"Found {len(unprocessed_papers)} unprocessed papers out of {len(all_papers)} total papers")
        return unprocessed_papers

def load_progress_tracker():
    """
    Load the progress tracker from disk or create a new one
    
    Returns:
        dict: Progress tracker
    """
    if os.path.exists(PROGRESS_TRACKER_PATH):
        with open(PROGRESS_TRACKER_PATH, 'r') as f:
            return json.load(f)
    else:
        return {
            "processed_papers": [],
            "last_update": None,
            "current_batch": [],
            "total_chunks": 0,
            "total_papers": 0,
            "total_metadata_chunks": 0
        }

def save_progress_tracker(tracker):
    """
    Save the progress tracker to disk
    
    Args:
        tracker: Progress tracker dictionary
    """
    os.makedirs(os.path.dirname(PROGRESS_TRACKER_PATH), exist_ok=True)
    tracker["last_update"] = datetime.now().isoformat()
    
    with open(PROGRESS_TRACKER_PATH, 'w') as f:
        json.dump(tracker, f, indent=2)

def ensure_paper_content(arxiv_id):
    """
    Ensure that we have the content for a paper.
    This function will download the content if it doesn't exist.
    
    Args:
        arxiv_id (str): arXiv ID
        
    Returns:
        bool: True if content is available, False otherwise
    """
    # Check if we already have ar5iv content
    ar5iv_path = os.path.join(AR5IV_DIR, f"{arxiv_id}.html")
    if os.path.exists(ar5iv_path):
        logger.info(f"Using existing ar5iv content for {arxiv_id}")
        return True
    
    # Download ar5iv content
    os.makedirs(AR5IV_DIR, exist_ok=True)
    try:
        logger.info(f"Downloading ar5iv content for {arxiv_id} from {AR5IV_URL.format(arxiv_id=arxiv_id)}")
        response = requests.get(AR5IV_URL.format(arxiv_id=arxiv_id))
        if response.status_code == 200:
            with open(ar5iv_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logger.info(f"Successfully downloaded ar5iv content for {arxiv_id}")
            return True
        else:
            logger.warning(f"Failed to download ar5iv content for {arxiv_id}: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error downloading ar5iv content for {arxiv_id}: {str(e)}")
        return False

def extract_text_from_ar5iv(arxiv_id):
    """
    Extract text from ar5iv HTML content
    
    Args:
        arxiv_id (str): arXiv ID
        
    Returns:
        dict: Dictionary with sections and their text
    """
    ar5iv_path = os.path.join(AR5IV_DIR, f"{arxiv_id}.html")
    if not os.path.exists(ar5iv_path):
        logger.warning(f"ar5iv content not found for {arxiv_id}")
        return {"metadata": {"title": "", "abstract": ""}}
    
    try:
        with open(ar5iv_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract metadata
        title = ""
        title_elem = soup.find('h1', class_='ltx_title')
        if title_elem:
            title = title_elem.get_text(strip=True)
        
        abstract = ""
        abstract_elem = soup.find('div', class_='ltx_abstract')
        if abstract_elem:
            abstract = abstract_elem.get_text(strip=True)
        
        # Extract sections
        sections = {}
        
        # Add metadata section
        sections["metadata"] = {
            "title": title,
            "abstract": abstract
        }
        
        # Process main content
        main_content = soup.find('div', class_='ltx_page_main')
        if main_content:
            # Find and remove bibliography/references sections
            bibliography_sections = main_content.find_all(['div', 'section'], 
                                                      class_=['ltx_bibliography', 'ltx_references'])
            for section in bibliography_sections:
                section.decompose()  # Remove from DOM
                
            # Find all section headers
            section_headers = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            for i, header in enumerate(section_headers):
                section_title = header.get_text(strip=True)
                
                # Skip sections that are likely references
                citation_section_patterns = [
                    'references', 'bibliography', 'citations', 'works cited', 'cited literature'
                ]
                if any(pattern in section_title.lower() for pattern in citation_section_patterns):
                    logger.info(f"Skipping citation section: {section_title}")
                    continue
                
                section_content = []
                
                # Get all content until the next header
                current = header.next_sibling
                while current and (i == len(section_headers) - 1 or current != section_headers[i + 1]):
                    if current.name and current.get_text(strip=True):
                        section_content.append(current.get_text(strip=True))
                    current = current.next_sibling
                    if not current:
                        break
                
                sections[section_title] = "\n\n".join(section_content)
            
            # If no sections were found, extract all paragraphs
            if len(sections) == 1:  # Only metadata
                paragraphs = main_content.find_all('p')
                section_content = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
                sections["main_content"] = "\n\n".join(section_content)
        
        logger.info(f"Successfully extracted text from ar5iv for {arxiv_id}")
        return sections
    
    except Exception as e:
        logger.error(f"Error extracting text from ar5iv for {arxiv_id}: {str(e)}")
        return {"metadata": {"title": "", "abstract": ""}}

def is_citation_text(text):
    """
    Check if a chunk of text is primarily citations
    
    Args:
        text (str): Text to check
        
    Returns:
        bool: True if text appears to be citations, False otherwise
    """
    # Count citation patterns
    citation_patterns = [
        r'\[\d+\]',  # [1], [2], etc.
        r'\(\d{4}\)',  # (2020), (2021), etc.
        r'\b[A-Z][a-z]+ et al\.',  # Smith et al.
        r'\b[A-Z][a-z]+ and [A-Z][a-z]+',  # Smith and Jones
    ]
    
    citation_count = 0
    for pattern in citation_patterns:
        citation_count += len(re.findall(pattern, text))
    
    # Count sentences
    sentences = re.split(r'[.!?]', text)
    sentence_count = sum(1 for s in sentences if len(s.strip()) > 0)
    
    if sentence_count == 0:
        return False
    
    # If more than 50% of sentences contain citations, consider it citation text
    citation_density = citation_count / sentence_count
    return citation_density > 0.5 and citation_count > 3

def normalize_scientific_text(text):
    """
    Normalize LaTeX and Unicode characters in scientific text.
    
    This function performs the following normalizations:
    1. Standardizes mathematical symbols and Greek letters
    2. Normalizes LaTeX notation for equations and formulas
    3. Cleans up table formatting and special characters
    4. Preserves semantic meaning of mathematical expressions
    
    Args:
        text (str): Scientific text to normalize
        
    Returns:
        str: Normalized text
    """
    # Dictionary of common LaTeX commands and their Unicode equivalents
    latex_to_unicode = {
        r'\\alpha': 'α', r'\\beta': 'β', r'\\gamma': 'γ', r'\\delta': 'δ',
        r'\\epsilon': 'ε', r'\\zeta': 'ζ', r'\\eta': 'η', r'\\theta': 'θ',
        r'\\iota': 'ι', r'\\kappa': 'κ', r'\\lambda': 'λ', r'\\mu': 'μ',
        r'\\nu': 'ν', r'\\xi': 'ξ', r'\\pi': 'π', r'\\rho': 'ρ',
        r'\\sigma': 'σ', r'\\tau': 'τ', r'\\upsilon': 'υ', r'\\phi': 'φ',
        r'\\chi': 'χ', r'\\psi': 'ψ', r'\\omega': 'ω',
        r'\\Gamma': 'Γ', r'\\Delta': 'Δ', r'\\Theta': 'Θ', r'\\Lambda': 'Λ',
        r'\\Xi': 'Ξ', r'\\Pi': 'Π', r'\\Sigma': 'Σ', r'\\Phi': 'Φ',
        r'\\Psi': 'Ψ', r'\\Omega': 'Ω',
        r'\\times': '×', r'\\div': '÷', r'\\pm': '±', r'\\mp': '∓',
        r'\\leq': '≤', r'\\geq': '≥', r'\\neq': '≠', r'\\approx': '≈',
        r'\\propto': '∝', r'\\partial': '∂', r'\\infty': '∞',
        r'\\sum': '∑', r'\\prod': '∏', r'\\int': '∫',
        r'\\rightarrow': '→', r'\\leftarrow': '←', r'\\Rightarrow': '⇒',
        r'\\Leftarrow': '⇐', r'\\forall': '∀', r'\\exists': '∃',
        r'\\in': '∈', r'\\subset': '⊂', r'\\supset': '⊃',
        r'\\cup': '∪', r'\\cap': '∩', r'\\emptyset': '∅'
    }
    
    # Replace LaTeX commands with Unicode equivalents
    for latex, unicode in latex_to_unicode.items():
        text = re.sub(latex, unicode, text)
    
    # Normalize common physics notation
    physics_patterns = {
        r'\\mathrm{([^}]+)}': r'\1',  # Remove \mathrm{}
        r'\\text{([^}]+)}': r'\1',    # Remove \text{}
        r'\\textbf{([^}]+)}': r'\1',  # Remove \textbf{}
        r'\\textit{([^}]+)}': r'\1',  # Remove \textit{}
        r'\\sqrt{([^}]+)}': r'sqrt(\1)',  # Convert \sqrt{x} to sqrt(x)
        r'\\frac{([^}]+)}{([^}]+)}': r'\1/\2',  # Convert \frac{a}{b} to a/b
        r'\\left\(': '(',  # Replace \left( with (
        r'\\right\)': ')',  # Replace \right) with )
        r'\\left\[': '[',  # Replace \left[ with [
        r'\\right\]': ']',  # Replace \right] with ]
        r'\^{([^}]+)}': r'^(\1)',  # Convert x^{abc} to x^(abc)
        r'_{([^}]+)}': r'_\1',      # Convert x_{abc} to x_abc
        r'\\bar{([^}]+)}': r'\1̄',   # Convert \bar{x} to x̄
        r'\\hat{([^}]+)}': r'\1̂',   # Convert \hat{x} to x̂
        r'\\vec{([^}]+)}': r'\1⃗',   # Convert \vec{x} to x⃗
        r'\\dot{([^}]+)}': r'\1̇',   # Convert \dot{x} to ẋ
        r'\\ddot{([^}]+)}': r'\1̈',  # Convert \ddot{x} to ẍ
    }
    
    # Apply physics notation patterns
    for pattern, replacement in physics_patterns.items():
        text = re.sub(pattern, replacement, text)
    
    # Clean up common LaTeX artifacts
    cleanup_patterns = {
        r'\\\\': ' ',  # Replace \\ (newline in LaTeX) with space
        r'\\;': ' ',   # Replace \; (space in LaTeX) with space
        r'\\,': ' ',   # Replace \, (small space in LaTeX) with space
        r'~': ' ',     # Replace ~ (non-breaking space) with space
        r'\\%': '%',   # Replace \% with %
        r'\\&': '&',   # Replace \& with &
        r'\\$': '$',   # Replace \$ with $
        r'\\#': '#',   # Replace \# with #
        r'\\_': '_',   # Replace \_ with _
        r'\\{': '{',   # Replace \{ with {
        r'\\}': '}',   # Replace \} with }
    }
    
    # Apply cleanup patterns
    for pattern, replacement in cleanup_patterns.items():
        text = re.sub(pattern, replacement, text)
    
    # Normalize subscripts and superscripts in chemical formulas and equations
    # Convert H_2O to H₂O, CO_2 to CO₂, etc.
    subscript_map = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', 
                     '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}
    
    def replace_subscript(match):
        digit = match.group(2)
        return match.group(1) + subscript_map.get(digit, digit)
    
    # Apply subscript replacement for patterns like H_2O
    text = re.sub(r'([A-Za-z])_(\d)', replace_subscript, text)
    
    # Normalize Unicode characters that might be inconsistently represented
    unicode_normalization = {
        '\u2212': '-',  # MINUS SIGN to hyphen
        '\u00d7': 'x',  # MULTIPLICATION SIGN to x
        '\u2013': '-',  # EN DASH to hyphen
        '\u2014': '-',  # EM DASH to hyphen
        '\u201c': '"',  # LEFT DOUBLE QUOTATION MARK to straight quote
        '\u201d': '"',  # RIGHT DOUBLE QUOTATION MARK to straight quote
        '\u2018': "'",  # LEFT SINGLE QUOTATION MARK to apostrophe
        '\u2019': "'",  # RIGHT SINGLE QUOTATION MARK to apostrophe
    }
    
    for unicode_char, replacement in unicode_normalization.items():
        text = text.replace(unicode_char, replacement)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def chunk_text(text, chunk_size, chunk_overlap, min_chunk_size):
    """
    Split text into chunks of specified size with overlap
    
    Args:
        text (str): Text to split
        chunk_size (int): Target chunk size in words
        chunk_overlap (int): Number of words to overlap between chunks
        min_chunk_size (int): Minimum chunk size in words
        
    Returns:
        list: List of text chunks
    """
    # Split text into words
    words = text.split()
    
    # If text is shorter than min_chunk_size, return as is
    if len(words) < min_chunk_size:
        return [text]
    
    # If text is shorter than chunk_size, return as is
    if len(words) <= chunk_size:
        return [text]
    
    # Create chunks
    chunks = []
    start = 0
    
    while start < len(words):
        # Calculate end position
        end = min(start + chunk_size, len(words))
        
        # Create chunk
        chunk = " ".join(words[start:end])
        
        # Only add chunk if it's not a citation
        if not is_citation_text(chunk):
            chunks.append(chunk)
        else:
            logger.debug(f"Filtered out citation chunk: {chunk[:100]}...")
        
        # Move start position for next chunk
        start += chunk_size - chunk_overlap
    
    return chunks

def process_paper(arxiv_id, tokenizer, model, faiss_index, next_index):
    """
    Process a paper: download, extract text, chunk, embed, and store
    
    Args:
        arxiv_id (str): arXiv ID
        tokenizer: Tokenizer for the embedding model
        model: Embedding model
        faiss_index: FAISS index
        next_index (int): Next available index in FAISS
        
    Returns:
        tuple: (chunks_added, next_index)
    """
    logger.info(f"DEBUG: Starting to process paper {arxiv_id}")
    chunks_added = 0
    
    # Ensure we have the content
    if not ensure_paper_content(arxiv_id):
        logger.warning(f"Failed to ensure content for {arxiv_id}")
        logger.info(f"DEBUG: Paper {arxiv_id} - Failed to ensure content")
        return chunks_added, next_index
    
    # Extract text from ar5iv
    sections = extract_text_from_ar5iv(arxiv_id)
    
    if not sections or len(sections) <= 1:  # Only metadata or empty
        logger.warning(f"No text extracted for {arxiv_id}")
        logger.info(f"DEBUG: Paper {arxiv_id} - No text extracted or only metadata")
        return chunks_added, next_index
    
    logger.info(f"DEBUG: Paper {arxiv_id} - Extracted {len(sections)} sections")
    
    # Process each section
    all_chunks = []
    
    for section_title, section_text in sections.items():
        if section_title == "metadata":
            # Special handling for metadata
            metadata = section_text
            metadata_text = f"Title: {metadata['title']}\nAbstract: {metadata['abstract']}"
            chunks = chunk_text(normalize_scientific_text(metadata_text), CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE)
            for chunk in chunks:
                all_chunks.append({
                    "section_title": "Metadata",
                    "chunk_text": chunk,
                    "chunk_position": len(all_chunks)
                })
        else:
            # Regular section
            if not section_text or len(section_text.strip()) < MIN_CHUNK_SIZE:
                continue
                
            chunks = chunk_text(normalize_scientific_text(section_text), CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE)
            for chunk in chunks:
                all_chunks.append({
                    "section_title": section_title,
                    "chunk_text": chunk,
                    "chunk_position": len(all_chunks)
                })
    
    # If no chunks were created, log and return
    if not all_chunks:
        logger.warning(f"No chunks created for {arxiv_id}")
        logger.info(f"DEBUG: Paper {arxiv_id} - No chunks created")
        return chunks_added, next_index
    
    logger.info(f"DEBUG: Paper {arxiv_id} - Created {len(all_chunks)} chunks")
    
    # Embed and store chunks
    with setup_database() as conn:
        for chunk in all_chunks:
            chunk_hash = hashlib.sha256(chunk["chunk_text"].encode()).hexdigest()
            embedding = generate_embedding(chunk["chunk_text"], tokenizer, model)
            faiss_index.add(np.array([embedding]))
            embedding_index = next_index
            next_index += 1
            
            conn.execute(
                "INSERT INTO chunks (chunk_id, arxiv_id, chunk_hash, embedding_index, chunk_text, section_title, chunk_position, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (chunk_hash, arxiv_id, chunk_hash, embedding_index, chunk["chunk_text"], chunk["section_title"], chunk["chunk_position"], datetime.now().isoformat())
            )
            conn.commit()
            chunks_added += 1
    
    logger.info(f"DEBUG: Paper {arxiv_id} - Embedded and stored {chunks_added} chunks")
    return chunks_added, next_index

def process_paper_metadata(arxiv_id, tokenizer, model, metadata_faiss_index, next_index):
    """
    Process a paper's metadata: extract, embed, and store
    
    Args:
        arxiv_id (str): arXiv ID
        tokenizer: Tokenizer for the embedding model
        model: Embedding model
        metadata_faiss_index: FAISS index for metadata
        next_index (int): Next available index in FAISS
        
    Returns:
        tuple: (success, next_index)
    """
    logger.info(f"DEBUG: Starting to process metadata for paper {arxiv_id}")
    logger.info(f"Processing metadata for paper {arxiv_id}")
    
    # Connect to databases
    with sqlite3.connect(METADATA_DB) as conn_metadata, sqlite3.connect(METADATA_INDEX_DB) as conn_embeddings:
        cursor_metadata = conn_metadata.execute(
            "SELECT title, summary, authors, categories, published, updated, comments, doi FROM papers WHERE arxiv_id = ?", 
            (arxiv_id,)
        )
        
        paper_data = cursor_metadata.fetchone()
        if not paper_data:
            logger.warning(f"No metadata found for paper {arxiv_id}")
            logger.info(f"DEBUG: Paper {arxiv_id} - No metadata found in database")
            return False, next_index
        
        logger.info(f"DEBUG: Paper {arxiv_id} - Found metadata in database")
        
        title, summary, authors, categories, published, updated, comments, doi = paper_data
        
        # Check if this paper's metadata has already been embedded
        cursor_check = conn_embeddings.execute("SELECT arxiv_id FROM metadata_embeddings WHERE arxiv_id = ?", (arxiv_id,))
        if cursor_check.fetchone():
            logger.info(f"DEBUG: Paper {arxiv_id} - Metadata already exists, skipping")
            return False, next_index
        
        # Combine all metadata fields into a single text
        combined_text = f"Title: {title}\nAbstract: {summary}\nAuthors: {authors}\nCategories: {categories}\nPublished: {published}\nUpdated: {updated}\nComments: {comments}\nDOI: {doi}"
        
        logger.info(f"DEBUG: Paper {arxiv_id} - Combined metadata text: {combined_text[:100]}...")
        
        # Generate embedding for the combined metadata
        embedding = generate_embedding(normalize_scientific_text(combined_text), tokenizer, model)
        
        logger.info(f"DEBUG: Paper {arxiv_id} - Generated metadata embedding")
        
        # Add to FAISS index
        metadata_faiss_index.add(np.array([embedding]))
        embedding_index = next_index
        next_index += 1
        
        logger.info(f"DEBUG: Paper {arxiv_id} - Added metadata embedding at index {embedding_index}")
        
        # Store in database
        conn_embeddings.execute(
            "INSERT INTO metadata_embeddings (arxiv_id, embedding_index, metadata_text, processed_at) VALUES (?, ?, ?, ?)",
            (arxiv_id, next_index, combined_text, datetime.now().isoformat())
        )
        
        conn_embeddings.commit()
        
        logger.info(f"DEBUG: Paper {arxiv_id} - Added metadata embedding to database")
        
        logger.info(f"Added metadata embedding for paper {arxiv_id}")
        logger.info(f"DEBUG: Paper {arxiv_id} - Finished processing metadata successfully")
    return True, next_index + 1

def generate_embedding(text, tokenizer, model):
    """
    Generate embedding for text
    
    Args:
        text (str): Text to embed
        tokenizer: Tokenizer for the embedding model
        model: Embedding model
        
    Returns:
        numpy.ndarray: Embedding vector
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DEVICE = torch.device(device)
    
    # Tokenize the text
    tokens = tokenizer.encode(text)
    
    # If text is shorter than the maximum length, embed directly
    if len(tokens) <= WINDOW_SIZE:
        # Tokenize and prepare for model
        inputs = tokenizer(text, padding=True, truncation=True, max_length=WINDOW_SIZE, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()  # Use [CLS] token embedding
        
        # Normalize for cosine similarity
        embedding = embeddings[0]
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
            
        return embedding
    
    # For longer text, use sliding window approach
    logger.debug(f"Using sliding window approach for text with {len(tokens)} tokens")
    
    # Create sliding windows
    windows = []
    for i in range(0, len(tokens) - WINDOW_SIZE + 1, WINDOW_STRIDE):
        window_tokens = tokens[i:i+WINDOW_SIZE]
        window_text = tokenizer.decode(window_tokens)
        windows.append(window_text)
    
    # If there's a final segment that would be missed, add it
    if len(tokens) % WINDOW_STRIDE != 0:
        final_window_tokens = tokens[-WINDOW_SIZE:] if len(tokens) >= WINDOW_SIZE else tokens
        final_window_text = tokenizer.decode(final_window_tokens)
        windows.append(final_window_text)
    
    # Get embedding for each window
    window_embeddings = []
    for window in windows:
        # Tokenize and prepare for model
        inputs = tokenizer(window, padding=True, truncation=True, max_length=WINDOW_SIZE, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
        
        window_embeddings.append(embeddings[0])
    
    # Aggregate embeddings (mean pooling)
    combined_embedding = np.mean(window_embeddings, axis=0)
    
    # Normalize for cosine similarity
    embedding_norm = np.linalg.norm(combined_embedding)
    if embedding_norm > 0:
        combined_embedding = combined_embedding / embedding_norm
    
    return combined_embedding

def get_total_papers():
    """
    Get total number of papers in the metadata database
    
    Returns:
        int: Total number of papers
    """
    with sqlite3.connect(METADATA_DB) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM papers")
        return cursor.fetchone()[0]

def load_or_create_faiss_index():
    """
    Load existing FAISS index or create a new one
    
    Returns:
        tuple: (faiss.Index, next_index)
    """
    if os.path.exists(FAISS_INDEX_PATH):
        logger.info(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
        index = faiss.read_index(FAISS_INDEX_PATH)
        next_index = index.ntotal
        logger.info(f"Loaded FAISS index with {next_index} vectors")
    else:
        logger.info("Creating new FAISS index")
        index = faiss.IndexFlatL2(EMBEDDING_DIM)  # L2 distance, can be converted to cosine with normalized vectors
        next_index = 0
    
    return index, next_index

def load_or_create_metadata_faiss_index():
    """
    Load existing metadata FAISS index or create a new one
    
    Returns:
        tuple: (faiss.Index, next_index)
    """
    os.makedirs(os.path.dirname(METADATA_FAISS_INDEX_PATH), exist_ok=True)
    
    if os.path.exists(METADATA_FAISS_INDEX_PATH):
        logger.info(f"Loading existing metadata FAISS index from {METADATA_FAISS_INDEX_PATH}")
        index = faiss.read_index(METADATA_FAISS_INDEX_PATH)
        next_index = index.ntotal
        logger.info(f"Loaded metadata FAISS index with {next_index} vectors")
    else:
        logger.info("Creating new metadata FAISS index")
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        next_index = 0
    
    return index, next_index

def save_faiss_index(index, index_path=FAISS_INDEX_PATH):
    """
    Save FAISS index to disk
    
    Args:
        index: FAISS index
        index_path: Path to save the index
    """
    logger.info(f"Saving FAISS index with {index.ntotal} vectors to {index_path}")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

def save_metadata_faiss_index(index, index_path=METADATA_FAISS_INDEX_PATH):
    """
    Save metadata FAISS index to disk
    
    Args:
        index: FAISS index
        index_path: Path to save the index
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    logger.info(f"Saving metadata FAISS index with {index.ntotal} vectors to {index_path}")
    faiss.write_index(index, index_path)

def run_content_embedding(force_update=False, specific_arxiv_id=None):
    """
    Run the content embedding process
    
    Args:
        force_update (bool): Whether to force update all papers
        specific_arxiv_id (str): Specific arXiv ID to process
        
    Returns:
        bool: Success status
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        # Debug: Print initial database and FAISS index state
        with sqlite3.connect(CHUNK_INDEX_DB) as conn_chunks:
            cursor = conn_chunks.execute("SELECT COUNT(DISTINCT arxiv_id) FROM chunks")
            initial_papers_with_chunks = cursor.fetchone()[0]
            
            cursor = conn_chunks.execute("SELECT COUNT(*) FROM chunks")
            initial_chunks = cursor.fetchone()[0]
            
            cursor = conn_chunks.execute("SELECT DISTINCT arxiv_id FROM chunks")
            initial_papers_list = [row[0] for row in cursor.fetchall()]
            
        with sqlite3.connect(METADATA_INDEX_DB) as conn_metadata:
            cursor = conn_metadata.execute("SELECT COUNT(*) FROM metadata_embeddings")
            initial_metadata = cursor.fetchone()[0]
            
            cursor = conn_metadata.execute("SELECT DISTINCT arxiv_id FROM metadata_embeddings")
            initial_metadata_papers = [row[0] for row in cursor.fetchall()]
        
        logger.info(f"DEBUG: INITIAL STATE - {initial_papers_with_chunks} papers with chunks, {initial_chunks} total chunks, {initial_metadata} metadata embeddings")
        logger.info(f"DEBUG: INITIAL PAPERS WITH CHUNKS: {initial_papers_list}")
        logger.info(f"DEBUG: INITIAL PAPERS WITH METADATA: {initial_metadata_papers}")
        # Get list of papers to process
        unprocessed_papers = get_unprocessed_papers(force_update, specific_arxiv_id)
        if not unprocessed_papers:
            logger.info("No papers to process")
            
            # Debug: Print final database and FAISS index state
            with sqlite3.connect(CHUNK_INDEX_DB) as conn_chunks:
                cursor = conn_chunks.execute("SELECT COUNT(DISTINCT arxiv_id) FROM chunks")
                final_papers_with_chunks = cursor.fetchone()[0]
                
                cursor = conn_chunks.execute("SELECT COUNT(*) FROM chunks")
                final_chunks = cursor.fetchone()[0]
                
            with sqlite3.connect(METADATA_INDEX_DB) as conn_metadata:
                cursor = conn_metadata.execute("SELECT COUNT(*) FROM metadata_embeddings")
                final_metadata = cursor.fetchone()[0]
            
            logger.info(f"DEBUG: FINAL STATE - {final_papers_with_chunks} papers with chunks, {final_chunks} total chunks, {final_metadata} metadata embeddings")
            return True
        
        # Load embedding model
        tokenizer, model = load_embedding_model()
        
        # Load or create FAISS indices
        content_faiss_index, content_next_index = load_or_create_faiss_index()
        metadata_faiss_index, metadata_next_index = load_or_create_metadata_faiss_index()
        
        # Load progress tracker
        progress_tracker = load_progress_tracker()
        
        # Process papers in batches
        total_papers = len(unprocessed_papers)
        total_content_chunks_added = 0
        total_metadata_papers_processed = 0
        
        for i, arxiv_id in enumerate(tqdm(unprocessed_papers, desc="Processing papers")):
            try:
                # Process content
                content_chunks_added, content_next_index = process_paper(
                    arxiv_id, tokenizer, model, content_faiss_index, content_next_index
                )
                total_content_chunks_added += content_chunks_added
                
                # Process metadata
                metadata_processed, metadata_next_index = process_paper_metadata(
                    arxiv_id, tokenizer, model, metadata_faiss_index, metadata_next_index
                )
                if metadata_processed:
                    total_metadata_papers_processed += 1
                
                # Update progress tracker
                if arxiv_id not in progress_tracker["processed_papers"]:
                    progress_tracker["processed_papers"].append(arxiv_id)
                progress_tracker["total_chunks"] += content_chunks_added
                progress_tracker["total_papers"] = len(progress_tracker["processed_papers"])
                
                # Save progress tracker
                save_progress_tracker(progress_tracker)
                
                # Save FAISS indices periodically
                if (i + 1) % MAX_PAPERS_PER_BATCH == 0 or i == total_papers - 1:
                    save_faiss_index(content_faiss_index)
                    save_metadata_faiss_index(metadata_faiss_index)
                    
                    logger.info(f"Progress: {i+1}/{total_papers} papers, {total_content_chunks_added} content chunks, {total_metadata_papers_processed} metadata embeddings")
                
            except Exception as e:
                logger.error(f"Error processing paper {arxiv_id}: {str(e)}")
                continue
        
        # Final save
        save_faiss_index(content_faiss_index)
        save_metadata_faiss_index(metadata_faiss_index)
        
        # Get the total number of papers and chunks in the database for accurate reporting
        with sqlite3.connect(CHUNK_INDEX_DB) as conn_chunks:
            cursor = conn_chunks.execute("SELECT COUNT(DISTINCT arxiv_id) FROM chunks")
            total_papers_with_chunks = cursor.fetchone()[0]
            
            cursor = conn_chunks.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            cursor = conn_chunks.execute("SELECT DISTINCT arxiv_id FROM chunks")
            final_papers_list = [row[0] for row in cursor.fetchall()]
            
        with sqlite3.connect(METADATA_INDEX_DB) as conn_metadata:
            cursor = conn_metadata.execute("SELECT COUNT(*) FROM metadata_embeddings")
            total_metadata = cursor.fetchone()[0]
            
            cursor = conn_metadata.execute("SELECT DISTINCT arxiv_id FROM metadata_embeddings")
            final_metadata_papers = [row[0] for row in cursor.fetchall()]
        
        # Debug: Compare initial and final states
        new_papers = set(final_papers_list) - set(initial_papers_list)
        new_metadata_papers = set(final_metadata_papers) - set(initial_metadata_papers)
        
        logger.info(f"DEBUG: FINAL STATE - {total_papers_with_chunks} papers with chunks, {total_chunks} total chunks, {total_metadata} metadata embeddings")
        logger.info(f"DEBUG: NEW PAPERS ADDED: {new_papers}")
        logger.info(f"DEBUG: NEW METADATA PAPERS ADDED: {new_metadata_papers}")
        logger.info(f"DEBUG: FINAL PAPERS WITH CHUNKS: {final_papers_list}")
        logger.info(f"DEBUG: FINAL PAPERS WITH METADATA: {final_metadata_papers}")
        
        # Check if FAISS indices were updated
        logger.info(f"DEBUG: FAISS content index has {content_faiss_index.ntotal} vectors")
        logger.info(f"DEBUG: FAISS metadata index has {metadata_faiss_index.ntotal} vectors")
            
        logger.info(f"Content embedding completed. Processed {len(progress_tracker['processed_papers'])} papers, {total_chunks} content chunks, {total_metadata} metadata embeddings")
        return True
        
    except Exception as e:
        logger.error(f"Error in content embedding: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for arXiv papers")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--max-papers", type=int, help="Maximum number of papers to process")
    
    args = parser.parse_args()
    
    run_content_embedding(batch_size=args.batch_size, max_papers=args.max_papers)
