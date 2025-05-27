#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalable Content Embedding Module

This module implements a scalable full-text extraction and embedding pipeline
with a priority hierarchy for content sources:
1. Pre-filtered local S2ORC subset
2. ar5iv HTML
3. GROBID XML
4. PDF text extraction

Each source yields structured sections and text paragraphs, which are chunked,
filtered, and embedded using intfloat/e5-small-v2.
"""

import os
import re
import json
import time
import glob
import uuid
import logging
import sqlite3
import tarfile
import hashlib
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Iterator, Any, Union, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from threading import Lock

# Text processing
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# ML libraries
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Import parameters
from content_embedding_scalable_parameters import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    CONTENT_SOURCES,
    METADATA_DB,
    CHUNK_DB,
    FAISS_INDEX,
    METADATA_FAISS_INDEX,
    S2ORC_DIR,
    AR5IV_DIR,
    GROBID_DIR,
    PDF_DIR,
    BATCH_SIZE,
    LOW_INFO_THRESHOLD,
    MAX_WORKERS,
    ARXIV_SOURCE_BASE_URL,
    ARXIV_PDF_BASE_URL,
    AR5IV_BASE_URL
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Type definitions
ParagraphType = Dict[str, str]
SectionType = Dict[str, Union[str, List[ParagraphType]]]
DocumentType = Dict[str, Union[str, List[SectionType]]]
ChunkType = Dict[str, Union[str, int, float]]

# Database management
@contextmanager
def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Create a database connection and yield it, closing it after use."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def initialize_chunk_db() -> None:
    """Initialize the chunk database if it doesn't exist."""
    os.makedirs(os.path.dirname(CHUNK_DB), exist_ok=True)
    
    with get_db_connection(CHUNK_DB) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            arxiv_id TEXT NOT NULL,
            section_title TEXT,
            chunk_index INTEGER,
            chunk_text TEXT,
            embedding_index INTEGER,
            processed_at TIMESTAMP
        )
        ''')
        
        # Create index on arxiv_id for faster lookups
        conn.execute('CREATE INDEX IF NOT EXISTS idx_arxiv_id ON chunks(arxiv_id)')
        
        # Create metadata embeddings table
        conn.execute('''
        CREATE TABLE IF NOT EXISTS metadata_embeddings (
            chunk_id TEXT PRIMARY KEY,
            arxiv_id TEXT NOT NULL UNIQUE,
            metadata_text TEXT,
            embedding_index INTEGER,
            processed_at TIMESTAMP
        )
        ''')
        
        # Create index on arxiv_id for metadata embeddings
        conn.execute('CREATE INDEX IF NOT EXISTS idx_metadata_arxiv_id ON metadata_embeddings(arxiv_id)')
        
        conn.commit()

def get_processed_papers() -> Set[str]:
    """Get the set of arxiv_ids that have already been processed."""
    if not os.path.exists(CHUNK_DB):
        return set()
        
    with get_db_connection(CHUNK_DB) as conn:
        cursor = conn.execute('SELECT DISTINCT arxiv_id FROM chunks')
        return {row['arxiv_id'] for row in cursor.fetchall()}

def get_processed_metadata_papers() -> Set[str]:
    """Get the set of arxiv_ids that have already had metadata processed."""
    if not os.path.exists(CHUNK_DB):
        return set()
        
    with get_db_connection(CHUNK_DB) as conn:
        cursor = conn.execute('SELECT arxiv_id FROM metadata_embeddings')
        return {row['arxiv_id'] for row in cursor.fetchall()}

def get_paper_metadata(arxiv_id: str, conn_metadata: sqlite3.Connection) -> Tuple[str, str]:
    """Get paper metadata (title and abstract) from the metadata database."""
    cursor = conn_metadata.execute(
        'SELECT title, abstract FROM papers WHERE arxiv_id = ?',
        (arxiv_id,)
    )
    row = cursor.fetchone()
    if row:
        return row['title'], row['abstract']
    return None, None

def generate_chunk_id(arxiv_id: str, section_title: str, chunk_index: int) -> str:
    """Generate a unique ID for a chunk based on its content."""
    content = f"{arxiv_id}_{section_title}_{chunk_index}"
    return hashlib.md5(content.encode()).hexdigest()

# Memory management
def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# S2ORC dictionary management
class S2ORCManager:
    """Manager for S2ORC data that loads and provides access to papers by arxiv_id."""
    
    def __init__(self, s2orc_dir: str):
        self.s2orc_dir = s2orc_dir
        self.paper_paths = {}
        self.paper_cache = {}  # Cache for frequently accessed papers
        self.cache_lock = Lock()  # Lock for thread-safe cache access
        self._index_papers()
        
    def _index_papers(self) -> None:
        """Index all S2ORC papers by arxiv_id."""
        logger.info("Indexing S2ORC papers...")
        for json_file in glob.glob(os.path.join(self.s2orc_dir, "*.json")):
            try:
                arxiv_id = os.path.basename(json_file).replace(".json", "")
                self.paper_paths[arxiv_id] = json_file
            except Exception as e:
                logger.error(f"Error indexing S2ORC paper {json_file}: {e}")
        logger.info(f"Indexed {len(self.paper_paths)} S2ORC papers")
        
        # Optionally pre-load frequently accessed papers
        # self._preload_common_papers()
    
    def _preload_common_papers(self, max_papers: int = 100) -> None:
        """Preload the most commonly accessed papers into memory."""
        # This could be based on access statistics or other criteria
        # For now, just load the first N papers
        count = 0
        for arxiv_id in list(self.paper_paths.keys())[:max_papers]:
            try:
                with open(self.paper_paths[arxiv_id], 'r', encoding='utf-8') as f:
                    self.paper_cache[arxiv_id] = json.load(f)
                count += 1
            except Exception as e:
                logger.error(f"Error preloading S2ORC paper {arxiv_id}: {e}")
        
        logger.info(f"Preloaded {count} S2ORC papers into memory cache")
    
    def get_paper(self, arxiv_id: str) -> Optional[Dict]:
        """Get a paper by arxiv_id with caching."""
        if arxiv_id not in self.paper_paths:
            return None
        
        # Check cache first
        with self.cache_lock:
            if arxiv_id in self.paper_cache:
                return self.paper_cache[arxiv_id]
            
        try:
            with open(self.paper_paths[arxiv_id], 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            # Add to cache if it's not too large
            with self.cache_lock:
                if len(self.paper_cache) < 1000:  # Limit cache size
                    self.paper_cache[arxiv_id] = paper_data
                
            return paper_data
        except Exception as e:
            logger.error(f"Error loading S2ORC paper {arxiv_id}: {e}")
            return None
            
    def has_paper(self, arxiv_id: str) -> bool:
        """Check if a paper exists in the S2ORC index."""
        return arxiv_id in self.paper_paths
        
    def clear_cache(self) -> None:
        """Clear the paper cache to free memory."""
        with self.cache_lock:
            self.paper_cache.clear()
        logger.info("Cleared S2ORC paper cache")

# Text extraction functions
def extract_text_from_s2orc(arxiv_id: str, s2orc_manager: S2ORCManager) -> Tuple[List[str], List[str]]:
    """Extract text from S2ORC JSON."""
    paper_data = s2orc_manager.get_paper(arxiv_id)
    if not paper_data:
        return [], []
        
    paragraphs = []
    section_titles = []
    
    # Extract text from body_text
    if 'body_text' in paper_data:
        for section in paper_data['body_text']:
            if 'section' in section and 'text' in section:
                section_titles.append(section['section'])
                paragraphs.append(section['text'])
    
    # Ensure alignment of paragraphs and section_titles
    if len(paragraphs) != len(section_titles):
        logger.warning(f"Misaligned sections in S2ORC for {arxiv_id}. Paragraphs: {len(paragraphs)}, Sections: {len(section_titles)}")
        # Pad or truncate to ensure equal length
        if len(paragraphs) > len(section_titles):
            for i in range(len(section_titles), len(paragraphs)):
                section_titles.append(f"Unnamed Section {i+1}")
        elif len(section_titles) > len(paragraphs):
            for i in range(len(paragraphs), len(section_titles)):
                paragraphs.append("")
    
    return paragraphs, section_titles

def extract_text_from_ar5iv(arxiv_id: str) -> Tuple[List[str], List[str]]:
    """Extract text from ar5iv HTML."""
    ar5iv_path = os.path.join(AR5IV_DIR, f"{arxiv_id}.html")
    if not os.path.exists(ar5iv_path):
        # Try to download from ar5iv
        ar5iv_url = f"{AR5IV_BASE_URL}/{arxiv_id}"
        try:
            response = requests.get(ar5iv_url)
            response.raise_for_status()
            
            # Save the HTML content
            os.makedirs(os.path.dirname(ar5iv_path), exist_ok=True)
            with open(ar5iv_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        except Exception as e:
            logger.error(f"Error downloading ar5iv HTML for {arxiv_id}: {e}")
            return [], []
    
    try:
        with open(ar5iv_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find and remove bibliography/references sections
        bibliography_sections = soup.find_all(['div', 'section'], 
                                          class_=['ltx_bibliography', 'ltx_references'])
        for section in bibliography_sections:
            section.decompose()  # Remove from DOM
        
        paragraphs = []
        section_titles = []
        
        # Extract sections and paragraphs
        sections = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for section in sections:
            section_title = section.get_text().strip()
            
            # Skip sections that are likely references
            citation_section_patterns = [
                'references', 'bibliography', 'citations', 'works cited', 'cited literature'
            ]
            if any(pattern in section_title.lower() for pattern in citation_section_patterns):
                logger.info(f"Skipping citation section: {section_title}")
                continue
                
            section_titles.append(section_title)
            
            # Get all paragraphs until the next section
            current = section.next_sibling
            section_text = []
            
            while current and current.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current.name == 'p':
                    section_text.append(current.get_text().strip())
                current = current.next_sibling
            
            if section_text:
                paragraphs.append(' '.join(section_text))
            else:
                # If no paragraphs found, add an empty one to maintain alignment with section_titles
                paragraphs.append("")
        
        # Ensure alignment of paragraphs and section_titles
        if len(paragraphs) != len(section_titles):
            logger.warning(f"Misaligned sections in ar5iv for {arxiv_id}. Paragraphs: {len(paragraphs)}, Sections: {len(section_titles)}")
            # Pad or truncate to ensure equal length
            if len(paragraphs) > len(section_titles):
                for i in range(len(section_titles), len(paragraphs)):
                    section_titles.append(f"Unnamed Section {i+1}")
            elif len(section_titles) > len(paragraphs):
                for i in range(len(paragraphs), len(section_titles)):
                    paragraphs.append("")
        
        return paragraphs, section_titles
    except Exception as e:
        logger.error(f"Error extracting text from ar5iv HTML for {arxiv_id}: {e}")
        return [], []

def extract_text_from_grobid(arxiv_id: str) -> Tuple[List[str], List[str]]:
    """Extract text from GROBID TEI XML."""
    grobid_path = os.path.join(GROBID_DIR, f"{arxiv_id}.tei.xml")
    if not os.path.exists(grobid_path):
        logger.warning(f"GROBID XML not found for {arxiv_id}")
        return [], []
    
    try:
        tree = ET.parse(grobid_path)
        root = tree.getroot()
        
        # Define XML namespaces
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        
        paragraphs = []
        section_titles = []
        
        # Extract sections
        for div in root.findall('.//tei:div', ns):
            # Get section title
            head = div.find('./tei:head', ns)
            section_title = head.text if head is not None and head.text else "Unnamed Section"
            
            # Skip sections that are likely references
            citation_section_patterns = [
                'references', 'bibliography', 'citations', 'works cited', 'cited literature'
            ]
            if any(pattern in section_title.lower() for pattern in citation_section_patterns):
                logger.info(f"Skipping citation section: {section_title}")
                continue
                
            section_titles.append(section_title)
            
            # Get paragraphs in this section
            section_paragraphs = []
            for p in div.findall('.//tei:p', ns):
                if p.text:
                    section_paragraphs.append(p.text.strip())
            
            if section_paragraphs:
                paragraphs.append(' '.join(section_paragraphs))
            else:
                # If no paragraphs found, add an empty one to maintain alignment with section_titles
                paragraphs.append("")
        
        # If no sections were found, try to extract paragraphs directly
        if not section_titles:
            body = root.find('.//tei:body', ns)
            if body is not None:
                for p in body.findall('.//tei:p', ns):
                    if p.text:
                        paragraphs.append(p.text.strip())
                        section_titles.append("Main Content")
        
        # Ensure alignment of paragraphs and section_titles
        if len(paragraphs) != len(section_titles):
            logger.warning(f"Misaligned sections in GROBID for {arxiv_id}. Paragraphs: {len(paragraphs)}, Sections: {len(section_titles)}")
            # Pad or truncate to ensure equal length
            if len(paragraphs) > len(section_titles):
                for i in range(len(section_titles), len(paragraphs)):
                    section_titles.append(f"Unnamed Section {i+1}")
            elif len(section_titles) > len(paragraphs):
                for i in range(len(paragraphs), len(section_titles)):
                    paragraphs.append("")
        
        return paragraphs, section_titles
    except Exception as e:
        logger.error(f"Error extracting text from GROBID XML for {arxiv_id}: {e}")
        return [], []

def extract_text_from_pdf(arxiv_id: str) -> Tuple[List[str], List[str]]:
    """Extract text from PDF using PyMuPDF."""
    pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    if not os.path.exists(pdf_path):
        # Try to download the PDF
        pdf_url = f"{ARXIV_PDF_BASE_URL}/{arxiv_id}.pdf"
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
            
            # Save the PDF content
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            logger.error(f"Error downloading PDF for {arxiv_id}: {e}")
            return [], []
    
    try:
        doc = fitz.open(pdf_path)
        
        # Extract text from each page
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        # Close the document
        doc.close()
        
        # Try to identify sections using regex patterns
        section_pattern = r'\n([A-Z][A-Za-z\s]{2,50})\n'
        sections = re.split(section_pattern, full_text)
        
        paragraphs = []
        section_titles = []
        
        # First element is text before any section
        if sections:
            paragraphs.append(sections[0].strip())
            section_titles.append("Introduction")
            
            # Process remaining sections
            for i in range(1, len(sections), 2):
                if i < len(sections):
                    section_titles.append(sections[i].strip())
                    if i + 1 < len(sections):
                        paragraphs.append(sections[i + 1].strip())
                    else:
                        paragraphs.append("")
        
        # Ensure alignment of paragraphs and section_titles
        if len(paragraphs) != len(section_titles):
            logger.warning(f"Misaligned sections in PDF for {arxiv_id}. Paragraphs: {len(paragraphs)}, Sections: {len(section_titles)}")
            # Pad or truncate to ensure equal length
            if len(paragraphs) > len(section_titles):
                for i in range(len(section_titles), len(paragraphs)):
                    section_titles.append(f"Unnamed Section {i+1}")
            elif len(section_titles) > len(paragraphs):
                for i in range(len(paragraphs), len(section_titles)):
                    paragraphs.append("")
        
        return paragraphs, section_titles
    except Exception as e:
        logger.error(f"Error extracting text from PDF for {arxiv_id}: {e}")
        return [], []

def ensure_paper_content(arxiv_id: str) -> Dict[str, bool]:
    """Ensure paper content is available for processing.
    
    Returns a dictionary indicating which sources are available.
    """
    # Check if any content source already has this paper
    s2orc_path = os.path.join(S2ORC_DIR, f"{arxiv_id}.json")
    ar5iv_path = os.path.join(AR5IV_DIR, f"{arxiv_id}.html")
    grobid_path = os.path.join(GROBID_DIR, f"{arxiv_id}.xml")
    pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    
    sources_available = {
        "s2orc": os.path.exists(s2orc_path),
        "ar5iv": os.path.exists(ar5iv_path),
        "grobid": os.path.exists(grobid_path),
        "pdf": os.path.exists(pdf_path)
    }
    
    # If S2ORC is not available, try to download the source and create S2ORC-like JSON
    if not sources_available["s2orc"]:
        sources_available["s2orc"] = download_paper_source(arxiv_id)
    
    # If PDF is not available and we need it as a fallback, download it
    if not sources_available["pdf"] and not any(sources_available.values()):
        sources_available["pdf"] = download_pdf(arxiv_id)
    
    return sources_available

def download_paper_source(arxiv_id: str) -> bool:
    """Download paper source from arXiv and create S2ORC-like JSON."""
    # Create directories if they don't exist
    os.makedirs(S2ORC_DIR, exist_ok=True)
    os.makedirs(AR5IV_DIR, exist_ok=True)
    os.makedirs(GROBID_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    
    # arXiv URL for the source
    url = f"{ARXIV_SOURCE_BASE_URL}/{arxiv_id}"
    
    try:
        logger.info(f"Downloading source for {arxiv_id} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Create a temporary directory to extract the source
        temp_dir = os.path.join(os.path.dirname(S2ORC_DIR), "temp", arxiv_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the source tarball
        tarball_path = os.path.join(temp_dir, f"{arxiv_id}.tar.gz")
        with open(tarball_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the source
        try:
            with tarfile.open(tarball_path) as tar:
                tar.extractall(path=temp_dir)
            
            # Look for LaTeX files
            tex_files = glob.glob(os.path.join(temp_dir, "*.tex"))
            if tex_files:
                # Create a simple JSON representation (S2ORC-like)
                s2orc_data = {
                    "paper_id": arxiv_id,
                    "body_text": []
                }
                
                # Process each TeX file
                for tex_file in tex_files:
                    try:
                        with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # More robust TeX parsing
                        # Look for sections, subsections, and subsubsections
                        section_patterns = [
                            r'\\section\{([^}]+)\}',
                            r'\\subsection\{([^}]+)\}',
                            r'\\subsubsection\{([^}]+)\}'
                        ]
                        
                        # Start with the whole document
                        current_text = content
                        current_section = "Document"
                        
                        # First pass: extract sections
                        for pattern in section_patterns:
                            sections = re.split(pattern, current_text)
                            if len(sections) > 1:  # Found sections
                                # Process each section
                                for i in range(1, len(sections), 2):
                                    if i < len(sections):
                                        section_title = sections[i].strip()
                                        section_text = sections[i+1] if i+1 < len(sections) else ""
                                        
                                        # Clean up the text
                                        section_text = re.sub(r'\\cite\{[^}]+\}', '', section_text)
                                        section_text = re.sub(r'\\ref\{[^}]+\}', '', section_text)
                                        section_text = re.sub(r'\\label\{[^}]+\}', '', section_text)
                                        
                                        # Add to S2ORC data
                                        s2orc_data["body_text"].append({
                                            "section": section_title,
                                            "text": section_text
                                        })
                                break  # Stop after finding sections at one level
                        
                        # If no sections found, add the whole document
                        if len(s2orc_data["body_text"]) == 0:
                            s2orc_data["body_text"].append({
                                "section": "Document",
                                "text": content
                            })
                    
                    except Exception as e:
                        logger.error(f"Error processing TeX file {tex_file}: {e}")
                
                # Save as S2ORC JSON
                s2orc_path = os.path.join(S2ORC_DIR, f"{arxiv_id}.json")
                with open(s2orc_path, 'w', encoding='utf-8') as f:
                    json.dump(s2orc_data, f, indent=2)
                
                logger.info(f"Created S2ORC-like JSON at {s2orc_path}")
                return True
            else:
                logger.warning(f"No LaTeX files found for {arxiv_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting source for {arxiv_id}: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading source for {arxiv_id}: {e}")
        return False

def download_pdf(arxiv_id: str) -> bool:
    """Download PDF from arXiv."""
    pdf_url = f"{ARXIV_PDF_BASE_URL}/{arxiv_id}.pdf"
    pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    
    try:
        logger.info(f"Downloading PDF for {arxiv_id} from {pdf_url}")
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # Save the PDF content
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded PDF to {pdf_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading PDF for {arxiv_id}: {e}")
        return False

def delete_source_files(arxiv_id: str) -> None:
    """Delete source files after processing to conserve storage."""
    # Delete temporary directory
    temp_dir = os.path.join(os.path.dirname(S2ORC_DIR), "temp", arxiv_id)
    if os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Deleted temporary directory {temp_dir}")
        except Exception as e:
            logger.error(f"Error deleting temporary directory {temp_dir}: {e}")
    
    # Optionally delete PDF after processing
    # Uncomment if you want to delete PDFs after processing
    # pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    # if os.path.exists(pdf_path):
    #     try:
    #         os.remove(pdf_path)
    #         logger.info(f"Deleted PDF {pdf_path}")
    #     except Exception as e:
    #         logger.error(f"Error deleting PDF {pdf_path}: {e}")

# Text chunking and filtering
def is_low_info_chunk(text: str, threshold: float = LOW_INFO_THRESHOLD) -> bool:
    """Check if a chunk has low information content."""
    if not text or len(text.strip()) == 0:
        return True
        
    # Count unique words
    words = text.lower().split()
    unique_words = set(words)
    
    # Calculate information density
    if len(words) == 0:
        return True
    
    # Ratio of unique words to total words
    unique_ratio = len(unique_words) / len(words)
    
    # Check for repetitive patterns
    repetition_score = 0
    if len(words) > 10:
        # Check for repeated n-grams
        for n in range(2, 5):  # Check for 2, 3, and 4-grams
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            ngram_counts = {}
            for ngram in ngrams:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            
            # Calculate repetition score
            if ngrams:
                max_count = max(ngram_counts.values()) if ngram_counts else 0
                repetition_score += max_count / len(ngrams)
    
    repetition_score = repetition_score / 3  # Average across 2, 3, and 4-grams
    
    # Combined score
    info_score = unique_ratio * (1 - repetition_score)
    
    return info_score < threshold

def is_citation_text(text: str) -> bool:
    """Check if a chunk of text is primarily citations."""
    # Common citation patterns
    citation_patterns = [
        r'\[\d+\]',                    # [1], [2], etc.
        r'\[\w+\s*\d{4}\]',            # [Smith 2020], [Jones et al. 2019]
        r'\(\w+\s*et\s*al\.\s*\d{4}\)', # (Smith et al. 2019)
        r'\(\w+\s*\d{4}\)',            # (Smith 2020)
    ]
    
    # Check if section title indicates references
    citation_section_patterns = [
        'references', 'bibliography', 'citations', 'works cited', 'cited literature'
    ]
    
    # If text starts with a citation section title, it's likely a citation section
    first_line = text.strip().split('\n')[0].lower() if text.strip() else ""
    if any(pattern in first_line for pattern in citation_section_patterns):
        return True
    
    # Check if text contains many citation patterns
    citation_count = 0
    for pattern in citation_patterns:
        citation_count += len(re.findall(pattern, text))
    
    # If more than 30% of the lines contain citation patterns, consider it a citation section
    lines = text.split('\n')
    if not lines:
        return False
        
    lines_with_citations = sum(1 for line in lines if any(re.search(pattern, line) for pattern in citation_patterns))
    
    if lines_with_citations / len(lines) > 0.3:
        return True
    
    return False

def chunk_text(text: str, section_title: str) -> List[str]:
    """Chunk text using a hybrid approach with sliding window for large chunks."""
    if not text or len(text.strip()) == 0:
        return []
        
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Filter out empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Initialize chunks
    chunks = []
    current_chunk = ""
    
    # Process each paragraph
    for paragraph in paragraphs:
        # If adding this paragraph would exceed CHUNK_SIZE
        if len(current_chunk) + len(paragraph) > CHUNK_SIZE:
            # If current chunk is not empty, add it to chunks
            if current_chunk:
                chunks.append(current_chunk)
                
            # If paragraph itself exceeds CHUNK_SIZE, use sliding window
            if len(paragraph) > CHUNK_SIZE:
                # Use sliding window for this paragraph
                words = paragraph.split()
                for i in range(0, len(words), WINDOW_STRIDE):
                    window = ' '.join(words[i:i + WINDOW_SIZE])
                    if len(window) >= MIN_CHUNK_SIZE and not is_low_info_chunk(window):
                        chunks.append(window)
            else:
                # Start new chunk with this paragraph
                current_chunk = paragraph
        else:
            # Add paragraph to current chunk with a space if needed
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if not empty
    if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE and not is_low_info_chunk(current_chunk):
        chunks.append(current_chunk)
    
    # Filter out low information chunks
    filtered_chunks = [chunk for chunk in chunks if not is_low_info_chunk(chunk) and not is_citation_text(chunk)]
    
    return filtered_chunks

# Embedding generation
def load_embedding_model() -> Tuple[AutoTokenizer, AutoModel]:
    """Load the embedding model and tokenizer."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device)
    return tokenizer, model

def generate_embedding(text: str, tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
    """Generate embedding for a text chunk using the specified model."""
    # Prepare inputs
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    # Normalize
    embedding = embeddings[0]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

def create_faiss_index(dim: int = EMBEDDING_DIM) -> faiss.IndexFlatIP:
    """Create a new FAISS index."""
    return faiss.IndexFlatIP(dim)

def load_faiss_index() -> Tuple[faiss.IndexFlatIP, int]:
    """Load existing FAISS index or create a new one."""
    next_index = 0
    
    if os.path.exists(FAISS_INDEX):
        try:
            logger.info(f"Loading existing FAISS index from {FAISS_INDEX}")
            faiss_index = faiss.read_index(FAISS_INDEX)
            
            # Get the next available index
            with get_db_connection(CHUNK_DB) as conn:
                cursor = conn.execute('SELECT MAX(embedding_index) as max_index FROM chunks')
                row = cursor.fetchone()
                if row and row['max_index'] is not None:
                    next_index = row['max_index'] + 1
            
            logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors, next index: {next_index}")
            return faiss_index, next_index
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
    
    logger.info("Creating new FAISS index")
    return create_faiss_index(), 0

def load_metadata_faiss_index() -> Tuple[faiss.IndexFlatIP, int]:
    """Load existing metadata FAISS index or create a new one."""
    next_index = 0
    
    if os.path.exists(METADATA_FAISS_INDEX):
        try:
            logger.info(f"Loading existing metadata FAISS index from {METADATA_FAISS_INDEX}")
            faiss_index = faiss.read_index(METADATA_FAISS_INDEX)
            
            # Get the next available index
            with get_db_connection(CHUNK_DB) as conn:
                cursor = conn.execute('SELECT MAX(embedding_index) as max_index FROM metadata_embeddings')
                row = cursor.fetchone()
                if row and row['max_index'] is not None:
                    next_index = row['max_index'] + 1
            
            logger.info(f"Loaded metadata FAISS index with {faiss_index.ntotal} vectors, next index: {next_index}")
            return faiss_index, next_index
        except Exception as e:
            logger.error(f"Error loading metadata FAISS index: {e}")
    
    logger.info("Creating new metadata FAISS index")
    return create_faiss_index(), 0

def save_faiss_index(faiss_index: faiss.IndexFlatIP, is_metadata: bool = False) -> None:
    """Save FAISS index to disk."""
    index_path = METADATA_FAISS_INDEX if is_metadata else FAISS_INDEX
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    try:
        logger.info(f"Saving {'metadata ' if is_metadata else ''}FAISS index with {faiss_index.ntotal} vectors to {index_path}")
        faiss.write_index(faiss_index, index_path)
    except Exception as e:
        logger.error(f"Error saving {'metadata ' if is_metadata else ''}FAISS index: {e}")

# Paper processing
def process_paper(arxiv_id: str, conn_metadata: sqlite3.Connection, tokenizer: AutoTokenizer, 
                 model: AutoModel, faiss_index: faiss.IndexFlatIP, metadata_faiss_index: faiss.IndexFlatIP,
                 conn_chunks: sqlite3.Connection, next_index: int, metadata_next_index: int,
                 s2orc_manager: S2ORCManager) -> Tuple[bool, int, int, int]:
    """Process a paper: extract text, chunk, embed, and store."""
    # Get paper metadata
    title, abstract = get_paper_metadata(arxiv_id, conn_metadata)
    if not title or not abstract:
        logger.warning(f"Skipping paper {arxiv_id} due to missing metadata")
        return False, 0, next_index, metadata_next_index

    # Process metadata embedding first
    metadata_embedded = process_metadata_embedding(
        arxiv_id, title, abstract, tokenizer, model, 
        metadata_faiss_index, conn_chunks, metadata_next_index
    )
    
    if metadata_embedded:
        metadata_next_index += 1
        logger.info(f"Processed metadata embedding for {arxiv_id}")

    # Download paper content if needed
    available_sources = ensure_paper_content(arxiv_id)
    if not any(available_sources.values()):
        logger.warning(f"No content sources available for {arxiv_id}")

    # Try each content source in order of preference
    paragraphs = None
    section_titles = None
    source_used = None

    for source in CONTENT_SOURCES:
        if source == "s2orc" and available_sources.get("s2orc", False):
            paragraphs, section_titles = extract_text_from_s2orc(arxiv_id, s2orc_manager)
        elif source == "ar5iv" and available_sources.get("ar5iv", False):
            paragraphs, section_titles = extract_text_from_ar5iv(arxiv_id)
        elif source == "grobid" and available_sources.get("grobid", False):
            paragraphs, section_titles = extract_text_from_grobid(arxiv_id)
        elif source == "pdf" and available_sources.get("pdf", False):
            paragraphs, section_titles = extract_text_from_pdf(arxiv_id)

        if paragraphs and section_titles:
            logger.info(f"Successfully extracted text from {source} for paper {arxiv_id}")
            source_used = source
            break

    # If no content was found, use only metadata
    if not paragraphs or not section_titles:
        logger.warning(f"No full text found for paper {arxiv_id}, using metadata only")
        paragraphs = []
        section_titles = []

    # Always add metadata as a section
    section_titles.insert(0, "Metadata")
    metadata_text = f"Title: {title}\n\nAbstract: {abstract}"
    paragraphs.insert(0, metadata_text)

    # Process each section
    chunks_processed = 0
    embeddings_batch = []
    chunk_data_batch = []

    for i, (section_title, text) in enumerate(zip(section_titles, paragraphs)):
        if not text or len(text.strip()) == 0:
            continue

        # Chunk the text
        chunks = chunk_text(text, section_title)
        
        # Process each chunk
        for j, chunk in enumerate(chunks):
            # Generate chunk ID
            chunk_id = generate_chunk_id(arxiv_id, section_title, j)
            
            # Generate embedding
            embedding = generate_embedding(chunk, tokenizer, model)
            
            # Add to batch
            embeddings_batch.append(embedding)
            chunk_data_batch.append({
                'chunk_id': chunk_id,
                'arxiv_id': arxiv_id,
                'section_title': section_title,
                'chunk_index': j,
                'chunk_text': chunk,
                'embedding_index': next_index + len(embeddings_batch) - 1,
                'processed_at': datetime.now().isoformat()
            })
            
            chunks_processed += 1
            
            # Process in batches to save memory
            if len(embeddings_batch) >= BATCH_SIZE:
                # Add embeddings to FAISS index
                embeddings_array = np.array(embeddings_batch).astype('float32')
                faiss_index.add(embeddings_array)
                
                # Save chunk data to database
                with conn_chunks:
                    for chunk_data in chunk_data_batch:
                        conn_chunks.execute(
                            '''
                            INSERT OR IGNORE INTO chunks 
                            (chunk_id, arxiv_id, section_title, chunk_index, chunk_text, embedding_index, processed_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''',
                            (
                                chunk_data['chunk_id'],
                                chunk_data['arxiv_id'],
                                chunk_data['section_title'],
                                chunk_data['chunk_index'],
                                chunk_data['chunk_text'],
                                chunk_data['embedding_index'],
                                chunk_data['processed_at']
                            )
                        )
                
                # Clear batches
                next_index += len(embeddings_batch)
                embeddings_batch = []
                chunk_data_batch = []
                
                # Clear GPU memory
                clear_gpu_memory()

    # Process any remaining chunks
    if embeddings_batch:
        # Add embeddings to FAISS index
        embeddings_array = np.array(embeddings_batch).astype('float32')
        faiss_index.add(embeddings_array)
        
        # Save chunk data to database
        with conn_chunks:
            for chunk_data in chunk_data_batch:
                conn_chunks.execute(
                    '''
                    INSERT OR IGNORE INTO chunks 
                    (chunk_id, arxiv_id, section_title, chunk_index, chunk_text, embedding_index, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        chunk_data['chunk_id'],
                        chunk_data['arxiv_id'],
                        chunk_data['section_title'],
                        chunk_data['chunk_index'],
                        chunk_data['chunk_text'],
                        chunk_data['embedding_index'],
                        chunk_data['processed_at']
                    )
                )
        
        next_index += len(embeddings_batch)
    
    # Delete source files to save space
    delete_source_files(arxiv_id)
    
    # Clear GPU memory
    clear_gpu_memory()
    
    return True, chunks_processed, next_index, metadata_next_index

def process_metadata_embedding(arxiv_id: str, title: str, abstract: str, 
                              tokenizer: AutoTokenizer, model: AutoModel,
                              metadata_faiss_index: faiss.IndexFlatIP, 
                              conn_chunks: sqlite3.Connection, 
                              metadata_next_index: int) -> bool:
    """Process metadata embedding for a paper."""
    # Check if already processed
    with conn_chunks:
        cursor = conn_chunks.execute(
            'SELECT chunk_id FROM metadata_embeddings WHERE arxiv_id = ?',
            (arxiv_id,)
        )
        if cursor.fetchone():
            logger.info(f"Metadata for {arxiv_id} already embedded, skipping")
            return False
    
    # Get additional metadata from the database
    with get_db_connection(METADATA_DB) as conn_metadata:
        cursor = conn_metadata.execute(
            'SELECT title, authors, categories, published FROM papers WHERE arxiv_id = ?',
            (arxiv_id,)
        )
        paper_data = cursor.fetchone()
        
        if not paper_data:
            logger.warning(f"Paper {arxiv_id} not found in metadata database")
            return False
            
        db_title = paper_data['title']
        authors = paper_data['authors']
        categories = paper_data['categories']
        published = paper_data['published']
    
    # Use title from metadata database if not provided
    if not title:
        title = db_title
    
    # Combine all metadata fields into a single text
    metadata_text = f"Title: {title}\n\n"
    
    if abstract:
        metadata_text += f"Abstract: {abstract}\n\n"
    
    if authors:
        metadata_text += f"Authors: {authors}\n\n"
    
    if categories:
        metadata_text += f"Categories: {categories}\n\n"
    
    if published:
        metadata_text += f"Published: {published}\n\n"
    
    # Generate chunk ID
    chunk_id = f"metadata_{arxiv_id}"
    
    # Generate embedding
    embedding = generate_embedding(metadata_text, tokenizer, model)
    
    # Add to FAISS index
    metadata_faiss_index.add(np.array([embedding]).astype('float32'))
    
    # Save to database
    with conn_chunks:
        conn_chunks.execute(
            '''
            INSERT OR IGNORE INTO metadata_embeddings 
            (chunk_id, arxiv_id, metadata_text, embedding_index, processed_at)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (
                chunk_id,
                arxiv_id,
                metadata_text,
                metadata_next_index,
                datetime.now().isoformat()
            )
        )
    
    return True

# Main function
def run_content_embedding(papers_to_process: Optional[List[str]] = None) -> None:
    """Run the content embedding pipeline."""
    logger.info("Starting content embedding pipeline")
    
    # Initialize database
    initialize_chunk_db()
    
    # Get list of already processed papers
    processed_papers = get_processed_papers()
    processed_metadata_papers = get_processed_metadata_papers()
    logger.info(f"Found {len(processed_papers)} already processed papers")
    logger.info(f"Found {len(processed_metadata_papers)} papers with metadata embeddings")
    
    # Load embedding model
    tokenizer, model = load_embedding_model()
    
    # Load FAISS indices
    faiss_index, next_index = load_faiss_index()
    metadata_faiss_index, metadata_next_index = load_metadata_faiss_index()
    
    # Initialize S2ORC manager
    s2orc_manager = S2ORCManager(S2ORC_DIR)
    
    # Connect to databases
    with get_db_connection(METADATA_DB) as conn_metadata, get_db_connection(CHUNK_DB) as conn_chunks:
        # Get list of papers to process
        if papers_to_process:
            # Filter out already processed papers
            papers_to_process = [p for p in papers_to_process if p not in processed_papers]
            logger.info(f"Processing {len(papers_to_process)} specified papers")
        else:
            # Get all papers from metadata database
            cursor = conn_metadata.execute('SELECT arxiv_id FROM papers')
            all_papers = [row['arxiv_id'] for row in cursor.fetchall()]
            
            # Filter out already processed papers
            papers_to_process = [p for p in all_papers if p not in processed_papers]
            logger.info(f"Found {len(all_papers)} papers in metadata database, {len(papers_to_process)} to process")
        
        # Process papers in parallel
        total_papers = len(papers_to_process)
        total_chunks = 0
        
        # Create locks for thread-safe access to shared resources
        faiss_lock = Lock()
        metadata_faiss_lock = Lock()
        next_index_lock = Lock()
        metadata_next_index_lock = Lock()
        
        # Define worker function for parallel processing
        def process_paper_worker(i: int, arxiv_id: str) -> Tuple[bool, int, int, int]:
            logger.info(f"Processing paper {i+1}/{total_papers}: {arxiv_id}")
            
            try:
                # Create thread-local database connections
                with get_db_connection(METADATA_DB) as worker_conn_metadata, \
                     get_db_connection(CHUNK_DB) as worker_conn_chunks:
                    
                    # Get current index values
                    with next_index_lock:
                        current_next_index = next_index
                    with metadata_next_index_lock:
                        current_metadata_next_index = metadata_next_index
                    
                    # Process the paper
                    success, chunks, new_next_index, new_metadata_next_index = process_paper(
                        arxiv_id, worker_conn_metadata, tokenizer, model, 
                        faiss_index, metadata_faiss_index, worker_conn_chunks, 
                        current_next_index, current_metadata_next_index, s2orc_manager
                    )
                    
                    # Update index values
                    with next_index_lock:
                        nonlocal next_index
                        next_index = max(next_index, new_next_index)
                    with metadata_next_index_lock:
                        nonlocal metadata_next_index
                        metadata_next_index = max(metadata_next_index, new_metadata_next_index)
                    
                    return success, chunks, new_next_index, new_metadata_next_index
            except Exception as e:
                logger.error(f"Error processing paper {arxiv_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False, 0, current_next_index, current_metadata_next_index
        
        # Process papers in batches to control memory usage
        batch_size = 100  # Process papers in batches of 100
        for batch_start in range(0, len(papers_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(papers_to_process))
            batch = papers_to_process[batch_start:batch_end]
            batch_total = batch_end - batch_start
            
            logger.info(f"Processing batch of {batch_total} papers ({batch_start+1}-{batch_end} of {total_papers})")
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                future_to_paper = {
                    executor.submit(
                        process_paper_worker, 
                        batch_start + i, 
                        arxiv_id
                    ): (i, arxiv_id) for i, arxiv_id in enumerate(batch)
                }
                
                # Process completed tasks
                batch_chunks = 0
                for future in as_completed(future_to_paper):
                    i, arxiv_id = future_to_paper[future]
                    try:
                        success, chunks, _, _ = future.result()
                        if success:
                            batch_chunks += chunks
                            logger.info(f"Processed paper {batch_start+i+1}/{total_papers}: {arxiv_id} with {chunks} chunks")
                    except Exception as e:
                        logger.error(f"Exception processing paper {arxiv_id}: {e}")
                
                total_chunks += batch_chunks
                
                # Save indices after each batch
                with faiss_lock:
                    save_faiss_index(faiss_index)
                with metadata_faiss_lock:
                    save_faiss_index(metadata_faiss_index, is_metadata=True)
                
                logger.info(f"Batch complete: {batch_chunks} chunks processed, {total_chunks} total chunks")
                
                # Clear S2ORC cache between batches
                s2orc_manager.clear_cache()
                
                # Clear GPU memory
                clear_gpu_memory()
    
    # Save final FAISS indices
    save_faiss_index(faiss_index)
    save_faiss_index(metadata_faiss_index, is_metadata=True)
    
    logger.info(f"Content embedding pipeline completed. Processed {total_papers} papers with {total_chunks} chunks.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Content Embedding Pipeline")
    parser.add_argument("--arxiv-ids", nargs="+", help="List of arXiv IDs to process")
    
    args = parser.parse_args()
    
    run_content_embedding(args.arxiv_ids)
