#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalable Content Embedding Parameters

This file contains all parameters for the scalable content embedding module.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Database paths
METADATA_DB = os.path.join(PROJECT_ROOT, "data", "papers.db")
CHUNK_DB = os.path.join(PROJECT_ROOT, "data", "chunk_index.db")
FAISS_INDEX = os.path.join(PROJECT_ROOT, "data", "content_embeddings_scalable.faiss")
METADATA_FAISS_INDEX = os.path.join(PROJECT_ROOT, "data", "metadata_embeddings_scalable.faiss")

# Content source directories
S2ORC_DIR = os.path.join(PROJECT_ROOT, "data", "s2orc")
AR5IV_DIR = os.path.join(PROJECT_ROOT, "data", "ar5iv")
GROBID_DIR = os.path.join(PROJECT_ROOT, "data", "grobid")
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdf")

# URL constants
ARXIV_SOURCE_BASE_URL = "https://arxiv.org/e-print"
ARXIV_PDF_BASE_URL = "https://arxiv.org/pdf"
AR5IV_BASE_URL = "https://ar5iv.labs.arxiv.org/html"

# Embedding model parameters
EMBEDDING_MODEL = "intfloat/e5-small-v2"
EMBEDDING_DIM = 384  # Dimension of the embedding vectors

# Chunking parameters
CHUNK_SIZE = 2048  # Maximum characters per chunk
CHUNK_OVERLAP = 256  # Character overlap between chunks
MIN_CHUNK_SIZE = 128  # Minimum characters for a valid chunk

# Sliding window parameters (for large chunks)
WINDOW_SIZE = 512  # Words per window
WINDOW_STRIDE = 256  # Words to stride between windows

# Content source preference order
CONTENT_SOURCES = [
    "s2orc",  # S2ORC-like JSON created from LaTeX source
    "ar5iv",  # ar5iv HTML
    "grobid",  # GROBID TEI XML
    "pdf"     # PDF text extraction
]

# Processing parameters
BATCH_SIZE = 32  # Number of chunks to process in a batch
LOW_INFO_THRESHOLD = 0.4  # Threshold for low information content
MAX_WORKERS = 4  # Maximum number of worker threads
