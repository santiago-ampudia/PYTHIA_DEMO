"""
submodule_metadata_embedding/content_embedding_parameters.py

This file contains the parameters for the content embedding module.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASES_DIR = os.path.join(BASE_DIR, "databases")

# Database paths
ARXIV_METADATA_DB = os.path.join(DATABASES_DIR, "arxiv_metadata.db")
CHUNK_INDEX_DB = os.path.join(DATABASES_DIR, "chunk_index.db")
METADATA_INDEX_DB = os.path.join(DATABASES_DIR, "metadata_index.db")

# Embedding model
EMBEDDING_MODEL = "intfloat/e5-small-v2"
EMBEDDING_DIMENSION = 384  # e5-small-v2 dimension
EMBEDDING_MAX_LENGTH = 512  # Maximum tokens the model can handle at once

# FAISS index paths
FAISS_INDEX_PATH = os.path.join(DATABASES_DIR, "content_embeddings.faiss")
METADATA_FAISS_INDEX_PATH = os.path.join(DATABASES_DIR, "metadata_embeddings.faiss")

# Chunking parameters
CHUNK_SIZE = 2000  # Target number of tokens per chunk (will be processed with sliding window)
CHUNK_OVERLAP = 200  # Number of tokens to overlap between chunks (10% of chunk size)
MIN_CHUNK_SIZE = 50  # Minimum number of words for a valid chunk (increased for larger chunks)

# Sliding window parameters
WINDOW_SIZE = 512  # Size of sliding window (must be <= EMBEDDING_MAX_LENGTH)
WINDOW_STRIDE = 256  # Stride between windows (typically half the window size)

# Processing parameters
BATCH_SIZE = 8  # Number of chunks to embed at once (reduced due to larger chunks)
MAX_PAPERS_PER_BATCH = 3  # Number of papers to process in a single batch (reduced due to larger chunks)

# Content source directories
LATEX_DIR = os.path.join(BASE_DIR, "data", "latex")
AR5IV_DIR = os.path.join(BASE_DIR, "data", "ar5iv")

# URL constants
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_SOURCE_URL = "https://arxiv.org/e-print/{arxiv_id}"
AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

# Content sources (ar5iv is now the only source)
CONTENT_SOURCES = ["ar5iv"]

# Progress tracking
PROGRESS_TRACKER_PATH = os.path.join(DATABASES_DIR, "progress_tracker_embedding.json")

# Checkpoint and backup directories
CHECKPOINT_DIR = os.path.join(DATABASES_DIR, "embedding_checkpoints")
BACKUP_DIR = os.path.join(DATABASES_DIR, "embedding_backups")
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N batches (reduced due to larger batches)
MAX_BACKUPS = 5  # Maximum number of backup files to keep
