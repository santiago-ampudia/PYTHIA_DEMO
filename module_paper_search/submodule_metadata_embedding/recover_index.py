#!/usr/bin/env python
"""
recover_index.py

A specialized script to recover the FAISS index from checkpoint backups
without needing to access the SQLite database.
"""

import os
import time
import logging
import numpy as np
import torch
import faiss
import argparse
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASES_DIR = os.path.join(MAIN_DIR, "databases")
BACKUP_DIR = os.path.join(DATABASES_DIR, "embedding_backups")
DEFAULT_FAISS_INDEX_PATH = os.path.join(DATABASES_DIR, "arxiv_embeddings.faiss")
DEFAULT_ARXIV_IDS_PATH = os.path.join(DATABASES_DIR, "arxiv_ids.npy")

# Set device (CPU, CUDA, or MPS)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

def find_latest_backup(backup_dir=BACKUP_DIR):
    """
    Find the latest backup file
    
    Args:
        backup_dir: Directory containing backups
    
    Returns:
        Path to the latest backup file, or None if not found
    """
    if not os.path.exists(backup_dir):
        logger.error(f"Backup directory {backup_dir} does not exist")
        return None
        
    # Look for the most recent backup
    backup_files = sorted(
        glob.glob(os.path.join(backup_dir, "embedding_checkpoint_backup_*.npz")),
        key=os.path.getmtime,
        reverse=True  # Most recent first
    )
    
    if backup_files:
        latest_backup = backup_files[0]
        logger.info(f"Found latest backup: {latest_backup}")
        return latest_backup
    
    logger.error("No backup files found")
    return None

def create_faiss_index(embeddings, chunk_size=1000):
    """
    Create a FAISS index from embeddings
    
    Args:
        embeddings: numpy array of embeddings
        chunk_size: number of vectors to process at once to avoid memory issues
        
    Returns:
        FAISS index
    """
    logger.info(f"Creating FAISS index with {len(embeddings)} vectors of dimension 768")
    
    # Create a flat index - exact search, no compression
    dimension = embeddings.shape[1]  # Should be 768 for SPECTER2
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Process in chunks to avoid memory issues
    num_chunks = (len(embeddings) + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(embeddings))
        
        logger.info(f"Processing chunk {i+1}/{num_chunks} (vectors {start_idx} to {end_idx})")
        
        # Get chunk of embeddings
        chunk_embeddings = embeddings[start_idx:end_idx]
        
        # Ensure embeddings are float32 and contiguous
        chunk_embeddings = np.ascontiguousarray(chunk_embeddings, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(chunk_embeddings)
        
        # Add vectors to the index
        index.add(chunk_embeddings)
        
        # Free memory
        del chunk_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
    logger.info(f"FAISS index created with {index.ntotal} vectors")
    return index

def save_faiss_index(index, arxiv_ids, index_path=DEFAULT_FAISS_INDEX_PATH, ids_path=DEFAULT_ARXIV_IDS_PATH):
    """
    Save FAISS index and corresponding arxiv_ids
    
    Args:
        index: FAISS index
        arxiv_ids: List of arxiv_ids
        index_path: Path to save the FAISS index
        ids_path: Path to save the arxiv_ids
    """
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(ids_path), exist_ok=True)
    
    logger.info(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)
    
    logger.info(f"Saving arxiv_ids to {ids_path}")
    np.save(ids_path, np.array(arxiv_ids))

def recover_from_backup(backup_path, faiss_index_path, arxiv_ids_path, chunk_size=1000):
    """
    Recover from a backup and create the FAISS index
    
    Args:
        backup_path: Path to the backup file
        faiss_index_path: Path to save the FAISS index
        arxiv_ids_path: Path to save the arxiv_ids
        chunk_size: Number of vectors to process at once
        
    Returns:
        tuple: (FAISS index, arxiv_ids)
    """
    logger.info(f"Recovering from backup at {backup_path}")
    
    try:
        # Load backup
        logger.info("Loading backup file (this may take a while for large backups)...")
        checkpoint = np.load(backup_path, allow_pickle=True)
        embeddings_list = list(checkpoint['embeddings'])
        arxiv_ids = list(checkpoint['arxiv_ids'])
        
        logger.info(f"Loaded backup with {len(arxiv_ids)} papers")
        
        # Stack embeddings
        logger.info("Stacking embeddings (this may take a while)...")
        try:
            # Process in batches to reduce memory usage
            all_embeddings = []
            batch_size = 100  # Process 100 batches at a time
            
            for i in range(0, len(embeddings_list), batch_size):
                batch = embeddings_list[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(embeddings_list) + batch_size - 1)//batch_size}")
                
                batch_embeddings = []
                for emb in batch:
                    if isinstance(emb, np.ndarray):
                        batch_embeddings.append(emb.astype(np.float32))
                    else:
                        batch_embeddings.append(emb)
                
                # Stack this batch
                stacked_batch = np.vstack(batch_embeddings)
                all_embeddings.append(stacked_batch)
                
                # Free memory
                del batch_embeddings, batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            # Combine all batches
            logger.info("Combining all batches...")
            all_embeddings = np.vstack(all_embeddings)
            
        except Exception as e:
            logger.error(f"Error stacking embeddings: {e}")
            raise
        
        # Free memory
        del embeddings_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Create FAISS index
        index = create_faiss_index(all_embeddings, chunk_size=chunk_size)
        
        # Free memory
        del all_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Save index and ids
        save_faiss_index(index, arxiv_ids, faiss_index_path, arxiv_ids_path)
        
        logger.info(f"Successfully recovered from backup and created index with {index.ntotal} vectors")
        
        return index, arxiv_ids
    except Exception as e:
        logger.error(f"Failed to recover from backup: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Recover FAISS index from checkpoint backup")
    parser.add_argument("--backup-path", type=str, default=None, 
                        help="Path to specific backup file (if not specified, will use latest)")
    parser.add_argument("--faiss-index-path", type=str, default=DEFAULT_FAISS_INDEX_PATH,
                        help=f"Path to save the FAISS index (default: {DEFAULT_FAISS_INDEX_PATH})")
    parser.add_argument("--arxiv-ids-path", type=str, default=DEFAULT_ARXIV_IDS_PATH,
                        help=f"Path to save the arxiv_ids (default: {DEFAULT_ARXIV_IDS_PATH})")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Number of vectors to process at once when creating FAISS index (default: 1000)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Find backup file
    backup_path = args.backup_path
    if not backup_path:
        backup_path = find_latest_backup()
        if not backup_path:
            logger.error("No backup file found. Please specify a backup file with --backup-path")
            return
    
    # Recover from backup
    try:
        recover_from_backup(
            backup_path,
            args.faiss_index_path,
            args.arxiv_ids_path,
            chunk_size=args.chunk_size
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Recovery completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Recovery failed: {e}")
        return

if __name__ == "__main__":
    main()
