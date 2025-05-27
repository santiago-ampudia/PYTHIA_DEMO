"""
submodule_metadata_embedding/metadata_embedding.py

This module implements Step 1 of the paper search pipeline: embedding of metadata.

Purpose: Create searchable semantic vectors for all papers using metadata (title + abstract).
Tool: HuggingFace (SPECTER2) + FAISS
Goal: Precompute semantic embeddings for all metadata entries

Process:
    (title + summary) → SPECTER2 embed → 768-dim vector
    ↓
    Store in FAISS index keyed by arXiv ID

Output: FAISS index of N vectors
"""

import os
import sqlite3
import logging
import torch
import faiss
import numpy as np
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse
import shutil
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "databases", "arxiv_metadata.db")
DEFAULT_FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "databases", "arxiv_embeddings.faiss")
DEFAULT_ARXIV_IDS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "databases", "arxiv_ids.npy")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "databases", "embedding_checkpoints")
BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "databases", "embedding_backups")
BATCH_SIZE = 64
CHECKPOINT_INTERVAL = 10
MAX_BACKUPS = 5  # Maximum number of backup files to keep

# Set device (CPU, CUDA, or MPS)
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

def load_model_and_tokenizer():
    """
    Load the SPECTER2 model and tokenizer
    
    Returns:
        tuple: (tokenizer, model)
    """
    logger.info(f"Loading SPECTER2 model and tokenizer from allenai/specter2_base")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoModel.from_pretrained("allenai/specter2_base")
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model

def get_paper_metadata(db_path=DEFAULT_DB_PATH, category_filter=None):
    """
    Retrieve paper metadata from the SQLite database
    
    Args:
        db_path: Path to the SQLite database
        category_filter: Optional category code or list of category codes to filter papers by
        
    Returns:
        list: List of tuples (arxiv_id, title, summary)
    """
    logger.info(f"Retrieving paper metadata from {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total count for progress tracking
    if category_filter:
        # Convert single category to list for consistent handling
        if isinstance(category_filter, str):
            category_filter = [category_filter]
            
        # Build query with multiple LIKE conditions for each category
        like_conditions = []
        for category in category_filter:
            like_conditions.append(f"categories LIKE '%{category}%'")
        
        where_clause = " OR ".join(like_conditions)
        count_query = f"SELECT COUNT(*) FROM papers WHERE {where_clause}"
        cursor.execute(count_query)
        total_papers = cursor.fetchone()[0]
        
        categories_str = ", ".join(category_filter)
        logger.info(f"Found {total_papers} papers with categories '{categories_str}' in the database")
        
        # Fetch filtered papers
        papers_query = f"SELECT arxiv_id, title, summary FROM papers WHERE {where_clause}"
        cursor.execute(papers_query)
    else:
        cursor.execute("SELECT COUNT(*) FROM papers")
        total_papers = cursor.fetchone()[0]
        logger.info(f"Found {total_papers} papers in the database")
        
        # Fetch all papers
        cursor.execute("SELECT arxiv_id, title, summary FROM papers")
    
    papers = cursor.fetchall()
    conn.close()
    
    return papers

def create_backup(checkpoint_path, backup_dir=BACKUP_DIR):
    """
    Create a backup of the checkpoint file
    
    Args:
        checkpoint_path: Path to the checkpoint file
        backup_dir: Directory to store backups
    
    Returns:
        Path to the backup file
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Cannot create backup: Checkpoint file {checkpoint_path} does not exist")
        return None
        
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create a timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"embedding_checkpoint_backup_{timestamp}.npz"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # Copy the checkpoint file to the backup location
    try:
        shutil.copy2(checkpoint_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
        
        # Clean up old backups if there are too many
        cleanup_old_backups(backup_dir)
        
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None

def cleanup_old_backups(backup_dir=BACKUP_DIR, max_backups=MAX_BACKUPS):
    """
    Remove old backup files if there are too many
    
    Args:
        backup_dir: Directory containing backups
        max_backups: Maximum number of backups to keep
    """
    if not os.path.exists(backup_dir):
        return
        
    # List all backup files sorted by modification time (oldest first)
    backup_files = sorted(
        glob.glob(os.path.join(backup_dir, "embedding_checkpoint_backup_*.npz")),
        key=os.path.getmtime
    )
    
    # Remove oldest backups if there are too many
    if len(backup_files) > max_backups:
        for old_backup in backup_files[:-max_backups]:
            try:
                os.remove(old_backup)
                logger.info(f"Removed old backup: {old_backup}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {old_backup}: {e}")

def find_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR, backup_dir=BACKUP_DIR):
    """
    Find the latest checkpoint file, checking both the checkpoint directory and backup directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        backup_dir: Directory containing backups
    
    Returns:
        Path to the latest checkpoint file, or None if not found
    """
    checkpoint_path = os.path.join(checkpoint_dir, "embedding_checkpoint.npz")
    
    # First check if the main checkpoint exists
    if os.path.exists(checkpoint_path):
        return checkpoint_path
        
    # If not, look for the most recent backup
    if os.path.exists(backup_dir):
        backup_files = sorted(
            glob.glob(os.path.join(backup_dir, "embedding_checkpoint_backup_*.npz")),
            key=os.path.getmtime,
            reverse=True  # Most recent first
        )
        
        if backup_files:
            latest_backup = backup_files[0]
            logger.info(f"Found backup checkpoint: {latest_backup}")
            
            # Restore the backup to the main checkpoint location
            os.makedirs(checkpoint_dir, exist_ok=True)
            try:
                shutil.copy2(latest_backup, checkpoint_path)
                logger.info(f"Restored checkpoint from backup: {latest_backup} -> {checkpoint_path}")
                return checkpoint_path
            except Exception as e:
                logger.error(f"Failed to restore checkpoint from backup: {e}")
                return latest_backup  # Return the backup path if restore failed
    
    return None

def generate_embeddings(papers, tokenizer, model, batch_size=BATCH_SIZE, checkpoint_interval=CHECKPOINT_INTERVAL, checkpoint_dir=CHECKPOINT_DIR, resume=True):
    """
    Generate embeddings for paper metadata using SPECTER2
    
    Args:
        papers: List of tuples (arxiv_id, title, summary)
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        batch_size: Batch size for processing
        checkpoint_interval: Save checkpoint every N batches
        checkpoint_dir: Directory to save checkpoints
        resume: Whether to resume from checkpoint if available
        
    Returns:
        tuple: (numpy array of embeddings, list of arxiv_ids)
    """
    logger.info(f"Generating embeddings for {len(papers)} papers using batch size {batch_size}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "embedding_checkpoint.npz")
    
    # Check if checkpoint exists and if we should resume
    start_batch = 0
    embeddings = []
    arxiv_ids = []
    
    # Try to find the latest checkpoint (either main or backup)
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, BACKUP_DIR) if resume else None
    
    if latest_checkpoint and resume:
        try:
            logger.info(f"Found checkpoint at {latest_checkpoint}, attempting to resume...")
            checkpoint = np.load(latest_checkpoint, allow_pickle=True)
            embeddings = list(checkpoint['embeddings'])
            arxiv_ids = list(checkpoint['arxiv_ids'])
            start_batch = len(arxiv_ids) // batch_size
            
            # Adjust papers to skip already processed ones
            papers = papers[len(arxiv_ids):]
            
            logger.info(f"Resumed from checkpoint. Already processed {len(arxiv_ids)} papers. {len(papers)} papers remaining.")
            
            if len(papers) == 0:
                logger.info("All papers were already processed. Returning existing embeddings.")
                # Ensure embeddings are properly stacked and have the right dtype
                stacked_embeddings = np.vstack([emb.astype(np.float32) if isinstance(emb, np.ndarray) else emb for emb in embeddings])
                return stacked_embeddings, arxiv_ids
                
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
            embeddings = []
            arxiv_ids = []
    
    # Process in batches
    total_batches = (len(papers) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(papers), batch_size), initial=start_batch, total=total_batches):
        batch_papers = papers[i:i+batch_size]
        batch_ids = [paper[0] for paper in batch_papers]
        
        # Combine title and summary for each paper
        batch_texts = [f"{paper[1]} {paper[2]}" for paper in batch_papers]
        
        # Tokenize
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          return_tensors="pt", max_length=512).to(DEVICE)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Use CLS token embedding as the paper embedding
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings.append(batch_embeddings)
        arxiv_ids.extend(batch_ids)
        
        # Free up memory
        del inputs, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.backends.mps.is_available():
            # Clear MPS cache if available
            torch.mps.empty_cache()
            
        # Save checkpoint every N batches
        current_batch = start_batch + (i // batch_size)
        if checkpoint_interval > 0 and current_batch > 0 and current_batch % checkpoint_interval == 0:
            logger.info(f"Saving checkpoint after processing {len(arxiv_ids)} papers...")
            # Save individual batch embeddings as separate arrays to preserve their structure
            np.savez(checkpoint_path, 
                    embeddings=np.array(embeddings, dtype=object),
                    arxiv_ids=np.array(arxiv_ids))
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Create a backup of the checkpoint
            create_backup(checkpoint_path)
    
    # Combine all batches and ensure correct dtype
    try:
        all_embeddings = np.vstack([emb.astype(np.float32) if isinstance(emb, np.ndarray) else emb for emb in embeddings])
    except Exception as e:
        logger.error(f"Error stacking embeddings: {e}")
        # Fallback: try to process each batch individually
        processed_embeddings = []
        for emb in embeddings:
            if isinstance(emb, np.ndarray):
                processed_embeddings.append(emb.astype(np.float32))
            else:
                processed_embeddings.append(emb)
        all_embeddings = np.vstack(processed_embeddings)
    
    # Save a final checkpoint before returning
    # This ensures we have a checkpoint even if the FAISS creation fails
    np.savez(checkpoint_path, 
            embeddings=np.array(embeddings, dtype=object),
            arxiv_ids=np.array(arxiv_ids))
    logger.info(f"Final checkpoint saved to {checkpoint_path}")
    create_backup(checkpoint_path)
    
    # DO NOT remove the checkpoint file here - it will be removed only after
    # successful FAISS index creation and saving
    
    return all_embeddings, arxiv_ids

def create_faiss_index(embeddings, arxiv_ids, chunk_size=10000):
    """
    Create a FAISS index from embeddings
    
    Args:
        embeddings: numpy array of embeddings
        arxiv_ids: list of arxiv_ids
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

def recover_from_checkpoint(checkpoint_path, faiss_index_path, arxiv_ids_path, chunk_size=10000):
    """
    Recover from a checkpoint and create the FAISS index
    
    Args:
        checkpoint_path: Path to the checkpoint file
        faiss_index_path: Path to save the FAISS index
        arxiv_ids_path: Path to save the arxiv_ids
        chunk_size: Number of vectors to process at once
        
    Returns:
        tuple: (FAISS index, arxiv_ids)
    """
    logger.info(f"Recovering from checkpoint at {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        embeddings_list = list(checkpoint['embeddings'])
        arxiv_ids = list(checkpoint['arxiv_ids'])
        
        logger.info(f"Loaded checkpoint with {len(arxiv_ids)} papers")
        
        # Stack embeddings
        logger.info("Stacking embeddings...")
        try:
            all_embeddings = np.vstack([emb.astype(np.float32) if isinstance(emb, np.ndarray) else emb for emb in embeddings_list])
        except Exception as e:
            logger.error(f"Error stacking embeddings: {e}")
            # Fallback: try to process each batch individually
            processed_embeddings = []
            for emb in embeddings_list:
                if isinstance(emb, np.ndarray):
                    processed_embeddings.append(emb.astype(np.float32))
                else:
                    processed_embeddings.append(emb)
            all_embeddings = np.vstack(processed_embeddings)
        
        # Free memory
        del embeddings_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Create FAISS index
        index = create_faiss_index(all_embeddings, arxiv_ids, chunk_size=chunk_size)
        
        # Free memory
        del all_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Save index and ids
        save_faiss_index(index, arxiv_ids, faiss_index_path, arxiv_ids_path)
        
        logger.info(f"Successfully recovered from checkpoint and created index with {index.ntotal} vectors")
        
        return index, arxiv_ids
    except Exception as e:
        logger.error(f"Failed to recover from checkpoint: {e}")
        raise

def save_faiss_index(index, arxiv_ids, index_path=DEFAULT_FAISS_INDEX_PATH, ids_path=DEFAULT_ARXIV_IDS_PATH):
    """
    Save FAISS index and corresponding arxiv_ids
    
    Args:
        index: FAISS index
        arxiv_ids: List of arxiv_ids
        index_path: Path to save the FAISS index
        ids_path: Path to save the arxiv_ids
    """
    logger.info(f"Saving FAISS index to {index_path}")
    faiss.write_index(index, index_path)
    
    logger.info(f"Saving arxiv_ids to {ids_path}")
    np.save(ids_path, np.array(arxiv_ids))

def load_faiss_index(index_path=DEFAULT_FAISS_INDEX_PATH, ids_path=DEFAULT_ARXIV_IDS_PATH):
    """
    Load FAISS index and corresponding arxiv_ids
    
    Returns:
        tuple: (faiss.Index, list of arxiv_ids)
    """
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    
    logger.info(f"Loading arxiv_ids from {ids_path}")
    arxiv_ids = np.load(ids_path).tolist()
    
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
    return index, arxiv_ids

def run_metadata_embedding(db_path=DEFAULT_DB_PATH, 
                          faiss_index_path=DEFAULT_FAISS_INDEX_PATH, 
                          arxiv_ids_path=DEFAULT_ARXIV_IDS_PATH,
                          checkpoint_dir=CHECKPOINT_DIR,
                          category_filter=None,
                          batch_size=BATCH_SIZE,
                          checkpoint_interval=CHECKPOINT_INTERVAL,
                          incremental=True,
                          resume=True,
                          chunk_size=10000):
    """
    Run the metadata embedding pipeline
    
    Args:
        db_path: Path to the SQLite database
        faiss_index_path: Path to save the FAISS index
        arxiv_ids_path: Path to save the arxiv_ids
        checkpoint_dir: Directory to save checkpoints
        category_filter: List of categories to filter by
        batch_size: Batch size for processing
        checkpoint_interval: Save checkpoint every N batches
        incremental: Whether to add only new papers to an existing index
        resume: Whether to resume from checkpoint if available
        chunk_size: Number of vectors to process at once when creating FAISS index
    """
    start_time = time.time()
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    os.makedirs(os.path.dirname(arxiv_ids_path), exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()
    
    # Get paper metadata
    papers = get_paper_metadata(db_path, category_filter)
    logger.info(f"Retrieved metadata for {len(papers)} papers")
    
    # Check if we should update an existing index
    if os.path.exists(faiss_index_path) and os.path.exists(arxiv_ids_path) and incremental:
        try:
            # Load existing index and IDs
            index, existing_ids = load_faiss_index(faiss_index_path, arxiv_ids_path)
            
            # Filter out papers that are already in the index
            existing_ids_set = set(existing_ids)
            new_papers = [paper for paper in papers if paper[0] not in existing_ids_set]
            
            if not new_papers:
                logger.info("No new papers to add to the index")
                elapsed_time = time.time() - start_time
                logger.info(f"Metadata embedding completed in {elapsed_time:.2f} seconds")
                return index, existing_ids
                
            logger.info(f"Adding {len(new_papers)} new papers to existing index with {len(existing_ids)} papers")
            
            # Generate embeddings for new papers
            new_embeddings, new_ids = generate_embeddings(
                new_papers, tokenizer, model, 
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
                resume=resume
            )
            
            # Normalize new vectors for cosine similarity
            new_embeddings = np.ascontiguousarray(new_embeddings, dtype=np.float32)
            faiss.normalize_L2(new_embeddings)
            
            # Add new vectors to the index
            index.add(new_embeddings)
            
            # Combine IDs
            combined_ids = existing_ids + new_ids
            
            # Save updated index and IDs
            save_faiss_index(index, combined_ids, faiss_index_path, arxiv_ids_path)
            
            # Only remove checkpoint after successful save
            checkpoint_path = os.path.join(checkpoint_dir, "embedding_checkpoint.npz")
            if os.path.exists(checkpoint_path):
                try:
                    # Create one last backup before removing
                    create_backup(checkpoint_path)
                    os.remove(checkpoint_path)
                    logger.info(f"Removed checkpoint file after successful completion")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint file: {e}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Added {len(new_ids)} new papers to index in {elapsed_time:.2f} seconds")
            
            return index, combined_ids
        except Exception as e:
            logger.error(f"Error updating existing index: {e}")
            logger.info("Falling back to creating a new index")
            # Continue with creating a new index
    
    # Check if we have a checkpoint but failed at the FAISS creation step
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, BACKUP_DIR) if resume else None
    
    if latest_checkpoint and resume:
        try:
            # Try to load the checkpoint and create the FAISS index directly
            logger.info(f"Found checkpoint at {latest_checkpoint}, attempting to create FAISS index from it...")
            
            # Use the recovery function to create the index from the checkpoint
            index, arxiv_ids = recover_from_checkpoint(
                latest_checkpoint, 
                faiss_index_path, 
                arxiv_ids_path,
                chunk_size=chunk_size
            )
            
            # Only remove checkpoint after successful save
            checkpoint_path = os.path.join(checkpoint_dir, "embedding_checkpoint.npz")
            if os.path.exists(checkpoint_path):
                try:
                    # Create one last backup before removing
                    create_backup(checkpoint_path)
                    os.remove(checkpoint_path)
                    logger.info(f"Removed checkpoint file after successful completion")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint file: {e}")
                
            elapsed_time = time.time() - start_time
            logger.info(f"Created index from checkpoint in {elapsed_time:.2f} seconds")
            
            return index, arxiv_ids
        except Exception as e:
            logger.error(f"Error creating index from checkpoint: {e}")
            logger.info("Continuing with normal process")
    
    # Process all papers (either new index or full rebuild requested)
    # Generate embeddings
    embeddings, arxiv_ids = generate_embeddings(
        papers, tokenizer, model, 
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        resume=resume
    )
    
    # Create FAISS index
    index = create_faiss_index(embeddings, arxiv_ids, chunk_size=chunk_size)
    
    # Save index and ids
    save_faiss_index(index, arxiv_ids, faiss_index_path, arxiv_ids_path)
    
    # Only remove checkpoint after successful save
    checkpoint_path = os.path.join(checkpoint_dir, "embedding_checkpoint.npz")
    if os.path.exists(checkpoint_path):
        try:
            # Create one last backup before removing
            create_backup(checkpoint_path)
            os.remove(checkpoint_path)
            logger.info(f"Removed checkpoint file after successful completion")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint file: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Metadata embedding completed in {elapsed_time:.2f} seconds")
    
    return index, arxiv_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for arXiv metadata")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to the SQLite database")
    parser.add_argument("--faiss-index-path", default=DEFAULT_FAISS_INDEX_PATH, help="Path to save the FAISS index")
    parser.add_argument("--arxiv-ids-path", default=DEFAULT_ARXIV_IDS_PATH, help="Path to save the arxiv_ids")
    parser.add_argument("--category-filter", nargs="+", help="Category codes to filter by (e.g., hep-ex hep-ph)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL, help="Save checkpoint every N batches")
    parser.add_argument("--no-incremental", action="store_true", help="Don't update existing index, create new one")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Number of vectors to process at once when creating FAISS index")
    
    args = parser.parse_args()
    
    run_metadata_embedding(
        db_path=args.db_path,
        faiss_index_path=args.faiss_index_path,
        arxiv_ids_path=args.arxiv_ids_path,
        checkpoint_dir=CHECKPOINT_DIR,
        category_filter=args.category_filter,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        incremental=not args.no_incremental,
        resume=not args.no_resume,
        chunk_size=args.chunk_size
    )
