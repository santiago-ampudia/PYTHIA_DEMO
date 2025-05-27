"""
submodule_chunk_similarity_selection/chunk_similarity_selection.py

This module implements Step 5 of the paper search pipeline: Chunk Similarity Selection.

Purpose: Select the most relevant chunks by computing cosine similarity between query 
embeddings and both metadata and chunk embeddings. This step performs fine-grained 
selection of text chunks from papers that passed the category preselection filter.

Process:
    1. Embed the topic_query, subtopic_query, and enhanced_query
    2. For each preselected paper:
        a. Compute metadata similarity scores with each query
        b. If any score exceeds threshold, retrieve and score all chunks from that paper
    3. Sort chunks by similarity scores and select top k/m/n chunks

Output: Lists of top chunks by topic, subtopic, and enhanced query

Note: The E5-small-v2 model used in this module produces 384-dimensional vectors.
"""

import os
import sqlite3
import logging
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
import torch
import time
import datetime
from .chunk_similarity_selection_parameters import (
    DB_PATH, 
    CHUNK_INDEX_DB_PATH,
    METADATA_INDEX_DB_PATH,
    METADATA_FAISS_INDEX_PATH,
    CHUNK_FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    THRESHOLD_METADATA_SCORE_TOPIC,
    THRESHOLD_METADATA_SCORE_SUBTOPIC,
    THRESHOLD_METADATA_SCORE_ENHANCED,
    TOP_K_TOPIC,
    TOP_M_SUBTOPIC,
    TOP_N_ENHANCED,
    TOP_K_TOPIC_ANSWER,
    TOP_M_SUBTOPIC_ANSWER,
    TOP_N_ENHANCED_ANSWER,
    TOP_K_TOPIC_RECOMMENDATION,
    TOP_M_SUBTOPIC_RECOMMENDATION,
    TOP_N_ENHANCED_RECOMMENDATION,
    SIMILARITY_RESULTS_PATH,
    DETAILED_RESULTS_PATH,
    SELECTED_CHUNKS_DB_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('chunk_similarity_selection')


class EmbeddingModel:
    """Class to handle text embedding using the specified model."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")
        
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embeddings for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embeddings (normalized)
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding as the sentence embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Normalize embeddings to unit length
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norm
        
        return normalized_embeddings[0]  # Return the first (and only) embedding


class FaissIndexWrapper:
    """Wrapper class for FAISS index operations."""
    
    def __init__(self, index_path: str):
        """
        Initialize FAISS index wrapper.
        
        Args:
            index_path: Path to the FAISS index
        """
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
    
    def search(self, query_vector: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the closest vectors to the query vector.
        
        Args:
            query_vector: The query vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        # Ensure query vector is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # Ensure vector is normalized
        faiss.normalize_L2(query_vector)
        
        # Search the index
        distances, indices = self.index.search(query_vector, k)
        
        return distances, indices
    
    def get_vector(self, idx: int) -> np.ndarray:
        """
        Get the vector at the specified index.
        
        Args:
            idx: Index of the vector to retrieve
            
        Returns:
            The vector as a numpy array
        """
        return self.index.reconstruct(idx)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    # Ensure vectors are normalized
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    
    return float(np.dot(vec1_norm, vec2_norm))


def get_metadata_embedding_index(arxiv_id: str, db_path: str = METADATA_INDEX_DB_PATH) -> int:
    """
    Get the embedding index for a paper's metadata.
    
    Args:
        arxiv_id: arXiv ID of the paper
        db_path: Path to the metadata index database
        
    Returns:
        Embedding index in the FAISS index
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT embedding_index FROM metadata_embeddings WHERE arxiv_id = ?", (arxiv_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return result[0]
        else:
            raise ValueError(f"No metadata embedding found for paper {arxiv_id}")
            
    except sqlite3.Error as e:
        logger.error(f"Database error when retrieving metadata embedding for {arxiv_id}: {e}")
        raise ValueError(f"Database error: {e}")


def get_chunk_embedding_indices(arxiv_id: str, db_path: str = CHUNK_INDEX_DB_PATH) -> List[tuple]:
    """
    Get all chunk embedding indices and texts associated with a paper.
    
    Args:
        arxiv_id: arXiv ID of the paper
        db_path: Path to the chunk index database
        
    Returns:
        List of tuples (embedding_index, chunk_text)
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT embedding_index, chunk_text FROM chunks WHERE arxiv_id = ?", (arxiv_id,))
        results = cursor.fetchall()
        
        conn.close()
        
        if results:
            return results
        else:
            return []
            
    except sqlite3.Error as e:
        logger.error(f"Database error when retrieving chunks for {arxiv_id}: {e}")
        return []


def get_paper_metadata(arxiv_id: str, db_path: str = DB_PATH) -> Dict[str, str]:
    """
    Get metadata for a paper.
    
    Args:
        arxiv_id: arXiv ID of the paper
        db_path: Path to the metadata database
        
    Returns:
        Dictionary with paper metadata
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT title, summary, authors, categories, published FROM papers WHERE arxiv_id = ?", (arxiv_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            title, summary, authors, categories, published = result
            return {
                "title": title,
                "summary": summary,
                "authors": authors,
                "categories": categories,
                "published": published
            }
        else:
            return {}
            
    except sqlite3.Error as e:
        logger.error(f"Database error when retrieving metadata for {arxiv_id}: {e}")
        return {}


def save_detailed_results(
    all_chunk_results: List[Dict[str, Any]],
    topic_query: str,
    subtopic_query: str,
    enhanced_query: str,
    output_path: str = DETAILED_RESULTS_PATH
) -> None:
    """
    Save detailed chunk results to a text file.
    
    Args:
        all_chunk_results: All chunk results with similarity scores
        topic_query: High-level research theme query
        subtopic_query: Specific focus/method query
        enhanced_query: Optimized query for semantic search
        output_path: Path to save results
    """
    # Sort chunks by different similarity scores
    top_chunks_enhanced = sorted(
        all_chunk_results,
        key=lambda x: x["sim_enhanced_chunk"],
        reverse=True
    )[:5]  # Top 5 chunks for enhanced query
    
    top_chunks_topic = sorted(
        all_chunk_results,
        key=lambda x: x["sim_topic_chunk"],
        reverse=True
    )[:5]  # Top 5 chunks for topic query
    
    top_chunks_subtopic = sorted(
        all_chunk_results,
        key=lambda x: x["sim_subtopic_chunk"],
        reverse=True
    )[:5]  # Top 5 chunks for subtopic query
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Remove the existing file if it exists
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            logger.info(f"Removed existing detailed results file: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to remove existing detailed results file: {e}")
    
    # Create detailed output file
    with open(output_path, 'w') as f:
        # File header
        f.write("=" * 80 + "\n")
        f.write("DETAILED CHUNK SIMILARITY RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Enhanced query section (original)
        f.write("-" * 80 + "\n")
        f.write("QUERY ORIGINAL:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{enhanced_query}\n\n")
        
        for i, chunk in enumerate(top_chunks_enhanced):
            arxiv_id = chunk["arxiv_id"]
            paper_metadata = get_paper_metadata(arxiv_id)
            
            f.write("*" * 80 + "\n")
            f.write(f"CHUNK #{i+1} FOR ORIGINAL QUERY\n")
            f.write("*" * 80 + "\n\n")
            
            # Paper info
            f.write("PAPER INFORMATION:\n")
            f.write(f"Paper ID: {arxiv_id}\n")
            f.write(f"Paper Title: {paper_metadata.get('title', 'N/A')}\n\n")
            
            # Metadata similarity
            f.write("METADATA SIMILARITY SCORE:\n")
            f.write(f"Similarity between paper metadata and original query: {chunk['sim_enhanced_metadata']:.4f}\n\n")
            
            # Chunk text
            f.write("CHUNK TEXT:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{chunk['chunk_text']}\n")
            f.write("-" * 40 + "\n\n")
            
            # Similarity scores
            f.write("CHUNK SIMILARITY SCORES:\n")
            f.write(f"Similarity between chunk and original query: {chunk['sim_enhanced_chunk']:.4f}\n")
            f.write(f"Similarity between chunk and topic query: {chunk['sim_topic_chunk']:.4f}\n")
            f.write(f"Similarity between chunk and subtopic query: {chunk['sim_subtopic_chunk']:.4f}\n\n")
        
        # Topic query section
        f.write("=" * 80 + "\n")
        f.write("QUERY TOPIC:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{topic_query}\n\n")
        
        for i, chunk in enumerate(top_chunks_topic):
            arxiv_id = chunk["arxiv_id"]
            paper_metadata = get_paper_metadata(arxiv_id)
            
            f.write("*" * 80 + "\n")
            f.write(f"CHUNK #{i+1} FOR TOPIC QUERY\n")
            f.write("*" * 80 + "\n\n")
            
            # Paper info
            f.write("PAPER INFORMATION:\n")
            f.write(f"Paper ID: {arxiv_id}\n")
            f.write(f"Paper Title: {paper_metadata.get('title', 'N/A')}\n\n")
            
            # Metadata similarity
            f.write("METADATA SIMILARITY SCORE:\n")
            f.write(f"Similarity between paper metadata and topic query: {chunk['sim_topic_metadata']:.4f}\n\n")
            
            # Chunk text
            f.write("CHUNK TEXT:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{chunk['chunk_text']}\n")
            f.write("-" * 40 + "\n\n")
            
            # Similarity scores
            f.write("CHUNK SIMILARITY SCORES:\n")
            f.write(f"Similarity between chunk and original query: {chunk['sim_enhanced_chunk']:.4f}\n")
            f.write(f"Similarity between chunk and topic query: {chunk['sim_topic_chunk']:.4f}\n")
            f.write(f"Similarity between chunk and subtopic query: {chunk['sim_subtopic_chunk']:.4f}\n\n")
        
        # Subtopic query section
        f.write("=" * 80 + "\n")
        f.write("QUERY SUBTOPIC:\n")
        f.write("=" * 80 + "\n")
        f.write(f"{subtopic_query}\n\n")
        
        for i, chunk in enumerate(top_chunks_subtopic):
            arxiv_id = chunk["arxiv_id"]
            paper_metadata = get_paper_metadata(arxiv_id)
            
            f.write("*" * 80 + "\n")
            f.write(f"CHUNK #{i+1} FOR SUBTOPIC QUERY\n")
            f.write("*" * 80 + "\n\n")
            
            # Paper info
            f.write("PAPER INFORMATION:\n")
            f.write(f"Paper ID: {arxiv_id}\n")
            f.write(f"Paper Title: {paper_metadata.get('title', 'N/A')}\n\n")
            
            # Metadata similarity
            f.write("METADATA SIMILARITY SCORE:\n")
            f.write(f"Similarity between paper metadata and subtopic query: {chunk['sim_subtopic_metadata']:.4f}\n\n")
            
            # Chunk text
            f.write("CHUNK TEXT:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{chunk['chunk_text']}\n")
            f.write("-" * 40 + "\n\n")
            
            # Similarity scores
            f.write("CHUNK SIMILARITY SCORES:\n")
            f.write(f"Similarity between chunk and original query: {chunk['sim_enhanced_chunk']:.4f}\n")
            f.write(f"Similarity between chunk and topic query: {chunk['sim_topic_chunk']:.4f}\n")
            f.write(f"Similarity between chunk and subtopic query: {chunk['sim_subtopic_chunk']:.4f}\n\n")
    
    logger.info(f"Saved detailed chunk results to {output_path}")


def select_chunks_by_similarity(
    topic_query: str,
    subtopic_query: str,
    enhanced_query: str,
    preselected_papers: List[Dict[str, Any]],
    threshold_metadata_score_topic: float = THRESHOLD_METADATA_SCORE_TOPIC,
    threshold_metadata_score_subtopic: float = THRESHOLD_METADATA_SCORE_SUBTOPIC,
    threshold_metadata_score_enhanced: float = THRESHOLD_METADATA_SCORE_ENHANCED,
    top_k_topic: int = TOP_K_TOPIC,
    top_m_subtopic: int = TOP_M_SUBTOPIC,
    top_n_enhanced: int = TOP_N_ENHANCED,
    metadata_index_path: str = METADATA_FAISS_INDEX_PATH,
    chunk_index_path: str = CHUNK_FAISS_INDEX_PATH,
    chunk_db_path: str = CHUNK_INDEX_DB_PATH,
    metadata_db_path: str = METADATA_INDEX_DB_PATH,
    model_name: str = EMBEDDING_MODEL
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Select the most relevant chunks by computing similarity between queries and embeddings.
    
    Args:
        topic_query: High-level research theme query
        subtopic_query: Specific focus/method query
        enhanced_query: Optimized query for semantic search
        preselected_papers: List of papers preselected by category
        threshold_metadata_score_topic: Minimum metadata similarity score to consider chunks
        threshold_metadata_score_subtopic: Minimum metadata similarity score to consider chunks
        threshold_metadata_score_enhanced: Minimum metadata similarity score to consider chunks
        top_k_topic: Number of top chunks to select by topic
        top_m_subtopic: Number of top chunks to select by subtopic
        top_n_enhanced: Number of top chunks to select by enhanced query
        metadata_index_path: Path to metadata embedding FAISS index
        chunk_index_path: Path to chunk embedding FAISS index
        chunk_db_path: Path to chunk index database
        metadata_db_path: Path to metadata index database
        model_name: Name of the embedding model
        
    Returns:
        Tuple of (top_chunks_topic, top_chunks_subtopic, top_chunks_enhanced, all_chunk_results)
    """
    start_time = time.time()
    logger.info("Starting chunk similarity selection")
    logger.info(f"Processing {len(preselected_papers)} preselected papers")
    
    # Load embedding model
    embedding_model = EmbeddingModel(model_name)
    
    # Embed queries
    logger.info("Embedding queries")
    topic_vector = embedding_model.embed(topic_query)
    subtopic_vector = embedding_model.embed(subtopic_query)
    enhanced_vector = embedding_model.embed(enhanced_query)
    
    # Load FAISS indices
    logger.info("Loading FAISS indices")
    metadata_index = FaissIndexWrapper(metadata_index_path)
    chunk_index = FaissIndexWrapper(chunk_index_path)
    
    # Process each paper
    all_chunk_results = []
    papers_processed = 0
    chunks_processed = 0
    papers_above_threshold = 0
    
    for paper in preselected_papers:
        arxiv_id = paper["arxiv_id"]
        papers_processed += 1
        
        try:
            # Get metadata embedding index from database
            metadata_idx = get_metadata_embedding_index(arxiv_id, metadata_db_path)
            
            # Get metadata vector from FAISS index
            metadata_vector = metadata_index.get_vector(metadata_idx)
            
            # Compute metadata similarity scores
            sim_topic_metadata = cosine_similarity(topic_vector, metadata_vector)
            sim_subtopic_metadata = cosine_similarity(subtopic_vector, metadata_vector)
            sim_enhanced_metadata = cosine_similarity(enhanced_vector, metadata_vector)
            
            # Check if any score exceeds threshold
            if (sim_topic_metadata >= threshold_metadata_score_topic or 
                sim_subtopic_metadata >= threshold_metadata_score_subtopic or 
                sim_enhanced_metadata >= threshold_metadata_score_enhanced):
                
                papers_above_threshold += 1
                
                # Get chunk embedding indices for this paper
                chunk_indices = get_chunk_embedding_indices(arxiv_id, chunk_db_path)
                
                # Process each chunk
                for chunk_idx, chunk_text in chunk_indices:
                    chunks_processed += 1
                    
                    try:
                        # Get chunk vector
                        chunk_vector = chunk_index.get_vector(chunk_idx)
                        
                        # Compute chunk similarity scores
                        sim_topic_chunk = cosine_similarity(topic_vector, chunk_vector)
                        sim_subtopic_chunk = cosine_similarity(subtopic_vector, chunk_vector)
                        sim_enhanced_chunk = cosine_similarity(enhanced_vector, chunk_vector)
                        
                        # Store result
                        chunk_result = {
                            "chunk_idx": chunk_idx,
                            "arxiv_id": arxiv_id,
                            "chunk_text": chunk_text,
                            "sim_topic_metadata": sim_topic_metadata,
                            "sim_subtopic_metadata": sim_subtopic_metadata,
                            "sim_enhanced_metadata": sim_enhanced_metadata,
                            "sim_topic_chunk": sim_topic_chunk,
                            "sim_subtopic_chunk": sim_subtopic_chunk,
                            "sim_enhanced_chunk": sim_enhanced_chunk
                        }
                        all_chunk_results.append(chunk_result)
                        
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Error processing chunk index {chunk_idx}: {e}")
                        continue
                    
                    # Log progress periodically
                    if chunks_processed % 1000 == 0:
                        elapsed = time.time() - start_time
                        logger.info(f"Processed {chunks_processed} chunks in {elapsed:.2f} seconds")
                        
        except (KeyError, ValueError) as e:
            logger.warning(f"Error processing paper {arxiv_id}: {e}")
            continue
        
        # Log progress periodically
        if papers_processed % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Processed {papers_processed}/{len(preselected_papers)} papers in {elapsed:.2f} seconds")
    
    # Sort chunks by similarity scores
    logger.info(f"Sorting {len(all_chunk_results)} chunk results")
    
    # Sort by topic similarity
    top_chunks_topic = sorted(
        all_chunk_results, 
        key=lambda x: x["sim_topic_chunk"], 
        reverse=True
    )[:top_k_topic]
    
    # Sort by subtopic similarity
    top_chunks_subtopic = sorted(
        all_chunk_results, 
        key=lambda x: x["sim_subtopic_chunk"], 
        reverse=True
    )[:top_m_subtopic]
    
    # Sort by enhanced similarity
    top_chunks_enhanced = sorted(
        all_chunk_results, 
        key=lambda x: x["sim_enhanced_chunk"], 
        reverse=True
    )[:top_n_enhanced]
    
    # Log summary
    elapsed = time.time() - start_time
    logger.info(f"Chunk similarity selection completed in {elapsed:.2f} seconds")
    logger.info(f"Processed {papers_processed} papers, {chunks_processed} chunks")
    logger.info(f"Papers above threshold: {papers_above_threshold}")
    logger.info(f"Selected {len(top_chunks_topic)} topic chunks, {len(top_chunks_subtopic)} subtopic chunks, {len(top_chunks_enhanced)} enhanced chunks")
    
    return top_chunks_topic, top_chunks_subtopic, top_chunks_enhanced, all_chunk_results


def save_similarity_results(
    top_chunks_topic: List[Dict[str, Any]],
    top_chunks_subtopic: List[Dict[str, Any]],
    top_chunks_enhanced: List[Dict[str, Any]],
    output_path: str = SIMILARITY_RESULTS_PATH
) -> None:
    """
    Save similarity results to a JSON file.
    
    Args:
        top_chunks_topic: Top chunks by topic similarity
        top_chunks_subtopic: Top chunks by subtopic similarity
        top_chunks_enhanced: Top chunks by enhanced similarity
        output_path: Path to save results
    """
    results = {
        "top_chunks_topic": top_chunks_topic,
        "top_chunks_subtopic": top_chunks_subtopic,
        "top_chunks_enhanced": top_chunks_enhanced
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved similarity results to {output_path}")


def save_selected_chunks_to_db(
    top_chunks_topic: List[Dict[str, Any]],
    top_chunks_subtopic: List[Dict[str, Any]],
    top_chunks_enhanced: List[Dict[str, Any]],
    topic_query: str,
    subtopic_query: str,
    enhanced_query: str,
    output_path: str = SELECTED_CHUNKS_DB_PATH
) -> None:
    """
    Save selected chunks to a SQLite database, mapping each query to its top chunks.
    
    Args:
        top_chunks_topic: Top chunks by topic similarity
        top_chunks_subtopic: Top chunks by subtopic similarity
        top_chunks_enhanced: Top chunks by enhanced similarity
        topic_query: High-level research theme query
        subtopic_query: Specific focus/method query
        enhanced_query: Optimized query for semantic search
        output_path: Path to save the database
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Remove existing database if it exists
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            logger.info(f"Removed existing selected chunks database: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to remove existing database: {e}")
    
    # Create or connect to the database
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_text TEXT NOT NULL,
        query_type TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS selected_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id INTEGER NOT NULL,
        chunk_idx INTEGER NOT NULL,
        arxiv_id TEXT NOT NULL,
        chunk_text TEXT NOT NULL,
        sim_enhanced_chunk REAL NOT NULL,
        sim_topic_chunk REAL NOT NULL,
        sim_subtopic_chunk REAL NOT NULL,
        sim_enhanced_metadata REAL NOT NULL,
        sim_topic_metadata REAL NOT NULL,
        sim_subtopic_metadata REAL NOT NULL,
        FOREIGN KEY (query_id) REFERENCES queries (id),
        UNIQUE (query_id, chunk_idx)
    )
    ''')
    
    # Get current timestamp
    timestamp = datetime.datetime.now().isoformat()
    
    # Insert queries
    cursor.execute(
        "INSERT INTO queries (query_text, query_type, timestamp) VALUES (?, ?, ?)",
        (topic_query, "topic", timestamp)
    )
    topic_query_id = cursor.lastrowid
    
    cursor.execute(
        "INSERT INTO queries (query_text, query_type, timestamp) VALUES (?, ?, ?)",
        (subtopic_query, "subtopic", timestamp)
    )
    subtopic_query_id = cursor.lastrowid
    
    cursor.execute(
        "INSERT INTO queries (query_text, query_type, timestamp) VALUES (?, ?, ?)",
        (enhanced_query, "enhanced", timestamp)
    )
    enhanced_query_id = cursor.lastrowid
    
    # Insert selected chunks for topic query
    for chunk in top_chunks_topic:
        cursor.execute(
            """
            INSERT OR REPLACE INTO selected_chunks (
                query_id, chunk_idx, arxiv_id, chunk_text, 
                sim_enhanced_chunk, sim_topic_chunk, sim_subtopic_chunk,
                sim_enhanced_metadata, sim_topic_metadata, sim_subtopic_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_query_id, 
                chunk["chunk_idx"], 
                chunk["arxiv_id"],
                chunk["chunk_text"],
                chunk["sim_enhanced_chunk"],
                chunk["sim_topic_chunk"],
                chunk["sim_subtopic_chunk"],
                chunk["sim_enhanced_metadata"],
                chunk["sim_topic_metadata"],
                chunk["sim_subtopic_metadata"]
            )
        )
    
    # Insert selected chunks for subtopic query
    for chunk in top_chunks_subtopic:
        cursor.execute(
            """
            INSERT OR REPLACE INTO selected_chunks (
                query_id, chunk_idx, arxiv_id, chunk_text, 
                sim_enhanced_chunk, sim_topic_chunk, sim_subtopic_chunk,
                sim_enhanced_metadata, sim_topic_metadata, sim_subtopic_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subtopic_query_id, 
                chunk["chunk_idx"], 
                chunk["arxiv_id"],
                chunk["chunk_text"],
                chunk["sim_enhanced_chunk"],
                chunk["sim_topic_chunk"],
                chunk["sim_subtopic_chunk"],
                chunk["sim_enhanced_metadata"],
                chunk["sim_topic_metadata"],
                chunk["sim_subtopic_metadata"]
            )
        )
    
    # Insert selected chunks for enhanced query
    for chunk in top_chunks_enhanced:
        cursor.execute(
            """
            INSERT OR REPLACE INTO selected_chunks (
                query_id, chunk_idx, arxiv_id, chunk_text, 
                sim_enhanced_chunk, sim_topic_chunk, sim_subtopic_chunk,
                sim_enhanced_metadata, sim_topic_metadata, sim_subtopic_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                enhanced_query_id, 
                chunk["chunk_idx"], 
                chunk["arxiv_id"],
                chunk["chunk_text"],
                chunk["sim_enhanced_chunk"],
                chunk["sim_topic_chunk"],
                chunk["sim_subtopic_chunk"],
                chunk["sim_enhanced_metadata"],
                chunk["sim_topic_metadata"],
                chunk["sim_subtopic_metadata"]
            )
        )
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    logger.info(f"Saved selected chunks to database: {output_path}")


def run_chunk_similarity_selection(
    topic_query: str,
    subtopic_query: str,
    enhanced_query: str,
    preselected_papers: List[Dict[str, Any]],
    search_mode: str = "answer"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run the chunk similarity selection process.
    
    Args:
        topic_query: High-level research theme query
        subtopic_query: Specific focus/method query
        enhanced_query: Optimized query for semantic search
        preselected_papers: List of papers preselected by category
        search_mode: Search mode, either "answer" or "recommendation"
        
    Returns:
        Tuple of (top_chunks_topic, top_chunks_subtopic, top_chunks_enhanced, all_chunk_results)
    """
    logger.info("Running chunk similarity selection")
    logger.info(f"Topic query: {topic_query}")
    logger.info(f"Subtopic query: {subtopic_query}")
    logger.info(f"Enhanced query: {enhanced_query}")
    logger.info(f"Number of preselected papers: {len(preselected_papers)}")
    logger.info(f"Search mode: {search_mode}")
    
    # Set the top-k parameters based on the search mode
    if search_mode == "recommendation":
        top_k_topic = TOP_K_TOPIC_RECOMMENDATION
        top_m_subtopic = TOP_M_SUBTOPIC_RECOMMENDATION
        top_n_enhanced = TOP_N_ENHANCED_RECOMMENDATION
        logger.info(f"Using recommendation mode parameters: top_k_topic={top_k_topic}, top_m_subtopic={top_m_subtopic}, top_n_enhanced={top_n_enhanced}")
    else:  # Default to "answer" mode
        top_k_topic = TOP_K_TOPIC_ANSWER
        top_m_subtopic = TOP_M_SUBTOPIC_ANSWER
        top_n_enhanced = TOP_N_ENHANCED_ANSWER
        logger.info(f"Using answer mode parameters: top_k_topic={top_k_topic}, top_m_subtopic={top_m_subtopic}, top_n_enhanced={top_n_enhanced}")
    
    # Select chunks by similarity
    top_chunks_topic, top_chunks_subtopic, top_chunks_enhanced, all_chunk_results = select_chunks_by_similarity(
        topic_query=topic_query,
        subtopic_query=subtopic_query,
        enhanced_query=enhanced_query,
        preselected_papers=preselected_papers,
        top_k_topic=top_k_topic,
        top_m_subtopic=top_m_subtopic,
        top_n_enhanced=top_n_enhanced,
        metadata_index_path=METADATA_FAISS_INDEX_PATH,
        chunk_index_path=CHUNK_FAISS_INDEX_PATH,
        chunk_db_path=CHUNK_INDEX_DB_PATH,
        metadata_db_path=METADATA_INDEX_DB_PATH
    )
    
    # Save similarity results to JSON
    save_similarity_results(
        top_chunks_topic=top_chunks_topic,
        top_chunks_subtopic=top_chunks_subtopic,
        top_chunks_enhanced=top_chunks_enhanced
    )
    
    # Save detailed results to text file
    save_detailed_results(
        all_chunk_results=all_chunk_results,
        topic_query=topic_query,
        subtopic_query=subtopic_query,
        enhanced_query=enhanced_query
    )
    
    # Save selected chunks to database
    save_selected_chunks_to_db(
        top_chunks_topic=top_chunks_topic,
        top_chunks_subtopic=top_chunks_subtopic,
        top_chunks_enhanced=top_chunks_enhanced,
        topic_query=topic_query,
        subtopic_query=subtopic_query,
        enhanced_query=enhanced_query
    )
    
    return top_chunks_topic, top_chunks_subtopic, top_chunks_enhanced, all_chunk_results
