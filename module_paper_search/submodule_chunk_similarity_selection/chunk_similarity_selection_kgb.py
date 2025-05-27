"""
submodule_chunk_similarity_selection/chunk_similarity_selection_kgb.py

This module implements an enhanced version of Step 5 of the paper search pipeline: Chunk Similarity Selection.
KGB (Keyword-Guided Batch) version supports processing multiple queries in a batch.

Purpose: Select the most relevant chunks by computing cosine similarity between query 
embeddings and both metadata and chunk embeddings. This step performs fine-grained 
selection of text chunks from papers that passed the category preselection filter.

Process:
    1. Embed all queries in the provided queries_list
    2. For each preselected paper:
        a. Compute metadata similarity scores with each query
        b. If any score exceeds threshold, retrieve and score all chunks from that paper
    3. Sort chunks by similarity scores and select top k chunks for each query

Output: List of dictionaries containing top chunks for each query

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
from .chunk_similarity_selection_kgb_parameters import (
    DB_PATH, 
    CHUNK_INDEX_DB_PATH,
    METADATA_INDEX_DB_PATH,
    METADATA_FAISS_INDEX_PATH,
    CHUNK_FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    THRESHOLD_METADATA_SCORE,
    TOP_K_CHUNKS,
    TOP_K_CHUNKS_ANSWER,
    TOP_K_CHUNKS_RECOMMENDATION,
    SIMILARITY_RESULTS_PATH,
    DETAILED_RESULTS_PATH,
    SELECTED_CHUNKS_DB_PATH,
    QUERY_TYPE_NAMES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('chunk_similarity_selection_kgb')


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


def get_chunk_embedding_indices(arxiv_id: str, db_path: str = CHUNK_INDEX_DB_PATH) -> List[Tuple[int, str]]:
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
        
        cursor.execute(
            "SELECT embedding_index, chunk_text FROM chunks WHERE arxiv_id = ?", 
            (arxiv_id,)
        )
        results = cursor.fetchall()
        
        conn.close()
        
        if results:
            return results
        else:
            logger.warning(f"No chunk embeddings found for paper {arxiv_id}")
            return []
            
    except sqlite3.Error as e:
        logger.error(f"Database error when retrieving chunk embeddings for {arxiv_id}: {e}")
        return []


def get_paper_metadata(arxiv_id: str, db_path: str = DB_PATH) -> Dict[str, Any]:
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
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", 
            (arxiv_id,)
        )
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return dict(result)
        else:
            logger.warning(f"No metadata found for paper {arxiv_id}")
            return {}
            
    except sqlite3.Error as e:
        logger.error(f"Database error when retrieving metadata for {arxiv_id}: {e}")
        return {}


def save_detailed_results(
    all_chunk_results: List[Dict[str, Any]],
    queries_list: List[str],
    output_path: str = DETAILED_RESULTS_PATH
) -> None:
    """
    Save detailed chunk results to a text file.
    
    Args:
        all_chunk_results: All chunk results with similarity scores
        queries_list: List of queries used for similarity search
        output_path: Path to save results
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sort chunks by highest similarity score across all queries
    sorted_results = sorted(
        all_chunk_results,
        key=lambda x: max([x[f"sim_query_{i}_chunk"] for i in range(len(queries_list))]),
        reverse=True
    )
    
    # Write results to file
    with open(output_path, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("DETAILED CHUNK SIMILARITY RESULTS\n")
        f.write(f"Generated on: {datetime.datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write queries
        f.write("QUERIES:\n")
        for i, query in enumerate(queries_list):
            query_type = QUERY_TYPE_NAMES[i] if i < len(QUERY_TYPE_NAMES) else f"query_{i}"
            f.write(f"{query_type}: {query}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Write chunk results
        f.write(f"TOP CHUNKS (sorted by highest similarity score):\n\n")
        
        for i, chunk in enumerate(sorted_results[:100]):  # Limit to top 100 for readability
            # Get paper metadata
            paper_metadata = get_paper_metadata(chunk["arxiv_id"])
            
            # Write chunk header
            f.write(f"CHUNK #{i+1}\n")
            f.write(f"arXiv ID: {chunk['arxiv_id']}\n")
            
            if paper_metadata:
                f.write(f"Title: {paper_metadata.get('title', 'N/A')}\n")
                f.write(f"Authors: {paper_metadata.get('authors', 'N/A')}\n")
                f.write(f"Categories: {paper_metadata.get('categories', 'N/A')}\n")
                f.write(f"Abstract: {paper_metadata.get('abstract', 'N/A')[:300]}...\n")
            
            # Write similarity scores for each query
            f.write("Similarity Scores:\n")
            for j in range(len(queries_list)):
                query_type = QUERY_TYPE_NAMES[j] if j < len(QUERY_TYPE_NAMES) else f"query_{j}"
                f.write(f"  {query_type} metadata: {chunk[f'sim_query_{j}_metadata']:.4f}\n")
                f.write(f"  {query_type} chunk: {chunk[f'sim_query_{j}_chunk']:.4f}\n")
            
            # Write chunk text
            f.write("\nChunk Text:\n")
            f.write(f"{chunk['chunk_text']}\n")
            
            f.write("-" * 80 + "\n\n")
    
    logger.info(f"Saved detailed results to {output_path}")


def select_chunks_by_similarity(
    queries_list: List[str],
    preselected_papers: List[Dict[str, Any]],
    threshold_metadata_score: float = THRESHOLD_METADATA_SCORE,
    top_k_chunks: int = TOP_K_CHUNKS,
    metadata_index_path: str = METADATA_FAISS_INDEX_PATH,
    chunk_index_path: str = CHUNK_FAISS_INDEX_PATH,
    chunk_db_path: str = CHUNK_INDEX_DB_PATH,
    metadata_db_path: str = METADATA_INDEX_DB_PATH,
    model_name: str = EMBEDDING_MODEL
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Select the most relevant chunks by computing similarity between queries and embeddings.
    
    Args:
        queries_list: List of queries to use for similarity search
        preselected_papers: List of papers preselected by category
        threshold_metadata_score: Minimum metadata similarity score to consider chunks
        top_k_chunks: Number of top chunks to select for each query
        metadata_index_path: Path to metadata embedding FAISS index
        chunk_index_path: Path to chunk embedding FAISS index
        chunk_db_path: Path to chunk index database
        metadata_db_path: Path to metadata index database
        model_name: Name of the embedding model
        
    Returns:
        Tuple of (list_of_top_chunks_per_query, all_chunk_results)
    """
    start_time = time.time()
    logger.info("Starting KGB chunk similarity selection")
    logger.info(f"Processing {len(preselected_papers)} preselected papers with {len(queries_list)} queries")
    
    # Load embedding model
    embedding_model = EmbeddingModel(model_name)
    
    # Embed all queries
    logger.info("Embedding queries")
    query_vectors = []
    for i, query in enumerate(queries_list):
        query_type = QUERY_TYPE_NAMES[i] if i < len(QUERY_TYPE_NAMES) else f"query_{i}"
        logger.info(f"Embedding {query_type} query: {query}")
        query_vectors.append(embedding_model.embed(query))
    
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
            
            # Compute metadata similarity scores for each query
            metadata_scores = []
            above_threshold = False
            
            for query_vector in query_vectors:
                sim_metadata = cosine_similarity(query_vector, metadata_vector)
                metadata_scores.append(sim_metadata)
                if sim_metadata >= threshold_metadata_score:
                    above_threshold = True
            
            # Check if any score exceeds threshold
            if above_threshold:
                papers_above_threshold += 1
                
                # Get chunk embedding indices for this paper
                chunk_indices = get_chunk_embedding_indices(arxiv_id, chunk_db_path)
                
                # Process each chunk
                for chunk_idx, chunk_text in chunk_indices:
                    chunks_processed += 1
                    
                    try:
                        # Get chunk vector
                        chunk_vector = chunk_index.get_vector(chunk_idx)
                        
                        # Compute chunk similarity scores for each query
                        chunk_scores = []
                        for query_vector in query_vectors:
                            sim_chunk = cosine_similarity(query_vector, chunk_vector)
                            chunk_scores.append(sim_chunk)
                        
                        # Store result with all similarity scores
                        chunk_result = {
                            "chunk_idx": chunk_idx,
                            "arxiv_id": arxiv_id,
                            "chunk_text": chunk_text
                        }
                        
                        # Add metadata and chunk similarity scores for each query
                        for i in range(len(queries_list)):
                            chunk_result[f"sim_query_{i}_metadata"] = metadata_scores[i]
                            chunk_result[f"sim_query_{i}_chunk"] = chunk_scores[i]
                        
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
    
    # Sort chunks by similarity scores for each query
    logger.info(f"Sorting {len(all_chunk_results)} chunk results")
    
    # Create a list to hold top chunks for each query
    top_chunks_per_query = []
    
    # For each query, sort chunks by similarity and select top k
    for i in range(len(queries_list)):
        query_type = QUERY_TYPE_NAMES[i] if i < len(QUERY_TYPE_NAMES) else f"query_{i}"
        logger.info(f"Selecting top {top_k_chunks} chunks for {query_type} query")
        
        top_chunks = sorted(
            all_chunk_results, 
            key=lambda x: x[f"sim_query_{i}_chunk"], 
            reverse=True
        )[:top_k_chunks]
        
        top_chunks_per_query.append(top_chunks)
    
    # Log summary
    elapsed = time.time() - start_time
    logger.info(f"KGB chunk similarity selection completed in {elapsed:.2f} seconds")
    logger.info(f"Processed {papers_processed} papers, {chunks_processed} chunks")
    logger.info(f"Papers above threshold: {papers_above_threshold}")
    
    for i in range(len(queries_list)):
        query_type = QUERY_TYPE_NAMES[i] if i < len(QUERY_TYPE_NAMES) else f"query_{i}"
        logger.info(f"Selected {len(top_chunks_per_query[i])} chunks for {query_type} query")
    
    return top_chunks_per_query, all_chunk_results


def save_chunks_with_scores_json(
    top_chunks_per_query: List[List[Dict[str, Any]]],
    queries_list: List[str],
    output_path: str = os.path.join(os.path.dirname(SIMILARITY_RESULTS_PATH), 'chunks_with_scores_kgb.json')
) -> None:
    """
    Save chunks with their scores for all queries to a JSON file.
    
    Args:
        top_chunks_per_query: List of top chunks for each query
        queries_list: List of queries used for similarity search
        output_path: Path to save the JSON file
    """
    logger.info(f"Saving chunks with scores to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a dictionary to store chunks by query type
    result = {
        "queries": {},
        "selected_chunks": {}
    }
    
    # Add queries for reference
    for i, query in enumerate(queries_list):
        query_type = QUERY_TYPE_NAMES[i] if i < len(QUERY_TYPE_NAMES) else f"query_{i}"
        result["queries"][query_type] = query
        result["selected_chunks"][query_type] = []
    
    # Helper function to extract chunk info with scores
    def extract_chunk_info(chunk, query_index):
        chunk_info = {
            "chunk_idx": chunk.get("chunk_idx", ""),
            "arxiv_id": chunk.get("arxiv_id", ""),
            "chunk_text": chunk.get("chunk_text", ""),
            "scores": {}
        }
        
        # Add scores for all queries
        for j in range(len(queries_list)):
            query_type = QUERY_TYPE_NAMES[j] if j < len(QUERY_TYPE_NAMES) else f"query_{j}"
            chunk_info["scores"][f"{query_type}_score"] = chunk.get(f"sim_query_{j}_chunk", 0)
        
        return chunk_info
    
    # Add chunks for each query type
    for i in range(len(queries_list)):
        query_type = QUERY_TYPE_NAMES[i] if i < len(QUERY_TYPE_NAMES) else f"query_{i}"
        for chunk in top_chunks_per_query[i]:
            result["selected_chunks"][query_type].append(extract_chunk_info(chunk, i))
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Saved chunks with scores to {output_path}")
    
    # Log summary
    for i in range(len(queries_list)):
        query_type = QUERY_TYPE_NAMES[i] if i < len(QUERY_TYPE_NAMES) else f"query_{i}"
        logger.info(f"{query_type} query: {len(result['selected_chunks'][query_type])} chunks")


def save_similarity_results(
    top_chunks_per_query: List[List[Dict[str, Any]]],
    queries_list: List[str],
    output_path: str = SIMILARITY_RESULTS_PATH
) -> None:
    """
    Save similarity results to a JSON file.
    
    Args:
        top_chunks_per_query: List of top chunks for each query
        queries_list: List of queries used for similarity search
        output_path: Path to save results
    """
    results = {
        "queries": {},
        "selected_chunks": {}
    }
    
    # Add queries for reference
    for i, query in enumerate(queries_list):
        query_type = QUERY_TYPE_NAMES[i] if i < len(QUERY_TYPE_NAMES) else f"query_{i}"
        results["queries"][query_type] = query
        results["selected_chunks"][query_type] = top_chunks_per_query[i]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved similarity results to {output_path}")
    
    # Log summary
    for i in range(len(queries_list)):
        query_type = QUERY_TYPE_NAMES[i] if i < len(QUERY_TYPE_NAMES) else f"query_{i}"
        logger.info(f"{query_type} query: {len(results['selected_chunks'][query_type])} chunks")


def save_selected_chunks_to_db(
    top_chunks_per_query: List[List[Dict[str, Any]]],
    queries_list: List[str],
    output_path: str = SELECTED_CHUNKS_DB_PATH
) -> None:
    """
    Save selected chunks to a SQLite database, mapping each query to its top chunks.
    
    Args:
        top_chunks_per_query: List of top chunks for each query
        queries_list: List of queries used for similarity search
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
    
    # Create a more flexible schema that can accommodate any number of queries
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS selected_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id INTEGER NOT NULL,
        chunk_idx INTEGER NOT NULL,
        arxiv_id TEXT NOT NULL,
        chunk_text TEXT NOT NULL,
        similarity_score REAL NOT NULL,
        FOREIGN KEY (query_id) REFERENCES queries (id),
        UNIQUE (query_id, chunk_idx)
    )
    ''')
    
    # Create a table to store all similarity scores
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS similarity_scores (
        chunk_id INTEGER NOT NULL,
        query_id INTEGER NOT NULL,
        metadata_score REAL NOT NULL,
        chunk_score REAL NOT NULL,
        FOREIGN KEY (chunk_id) REFERENCES selected_chunks (id),
        FOREIGN KEY (query_id) REFERENCES queries (id),
        UNIQUE (chunk_id, query_id)
    )
    ''')
    
    # Get current timestamp
    timestamp = datetime.datetime.now().isoformat()
    
    # Insert queries and get their IDs
    query_ids = []
    for i, query in enumerate(queries_list):
        query_type = QUERY_TYPE_NAMES[i] if i < len(QUERY_TYPE_NAMES) else f"query_{i}"
        cursor.execute(
            "INSERT INTO queries (query_text, query_type, timestamp) VALUES (?, ?, ?)",
            (query, query_type, timestamp)
        )
        query_ids.append(cursor.lastrowid)
    
    # Insert selected chunks for each query
    for i, (query_id, top_chunks) in enumerate(zip(query_ids, top_chunks_per_query)):
        for chunk in top_chunks:
            # Insert chunk
            cursor.execute(
                """
                INSERT OR REPLACE INTO selected_chunks (
                    query_id, chunk_idx, arxiv_id, chunk_text, similarity_score
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    query_id, 
                    chunk["chunk_idx"], 
                    chunk["arxiv_id"],
                    chunk["chunk_text"],
                    chunk[f"sim_query_{i}_chunk"]
                )
            )
            chunk_id = cursor.lastrowid
            
            # Insert similarity scores for all queries
            for j in range(len(queries_list)):
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO similarity_scores (
                        chunk_id, query_id, metadata_score, chunk_score
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        query_ids[j],
                        chunk[f"sim_query_{j}_metadata"],
                        chunk[f"sim_query_{j}_chunk"]
                    )
                )
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    logger.info(f"Saved selected chunks to database: {output_path}")


def run_chunk_similarity_selection_kgb(
    queries_list: List[str],
    preselected_papers: List[Dict[str, Any]],
    search_mode: str = "answer"
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Run the KGB chunk similarity selection process.
    
    Args:
        queries_list: List of queries to use for similarity search
        preselected_papers: List of papers preselected by category
        search_mode: Search mode, either "answer" or "recommendation"
        
    Returns:
        Tuple of (top_chunks_per_query, all_chunk_results)
    """
    logger.info("Running KGB chunk similarity selection")
    logger.info(f"Number of queries: {len(queries_list)}")
    for i, query in enumerate(queries_list):
        query_type = QUERY_TYPE_NAMES[i] if i < len(QUERY_TYPE_NAMES) else f"query_{i}"
        logger.info(f"{query_type} query: {query}")
    logger.info(f"Number of preselected papers: {len(preselected_papers)}")
    logger.info(f"Search mode: {search_mode}")
    
    # Set the top-k parameter based on the search mode
    if search_mode == "recommendation":
        top_k_chunks = TOP_K_CHUNKS_RECOMMENDATION
        logger.info(f"Using recommendation mode parameters: top_k_chunks={top_k_chunks}")
    else:  # Default to "answer" mode
        top_k_chunks = TOP_K_CHUNKS_ANSWER
        logger.info(f"Using answer mode parameters: top_k_chunks={top_k_chunks}")
    
    # Select chunks by similarity
    top_chunks_per_query, all_chunk_results = select_chunks_by_similarity(
        queries_list=queries_list,
        preselected_papers=preselected_papers,
        top_k_chunks=top_k_chunks
    )
    
    # Save similarity results to JSON
    save_similarity_results(
        top_chunks_per_query=top_chunks_per_query,
        queries_list=queries_list
    )
    
    # Save chunks with scores to JSON (similar to recommendation format)
    save_chunks_with_scores_json(
        top_chunks_per_query=top_chunks_per_query,
        queries_list=queries_list
    )
    
    # Save detailed results to text file
    save_detailed_results(
        all_chunk_results=all_chunk_results,
        queries_list=queries_list
    )
    
    # Save selected chunks to database
    save_selected_chunks_to_db(
        top_chunks_per_query=top_chunks_per_query,
        queries_list=queries_list
    )
    
    return top_chunks_per_query, all_chunk_results
