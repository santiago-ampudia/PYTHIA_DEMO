"""
submodule_chunk_weight_determination/chunk_weight_determination_kgb.py

This module implements an enhanced version of Step 6 of the paper search pipeline: Chunk Weight Determination.
KGB (Keyword-Guided Batch) version supports processing multiple queries in a batch.

Purpose: Compute a final, normalized weight for each candidate chunk based on its similarity 
to the current query and neighboring queries (previous and next), applying a weighted formula
that considers the context of each query in the sequence.

Process:
    1. Apply a preselection cut to only work with chunks above a similarity threshold
    2. For each query in the list:
        a. For each chunk, compute a relevance score based on weighted similarity scores:
           score = weight_query_i-1 * sim_i-1 + weight_query_i * sim_i + weight_query_i+1 * sim_i+1
        b. Handle edge cases (first and last queries) with adjusted weights
    3. Normalize all weights relative to the maximum weight for each query
    4. Create both per-query rankings and a global ranking

Output: 
    1. Per-query lists of chunks with normalized weights
    2. Global list of all chunks with normalized weights
"""

import os
import sqlite3
import logging
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import datetime
from .chunk_weight_determination_kgb_parameters import (
    INPUT_DB_PATH,
    WEIGHTED_CHUNKS_DB_PATH,
    WEIGHTED_CHUNKS_JSON_PATH,
    # Query weight parameters
    CURRENT_QUERY_WEIGHT,
    PREVIOUS_QUERY_WEIGHT,
    NEXT_QUERY_WEIGHT,
    # Adjusted weights for edge cases
    FIRST_QUERY_WEIGHT,
    FIRST_QUERY_NEXT_WEIGHT,
    LAST_QUERY_WEIGHT,
    LAST_QUERY_PREVIOUS_WEIGHT,
    # Preselection threshold
    SIMILARITY_PRESELECTION_THRESHOLD,
    # Metadata similarity boost parameters
    METADATA_BOOST_WEIGHT,
    METADATA_BOOST_THRESHOLD,
    METADATA_BOOST_FIRST_QUERY,
    METADATA_BOOST_LAST_QUERY
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('chunk_weight_determination_kgb')


def get_scored_chunks_kgb(db_path: str = INPUT_DB_PATH) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Retrieve scored chunks from the KGB database.
    
    Args:
        db_path: Path to the database containing scored chunks
        
    Returns:
        Tuple of (scored_chunks, queries_list)
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get the most recent queries
        cursor.execute("""
            SELECT q.id, q.query_text, q.query_type
            FROM queries q
            ORDER BY q.id
        """)
        
        queries = cursor.fetchall()
        queries_list = [row['query_text'] for row in queries]
        query_ids = [row['id'] for row in queries]
        query_types = [row['query_type'] for row in queries]
        
        # Get all chunks for these queries
        scored_chunks_by_query = []
        
        for i, query_id in enumerate(query_ids):
            query_chunks = []
            
            cursor.execute("""
                SELECT 
                    sc.id,
                    sc.chunk_idx,
                    sc.arxiv_id,
                    sc.chunk_text,
                    sc.similarity_score,
                    q.query_type
                FROM selected_chunks sc
                JOIN queries q ON sc.query_id = q.id
                WHERE sc.query_id = ?
            """, (query_id,))
            
            chunks = cursor.fetchall()
            
            for chunk in chunks:
                chunk_dict = dict(chunk)
                chunk_dict['chunk_id'] = f"{chunk_dict['arxiv_id']}_{chunk_dict['chunk_idx']}"
                chunk_dict['query_index'] = i
                chunk_dict['query_type'] = query_types[i]
                
                # Get similarity scores for this chunk with all queries
                cursor.execute("""
                    SELECT 
                        q.id as query_id,
                        ss.chunk_score,
                        ss.metadata_score,
                        q.query_type
                    FROM similarity_scores ss
                    JOIN queries q ON ss.query_id = q.id
                    WHERE ss.chunk_id = ?
                """, (chunk_dict['id'],))
                
                scores = cursor.fetchall()
                
                # Add similarity scores to chunk dictionary
                for score in scores:
                    score_dict = dict(score)
                    score_idx = query_ids.index(score_dict['query_id'])
                    chunk_dict[f'sim_query_{score_idx}_chunk'] = score_dict['chunk_score']
                    chunk_dict[f'sim_query_{score_idx}_metadata'] = score_dict['metadata_score']
                
                query_chunks.append(chunk_dict)
            
            scored_chunks_by_query.append(query_chunks)
        
        conn.close()
        
        # Flatten the list for global operations
        all_chunks = [chunk for query_chunks in scored_chunks_by_query for chunk in query_chunks]
        
        logger.info(f"Retrieved {len(all_chunks)} scored chunks for {len(queries_list)} queries")
        
        return scored_chunks_by_query, queries_list
        
    except sqlite3.Error as e:
        logger.error(f"Database error when retrieving scored chunks: {e}")
        raise ValueError(f"Database error: {e}")


def determine_chunk_weights_kgb(
    scored_chunks_by_query: List[List[Dict[str, Any]]],
    queries_list: List[str],
    current_query_weight: float = CURRENT_QUERY_WEIGHT,
    previous_query_weight: float = PREVIOUS_QUERY_WEIGHT,
    next_query_weight: float = NEXT_QUERY_WEIGHT,
    first_query_weight: float = FIRST_QUERY_WEIGHT,
    first_query_next_weight: float = FIRST_QUERY_NEXT_WEIGHT,
    last_query_weight: float = LAST_QUERY_WEIGHT,
    last_query_previous_weight: float = LAST_QUERY_PREVIOUS_WEIGHT,
    similarity_threshold: float = SIMILARITY_PRESELECTION_THRESHOLD,
    metadata_boost_weight: float = METADATA_BOOST_WEIGHT,
    metadata_boost_threshold: float = METADATA_BOOST_THRESHOLD,
    metadata_boost_first_query: bool = METADATA_BOOST_FIRST_QUERY,
    metadata_boost_last_query: bool = METADATA_BOOST_LAST_QUERY
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Determine weights for each chunk based on similarity scores with current and neighboring queries.
    Applies a conservative boost for chunks with high metadata similarity to edge queries.
    
    Args:
        scored_chunks_by_query: List of chunks for each query with similarity scores
        queries_list: List of queries used for similarity search
        current_query_weight: Weight for the current query
        previous_query_weight: Weight for the previous query
        next_query_weight: Weight for the next query
        first_query_weight: Weight for the first query when it's the current query
        first_query_next_weight: Weight for the next query when first query is current
        last_query_weight: Weight for the last query when it's the current query
        last_query_previous_weight: Weight for the previous query when last query is current
        similarity_threshold: Minimum similarity score to consider chunks
        metadata_boost_weight: Weight for metadata similarity boost
        metadata_boost_threshold: Minimum metadata similarity threshold to apply boost
        metadata_boost_first_query: Whether to boost based on first query metadata similarity
        metadata_boost_last_query: Whether to boost based on last query metadata similarity
        
    Returns:
        Tuple of (weighted_chunks_by_query, all_weighted_chunks)
    """
    logger.info("Determining chunk weights using KGB method...")
    
    num_queries = len(queries_list)
    weighted_chunks_by_query = []
    all_weighted_chunks = []
    
    # Process each query
    for i, query_chunks in enumerate(scored_chunks_by_query):
        query_type = f"query_{i}"
        logger.info(f"Processing chunks for query {i} ({query_type})")
        
        # Apply preselection cut - only include chunks that pass the similarity threshold
        filtered_chunks = []
        
        for chunk in query_chunks:
            # Check if the chunk similarity score exceeds the threshold
            if chunk[f'sim_query_{i}_chunk'] >= similarity_threshold:
                filtered_chunks.append(chunk)
        
        logger.info(f"Applied preselection cut for query {i}: {len(filtered_chunks)}/{len(query_chunks)} chunks remain")
        
        # Calculate weights for each chunk
        weighted_chunks = []
        for chunk in filtered_chunks:
            # Determine weights based on position in query sequence
            if i == 0:  # First query
                # For the first query, use special weights
                w_current = first_query_weight
                w_previous = 0.0  # No previous query
                w_next = first_query_next_weight
            elif i == num_queries - 1:  # Last query
                # For the last query, use special weights
                w_current = last_query_weight
                w_previous = last_query_previous_weight
                w_next = 0.0  # No next query
            else:  # Middle queries
                # For middle queries, use standard weights
                w_current = current_query_weight
                w_previous = previous_query_weight
                w_next = next_query_weight
            
            # Calculate relevance score as weighted sum of chunk similarities
            relevance_score = w_current * chunk[f'sim_query_{i}_chunk']
            
            # Add previous query similarity if applicable
            if i > 0:
                relevance_score += w_previous * chunk.get(f'sim_query_{i-1}_chunk', 0.0)
            
            # Add next query similarity if applicable
            if i < num_queries - 1:
                relevance_score += w_next * chunk.get(f'sim_query_{i+1}_chunk', 0.0)
            
            # Apply metadata similarity boost for edge queries if applicable
            metadata_boost = 0.0
            
            # Boost based on first query metadata similarity if enabled
            if metadata_boost_first_query and chunk.get(f'sim_query_0_metadata', 0) >= metadata_boost_threshold:
                metadata_boost += metadata_boost_weight * chunk.get(f'sim_query_0_metadata', 0)
                
            # Boost based on last query metadata similarity if enabled
            last_query_idx = num_queries - 1
            if metadata_boost_last_query and chunk.get(f'sim_query_{last_query_idx}_metadata', 0) >= metadata_boost_threshold:
                metadata_boost += metadata_boost_weight * chunk.get(f'sim_query_{last_query_idx}_metadata', 0)
                
            # Add the metadata boost to the relevance score
            if metadata_boost > 0:
                logger.debug(f"Applied metadata boost of {metadata_boost:.4f} to chunk {chunk['chunk_id']}")
                relevance_score += metadata_boost
            
            # Create weighted chunk entry
            weighted_chunk = {
                "chunk_id": chunk['chunk_id'],
                "arxiv_id": chunk['arxiv_id'],
                "chunk_idx": chunk['chunk_idx'],
                "chunk_text": chunk['chunk_text'],
                "query_index": i,
                "query_type": query_type,
                "query_text": queries_list[i],
                "final_weight": relevance_score
            }
            
            # Add all similarity scores for reference
            for j in range(num_queries):
                weighted_chunk[f'sim_query_{j}_chunk'] = chunk.get(f'sim_query_{j}_chunk', 0.0)
                weighted_chunk[f'sim_query_{j}_metadata'] = chunk.get(f'sim_query_{j}_metadata', 0.0)
            
            weighted_chunks.append(weighted_chunk)
        
        # Find maximum weight for normalization within this query
        if weighted_chunks:
            max_weight = max(chunk['final_weight'] for chunk in weighted_chunks)
            
            # Normalize weights
            for chunk in weighted_chunks:
                chunk['normalized_weight'] = chunk['final_weight'] / max_weight
        else:
            logger.warning(f"No chunks passed the preselection cut for query {i}")
        
        # Sort by normalized weight (descending)
        weighted_chunks.sort(key=lambda x: x.get('normalized_weight', 0), reverse=True)
        
        weighted_chunks_by_query.append(weighted_chunks)
        all_weighted_chunks.extend(weighted_chunks)
    
    # Sort all chunks by normalized weight (descending) for global ranking
    all_weighted_chunks.sort(key=lambda x: x.get('normalized_weight', 0), reverse=True)
    
    logger.info(f"Determined weights for {len(all_weighted_chunks)} chunks across {num_queries} queries")
    return weighted_chunks_by_query, all_weighted_chunks


def save_weighted_chunks_to_db_kgb(
    weighted_chunks_by_query: List[List[Dict[str, Any]]],
    all_weighted_chunks: List[Dict[str, Any]],
    queries_list: List[str],
    db_path: str = WEIGHTED_CHUNKS_DB_PATH
) -> None:
    """
    Save weighted chunks to a SQLite database.
    
    Args:
        weighted_chunks_by_query: List of weighted chunks for each query
        all_weighted_chunks: List of all weighted chunks (global ranking)
        queries_list: List of queries used for similarity search
        db_path: Path to save the database
    """
    try:
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Remove existing database if it exists
        if os.path.exists(db_path):
            logger.info(f"Removing existing weighted chunks database: {db_path}")
            os.remove(db_path)
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                query_type TEXT NOT NULL,
                query_index INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weighted_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                arxiv_id TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                final_weight REAL NOT NULL,
                normalized_weight REAL NOT NULL,

                global_rank INTEGER NOT NULL,
                query_rank INTEGER NOT NULL,
                FOREIGN KEY (query_id) REFERENCES queries (id),
                UNIQUE (query_id, chunk_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS similarity_scores (
                chunk_id INTEGER NOT NULL,
                query_id INTEGER NOT NULL,
                chunk_score REAL NOT NULL,
                metadata_score REAL NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES weighted_chunks (id),
                FOREIGN KEY (query_id) REFERENCES queries (id),
                UNIQUE (chunk_id, query_id)
            )
        """)
        
        # Insert queries
        query_ids = {}
        timestamp = datetime.datetime.now().isoformat()
        
        for i, query_text in enumerate(queries_list):
            query_type = f"query_{i}"
            cursor.execute(
                "INSERT INTO queries (query_text, query_type, query_index, timestamp) VALUES (?, ?, ?, ?)",
                (query_text, query_type, i, timestamp)
            )
            query_ids[i] = cursor.lastrowid
        
        # Insert weighted chunks with global and per-query rankings
        # First, assign global rankings
        for global_rank, chunk in enumerate(all_weighted_chunks):
            chunk['global_rank'] = global_rank + 1  # 1-based ranking
        
        # Then, assign per-query rankings and insert
        for i, query_chunks in enumerate(weighted_chunks_by_query):
            for query_rank, chunk in enumerate(query_chunks):
                chunk['query_rank'] = query_rank + 1  # 1-based ranking
                
                # Find this chunk in the global ranking to get its global rank
                global_chunk = next((c for c in all_weighted_chunks if c['chunk_id'] == chunk['chunk_id'] and c['query_index'] == i), None)
                if global_chunk:
                    chunk['global_rank'] = global_chunk['global_rank']
                else:
                    chunk['global_rank'] = 0  # Should never happen
                
                # Insert the chunk
                cursor.execute("""
                    INSERT INTO weighted_chunks (
                        query_id, chunk_id, arxiv_id, chunk_idx, chunk_text,
                        final_weight, normalized_weight,
                        global_rank, query_rank
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_ids[i], chunk['chunk_id'], chunk['arxiv_id'], chunk['chunk_idx'], chunk['chunk_text'],
                    chunk['final_weight'], chunk['normalized_weight'],
                    chunk['global_rank'], chunk['query_rank']
                ))
                
                chunk_id = cursor.lastrowid
                
                # Insert similarity scores for all queries
                for j in range(len(queries_list)):
                    cursor.execute("""
                        INSERT INTO similarity_scores (
                            chunk_id, query_id, chunk_score, metadata_score
                        ) VALUES (?, ?, ?, ?)
                    """, (
                        chunk_id, query_ids[j], 
                        chunk.get(f'sim_query_{j}_chunk', 0.0),
                        chunk.get(f'sim_query_{j}_metadata', 0.0)
                    ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(all_weighted_chunks)} weighted chunks to database: {db_path}")
        
    except sqlite3.Error as e:
        logger.error(f"Database error when saving weighted chunks: {e}")
        raise ValueError(f"Database error: {e}")


def save_weighted_chunks_to_json_kgb(
    weighted_chunks_by_query: List[List[Dict[str, Any]]],
    all_weighted_chunks: List[Dict[str, Any]],
    queries_list: List[str],
    output_path: str = WEIGHTED_CHUNKS_JSON_PATH,
    current_query_weight: float = CURRENT_QUERY_WEIGHT,
    previous_query_weight: float = PREVIOUS_QUERY_WEIGHT,
    next_query_weight: float = NEXT_QUERY_WEIGHT,
    first_query_weight: float = FIRST_QUERY_WEIGHT,
    first_query_next_weight: float = FIRST_QUERY_NEXT_WEIGHT,
    last_query_weight: float = LAST_QUERY_WEIGHT,
    last_query_previous_weight: float = LAST_QUERY_PREVIOUS_WEIGHT,
    similarity_threshold: float = SIMILARITY_PRESELECTION_THRESHOLD,
    metadata_boost_weight: float = METADATA_BOOST_WEIGHT,
    metadata_boost_threshold: float = METADATA_BOOST_THRESHOLD,
    metadata_boost_first_query: bool = METADATA_BOOST_FIRST_QUERY,
    metadata_boost_last_query: bool = METADATA_BOOST_LAST_QUERY
) -> None:
    """
    Save weighted chunks to a JSON file.
    
    Args:
        weighted_chunks_by_query: List of weighted chunks for each query
        all_weighted_chunks: List of all weighted chunks (global ranking)
        queries_list: List of queries used for similarity search
        output_path: Path to save the JSON file
        search_mode: Search mode ("answer" or "recommendation")
        current_query_weight: Weight for the current query
        previous_query_weight: Weight for the previous query
        next_query_weight: Weight for the next query
        first_query_weight: Weight for the first query when it's the current query
        first_query_next_weight: Weight for the next query when first query is current
        last_query_weight: Weight for the last query when it's the current query
        last_query_previous_weight: Weight for the previous query when last query is current
        similarity_threshold: Similarity threshold for preselection
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare queries data
        queries_data = []
        for i, query in enumerate(queries_list):
            query_type = f"query_{i}"
            queries_data.append({
                "query_index": i,
                "query_type": query_type,
                "query_text": query,
                "num_chunks": len(weighted_chunks_by_query[i])
            })
        
        # Prepare data for JSON
        output_data = {
            "queries": queries_data,
            "per_query_chunks": {},
            "global_ranking": all_weighted_chunks,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": {
                "num_queries": len(queries_list),
                "num_chunks_total": len(all_weighted_chunks),
                "parameters": {
                    "mode": "answer",
                    "current_query_weight": current_query_weight,
                    "previous_query_weight": previous_query_weight,
                    "next_query_weight": next_query_weight,
                    "first_query_weight": first_query_weight,
                    "first_query_next_weight": first_query_next_weight,
                    "last_query_weight": last_query_weight,
                    "last_query_previous_weight": last_query_previous_weight,
                    "similarity_threshold": similarity_threshold,
                    "metadata_boost_first_query": metadata_boost_first_query,
                    "metadata_boost_last_query": metadata_boost_last_query,
                    "metadata_boost_threshold": metadata_boost_threshold,
                    "metadata_boost_weight": metadata_boost_weight
                }
            }
        }
        
        # Add per-query chunks
        for i, query_chunks in enumerate(weighted_chunks_by_query):
            query_type = f"query_{i}"
            output_data["per_query_chunks"][query_type] = query_chunks
        
        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved weighted chunks to JSON file: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving weighted chunks to JSON: {e}")
        raise ValueError(f"JSON save error: {e}")


def run_chunk_weight_determination_kgb(
    queries_list: List[str],
    metadata_boost_weight: float = METADATA_BOOST_WEIGHT,
    metadata_boost_threshold: float = METADATA_BOOST_THRESHOLD,
    metadata_boost_first_query: bool = METADATA_BOOST_FIRST_QUERY,
    metadata_boost_last_query: bool = METADATA_BOOST_LAST_QUERY
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Run the KGB chunk weight determination process for answer mode.
    
    Args:
        queries_list: List of queries used for similarity search
    
    Returns:
        Tuple of (weighted_chunks_by_query, all_weighted_chunks)
    """
    start_time = time.time()
    logger.info("Starting KGB chunk weight determination process...")
    
    # Get scored chunks from database
    scored_chunks_by_query, queries = get_scored_chunks_kgb()
    
    # Verify that the queries match
    if len(queries) != len(queries_list):
        logger.warning(f"Number of queries in database ({len(queries)}) doesn't match provided queries list ({len(queries_list)})")
    
    # Determine chunk weights
    weighted_chunks_by_query, all_weighted_chunks = determine_chunk_weights_kgb(
        scored_chunks_by_query=scored_chunks_by_query,
        queries_list=queries,
        similarity_threshold=SIMILARITY_PRESELECTION_THRESHOLD,
        metadata_boost_weight=metadata_boost_weight,
        metadata_boost_threshold=metadata_boost_threshold,
        metadata_boost_first_query=metadata_boost_first_query,
        metadata_boost_last_query=metadata_boost_last_query
    )
    
    # Save results
    save_weighted_chunks_to_db_kgb(
        weighted_chunks_by_query=weighted_chunks_by_query,
        all_weighted_chunks=all_weighted_chunks,
        queries_list=queries
    )
    
    save_weighted_chunks_to_json_kgb(
        weighted_chunks_by_query=weighted_chunks_by_query,
        all_weighted_chunks=all_weighted_chunks,
        queries_list=queries,
        similarity_threshold=SIMILARITY_PRESELECTION_THRESHOLD,
        metadata_boost_weight=metadata_boost_weight,
        metadata_boost_threshold=metadata_boost_threshold,
        metadata_boost_first_query=metadata_boost_first_query,
        metadata_boost_last_query=metadata_boost_last_query
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"KGB chunk weight determination completed in {elapsed_time:.2f} seconds")
    
    return weighted_chunks_by_query, all_weighted_chunks


if __name__ == "__main__":
    run_chunk_weight_determination_kgb(["test_query"])
