"""
submodule_chunk_weight_determination/chunk_weight_determination.py

This module implements Step 6 of the paper search pipeline: Chunk Weight Determination.

Purpose: Compute a final, normalized weight for each candidate chunk based on its similarity 
to the enhanced query, subtopic, and topic, and apply a metadata-based boost that favors 
papers whose summaries/titles are especially relevant to the original query.

Process:
    1. Apply a preselection cut to only work with chunks above a similarity threshold
    2. For each chunk, compute a relevance score based on weighted similarity scores
    3. Apply a metadata boost if the metadata similarity exceeds a threshold
    4. Normalize all weights relative to the maximum weight
    5. Sort chunks by normalized weight

Output: List of chunks with final and normalized weights
"""

import os
import sqlite3
import logging
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import datetime
from .chunk_weight_determination_parameters import (
    INPUT_DB_PATH,
    WEIGHTED_CHUNKS_DB_PATH,
    WEIGHTED_CHUNKS_JSON_PATH,
    # Mode-specific weight parameters
    ENHANCED_QUERY_WEIGHT_ANSWER,
    SUBTOPIC_QUERY_WEIGHT_ANSWER,
    TOPIC_QUERY_WEIGHT_ANSWER,
    ENHANCED_QUERY_WEIGHT_RECOMMENDATION,
    SUBTOPIC_QUERY_WEIGHT_RECOMMENDATION,
    TOPIC_QUERY_WEIGHT_RECOMMENDATION,
    # Mode-specific metadata boost parameters
    METADATA_BOOST_FACTOR_ANSWER,
    METADATA_THRESHOLD_TOPIC_ANSWER,
    METADATA_THRESHOLD_SUBTOPIC_ANSWER,
    METADATA_THRESHOLD_ENHANCED_ANSWER,
    METADATA_BOOST_FACTOR_RECOMMENDATION,
    METADATA_THRESHOLD_TOPIC_RECOMMENDATION,
    METADATA_THRESHOLD_SUBTOPIC_RECOMMENDATION,
    METADATA_THRESHOLD_ENHANCED_RECOMMENDATION,
    # Mode-specific preselection thresholds
    SIMILARITY_PRESELECTION_THRESHOLD_ANSWER,
    SIMILARITY_PRESELECTION_THRESHOLD_RECOMMENDATION,
    # Flag to ensure top chunks per query
    ENSURE_TOP_CHUNKS_PER_QUERY
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('chunk_weight_determination')


def get_scored_chunks(db_path: str = INPUT_DB_PATH) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Retrieve scored chunks from the database.
    
    Args:
        db_path: Path to the database containing scored chunks
        
    Returns:
        Tuple of (scored_chunks, queries_dict)
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get the most recent queries (one of each type)
        cursor.execute("""
            SELECT q.id, q.query_text, q.query_type
            FROM queries q
            INNER JOIN (
                SELECT query_type, MAX(timestamp) as max_timestamp
                FROM queries
                GROUP BY query_type
            ) latest ON q.query_type = latest.query_type AND q.timestamp = latest.max_timestamp
        """)
        
        queries = cursor.fetchall()
        queries_dict = {row['query_type']: row['query_text'] for row in queries}
        query_ids = {row['query_type']: row['id'] for row in queries}
        
        # Get all chunks for these queries
        scored_chunks = []
        seen_chunk_ids = set()  # Track unique chunk IDs to avoid duplicates
        
        for query_type, query_id in query_ids.items():
            cursor.execute("""
                SELECT 
                    chunk_idx,
                    arxiv_id,
                    chunk_text,
                    sim_enhanced_chunk,
                    sim_topic_chunk,
                    sim_subtopic_chunk,
                    sim_enhanced_metadata,
                    sim_topic_metadata,
                    sim_subtopic_metadata
                FROM selected_chunks
                WHERE query_id = ?
            """, (query_id,))
            
            chunks = cursor.fetchall()
            for chunk in chunks:
                chunk_dict = dict(chunk)
                chunk_dict['chunk_id'] = f"{chunk_dict['arxiv_id']}_{chunk_dict['chunk_idx']}"
                
                # Only add the chunk if we haven't seen it before
                if chunk_dict['chunk_id'] not in seen_chunk_ids:
                    scored_chunks.append(chunk_dict)
                    seen_chunk_ids.add(chunk_dict['chunk_id'])
        
        conn.close()
        
        logger.info(f"Retrieved {len(scored_chunks)} unique scored chunks from database")
        return scored_chunks, queries_dict
        
    except sqlite3.Error as e:
        logger.error(f"Database error when retrieving scored chunks: {e}")
        raise ValueError(f"Database error: {e}")


def determine_chunk_weights(
    scored_chunks: List[Dict[str, Any]],
    enhanced_query_weight: float = ENHANCED_QUERY_WEIGHT_ANSWER,
    subtopic_query_weight: float = SUBTOPIC_QUERY_WEIGHT_ANSWER,
    topic_query_weight: float = TOPIC_QUERY_WEIGHT_ANSWER,
    metadata_boost_factor: float = METADATA_BOOST_FACTOR_ANSWER,
    metadata_threshold_topic: float = METADATA_THRESHOLD_TOPIC_ANSWER,
    metadata_threshold_subtopic: float = METADATA_THRESHOLD_SUBTOPIC_ANSWER,
    metadata_threshold_enhanced: float = METADATA_THRESHOLD_ENHANCED_ANSWER,
    similarity_threshold: float = SIMILARITY_PRESELECTION_THRESHOLD_ANSWER,
    ensure_top_chunks: bool = ENSURE_TOP_CHUNKS_PER_QUERY
) -> List[Dict[str, Any]]:
    """
    Determine weights for each chunk based on similarity scores and metadata boost.
    
    Args:
        scored_chunks: List of chunks with similarity scores
        enhanced_query_weight: Weight for enhanced query similarity (αₑ)
        subtopic_query_weight: Weight for subtopic query similarity (αₛ)
        topic_query_weight: Weight for topic query similarity (αₜ)
        metadata_boost_factor: Boost factor for metadata relevance (γ)
        metadata_threshold_topic: Threshold for applying metadata boost (τ)
        metadata_threshold_subtopic: Threshold for applying metadata boost (τ)
        metadata_threshold_enhanced: Threshold for applying metadata boost (τ)
        similarity_threshold: Minimum similarity score to consider chunks
        
    Returns:
        List of chunks with final and normalized weights
    """
    logger.info("Determining chunk weights...")
    
    # Apply preselection cut
    filtered_chunks = []
    
    # If ensure_top_chunks is True, find the top chunk for each query type
    top_enhanced_chunk = None
    top_subtopic_chunk = None
    top_topic_chunk = None
    
    if ensure_top_chunks:
        # Find top chunks for each query type
        for chunk in scored_chunks:
            # Initialize top chunks if not set
            if top_enhanced_chunk is None or chunk['sim_enhanced_chunk'] > top_enhanced_chunk['sim_enhanced_chunk']:
                top_enhanced_chunk = chunk
            if top_subtopic_chunk is None or chunk['sim_subtopic_chunk'] > top_subtopic_chunk['sim_subtopic_chunk']:
                top_subtopic_chunk = chunk
            if top_topic_chunk is None or chunk['sim_topic_chunk'] > top_topic_chunk['sim_topic_chunk']:
                top_topic_chunk = chunk
    
    # Add chunks that pass the similarity threshold
    for chunk in scored_chunks:
        # Check if any of the chunk similarity scores exceeds the threshold
        if (chunk['sim_enhanced_chunk'] >= similarity_threshold or
            chunk['sim_subtopic_chunk'] >= similarity_threshold or
            chunk['sim_topic_chunk'] >= similarity_threshold):
            filtered_chunks.append(chunk)
    
    # Ensure top chunks are included even if they don't pass the threshold
    if ensure_top_chunks:
        # Check if top enhanced chunk is already included
        if top_enhanced_chunk and top_enhanced_chunk not in filtered_chunks:
            filtered_chunks.append(top_enhanced_chunk)
            logger.info(f"Added top enhanced chunk (sim={top_enhanced_chunk['sim_enhanced_chunk']:.4f}) to ensure representation")
        
        # Check if top subtopic chunk is already included
        if top_subtopic_chunk and top_subtopic_chunk not in filtered_chunks:
            filtered_chunks.append(top_subtopic_chunk)
            logger.info(f"Added top subtopic chunk (sim={top_subtopic_chunk['sim_subtopic_chunk']:.4f}) to ensure representation")
        
        # Check if top topic chunk is already included
        if top_topic_chunk and top_topic_chunk not in filtered_chunks:
            filtered_chunks.append(top_topic_chunk)
            logger.info(f"Added top topic chunk (sim={top_topic_chunk['sim_topic_chunk']:.4f}) to ensure representation")
    
    logger.info(f"Applied preselection cut: {len(filtered_chunks)}/{len(scored_chunks)} chunks remain")
    
    # Calculate weights for each chunk
    weighted_chunks = []
    for chunk in filtered_chunks:
        # Check if this is one of the top chunks for any query type
        is_top_enhanced = ensure_top_chunks and chunk is top_enhanced_chunk
        is_top_subtopic = ensure_top_chunks and chunk is top_subtopic_chunk
        is_top_topic = ensure_top_chunks and chunk is top_topic_chunk
        is_top_chunk = is_top_enhanced or is_top_subtopic or is_top_topic
        
        # Calculate relevance score as weighted sum of chunk similarities
        relevance_score = (
            enhanced_query_weight * chunk['sim_enhanced_chunk'] +
            subtopic_query_weight * chunk['sim_subtopic_chunk'] +
            topic_query_weight * chunk['sim_topic_chunk']
        )
        
        # Calculate metadata score as max of metadata similarities
        metadata_score = max(
            chunk['sim_enhanced_metadata'],
            chunk['sim_subtopic_metadata'],
            chunk['sim_topic_metadata']
        )
        if metadata_score == chunk['sim_enhanced_metadata']:
            metadata_treshold = metadata_threshold_enhanced
        elif metadata_score == chunk['sim_subtopic_metadata']:
            metadata_treshold = metadata_threshold_subtopic
        else:
            metadata_treshold = metadata_threshold_topic
        
        # Apply metadata boost if score exceeds threshold
        if metadata_score > metadata_treshold:
            meta_boost = 1 + metadata_boost_factor * (metadata_score - metadata_treshold)
        else:
            meta_boost = 1
        
        # Calculate final weight
        final_weight = relevance_score * meta_boost
        
        # If this is one of the top chunks and we're in recommendation mode, assign a very high weight
        # In answer mode, we just use the calculated weights without artificial boosting
        if is_top_chunk and enhanced_query_weight == ENHANCED_QUERY_WEIGHT_RECOMMENDATION:
            # Store the original weight for reference
            original_weight = final_weight
            # Set a very high weight to ensure it's at the top
            final_weight = 1000.0
            
            # Log which top chunk this is and its weight boost
            if is_top_enhanced:
                logger.info(f"Boosting weight for top enhanced chunk: {original_weight:.4f} -> {final_weight:.4f}")
            elif is_top_subtopic:
                logger.info(f"Boosting weight for top subtopic chunk: {original_weight:.4f} -> {final_weight:.4f}")
            elif is_top_topic:
                logger.info(f"Boosting weight for top topic chunk: {original_weight:.4f} -> {final_weight:.4f}")
        
        # Create weighted chunk entry
        weighted_chunk = {
            "chunk_id": chunk['chunk_id'],
            "arxiv_id": chunk['arxiv_id'],
            "chunk_idx": chunk['chunk_idx'],
            "chunk_text": chunk['chunk_text'],
            "final_weight": final_weight,
            "relevance_score": relevance_score,
            "metadata_score": metadata_score,
            "meta_boost": meta_boost,
            "sim_enhanced_chunk": chunk['sim_enhanced_chunk'],
            "sim_subtopic_chunk": chunk['sim_subtopic_chunk'],
            "sim_topic_chunk": chunk['sim_topic_chunk'],
            "sim_enhanced_metadata": chunk['sim_enhanced_metadata'],
            "sim_subtopic_metadata": chunk['sim_subtopic_metadata'],
            "sim_topic_metadata": chunk['sim_topic_metadata']
        }
        
        weighted_chunks.append(weighted_chunk)
    
    # Find maximum weight for normalization
    if weighted_chunks:
        max_weight = max(chunk['final_weight'] for chunk in weighted_chunks)
        
        # Normalize weights
        for chunk in weighted_chunks:
            chunk['normalized_weight'] = chunk['final_weight'] / max_weight
    else:
        logger.warning("No chunks passed the preselection cut")
    
    # Sort by normalized weight (descending)
    weighted_chunks.sort(key=lambda x: x.get('normalized_weight', 0), reverse=True)
    
    logger.info(f"Determined weights for {len(weighted_chunks)} chunks")
    return weighted_chunks


def save_weighted_chunks_to_db(
    weighted_chunks: List[Dict[str, Any]],
    queries: Dict[str, str],
    db_path: str = WEIGHTED_CHUNKS_DB_PATH
) -> None:
    """
    Save weighted chunks to a SQLite database.
    
    Args:
        weighted_chunks: List of chunks with weights
        queries: Dictionary of query texts by type
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
                relevance_score REAL NOT NULL,
                metadata_score REAL NOT NULL,
                meta_boost REAL NOT NULL,
                sim_enhanced_chunk REAL NOT NULL,
                sim_subtopic_chunk REAL NOT NULL,
                sim_topic_chunk REAL NOT NULL,
                sim_enhanced_metadata REAL NOT NULL,
                sim_subtopic_metadata REAL NOT NULL,
                sim_topic_metadata REAL NOT NULL,
                FOREIGN KEY (query_id) REFERENCES queries (id),
                UNIQUE (query_id, chunk_id)
            )
        """)
        
        # Insert queries
        query_ids = {}
        timestamp = datetime.datetime.now().isoformat()
        
        for query_type, query_text in queries.items():
            cursor.execute(
                "INSERT INTO queries (query_text, query_type, timestamp) VALUES (?, ?, ?)",
                (query_text, query_type, timestamp)
            )
            query_ids[query_type] = cursor.lastrowid
        
        # Insert weighted chunks
        for chunk in weighted_chunks:
            # Use the enhanced query ID for all chunks
            query_id = query_ids.get('enhanced', list(query_ids.values())[0])
            
            try:
                cursor.execute("""
                    INSERT INTO weighted_chunks (
                        query_id, chunk_id, arxiv_id, chunk_idx, chunk_text,
                        final_weight, normalized_weight, relevance_score,
                        metadata_score, meta_boost,
                        sim_enhanced_chunk, sim_subtopic_chunk, sim_topic_chunk,
                        sim_enhanced_metadata, sim_subtopic_metadata, sim_topic_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id, chunk['chunk_id'], chunk['arxiv_id'], chunk['chunk_idx'], chunk['chunk_text'],
                    chunk['final_weight'], chunk['normalized_weight'], chunk['relevance_score'],
                    chunk['metadata_score'], chunk['meta_boost'],
                    chunk['sim_enhanced_chunk'], chunk['sim_subtopic_chunk'], chunk['sim_topic_chunk'],
                    chunk['sim_enhanced_metadata'], chunk['sim_subtopic_metadata'], chunk['sim_topic_metadata']
                ))
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    # Update existing entry instead of inserting
                    logger.debug(f"Updating existing chunk: {chunk['chunk_id']}")
                    cursor.execute("""
                        UPDATE weighted_chunks SET
                            final_weight = ?, normalized_weight = ?, relevance_score = ?,
                            metadata_score = ?, meta_boost = ?,
                            sim_enhanced_chunk = ?, sim_subtopic_chunk = ?, sim_topic_chunk = ?,
                            sim_enhanced_metadata = ?, sim_subtopic_metadata = ?, sim_topic_metadata = ?
                        WHERE query_id = ? AND chunk_id = ?
                    """, (
                        chunk['final_weight'], chunk['normalized_weight'], chunk['relevance_score'],
                        chunk['metadata_score'], chunk['meta_boost'],
                        chunk['sim_enhanced_chunk'], chunk['sim_subtopic_chunk'], chunk['sim_topic_chunk'],
                        chunk['sim_enhanced_metadata'], chunk['sim_subtopic_metadata'], chunk['sim_topic_metadata'],
                        query_id, chunk['chunk_id']
                    ))
                else:
                    raise
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(weighted_chunks)} weighted chunks to database: {db_path}")
        
    except sqlite3.Error as e:
        logger.error(f"Database error when saving weighted chunks: {e}")
        raise ValueError(f"Database error: {e}")


def save_weighted_chunks_to_json(
    weighted_chunks: List[Dict[str, Any]],
    queries: Dict[str, str],
    output_path: str = WEIGHTED_CHUNKS_JSON_PATH,
    search_mode: str = "answer",
    enhanced_query_weight: float = None,
    subtopic_query_weight: float = None,
    topic_query_weight: float = None,
    metadata_boost_factor: float = None,
    metadata_threshold_topic: float = None,
    metadata_threshold_subtopic: float = None,
    metadata_threshold_enhanced: float = None,
    similarity_threshold: float = None
) -> None:
    """
    Save weighted chunks to a JSON file.
    
    Args:
        weighted_chunks: List of chunks with weights
        queries: Dictionary of query texts by type
        output_path: Path to save the JSON file
        search_mode: Search mode ("answer" or "recommendation")
        enhanced_query_weight: Weight for enhanced query similarity
        subtopic_query_weight: Weight for subtopic query similarity
        topic_query_weight: Weight for topic query similarity
        metadata_boost_factor: Boost factor for metadata similarity
        metadata_threshold_topic: Threshold for topic metadata similarity
        metadata_threshold_subtopic: Threshold for subtopic metadata similarity
        metadata_threshold_enhanced: Threshold for enhanced metadata similarity
        similarity_threshold: Similarity threshold for preselection
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for JSON
        output_data = {
            "queries": queries,
            "weighted_chunks": weighted_chunks,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": {
                "num_chunks": len(weighted_chunks),
                "parameters": {
                    "search_mode": search_mode,
                    "enhanced_query_weight": enhanced_query_weight,
                    "subtopic_query_weight": subtopic_query_weight,
                    "topic_query_weight": topic_query_weight,
                    "metadata_boost_factor": metadata_boost_factor,
                    "metadata_threshold_topic": metadata_threshold_topic,
                    "metadata_threshold_subtopic": metadata_threshold_subtopic,
                    "metadata_threshold_enhanced": metadata_threshold_enhanced,
                    "similarity_threshold": similarity_threshold
                }
            }
        }
        
        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved weighted chunks to JSON file: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving weighted chunks to JSON: {e}")
        raise ValueError(f"JSON save error: {e}")


def run_chunk_weight_determination(search_mode: str = "answer") -> List[Dict[str, Any]]:
    """
    Run the chunk weight determination process.
    
    Args:
        search_mode: Search mode, either "answer" or "recommendation"
    
    Returns:
        List of weighted chunks sorted by normalized weight
    """
    start_time = time.time()
    logger.info(f"Starting chunk weight determination process for {search_mode} mode...")
    
    # Get scored chunks from database
    scored_chunks, queries = get_scored_chunks()
    
    # Set parameters based on search mode
    if search_mode == "recommendation":
        logger.info("Using recommendation mode parameters for weight determination")
        enhanced_query_weight = ENHANCED_QUERY_WEIGHT_RECOMMENDATION
        subtopic_query_weight = SUBTOPIC_QUERY_WEIGHT_RECOMMENDATION
        topic_query_weight = TOPIC_QUERY_WEIGHT_RECOMMENDATION
        metadata_boost_factor = METADATA_BOOST_FACTOR_RECOMMENDATION
        metadata_threshold_topic = METADATA_THRESHOLD_TOPIC_RECOMMENDATION
        metadata_threshold_subtopic = METADATA_THRESHOLD_SUBTOPIC_RECOMMENDATION
        metadata_threshold_enhanced = METADATA_THRESHOLD_ENHANCED_RECOMMENDATION
        similarity_threshold = SIMILARITY_PRESELECTION_THRESHOLD_RECOMMENDATION
    else:  # Default to "answer" mode
        logger.info("Using answer mode parameters for weight determination")
        enhanced_query_weight = ENHANCED_QUERY_WEIGHT_ANSWER
        subtopic_query_weight = SUBTOPIC_QUERY_WEIGHT_ANSWER
        topic_query_weight = TOPIC_QUERY_WEIGHT_ANSWER
        metadata_boost_factor = METADATA_BOOST_FACTOR_ANSWER
        metadata_threshold_topic = METADATA_THRESHOLD_TOPIC_ANSWER
        metadata_threshold_subtopic = METADATA_THRESHOLD_SUBTOPIC_ANSWER
        metadata_threshold_enhanced = METADATA_THRESHOLD_ENHANCED_ANSWER
        similarity_threshold = SIMILARITY_PRESELECTION_THRESHOLD_ANSWER
    
    # Determine chunk weights with mode-specific parameters
    weighted_chunks = determine_chunk_weights(
        scored_chunks=scored_chunks,
        enhanced_query_weight=enhanced_query_weight,
        subtopic_query_weight=subtopic_query_weight,
        topic_query_weight=topic_query_weight,
        metadata_boost_factor=metadata_boost_factor,
        metadata_threshold_topic=metadata_threshold_topic,
        metadata_threshold_subtopic=metadata_threshold_subtopic,
        metadata_threshold_enhanced=metadata_threshold_enhanced,
        similarity_threshold=similarity_threshold,
        ensure_top_chunks=ENSURE_TOP_CHUNKS_PER_QUERY
    )
    
    # Save results
    save_weighted_chunks_to_db(weighted_chunks, queries)
    save_weighted_chunks_to_json(
        weighted_chunks=weighted_chunks, 
        queries=queries,
        search_mode=search_mode,
        enhanced_query_weight=enhanced_query_weight,
        subtopic_query_weight=subtopic_query_weight,
        topic_query_weight=topic_query_weight,
        metadata_boost_factor=metadata_boost_factor,
        metadata_threshold_topic=metadata_threshold_topic,
        metadata_threshold_subtopic=metadata_threshold_subtopic,
        metadata_threshold_enhanced=metadata_threshold_enhanced,
        similarity_threshold=similarity_threshold
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Chunk weight determination completed in {elapsed_time:.2f} seconds")
    
    return weighted_chunks


if __name__ == "__main__":
    run_chunk_weight_determination()
