"""
KGB LLM-based chunk weight determination.

This module implements the KGB LLM-based chunk weight determination process.
It takes the weighted chunks from the KGB chunk weight determination process,
selects the top K chunks per query, and uses an LLM to re-weight them based on
their relevance to the enhanced query.
"""

import os
import json
import time
import sqlite3
import logging
import requests
import datetime
from typing import List, Dict, Any, Tuple, Set
from pathlib import Path

from .chunk_weight_determination_kgb_llm_parameters import (
    INPUT_DB_PATH,
    WEIGHTED_CHUNKS_DB_PATH,
    WEIGHTED_CHUNKS_JSON_PATH,
    TOP_K_CHUNKS_PER_QUERY,
    LLM_BATCH_SIZE,
    ORIGINAL_SCORE_WEIGHT,
    LLM_SCORE_WEIGHT,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    CHUNK_TEXT_TEMPLATE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_weighted_chunks_from_kgb_db(db_path: str = INPUT_DB_PATH) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """
    Get weighted chunks from KGB database.
    
    Args:
        db_path: Path to the database
    
    Returns:
        Tuple of (weighted_chunks_by_query, queries_list)
    """
    logger = logging.getLogger(__name__)
    chunk_id_count = {}
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get queries
        cursor.execute("""
            SELECT id, query_text, query_type, query_index
            FROM queries
            ORDER BY query_index
        """)
        queries = cursor.fetchall()
        
        if not queries:
            logger.error("No queries found in the database")
            raise ValueError("No queries found in the database")
        
        queries_list = [query['query_text'] for query in queries]
        query_ids = [query['id'] for query in queries]
        
        # Get weighted chunks for each query
        weighted_chunks_by_query = []
        
        for query_id in query_ids:
            cursor.execute("""
                SELECT wc.*, q.query_text, q.query_type, q.query_index
                FROM weighted_chunks wc
                JOIN queries q ON wc.query_id = q.id
                WHERE wc.query_id = ?
                ORDER BY wc.query_rank
            """, (query_id,))
            
            query_chunks = []
            for row in cursor.fetchall():
                chunk = dict(row)
                chunk_id = chunk['chunk_id']
                
                # Track duplicate chunks
                if chunk_id in chunk_id_count:
                    chunk_id_count[chunk_id] += 1
                else:
                    chunk_id_count[chunk_id] = 1
                    
                query_chunks.append(chunk)
            
            weighted_chunks_by_query.append(query_chunks)
        
        conn.close()
        
        # Log duplicate chunks
        duplicates = {chunk_id: count for chunk_id, count in chunk_id_count.items() if count > 1}
        if duplicates:
            logger.info(f"Found {len(duplicates)} duplicate chunks across queries")
            for chunk_id, count in duplicates.items():
                logger.info(f"  Chunk {chunk_id} appears {count} times")
        else:
            logger.info("No duplicate chunks found across queries")
        
        logger.info(f"Retrieved {sum(len(chunks) for chunks in weighted_chunks_by_query)} weighted chunks from KGB database")
        return weighted_chunks_by_query, queries_list
        
    except sqlite3.Error as e:
        logger.error(f"Database error when retrieving weighted chunks: {e}")
        raise ValueError(f"Database error: {e}")

def select_top_k_chunks_per_query(
    weighted_chunks_by_query: List[List[Dict[str, Any]]],
    top_k: int = TOP_K_CHUNKS_PER_QUERY
) -> Tuple[List[List[Dict[str, Any]]], int]:
    """
    Select the top K chunks per query based on their normalized weight.
    Handles duplicates by keeping only the highest-scoring instance of each chunk.
    
    Args:
        weighted_chunks_by_query: List of weighted chunks for each query
        top_k: Number of top chunks to select per query
    
    Returns:
        Tuple of (List of top K chunks for each query, Total unique chunks)
    """
    # First, count total chunks before deduplication
    total_chunks = sum(len(chunks) for chunks in weighted_chunks_by_query)
    logger.info(f"Starting with {total_chunks} total chunks across all queries")
    
    # Identify all duplicate chunks and find the highest-scoring instance
    chunk_id_to_best_chunk = {}
    chunk_id_to_best_score = {}
    chunk_id_to_query_idx = {}
    
    # Find the best instance of each chunk across all queries
    for i, query_chunks in enumerate(weighted_chunks_by_query):
        for chunk in query_chunks:
            chunk_id = chunk['chunk_id']
            score = chunk.get('normalized_weight', 0)
            
            if chunk_id not in chunk_id_to_best_score or score > chunk_id_to_best_score[chunk_id]:
                chunk_id_to_best_score[chunk_id] = score
                chunk_id_to_best_chunk[chunk_id] = chunk
                chunk_id_to_query_idx[chunk_id] = i
    
    # Count unique chunks
    unique_chunks_count = len(chunk_id_to_best_chunk)
    logger.info(f"Found {unique_chunks_count} unique chunks after deduplication")
    
    # Log duplicate information
    duplicate_counts = {}
    for query_chunks in weighted_chunks_by_query:
        for chunk in query_chunks:
            chunk_id = chunk['chunk_id']
            if chunk_id in duplicate_counts:
                duplicate_counts[chunk_id] += 1
            else:
                duplicate_counts[chunk_id] = 1
    
    duplicates = {chunk_id: count for chunk_id, count in duplicate_counts.items() if count > 1}
    if duplicates:
        logger.info(f"Found {len(duplicates)} duplicate chunks across queries")
        for chunk_id, count in duplicates.items():
            best_score = chunk_id_to_best_score[chunk_id]
            query_idx = chunk_id_to_query_idx[chunk_id]
            logger.info(f"  Chunk {chunk_id} appears {count} times, keeping instance from query {query_idx} with score {best_score:.4f}")
    
    # Now select top K chunks per query, skipping duplicates that aren't the best instance
    top_k_chunks_by_query = []
    total_selected_chunks = 0
    
    for i, query_chunks in enumerate(weighted_chunks_by_query):
        # Filter out duplicates that aren't the best instance
        filtered_chunks = []
        for chunk in query_chunks:
            chunk_id = chunk['chunk_id']
            if chunk is chunk_id_to_best_chunk[chunk_id]:  # Only keep the best instance
                filtered_chunks.append(chunk)
        
        # Sort by normalized weight (descending)
        sorted_chunks = sorted(filtered_chunks, key=lambda x: x.get('normalized_weight', 0), reverse=True)
        
        # Select top K chunks
        top_chunks = sorted_chunks[:top_k]
        total_selected_chunks += len(top_chunks)
        
        logger.info(f"Selected top {len(top_chunks)} unique chunks for query {i} (after removing duplicates)")
        top_k_chunks_by_query.append(top_chunks)
    
    logger.info(f"Selected {total_selected_chunks} chunks in total across all queries")
    
    # Count how many unique chunks we have in our final selection
    selected_chunk_ids = set()
    for query_chunks in top_k_chunks_by_query:
        for chunk in query_chunks:
            selected_chunk_ids.add(chunk['chunk_id'])
    
    unique_selected_count = len(selected_chunk_ids)
    logger.info(f"Of these, {unique_selected_count} are unique chunks")
    
    return top_k_chunks_by_query, unique_selected_count

def prepare_batches(
    top_k_chunks_by_query: List[List[Dict[str, Any]]],
    expected_unique_count: int,
    batch_size: int = LLM_BATCH_SIZE
) -> List[List[Dict[str, Any]]]:
    """
    Prepare batches of chunks for LLM processing.
    Ensures all unique chunks are included exactly once.
    
    Args:
        top_k_chunks_by_query: List of top K chunks for each query
        expected_unique_count: Expected number of unique chunks
        batch_size: Size of each batch
    
    Returns:
        List of batches, where each batch is a list of chunks
    """
    # Flatten the list of chunks while ensuring each unique chunk appears exactly once
    unique_chunks = {}
    for query_chunks in top_k_chunks_by_query:
        for chunk in query_chunks:
            chunk_id = chunk['chunk_id']
            if chunk_id not in unique_chunks:
                unique_chunks[chunk_id] = chunk
    
    # Convert to list and verify count
    unique_chunks_list = list(unique_chunks.values())
    actual_unique_count = len(unique_chunks_list)
    
    if actual_unique_count != expected_unique_count:
        logger.warning(f"Expected {expected_unique_count} unique chunks but found {actual_unique_count}")
    
    # Create batches
    batches = []
    for i in range(0, len(unique_chunks_list), batch_size):
        batch = unique_chunks_list[i:i+batch_size]
        batches.append(batch)
    
    # Calculate expected number of batches and chunks in last batch
    expected_batches = (actual_unique_count + batch_size - 1) // batch_size
    expected_last_batch_size = actual_unique_count % batch_size
    if expected_last_batch_size == 0 and actual_unique_count > 0:
        expected_last_batch_size = batch_size
    
    # Log batch information
    batch_sizes = [len(batch) for batch in batches]
    logger.info(f"Prepared {len(batches)} batches: {batch_sizes}")
    logger.info(f"Total unique chunks to process: {actual_unique_count}")
    
    # Verify batch count and last batch size
    if len(batches) != expected_batches:
        logger.error(f"Expected {expected_batches} batches but created {len(batches)}")
    elif batches and len(batches[-1]) != expected_last_batch_size:
        logger.error(f"Expected last batch size {expected_last_batch_size} but got {len(batches[-1])}")
    
    return batches

def call_llm_api(
    batch: List[Dict[str, Any]],
    enhanced_query: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS
) -> List[Dict[str, Any]]:
    """
    Call the LLM API to process a batch of chunks.
    
    Args:
        batch: Batch of chunks to process
        enhanced_query: Enhanced query to use for relevance scoring
        model: LLM model to use
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens for LLM response
    
    Returns:
        List of processed chunks with summaries and relevance scores
    """
    # Prepare chunk texts
    chunk_texts = ""
    for i, chunk in enumerate(batch):
        chunk_texts += CHUNK_TEXT_TEMPLATE.format(
            index=i+1,
            chunk_id=chunk['chunk_id'],
            text=chunk['chunk_text']
        )
    
    # Prepare user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        enhanced_query=enhanced_query,
        chunk_texts=chunk_texts
    )
    
    # Prepare API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Call API
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        llm_response = result['choices'][0]['message']['content']
        
        # Parse JSON response
        try:
            llm_results = json.loads(llm_response)
            
            # Map LLM results to chunks by chunk_id
            chunk_id_to_llm_result = {item['chunk_id']: item for item in llm_results}
            
            # Update chunks with LLM results
            processed_chunks = []
            for chunk in batch:
                chunk_id = chunk['chunk_id']
                if chunk_id in chunk_id_to_llm_result:
                    llm_result = chunk_id_to_llm_result[chunk_id]
                    chunk['llm_summary'] = llm_result.get('summary', '')
                    chunk['llm_relevance_score'] = llm_result.get('relevance_score', 0.0)
                    chunk['llm_reasoning'] = llm_result.get('reasoning', '')
                else:
                    logger.warning(f"LLM did not return results for chunk {chunk_id}")
                    chunk['llm_summary'] = ''
                    chunk['llm_relevance_score'] = 0.0
                    chunk['llm_reasoning'] = 'No results returned by LLM'
                
                processed_chunks.append(chunk)
            
            return processed_chunks
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"LLM response: {llm_response}")
            
            # Return chunks with default values
            for chunk in batch:
                chunk['llm_summary'] = ''
                chunk['llm_relevance_score'] = 0.0
                chunk['llm_reasoning'] = f'Error parsing LLM response: {e}'
            
            return batch
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        
        # Return chunks with default values
        for chunk in batch:
            chunk['llm_summary'] = ''
            chunk['llm_relevance_score'] = 0.0
            chunk['llm_reasoning'] = f'API request failed: {e}'
        
        return batch

# This function has been replaced by direct calls to call_llm_api in determine_chunk_weights_kgb_llm

def combine_scores(
    weighted_chunks_by_query: List[List[Dict[str, Any]]],
    chunk_id_to_processed: Dict[str, Dict[str, Any]],
    original_weight: float = ORIGINAL_SCORE_WEIGHT,
    llm_weight: float = LLM_SCORE_WEIGHT
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Combine original scores with LLM-assigned scores.
    Only includes chunks that were processed by the LLM in the final rankings.
    
    Args:
        weighted_chunks_by_query: List of weighted chunks for each query
        chunk_id_to_processed: Dictionary mapping chunk_id to processed chunk
        original_weight: Weight for original similarity-based score
        llm_weight: Weight for LLM-assigned relevance score
    
    Returns:
        Tuple of (reweighted_chunks_by_query, all_reweighted_chunks)
    """
    # Count total chunks before filtering
    total_chunks = sum(len(chunks) for chunks in weighted_chunks_by_query)
    logger.info(f"Starting with {total_chunks} total chunks before filtering")
    
    # Track processed and unprocessed chunks
    processed_chunk_ids = set(chunk_id_to_processed.keys())
    logger.info(f"Found {len(processed_chunk_ids)} unique chunks processed by LLM")
    
    # Create a set of all chunk IDs from the original weighted chunks
    all_chunk_ids = set()
    for query_chunks in weighted_chunks_by_query:
        for chunk in query_chunks:
            all_chunk_ids.add(chunk['chunk_id'])
    
    # Calculate unprocessed chunks
    unprocessed_chunk_ids = all_chunk_ids - processed_chunk_ids
    logger.info(f"Found {len(unprocessed_chunk_ids)} chunks that were not processed by LLM")
    
    # Create reweighted chunks by query
    reweighted_chunks_by_query = []
    processed_chunks_per_query = []
    
    # Track unique chunks that have been added to the final results
    # This ensures we don't count duplicates in the final tally
    added_chunk_ids = set()
    
    # Apply LLM scores to chunks
    for i, query_chunks in enumerate(weighted_chunks_by_query):
        reweighted_query_chunks = []
        query_processed_count = 0
        
        for chunk in query_chunks:
            chunk_id = chunk['chunk_id']
            
            # Only include chunks that were processed by the LLM
            if chunk_id in chunk_id_to_processed:
                processed_chunk = chunk_id_to_processed[chunk_id]
                query_processed_count += 1
                
                # Get LLM-assigned relevance score
                llm_score = processed_chunk.get('llm_relevance_score', 0.0)
                
                # Get original normalized weight
                original_score = chunk.get('normalized_weight', 0.0)
                
                # Combine scores
                combined_score = (original_weight * original_score) + (llm_weight * llm_score)
                
                # Create reweighted chunk
                reweighted_chunk = chunk.copy()
                reweighted_chunk['llm_summary'] = processed_chunk.get('llm_summary', '')
                reweighted_chunk['llm_relevance_score'] = llm_score
                reweighted_chunk['llm_reasoning'] = processed_chunk.get('llm_reasoning', '')
                reweighted_chunk['original_weight'] = original_score
                reweighted_chunk['final_weight'] = combined_score
                
                reweighted_query_chunks.append(reweighted_chunk)
                added_chunk_ids.add(chunk_id)
        
        # Track how many chunks were processed for this query
        processed_chunks_per_query.append(query_processed_count)
        
        # Only add this query's chunks if we have any
        if reweighted_query_chunks:
            # Sort by combined score (descending)
            reweighted_query_chunks.sort(key=lambda x: x.get('final_weight', 0), reverse=True)
            
            # Add query rank
            for j, chunk in enumerate(reweighted_query_chunks):
                chunk['query_rank'] = j + 1
            
            reweighted_chunks_by_query.append(reweighted_query_chunks)
        else:
            # Add an empty list for this query if no chunks were processed
            reweighted_chunks_by_query.append([])
            logger.warning(f"No chunks were processed by LLM for query {i}")
    
    # Combine all reweighted chunks into a single list
    all_reweighted_chunks = []
    for query_chunks in reweighted_chunks_by_query:
        all_reweighted_chunks.extend(query_chunks)
    
    # Sort all chunks by combined score (descending)
    all_reweighted_chunks.sort(key=lambda x: x.get('final_weight', 0), reverse=True)
    
    # Add global rank
    for i, chunk in enumerate(all_reweighted_chunks):
        chunk['global_rank'] = i + 1
    
    # Log information about processed vs. unprocessed chunks
    logger.info(f"Combined scores for {len(all_reweighted_chunks)} chunks (from {len(added_chunk_ids)} unique chunks)")
    if len(all_reweighted_chunks) != len(processed_chunk_ids):
        logger.warning(f"Discrepancy in chunk counts: {len(all_reweighted_chunks)} chunks in final ranking vs {len(processed_chunk_ids)} processed by LLM")
        # Explain the discrepancy
        duplicate_count = len(all_reweighted_chunks) - len(added_chunk_ids)
        if duplicate_count > 0:
            logger.info(f"The discrepancy is due to {duplicate_count} duplicate chunks appearing in multiple queries")
    
    # Log per-query chunk counts
    for i, count in enumerate(processed_chunks_per_query):
        logger.info(f"Query {i}: {count} chunks processed by LLM, {len(reweighted_chunks_by_query[i])} in final ranking")
    
    return reweighted_chunks_by_query, all_reweighted_chunks

def save_reweighted_chunks_to_db(
    reweighted_chunks_by_query: List[List[Dict[str, Any]]],
    all_reweighted_chunks: List[Dict[str, Any]],
    queries_list: List[str],
    db_path: str = WEIGHTED_CHUNKS_DB_PATH
) -> None:
    """
    Save reweighted chunks to a SQLite database.
    Only includes chunks that were processed by the LLM.
    
    Args:
        reweighted_chunks_by_query: List of reweighted chunks for each query
        all_reweighted_chunks: List of all reweighted chunks
        queries_list: List of queries
        db_path: Path to save the database
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
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
            CREATE TABLE IF NOT EXISTS reweighted_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                arxiv_id TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                original_weight REAL NOT NULL,
                llm_relevance_score REAL NOT NULL,
                final_weight REAL NOT NULL,
                llm_summary TEXT,
                llm_reasoning TEXT,
                global_rank INTEGER NOT NULL,
                query_rank INTEGER NOT NULL,
                FOREIGN KEY (query_id) REFERENCES queries (id),
                UNIQUE (query_id, chunk_id)
            )
        """)
        
        # Insert queries
        cursor.execute("DELETE FROM queries")
        cursor.execute("DELETE FROM reweighted_chunks")
        
        query_ids = []
        timestamp = datetime.datetime.now().isoformat()
        
        for i, query_text in enumerate(queries_list):
            query_type = f"query_{i}"
            cursor.execute(
                "INSERT INTO queries (query_text, query_type, query_index, timestamp) VALUES (?, ?, ?, ?)",
                (query_text, query_type, i, timestamp)
            )
            query_ids.append(cursor.lastrowid)
        
        # Insert reweighted chunks
        for i, query_chunks in enumerate(reweighted_chunks_by_query):
            # Skip empty query chunks (no chunks processed by LLM)
            if not query_chunks:
                logger.info(f"No chunks to save for query {i}")
                continue
                
            for chunk in query_chunks:
                cursor.execute("""
                    INSERT INTO reweighted_chunks (
                        query_id, chunk_id, arxiv_id, chunk_idx, chunk_text,
                        original_weight, llm_relevance_score, final_weight,
                        llm_summary, llm_reasoning,
                        global_rank, query_rank
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_ids[i], chunk['chunk_id'], chunk['arxiv_id'], chunk['chunk_idx'], chunk['chunk_text'],
                    chunk['original_weight'], chunk['llm_relevance_score'], chunk['final_weight'],
                    chunk.get('llm_summary', ''), chunk.get('llm_reasoning', ''),
                    chunk['global_rank'], chunk['query_rank']
                ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(all_reweighted_chunks)} reweighted chunks to database: {db_path}")
        
    except sqlite3.Error as e:
        logger.error(f"Database error when saving reweighted chunks: {e}")
        raise ValueError(f"Database error: {e}")

def save_reweighted_chunks_to_json(
    reweighted_chunks_by_query: List[List[Dict[str, Any]]],
    all_reweighted_chunks: List[Dict[str, Any]],
    queries_list: List[str],
    enhanced_query: str,
    output_path: str = WEIGHTED_CHUNKS_JSON_PATH,
    original_weight: float = ORIGINAL_SCORE_WEIGHT,
    llm_weight: float = LLM_SCORE_WEIGHT
) -> None:
    """
    Save reweighted chunks to a JSON file.
    Only includes chunks that were processed by the LLM.
    
    Args:
        reweighted_chunks_by_query: List of reweighted chunks for each query
        all_reweighted_chunks: List of all reweighted chunks
        queries_list: List of queries
        enhanced_query: Enhanced query used for LLM relevance scoring
        output_path: Path to save the JSON file
        original_weight: Weight for original similarity-based score
        llm_weight: Weight for LLM-assigned relevance score
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare queries data
        queries_data = []
        for i, query in enumerate(queries_list):
            query_type = f"query_{i}"
            chunk_count = len(reweighted_chunks_by_query[i]) if i < len(reweighted_chunks_by_query) else 0
            queries_data.append({
                "query_index": i,
                "query_type": query_type,
                "query_text": query,
                "num_chunks": chunk_count
            })
        
        # Prepare data for JSON
        output_data = {
            "queries": queries_data,
            "enhanced_query": enhanced_query,
            "per_query_chunks": {},
            "global_ranking": all_reweighted_chunks,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": {
                "num_queries": len(queries_list),
                "num_chunks_total": len(all_reweighted_chunks),
                "parameters": {
                    "mode": "answer",
                    "original_score_weight": original_weight,
                    "llm_score_weight": llm_weight,
                    "top_k_chunks_per_query": TOP_K_CHUNKS_PER_QUERY,
                    "llm_batch_size": LLM_BATCH_SIZE,
                    "llm_model": LLM_MODEL
                }
            }
        }
        
        # Add per-query chunks
        for i, query_chunks in enumerate(reweighted_chunks_by_query):
            query_type = f"query_{i}"
            output_data["per_query_chunks"][query_type] = query_chunks
        
        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved reweighted chunks to JSON file: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving reweighted chunks to JSON: {e}")
        raise ValueError(f"JSON save error: {e}")

def deduplicate_chunks_across_queries(weighted_chunks_by_query: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """
    Deduplicate chunks across queries, keeping only the highest-scoring instance of each chunk.
    
    Args:
        weighted_chunks_by_query: List of weighted chunks for each query
        
    Returns:
        List of deduplicated weighted chunks for each query
    """
    logger = logging.getLogger(__name__)
    
    # First, identify all duplicate chunks and find the highest-scoring instance
    chunk_id_to_best_chunk = {}
    chunk_id_to_best_score = {}
    chunk_id_to_query_idx = {}
    chunk_id_to_rank = {}
    
    # Find the best instance of each chunk across all queries
    for i, query_chunks in enumerate(weighted_chunks_by_query):
        for j, chunk in enumerate(query_chunks):
            chunk_id = chunk['chunk_id']
            score = chunk.get('normalized_weight', 0)
            
            if chunk_id not in chunk_id_to_best_score or score > chunk_id_to_best_score[chunk_id]:
                chunk_id_to_best_score[chunk_id] = score
                chunk_id_to_best_chunk[chunk_id] = chunk
                chunk_id_to_query_idx[chunk_id] = i
                chunk_id_to_rank[chunk_id] = j
    
    # Create new deduplicated chunks by query
    deduplicated_chunks_by_query = [[] for _ in range(len(weighted_chunks_by_query))]
    
    # Place each chunk in its original query if it's the best instance
    for chunk_id, chunk in chunk_id_to_best_chunk.items():
        query_idx = chunk_id_to_query_idx[chunk_id]
        deduplicated_chunks_by_query[query_idx].append(chunk)
    
    # Sort chunks within each query by their original rank
    for i, query_chunks in enumerate(deduplicated_chunks_by_query):
        query_chunks.sort(key=lambda x: x.get('query_rank', 0))
    
    # Log deduplication results
    total_original = sum(len(chunks) for chunks in weighted_chunks_by_query)
    total_deduplicated = sum(len(chunks) for chunks in deduplicated_chunks_by_query)
    duplicates_removed = total_original - total_deduplicated
    
    logger.info(f"Deduplicated chunks across queries: {total_original} → {total_deduplicated} chunks")
    logger.info(f"Removed {duplicates_removed} duplicate chunks")
    
    return deduplicated_chunks_by_query

def determine_chunk_weights_kgb_llm(
    enhanced_query: str,
    top_k: int = TOP_K_CHUNKS_PER_QUERY,
    batch_size: int = LLM_BATCH_SIZE,
    original_weight: float = ORIGINAL_SCORE_WEIGHT,
    llm_weight: float = LLM_SCORE_WEIGHT
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Determine chunk weights using KGB LLM-based method.
    
    Args:
        enhanced_query: Enhanced query used for LLM relevance scoring
        top_k: Number of top chunks to select per query
        batch_size: Size of each batch for LLM processing
        original_weight: Weight for original similarity-based score
        llm_weight: Weight for LLM-assigned relevance score
    
    Returns:
        Tuple of (reweighted_chunks_by_query, all_reweighted_chunks)
    """
    start_time = time.time()
    logger.info("Starting KGB LLM-based chunk weight determination process...")
    
    # Get weighted chunks from KGB database
    weighted_chunks_by_query, queries_list = get_weighted_chunks_from_kgb_db()
    total_initial_chunks = sum(len(chunks) for chunks in weighted_chunks_by_query)
    logger.info(f"Retrieved {total_initial_chunks} weighted chunks from KGB database")
    
    # Deduplicate chunks across queries before selecting top K
    deduplicated_chunks_by_query = deduplicate_chunks_across_queries(weighted_chunks_by_query)
    
    # Select top K chunks per query from deduplicated chunks
    top_k_chunks_by_query = []
    total_selected = 0
    
    for i, query_chunks in enumerate(deduplicated_chunks_by_query):
        # Sort by normalized weight (descending)
        sorted_chunks = sorted(query_chunks, key=lambda x: x.get('normalized_weight', 0), reverse=True)
        
        # Select top K chunks
        top_chunks = sorted_chunks[:top_k]
        total_selected += len(top_chunks)
        
        logger.info(f"Selected top {len(top_chunks)} chunks for query {i}")
        top_k_chunks_by_query.append(top_chunks)
    
    logger.info(f"Selected {total_selected} chunks in total across all queries")
    
    # Count unique chunks to process
    unique_chunk_ids = set()
    for query_chunks in top_k_chunks_by_query:
        for chunk in query_chunks:
            unique_chunk_ids.add(chunk['chunk_id'])
    
    unique_chunk_count = len(unique_chunk_ids)
    logger.info(f"These selected chunks represent {unique_chunk_count} unique chunks")
    
    # Prepare batches for LLM processing
    batches = []
    all_chunks_to_process = []
    
    # Collect all unique chunks to process
    processed_chunk_ids = set()
    for query_chunks in top_k_chunks_by_query:
        for chunk in query_chunks:
            chunk_id = chunk['chunk_id']
            if chunk_id not in processed_chunk_ids:
                all_chunks_to_process.append(chunk)
                processed_chunk_ids.add(chunk_id)
    
    # Create batches
    for i in range(0, len(all_chunks_to_process), batch_size):
        batch = all_chunks_to_process[i:i+batch_size]
        batches.append(batch)
    
    batch_sizes = [len(batch) for batch in batches]
    logger.info(f"Prepared {len(batches)} batches: {batch_sizes}")
    logger.info(f"Total unique chunks to process: {len(all_chunks_to_process)}")
    
    # Process chunks with LLM
    chunk_id_to_processed = {}
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} chunks")
        processed_batch = call_llm_api(batch, enhanced_query)
        
        # Add processed chunks to the map
        for chunk in processed_batch:
            chunk_id = chunk['chunk_id']
            chunk_id_to_processed[chunk_id] = chunk
    
    logger.info(f"Processed {len(chunk_id_to_processed)} chunks with LLM")
    
    # Create reweighted chunks by query
    reweighted_chunks_by_query = []
    all_reweighted_chunks = []
    
    for i, query_chunks in enumerate(top_k_chunks_by_query):
        reweighted_query_chunks = []
        
        for chunk in query_chunks:
            chunk_id = chunk['chunk_id']
            
            if chunk_id in chunk_id_to_processed:
                processed_chunk = chunk_id_to_processed[chunk_id]
                
                # Get LLM-assigned relevance score
                llm_score = processed_chunk.get('llm_relevance_score', 0.0)
                
                # Get original normalized weight
                original_score = chunk.get('normalized_weight', 0.0)
                
                # Combine scores
                combined_score = (original_weight * original_score) + (llm_weight * llm_score)
                
                # Create reweighted chunk
                reweighted_chunk = chunk.copy()
                reweighted_chunk['llm_summary'] = processed_chunk.get('llm_summary', '')
                reweighted_chunk['llm_relevance_score'] = llm_score
                reweighted_chunk['llm_reasoning'] = processed_chunk.get('llm_reasoning', '')
                reweighted_chunk['original_weight'] = original_score
                reweighted_chunk['final_weight'] = combined_score
                
                reweighted_query_chunks.append(reweighted_chunk)
            else:
                logger.warning(f"Chunk {chunk_id} was not processed by LLM")
        
        # Sort by combined score (descending)
        reweighted_query_chunks.sort(key=lambda x: x.get('final_weight', 0), reverse=True)
        
        # Add query rank
        for j, chunk in enumerate(reweighted_query_chunks):
            chunk['query_rank'] = j + 1
        
        reweighted_chunks_by_query.append(reweighted_query_chunks)
        all_reweighted_chunks.extend(reweighted_query_chunks)
    
    # Sort all chunks by combined score (descending)
    all_reweighted_chunks.sort(key=lambda x: x.get('final_weight', 0), reverse=True)
    
    # Add global rank
    for i, chunk in enumerate(all_reweighted_chunks):
        chunk['global_rank'] = i + 1
    
    # Log final results
    logger.info(f"Final reweighted chunks: {len(all_reweighted_chunks)} chunks in total")
    for i, query_chunks in enumerate(reweighted_chunks_by_query):
        logger.info(f"Query {i}: {len(query_chunks)} chunks in final ranking")
    
    # Save results
    save_reweighted_chunks_to_db(
        reweighted_chunks_by_query=reweighted_chunks_by_query,
        all_reweighted_chunks=all_reweighted_chunks,
        queries_list=queries_list
    )
    
    save_reweighted_chunks_to_json(
        reweighted_chunks_by_query=reweighted_chunks_by_query,
        all_reweighted_chunks=all_reweighted_chunks,
        queries_list=queries_list,
        enhanced_query=enhanced_query,
        original_weight=original_weight,
        llm_weight=llm_weight
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"KGB LLM-based chunk weight determination completed in {elapsed_time:.2f} seconds")
    
    return reweighted_chunks_by_query, all_reweighted_chunks
