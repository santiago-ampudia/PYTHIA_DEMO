"""
submodule_chunk_weight_determination/chunk_llm_weight_determination.py

This module implements Step 7 of the paper search pipeline: Chunk Summarization Reweighting.

Purpose: Re-rank the top-k retrieved chunks using LLM summaries that judge their relevance 
to the original query. Implements deduplication, minibatching, retry logic, and structured 
scoring to improve efficiency and robustness.

Process:
    1. Select top-k_summary chunks from weighted_chunks, sorted by normalized_weight
    2. Deduplicate highly similar chunks using cosine similarity
    3. Batch remaining chunks into minibatches for LLM processing
    4. For each minibatch, run LLM to generate summaries and relevance scores
    5. Compute final adjusted weights using a weighted combination of LLM scores and original weights
    6. Sort chunks by final adjusted weight

Output: List of chunks with summaries and adjusted weights
"""

import os
import sqlite3
import logging
import json
import numpy as np
import time
import datetime
import asyncio
import aiohttp
import requests
from openai import AsyncOpenAI
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunk_llm_weight_determination_parameters import (
    INPUT_DB_PATH,
    SUMMARIZED_CHUNKS_DB_PATH,
    SUMMARIZED_CHUNKS_JSON_PATH,
    K_SUMMARY,
    SUMMARY_LENGTH,
    LAMBDA,
    LAMBDA_ANSWER,
    LAMBDA_RECOMMENDATION,
    SIMILARITY_THRESHOLD,
    MAX_WORKERS,
    BATCH_SIZE,
    MAX_CHUNK_LENGTH,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    LLM_RETRY_ATTEMPTS,
    LLM_DEFAULT_SCORE,
    DEFAULT_OPENAI_API_KEY,
    SYSTEM_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE_ANSWER,
    SYSTEM_PROMPT_TEMPLATE_RECOMMENDATION,
    USER_PROMPT_TEMPLATE,
    CHUNK_TEXT_TEMPLATE,
    K_SUMMARY_ANSWER,
    K_SUMMARY_RECOMMENDATION,
    SIMILARITY_THRESHOLD_ANSWER,
    SIMILARITY_THRESHOLD_RECOMMENDATION,
    MAX_TWEET_LENGTH,
    LLM_BATCH_DELAY
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('chunk_llm_weight_determination')


def get_weighted_chunks(db_path: str = INPUT_DB_PATH) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Retrieve weighted chunks from the database.
    
    Args:
        db_path: Path to the database containing weighted chunks
        
    Returns:
        Tuple of (weighted_chunks, queries_dict)
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
        
        # Get enhanced query ID
        enhanced_query_id = query_ids.get('enhanced', list(query_ids.values())[0])
        
        # Get all weighted chunks for the enhanced query
        cursor.execute("""
            SELECT 
                chunk_id,
                arxiv_id,
                chunk_idx,
                chunk_text,
                final_weight,
                normalized_weight
            FROM weighted_chunks
            WHERE query_id = ?
            ORDER BY normalized_weight DESC
        """, (enhanced_query_id,))
        
        weighted_chunks = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        logger.info(f"Retrieved {len(weighted_chunks)} weighted chunks from database")
        return weighted_chunks, queries_dict
        
    except sqlite3.Error as e:
        logger.error(f"Database error when retrieving weighted chunks: {e}")
        raise ValueError(f"Database error: {e}")


def deduplicate_chunks(chunks: List[Dict[str, Any]], similarity_threshold: float = SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
    """
    Deduplicate chunks based on text similarity.
    
    Args:
        chunks: List of chunks to deduplicate
        similarity_threshold: Cosine similarity threshold for deduplication
        
    Returns:
        Deduplicated list of chunks
    """
    if not chunks:
        return []
    
    logger.info(f"Deduplicating {len(chunks)} chunks...")
    
    # Extract chunk texts
    chunk_texts = [chunk['chunk_text'] for chunk in chunks]
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    
    # Compute pairwise cosine similarities
    similarities = cosine_similarity(tfidf_matrix)
    
    # Track which chunks to keep
    keep_indices = set()
    
    # Process chunks in order of normalized weight (highest first)
    for i in range(len(chunks)):
        # If this chunk is already marked as a duplicate, skip it
        if i not in keep_indices:
            # Check if this chunk is similar to any kept chunk
            is_duplicate = False
            for j in keep_indices:
                if similarities[i, j] > similarity_threshold:
                    is_duplicate = True
                    break
            
            # If not a duplicate, keep it
            if not is_duplicate:
                keep_indices.add(i)
    
    # Create deduplicated list
    deduplicated_chunks = [chunks[i] for i in sorted(keep_indices)]
    
    logger.info(f"Deduplication complete: {len(deduplicated_chunks)}/{len(chunks)} chunks remain")
    return deduplicated_chunks


def create_similarity_batches(chunks: List[Dict[str, Any]], batch_size: int, similarity_threshold: float) -> List[List[Dict[str, Any]]]:
    """
    Create batches of chunks based on similarity.
    
    Args:
        chunks: List of chunks to batch
        batch_size: Number of chunks per batch
        similarity_threshold: Similarity threshold for considering chunks as similar
        
    Returns:
        List of batches, where each batch is a list of chunks
    """
    if not chunks:
        return []
    
    logger.info(f"Creating similarity-based batches from {len(chunks)} chunks...")
    
    # Extract chunk texts
    chunk_texts = [chunk['chunk_text'] for chunk in chunks]
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    
    # Compute pairwise cosine similarities
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Initialize batches and used chunks
    batches = []
    used_indices = set()
    
    # Process until all chunks are assigned to batches
    while len(used_indices) < len(chunks):
        current_batch_indices = []
        
        # Find first unused chunk as seed
        seed_index = None
        for i in range(len(chunks)):
            if i not in used_indices:
                seed_index = i
                break
        
        if seed_index is None:
            break  # All chunks have been used
        
        # Add seed chunk to batch
        current_batch_indices.append(seed_index)
        used_indices.add(seed_index)
        
        # Find similar chunks to complete the batch
        for i in range(len(chunks)):
            if i not in used_indices and len(current_batch_indices) < batch_size:
                # Check if chunk is similar to any in current batch
                is_similar = any(similarity_matrix[i][j] > similarity_threshold 
                               for j in current_batch_indices)
                if is_similar:
                    current_batch_indices.append(i)
                    used_indices.add(i)
        
        # If batch isn't full, add most similar remaining chunks
        while len(current_batch_indices) < batch_size and len(used_indices) < len(chunks):
            best_similarity = 0
            best_chunk = None
            
            for i in range(len(chunks)):
                if i not in used_indices:
                    # Calculate average similarity to current batch
                    avg_similarity = sum(similarity_matrix[i][j] for j in current_batch_indices) / len(current_batch_indices)
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_chunk = i
            
            if best_chunk is not None:
                current_batch_indices.append(best_chunk)
                used_indices.add(best_chunk)
            else:
                break
        
        # Add completed batch
        batches.append([chunks[i] for i in current_batch_indices])
    
    logger.info(f"Created {len(batches)} similarity-based batches")
    return batches


def create_llm_prompt(chunks: List[Dict[str, Any]], enhanced_query: str, search_mode: str = "answer", summary_length: int = SUMMARY_LENGTH, max_tweet_length: int = MAX_TWEET_LENGTH) -> Dict[str, str]:
    """
    Create a prompt for the LLM to summarize chunks.
    
    Args:
        chunks: List of chunks to summarize
        enhanced_query: Enhanced query text
        search_mode: Search mode ("answer" or "recommendation")
        summary_length: Target summary length in words for answer mode
        max_tweet_length: Maximum length of tweet in characters for recommendation mode
        
    Returns:
        Formatted prompt for the LLM
    """
    # Select the appropriate system prompt based on search mode
    if search_mode == "recommendation":
        system_prompt = SYSTEM_PROMPT_TEMPLATE_RECOMMENDATION.format(max_tweet_length=max_tweet_length)
    else:  # Default to answer mode
        system_prompt = SYSTEM_PROMPT_TEMPLATE_ANSWER.format(summary_length=summary_length)
    
    chunk_texts = ""
    for i, chunk in enumerate(chunks):
        # Truncate chunk text if it's too long
        chunk_text = chunk['chunk_text']
        if len(chunk_text) > MAX_CHUNK_LENGTH:
            logger.info(f"Truncating chunk {i+1} from {len(chunk_text)} to {MAX_CHUNK_LENGTH} characters")
            chunk_text = chunk_text[:MAX_CHUNK_LENGTH] + "... [truncated]"
        
        # For recommendation mode, include the arXiv ID in the chunk text
        if search_mode == "recommendation":
            arxiv_id = chunk.get('arxiv_id', 'unknown')
            chunk_texts += CHUNK_TEXT_TEMPLATE.format(
                index=i+1, 
                text=f"ArXiv ID: {arxiv_id}\n{chunk_text}"
            )
        else:
            chunk_texts += CHUNK_TEXT_TEMPLATE.format(index=i+1, text=chunk_text)
    
    user_prompt = USER_PROMPT_TEMPLATE.format(enhanced_query=enhanced_query, chunk_texts=chunk_texts)
    
    return {"system": system_prompt, "user": user_prompt}


def get_openai_api_key():
    """Get OpenAI API key from environment variable."""
    # Get from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
        
    return api_key


def sanitize_json_string(json_str: str) -> str:
    """
    Sanitize a JSON string to handle problematic Unicode escape sequences.
    
    Args:
        json_str: The JSON string to sanitize
        
    Returns:
        Sanitized JSON string
    """
    # Replace problematic escape sequences
    # This handles cases where \u is followed by invalid hex digits
    import re
    
    # Replace invalid \uXXXX escape sequences with a placeholder
    sanitized = re.sub(r'\\u(?![0-9a-fA-F]{4})[0-9a-fA-F]{0,3}', lambda m: f'\\\\u{m.group(0)[2:]}', json_str)
    
    # Replace any remaining problematic escape sequences
    sanitized = sanitized.replace('\\', '\\\\').replace('\\"', '\\\\"')
    sanitized = re.sub(r'(?<!\\)\\(?!["\\\/bfnrtu])', r'\\\\', sanitized)
    
    return sanitized


async def process_chunk_batch_async(chunks: List[Dict[str, Any]], enhanced_query: str, search_mode: str = "answer") -> List[Dict[str, Any]]:
    """
    Process a batch of chunks asynchronously using the LLM API.
    
    Args:
        chunks: List of chunks to process
        enhanced_query: Enhanced query text
        search_mode: Search mode ("answer" or "recommendation")
        
    Returns:
        List of chunks with summaries and relevance scores
    """
    # Create prompt for the batch based on search mode
    prompt = create_llm_prompt(chunks, enhanced_query, search_mode=search_mode)
    
    # Call LLM API
    try:
        # Use OpenAI API to generate summaries and relevance scores
        api_key = get_openai_api_key()
        client = AsyncOpenAI(api_key=api_key)
        
        logger.info(f"Sending batch of {len(chunks)} chunks to LLM for summarization")
        
        # Implement retry logic with exponential backoff
        max_retries = LLM_RETRY_ATTEMPTS
        retry_count = 0
        backoff_time = 1  # Start with 1 second backoff
        
        while retry_count <= max_retries:
            try:
                response = await client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                    ],
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    timeout=LLM_TIMEOUT
                )
                break  # If successful, break out of the retry loop
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise  # Re-raise the exception if we've exceeded max retries
                
                # Log the error and retry after backoff
                logger.warning(f"LLM API call failed (attempt {retry_count}/{max_retries}): {str(e)}")
                logger.info(f"Retrying after {backoff_time} seconds...")
                await asyncio.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
        
        # Parse the response
        try:
            # Extract the JSON content from the response
            content = response.choices[0].message.content
            logger.info(f"Received response from LLM, parsing JSON")
            
            # Try to parse the JSON content
            try:
                # Sanitize the JSON string before parsing
                sanitized_content = sanitize_json_string(content)
                llm_results = json.loads(sanitized_content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the text
                # Sometimes LLMs wrap JSON in markdown or add explanatory text
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|^\s*(\[[\s\S]*\]|\{[\s\S]*\})\s*$', content)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    # Sanitize the extracted JSON string
                    sanitized_json_str = sanitize_json_string(json_str)
                    llm_results = json.loads(sanitized_json_str)
                else:
                    raise ValueError("Could not extract JSON from LLM response")
            
            # Process the response based on its format and search mode
            summarized_chunks = []
            
            # Handle recommendation mode differently
            if search_mode == "recommendation":
                # For recommendation mode, we expect a single dictionary with tweet, relevance_score, and paper_ids
                if isinstance(llm_results, dict):
                    logger.info("Processing recommendation mode response")
                    
                    # Create a copy of each chunk to return
                    for chunk in chunks:
                        summarized_chunk = chunk.copy()
                        
                        # Extract tweet text
                        tweet = llm_results.get("tweet", "")
                        if not tweet or not isinstance(tweet, str):
                            logger.warning("Invalid or missing tweet in recommendation result")
                            tweet = f"Failed to generate a tweet for this batch of chunks"
                        
                        # Extract relevance score
                        score = llm_results.get("relevance_score", None)
                        if score is None or not isinstance(score, (int, float)) or score < 0 or score > 1:
                            logger.warning(f"Invalid relevance score in recommendation result: {score}")
                            score = chunk.get('normalized_weight', LLM_DEFAULT_SCORE)
                        
                        # Extract paper IDs
                        paper_ids = llm_results.get("paper_ids", [])
                        if not isinstance(paper_ids, list):
                            logger.warning(f"Invalid paper_ids format: {paper_ids}")
                            paper_ids = [chunk.get('arxiv_id', 'unknown')]
                        
                        # Store the tweet information in the chunk
                        summarized_chunk["tweet"] = tweet
                        summarized_chunk["relevance_score"] = score
                        summarized_chunk["paper_ids"] = paper_ids
                        
                        # For compatibility with answer mode, also store as llm_summary and llm_relevance_score
                        summarized_chunk["llm_summary"] = tweet
                        summarized_chunk["llm_relevance_score"] = score
                        
                        summarized_chunks.append(summarized_chunk)
                else:
                    # Unexpected format for recommendation mode
                    logger.error(f"Unexpected response format for recommendation mode: {type(llm_results)}")
                    for chunk in chunks:
                        summarized_chunk = chunk.copy()
                        summarized_chunk["tweet"] = f"LLM returned unexpected format: {type(llm_results)}"
                        summarized_chunk["relevance_score"] = chunk.get('normalized_weight', LLM_DEFAULT_SCORE)
                        summarized_chunk["paper_ids"] = [chunk.get('arxiv_id', 'unknown')]
                        summarized_chunk["llm_summary"] = summarized_chunk["tweet"]
                        summarized_chunk["llm_relevance_score"] = summarized_chunk["relevance_score"]
                        summarized_chunks.append(summarized_chunk)
            else:
                # Handle answer mode (original behavior)
                # Handle different response formats
                if isinstance(llm_results, list):
                    # Response is a list of objects (expected format)
                    logger.info(f"LLM returned a list with {len(llm_results)} items")
                    for i, chunk in enumerate(chunks):
                        summarized_chunk = chunk.copy()
                        if i < len(llm_results):
                            # Extract data from the corresponding result
                            result = llm_results[i]
                            if isinstance(result, dict):
                                # Extract summary and score with validation
                                summary = result.get("summary", "")
                                if not summary or not isinstance(summary, str):
                                    logger.warning(f"Invalid or missing summary for chunk {i+1}")
                                    summary = f"LLM returned invalid summary format for chunk {i+1}"
                                
                                # Extract and validate relevance score
                                score = result.get("relevance_score", None)
                                if score is None or not isinstance(score, (int, float)) or score < 0 or score > 1:
                                    logger.warning(f"Invalid relevance score for chunk {i+1}: {score}")
                                    # Use the normalized weight as a fallback instead of a default score
                                    # This preserves the original ranking rather than introducing a random value
                                    score = chunk.get('normalized_weight', LLM_DEFAULT_SCORE)
                                
                                summarized_chunk["llm_summary"] = summary
                                summarized_chunk["llm_relevance_score"] = score
                            else:
                                logger.warning(f"Result for chunk {i+1} is not a dictionary: {result}")
                                summarized_chunk["llm_summary"] = f"LLM returned invalid format for chunk {i+1}"
                                summarized_chunk["llm_relevance_score"] = chunk.get('normalized_weight', LLM_DEFAULT_SCORE)
                        else:
                            # LLM returned fewer results than expected
                            logger.warning(f"LLM returned fewer results than expected. Using original weight for chunk {i+1}")
                            summarized_chunk["llm_summary"] = f"No summary generated for this chunk"
                            summarized_chunk["llm_relevance_score"] = chunk.get('normalized_weight', LLM_DEFAULT_SCORE)
                        
                        summarized_chunks.append(summarized_chunk)
                
                elif isinstance(llm_results, dict):
                    # Response is a single object (sometimes happens with single chunk batches)
                    logger.info("LLM returned a single dictionary object")
                    if len(chunks) == 1:
                        # If we only sent one chunk, this is fine
                        summarized_chunk = chunks[0].copy()
                        
                        # Extract and validate summary
                        summary = llm_results.get("summary", "")
                        if not summary or not isinstance(summary, str):
                            logger.warning("Invalid or missing summary in single result")
                            summary = "LLM returned invalid summary format"
                        
                        # Extract and validate relevance score
                        score = llm_results.get("relevance_score", None)
                        if score is None or not isinstance(score, (int, float)) or score < 0 or score > 1:
                            logger.warning(f"Invalid relevance score in single result: {score}")
                            score = summarized_chunk.get('normalized_weight', LLM_DEFAULT_SCORE)
                        
                        summarized_chunk["llm_summary"] = summary
                        summarized_chunk["llm_relevance_score"] = score
                        summarized_chunks.append(summarized_chunk)
                    else:
                        # If we sent multiple chunks but got a single result, use it for the first chunk
                        # and use normalized weights for the rest
                        logger.warning(f"LLM returned a single result for {len(chunks)} chunks")
                        for i, chunk in enumerate(chunks):
                            summarized_chunk = chunk.copy()
                            if i == 0:
                                # Use the single result for the first chunk
                                summary = llm_results.get("summary", "")
                                if not summary or not isinstance(summary, str):
                                    summary = "LLM returned invalid summary format"
                                
                                score = llm_results.get("relevance_score", None)
                                if score is None or not isinstance(score, (int, float)) or score < 0 or score > 1:
                                    score = chunk.get('normalized_weight', LLM_DEFAULT_SCORE)
                                
                                summarized_chunk["llm_summary"] = summary
                                summarized_chunk["llm_relevance_score"] = score
                            else:
                                # Use normalized weights for other chunks
                                summarized_chunk["llm_summary"] = f"LLM did not return a summary for this chunk"
                                summarized_chunk["llm_relevance_score"] = chunk.get('normalized_weight', LLM_DEFAULT_SCORE)
                            
                            summarized_chunks.append(summarized_chunk)
                else:
                    # Unexpected response format
                    logger.error(f"Unexpected response format from LLM: {type(llm_results)}")
                    for chunk in chunks:
                        summarized_chunk = chunk.copy()
                        summarized_chunk["llm_summary"] = f"LLM returned unexpected format: {type(llm_results)}"
                        summarized_chunk["llm_relevance_score"] = chunk.get('normalized_weight', LLM_DEFAULT_SCORE)
                        summarized_chunks.append(summarized_chunk)
            
            return summarized_chunks
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Raw response content: {response.choices[0].message.content if hasattr(response, 'choices') else 'No content'}")
            
            # Apply fallback strategy for failed parsing - use normalized weights
            summarized_chunks = []
            for chunk in chunks:
                summarized_chunk = chunk.copy()
                summarized_chunk["llm_summary"] = f"Failed to parse LLM response: {str(e)[:100]}"
                # Use normalized weight instead of default score to preserve ranking
                summarized_chunk["llm_relevance_score"] = chunk.get('normalized_weight', LLM_DEFAULT_SCORE)
                summarized_chunks.append(summarized_chunk)
            
            return summarized_chunks
        
    except Exception as e:
        logger.error(f"Error processing batch with LLM: {e}")
        
        # Apply fallback strategy for failed batch
        summarized_chunks = []
        for chunk in chunks:
            summarized_chunk = chunk.copy()
            summarized_chunk["llm_summary"] = f"Failed to generate summary: {str(e)[:100]}"
            summarized_chunk["llm_relevance_score"] = chunk.get('normalized_weight', LLM_DEFAULT_SCORE)
            summarized_chunks.append(summarized_chunk)
        
        return summarized_chunks


async def process_chunks_with_llm(chunks: List[Dict[str, Any]], enhanced_query: str, search_mode: str = "answer", batch_size: int = BATCH_SIZE) -> List[Dict[str, Any]]:
    """
    Process chunks with LLM in batches.
    
    Args:
        chunks: List of chunks to process
        enhanced_query: Enhanced query text
        search_mode: Search mode ("answer" or "recommendation")
        batch_size: Number of chunks per batch
        
    Returns:
        List of chunks with summaries and relevance scores
    """
    logger.info(f"Processing {len(chunks)} chunks with LLM in {search_mode} mode")
    
    # For recommendation mode, create similarity-based batches
    if search_mode == "recommendation":
        # Use similarity-based clustering to form batches
        batches = create_similarity_batches(chunks, batch_size, SIMILARITY_THRESHOLD_RECOMMENDATION)
        logger.info(f"Created {len(batches)} similarity-based batches for recommendation mode")
    else:
        # For answer mode, use the original simple batching
        batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
        logger.info(f"Split into {len(batches)} sequential batches for answer mode")
    
    # Process batches with rate limiting
    summarized_chunks = []
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)}")
        
        # Add a small delay between batches to avoid rate limiting
        if i > 0:
            await asyncio.sleep(LLM_BATCH_DELAY)
        
        # Process the batch with the appropriate mode
        batch_result = await process_chunk_batch_async(batch, enhanced_query, search_mode=search_mode)
        
        # For recommendation mode, we need to handle the tweet format differently
        if search_mode == "recommendation" and batch_result:
            # In recommendation mode, the first chunk in the batch contains the tweet
            # We need to mark all chunks in the batch as being part of this tweet
            for chunk in batch_result:
                # Store the batch ID (using the first chunk's ID as the batch ID)
                chunk["batch_id"] = batch_result[0].get("id", f"batch_{i}")
                # Store the tweet text in all chunks of this batch
                if "tweet" in batch_result[0]:
                    chunk["tweet"] = batch_result[0]["tweet"]
                    chunk["tweet_relevance_score"] = batch_result[0].get("relevance_score", 0.0)
                    chunk["paper_ids"] = batch_result[0].get("paper_ids", [])
        
        summarized_chunks.extend(batch_result)
    
    logger.info(f"LLM processing complete for {len(summarized_chunks)} chunks")
    return summarized_chunks


def compute_final_weights(
    summarized_chunks: List[Dict[str, Any]], 
    search_mode: str = "answer"
) -> List[Dict[str, Any]]:
    """
    Compute final adjusted weights based on LLM relevance scores and original weights.
    
    Args:
        summarized_chunks: List of chunks with summaries and relevance scores
        search_mode: Search mode ("answer" or "recommendation")
        
    Returns:
        List of chunks with final adjusted weights
    """
    logger.info(f"Computing final adjusted weights for {search_mode} mode...")
    
    # Select the appropriate lambda weight based on search mode
    if search_mode == "recommendation":
        lambda_weight = LAMBDA_RECOMMENDATION
    else:
        lambda_weight = LAMBDA_ANSWER
    
    logger.info(f"Using lambda weight of {lambda_weight} for {search_mode} mode")
    
    if search_mode == "recommendation":
        # For recommendation mode, we need to handle tweets
        # Group chunks by batch_id (tweets)
        batch_groups = {}
        for chunk in summarized_chunks:
            batch_id = chunk.get("batch_id", None)
            if batch_id:
                if batch_id not in batch_groups:
                    batch_groups[batch_id] = []
                batch_groups[batch_id].append(chunk)
        
        # Process each batch (tweet)
        for batch_id, batch_chunks in batch_groups.items():
            if not batch_chunks:
                continue
                
            # Calculate the average normalized weight of chunks in this batch
            avg_normalized_weight = sum(chunk.get("normalized_weight", 0) for chunk in batch_chunks) / len(batch_chunks)
            
            # Get the tweet relevance score (should be the same for all chunks in the batch)
            tweet_relevance_score = batch_chunks[0].get("relevance_score", batch_chunks[0].get("llm_relevance_score", 0))
            
            # Calculate the final weight for this tweet
            tweet_final_weight = (
                lambda_weight * tweet_relevance_score + 
                (1 - lambda_weight) * avg_normalized_weight
            )
            
            # Assign this final weight to all chunks in the batch
            for chunk in batch_chunks:
                chunk["final_weight_adjusted"] = tweet_final_weight
                # Also store the component weights for reference
                chunk["tweet_relevance_score"] = tweet_relevance_score
                chunk["avg_normalized_weight"] = avg_normalized_weight
                # Store the lambda used for reference
                chunk["lambda_weight"] = lambda_weight
    else:
        # Original answer mode behavior
        for chunk in summarized_chunks:
            # Compute final adjusted weight
            chunk["final_weight_adjusted"] = (
                lambda_weight * chunk["llm_relevance_score"] + 
                (1 - lambda_weight) * chunk["normalized_weight"]
            )
            # Store the lambda used for reference
            chunk["lambda_weight"] = lambda_weight
    
    # Sort by final adjusted weight (descending)
    summarized_chunks.sort(key=lambda x: x.get("final_weight_adjusted", 0), reverse=True)
    
    logger.info(f"Computed final adjusted weights for {len(summarized_chunks)} chunks")
    return summarized_chunks


def save_summarized_chunks_to_db(
    summarized_chunks: List[Dict[str, Any]],
    queries: Dict[str, str],
    db_path: str = SUMMARIZED_CHUNKS_DB_PATH
) -> None:
    """
    Save summarized chunks to a SQLite database.
    
    Args:
        summarized_chunks: List of chunks with summaries and adjusted weights
        queries: Dictionary of query texts by type
        db_path: Path to save the database
    """
    try:
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Remove existing database if it exists
        if os.path.exists(db_path):
            logger.info(f"Removing existing summarized chunks database: {db_path}")
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
            CREATE TABLE IF NOT EXISTS summarized_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                arxiv_id TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                llm_summary TEXT NOT NULL,
                llm_relevance_score REAL NOT NULL,
                normalized_weight REAL NOT NULL,
                final_weight_adjusted REAL NOT NULL,
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
        
        # Insert summarized chunks
        for chunk in summarized_chunks:
            # Use the enhanced query ID for all chunks
            query_id = query_ids.get('enhanced', list(query_ids.values())[0])
            
            try:
                cursor.execute("""
                    INSERT INTO summarized_chunks (
                        query_id, chunk_id, arxiv_id, chunk_idx,
                        llm_summary, llm_relevance_score,
                        normalized_weight, final_weight_adjusted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id, chunk['chunk_id'], chunk['arxiv_id'], chunk['chunk_idx'],
                    chunk['llm_summary'], chunk['llm_relevance_score'],
                    chunk['normalized_weight'], chunk['final_weight_adjusted']
                ))
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    # Update existing entry instead of inserting
                    logger.debug(f"Updating existing chunk: {chunk['chunk_id']}")
                    cursor.execute("""
                        UPDATE summarized_chunks SET
                            llm_summary = ?, llm_relevance_score = ?,
                            normalized_weight = ?, final_weight_adjusted = ?
                        WHERE query_id = ? AND chunk_id = ?
                    """, (
                        chunk['llm_summary'], chunk['llm_relevance_score'],
                        chunk['normalized_weight'], chunk['final_weight_adjusted'],
                        query_id, chunk['chunk_id']
                    ))
                else:
                    raise
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(summarized_chunks)} summarized chunks to database: {db_path}")
        
    except sqlite3.Error as e:
        logger.error(f"Database error when saving summarized chunks: {e}")
        raise ValueError(f"Database error: {e}")


def save_summarized_chunks_to_json(
    summarized_chunks: List[Dict[str, Any]],
    queries: Dict[str, str],
    output_path: str = SUMMARIZED_CHUNKS_JSON_PATH,
    search_mode: str = "answer"
) -> None:
    """
    Save summarized chunks to a JSON file.
    
    Args:
        summarized_chunks: List of chunks with summaries and adjusted weights
        queries: Dictionary of query texts by type
        output_path: Path to save the JSON file
        search_mode: Search mode ("answer" or "recommendation")
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for JSON
        output_data = {
            "queries": queries,
            "summarized_chunks": summarized_chunks,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": {
                "num_chunks": len(summarized_chunks),
                "parameters": {
                    "search_mode": search_mode,
                    "k_summary": K_SUMMARY_RECOMMENDATION if search_mode == "recommendation" else K_SUMMARY_ANSWER,
                    "summary_length": SUMMARY_LENGTH,
                    "lambda": LAMBDA_RECOMMENDATION if search_mode == "recommendation" else LAMBDA_ANSWER,
                    "similarity_threshold": SIMILARITY_THRESHOLD_RECOMMENDATION if search_mode == "recommendation" else SIMILARITY_THRESHOLD_ANSWER,
                    "batch_size": BATCH_SIZE,
                    "llm_model": LLM_MODEL
                }
            }
        }
        
        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved summarized chunks to JSON file: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving summarized chunks to JSON: {e}")
        raise ValueError(f"JSON save error: {e}")


async def run_chunk_llm_weight_determination_async(search_mode: str = "answer") -> List[Dict[str, Any]]:
    """
    Run the chunk LLM weight determination process asynchronously.
    
    Args:
        search_mode: Search mode ("answer" or "recommendation")
    
    Returns:
        List of summarized chunks sorted by final adjusted weight
    """
    start_time = time.time()
    logger.info(f"Starting chunk LLM weight determination process in {search_mode} mode...")
    
    # Get weighted chunks from database
    weighted_chunks, queries = get_weighted_chunks()
    
    # Get enhanced query
    enhanced_query = queries.get('enhanced', list(queries.values())[0])
    
    # Select top-k chunks based on search mode
    k_summary = K_SUMMARY_RECOMMENDATION if search_mode == "recommendation" else K_SUMMARY_ANSWER
    top_k_chunks = weighted_chunks[:k_summary]
    logger.info(f"Selected top {len(top_k_chunks)} chunks for LLM processing in {search_mode} mode")
    
    # For answer mode, deduplicate chunks
    # For recommendation mode, we'll handle similarity in the batching process
    if search_mode == "answer":
        logger.info("Deduplicating chunks for answer mode")
        processed_chunks = deduplicate_chunks(top_k_chunks, SIMILARITY_THRESHOLD_ANSWER)
    else:
        logger.info("Skipping deduplication for recommendation mode - will use similarity-based batching instead")
        processed_chunks = top_k_chunks
    
    # Process chunks with LLM using the appropriate mode
    summarized_chunks = await process_chunks_with_llm(processed_chunks, enhanced_query, search_mode=search_mode)
    
    # Compute final adjusted weights
    summarized_chunks = compute_final_weights(summarized_chunks, search_mode=search_mode)
    
    # Save results
    save_summarized_chunks_to_db(summarized_chunks, queries)
    save_summarized_chunks_to_json(summarized_chunks, queries, search_mode=search_mode)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Chunk LLM weight determination completed in {elapsed_time:.2f} seconds for {search_mode} mode")
    
    return summarized_chunks


def run_chunk_llm_weight_determination(search_mode: str = "answer") -> List[Dict[str, Any]]:
    """
    Run the chunk LLM weight determination process.
    
    Args:
        search_mode: Search mode ("answer" or "recommendation")
        
    Returns:
        List of summarized chunks sorted by final adjusted weight
    """
    # Run the async function in the event loop
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_chunk_llm_weight_determination_async(search_mode))


if __name__ == "__main__":
    run_chunk_llm_weight_determination()
