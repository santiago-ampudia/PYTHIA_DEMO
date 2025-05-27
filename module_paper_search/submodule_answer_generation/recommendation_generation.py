"""
module_paper_search/submodule_answer_generation/recommendation_generation.py

This module handles the generation and evaluation of tweet-style recommendations
using the top-n summarized chunks from the previous pipeline steps.
"""

import logging
import json
import os
import datetime
import copy
from typing import Dict, List, Optional, Tuple, Any, Set

# Import parameters
from module_paper_search.submodule_answer_generation.recommendation_generation_parameters import (
    N_RECOMMENDATIONS,
    RECOMMENDATION_QUALITY_THRESHOLD,
    OUTPUT_DIR,
    MAX_RETRY_ATTEMPTS,
    DEFAULT_OPENAI_API_KEY,
    TWEET_TEXT_KEY,
    PAPER_IDS_KEY,
    BATCH_ID_KEY
)

from module_paper_search.submodule_chunk_weight_determination.chunk_weight_determination_parameters import (
    ENHANCED_QUERY_WEIGHT_RECOMMENDATION,
    SUBTOPIC_QUERY_WEIGHT_RECOMMENDATION,
    TOPIC_QUERY_WEIGHT_RECOMMENDATION,
    ENHANCED_QUERY_WEIGHT_RECOMMENDATION_RETRY,
    SUBTOPIC_QUERY_WEIGHT_RECOMMENDATION_RETRY,
    TOPIC_QUERY_WEIGHT_RECOMMENDATION_RETRY,
    METADATA_BOOST_FACTOR_RECOMMENDATION,
    METADATA_THRESHOLD_ENHANCED_RECOMMENDATION,
    METADATA_THRESHOLD_SUBTOPIC_RECOMMENDATION,
    METADATA_THRESHOLD_TOPIC_RECOMMENDATION
)

from module_paper_search.submodule_chunk_weight_determination.chunk_weight_determination import (
    determine_chunk_weights
)

from module_paper_search.submodule_chunk_weight_determination.chunk_llm_weight_determination import (
    run_chunk_llm_weight_determination_async
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('recommendation_generation')


def select_top_tweets(summarized_chunks: List[Dict[str, Any]], n: int = N_RECOMMENDATIONS) -> List[Dict[str, Any]]:
    """
    Select the top n tweets from summarized_chunks based on final_weight_adjusted.
    
    Args:
        summarized_chunks: List of dictionaries containing chunk information
        n: Number of top tweets to select
        
    Returns:
        List of the top n tweets (represented by their first chunk)
    """
    logger.info(f"Selecting top {n} tweets from {len(summarized_chunks)} chunks")
    
    # Group chunks by batch_id (tweets)
    batch_groups = {}
    for chunk in summarized_chunks:
        batch_id = chunk.get("batch_id", None)
        if batch_id:
            if batch_id not in batch_groups:
                batch_groups[batch_id] = []
            batch_groups[batch_id].append(chunk)
    
    # Get the first chunk from each batch (all chunks in a batch have the same weight)
    tweet_representatives = []
    for batch_id, batch_chunks in batch_groups.items():
        if batch_chunks:
            # Sort by final_weight_adjusted (descending) within the batch
            batch_chunks.sort(key=lambda x: x.get("final_weight_adjusted", 0), reverse=True)
            tweet_representatives.append(batch_chunks[0])
    
    # Sort tweet representatives by final_weight_adjusted (descending)
    tweet_representatives.sort(key=lambda x: x.get("final_weight_adjusted", 0), reverse=True)
    
    # Select top n
    top_tweets = tweet_representatives[:n]
    
    logger.info(f"Selected top {len(top_tweets)} tweets")
    return top_tweets


def get_high_quality_tweets(summarized_chunks: List[Dict[str, Any]], 
                           n: int = N_RECOMMENDATIONS, 
                           threshold: float = RECOMMENDATION_QUALITY_THRESHOLD) -> List[Dict[str, Any]]:
    """
    Get tweets that have a final_weight_adjusted above the threshold.
    
    Args:
        summarized_chunks: List of dictionaries containing chunk information
        n: Maximum number of tweets to return
        threshold: Minimum final_weight_adjusted score for a high-quality tweet
        
    Returns:
        List of high-quality tweets
    """
    # Get the top n tweets
    top_tweets = select_top_tweets(summarized_chunks, n)
    
    # Filter for those above the threshold
    high_quality_tweets = [tweet for tweet in top_tweets if tweet.get("final_weight_adjusted", 0) >= threshold]
    
    logger.info(f"Found {len(high_quality_tweets)} high-quality tweets (score >= {threshold})")
    return high_quality_tweets


async def retry_with_different_weights(original_chunks: List[Dict[str, Any]], 
                                      queries: Dict[str, str], 
                                      used_chunk_ids: Set[str]) -> List[Dict[str, Any]]:
    """
    Retry the chunk weight determination with different weights and excluding used chunks.
    
    Args:
        original_chunks: Original chunks from the similarity search
        queries: Dictionary containing enhanced_query, subtopic_query, and topic_query
        used_chunk_ids: Set of chunk IDs that have already been used
        
    Returns:
        List of newly processed chunks with adjusted weights
    """
    logger.info("Retrying with different weights and excluding used chunks")
    
    # Filter out used chunks
    # The original_chunks from similarity search might have different key structure
    # than the summarized chunks, so we need to be careful with the filtering
    filtered_chunks = []
    for chunk in original_chunks:
        # Check if this chunk has already been used
        # The ID might be in different keys depending on the source
        chunk_id = chunk.get("chunk_id", chunk.get("id", None))
        if chunk_id and chunk_id not in used_chunk_ids:
            filtered_chunks.append(chunk)
    
    if len(filtered_chunks) == 0:
        logger.warning("No unused chunks available for retry")
        return []
    
    logger.info(f"Using {len(filtered_chunks)} unused chunks for retry")
    
    # Determine weights with retry parameters
    weighted_chunks = determine_chunk_weights(
        filtered_chunks, 
        queries["enhanced_query"], 
        queries["subtopic_query"], 
        queries["topic_query"],
        enhanced_query_weight=ENHANCED_QUERY_WEIGHT_RECOMMENDATION_RETRY,
        subtopic_query_weight=SUBTOPIC_QUERY_WEIGHT_RECOMMENDATION_RETRY,
        topic_query_weight=TOPIC_QUERY_WEIGHT_RECOMMENDATION_RETRY,
        metadata_boost_factor=METADATA_BOOST_FACTOR_RECOMMENDATION,
        metadata_threshold_enhanced=METADATA_THRESHOLD_ENHANCED_RECOMMENDATION,
        metadata_threshold_subtopic=METADATA_THRESHOLD_SUBTOPIC_RECOMMENDATION,
        metadata_threshold_topic=METADATA_THRESHOLD_TOPIC_RECOMMENDATION
    )
    
    # Process with LLM again
    summarized_chunks = await run_chunk_llm_weight_determination_async(search_mode="recommendation")
    
    logger.info(f"Retry produced {len(summarized_chunks)} summarized chunks")
    return summarized_chunks


def save_recommendations(recommendations: List[Dict[str, Any]], 
                        enhanced_query: str,
                        output_dir: str = OUTPUT_DIR) -> None:
    """
    Save the generated recommendations to a file.
    
    Args:
        recommendations: List of recommendation tweets
        enhanced_query: The query that was used
        output_dir: Directory to save the recommendations in
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filenames as requested
    tweets_txt_filename = os.path.join(output_dir, "tweets.txt")
    tweets_json_filename = os.path.join(output_dir, "tweets.json")
    
    # Prepare data to save in JSON
    output_data = {
        "query": enhanced_query,
        "timestamp": str(datetime.datetime.now()),
        "recommendations": []
    }
    
    # Prepare text content for tweets.txt
    text_content = [f"Query: {enhanced_query}\n\nRecommendations:\n"]
    
    # Add each recommendation
    for i, tweet in enumerate(recommendations):
        tweet_text = tweet.get("tweet", tweet.get(TWEET_TEXT_KEY, ""))
        paper_ids = tweet.get(PAPER_IDS_KEY, [])
        score = tweet.get("final_weight_adjusted", 0)
        
        # Add to JSON structure
        recommendation_data = {
            "tweet_text": tweet_text,
            "score": score,
            "paper_ids": paper_ids,
            "batch_id": tweet.get(BATCH_ID_KEY, ""),
            "component_scores": {
                "tweet_relevance_score": tweet.get("tweet_relevance_score", 0),
                "avg_normalized_weight": tweet.get("avg_normalized_weight", 0),
                "lambda_weight": tweet.get("lambda_weight", 0)
            }
        }
        output_data["recommendations"].append(recommendation_data)
        
        # Add to text content
        text_content.append(f"[{i+1}] {tweet_text}")
        if paper_ids:
            text_content.append(f"    Papers: {', '.join(paper_ids)}")
        text_content.append(f"    Score: {score:.4f}\n")
    
    # Save to JSON file
    with open(tweets_json_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save to text file
    with open(tweets_txt_filename, 'w') as f:
        f.write('\n'.join(text_content))
    
    logger.info(f"Saved {len(recommendations)} recommendations to {tweets_txt_filename} and {tweets_json_filename}")


async def run_recommendation_generation(
    summarized_chunks: List[Dict[str, Any]],
    original_chunks: List[Dict[str, Any]],
    queries: Dict[str, str],
    n_recommendations: int = N_RECOMMENDATIONS,
    quality_threshold: float = RECOMMENDATION_QUALITY_THRESHOLD,
    output_dir: str = OUTPUT_DIR,
    max_retry_attempts: int = MAX_RETRY_ATTEMPTS
) -> List[Dict[str, Any]]:
    """
    Main function to run the recommendation generation process.
    
    Args:
        summarized_chunks: List of dictionaries containing processed chunks with tweets
        original_chunks: Original chunks from similarity search (for retry)
        queries: Dictionary containing enhanced_query, subtopic_query, and topic_query
        n_recommendations: Number of recommendations to generate
        quality_threshold: Minimum quality score for recommendations
        output_dir: Directory to save the recommendations in
        
    Returns:
        List of recommendation tweets
    """
    logger.info("Starting recommendation generation process")
    
    # Get high-quality tweets
    high_quality_tweets = get_high_quality_tweets(
        summarized_chunks, 
        n=n_recommendations, 
        threshold=quality_threshold
    )
    
    # If we have enough high-quality tweets, use them
    if len(high_quality_tweets) >= n_recommendations:
        logger.info(f"Found {len(high_quality_tweets)} high-quality tweets, using top {n_recommendations}")
        final_recommendations = high_quality_tweets[:n_recommendations]
    else:
        # Not enough high-quality tweets, need to retry with different weights
        logger.info(f"Only found {len(high_quality_tweets)} high-quality tweets, retrying with different weights")
        
        # Get the chunk IDs that have already been used
        used_chunk_ids = set()
        for chunk in summarized_chunks:
            batch_id = chunk.get("batch_id", None)
            if batch_id:
                # For each batch that resulted in a high-quality tweet, mark all its chunks as used
                for high_quality_tweet in high_quality_tweets:
                    if high_quality_tweet.get("batch_id", None) == batch_id:
                        # Get the chunk ID, which might be stored in different keys
                        chunk_id = chunk.get("chunk_id", chunk.get("id", ""))
                        if chunk_id:
                            used_chunk_ids.add(chunk_id)
        
        logger.info(f"Marked {len(used_chunk_ids)} chunks as used")
        
        # Retry with different weights
        retry_chunks = await retry_with_different_weights(
            original_chunks, 
            queries, 
            used_chunk_ids
        )
        
        # Get high-quality tweets from retry
        retry_high_quality_tweets = get_high_quality_tweets(
            retry_chunks, 
            n=n_recommendations - len(high_quality_tweets), 
            threshold=quality_threshold
        )
        
        # Combine original high-quality tweets with retry high-quality tweets
        final_recommendations = high_quality_tweets + retry_high_quality_tweets
        
        # If we still don't have enough, just use the best ones we have
        if len(final_recommendations) < n_recommendations:
            logger.info(f"Still only have {len(final_recommendations)} high-quality tweets after retry")
            
            # Get all tweets from both original and retry
            all_tweets = select_top_tweets(summarized_chunks + retry_chunks, n=n_recommendations)
            
            # Filter out those that are already in final_recommendations
            existing_batch_ids = {tweet.get("batch_id", "") for tweet in final_recommendations}
            additional_tweets = [
                tweet for tweet in all_tweets 
                if tweet.get("batch_id", "") not in existing_batch_ids
            ]
            
            # Add as many as needed to reach n_recommendations
            final_recommendations.extend(
                additional_tweets[:n_recommendations - len(final_recommendations)]
            )
    
    # Save the recommendations
    save_recommendations(
        final_recommendations,
        queries["enhanced_query"],
        output_dir
    )
    
    logger.info(f"Generated {len(final_recommendations)} recommendations")
    return final_recommendations
