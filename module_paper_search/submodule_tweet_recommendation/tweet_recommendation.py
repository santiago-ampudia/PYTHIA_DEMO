"""
submodule_tweet_recommendation/tweet_recommendation.py

This module implements Tweet Recommendation generation based on the chunks selected by the
chunk similarity selection process for recommendation mode.

Purpose: Generate tweet-style recommendations for each of the five specialized queries
about a GitHub repository. Each tweet will be based on the content of the selected chunks
and will include proper citations to the source papers.

Process:
    1. For each of the five queries:
        a. Format the chunks into a structured input
        b. Call the OpenAI API with the query and chunks
        c. Parse the response to extract the generated tweets
        d. Save the tweets to a JSON file
    2. Combine all tweets into a single JSON file

Output: JSON files containing tweet recommendations for each query
"""

import os
import json
import logging
import re
from typing import List, Dict, Any
from openai import OpenAI
import datetime
from pathlib import Path

# Import parameters from the parameters file
from .tweet_recommendation_parameters import (
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    NUM_TWEETS,
    SYSTEM_PROMPT,
    USER_PROMPT,
    TWEETS_ARCHITECTURE_PATH,
    TWEETS_TECHNICAL_PATH,
    TWEETS_ALGORITHMIC_PATH,
    TWEETS_DOMAIN_PATH,
    TWEETS_INTEGRATION_PATH,
    TWEETS_ALL_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tweet_recommendation')


def get_openai_api_key() -> str:
    """
    Get the OpenAI API key from environment variables.
    
    Returns:
        str: The OpenAI API key
    """
    # Get from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
        
    return api_key


def format_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Format chunks into a structured text for the prompt.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        str: Formatted chunks text
    """
    formatted_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Extract key information
        arxiv_id = chunk.get("arxiv_id", "unknown")
        title = chunk.get("title", "Untitled")
        authors = chunk.get("authors", "Unknown")
        chunk_text = chunk.get("chunk_text", "").strip()
        
        # Format the chunk
        formatted_chunk = f"CHUNK {i+1} [ID: {arxiv_id}]\n"
        formatted_chunk += f"Title: {title}\n"
        formatted_chunk += f"Authors: {authors}\n"
        formatted_chunk += f"Content: {chunk_text}\n"
        
        formatted_chunks.append(formatted_chunk)
    
    return "\n\n".join(formatted_chunks)


def generate_tweets(query: str, chunks: List[Dict[str, Any]], num_tweets: int = NUM_TWEETS) -> List[str]:
    """
    Generate tweet recommendations based on the query and chunks.
    
    Args:
        query: The specialized query about the repository
        chunks: List of chunk dictionaries
        num_tweets: Number of tweets to generate
        
    Returns:
        List[str]: Generated tweets
    """
    logger.info(f"Generating {num_tweets} tweets for query: {query[:50]}...")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_openai_api_key())
        
        # Format chunks for the prompt
        formatted_chunks = format_chunks_for_prompt(chunks)
        
        # Format the prompts
        formatted_system_prompt = SYSTEM_PROMPT.format(num_tweets=num_tweets)
        formatted_user_prompt = USER_PROMPT.format(
            query=query,
            chunks=formatted_chunks,
            num_tweets=num_tweets
        )
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": formatted_user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # Extract the response content
        content = response.choices[0].message.content
        
        # Parse the tweets
        tweets = []
        tweet_pattern = r"Tweet \d+:(.*?)(?=Tweet \d+:|$)"
        matches = re.finditer(tweet_pattern, content, re.DOTALL)
        
        for match in matches:
            tweet_text = match.group(1).strip()
            tweets.append(tweet_text)
        
        # If no matches found, try to split by newlines
        if not tweets:
            lines = content.split('\n')
            current_tweet = ""
            
            for line in lines:
                if line.startswith("Tweet") and current_tweet:
                    tweets.append(current_tweet.strip())
                    current_tweet = ""
                else:
                    current_tweet += line + "\n"
            
            if current_tweet:
                tweets.append(current_tweet.strip())
        
        # If still no tweets, use the whole content
        if not tweets:
            tweets = [content.strip()]
        
        logger.info(f"Generated {len(tweets)} tweets successfully")
        return tweets
        
    except Exception as e:
        logger.error(f"Error generating tweets: {str(e)}")
        return [f"Error generating tweet recommendations: {str(e)}"]


def save_tweets_to_json(tweets: List[str], query: str, query_type: str, output_path: str) -> None:
    """
    Save generated tweets to a JSON file.
    
    Args:
        tweets: List of generated tweets
        query: The specialized query about the repository
        query_type: Type of the query (architecture, technical, etc.)
        output_path: Path to save the JSON file
    """
    logger.info(f"Saving tweets to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare the data
    data = {
        "query_type": query_type,
        "query": query,
        "tweets": tweets,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved tweets to {output_path}")


def generate_and_save_tweets_for_query(
    query: str,
    chunks: List[Dict[str, Any]],
    query_type: str,
    output_path: str,
    num_tweets: int = NUM_TWEETS
) -> List[Dict[str, Any]]:
    """
    Generate and save tweets for a specific query.
    
    Args:
        query: The specialized query about the repository
        chunks: List of chunk dictionaries
        query_type: Type of the query (architecture, technical, etc.)
        output_path: Path to save the JSON file
        num_tweets: Number of tweets to generate
        
    Returns:
        List[Dict[str, Any]]: Generated tweets with metadata
    """
    logger.info(f"Processing {query_type} query")
    
    # Generate tweets
    tweets = generate_tweets(query, chunks, num_tweets)
    
    # Save tweets to JSON
    save_tweets_to_json(tweets, query, query_type, output_path)
    
    # Prepare tweets with metadata for the combined file
    tweets_with_metadata = []
    for i, tweet in enumerate(tweets):
        tweet_with_metadata = {
            "tweet_id": f"{query_type}_{i+1}",
            "query_type": query_type,
            "tweet_text": tweet
        }
        tweets_with_metadata.append(tweet_with_metadata)
    
    return tweets_with_metadata


def save_all_tweets(all_tweets: List[Dict[str, Any]], output_path: str = TWEETS_ALL_PATH) -> None:
    """
    Save all generated tweets to a single JSON file.
    
    Args:
        all_tweets: List of all generated tweets with metadata
        output_path: Path to save the combined JSON file
    """
    logger.info(f"Saving all tweets to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare the data
    data = {
        "tweets": all_tweets,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved all tweets to {output_path}")


def run_tweet_recommendation(
    architecture_query: str,
    technical_implementation_query: str,
    algorithmic_approach_query: str,
    domain_specific_query: str,
    integration_pipeline_query: str,
    top_chunks_architecture: List[Dict[str, Any]],
    top_chunks_technical: List[Dict[str, Any]],
    top_chunks_algorithmic: List[Dict[str, Any]],
    top_chunks_domain: List[Dict[str, Any]],
    top_chunks_integration: List[Dict[str, Any]],
    file_suffix: str = ""
) -> Dict[str, List[str]]:
    """
    Run the tweet recommendation process for all five specialized queries.
    
    Args:
        architecture_query: System architecture, design patterns, and overall structure
        technical_implementation_query: Specific technologies, libraries, and frameworks
        algorithmic_approach_query: Algorithms, mathematical models, and computational techniques
        domain_specific_query: Specific academic domain and research methodologies
        integration_pipeline_query: Component interactions and pipeline structure
        top_chunks_architecture: Top chunks by architecture similarity
        top_chunks_technical: Top chunks by technical implementation similarity
        top_chunks_algorithmic: Top chunks by algorithmic approach similarity
        top_chunks_domain: Top chunks by domain-specific similarity
        top_chunks_integration: Top chunks by integration pipeline similarity
        file_suffix: Optional suffix to add to output file names (e.g., "(1)" for recent work)
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping query types to lists of generated tweets
    """
    logger.info("Running tweet recommendation")
    
    # Initialize all tweets list
    all_tweets = []
    
    # Generate custom file paths based on suffix
    def get_custom_path(base_path: str, suffix: str) -> str:
        if suffix:
            # Insert suffix before the file extension
            path_obj = Path(base_path)
            return str(path_obj.with_name(f"{path_obj.stem}{suffix}{path_obj.suffix}"))
        return base_path
    
    # Process architecture query
    architecture_tweets = generate_and_save_tweets_for_query(
        query=architecture_query,
        chunks=top_chunks_architecture,
        query_type="architecture",
        output_path=get_custom_path(TWEETS_ARCHITECTURE_PATH, file_suffix)
    )
    all_tweets.extend(architecture_tweets)
    
    # Process technical implementation query
    technical_tweets = generate_and_save_tweets_for_query(
        query=technical_implementation_query,
        chunks=top_chunks_technical,
        query_type="technical",
        output_path=get_custom_path(TWEETS_TECHNICAL_PATH, file_suffix)
    )
    all_tweets.extend(technical_tweets)
    
    # Process algorithmic approach query
    algorithmic_tweets = generate_and_save_tweets_for_query(
        query=algorithmic_approach_query,
        chunks=top_chunks_algorithmic,
        query_type="algorithmic",
        output_path=get_custom_path(TWEETS_ALGORITHMIC_PATH, file_suffix)
    )
    all_tweets.extend(algorithmic_tweets)
    
    # Process domain-specific query
    domain_tweets = generate_and_save_tweets_for_query(
        query=domain_specific_query,
        chunks=top_chunks_domain,
        query_type="domain",
        output_path=get_custom_path(TWEETS_DOMAIN_PATH, file_suffix)
    )
    all_tweets.extend(domain_tweets)
    
    # Process integration pipeline query
    integration_tweets = generate_and_save_tweets_for_query(
        query=integration_pipeline_query,
        chunks=top_chunks_integration,
        query_type="integration",
        output_path=get_custom_path(TWEETS_INTEGRATION_PATH, file_suffix)
    )
    all_tweets.extend(integration_tweets)
    
    # Save all tweets to a single file
    save_all_tweets(all_tweets, get_custom_path(TWEETS_ALL_PATH, file_suffix))
    
    # Prepare result
    result = {
        "architecture": [tweet["tweet_text"] for tweet in architecture_tweets],
        "technical": [tweet["tweet_text"] for tweet in technical_tweets],
        "algorithmic": [tweet["tweet_text"] for tweet in algorithmic_tweets],
        "domain": [tweet["tweet_text"] for tweet in domain_tweets],
        "integration": [tweet["tweet_text"] for tweet in integration_tweets]
    }
    
    logger.info("Tweet recommendation completed")
    return result


if __name__ == "__main__":
    # For testing purposes
    import sys
    import os
    
    # Add the project root to the Python path to allow imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    
    # Example query and chunks
    query = "Microservices architecture with event-driven communication patterns"
    chunks = [
        {
            "arxiv_id": "2101.00123",
            "title": "Example Paper 1",
            "authors": "Author 1, Author 2",
            "chunk_text": "Microservices architectures benefit from event-driven patterns for asynchronous communication."
        },
        {
            "arxiv_id": "2102.00456",
            "title": "Example Paper 2",
            "authors": "Author 3, Author 4",
            "chunk_text": "Event sourcing is a powerful pattern for maintaining consistency in distributed systems."
        }
    ]
    
    # Generate tweets
    tweets = generate_tweets(query, chunks, 2)
    
    # Print the results
    print("\nGenerated Tweets:")
    for i, tweet in enumerate(tweets):
        print(f"\nTweet {i+1}:")
        print(tweet)
