"""
submodule_chunk_similarity_selection/chunk_similarity_selection_recommendation.py

This module implements Chunk Similarity Selection for recommendation mode.

Purpose: Select the most relevant chunks by computing cosine similarity between the five specialized 
query embeddings and both metadata and chunk embeddings. This step performs fine-grained 
selection of text chunks from papers that passed the category preselection filter.

Process:
    1. Embed the five specialized queries (architecture, technical, algorithmic, domain, integration)
    2. For each preselected paper:
        a. Compute metadata similarity scores with each query
        b. If any score exceeds threshold, retrieve and score all chunks from that paper
    3. Sort chunks by similarity scores and select top k chunks for each query

Output: Lists of top chunks for each of the five specialized queries
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

# Import from the original chunk similarity selection module
from module_paper_search.submodule_chunk_similarity_selection.chunk_similarity_selection import (
    EmbeddingModel,
    FaissIndexWrapper,
    cosine_similarity,
    get_metadata_embedding_index,
    get_chunk_embedding_indices,
    get_paper_metadata
)

# Import parameters for recommendation mode
from .chunk_similarity_selection_recommendation_parameters import (
    DB_PATH, 
    CHUNK_INDEX_DB_PATH,
    METADATA_INDEX_DB_PATH,
    METADATA_FAISS_INDEX_PATH,
    CHUNK_FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    THRESHOLD_METADATA_SCORE_ARCHITECTURE,
    THRESHOLD_METADATA_SCORE_TECHNICAL,
    THRESHOLD_METADATA_SCORE_ALGORITHMIC,
    THRESHOLD_METADATA_SCORE_DOMAIN,
    THRESHOLD_METADATA_SCORE_INTEGRATION,
    TOP_K_ARCHITECTURE,
    TOP_K_TECHNICAL,
    TOP_K_ALGORITHMIC,
    TOP_K_DOMAIN,
    TOP_K_INTEGRATION,
    DETAILED_RESULTS_RECOMMENDATION_PATH,
    SIMILARITY_RESULTS_RECOMMENDATION_PATH,
    SELECTED_CHUNKS_RECOMMENDATION_DB_PATH
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('chunk_similarity_selection_recommendation')


def save_detailed_results_recommendation(
    all_chunk_results: List[Dict[str, Any]],
    architecture_query: str,
    technical_implementation_query: str,
    algorithmic_approach_query: str,
    domain_specific_query: str,
    integration_pipeline_query: str,
    output_path: str = DETAILED_RESULTS_RECOMMENDATION_PATH
) -> None:
    """
    Save detailed chunk results to a text file for recommendation mode.
    
    Args:
        all_chunk_results: All chunk results with similarity scores
        architecture_query: System architecture, design patterns, and overall structure
        technical_implementation_query: Specific technologies, libraries, and frameworks
        algorithmic_approach_query: Algorithms, mathematical models, and computational techniques
        domain_specific_query: Specific academic domain and research methodologies
        integration_pipeline_query: Component interactions and pipeline structure
        output_path: Path to save results
    """
    logger.info(f"Saving detailed chunk results to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sort all chunks by combined score
    sorted_chunks = sorted(all_chunk_results, key=lambda x: x.get('combined_score', 0), reverse=True)
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("DETAILED CHUNK SIMILARITY RESULTS FOR RECOMMENDATION MODE\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write queries
        f.write("QUERIES:\n")
        f.write(f"Architecture Query: {architecture_query}\n")
        f.write(f"Technical Implementation Query: {technical_implementation_query}\n")
        f.write(f"Algorithmic Approach Query: {algorithmic_approach_query}\n")
        f.write(f"Domain-Specific Query: {domain_specific_query}\n")
        f.write(f"Integration Pipeline Query: {integration_pipeline_query}\n\n")
        
        # Write summary
        f.write("SUMMARY:\n")
        f.write(f"Total chunks processed: {len(all_chunk_results)}\n")
        f.write(f"Top combined score: {sorted_chunks[0].get('combined_score', 0) if sorted_chunks else 0}\n")
        f.write(f"Lowest combined score: {sorted_chunks[-1].get('combined_score', 0) if sorted_chunks else 0}\n\n")
        
        # Write detailed results for each chunk
        f.write("DETAILED RESULTS:\n")
        for i, chunk in enumerate(sorted_chunks[:100]):  # Limit to top 100 for readability
            f.write("-" * 80 + "\n")
            f.write(f"CHUNK #{i+1}\n")
            f.write(f"Paper: {chunk.get('title', 'Unknown')}\n")
            f.write(f"Authors: {chunk.get('authors', 'Unknown')}\n")
            f.write(f"arXiv ID: {chunk.get('arxiv_id', 'Unknown')}\n")
            f.write(f"Categories: {chunk.get('categories', 'Unknown')}\n")
            f.write(f"Published: {chunk.get('published', 'Unknown')}\n")
            f.write(f"Combined Score: {chunk.get('combined_score', 0)}\n")
            f.write(f"Architecture Score: {chunk.get('architecture_score', 0)}\n")
            f.write(f"Technical Score: {chunk.get('technical_score', 0)}\n")
            f.write(f"Algorithmic Score: {chunk.get('algorithmic_score', 0)}\n")
            f.write(f"Domain Score: {chunk.get('domain_score', 0)}\n")
            f.write(f"Integration Score: {chunk.get('integration_score', 0)}\n")
            f.write("\nCHUNK TEXT:\n")
            f.write(f"{chunk.get('chunk_text', 'No text available')}\n\n")
    
    logger.info(f"Saved detailed results to {output_path}")


def select_chunks_by_similarity_recommendation(
    architecture_query: str,
    technical_implementation_query: str,
    algorithmic_approach_query: str,
    domain_specific_query: str,
    integration_pipeline_query: str,
    preselected_papers: List[Dict[str, Any]],
    threshold_metadata_score_architecture: float = THRESHOLD_METADATA_SCORE_ARCHITECTURE,
    threshold_metadata_score_technical: float = THRESHOLD_METADATA_SCORE_TECHNICAL,
    threshold_metadata_score_algorithmic: float = THRESHOLD_METADATA_SCORE_ALGORITHMIC,
    threshold_metadata_score_domain: float = THRESHOLD_METADATA_SCORE_DOMAIN,
    threshold_metadata_score_integration: float = THRESHOLD_METADATA_SCORE_INTEGRATION,
    top_k_architecture: int = TOP_K_ARCHITECTURE,
    top_k_technical: int = TOP_K_TECHNICAL,
    top_k_algorithmic: int = TOP_K_ALGORITHMIC,
    top_k_domain: int = TOP_K_DOMAIN,
    top_k_integration: int = TOP_K_INTEGRATION,
    metadata_index_path: str = METADATA_FAISS_INDEX_PATH,
    chunk_index_path: str = CHUNK_FAISS_INDEX_PATH,
    chunk_db_path: str = CHUNK_INDEX_DB_PATH,
    metadata_db_path: str = METADATA_INDEX_DB_PATH,
    model_name: str = EMBEDDING_MODEL
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Select the most relevant chunks by computing similarity between queries and embeddings for recommendation mode.
    
    Args:
        architecture_query: System architecture, design patterns, and overall structure
        technical_implementation_query: Specific technologies, libraries, and frameworks
        algorithmic_approach_query: Algorithms, mathematical models, and computational techniques
        domain_specific_query: Specific academic domain and research methodologies
        integration_pipeline_query: Component interactions and pipeline structure
        preselected_papers: List of papers preselected by category
        threshold_metadata_score_architecture: Minimum metadata similarity score for architecture query
        threshold_metadata_score_technical: Minimum metadata similarity score for technical query
        threshold_metadata_score_algorithmic: Minimum metadata similarity score for algorithmic query
        threshold_metadata_score_domain: Minimum metadata similarity score for domain query
        threshold_metadata_score_integration: Minimum metadata similarity score for integration query
        top_k_architecture: Number of top chunks to select by architecture query
        top_k_technical: Number of top chunks to select by technical query
        top_k_algorithmic: Number of top chunks to select by algorithmic query
        top_k_domain: Number of top chunks to select by domain query
        top_k_integration: Number of top chunks to select by integration query
        metadata_index_path: Path to metadata embedding FAISS index
        chunk_index_path: Path to chunk embedding FAISS index
        chunk_db_path: Path to chunk index database
        metadata_db_path: Path to metadata index database
        model_name: Name of the embedding model
        
    Returns:
        Tuple of (top_chunks_architecture, top_chunks_technical, top_chunks_algorithmic, 
                  top_chunks_domain, top_chunks_integration, all_chunk_results)
    """
    logger.info("Selecting chunks by similarity for recommendation mode")
    
    # Initialize embedding model
    embedding_model = EmbeddingModel(model_name=model_name)
    
    # Embed queries
    logger.info("Embedding queries")
    architecture_query_embedding = embedding_model.embed(architecture_query)
    technical_query_embedding = embedding_model.embed(technical_implementation_query)
    algorithmic_query_embedding = embedding_model.embed(algorithmic_approach_query)
    domain_query_embedding = embedding_model.embed(domain_specific_query)
    integration_query_embedding = embedding_model.embed(integration_pipeline_query)
    
    # Load FAISS indices
    metadata_index = FaissIndexWrapper(metadata_index_path)
    chunk_index = FaissIndexWrapper(chunk_index_path)
    
    # Initialize results
    all_chunk_results = []
    
    # Process each preselected paper
    logger.info(f"Processing {len(preselected_papers)} preselected papers")
    for paper in preselected_papers:
        arxiv_id = paper["arxiv_id"]
        
        try:
            # Get metadata embedding index
            metadata_idx = get_metadata_embedding_index(arxiv_id, metadata_db_path)
            
            # Get metadata embedding vector
            metadata_vector = metadata_index.get_vector(metadata_idx)
            
            # Compute metadata similarity scores
            architecture_metadata_score = cosine_similarity(architecture_query_embedding, metadata_vector)
            technical_metadata_score = cosine_similarity(technical_query_embedding, metadata_vector)
            algorithmic_metadata_score = cosine_similarity(algorithmic_query_embedding, metadata_vector)
            domain_metadata_score = cosine_similarity(domain_query_embedding, metadata_vector)
            integration_metadata_score = cosine_similarity(integration_query_embedding, metadata_vector)
            
            # Check if any metadata score exceeds threshold
            if (architecture_metadata_score >= threshold_metadata_score_architecture or
                technical_metadata_score >= threshold_metadata_score_technical or
                algorithmic_metadata_score >= threshold_metadata_score_algorithmic or
                domain_metadata_score >= threshold_metadata_score_domain or
                integration_metadata_score >= threshold_metadata_score_integration):
                
                # Get chunk embedding indices and texts
                chunk_data = get_chunk_embedding_indices(arxiv_id, chunk_db_path)
                
                # Process each chunk
                for chunk_idx, chunk_text in chunk_data:
                    # Get chunk embedding vector
                    chunk_vector = chunk_index.get_vector(chunk_idx)
                    
                    # Compute chunk similarity scores
                    architecture_chunk_score = cosine_similarity(architecture_query_embedding, chunk_vector)
                    technical_chunk_score = cosine_similarity(technical_query_embedding, chunk_vector)
                    algorithmic_chunk_score = cosine_similarity(algorithmic_query_embedding, chunk_vector)
                    domain_chunk_score = cosine_similarity(domain_query_embedding, chunk_vector)
                    integration_chunk_score = cosine_similarity(integration_query_embedding, chunk_vector)
                    
                    # Compute combined score (average of all scores)
                    combined_score = (architecture_chunk_score + technical_chunk_score + 
                                     algorithmic_chunk_score + domain_chunk_score + 
                                     integration_chunk_score) / 5.0
                    
                    # Add to results
                    chunk_result = {
                        "arxiv_id": arxiv_id,
                        "chunk_idx": chunk_idx,
                        "chunk_text": chunk_text,
                        "architecture_score": architecture_chunk_score,
                        "technical_score": technical_chunk_score,
                        "algorithmic_score": algorithmic_chunk_score,
                        "domain_score": domain_chunk_score,
                        "integration_score": integration_chunk_score,
                        "combined_score": combined_score,
                        **paper  # Include all paper metadata
                    }
                    all_chunk_results.append(chunk_result)
                    
        except Exception as e:
            logger.error(f"Error processing paper {arxiv_id}: {str(e)}")
    
    # Sort chunks by each score type
    logger.info(f"Sorting {len(all_chunk_results)} chunks by scores")
    
    # Sort by architecture score
    top_chunks_architecture = sorted(
        all_chunk_results, 
        key=lambda x: x["architecture_score"], 
        reverse=True
    )[:top_k_architecture]
    
    # Sort by technical score
    top_chunks_technical = sorted(
        all_chunk_results, 
        key=lambda x: x["technical_score"], 
        reverse=True
    )[:top_k_technical]
    
    # Sort by algorithmic score
    top_chunks_algorithmic = sorted(
        all_chunk_results, 
        key=lambda x: x["algorithmic_score"], 
        reverse=True
    )[:top_k_algorithmic]
    
    # Sort by domain score
    top_chunks_domain = sorted(
        all_chunk_results, 
        key=lambda x: x["domain_score"], 
        reverse=True
    )[:top_k_domain]
    
    # Sort by integration score
    top_chunks_integration = sorted(
        all_chunk_results, 
        key=lambda x: x["integration_score"], 
        reverse=True
    )[:top_k_integration]
    
    logger.info(f"Selected top chunks: {top_k_architecture} architecture, {top_k_technical} technical, "
                f"{top_k_algorithmic} algorithmic, {top_k_domain} domain, {top_k_integration} integration")
    
    return (top_chunks_architecture, top_chunks_technical, top_chunks_algorithmic, 
            top_chunks_domain, top_chunks_integration, all_chunk_results)


def save_chunks_with_scores_json(
    top_chunks_architecture: List[Dict[str, Any]],
    top_chunks_technical: List[Dict[str, Any]],
    top_chunks_algorithmic: List[Dict[str, Any]],
    top_chunks_domain: List[Dict[str, Any]],
    top_chunks_integration: List[Dict[str, Any]],
    all_chunk_results: List[Dict[str, Any]],
    architecture_query: str,
    technical_implementation_query: str,
    algorithmic_approach_query: str,
    domain_specific_query: str,
    integration_pipeline_query: str,
    output_path: str = os.path.join(os.path.dirname(SIMILARITY_RESULTS_RECOMMENDATION_PATH), 'chunks_with_scores.json')
) -> None:
    """
    Save chunks with their scores for all five queries to a JSON file.
    
    Args:
        top_chunks_architecture: Top chunks by architecture similarity
        top_chunks_technical: Top chunks by technical implementation similarity
        top_chunks_algorithmic: Top chunks by algorithmic approach similarity
        top_chunks_domain: Top chunks by domain-specific similarity
        top_chunks_integration: Top chunks by integration pipeline similarity
        all_chunk_results: All chunks with their scores
        architecture_query: System architecture query
        technical_implementation_query: Technical implementation query
        algorithmic_approach_query: Algorithmic approach query
        domain_specific_query: Domain-specific query
        integration_pipeline_query: Integration pipeline query
        output_path: Path to save the JSON file
    """
    logger.info(f"Saving chunks with scores to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a dictionary to store chunks by query type
    result = {
        "queries": {
            "architecture_query": architecture_query,
            "technical_query": technical_implementation_query,
            "algorithmic_query": algorithmic_approach_query,
            "domain_query": domain_specific_query,
            "integration_query": integration_pipeline_query
        },
        "selected_chunks": {
            "architecture_query": [],
            "technical_query": [],
            "algorithmic_query": [],
            "domain_query": [],
            "integration_query": []
        }
    }
    
    # Helper function to extract chunk info with scores
    def extract_chunk_info(chunk):
        return {
            "chunk_id": chunk.get("chunk_id", ""),
            "arxiv_id": chunk.get("arxiv_id", ""),
            "title": chunk.get("title", ""),
            "chunk_text": chunk.get("chunk_text", ""),
            "scores": {
                "architecture_score": chunk.get("architecture_score", 0),
                "technical_score": chunk.get("technical_score", 0),
                "algorithmic_score": chunk.get("algorithmic_score", 0),
                "domain_score": chunk.get("domain_score", 0),
                "integration_score": chunk.get("integration_score", 0)
            }
        }
    
    # Add chunks for each query type
    for chunk in top_chunks_architecture:
        result["selected_chunks"]["architecture_query"].append(extract_chunk_info(chunk))
    
    for chunk in top_chunks_technical:
        result["selected_chunks"]["technical_query"].append(extract_chunk_info(chunk))
    
    for chunk in top_chunks_algorithmic:
        result["selected_chunks"]["algorithmic_query"].append(extract_chunk_info(chunk))
    
    for chunk in top_chunks_domain:
        result["selected_chunks"]["domain_query"].append(extract_chunk_info(chunk))
    
    for chunk in top_chunks_integration:
        result["selected_chunks"]["integration_query"].append(extract_chunk_info(chunk))
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Saved chunks with scores to {output_path}")
    logger.info(f"Architecture query: {len(result['selected_chunks']['architecture_query'])} chunks")
    logger.info(f"Technical query: {len(result['selected_chunks']['technical_query'])} chunks")
    logger.info(f"Algorithmic query: {len(result['selected_chunks']['algorithmic_query'])} chunks")
    logger.info(f"Domain query: {len(result['selected_chunks']['domain_query'])} chunks")
    logger.info(f"Integration query: {len(result['selected_chunks']['integration_query'])} chunks")


def save_similarity_results_recommendation(
    top_chunks_architecture: List[Dict[str, Any]],
    top_chunks_technical: List[Dict[str, Any]],
    top_chunks_algorithmic: List[Dict[str, Any]],
    top_chunks_domain: List[Dict[str, Any]],
    top_chunks_integration: List[Dict[str, Any]],
    output_path: str = SIMILARITY_RESULTS_RECOMMENDATION_PATH
) -> None:
    """
    Save similarity results to a JSON file for recommendation mode.
    
    Args:
        top_chunks_architecture: Top chunks by architecture similarity
        top_chunks_technical: Top chunks by technical implementation similarity
        top_chunks_algorithmic: Top chunks by algorithmic approach similarity
        top_chunks_domain: Top chunks by domain-specific similarity
        top_chunks_integration: Top chunks by integration pipeline similarity
        output_path: Path to save results
    """
    logger.info(f"Saving similarity results to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare results in the requested format
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "queries": {
            "architecture_query": {
                "chunks": []
            },
            "technical_implementation_query": {
                "chunks": []
            },
            "algorithmic_approach_query": {
                "chunks": []
            },
            "domain_specific_query": {
                "chunks": []
            },
            "integration_pipeline_query": {
                "chunks": []
            }
        }
    }
    
    # Process architecture query chunks
    for chunk in top_chunks_architecture:
        chunk_info = {
            "arxiv_id": chunk.get("arxiv_id", ""),
            "title": chunk.get("title", ""),
            "authors": chunk.get("authors", ""),
            "published": chunk.get("published", ""),
            "categories": chunk.get("categories", ""),
            "chunk_text": chunk.get("chunk_text", ""),
            "scores": {
                "architecture_score": chunk.get("architecture_score", 0),
                "technical_score": chunk.get("technical_score", 0),
                "algorithmic_score": chunk.get("algorithmic_score", 0),
                "domain_score": chunk.get("domain_score", 0),
                "integration_score": chunk.get("integration_score", 0),
                "combined_score": chunk.get("combined_score", 0)
            },
            "selected_for": "architecture_query"
        }
        results["queries"]["architecture_query"]["chunks"].append(chunk_info)
    
    # Process technical implementation query chunks
    for chunk in top_chunks_technical:
        chunk_info = {
            "arxiv_id": chunk.get("arxiv_id", ""),
            "title": chunk.get("title", ""),
            "authors": chunk.get("authors", ""),
            "published": chunk.get("published", ""),
            "categories": chunk.get("categories", ""),
            "chunk_text": chunk.get("chunk_text", ""),
            "scores": {
                "architecture_score": chunk.get("architecture_score", 0),
                "technical_score": chunk.get("technical_score", 0),
                "algorithmic_score": chunk.get("algorithmic_score", 0),
                "domain_score": chunk.get("domain_score", 0),
                "integration_score": chunk.get("integration_score", 0),
                "combined_score": chunk.get("combined_score", 0)
            },
            "selected_for": "technical_implementation_query"
        }
        results["queries"]["technical_implementation_query"]["chunks"].append(chunk_info)
    
    # Process algorithmic approach query chunks
    for chunk in top_chunks_algorithmic:
        chunk_info = {
            "arxiv_id": chunk.get("arxiv_id", ""),
            "title": chunk.get("title", ""),
            "authors": chunk.get("authors", ""),
            "published": chunk.get("published", ""),
            "categories": chunk.get("categories", ""),
            "chunk_text": chunk.get("chunk_text", ""),
            "scores": {
                "architecture_score": chunk.get("architecture_score", 0),
                "technical_score": chunk.get("technical_score", 0),
                "algorithmic_score": chunk.get("algorithmic_score", 0),
                "domain_score": chunk.get("domain_score", 0),
                "integration_score": chunk.get("integration_score", 0),
                "combined_score": chunk.get("combined_score", 0)
            },
            "selected_for": "algorithmic_approach_query"
        }
        results["queries"]["algorithmic_approach_query"]["chunks"].append(chunk_info)
    
    # Process domain-specific query chunks
    for chunk in top_chunks_domain:
        chunk_info = {
            "arxiv_id": chunk.get("arxiv_id", ""),
            "title": chunk.get("title", ""),
            "authors": chunk.get("authors", ""),
            "published": chunk.get("published", ""),
            "categories": chunk.get("categories", ""),
            "chunk_text": chunk.get("chunk_text", ""),
            "scores": {
                "architecture_score": chunk.get("architecture_score", 0),
                "technical_score": chunk.get("technical_score", 0),
                "algorithmic_score": chunk.get("algorithmic_score", 0),
                "domain_score": chunk.get("domain_score", 0),
                "integration_score": chunk.get("integration_score", 0),
                "combined_score": chunk.get("combined_score", 0)
            },
            "selected_for": "domain_specific_query"
        }
        results["queries"]["domain_specific_query"]["chunks"].append(chunk_info)
    
    # Process integration pipeline query chunks
    for chunk in top_chunks_integration:
        chunk_info = {
            "arxiv_id": chunk.get("arxiv_id", ""),
            "title": chunk.get("title", ""),
            "authors": chunk.get("authors", ""),
            "published": chunk.get("published", ""),
            "categories": chunk.get("categories", ""),
            "chunk_text": chunk.get("chunk_text", ""),
            "scores": {
                "architecture_score": chunk.get("architecture_score", 0),
                "technical_score": chunk.get("technical_score", 0),
                "algorithmic_score": chunk.get("algorithmic_score", 0),
                "domain_score": chunk.get("domain_score", 0),
                "integration_score": chunk.get("integration_score", 0),
                "combined_score": chunk.get("combined_score", 0)
            },
            "selected_for": "integration_pipeline_query"
        }
        results["queries"]["integration_pipeline_query"]["chunks"].append(chunk_info)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved similarity results to {output_path}")
    
    # Also save a more detailed version with full chunk information
    detailed_output_path = output_path.replace('.json', '_detailed.json')
    with open(detailed_output_path, 'w') as f:
        json.dump({
            "architecture_chunks": top_chunks_architecture,
            "technical_chunks": top_chunks_technical,
            "algorithmic_chunks": top_chunks_algorithmic,
            "domain_chunks": top_chunks_domain,
            "integration_chunks": top_chunks_integration,
            "timestamp": datetime.datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"Saved detailed similarity results to {detailed_output_path}")


def save_selected_chunks_to_db_recommendation(
    top_chunks_architecture: List[Dict[str, Any]],
    top_chunks_technical: List[Dict[str, Any]],
    top_chunks_algorithmic: List[Dict[str, Any]],
    top_chunks_domain: List[Dict[str, Any]],
    top_chunks_integration: List[Dict[str, Any]],
    architecture_query: str,
    technical_implementation_query: str,
    algorithmic_approach_query: str,
    domain_specific_query: str,
    integration_pipeline_query: str,
    output_path: str = SELECTED_CHUNKS_RECOMMENDATION_DB_PATH
) -> None:
    """
    Save selected chunks to a SQLite database for recommendation mode.
    
    Args:
        top_chunks_architecture: Top chunks by architecture similarity
        top_chunks_technical: Top chunks by technical implementation similarity
        top_chunks_algorithmic: Top chunks by algorithmic approach similarity
        top_chunks_domain: Top chunks by domain-specific similarity
        top_chunks_integration: Top chunks by integration pipeline similarity
        architecture_query: System architecture, design patterns, and overall structure
        technical_implementation_query: Specific technologies, libraries, and frameworks
        algorithmic_approach_query: Algorithms, mathematical models, and computational techniques
        domain_specific_query: Specific academic domain and research methodologies
        integration_pipeline_query: Component interactions and pipeline structure
        output_path: Path to save the database
    """
    logger.info(f"Saving selected chunks to database at {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Connect to the database (will create it if it doesn't exist)
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        query_id INTEGER PRIMARY KEY,
        query_type TEXT,
        query_text TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS selected_chunks (
        chunk_id INTEGER PRIMARY KEY,
        query_id INTEGER,
        arxiv_id TEXT,
        chunk_idx INTEGER,
        chunk_text TEXT,
        similarity_score REAL,
        title TEXT,
        authors TEXT,
        summary TEXT,
        categories TEXT,
        published TEXT,
        FOREIGN KEY (query_id) REFERENCES queries (query_id)
    )
    """)
    
    # Clear existing data
    cursor.execute("DELETE FROM selected_chunks")
    cursor.execute("DELETE FROM queries")
    
    # Insert queries
    queries = [
        ("architecture", architecture_query),
        ("technical", technical_implementation_query),
        ("algorithmic", algorithmic_approach_query),
        ("domain", domain_specific_query),
        ("integration", integration_pipeline_query)
    ]
    
    for query_type, query_text in queries:
        cursor.execute(
            "INSERT INTO queries (query_type, query_text) VALUES (?, ?)",
            (query_type, query_text)
        )
    
    # Get query IDs
    cursor.execute("SELECT query_id, query_type FROM queries")
    query_ids = {query_type: query_id for query_id, query_type in cursor.fetchall()}
    
    # Insert chunks for each query type
    chunk_data = [
        (query_ids["architecture"], top_chunks_architecture, "architecture_score"),
        (query_ids["technical"], top_chunks_technical, "technical_score"),
        (query_ids["algorithmic"], top_chunks_algorithmic, "algorithmic_score"),
        (query_ids["domain"], top_chunks_domain, "domain_score"),
        (query_ids["integration"], top_chunks_integration, "integration_score")
    ]
    
    for query_id, chunks, score_key in chunk_data:
        for chunk in chunks:
            cursor.execute("""
            INSERT INTO selected_chunks (
                query_id, arxiv_id, chunk_idx, chunk_text, similarity_score,
                title, authors, summary, categories, published
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id,
                chunk["arxiv_id"],
                chunk["chunk_idx"],
                chunk["chunk_text"],
                chunk[score_key],
                chunk.get("title", ""),
                chunk.get("authors", ""),
                chunk.get("summary", ""),
                chunk.get("categories", ""),
                chunk.get("published", "")
            ))
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    logger.info(f"Saved selected chunks to database at {output_path}")


def run_chunk_similarity_selection_recommendation(
    architecture_query: str,
    technical_implementation_query: str,
    algorithmic_approach_query: str,
    domain_specific_query: str,
    integration_pipeline_query: str,
    preselected_papers: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run the chunk similarity selection process for recommendation mode.
    
    Args:
        architecture_query: System architecture, design patterns, and overall structure
        technical_implementation_query: Specific technologies, libraries, and frameworks
        algorithmic_approach_query: Algorithms, mathematical models, and computational techniques
        domain_specific_query: Specific academic domain and research methodologies
        integration_pipeline_query: Component interactions and pipeline structure
        preselected_papers: List of papers preselected by category
        
    Returns:
        Tuple of (top_chunks_architecture, top_chunks_technical, top_chunks_algorithmic, 
                  top_chunks_domain, top_chunks_integration, all_chunk_results)
    """
    logger.info("Running chunk similarity selection for recommendation mode")
    logger.info(f"Architecture query: {architecture_query}")
    logger.info(f"Technical Implementation query: {technical_implementation_query}")
    logger.info(f"Algorithmic Approach query: {algorithmic_approach_query}")
    logger.info(f"Domain-Specific query: {domain_specific_query}")
    logger.info(f"Integration Pipeline query: {integration_pipeline_query}")
    logger.info(f"Number of preselected papers: {len(preselected_papers)}")
    
    # Select chunks by similarity
    (top_chunks_architecture, top_chunks_technical, top_chunks_algorithmic,
     top_chunks_domain, top_chunks_integration, all_chunk_results) = select_chunks_by_similarity_recommendation(
        architecture_query=architecture_query,
        technical_implementation_query=technical_implementation_query,
        algorithmic_approach_query=algorithmic_approach_query,
        domain_specific_query=domain_specific_query,
        integration_pipeline_query=integration_pipeline_query,
        preselected_papers=preselected_papers,
        metadata_index_path=METADATA_FAISS_INDEX_PATH,
        chunk_index_path=CHUNK_FAISS_INDEX_PATH,
        chunk_db_path=CHUNK_INDEX_DB_PATH,
        metadata_db_path=METADATA_INDEX_DB_PATH
    )
    
    # Save similarity results to JSON
    save_similarity_results_recommendation(
        top_chunks_architecture=top_chunks_architecture,
        top_chunks_technical=top_chunks_technical,
        top_chunks_algorithmic=top_chunks_algorithmic,
        top_chunks_domain=top_chunks_domain,
        top_chunks_integration=top_chunks_integration
    )
    
    # Save detailed results to text file
    save_detailed_results_recommendation(
        all_chunk_results=all_chunk_results,
        architecture_query=architecture_query,
        technical_implementation_query=technical_implementation_query,
        algorithmic_approach_query=algorithmic_approach_query,
        domain_specific_query=domain_specific_query,
        integration_pipeline_query=integration_pipeline_query
    )
    
    # Save selected chunks to database
    save_selected_chunks_to_db_recommendation(
        top_chunks_architecture=top_chunks_architecture,
        top_chunks_technical=top_chunks_technical,
        top_chunks_algorithmic=top_chunks_algorithmic,
        top_chunks_domain=top_chunks_domain,
        top_chunks_integration=top_chunks_integration,
        architecture_query=architecture_query,
        technical_implementation_query=technical_implementation_query,
        algorithmic_approach_query=algorithmic_approach_query,
        domain_specific_query=domain_specific_query,
        integration_pipeline_query=integration_pipeline_query
    )
    
    # Save chunks with their scores to JSON file
    save_chunks_with_scores_json(
        top_chunks_architecture=top_chunks_architecture,
        top_chunks_technical=top_chunks_technical,
        top_chunks_algorithmic=top_chunks_algorithmic,
        top_chunks_domain=top_chunks_domain,
        top_chunks_integration=top_chunks_integration,
        all_chunk_results=all_chunk_results,
        architecture_query=architecture_query,
        technical_implementation_query=technical_implementation_query,
        algorithmic_approach_query=algorithmic_approach_query,
        domain_specific_query=domain_specific_query,
        integration_pipeline_query=integration_pipeline_query
    )
    
    logger.info("Chunk similarity selection for recommendation mode completed")
    
    return (top_chunks_architecture, top_chunks_technical, top_chunks_algorithmic,
            top_chunks_domain, top_chunks_integration, all_chunk_results)


if __name__ == "__main__":
    # For testing purposes
    import sys
    import os
    
    # Add the project root to the Python path to allow imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    
    # Example queries
    architecture_query = "Microservices architecture with event-driven communication patterns for distributed ML training pipelines"
    technical_implementation_query = "Python-based implementation using PyTorch for deep learning models, FastAPI for REST endpoints"
    algorithmic_approach_query = "Gradient-boosted decision trees with custom loss functions for multi-class classification problems"
    domain_specific_query = "Natural language processing for sentiment analysis in social media data"
    integration_pipeline_query = "End-to-end ML pipeline with data ingestion, preprocessing, feature extraction, model training"
    
    # Example preselected papers (would normally come from category_paper_preselection_recommendation)
    preselected_papers = [
        {
            "arxiv_id": "2101.00123",
            "title": "Example Paper 1",
            "summary": "This is an example paper summary.",
            "published": "2021-01-01",
            "categories": "cs.LG cs.AI",
            "authors": "Author 1, Author 2"
        }
    ]
    
    # Run the chunk similarity selection for recommendation mode
    (top_chunks_architecture, top_chunks_technical, top_chunks_algorithmic,
     top_chunks_domain, top_chunks_integration, all_results) = run_chunk_similarity_selection_recommendation(
        architecture_query=architecture_query,
        technical_implementation_query=technical_implementation_query,
        algorithmic_approach_query=algorithmic_approach_query,
        domain_specific_query=domain_specific_query,
        integration_pipeline_query=integration_pipeline_query,
        preselected_papers=preselected_papers
    )
    
    # Print results
    print(f"\nFound {len(top_chunks_architecture)} top architecture chunks")
    print(f"Found {len(top_chunks_technical)} top technical implementation chunks")
    print(f"Found {len(top_chunks_algorithmic)} top algorithmic approach chunks")
    print(f"Found {len(top_chunks_domain)} top domain-specific chunks")
    print(f"Found {len(top_chunks_integration)} top integration pipeline chunks")
