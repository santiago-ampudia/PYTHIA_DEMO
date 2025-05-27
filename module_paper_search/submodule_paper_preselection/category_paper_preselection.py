"""
submodule_paper_preselection/category_paper_preselection.py

This module implements Step 4 of the paper search pipeline: Paper Preselection by Category.

Purpose: Select candidate papers from the local SQLite metadata database by filtering on 
predicted arXiv categories. This significantly reduces the number of papers sent into 
chunk-level similarity search.

Process:
    1. Take predicted_categories from arXiv category prediction
    2. Build SQL query to filter papers by these categories
    3. Execute query on arxiv_metadata.db
    4. Parse results into structured list of metadata entries

Output: List of paper metadata entries to be passed to chunk selection
"""

import os
import sqlite3
import logging
import json
from typing import List, Dict, Any
from .paper_preselection_parameters import DB_PATH, PRESELECTION_DB_PATH, PRESELECTION_JSON_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('paper_preselection')


def preselect_papers_by_categories(predicted_categories: List[str], db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Select papers from the database based on predicted arXiv categories.
    
    Args:
        predicted_categories: List of predicted arXiv categories
        db_path: Path to the SQLite database
        
    Returns:
        List of paper metadata entries
    """
    logger.info(f"Preselecting papers by categories: {predicted_categories}")
    
    # Check if database exists
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        return []
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    # Build the SQL query with placeholders for categories
    placeholders = ','.join(['?' for _ in predicted_categories])
    
    # The LIKE conditions for each category
    like_conditions = []
    for _ in predicted_categories:
        like_conditions.append("categories LIKE ?")
    
    # Join the LIKE conditions with OR
    where_clause = ' OR '.join(like_conditions)
    
    # Complete SQL query
    query = f"""
    SELECT * FROM papers
    WHERE {where_clause}
    """
    
    # Prepare parameters for the LIKE conditions (adding % for partial matches)
    params = [f"%{category}%" for category in predicted_categories]
    
    try:
        # Execute the query
        logger.info(f"Executing SQL query: {query} with params: {params}")
        cursor.execute(query, params)
        
        # Fetch all results
        rows = cursor.fetchall()
        logger.info(f"Found {len(rows)} papers matching the categories")
        
        # Convert rows to dictionaries
        preselected_papers = []
        for row in rows:
            paper = {
                "arxiv_id": row["arxiv_id"],
                "title": row["title"],
                "summary": row["summary"],
                "published": row["published"],
                "updated": row["updated"],
                "categories": row["categories"],
                "datestamp": row["datestamp"],
                "authors": row["authors"],
                "comments": row["comments"],
                "doi": row["doi"]
            }
            preselected_papers.append(paper)
        
        return preselected_papers
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        return []
    finally:
        conn.close()


def create_preselection_database(preselected_papers: List[Dict[str, Any]], db_path: str = PRESELECTION_DB_PATH) -> None:
    """
    Create a database with preselected papers.
    
    Args:
        preselected_papers: List of paper metadata entries
        db_path: Path to save the SQLite database
    """
    logger.info(f"Creating preselection database at {db_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to the database (will create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS preselected_papers (
        arxiv_id TEXT PRIMARY KEY,
        title TEXT,
        summary TEXT,
        published TEXT,
        updated TEXT,
        categories TEXT,
        datestamp TEXT,
        authors TEXT,
        comments TEXT,
        doi TEXT
    )
    """)
    
    # Clear existing data
    cursor.execute("DELETE FROM preselected_papers")
    
    # Insert papers
    for paper in preselected_papers:
        cursor.execute("""
        INSERT INTO preselected_papers (
            arxiv_id, title, summary, published, updated, 
            categories, datestamp, authors, comments, doi
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper["arxiv_id"], paper["title"], paper["summary"], 
            paper["published"], paper["updated"], paper["categories"], 
            paper["datestamp"], paper["authors"], paper["comments"], paper["doi"]
        ))
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    logger.info(f"Created preselection database with {len(preselected_papers)} papers")


def save_preselection_json(preselected_papers: List[Dict[str, Any]], json_path: str = PRESELECTION_JSON_PATH) -> None:
    """
    Save preselected papers to a JSON file.
    
    Args:
        preselected_papers: List of paper metadata entries
        json_path: Path to save the JSON file
    """
    logger.info(f"Saving preselection results to JSON at {json_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Save to JSON
    with open(json_path, 'w') as f:
        json.dump(preselected_papers, f, indent=2)
    
    logger.info(f"Saved preselection results to {json_path}")


def run_category_paper_preselection(predicted_categories: List[str]) -> List[Dict[str, Any]]:
    """
    Run the category-based paper preselection process.
    
    Args:
        predicted_categories: List of predicted arXiv categories
        
    Returns:
        List of preselected paper metadata entries
    """
    logger.info("Starting category-based paper preselection")
    
    # Step 1: Preselect papers by categories
    preselected_papers = preselect_papers_by_categories(predicted_categories)
    
    if not preselected_papers:
        logger.warning("No papers found matching the predicted categories")
        return []
    
    # Step 2: Create preselection database
    create_preselection_database(preselected_papers)
    
    # Step 3: Save preselection results to JSON
    save_preselection_json(preselected_papers)
    
    logger.info(f"Category-based paper preselection completed. Found {len(preselected_papers)} papers.")
    
    return preselected_papers


def print_preselection_results(preselected_papers: List[Dict[str, Any]]) -> None:
    """
    Print the preselection results.
    
    Args:
        preselected_papers: List of paper metadata entries
    """
    logger.info(f"Preselection Results: {len(preselected_papers)} papers")
    
    for i, paper in enumerate(preselected_papers[:10], 1):  # Print only first 10 papers
        logger.info(f"\n{i}. {paper['title']}")
        logger.info(f"   arXiv ID: {paper['arxiv_id']}")
        logger.info(f"   Categories: {paper['categories']}")
        logger.info(f"   Published: {paper['published']}")
        
    if len(preselected_papers) > 10:
        logger.info(f"\n... and {len(preselected_papers) - 10} more papers")


if __name__ == "__main__":
    # For testing purposes
    import sys
    
    # Test with some sample categories
    test_categories = ["hep-ex", "hep-th"]
    
    # Run preselection
    preselected_papers = run_category_paper_preselection(test_categories)
    
    # Print results
    print_preselection_results(preselected_papers)
