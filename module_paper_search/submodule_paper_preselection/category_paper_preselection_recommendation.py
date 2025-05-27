"""
submodule_paper_preselection/category_paper_preselection_recommendation.py

This module implements Paper Preselection by Category for recommendation mode.

Purpose: Select candidate papers from the local SQLite metadata database by filtering on 
predicted arXiv categories for GitHub repository recommendations. This significantly reduces 
the number of papers sent into chunk-level similarity search.

Process:
    1. Take predicted_categories from arXiv category prediction for recommendation mode
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
from .paper_preselection_recommendation_parameters import (
    DB_PATH, 
    PRESELECTION_RECOMMENDATION_DB_PATH, 
    PRESELECTION_RECOMMENDATION_JSON_PATH,
    MAX_PAPERS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('paper_preselection_recommendation')


def preselect_papers_by_categories_recommendation(predicted_categories: List[str], db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Select papers from the database based on predicted arXiv categories for recommendation mode.
    
    Args:
        predicted_categories: List of predicted arXiv categories
        db_path: Path to the SQLite database
        
    Returns:
        List of paper metadata entries
    """
    logger.info(f"Preselecting papers by categories for recommendation mode: {predicted_categories}")
    
    # Debug: Log all available papers and their categories before filtering
    try:
        debug_conn = sqlite3.connect(db_path)
        debug_conn.row_factory = sqlite3.Row
        debug_cursor = debug_conn.cursor()
        
        # Get all papers
        debug_cursor.execute("SELECT arxiv_id, categories FROM papers")
        all_papers = debug_cursor.fetchall()
        logger.info(f"DEBUG: Total papers in database before filtering: {len(all_papers)}")
        
        # Log each paper's categories to see if they match our predicted categories
        paper_categories = {}
        for paper in all_papers:
            paper_categories[paper['arxiv_id']] = paper['categories']
        
        logger.info(f"DEBUG: Paper categories: {paper_categories}")
        
        # Check which papers should match our categories
        matching_papers = []
        for paper_id, categories in paper_categories.items():
            for predicted_cat in predicted_categories:
                if predicted_cat in categories:
                    matching_papers.append(paper_id)
                    break
        
        logger.info(f"DEBUG: Papers that should match our categories: {matching_papers}")
        logger.info(f"DEBUG: Expected match count: {len(matching_papers)}")
        
        debug_conn.close()
    except Exception as e:
        logger.error(f"DEBUG: Error in debug logging: {str(e)}")
    
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
    
    # Complete SQL query with limit
    query = f"""
    SELECT * FROM papers
    WHERE {where_clause}
    ORDER BY published DESC
    LIMIT {MAX_PAPERS}
    """
    
    # Prepare parameters for the LIKE conditions (adding % for partial matches)
    params = [f"%{category}%" for category in predicted_categories]
    
    try:
        # Execute the query
        logger.info(f"Executing SQL query: {query} with params: {params}")
        cursor.execute(query, params)
        
        # Fetch all results
        rows = cursor.fetchall()
        logger.info(f"Found {len(rows)} papers matching the categories for recommendation mode")
        
        # Debug: Log which papers were actually selected
        selected_paper_ids = [row['arxiv_id'] for row in rows]
        logger.info(f"DEBUG: Selected paper IDs: {selected_paper_ids}")
        
        # Debug: For each paper, log why it was selected (which category matched)
        for row in rows:
            paper_id = row['arxiv_id']
            paper_categories = row['categories']
            matching_categories = [cat for cat in predicted_categories if cat in paper_categories]
            logger.info(f"DEBUG: Paper {paper_id} was selected because it matches categories: {matching_categories}")
        
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


def create_preselection_database_recommendation(preselected_papers: List[Dict[str, Any]], db_path: str = PRESELECTION_RECOMMENDATION_DB_PATH) -> None:
    """
    Create a database with preselected papers for recommendation mode.
    
    Args:
        preselected_papers: List of paper metadata entries
        db_path: Path to save the SQLite database
    """
    logger.info(f"Creating preselection database for recommendation mode at {db_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to the database (will create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS preselected_papers_recommendation (
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
    cursor.execute("DELETE FROM preselected_papers_recommendation")
    
    # Insert papers
    for paper in preselected_papers:
        cursor.execute("""
        INSERT INTO preselected_papers_recommendation (
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
    
    logger.info(f"Created preselection database for recommendation mode with {len(preselected_papers)} papers")


def save_preselection_json_recommendation(preselected_papers: List[Dict[str, Any]], json_path: str = PRESELECTION_RECOMMENDATION_JSON_PATH) -> None:
    """
    Save preselected papers to a JSON file for recommendation mode.
    
    Args:
        preselected_papers: List of paper metadata entries
        json_path: Path to save the JSON file
    """
    logger.info(f"Saving preselection results for recommendation mode to JSON at {json_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Save to JSON
    with open(json_path, 'w') as f:
        json.dump(preselected_papers, f, indent=2)
    
    logger.info(f"Saved preselection results for recommendation mode to {json_path}")


def run_category_paper_preselection_recommendation(predicted_categories: List[str]) -> List[Dict[str, Any]]:
    """
    Run the category-based paper preselection process for recommendation mode.
    
    Args:
        predicted_categories: List of predicted arXiv categories
        
    Returns:
        List of preselected paper metadata entries
    """
    logger.info("Starting category-based paper preselection for recommendation mode")
    
    # Debug: Check the database path
    logger.info(f"DEBUG: Using database at {DB_PATH}")
    if not os.path.exists(DB_PATH):
        logger.error(f"DEBUG: Database file does not exist at {DB_PATH}")
    else:
        # Debug: Check database content
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers")
            count = cursor.fetchone()[0]
            logger.info(f"DEBUG: Database contains {count} papers")
            
            # Check table structure
            cursor.execute("PRAGMA table_info(papers)")
            columns = cursor.fetchall()
            logger.info(f"DEBUG: Database table structure: {columns}")
            conn.close()
        except Exception as e:
            logger.error(f"DEBUG: Error checking database: {str(e)}")
    
    # Step 1: Preselect papers by categories
    preselected_papers = preselect_papers_by_categories_recommendation(predicted_categories)
    
    # Step 2: Create a database with preselected papers
    create_preselection_database_recommendation(preselected_papers)
    
    # Step 3: Save preselected papers to JSON
    save_preselection_json_recommendation(preselected_papers)
    
    # Step 4: Return the preselected papers
    return preselected_papers


def print_preselection_results_recommendation(preselected_papers: List[Dict[str, Any]]) -> None:
    """
    Print the preselection results for recommendation mode.
    
    Args:
        preselected_papers: List of paper metadata entries
    """
    print(f"\nPreselected {len(preselected_papers)} papers for recommendation mode:")
    
    # Print the first 5 papers
    for i, paper in enumerate(preselected_papers[:5]):
        print(f"\n{i+1}. {paper['title']}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        print(f"   Categories: {paper['categories']}")
        print(f"   Published: {paper['published']}")
        
    if len(preselected_papers) > 5:
        print(f"\n... and {len(preselected_papers) - 5} more papers.")


if __name__ == "__main__":
    # For testing purposes
    import sys
    import os
    
    # Add the project root to the Python path to allow imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    
    # Example predicted categories
    predicted_categories = ["cs.SE", "cs.LG", "cs.AI"]
    
    # Run the category-based paper preselection
    preselected_papers = run_category_paper_preselection_recommendation(predicted_categories)
    
    # Print the results
    print_preselection_results_recommendation(preselected_papers)
