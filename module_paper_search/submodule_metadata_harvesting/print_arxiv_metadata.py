#!/usr/bin/env python3
"""
Script to print all metadata for papers in the arxiv_metadata.db database
with 5 lines of spacing between each paper.
"""

import os
import sqlite3
import sys

# Get the same DB_PATH as used in the harvesting script
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASES_DIR = os.path.join(MAIN_DIR, "databases")
DB_PATH = os.path.join(DATABASES_DIR, "arxiv_metadata.db")

def print_all_papers_metadata():
    """
    Prints all metadata for each paper in the database with 5 lines of spacing between papers.
    """
    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        sys.exit(1)
    
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all papers from the database with all metadata fields
        cursor.execute("SELECT arxiv_id, title, summary, published, updated, categories, datestamp, authors, comments, doi FROM papers")
        papers = cursor.fetchall()
        
        if not papers:
            print("No papers found in the database.")
            conn.close()
            return
        
        # Print metadata for each paper with spacing
        for i, paper in enumerate(papers):
            arxiv_id, title, summary, published, updated, categories, datestamp, authors, comments, doi = paper
            
            print(f"Paper #{i+1}")
            print(f"ArXiv ID: {arxiv_id}")
            print(f"Title: {title}")
            print(f"Authors: {authors}")
            print(f"Published: {published}")
            print(f"Updated: {updated}")
            print(f"Repository Datestamp: {datestamp}")
            print(f"Categories: {categories}")
            
            # Print additional metadata if available
            if comments:
                print(f"Comments: {comments}")
            if doi:
                print(f"DOI: {doi}")
                
            print(f"Summary: {summary}")
            
            # Add 5 lines of spacing between papers (except after the last paper)
            if i < len(papers) - 1:
                print("\n\n\n\n\n")
        
        # Close the database connection
        conn.close()
        print(f"\nTotal papers printed: {len(papers)}")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print_all_papers_metadata()
