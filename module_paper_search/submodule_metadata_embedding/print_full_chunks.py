#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print Full Chunks

This script prints out ALL the chunks in the embedded database WITHOUT ANY TRUNCATION.
"""

import os
import sqlite3
from module_paper_search.submodule_metadata_embedding.content_embedding_parameters import CHUNK_INDEX_DB

def print_full_chunks():
    """Print all chunks in the database without any truncation."""
    # Check if database exists
    if not os.path.exists(CHUNK_INDEX_DB):
        print(f"Database not found at {CHUNK_INDEX_DB}")
        return
    
    # Connect to the database
    conn = sqlite3.connect(CHUNK_INDEX_DB)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    
    try:
        # Query all chunks
        cursor = conn.execute('''
            SELECT arxiv_id, section_title, chunk_text, embedding_index
            FROM chunks
            ORDER BY arxiv_id
        ''')
        
        rows = cursor.fetchall()
        
        if not rows:
            print("No chunks found in the database.")
            return
        
        print(f"Found {len(rows)} chunks in the database:\n")
        
        # Group by paper
        current_paper = None
        chunk_count = 0
        
        for row in rows:
            # Print paper header if we're starting a new paper
            if current_paper != row['arxiv_id']:
                current_paper = row['arxiv_id']
                chunk_count = 0
                print(f"\n{'='*80}")
                print(f"PAPER: {current_paper}")
                print(f"{'='*80}")
            
            # Print chunk information
            chunk_count += 1
            print(f"\n--- Chunk {chunk_count} (Section: {row['section_title']}) ---")
            print(f"Embedding Index: {row['embedding_index']}")
            print()
            
            # Print the FULL chunk text with NO truncation
            print(row['chunk_text'])
            
            print(f"\n--- End of Chunk {chunk_count} ---\n")
    
    finally:
        # Close the connection
        conn.close()

if __name__ == "__main__":
    print_full_chunks()
