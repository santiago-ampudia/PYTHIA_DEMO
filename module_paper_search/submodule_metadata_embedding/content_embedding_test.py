#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content Embedding Test

This script prints out all the chunks in the embedded database.
"""

import os
import sqlite3
import textwrap
from module_paper_search.submodule_metadata_embedding.content_embedding_parameters import CHUNK_INDEX_DB, DATABASES_DIR

def print_all_chunks(save_to_file=True):
    """Print all chunks in the database."""
    # Check if database exists
    if not os.path.exists(CHUNK_INDEX_DB):
        print(f"Database not found at {CHUNK_INDEX_DB}")
        return
    
    # Connect to the database
    conn = sqlite3.connect(CHUNK_INDEX_DB)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    
    # Prepare output file if saving to file
    output_file = None
    if save_to_file:
        output_path = os.path.join(DATABASES_DIR, "chunks_export.txt")
        output_file = open(output_path, 'w', encoding='utf-8')
        print(f"Saving full chunks to: {output_path}")
    
    try:
        # First, let's check the schema to see what columns are available
        cursor = conn.execute("PRAGMA table_info(chunks)")
        columns = [row['name'] for row in cursor.fetchall()]
        print(f"Available columns in the chunks table: {columns}")
        
        # Query all chunks with the correct column names
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
                header = f"\n{'='*80}\nPAPER: {current_paper}\n{'='*80}"
                print(header)
                if output_file:
                    output_file.write(header + "\n")
            
            # Print chunk information
            chunk_count += 1
            chunk_header = f"\n--- Chunk {chunk_count} (Section: {row['section_title']}) ---"
            embedding_info = f"Embedding Index: {row['embedding_index']}"
            
            print(chunk_header)
            print(embedding_info)
            if output_file:
                output_file.write(chunk_header + "\n")
                output_file.write(embedding_info + "\n\n")
            
            # Format and print the chunk text (wrapped for readability)
            chunk_text = row['chunk_text']
            wrapped_text = textwrap.fill(chunk_text[:500], width=80)
            
            if len(chunk_text) > 500:
                wrapped_text += "... [truncated for console display]"
            
            # Print truncated version to console
            print(wrapped_text)
            
            # Write full text to file
            if output_file:
                # Write the full text to file without truncation
                output_file.write(chunk_text + "\n")
            
            chunk_footer = f"--- End of Chunk {chunk_count} ---\n"
            print(chunk_footer)
            if output_file:
                output_file.write(chunk_footer + "\n")
    
    finally:
        # Close the connection
        conn.close()
        if output_file:
            output_file.close()
            print(f"\nFull chunks have been saved to: {output_path}")

if __name__ == "__main__":
    print_all_chunks()
