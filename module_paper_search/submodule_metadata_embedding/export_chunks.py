#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Chunks

This script exports all chunks from the database to a text file.
"""

import os
import sqlite3
from module_paper_search.submodule_metadata_embedding.content_embedding_parameters import CHUNK_INDEX_DB, DATABASES_DIR

def export_chunks():
    """Export all chunks to a text file."""
    # Check if database exists
    if not os.path.exists(CHUNK_INDEX_DB):
        print(f"Database not found at {CHUNK_INDEX_DB}")
        return
    
    # Connect to the database
    conn = sqlite3.connect(CHUNK_INDEX_DB)
    conn.row_factory = sqlite3.Row  # This allows accessing columns by name
    
    # Prepare output file
    output_path = os.path.join(DATABASES_DIR, "chunks_export_full.txt")
    with open(output_path, 'w', encoding='utf-8') as output_file:
        # Query all chunks
        cursor = conn.execute('''
            SELECT arxiv_id, section_title, chunk_text, embedding_index
            FROM chunks
            ORDER BY arxiv_id, embedding_index
        ''')
        
        rows = cursor.fetchall()
        
        if not rows:
            print("No chunks found in the database.")
            return
        
        print(f"Exporting {len(rows)} chunks to {output_path}")
        
        # Group by paper
        current_paper = None
        
        for row in rows:
            # Print paper header if we're starting a new paper
            if current_paper != row['arxiv_id']:
                current_paper = row['arxiv_id']
                header = f"\n\n{'='*80}\nPAPER: {current_paper}\n{'='*80}\n\n"
                output_file.write(header)
            
            # Write chunk information
            chunk_header = f"--- Chunk (Section: {row['section_title']}) ---"
            embedding_info = f"Embedding Index: {row['embedding_index']}"
            
            output_file.write(chunk_header + "\n")
            output_file.write(embedding_info + "\n\n")
            
            # Write the full text to file without truncation
            output_file.write(row['chunk_text'] + "\n\n")
            
            chunk_footer = f"--- End of Chunk ---\n\n"
            output_file.write(chunk_footer)
    
    # Close the connection
    conn.close()
    print(f"\nAll chunks have been saved to: {output_path}")

if __name__ == "__main__":
    export_chunks()
