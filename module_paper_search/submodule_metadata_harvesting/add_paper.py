#!/usr/bin/env python
"""
Simple script to add a specific paper to the database without running the entire pipeline.
This avoids making unnecessary API calls to the LLM services.
"""

import sys
import logging
from module_paper_search.submodule_metadata_harvesting.arxiv_metadata_harvesting import download_specific_paper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to add papers to the database."""
    if len(sys.argv) < 2:
        print("Usage: python module_paper_search/submodule_metadata_harvesting/add_paper.py <arxiv_id> [<arxiv_id2> ...]")
        print("Example: python module_paper_search/submodule_metadata_harvesting/add_paper.py 2306.10057 2304.12244")
        return
    
    # Get the list of arXiv IDs from command line arguments
    arxiv_ids = sys.argv[1:]
    
    success_count = 0
    for arxiv_id in arxiv_ids:
        print(f"Attempting to download paper with arXiv ID: {arxiv_id}")
        if download_specific_paper(arxiv_id):
            print(f"✅ Successfully added paper {arxiv_id} to the database")
            success_count += 1
        else:
            print(f"❌ Failed to add paper {arxiv_id} (may already exist in database)")
    
    print(f"\nSummary: Added {success_count} out of {len(arxiv_ids)} papers to the database")

if __name__ == "__main__":
    main()
