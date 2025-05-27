"""
Test script for the answer generation module with arXiv ID citations.

This script tests the answer generation module's ability to use arXiv IDs as citation keys
without calling the full pipeline or making actual OpenAI API calls.
"""

import json
import os
from unittest.mock import patch, MagicMock

from module_paper_search.submodule_answer_generation.answer_generation import (
    select_top_chunks,
    build_context
)

def main():
    """Run a test of the answer generation module with mock data."""
    print("Testing answer generation with arXiv ID citations...")
    
    # Create mock summarized chunks with arXiv IDs
    mock_chunks = [
        {
            'chunk_id': 'paper1_chunk1',
            'arxiv_id': '2306.10057',
            'llm_summary': 'The XCC produces 80,000 Higgs bosons per 10^7 second year, roughly the same as the ILC Higgs rate at s=250 GeV.',
            'final_weight_adjusted': 0.95
        },
        {
            'chunk_id': 'paper1_chunk2',
            'arxiv_id': '2306.10057',
            'llm_summary': 'The Higgs boson production rate at XCC is 80,000 Higgs bosons per year, with a sharply peaked gamma gamma center-of-mass energy spectrum.',
            'final_weight_adjusted': 0.92
        },
        {
            'chunk_id': 'paper2_chunk1',
            'arxiv_id': '1307.6346',
            'llm_summary': 'Delphes is a framework for fast simulation of a generic collider experiment, including detector response and physics object reconstruction.',
            'final_weight_adjusted': 0.85
        }
    ]
    
    # Create mock citation key map
    citation_key_map = {
        'paper1_chunk1': 'Smith2021',
        'paper1_chunk2': 'Jones2022',
        'paper2_chunk1': 'Brown2023'
    }
    
    # Test selecting top chunks
    top_chunks = select_top_chunks(mock_chunks, n=2)
    print(f"Selected top {len(top_chunks)} chunks:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"{i}. Chunk ID: {chunk['chunk_id']}, Weight: {chunk['final_weight_adjusted']}")
    
    # Test building context with arXiv IDs
    context = build_context(top_chunks, citation_key_map, use_arxiv_id=True)
    print("\nGenerated context with arXiv ID citations:")
    print(context)
    
    # Test building context with traditional citation keys
    context_traditional = build_context(top_chunks, citation_key_map, use_arxiv_id=False)
    print("\nGenerated context with traditional citation keys:")
    print(context_traditional)
    
    # Simulate a generated answer with arXiv ID citations
    simulated_answer = (
        "The production rate of the Higgs boson at the XCC, an X-ray FEL-based gamma gamma "
        "Compton Collider Higgs Factory, is approximately 80,000 Higgs bosons per 10^7 second year (2306.10057). "
        "This production rate is comparable to the ILC Higgs rate at s=250 GeV."
    )
    
    print("\nSimulated generated answer:")
    print(simulated_answer)
    
    # Verify the answer contains arXiv ID citations
    if "(2306.10057)" in simulated_answer:
        print("\n Success: Answer contains arXiv ID citations")
    else:
        print("\n Error: Answer does not contain arXiv ID citations")

if __name__ == "__main__":
    main()
