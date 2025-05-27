"""
module_paper_search/submodule_answer_generation/test_answer_generation.py

This file contains tests for the answer generation module.
"""

import unittest
from unittest.mock import patch, MagicMock
import json

from module_paper_search.submodule_answer_generation.answer_generation import (
    select_top_chunks,
    build_context,
    generate_answer,
    evaluate_answer,
    run_answer_generation
)


class TestAnswerGeneration(unittest.TestCase):
    """Test cases for the answer generation module."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample summarized chunks for testing
        self.summarized_chunks = [
            {
                'chunk_id': 'chunk1',
                'arxiv_id': '2101.00001',
                'llm_summary': 'This paper discusses quantum computing applications.',
                'final_weight_adjusted': 0.9
            },
            {
                'chunk_id': 'chunk2',
                'arxiv_id': '2101.00002',
                'llm_summary': 'This paper explores neural network architectures.',
                'final_weight_adjusted': 0.8
            },
            {
                'chunk_id': 'chunk3',
                'arxiv_id': '2101.00003',
                'llm_summary': 'This paper analyzes climate change models.',
                'final_weight_adjusted': 0.7
            }
        ]
        
        # Sample citation key map
        self.citation_key_map = {
            'chunk1': 'Smith2021',
            'chunk2': 'Jones2022',
            'chunk3': 'Brown2023'
        }
        
        # Sample query
        self.enhanced_query = "What are the latest developments in quantum computing?"

    def test_select_top_chunks(self):
        """Test selecting top chunks based on weight."""
        top_chunks = select_top_chunks(self.summarized_chunks, n=2)
        self.assertEqual(len(top_chunks), 2)
        self.assertEqual(top_chunks[0]['chunk_id'], 'chunk1')
        self.assertEqual(top_chunks[1]['chunk_id'], 'chunk2')

    def test_build_context(self):
        """Test building context from chunks."""
        context = build_context(self.summarized_chunks[:2], self.citation_key_map)
        self.assertIn('[Smith2021]', context)
        self.assertIn('[Jones2022]', context)
        self.assertIn('quantum computing', context)
        self.assertIn('neural network', context)

    def test_build_context_with_arxiv_id(self):
        """Test that build_context uses arXiv IDs as citation keys."""
        # Use the first 2 chunks from the sample data
        top_chunks = self.summarized_chunks[:2]
        
        # Build context with citation keys
        context = build_context(top_chunks, self.citation_key_map, use_arxiv_id=True)
        
        # Check that the context contains the arXiv IDs instead of the citation keys
        self.assertIn('[2101.00001]', context)
        self.assertIn('[2101.00002]', context)
        self.assertNotIn('[Smith2021]', context)
        self.assertNotIn('[Jones2022]', context)

    @patch('openai.chat.completions.create')
    def test_generate_answer(self, mock_openai):
        """Test generating an answer."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated answer about quantum computing with citations (Smith2021)."
        mock_openai.return_value = mock_response
        
        context = build_context(self.summarized_chunks[:2], self.citation_key_map)
        answer = generate_answer(context, self.enhanced_query)
        
        self.assertIn("Generated answer", answer)
        self.assertIn("(Smith2021)", answer)
        mock_openai.assert_called_once()

    @patch('openai.chat.completions.create')
    def test_evaluate_answer(self, mock_openai):
        """Test evaluating an answer."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.85"
        mock_openai.return_value = mock_response
        
        context = build_context(self.summarized_chunks[:2], self.citation_key_map)
        answer_text = "Sample answer about quantum computing with citations (Smith2021)."
        score = evaluate_answer(context, self.enhanced_query, answer_text)
        
        self.assertEqual(score, 0.85)
        mock_openai.assert_called_once()

    @patch('module_paper_search.submodule_answer_generation.answer_generation.generate_answer')
    @patch('module_paper_search.submodule_answer_generation.answer_generation.evaluate_answer')
    def test_run_answer_generation(self, mock_evaluate, mock_generate):
        """Test the full answer generation pipeline."""
        # Mock function returns
        mock_generate.return_value = "Generated answer with citations."
        mock_evaluate.return_value = 0.9
        
        answer_text, score = run_answer_generation(
            self.summarized_chunks,
            self.enhanced_query,
            self.citation_key_map,
            n_answer=2
        )
        
        self.assertEqual(answer_text, "Generated answer with citations.")
        self.assertEqual(score, 0.9)
        mock_generate.assert_called_once()
        mock_evaluate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
