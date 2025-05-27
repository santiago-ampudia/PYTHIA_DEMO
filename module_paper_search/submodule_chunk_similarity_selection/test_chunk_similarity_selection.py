"""
submodule_chunk_similarity_selection/test_chunk_similarity_selection.py

This module contains tests for the chunk similarity selection functionality.
"""

import unittest
import os
import json
import numpy as np
import tempfile
import sqlite3
import faiss
from unittest.mock import patch, MagicMock

from .chunk_similarity_selection import (
    EmbeddingModel,
    FaissIndexWrapper,
    cosine_similarity,
    get_chunk_ids_for_paper,
    select_chunks_by_similarity,
    save_similarity_results,
    run_chunk_similarity_selection
)


class TestChunkSimilaritySelection(unittest.TestCase):
    """Test cases for chunk similarity selection module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories and files for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name
        
        # Create mock data
        self.topic_query = "high energy physics collision experiments"
        self.subtopic_query = "Higgs boson detection methods"
        self.enhanced_query = "LHC ATLAS CMS Higgs boson detection methods in high energy physics collision experiments"
        
        self.preselected_papers = [
            {"arxiv_id": "2201.00001", "title": "Paper 1"},
            {"arxiv_id": "2201.00002", "title": "Paper 2"},
            {"arxiv_id": "2201.00003", "title": "Paper 3"}
        ]
        
        # Create mock vectors (dimension 384 for E5-small-v2)
        self.mock_dim = 384
        self.mock_topic_vector = np.random.rand(self.mock_dim).astype(np.float32)
        self.mock_topic_vector = self.mock_topic_vector / np.linalg.norm(self.mock_topic_vector)
        
        self.mock_subtopic_vector = np.random.rand(self.mock_dim).astype(np.float32)
        self.mock_subtopic_vector = self.mock_subtopic_vector / np.linalg.norm(self.mock_subtopic_vector)
        
        self.mock_enhanced_vector = np.random.rand(self.mock_dim).astype(np.float32)
        self.mock_enhanced_vector = self.mock_enhanced_vector / np.linalg.norm(self.mock_enhanced_vector)
        
        # Create mock database
        self.db_path = os.path.join(self.test_dir, "test_chunk_index.db")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE chunk_index (chunk_id TEXT, arxiv_id TEXT)")
        
        # Insert test data
        test_data = [
            ("chunk1", "2201.00001"),
            ("chunk2", "2201.00001"),
            ("chunk3", "2201.00002"),
            ("chunk4", "2201.00003"),
            ("chunk5", "2201.00003")
        ]
        cursor.executemany("INSERT INTO chunk_index VALUES (?, ?)", test_data)
        conn.commit()
        conn.close()
        
        # Create mock FAISS indices
        self.metadata_index_path = os.path.join(self.test_dir, "metadata_index")
        self.chunk_index_path = os.path.join(self.test_dir, "chunk_index")
        
        # Create metadata index
        metadata_index = faiss.IndexFlatIP(self.mock_dim)
        metadata_vectors = np.array([
            np.random.rand(self.mock_dim).astype(np.float32),
            np.random.rand(self.mock_dim).astype(np.float32),
            np.random.rand(self.mock_dim).astype(np.float32)
        ])
        for i in range(metadata_vectors.shape[0]):
            metadata_vectors[i] = metadata_vectors[i] / np.linalg.norm(metadata_vectors[i])
        metadata_index.add(metadata_vectors)
        faiss.write_index(metadata_index, self.metadata_index_path)
        
        # Create metadata ID map
        self.metadata_id_map_path = os.path.join(self.test_dir, "metadata_id_map.json")
        metadata_id_map = {
            "2201.00001": 0,
            "2201.00002": 1,
            "2201.00003": 2
        }
        with open(self.metadata_id_map_path, 'w') as f:
            json.dump(metadata_id_map, f)
        
        # Create chunk index
        chunk_index = faiss.IndexFlatIP(self.mock_dim)
        chunk_vectors = np.array([
            np.random.rand(self.mock_dim).astype(np.float32),
            np.random.rand(self.mock_dim).astype(np.float32),
            np.random.rand(self.mock_dim).astype(np.float32),
            np.random.rand(self.mock_dim).astype(np.float32),
            np.random.rand(self.mock_dim).astype(np.float32)
        ])
        for i in range(chunk_vectors.shape[0]):
            chunk_vectors[i] = chunk_vectors[i] / np.linalg.norm(chunk_vectors[i])
        chunk_index.add(chunk_vectors)
        faiss.write_index(chunk_index, self.chunk_index_path)
        
        # Create chunk ID map
        self.chunk_id_map_path = os.path.join(self.test_dir, "chunk_id_map.json")
        chunk_id_map = {
            "chunk1": 0,
            "chunk2": 1,
            "chunk3": 2,
            "chunk4": 3,
            "chunk5": 4
        }
        with open(self.chunk_id_map_path, 'w') as f:
            json.dump(chunk_id_map, f)
            
        # Create output path
        self.output_path = os.path.join(self.test_dir, "test_results.json")
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Create two random vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        vec3 = np.array([1.0, 1.0, 0.0])
        
        # Calculate cosine similarity
        sim12 = cosine_similarity(vec1, vec2)
        sim13 = cosine_similarity(vec1, vec3)
        
        # Check results
        self.assertAlmostEqual(sim12, 0.0)
        self.assertAlmostEqual(sim13, 1.0 / np.sqrt(2))
    
    def test_get_chunk_ids_for_paper(self):
        """Test retrieving chunk IDs for a paper."""
        # Get chunk IDs for paper
        chunk_ids = get_chunk_ids_for_paper("2201.00001", self.db_path)
        
        # Check results
        self.assertEqual(len(chunk_ids), 2)
        self.assertIn("chunk1", chunk_ids)
        self.assertIn("chunk2", chunk_ids)
        
        # Test with non-existent paper
        chunk_ids = get_chunk_ids_for_paper("non_existent", self.db_path)
        self.assertEqual(len(chunk_ids), 0)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_embedding_model(self, mock_model, mock_tokenizer):
        """Test embedding model."""
        # Mock tokenizer and model
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # Mock model output
        mock_outputs = MagicMock()
        mock_hidden_state = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
        mock_outputs.last_hidden_state = mock_hidden_state
        mock_model.return_value.return_value = mock_outputs
        
        # Create embedding model
        model = EmbeddingModel("test_model")
        
        # Test embed method
        embedding = model.embed("test text")
        
        # Check results
        self.assertEqual(embedding.shape, (384,))
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=6)
    
    def test_faiss_index_wrapper(self):
        """Test FAISS index wrapper."""
        # Create wrapper
        wrapper = FaissIndexWrapper(self.metadata_index_path, self.metadata_id_map_path)
        
        # Test id_to_index
        idx = wrapper.id_to_index("2201.00001")
        self.assertEqual(idx, 0)
        
        # Test index_to_id
        id_str = wrapper.index_to_id(0)
        self.assertEqual(id_str, "2201.00001")
        
        # Test get_vector
        vector = wrapper.get_vector(0)
        self.assertEqual(vector.shape, (self.mock_dim,))
        
        # Test search
        query = np.random.rand(self.mock_dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        distances, indices = wrapper.search(query, k=2)
        self.assertEqual(distances.shape, (1, 2))
        self.assertEqual(indices.shape, (1, 2))
    
    def test_save_similarity_results(self):
        """Test saving similarity results."""
        # Create mock results
        top_chunks_topic = [{"chunk_id": "chunk1", "sim_topic_chunk": 0.9}]
        top_chunks_subtopic = [{"chunk_id": "chunk2", "sim_subtopic_chunk": 0.8}]
        top_chunks_enhanced = [{"chunk_id": "chunk3", "sim_enhanced_chunk": 0.7}]
        
        # Save results
        save_similarity_results(
            top_chunks_topic,
            top_chunks_subtopic,
            top_chunks_enhanced,
            self.output_path
        )
        
        # Check if file exists
        self.assertTrue(os.path.exists(self.output_path))
        
        # Load results
        with open(self.output_path, 'r') as f:
            results = json.load(f)
        
        # Check results
        self.assertEqual(len(results["top_chunks_topic"]), 1)
        self.assertEqual(len(results["top_chunks_subtopic"]), 1)
        self.assertEqual(len(results["top_chunks_enhanced"]), 1)
        self.assertEqual(results["top_chunks_topic"][0]["chunk_id"], "chunk1")
        self.assertEqual(results["top_chunks_subtopic"][0]["chunk_id"], "chunk2")
        self.assertEqual(results["top_chunks_enhanced"][0]["chunk_id"], "chunk3")
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_select_chunks_by_similarity(self, mock_model, mock_tokenizer):
        """Test selecting chunks by similarity."""
        # Mock embedding model
        with patch('torch.no_grad'):
            with patch.object(EmbeddingModel, 'embed') as mock_embed:
                # Mock embed method to return our pre-defined vectors
                mock_embed.side_effect = [
                    self.mock_topic_vector,
                    self.mock_subtopic_vector,
                    self.mock_enhanced_vector
                ]
                
                # Run selection
                top_chunks_topic, top_chunks_subtopic, top_chunks_enhanced = select_chunks_by_similarity(
                    topic_query=self.topic_query,
                    subtopic_query=self.subtopic_query,
                    enhanced_query=self.enhanced_query,
                    preselected_papers=self.preselected_papers,
                    threshold_metadata_score=0.0,  # Set to 0 to include all papers
                    top_k_topic=2,
                    top_m_subtopic=2,
                    top_n_enhanced=2,
                    metadata_index_path=self.metadata_index_path,
                    chunk_index_path=self.chunk_index_path,
                    chunk_db_path=self.db_path,
                    model_name="test_model"
                )
                
                # Check results
                self.assertEqual(len(top_chunks_topic), 2)
                self.assertEqual(len(top_chunks_subtopic), 2)
                self.assertEqual(len(top_chunks_enhanced), 2)
                
                # Check that each result has the expected fields
                for result in top_chunks_topic:
                    self.assertIn("chunk_id", result)
                    self.assertIn("arxiv_id", result)
                    self.assertIn("sim_topic_metadata", result)
                    self.assertIn("sim_topic_chunk", result)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_run_chunk_similarity_selection(self, mock_model, mock_tokenizer):
        """Test running the full chunk similarity selection process."""
        # Mock embedding model
        with patch('torch.no_grad'):
            with patch.object(EmbeddingModel, 'embed') as mock_embed:
                # Mock embed method to return our pre-defined vectors
                mock_embed.side_effect = [
                    self.mock_topic_vector,
                    self.mock_subtopic_vector,
                    self.mock_enhanced_vector
                ]
                
                # Mock select_chunks_by_similarity
                with patch('chunk_similarity_selection.select_chunks_by_similarity') as mock_select:
                    mock_top_chunks_topic = [{"chunk_id": "chunk1", "sim_topic_chunk": 0.9}]
                    mock_top_chunks_subtopic = [{"chunk_id": "chunk2", "sim_subtopic_chunk": 0.8}]
                    mock_top_chunks_enhanced = [{"chunk_id": "chunk3", "sim_enhanced_chunk": 0.7}]
                    mock_select.return_value = (
                        mock_top_chunks_topic,
                        mock_top_chunks_subtopic,
                        mock_top_chunks_enhanced
                    )
                    
                    # Mock save_similarity_results
                    with patch('chunk_similarity_selection.save_similarity_results') as mock_save:
                        # Run the function
                        top_chunks_topic, top_chunks_subtopic, top_chunks_enhanced = run_chunk_similarity_selection(
                            topic_query=self.topic_query,
                            subtopic_query=self.subtopic_query,
                            enhanced_query=self.enhanced_query,
                            preselected_papers=self.preselected_papers
                        )
                        
                        # Check results
                        self.assertEqual(top_chunks_topic, mock_top_chunks_topic)
                        self.assertEqual(top_chunks_subtopic, mock_top_chunks_subtopic)
                        self.assertEqual(top_chunks_enhanced, mock_top_chunks_enhanced)
                        
                        # Check that save_similarity_results was called
                        mock_save.assert_called_once()


if __name__ == '__main__':
    unittest.main()
