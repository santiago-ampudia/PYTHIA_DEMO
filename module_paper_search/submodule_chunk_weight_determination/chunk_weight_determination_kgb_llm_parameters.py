"""
Parameters for KGB LLM-based chunk weight determination.

This module contains the parameters for the KGB LLM-based chunk weight determination process.
"""

import os
from pathlib import Path

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Directory for databases
DATABASES_DIR = os.path.join(BASE_DIR, "databases")

# Input database path (from KGB chunk weight determination)
INPUT_DB_PATH = os.path.join(DATABASES_DIR, "weighted_chunks_kgb.db")

# Output database path
WEIGHTED_CHUNKS_DB_PATH = os.path.join(DATABASES_DIR, "weighted_chunks_kgb_llm.db")

# Output JSON file for weighted chunks
WEIGHTED_CHUNKS_JSON_PATH = os.path.join(DATABASES_DIR, "weighted_chunks_kgb_llm.json")

# Ensure databases directory exists
os.makedirs(DATABASES_DIR, exist_ok=True)

# Number of top chunks to select per query for LLM processing
TOP_K_CHUNKS_PER_QUERY = 5

# Batch size for LLM processing
LLM_BATCH_SIZE = 5

# Weight factors for combining scores
ORIGINAL_SCORE_WEIGHT = 0.6  # Weight for the original similarity-based score
LLM_SCORE_WEIGHT = 0.4       # Weight for the LLM-assigned relevance score

# LLM parameters
LLM_MODEL = "gpt-4-turbo"
LLM_TEMPERATURE = 0.0  # Low temperature for more deterministic outputs
LLM_MAX_TOKENS = 2000  # Maximum tokens for response

# System prompt for the LLM
SYSTEM_PROMPT = """You are a highly specialized scientific research assistant tasked with summarizing and evaluating scientific paper chunks for relevance to a research question.

For each chunk of text from a scientific paper, you will:
1. Create a precise, technical summary that captures the essence of the chunk
2. Assign a relevance score (0.0 to 1.0) indicating how useful this chunk would be for answering the research question

Guidelines for summaries:
- Maintain all technical terminology, scientific terms, quantities, and specific data
- Use scientific paper style writing (formal, precise, technical)
- Focus ONLY on what is explicitly stated in the chunk (do not add external knowledge)
- Do not reference other chunks in the batch
- Ensure the summary is comprehensive yet concise
- Preserve all mathematical formulas, equations, and numerical values

Guidelines for relevance scoring:
- Consider multi-hop reasoning: chunks that provide useful intermediate information should receive high scores even if they don't directly answer the question
- Evaluate if the chunk contains information that would be useful in constructing a path to the answer
- Use the following scale:
  - 0.9-1.0: Crucial information that directly addresses key aspects of the question or provides essential connecting information
  - 0.7-0.8: Important information that significantly contributes to understanding or connecting concepts in the question
  - 0.4-0.6: Moderately useful information that provides some relevant context or partial connections
  - 0.1-0.3: Minimally useful information with limited relevance or very indirect connections
  - 0.0: Not useful at all for answering the question

Your response must be in valid JSON format with the following structure:
[
  {
    "chunk_id": "chunk identifier",
    "summary": "technical summary of the chunk",
    "relevance_score": 0.85,
    "reasoning": "brief explanation of why this score was assigned"
  },
  ...
]
"""

# User prompt template
USER_PROMPT_TEMPLATE = """Research question: {enhanced_query}

Please analyze the following chunks from scientific papers, create technical summaries, and assign relevance scores based on how useful each chunk would be for answering the research question.

{chunk_texts}

Remember to focus only on the content within each chunk, maintain all technical terminology, and consider multi-hop reasoning when assigning relevance scores.
"""

# Chunk text template
CHUNK_TEXT_TEMPLATE = """=== Chunk {index} (ID: {chunk_id}) ===
{text}

"""
