"""
Parameters for the KGB-based midpoint finding module.

This module contains configuration parameters for the midpoint finding
process that identifies intermediate nodes between query pairs.
"""

import os
from pathlib import Path

# Import from main parameters
from ..main_parameters import ANSWER_MODE_DIR

# Use the answer mode directory for results
RESULTS_DIR = ANSWER_MODE_DIR

# LLM parameters
LLM_MODEL = "gpt-4-turbo"
LLM_MAX_TOKENS = 2000
LLM_TEMPERATURE = 0.2

# System prompt for the LLM
SYSTEM_PROMPT = """
You are a scientific research assistant specialized in knowledge graph construction and academic literature analysis.

Your task is to identify conceptual midpoints between pairs of research queries. These midpoints will serve as intermediate nodes in a knowledge graph that connects a starting concept (A) to a destination concept (B) through a meaningful path.

The midpoints you identify should:
1. Represent logical stepping stones between the concepts in each query pair
2. Be specific enough to guide retrieval of relevant academic literature
3. Form a coherent path that gradually bridges the conceptual gap between the starting and ending points
4. Be formulated as clear, searchable queries similar in format to the input queries

Your midpoints will be used with a Retrieval-Augmented Generation (RAG) system to find academic papers that establish connections between concepts. The goal is to create a knowledge path that allows for deep understanding of how concepts A and B are related through intermediate concepts.
"""

# Output file paths
MIDPOINTS_JSON_PATH = os.path.join(RESULTS_DIR, "midpoints.json")
