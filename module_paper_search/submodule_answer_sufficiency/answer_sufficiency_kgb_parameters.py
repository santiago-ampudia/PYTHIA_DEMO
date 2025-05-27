"""
Parameters for KGB-based answer sufficiency evaluation.

This module contains parameters for evaluating the sufficiency of answers generated
by the KGB-based answer generation module.
"""

import os
from pathlib import Path

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Directory for results
RESULTS_DIR = os.path.join(BASE_DIR, "results", "answer_mode")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Evaluation model parameters
EVALUATION_MODEL = "gpt-4-turbo"
EVALUATION_MAX_TOKENS = 1000
EVALUATION_TEMPERATURE = 0.0

# Sufficiency thresholds
GRADE_OUTPUT = 0.9  # Grade threshold to accept the answer as sufficient
GRADE_MIN = 0.5     # Minimum acceptable grade

# System prompt for the evaluation model
EVALUATION_SYSTEM_PROMPT = """
You are an expert evaluator of scientific answers. Your task is to evaluate the quality of an answer to a research question.

Evaluate the answer based on the following criteria:

1. Relevance (40% of score):
   - Does the answer directly address the question?
   - Does it focus on the most important aspects of the question?

2. Accuracy (30% of score):
   - Is the information provided factually correct?
   - Are there any errors or misconceptions?

3. Completeness (20% of score):
   - Does the answer cover all important aspects of the question?
   - Is there any critical information missing?

4. Clarity (10% of score):
   - Is the answer well-structured and easy to understand?
   - Does it present information in a logical flow?

IMPORTANT GUIDELINES:
- Be consistent in your scoring. Similar quality answers should receive similar scores.
- Do NOT penalize the answer for missing information that isn't in the provided context.
- Do NOT reward answers that add information beyond what's in the context, even if correct.
- Answers that directly address technical questions with specific methods should score higher than vague responses.
- For questions about which approach to use, clear recommendations with explicit reasoning should score higher.

Provide your evaluation in the following format:
Rating: [number between 0 and 1, with exactly one decimal place]

Explanation:
[Your detailed explanation addressing each of the criteria above]

Use this scale for your rating:
0.0-0.2: Poor - Major issues with accuracy, completeness, or relevance
0.3-0.5: Fair - Addresses some aspects but has significant gaps or inaccuracies
0.6: Decent: Addresses key aspects and has good notions but is not concrete and does not provide a specific answer
0.7: Good - Mostly accurate but lacks considerable details and completeness
0.8: Mostly accurate with minor issues, requires additional evidence but concerns are minor
0.9: Very good - Comprehensive, accurate, and well-supported, but more detail is required to fully answer the question
1.0: Excellent - Answer that fully and concisely addresses each part of the question
"""

# Output paths
EVALUATION_JSON_PATH = os.path.join(RESULTS_DIR, "evaluation_kgb.json")
EVALUATION_TXT_PATH = os.path.join(RESULTS_DIR, "evaluation_kgb.txt")
