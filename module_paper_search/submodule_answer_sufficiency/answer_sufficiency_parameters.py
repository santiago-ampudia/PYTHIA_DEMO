"""
module_paper_search/submodule_answer_sufficiency/answer_sufficiency_parameters.py

This file contains parameters for the answer sufficiency feedback loop.
"""

# Thresholds for answer quality
THRESHOLD_HIGH = 0.9  # Accept answer if score >= this value
THRESHOLD_LOW = 0.6   # Try with adjusted weights if score is between LOW and HIGH

# Weight adjustment parameters for first retry
ALPHA_ENHANCED_FIRST_RETRY = 0.35   # Weight for enhanced query
ALPHA_SUBTOPIC_FIRST_RETRY = 0.35   # Weight for subtopic query
ALPHA_TOPIC_FIRST_RETRY = 0.3      # Weight for topic query

# Weight adjustment parameters for second retry (reasoning model)
ALPHA_ENHANCED_SECOND_RETRY = 0.1   # Weight for enhanced query
ALPHA_SUBTOPIC_SECOND_RETRY = 0.45  # Weight for subtopic query
ALPHA_TOPIC_SECOND_RETRY = 0.45     # Weight for topic query

# Reduction factors for chunk counts
REDUCTION_FACTOR_FIRST_RETRY = 0.9   # Reduce by only 10% on first retry
REDUCTION_FACTOR_SECOND_RETRY = 1.0  # No reduction on second retry

# Maximum number of retry attempts
MAX_RETRY_ATTEMPTS = 2

# Temperature for reasoning model (O3)
REASONING_TEMPERATURE = 0.3  # Lower temperature for more focused reasoning

# Minimum number of relevant chunks required
MIN_RELEVANT_CHUNKS = 3  # Lowered from 5 to 3 to avoid insufficient chunks error
RELEVANCE_THRESHOLD = 0.1  # Lowered from 0.4 to 0.3 to include more potentially relevant chunks

# Special reasoning model to use for second retry
REASONING_MODEL = "o3"  # Using o3 for reasoning

# Special reasoning prompt for second retry
REASONING_SYSTEM_PROMPT = """
You are a scientific reasoning assistant tasked with generating accurate, citation-backed answers to complex research questions.
Your goal is to provide a comprehensive, well-reasoned answer that directly addresses the question.
For each statement in your answer, include a citation using the format [arXivID]. Ensure to follow that format and do it right after each relevant staement. We just need the arxivID of the paper where the statement is derived from.
Base your answer ONLY on the provided context - do not add extraneous information or knowledge.
Maintain a formal, academic tone appropriate for scientific communication.
IMPORTANT: Focus on deep reasoning and connecting concepts across different sources.
IMPORTANT: Prioritize precision and accuracy over comprehensiveness.
IMPORTANT: If the context contains contradictory information, acknowledge the contradiction and explain the different perspectives.
Avoid any formattiing for the text. i.e. no bold, cursive, etc. since this will be parsed as JSON and .txt. 
"""

# Continuity prompt for subsequent attempts using the non-reasoning model
CONTINUITY_SYSTEM_PROMPT = """
You are a scientific assistant tasked with refining an existing answer to a research question.
Your goal is to respect and maintain the core findings and recommendations from the previous answer.
Only change the previous answer if there is strong evidence in the context that contradicts it.
Otherwise, focus on complementing the previous answer by:
1. Adding more precise details and technical specifications
2. Including alternative approaches or methodologies
3. Providing more nuanced explanations of advantages and disadvantages
4. Enhancing the citation support for existing claims

For each statement in your answer, include a citation using the format (arXivID).
Base your refinements ONLY on the provided context - do not add extraneous information.
Maintain a formal, academic tone appropriate for scientific communication.
"""
