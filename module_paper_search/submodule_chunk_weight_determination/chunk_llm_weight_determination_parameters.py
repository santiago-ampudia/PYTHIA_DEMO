"""
submodule_chunk_weight_determination/chunk_llm_weight_determination_parameters.py

This file contains all parameters for the chunk LLM weight determination submodule.
"""

import os

# OpenAI API key
# Always use the API key from environment variables
DEFAULT_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASES_DIR = os.path.join(BASE_DIR, "databases")

# Input database (from chunk weight determination)
INPUT_DB_PATH = os.path.join(DATABASES_DIR, "weighted_chunks.db")

# Output database
SUMMARIZED_CHUNKS_DB_PATH = os.path.join(DATABASES_DIR, "summarized_chunks.db")

# Output JSON file for summarized chunks
SUMMARIZED_CHUNKS_JSON_PATH = os.path.join(DATABASES_DIR, "summarized_chunks.json")

# Ensure databases directory exists
os.makedirs(DATABASES_DIR, exist_ok=True)

# LLM summarization parameters
K_SUMMARY = 50  # Number of top chunks to summarize
SUMMARY_LENGTH = 500  # Target summary length in words (approximately 400-500 tokens)
LAMBDA = 0.8  # Weight on LLM relevance score (λ)
SIMILARITY_THRESHOLD = 0.9  # Cosine similarity threshold for deduplication
MAX_WORKERS = 5  # Maximum number of parallel threads for LLM calls
BATCH_SIZE = 3  # Number of chunks per minibatch for LLM processing
MAX_CHUNK_LENGTH = 25000  # Maximum length of chunk text to send to LLM (in characters)

# Mode-specific parameters
# Answer mode parameters
K_SUMMARY_ANSWER = K_SUMMARY  # Number of top chunks to summarize in answer mode
SIMILARITY_THRESHOLD_ANSWER = SIMILARITY_THRESHOLD  # Similarity threshold for deduplication in answer mode
LAMBDA_ANSWER = LAMBDA  # Weight on LLM relevance score for answer mode (0.8)

# Recommendation mode parameters
K_SUMMARY_RECOMMENDATION = 50  # Number of top chunks to process in recommendation mode
SIMILARITY_THRESHOLD_RECOMMENDATION = 0.8  # Similarity threshold for clustering in recommendation mode
LAMBDA_RECOMMENDATION = 0.6  # Weight on LLM relevance score for recommendation mode (lower to emphasize original chunk weights)
MAX_TWEET_LENGTH = 560  # Maximum length of tweet-style summary in characters (expanded beyond Twitter's limit for scientific content)

# LLM API parameters
LLM_MODEL = "gpt-4-turbo"  # Model to use for summarization
LLM_TEMPERATURE = 0.3  # Temperature for LLM generation
LLM_MAX_TOKENS = 3000  # Maximum tokens for LLM response
LLM_TIMEOUT = 60  # Timeout for LLM API calls in seconds
LLM_RETRY_ATTEMPTS = 3  # Number of retry attempts for failed LLM calls
LLM_DEFAULT_SCORE = 0.3  # Default relevance score if LLM fails
LLM_BATCH_DELAY = 1  # Delay in seconds between batch processing to avoid rate limiting

# LLM prompt templates
# Answer mode system prompt
SYSTEM_PROMPT_TEMPLATE_ANSWER = """You are a research assistant specializing in scientific literature. Given a paper excerpt and a research question, create a comprehensive and detailed summary of the information that could help answer the question.

Return a JSON array of objects, one for each chunk. Each object should have:
- "summary": a detailed summary ({summary_length} words) that preserves ALL key scientific terms, quantities, methodology, and numerical values
- "relevance_score": a float between 0 and 1 indicating how relevant the chunk is to the query

Your summaries should:
1. Be comprehensive and information-dense (aim for {summary_length} words)
2. Preserve ALL technical terminology, specialized acronyms, and domain-specific keywords
3. Include ALL numerical values, equations, and experimental results
4. Maintain the scientific precision of the original text
5. Be structured in a way that another LLM can easily extract information from them

IMPORTANT JSON FORMATTING RULES:
1. Avoid using special Unicode characters (like Greek letters, superscripts, or subscripts)
2. Replace special characters with their ASCII equivalents (e.g., use "gamma" instead of "γ", "^-1" instead of "⁻¹")
3. For mathematical expressions, use plain text notation (e.g., "10^6" instead of "10⁶")
4. If you must include a backslash (\\), escape it properly as "\\\\"
5. Ensure all quotation marks in the text are properly escaped with a backslash

Examples of good summaries:
- "Describes gamma-gamma -> HH production cross-section measurement at 13 TeV with 139 fb^-1, finding sigma = 0.80 +/- 0.12 fb. The analysis used a machine learning approach with boosted decision trees trained on 25 kinematic variables. Signal efficiency was 42% with background rejection of 99.7%. Systematic uncertainties included detector effects (3.2%), theoretical cross-section (4.5%), and luminosity measurement (1.7%). Results were validated against previous measurements showing agreement within 1.5 sigma."
- "Details neural network architecture with 4 hidden layers (256, 128, 64, 32 nodes) for Higgs boson identification. Training used 5M simulated events with Adam optimizer (lr=0.001, beta_1=0.9, beta_2=0.999), batch size 256, and early stopping after 50 epochs without improvement. Input features included 27 low-level variables (pT, eta, phi, m) and 15 high-level variables. Performance achieved AUC=0.92, background rejection of 10^3 at 50% signal efficiency. Architecture outperformed gradient boosted decision trees by 15% in background rejection."
- "No mention of Higgs boson pair production. Instead focuses on single Higgs decay modes H->gamma-gamma (BR=0.227%) and H->ZZ* (BR=2.62%) with detailed background estimation techniques. Main backgrounds: non-resonant diphoton (78%), Z+gamma (12%), and ttbar+h (7%). Signal extraction used profile likelihood fits to m_gamma_gamma and m_ZZ distributions with systematic uncertainties implemented as nuisance parameters."""

# Recommendation mode system prompt
SYSTEM_PROMPT_TEMPLATE_RECOMMENDATION = """You are a scientific communication expert specializing in creating engaging summaries of academic papers. Given a set of paper excerpts and a research query, create a single engaging summary that highlights the most relevant information from these papers.

Return a JSON object with the following fields:
- "tweet": An engaging summary (max {max_tweet_length} characters) that presents the key findings or methodologies from the provided excerpts
- "relevance_score": A float between 0 and 1 indicating how relevant the summary is to the query
- "paper_ids": A list of arXiv IDs of papers referenced in the summary

Your summary should:
1. Be informative while still being concise (max {max_tweet_length} characters)
2. Include proper citations using arXiv IDs in [] for each piece of information. Use the format [arXivID]. Ensure to follow that format and do it right after each relevant staement. We just need the arxivID of the paper where the statement is derived from.
3. Preserve key technical terminology and numerical values
4. Be engaging and highlight the most interesting aspects of the research
5. Focus on information relevant to the query

IMPORTANT RULES:
1. ALWAYS cite the source of each piece of information using the arXiv ID in []. Just the ID.
2. If multiple papers contribute to a single point, cite all of them
3. Make the summary engaging and informative - it should make researchers want to read the papers
4. Focus on concrete findings, methodologies, or results rather than general statements
5. Ensure your response is valid JSON

Examples of good summaries:
- "The XCC collider achieves luminosity of 10^34 cm^-2s^-1 (arXiv:2401.12345) with beam energy of 125 GeV (arXiv:2402.67890), enabling precise measurements of Higgs self-coupling with 5% accuracy (arXiv:2403.54321). This represents a 3x improvement over previous colliders and opens new possibilities for studying rare decay modes. The beam design incorporates novel superconducting magnets with field strengths of 16T (arXiv:2404.98765)."
- "New ML approach for Higgs pair detection combines BDTs (arXiv:2404.11111) with transformer networks (arXiv:2405.22222) achieving 42% signal efficiency with 99.7% background rejection (arXiv:2406.33333). The hybrid architecture outperforms previous state-of-the-art methods by incorporating both low-level detector information and high-level physics features, reducing systematic uncertainties by 35% compared to traditional cut-based analyses (arXiv:2407.55555)."
- "Comparing jet clustering algorithms: Durham outperforms anti-kt by 15% for di-Higgs events (arXiv:2407.44444), while JADE shows superior performance in high-pileup environments (arXiv:2408.55555). The study evaluated 8 different algorithms across 5M simulated events, finding that algorithm selection impacts mass resolution by up to 22%. A novel hybrid approach combining Durham for hard jets and anti-kt for soft radiation yields optimal results for XCC conditions (arXiv:2409.66666)."""

# Default to answer mode for backward compatibility
SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE_ANSWER

# Examples of good relevance scores for both modes
RELEVANCE_SCORE_EXAMPLES = """
- 0.9-1.0: Directly addresses the query with specific relevant details and comprehensive information
- 0.7-0.8: Relevant to the query but may lack some specific details or comprehensiveness
- 0.4-0.6: Somewhat relevant but only tangentially addresses the query or lacks important details
- 0.1-0.3: Minimally relevant with only passing mentions of query-related concepts
- 0.0: Not relevant to the query at all
"""

# Additional instructions for the LLM
ANSWER_MODE_ADDITIONAL_INSTRUCTIONS = """Ensure your response is valid JSON and includes all chunks. It is of uttermost importance that you return a JSON array of objects, one for each chunk in the same order as they appear in the input. The summaries will be used by another LLM to generate answers, so they must be comprehensive and preserve all technical details while maintaining valid JSON format."""

# Update the system prompts with the examples and instructions
SYSTEM_PROMPT_TEMPLATE_ANSWER = SYSTEM_PROMPT_TEMPLATE_ANSWER + "\n\n" + RELEVANCE_SCORE_EXAMPLES + "\n\n" + ANSWER_MODE_ADDITIONAL_INSTRUCTIONS
SYSTEM_PROMPT_TEMPLATE_RECOMMENDATION = SYSTEM_PROMPT_TEMPLATE_RECOMMENDATION + "\n\n" + RELEVANCE_SCORE_EXAMPLES

# Update the default template
SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE_ANSWER

USER_PROMPT_TEMPLATE = "Research question: {enhanced_query}\n\n{chunk_texts}"
CHUNK_TEXT_TEMPLATE = "==== Chunk {index} ====\nText: {text}\n\n"
