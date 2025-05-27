"""
module_paper_search/submodule_answer_generation/answer_generation.py

This module handles the generation and evaluation of citation-backed answers
using the top-n summarized chunks from the previous pipeline steps.
"""

import logging
import os
import json
import datetime
import openai
from openai import OpenAI
from typing import Dict, List, Optional, Tuple, Any, Set

# Import output directory from the central module
from module_paper_search.output_directories import ANSWER_OUTPUT_DIR

# Import parameters
from module_paper_search.submodule_answer_generation.answer_generation_parameters import (
    N_ANSWER,
    ANSWER_LENGTH_MIN,
    ANSWER_LENGTH_MAX,
    GENERATION_MODEL,
    EVALUATION_MODEL,
    GENERATION_SYSTEM_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    DEFAULT_OPENAI_API_KEY
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('answer_generation')


def get_openai_api_key() -> str:
    """
    Get the OpenAI API key from environment variables.
    
    Returns:
        str: The OpenAI API key
    """
    # Get from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
        
    return api_key


def select_top_chunks(summarized_chunks: List[Dict[str, Any]], n: int = N_ANSWER) -> List[Dict[str, Any]]:
    """
    Select the top n chunks from summarized_chunks based on final_weight_adjusted.
    
    Args:
        summarized_chunks: List of dictionaries containing chunk information
        n: Number of top chunks to select
        
    Returns:
        List of the top n chunks
    """
    # Sort chunks by final_weight_adjusted in descending order and take top n
    sorted_chunks = sorted(
        summarized_chunks, 
        key=lambda x: x.get('final_weight_adjusted', 0), 
        reverse=True
    )
    return sorted_chunks[:n]


def build_context(top_chunks: List[Dict[str, Any]], citation_key_map: Dict[str, str], use_arxiv_id: bool = True) -> str:
    """
    Build context from top chunks with citation keys.
    
    Args:
        top_chunks: List of top chunks to include in context
        citation_key_map: Dictionary mapping chunk_id to citation key
        use_arxiv_id: Whether to use arXiv ID as citation key (default: True)
        
    Returns:
        String containing formatted context with citation keys
    """
    context_lines = []
    
    for chunk in top_chunks:
        chunk_id = chunk.get('chunk_id')
        summary = chunk.get('llm_summary', '')
        arxiv_id = chunk.get('arxiv_id', '')
        
        # Use the arXiv ID as the citation key if available and requested
        if use_arxiv_id and arxiv_id:
            citation_key = arxiv_id
        else:
            # Fall back to the citation_key_map
            citation_key = citation_key_map.get(chunk_id, f"Cite{len(context_lines)+1}")
        
        # Format the context line with citation key
        context_line = f"[{citation_key}] {summary}"
        context_lines.append(context_line)
    
    return "\n".join(context_lines)


def generate_answer(
    context: str, 
    enhanced_query: str,
    previous_answer: Optional[str] = None,
    previous_score: Optional[float] = None,
    model: str = GENERATION_MODEL,
    temperature: float = 0.3,
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate a citation-backed answer using the provided context.
    
    Args:
        context: String containing formatted context with citation keys
        enhanced_query: The enhanced query to answer
        previous_answer: Optional previous answer to improve upon
        previous_score: Optional score of the previous answer
        model: Model to use for generation
        temperature: Temperature to use for generation
        system_prompt: System prompt to use for generation (if None, use default)
        
    Returns:
        Generated answer text
    """
    # Use provided system prompt or default
    if system_prompt is None:
        system_prompt = GENERATION_SYSTEM_PROMPT
        
    # Build system prompt with previous answer if available
    if previous_answer and previous_score is not None:
        system_prompt += f"\n\nA previous answer was: \"{previous_answer}\" (score={previous_score}). Improve it using the context provided."
    
    # Build user prompt
    user_prompt = f"Context:\n{context}\n\nQuestion: {enhanced_query}\n\nAnswer ({ANSWER_LENGTH_MIN},{ANSWER_LENGTH_MAX} words):"
    
    try:
        # Get the OpenAI API key
        api_key = get_openai_api_key()
        
        # Initialize the OpenAI client with the API key
        client = OpenAI(api_key=api_key)
        
        # Call the OpenAI API with appropriate parameters based on model
        # For the 'o3' model, we need to handle parameters differently
        if model.lower() == "o3":
            logger.info(f"Using o3 model with default parameters")
            # For o3, don't specify temperature or max_completion_tokens
            # as it only supports default values
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
        else:
            # For other models like GPT-4, use the standard parameters
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=1500
            )
        
        answer_text = response.choices[0].message.content.strip()
        return answer_text
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error generating answer: {e}"


def evaluate_answer(context: str, enhanced_query: str, answer_text: str) -> Tuple[float, str]:
    """
    Evaluate the quality of the generated answer.
    
    Args:
        context: String containing formatted context with citation keys
        enhanced_query: The enhanced query being answered
        answer_text: The generated answer to evaluate
        
    Returns:
        Tuple[float, str]: A tuple containing:
            - Float score between 0 and 1 indicating answer quality
            - String containing the full evaluation explanation
    """
    # Build system prompt
    system_prompt = EVALUATION_SYSTEM_PROMPT
    
    # Build user prompt
    user_prompt = f"Question: {enhanced_query}\n\nContext:\n{context}\n\nAnswer:\n{answer_text}"
    
    try:
        # Get the OpenAI API key
        api_key = get_openai_api_key()
        
        # Initialize the OpenAI client with the API key
        client = OpenAI(api_key=api_key)
        
        # Call the OpenAI API with appropriate parameters based on model
        if EVALUATION_MODEL.lower() == "o3":
            logger.info(f"Using o3 model for evaluation with default parameters")
            # For o3, don't specify temperature or max_tokens
            response = client.chat.completions.create(
                model=EVALUATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
        else:
            # For other models like GPT-4, use the standard parameters
            response = client.chat.completions.create(
                model=EVALUATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=200  # Increased to capture the full explanation
            )
        
        # Extract the full response
        full_evaluation = response.choices[0].message.content.strip()
        logger.info(f"Raw evaluation response: {full_evaluation}")
        
        # Try to parse the score as a float
        try:
            # First try direct conversion (in case the LLM follows instructions exactly)
            try:
                score = float(full_evaluation)
                # If we get here, the full_evaluation is just a number with no explanation
                explanation = f"Score: {score}"
            except ValueError:
                # If that fails, try to extract a number using regex
                import re
                # Look for patterns like "Rating: 0.9" or just a number like "0.9"
                score_match = re.search(r'(?:rating|score)?\s*:?\s*(\d+(?:\.\d+)?)', full_evaluation, re.IGNORECASE)
                if score_match:
                    score = float(score_match.group(1))
                    explanation = full_evaluation  # Keep the full explanation
                else:
                    raise ValueError(f"Could not extract score from: {full_evaluation}")
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))
            logger.info(f"Extracted score: {score}")
            return score, explanation
        except ValueError as e:
            logger.warning(f"Could not parse score from response: {full_evaluation}. Error: {e}")
            return 0.5, f"Failed to parse score. Raw evaluation: {full_evaluation}"
        
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        return 0.5, f"Error during evaluation: {str(e)}"


def run_answer_generation(
    summarized_chunks: List[Dict[str, Any]],
    enhanced_query: str,
    citation_key_map: Dict[str, str],
    n_answer: int = N_ANSWER,
    previous_answer: Optional[str] = None,
    previous_score: Optional[float] = None,
    model: str = GENERATION_MODEL,
    temperature: float = 0.3,
    system_prompt: Optional[str] = None
) -> Tuple[str, float, str, List[Dict[str, Any]]]:
    """
    Main function to run the answer generation and evaluation process.
    
    Args:
        summarized_chunks: List of dictionaries containing chunk information
        enhanced_query: The enhanced query to answer
        citation_key_map: Dictionary mapping chunk_id to citation key
        n_answer: Number of top chunks to use for answer generation
        previous_answer: Optional previous answer to improve upon
        previous_score: Optional score of the previous answer
        model: Model to use for generation
        temperature: Temperature to use for generation
        system_prompt: System prompt to use for generation (if None, use default)
        
    Returns:
        Tuple containing (answer_text, answer_quality_score, evaluation_explanation, top_chunks)
    """
    logger.info("Starting answer generation process")
    
    # Step 1: Select top n chunks
    top_chunks = select_top_chunks(summarized_chunks, n=n_answer)
    logger.info(f"Selected top {len(top_chunks)} chunks for answer generation")
    
    # Step 2: Build context from top chunks
    context = build_context(top_chunks, citation_key_map)
    logger.info(f"Built context with {len(context.split('\\n'))} lines")
    
    # Step 3: Generate answer
    answer_text = generate_answer(
        context=context,
        enhanced_query=enhanced_query,
        previous_answer=previous_answer,
        previous_score=previous_score,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt
    )
    logger.info(f"Generated answer with {len(answer_text.split())} words")
    
    # Step 4: Evaluate answer quality
    answer_quality_score, evaluation_explanation = evaluate_answer(
        context=context,
        enhanced_query=enhanced_query,
        answer_text=answer_text
    )
    logger.info(f"Evaluated answer with quality score: {answer_quality_score}")
    logger.info(f"Full evaluation explanation: {evaluation_explanation}")
    
    return answer_text, answer_quality_score, evaluation_explanation, top_chunks


def save_attempt_details(
    attempt_number: int,
    answer_text: str, 
    answer_quality_score: float, 
    evaluation_explanation: str,
    enhanced_query: str,
    all_chunks: List[Dict[str, Any]],
    selected_chunks: List[Dict[str, Any]],
    used_chunk_ids: Set[str],
    output_dir: str = None
) -> None:
    """
    Save detailed information about a specific attempt in the answer sufficiency feedback loop.
    
    Args:
        attempt_number: The attempt number (0 for initial, 1 for first retry, etc.)
        answer_text: The generated answer
        answer_quality_score: The quality score of the answer
        evaluation_explanation: The full explanation from the LLM
        enhanced_query: The query that was answered
        all_chunks: All available chunks with their weights and scores
        selected_chunks: Chunks selected for this attempt
        used_chunk_ids: Set of chunk IDs that have been used in previous attempts
        output_dir: Directory to save the information in
    """
    # If no output directory is specified, use the default answer output directory
    if output_dir is None:
        output_dir = ANSWER_OUTPUT_DIR
        
    # Create attempt-specific directory
    attempt_dir = os.path.join(output_dir, f"attempt_{attempt_number}")
    os.makedirs(attempt_dir, exist_ok=True)
    
    # Create filenames as requested
    answer_txt_filename = os.path.join(attempt_dir, "answer.txt")
    answer_json_filename = os.path.join(attempt_dir, "answer.json")
    
    # Save the answer as a text file
    with open(answer_txt_filename, 'w') as f:
        f.write(f"Query: {enhanced_query}\n\n")
        f.write(f"Answer: {answer_text}\n\n")
        f.write(f"Quality Score: {answer_quality_score}\n")
        f.write(f"Evaluation Explanation:\n{evaluation_explanation}\n")
    
    # Create detailed output dictionary
    details = {
        "query": enhanced_query,
        "answer": answer_text,
        "quality_score": answer_quality_score,
        "evaluation_explanation": evaluation_explanation,
        "timestamp": str(datetime.datetime.now()),
        "attempt_number": attempt_number,
        "chunks_used": []
    }
    
    # Add detailed information about each chunk used
    for chunk in selected_chunks:
        chunk_info = {
            "chunk_id": chunk.get("chunk_id"),
            "arxiv_id": chunk.get("arxiv_id"),
            "final_weight_adjusted": chunk.get("final_weight_adjusted"),
            "llm_summary": chunk.get("llm_summary"),
            "topic_similarity": chunk.get("topic_similarity"),
            "subtopic_similarity": chunk.get("subtopic_similarity"),
            "enhanced_similarity": chunk.get("enhanced_similarity")
        }
        details["chunks_used"].append(chunk_info)
    
    # Save the detailed information as a JSON file
    with open(answer_json_filename, 'w') as f:
        json.dump(details, f, indent=2)
    
    logger.info(f"Saved attempt {attempt_number} answer to {answer_txt_filename}")
    logger.info(f"Saved attempt {attempt_number} detailed information to {answer_json_filename}")


def save_answer(
    answer_text: str, 
    answer_quality_score: float, 
    evaluation_explanation: str,
    enhanced_query: str,
    top_chunks: List[Dict[str, Any]],
    output_dir: str = None
) -> None:
    """
    Save the generated answer and its quality score to a file.
    
    Args:
        answer_text: The generated answer
        answer_quality_score: The quality score of the answer
        evaluation_explanation: The full explanation from the LLM
        enhanced_query: The query that was answered
        top_chunks: List of top chunks used to generate the answer
        output_dir: Directory to save the answer in
    """
    # If no output directory is specified, use the default answer output directory
    if output_dir is None:
        output_dir = ANSWER_OUTPUT_DIR
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filenames - simple names as requested
    answer_filename = os.path.join(output_dir, "answer.txt")
    details_filename = os.path.join(output_dir, "answer.json")
    
    # Save the answer as a text file
    with open(answer_filename, 'w') as f:
        f.write(f"Query: {enhanced_query}\n\n")
        f.write(f"Answer: {answer_text}\n\n")
        f.write(f"Quality Score: {answer_quality_score}\n")
        f.write(f"Evaluation Explanation:\n{evaluation_explanation}\n")
    
    # Create detailed output dictionary
    details = {
        "query": enhanced_query,
        "answer": answer_text,
        "quality_score": answer_quality_score,
        "evaluation_explanation": evaluation_explanation,
        "timestamp": str(datetime.datetime.now()),
        "chunks_used": []
    }
    
    # Add detailed information about each chunk used
    for chunk in top_chunks:
        chunk_info = {
            "chunk_id": chunk.get("chunk_id"),
            "arxiv_id": chunk.get("arxiv_id"),
            "final_weight_adjusted": chunk.get("final_weight_adjusted"),
            "llm_summary": chunk.get("llm_summary"),
            "topic_similarity": chunk.get("topic_similarity"),
            "subtopic_similarity": chunk.get("subtopic_similarity"),
            "enhanced_similarity": chunk.get("enhanced_similarity")
        }
        details["chunks_used"].append(chunk_info)
    
    # Save the detailed information as a JSON file
    with open(details_filename, 'w') as f:
        json.dump(details, f, indent=2)
    
    logger.info(f"Saved final answer to {answer_filename}")
    logger.info(f"Saved final answer detailed information to {details_filename}")
