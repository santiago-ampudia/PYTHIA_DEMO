"""
module_paper_search/submodule_answer_sufficiency/answer_sufficiency.py

This module implements the answer sufficiency feedback loop that evaluates
the quality of generated answers and takes appropriate actions based on that evaluation.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set

# Import output directory from the central module
from module_paper_search.output_directories import ANSWER_OUTPUT_DIR

# Import parameters
from module_paper_search.submodule_answer_sufficiency.answer_sufficiency_parameters import (
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
    ALPHA_ENHANCED_FIRST_RETRY,
    ALPHA_SUBTOPIC_FIRST_RETRY,
    ALPHA_TOPIC_FIRST_RETRY,
    ALPHA_ENHANCED_SECOND_RETRY,
    ALPHA_SUBTOPIC_SECOND_RETRY,
    ALPHA_TOPIC_SECOND_RETRY,
    CONTINUITY_SYSTEM_PROMPT,
    REDUCTION_FACTOR_FIRST_RETRY,
    REDUCTION_FACTOR_SECOND_RETRY,
    MAX_RETRY_ATTEMPTS,
    MIN_RELEVANT_CHUNKS,
    RELEVANCE_THRESHOLD,
    REASONING_MODEL,
    REASONING_SYSTEM_PROMPT,
    REASONING_TEMPERATURE
)

# Import answer generation functions
from module_paper_search.submodule_answer_generation.answer_generation import run_answer_generation, save_attempt_details
from module_paper_search.submodule_answer_generation.answer_generation_parameters import GENERATION_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('answer_sufficiency')


def check_chunk_pool_sufficiency(
    similarity_scores: List[Dict[str, Any]],
    threshold: float = RELEVANCE_THRESHOLD,
    min_chunks: int = MIN_RELEVANT_CHUNKS
) -> bool:
    """
    Check if there are enough relevant chunks in the pool.
    
    Args:
        similarity_scores: List of dictionaries containing similarity scores
        threshold: Minimum similarity score to consider a chunk relevant
        min_chunks: Minimum number of relevant chunks required
        
    Returns:
        bool: True if there are enough relevant chunks, False otherwise
    """
    # Count chunks with similarity score above threshold for any query type
    relevant_chunks = 0
    for score_dict in similarity_scores:
        # Check for both naming conventions to make the function more robust
        if (score_dict.get('sim_topic_chunk', 0) > threshold or
            score_dict.get('sim_subtopic_chunk', 0) > threshold or
            score_dict.get('sim_enhanced_chunk', 0) > threshold or
            score_dict.get('topic_similarity', 0) > threshold or
            score_dict.get('subtopic_similarity', 0) > threshold or
            score_dict.get('enhanced_similarity', 0) > threshold):
            relevant_chunks += 1
    
    logger.info(f"Found {relevant_chunks} relevant chunks (threshold: {threshold}, minimum required: {min_chunks})")
    return relevant_chunks >= min_chunks


def adjust_parameters_for_retry(
    attempt_count: int,
    k_topic: int,
    k_subtopic: int,
    k_enhanced: int,
    n_final: int
) -> Tuple[float, float, float, int, int, int, int, str, float, str]:
    """
    Adjust parameters for retry based on attempt count.
    
    Args:
        attempt_count: Number of previous attempts
        k_topic: Current k_topic value
        k_subtopic: Current k_subtopic value
        k_enhanced: Current k_enhanced value
        n_final: Current n_final value
        
    Returns:
        Tuple containing:
        - alpha_e: Weight for enhanced query
        - alpha_s: Weight for subtopic query
        - alpha_t: Weight for topic query
        - new_k_topic: Adjusted k_topic value
        - new_k_subtopic: Adjusted k_subtopic value
        - new_k_enhanced: Adjusted k_enhanced value
        - new_n_final: Adjusted n_final value
        - model: Model to use for generation
        - temperature: Temperature to use for generation
        - system_prompt: System prompt to use for generation
    """
    if attempt_count == 1:
        # First retry
        reduction_factor = REDUCTION_FACTOR_FIRST_RETRY
        alpha_e = ALPHA_ENHANCED_FIRST_RETRY
        alpha_s = ALPHA_SUBTOPIC_FIRST_RETRY
        alpha_t = ALPHA_TOPIC_FIRST_RETRY
        model = GENERATION_MODEL
        temperature = 0.3  # Default temperature
        system_prompt = None  # Use default system prompt
        
        logger.info(f"First retry: adjusting weights to αₑ={alpha_e}, αₛ={alpha_s}, αₜ={alpha_t}")
        logger.info(f"Reducing chunk counts by {(1-reduction_factor)*100}%")
    else:
        # Second retry (reasoning model)
        reduction_factor = REDUCTION_FACTOR_SECOND_RETRY
        alpha_e = ALPHA_ENHANCED_SECOND_RETRY
        alpha_s = ALPHA_SUBTOPIC_SECOND_RETRY
        alpha_t = ALPHA_TOPIC_SECOND_RETRY
        model = REASONING_MODEL
        temperature = REASONING_TEMPERATURE
        system_prompt = REASONING_SYSTEM_PROMPT
        
        logger.info(f"Second retry: using reasoning model with αₑ={alpha_e}, αₛ={alpha_s}, αₜ={alpha_t}")
        logger.info(f"Reducing chunk counts by {(1-reduction_factor)*100}%")
    
    # Apply reduction factor to k values and n_final
    new_k_topic = max(1, int(k_topic * reduction_factor))
    new_k_subtopic = max(1, int(k_subtopic * reduction_factor))
    new_k_enhanced = max(1, int(k_enhanced * reduction_factor))
    new_n_final = max(1, int(n_final * reduction_factor))
    
    logger.info(f"Adjusted parameters: k_topic={new_k_topic}, k_subtopic={new_k_subtopic}, k_enhanced={new_k_enhanced}, n_final={new_n_final}")
    
    return (
        alpha_e, alpha_s, alpha_t,
        new_k_topic, new_k_subtopic, new_k_enhanced, new_n_final,
        model, temperature, system_prompt
    )


def select_system_prompt(system_prompt, previous_attempt_count, previous_answer=None):
    """
    Select the appropriate system prompt based on attempt count and previous answer.
    
    For subsequent attempts with a non-reasoning model, use the continuity prompt
    to respect the previous answer more and only change it if there's strong evidence.
    
    Args:
        system_prompt: The default system prompt
        previous_attempt_count: Number of previous attempts
        previous_answer: The previous answer text (if any)
        
    Returns:
        str: The selected system prompt
    """
    # For subsequent attempts with a non-reasoning model, use the continuity prompt
    if previous_attempt_count > 0 and previous_answer:
        logger.info(f"Using continuity prompt for attempt {previous_attempt_count + 1} to respect previous answer more")
        return CONTINUITY_SYSTEM_PROMPT
    else:
        return system_prompt


def run_answer_sufficiency_feedback_loop(
    answer_text: str,
    answer_quality_score: float,
    previous_answer: Optional[str] = None,
    previous_score: Optional[float] = None,
    previous_attempt_count: int = 0,
    summarized_chunks: List[Dict[str, Any]] = [],
    similarity_scores: List[Dict[str, Any]] = [],
    used_chunk_ids: Set[str] = set(),
    enhanced_query: str = "",
    citation_key_map: Dict[str, str] = {},
    threshold_high: float = THRESHOLD_HIGH,
    threshold_low: float = THRESHOLD_LOW,
    alpha_e: float = 0.5,
    alpha_s: float = 0.3,
    alpha_t: float = 0.2,
    k_topic: int = 50,
    k_subtopic: int = 50,
    k_enhanced: int = 50,
    n_final: int = 5,
    used_reasoning_model: bool = False,
    evaluation_explanation: str = ""
) -> Tuple[str, float, str, List[Dict[str, Any]]]:
    """
    Run the answer sufficiency feedback loop.
    
    Args:
        answer_text: The current answer text
        answer_quality_score: The quality score of the current answer
        previous_answer: The previous answer text (if any)
        previous_score: The quality score of the previous answer (if any)
        previous_attempt_count: Number of previous attempts
        summarized_chunks: List of dictionaries containing chunk information
        similarity_scores: List of dictionaries containing similarity scores
        used_chunk_ids: Set of chunk IDs that have already been used
        enhanced_query: The enhanced query to answer
        citation_key_map: Dictionary mapping chunk_id to citation key
        threshold_high: High threshold for answer quality
        threshold_low: Low threshold for answer quality
        alpha_e: Weight for enhanced query
        alpha_s: Weight for subtopic query
        alpha_t: Weight for topic query
        k_topic: Number of top chunks to select for topic query
        k_subtopic: Number of top chunks to select for subtopic query
        k_enhanced: Number of top chunks to select for enhanced query
        n_final: Number of top chunks to use for answer generation
        used_reasoning_model: Whether the reasoning model was used in previous attempts
        
    Returns:
        Tuple containing (final_answer, final_score, evaluation_explanation, top_chunks)
    """
    logger.info(f"Running answer sufficiency feedback loop with score: {answer_quality_score}, attempt: {previous_attempt_count}, used_reasoning_model: {used_reasoning_model}")
    logger.info(f"Generated answer: {answer_text}")
    
    # Get the top chunks that were used for this answer
    top_chunks = [chunk for chunk in summarized_chunks if chunk.get('chunk_id') in used_chunk_ids]
    top_chunks = top_chunks[:n_final] if top_chunks else summarized_chunks[:n_final]
    
    # Save detailed information about the current attempt
    save_attempt_details(
        attempt_number=previous_attempt_count,
        answer_text=answer_text,
        answer_quality_score=answer_quality_score,
        evaluation_explanation=evaluation_explanation,
        enhanced_query=enhanced_query,
        all_chunks=summarized_chunks,
        selected_chunks=top_chunks,
        used_chunk_ids=used_chunk_ids,
        output_dir=ANSWER_OUTPUT_DIR
    )
    
    # Maximum attempts reached - return what we have
    if previous_attempt_count >= MAX_RETRY_ATTEMPTS:
        logger.info(f"Maximum attempts reached ({previous_attempt_count}). Returning current answer with score {answer_quality_score}")
        return answer_text, answer_quality_score, f"Final answer after {previous_attempt_count} attempts. Score: {answer_quality_score}", top_chunks
    
    # INITIAL ANSWER GENERATION (attempt 0)
    if previous_attempt_count == 0:
        # If score is perfect (1.0) or > threshold_high, accept it
        if answer_quality_score == 1.0 or answer_quality_score > threshold_high:
            logger.info(f"Initial answer quality score {answer_quality_score} is perfect or > {threshold_high}, accepting answer")
            # Use the provided evaluation_explanation or a default message
            explanation = evaluation_explanation if evaluation_explanation else f"Answer accepted with score {answer_quality_score}"
            return answer_text, answer_quality_score, explanation, top_chunks
        
        # If score is medium (between threshold_low and threshold_high), use scenario 2 (first retry with adjusted weights)
        elif threshold_low < answer_quality_score < threshold_high:
            logger.info(f"Initial answer quality score {answer_quality_score} between {threshold_low}-{threshold_high}, using scenario 2 (adjusted weights)")
            
            # Get parameters for first retry (scenario 2)
            (new_alpha_e, new_alpha_s, new_alpha_t,
             new_k_topic, new_k_subtopic, new_k_enhanced, new_n_final,
             model, temperature, system_prompt) = adjust_parameters_for_retry(
                1,  # First retry
                k_topic, k_subtopic, k_enhanced, n_final
            )
            
            # Instead of filtering out chunks, we'll use all chunks but prioritize unused ones
            # This ensures we don't lose valuable context
            filtered_chunks = summarized_chunks.copy()
            
            # Mark previously used chunks with a lower weight to prioritize new information
            for chunk in filtered_chunks:
                if chunk.get('chunk_id') in used_chunk_ids:
                    # Reduce the weight but don't remove it completely
                    if 'final_weight_adjusted' in chunk:
                        chunk['final_weight_adjusted'] *= 0.8  # Reduce weight by 20%
            
            # Update used_chunk_ids - only track chunks that were actually used for answer generation
            # We'll track these after the answer generation
            
            # Select appropriate system prompt based on attempt count and previous answer
            current_system_prompt = select_system_prompt(system_prompt, previous_attempt_count, previous_answer)
            
            # Run answer generation with scenario 2 parameters
            new_answer_text, new_answer_quality_score, evaluation_explanation, new_top_chunks = run_answer_generation(
                summarized_chunks=filtered_chunks,
                enhanced_query=enhanced_query,
                citation_key_map=citation_key_map,
                n_answer=new_n_final,
                previous_answer=answer_text,
                previous_score=answer_quality_score,
                model=model,
                temperature=temperature,
                system_prompt=current_system_prompt
            )
            
            # Now update used_chunk_ids with only the chunks that were actually used
            for chunk in new_top_chunks:
                chunk_id = chunk.get('chunk_id', '')
                if chunk_id:
                    used_chunk_ids.add(chunk_id)
            
            # Save detailed information about this attempt
            save_attempt_details(
                attempt_number=previous_attempt_count + 1,
                answer_text=new_answer_text,
                answer_quality_score=new_answer_quality_score,
                evaluation_explanation=evaluation_explanation,
                enhanced_query=enhanced_query,
                all_chunks=filtered_chunks,
                selected_chunks=new_top_chunks,
                used_chunk_ids=used_chunk_ids,
                output_dir=ANSWER_OUTPUT_DIR
            )
            
            # Continue with feedback loop
            return run_answer_sufficiency_feedback_loop(
                answer_text=new_answer_text,
                answer_quality_score=new_answer_quality_score,
                previous_answer=answer_text,
                previous_score=answer_quality_score,
                previous_attempt_count=previous_attempt_count + 1,
                summarized_chunks=filtered_chunks,
                similarity_scores=similarity_scores,
                used_chunk_ids=used_chunk_ids,
                enhanced_query=enhanced_query,
                citation_key_map=citation_key_map,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                alpha_e=new_alpha_e,
                alpha_s=new_alpha_s,
                alpha_t=new_alpha_t,
                k_topic=new_k_topic,
                k_subtopic=new_k_subtopic,
                k_enhanced=new_k_enhanced,
                n_final=new_n_final,
                used_reasoning_model=False,
                evaluation_explanation=evaluation_explanation
            )
        
        # If score is low (<= 0.7), go straight to reasoning model
        elif answer_quality_score <= threshold_low:  # <= 0.7
            logger.info(f"Initial answer quality score {answer_quality_score} <= {threshold_low}, going straight to reasoning model")
            
            # Check if we have enough relevant chunks
            if not check_chunk_pool_sufficiency(similarity_scores):
                logger.warning("Insufficient relevant context in chunk pool")
                return "Insufficient relevant context", 0.0, "Not enough relevant chunks found", []
            
            # Get parameters for reasoning model
            (new_alpha_e, new_alpha_s, new_alpha_t,
             new_k_topic, new_k_subtopic, new_k_enhanced, new_n_final,
             model, temperature, system_prompt) = adjust_parameters_for_retry(
                MAX_RETRY_ATTEMPTS,  # Force using reasoning model parameters
                k_topic, k_subtopic, k_enhanced, n_final
            )
            
            # Instead of filtering out chunks, we'll use all chunks but prioritize unused ones
            # This ensures we don't lose valuable context
            filtered_chunks = summarized_chunks.copy()
            
            # Mark previously used chunks with a lower weight to prioritize new information
            for chunk in filtered_chunks:
                if chunk.get('chunk_id') in used_chunk_ids:
                    # Reduce the weight but don't remove it completely
                    if 'final_weight_adjusted' in chunk:
                        chunk['final_weight_adjusted'] *= 0.8  # Reduce weight by 20%
            
            # Update used_chunk_ids - only track chunks that were actually used for answer generation
            # We'll track these after the answer generation
            
            # Run answer generation with reasoning model
            new_answer_text, new_answer_quality_score, evaluation_explanation, new_top_chunks = run_answer_generation(
                summarized_chunks=filtered_chunks,
                enhanced_query=enhanced_query,
                citation_key_map=citation_key_map,
                n_answer=new_n_final,
                previous_answer=answer_text,
                previous_score=answer_quality_score,
                model=model,
                temperature=temperature,
                system_prompt=system_prompt
            )
            
            # Now update used_chunk_ids with only the chunks that were actually used
            for chunk in new_top_chunks:
                chunk_id = chunk.get('chunk_id', '')
                if chunk_id:
                    used_chunk_ids.add(chunk_id)
            
            # Save detailed information about this attempt
            save_attempt_details(
                attempt_number=previous_attempt_count + 1,
                answer_text=new_answer_text,
                answer_quality_score=new_answer_quality_score,
                evaluation_explanation=evaluation_explanation,
                enhanced_query=enhanced_query,
                all_chunks=filtered_chunks,
                selected_chunks=new_top_chunks,
                used_chunk_ids=used_chunk_ids,
                output_dir=ANSWER_OUTPUT_DIR
            )
            
            # Continue with feedback loop
            return run_answer_sufficiency_feedback_loop(
                answer_text=new_answer_text,
                answer_quality_score=new_answer_quality_score,
                previous_answer=answer_text,
                previous_score=answer_quality_score,
                previous_attempt_count=previous_attempt_count + 1,
                summarized_chunks=filtered_chunks,
                similarity_scores=similarity_scores,
                used_chunk_ids=used_chunk_ids,
                enhanced_query=enhanced_query,
                citation_key_map=citation_key_map,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                alpha_e=new_alpha_e,
                alpha_s=new_alpha_s,
                alpha_t=new_alpha_t,
                k_topic=new_k_topic,
                k_subtopic=new_k_subtopic,
                k_enhanced=new_k_enhanced,
                n_final=new_n_final,
                used_reasoning_model=True,
                evaluation_explanation=evaluation_explanation
            )
        
        # Default case for initial attempt - should never reach here with proper thresholds
        # but added as a safeguard
        else:
            logger.info(f"Initial answer with score {answer_quality_score} didn't match any condition, using scenario 2 (adjusted weights)")
            
            # Get parameters for first retry (scenario 2)
            (new_alpha_e, new_alpha_s, new_alpha_t,
             new_k_topic, new_k_subtopic, new_k_enhanced, new_n_final,
             model, temperature, system_prompt) = adjust_parameters_for_retry(
                1,  # First retry
                k_topic, k_subtopic, k_enhanced, n_final
            )
            
            # Instead of filtering out chunks, we'll use all chunks but prioritize unused ones
            # This ensures we don't lose valuable context
            filtered_chunks = summarized_chunks.copy()
            
            # Mark previously used chunks with a lower weight to prioritize new information
            for chunk in filtered_chunks:
                if chunk.get('chunk_id') in used_chunk_ids:
                    # Reduce the weight but don't remove it completely
                    if 'final_weight_adjusted' in chunk:
                        chunk['final_weight_adjusted'] *= 0.8  # Reduce weight by 20%
            
            # Select appropriate system prompt based on attempt count and previous answer
            current_system_prompt = select_system_prompt(system_prompt, previous_attempt_count, previous_answer)
            
            # Run answer generation with scenario 2 parameters
            new_answer_text, new_answer_quality_score, evaluation_explanation, new_top_chunks = run_answer_generation(
                summarized_chunks=filtered_chunks,
                enhanced_query=enhanced_query,
                citation_key_map=citation_key_map,
                n_answer=new_n_final,
                previous_answer=answer_text,
                previous_score=answer_quality_score,
                model=model,
                temperature=temperature,
                system_prompt=current_system_prompt
            )
            
            # Now update used_chunk_ids with only the chunks that were actually used
            for chunk in new_top_chunks:
                chunk_id = chunk.get('chunk_id', '')
                if chunk_id:
                    used_chunk_ids.add(chunk_id)
            
            # Save detailed information about this attempt
            save_attempt_details(
                attempt_number=previous_attempt_count + 1,
                answer_text=new_answer_text,
                answer_quality_score=new_answer_quality_score,
                evaluation_explanation=evaluation_explanation,
                enhanced_query=enhanced_query,
                all_chunks=filtered_chunks,
                selected_chunks=new_top_chunks,
                used_chunk_ids=used_chunk_ids,
                output_dir=ANSWER_OUTPUT_DIR
            )
            
            # Continue with feedback loop
            return run_answer_sufficiency_feedback_loop(
                answer_text=new_answer_text,
                answer_quality_score=new_answer_quality_score,
                previous_answer=answer_text,
                previous_score=answer_quality_score,
                previous_attempt_count=previous_attempt_count + 1,
                summarized_chunks=filtered_chunks,
                similarity_scores=similarity_scores,
                used_chunk_ids=used_chunk_ids,
                enhanced_query=enhanced_query,
                citation_key_map=citation_key_map,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                alpha_e=new_alpha_e,
                alpha_s=new_alpha_s,
                alpha_t=new_alpha_t,
                k_topic=new_k_topic,
                k_subtopic=new_k_subtopic,
                k_enhanced=new_k_enhanced,
                n_final=new_n_final,
                used_reasoning_model=False,
                evaluation_explanation=evaluation_explanation
            )
    
    # FIRST RETRY (attempt 1)
    elif previous_attempt_count == 1:
        # If scenario 2 was used (not reasoning model)
        if not used_reasoning_model:
            # If score is high enough (>= threshold_high), accept it
            if answer_quality_score >= threshold_high:
                logger.info(f"After scenario 2, answer quality score {answer_quality_score} >= {threshold_high}, accepting answer")
                # Use the provided evaluation_explanation or a default message
                explanation = evaluation_explanation if evaluation_explanation else f"Answer accepted with score {answer_quality_score}"
                return answer_text, answer_quality_score, explanation, top_chunks
            
            # If score is not high enough (< threshold_high), run reasoning model
            else:  # < threshold_high
                logger.info(f"After scenario 2, answer quality score {answer_quality_score} < {threshold_high}, running reasoning model")
                
                # Check if we have enough relevant chunks
                if not check_chunk_pool_sufficiency(similarity_scores):
                    logger.warning("Insufficient relevant context in chunk pool")
                    return "Insufficient relevant context", 0.0, "Not enough relevant chunks found", []
                
                # Get parameters for reasoning model
                (new_alpha_e, new_alpha_s, new_alpha_t,
                 new_k_topic, new_k_subtopic, new_k_enhanced, new_n_final,
                 model, temperature, system_prompt) = adjust_parameters_for_retry(
                    MAX_RETRY_ATTEMPTS,  # Force using reasoning model parameters
                    k_topic, k_subtopic, k_enhanced, n_final
                )
                
                # Filter out already used chunks
                filtered_chunks = [chunk for chunk in summarized_chunks if chunk.get('chunk_id') not in used_chunk_ids]
                
                # Update used_chunk_ids - only track chunks that were actually used for answer generation
                # We'll track these after the answer generation
                
                # Run answer generation with reasoning model
                new_answer_text, new_answer_quality_score, evaluation_explanation, new_top_chunks = run_answer_generation(
                    summarized_chunks=filtered_chunks,
                    enhanced_query=enhanced_query,
                    citation_key_map=citation_key_map,
                    n_answer=new_n_final,
                    previous_answer=answer_text,
                    previous_score=answer_quality_score,
                    model=model,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                
                # Now update used_chunk_ids with only the chunks that were actually used
                for chunk in new_top_chunks:
                    chunk_id = chunk.get('chunk_id', '')
                    if chunk_id:
                        used_chunk_ids.add(chunk_id)
                
                # Continue with feedback loop
                return run_answer_sufficiency_feedback_loop(
                    answer_text=new_answer_text,
                    answer_quality_score=new_answer_quality_score,
                    previous_answer=answer_text,
                    previous_score=answer_quality_score,
                    previous_attempt_count=previous_attempt_count + 1,
                    summarized_chunks=filtered_chunks,
                    similarity_scores=similarity_scores,
                    used_chunk_ids=used_chunk_ids,
                    enhanced_query=enhanced_query,
                    citation_key_map=citation_key_map,
                    threshold_high=threshold_high,
                    threshold_low=threshold_low,
                    alpha_e=new_alpha_e,
                    alpha_s=new_alpha_s,
                    alpha_t=new_alpha_t,
                    k_topic=new_k_topic,
                    k_subtopic=new_k_subtopic,
                    k_enhanced=new_k_enhanced,
                    n_final=new_n_final,
                    used_reasoning_model=True,
                    evaluation_explanation=evaluation_explanation
                )
        
        # If reasoning model was used
        else:  # used_reasoning_model == True
            # If score is perfect (1.0) or > threshold_high, accept it
            if answer_quality_score == 1.0 or answer_quality_score > threshold_high:
                logger.info(f"After reasoning model, answer quality score {answer_quality_score} is perfect or > {threshold_high}, accepting answer")
                # Use the provided evaluation_explanation or a default message
                explanation = evaluation_explanation if evaluation_explanation else f"Answer accepted with score {answer_quality_score}"
                return answer_text, answer_quality_score, explanation, top_chunks
            
            # If score is medium (between threshold_low and threshold_high), run scenario 2
            elif threshold_low < answer_quality_score < threshold_high:
                logger.info(f"After reasoning model, answer quality score {answer_quality_score} between {threshold_low}-{threshold_high}, running scenario 2")
                
                # Get parameters for scenario 2
                (new_alpha_e, new_alpha_s, new_alpha_t,
                 new_k_topic, new_k_subtopic, new_k_enhanced, new_n_final,
                 model, temperature, system_prompt) = adjust_parameters_for_retry(
                    1,  # First retry parameters (scenario 2)
                    k_topic, k_subtopic, k_enhanced, n_final
                )
                
                # Filter out already used chunks
                filtered_chunks = [chunk for chunk in summarized_chunks if chunk.get('chunk_id') not in used_chunk_ids]
                
                # Update used_chunk_ids - only track chunks that were actually used for answer generation
                # We'll track these after the answer generation
                
                # Run answer generation with scenario 2 parameters
                new_answer_text, new_answer_quality_score, evaluation_explanation, new_top_chunks = run_answer_generation(
                    summarized_chunks=filtered_chunks,
                    enhanced_query=enhanced_query,
                    citation_key_map=citation_key_map,
                    n_answer=new_n_final,
                    previous_answer=answer_text,
                    previous_score=answer_quality_score,
                    model=model,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                
                # Now update used_chunk_ids with only the chunks that were actually used
                for chunk in new_top_chunks:
                    chunk_id = chunk.get('chunk_id', '')
                    if chunk_id:
                        used_chunk_ids.add(chunk_id)
                
                # Continue with feedback loop
                return run_answer_sufficiency_feedback_loop(
                    answer_text=new_answer_text,
                    answer_quality_score=new_answer_quality_score,
                    previous_answer=answer_text,
                    previous_score=answer_quality_score,
                    previous_attempt_count=previous_attempt_count + 1,
                    summarized_chunks=filtered_chunks,
                    similarity_scores=similarity_scores,
                    used_chunk_ids=used_chunk_ids,
                    enhanced_query=enhanced_query,
                    citation_key_map=citation_key_map,
                    threshold_high=threshold_high,
                    threshold_low=threshold_low,
                    alpha_e=new_alpha_e,
                    alpha_s=new_alpha_s,
                    alpha_t=new_alpha_t,
                    k_topic=new_k_topic,
                    k_subtopic=new_k_subtopic,
                    k_enhanced=new_k_enhanced,
                    n_final=new_n_final,
                    used_reasoning_model=False,
                    evaluation_explanation=evaluation_explanation
                )
            
            # If score is low (<= threshold_low), run reasoning model again
            elif answer_quality_score <= threshold_low:
                logger.info(f"After reasoning model, answer quality score {answer_quality_score} <= {threshold_low}, running reasoning model again")
                
                # Check if we have enough relevant chunks
                if not check_chunk_pool_sufficiency(similarity_scores):
                    logger.warning("Insufficient relevant context in chunk pool")
                    return "Insufficient relevant context", 0.0, "Not enough relevant chunks found", []
                
                # Get parameters for reasoning model
                (new_alpha_e, new_alpha_s, new_alpha_t,
                 new_k_topic, new_k_subtopic, new_k_enhanced, new_n_final,
                 model, temperature, system_prompt) = adjust_parameters_for_retry(
                    MAX_RETRY_ATTEMPTS,  # Force using reasoning model parameters
                    k_topic, k_subtopic, k_enhanced, n_final
                )
                
                # Filter out already used chunks
                filtered_chunks = [chunk for chunk in summarized_chunks if chunk.get('chunk_id') not in used_chunk_ids]
                
                # Run answer generation with reasoning model
                new_answer_text, new_answer_quality_score, evaluation_explanation, new_top_chunks = run_answer_generation(
                    summarized_chunks=filtered_chunks,
                    enhanced_query=enhanced_query,
                    citation_key_map=citation_key_map,
                    n_answer=new_n_final,
                    previous_answer=answer_text,
                    previous_score=answer_quality_score,
                    model=model,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
                
                # Now update used_chunk_ids with only the chunks that were actually used
                for chunk in new_top_chunks:
                    chunk_id = chunk.get('chunk_id', '')
                    if chunk_id:
                        used_chunk_ids.add(chunk_id)
                
                # Continue with feedback loop
                return run_answer_sufficiency_feedback_loop(
                    answer_text=new_answer_text,
                    answer_quality_score=new_answer_quality_score,
                    previous_answer=answer_text,
                    previous_score=answer_quality_score,
                    previous_attempt_count=previous_attempt_count + 1,
                    summarized_chunks=filtered_chunks,
                    similarity_scores=similarity_scores,
                    used_chunk_ids=used_chunk_ids,
                    enhanced_query=enhanced_query,
                    citation_key_map=citation_key_map,
                    threshold_high=threshold_high,
                    threshold_low=threshold_low,
                    alpha_e=new_alpha_e,
                    alpha_s=new_alpha_s,
                    alpha_t=new_alpha_t,
                    k_topic=new_k_topic,
                    k_subtopic=new_k_subtopic,
                    k_enhanced=new_k_enhanced,
                    n_final=new_n_final,
                    used_reasoning_model=True,
                    evaluation_explanation=evaluation_explanation
                )
            
            # Default case - if we somehow reach here, return the current answer
            else:
                logger.info(f"Unhandled case in feedback loop with score {answer_quality_score}, returning current answer")
                return answer_text, answer_quality_score, evaluation_explanation, top_chunks
    
    # SECOND RETRY (attempt 2) - Always return this answer with explanation
    else:  # previous_attempt_count == 2
        logger.info(f"Second retry completed. Final answer quality score: {answer_quality_score}")
        explanation = f"Final answer after {previous_attempt_count} attempts. Score: {answer_quality_score}. "
        explanation += f"This answer was generated using {'the reasoning model' if used_reasoning_model else 'adjusted weights'}."
        explanation += f"\n\nEvaluation explanation: {evaluation_explanation if 'evaluation_explanation' in locals() else ''}"  
        return answer_text, answer_quality_score, explanation, top_chunks
