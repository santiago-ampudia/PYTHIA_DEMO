"""
module_paper_search/main.py

This is the main entry point for the paper search pipeline. It orchestrates the extended pipeline for finding the most relevant paper chunks given a user query.

Each step in the pipeline is implemented as a separate submodule for modularity and maintainability.

Pipeline Steps (each in its own submodule):
    0. Metadata Harvesting
    1. Embed Metadata and Content
    2. Query Understanding & Decomposition
    3. arXiv Category Prediction
    4. Paper Preselection by Category
    5. FAISS Semantic Search
    6. Chunk Weight Determination
    7. Candidate Grouping
    8a. Grounded Section Header Prediction
    8b. Section Matching for Rank 3k–10k Papers
    9. On-Demand PDF Download
    10. Chunking + LLM Keyword Filtering
    11. Embedding and Scoring
    12. Top-K Selection + Output

To add a new step, create a new Python file in this folder and import it here.

This script should be run as the main entry point for the paper search pipeline.
"""

import argparse
import logging
import sys
import os

# Add the project root to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the metadata harvesting submodule
from module_paper_search.submodule_metadata_harvesting.arxiv_metadata_harvesting_scheduler import run_harvesting_scheduler
# Import the metadata embedding submodule
from module_paper_search.submodule_metadata_embedding import metadata_embedding
# Import the content embedding submodule
from module_paper_search.submodule_metadata_embedding import content_embedding
# Import the query understanding submodules
from module_paper_search.submodule_query_understanding import run_query_understanding
from module_paper_search.submodule_query_understanding.query_understanding_recommendation import run_query_understanding_recommendation
# Import the arXiv category prediction submodules
from module_paper_search.submodule_arxiv_category_prediction.arxiv_category_prediction import run_arxiv_category_prediction
from module_paper_search.submodule_arxiv_category_prediction.arxiv_category_prediction_recommendation import run_arxiv_category_prediction_recommendation
# Import the paper preselection submodules
from module_paper_search.submodule_paper_preselection.category_paper_preselection import run_category_paper_preselection
from module_paper_search.submodule_paper_preselection.category_paper_preselection_recommendation import run_category_paper_preselection_recommendation
# Import the chunk similarity selection submodules
from module_paper_search.submodule_chunk_similarity_selection.chunk_similarity_selection import run_chunk_similarity_selection
from module_paper_search.submodule_chunk_similarity_selection.chunk_similarity_selection_recommendation import run_chunk_similarity_selection_recommendation
from module_paper_search.submodule_chunk_similarity_selection.chunk_similarity_selection_kgb import run_chunk_similarity_selection_kgb
# Import the chunk weight determination submodule
from module_paper_search.submodule_chunk_weight_determination import run_chunk_weight_determination
from module_paper_search.submodule_chunk_weight_determination.chunk_weight_determination_kgb import run_chunk_weight_determination_kgb
from module_paper_search.submodule_chunk_weight_determination.chunk_weight_determination_kgb_llm import determine_chunk_weights_kgb_llm
# Import the chunk LLM weight determination submodule
from module_paper_search.submodule_chunk_weight_determination.chunk_llm_weight_determination import run_chunk_llm_weight_determination
# Import the answer generation and evaluation submodule
from module_paper_search.submodule_answer_generation.answer_generation import run_answer_generation, save_answer
from module_paper_search.submodule_answer_generation.recommendation_generation import run_recommendation_generation
from module_paper_search.submodule_answer_generation.answer_generation_kgb import run_answer_generation_kgb
from module_paper_search.submodule_answer_sufficiency.answer_sufficiency_kgb import run_answer_sufficiency_kgb
# Import the tweet recommendation submodule
from module_paper_search.submodule_tweet_recommendation import run_tweet_recommendation
from module_paper_search.output_directories import ANSWER_OUTPUT_DIR, RECOMMENDATION_OUTPUT_DIR
from module_paper_search.submodule_answer_generation.recommendation_generation_parameters import (
    N_RECOMMENDATIONS,
    RECOMMENDATION_QUALITY_THRESHOLD,
    OUTPUT_DIR
)
# Import the answer sufficiency feedback loop
from module_paper_search.submodule_answer_sufficiency.answer_sufficiency import run_answer_sufficiency_feedback_loop
# Import the query input module
from module_query_obtention.query_input import get_user_query
# Import the search mode module
from module_query_obtention.search_mode import get_search_mode
# Import the midpoint finding submodule
from module_paper_search.submodule_midpoint_finding_kgb.midpoint_finding_kgb import run_midpoint_finding_kgb
# Import main parameters
from module_paper_search.main_parameters import MAX_LOOPS, DEFAULT_QUERIES_ORDER

# Configure logging
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('paper_search_pipeline')


def main(args):
    """
    Main function to run the paper search pipeline.
    Each pipeline step will be called in sequence here as submodules are implemented.
    
    Args:
        args: Command line arguments
    """
    logger.info("Paper search pipeline started.")
    
    # Determine the search mode (answer or recommendation)
    search_mode = get_search_mode()
    logger.info(f"Search mode: {search_mode}")

    # Get the ser query
    user_query = get_user_query()
    logger.info(f"User query: {user_query}")

    # Use custom DB path if provided
    db_path = args.db_path if args.db_path else None

    # Step 0: Metadata Harvesting
    # The scheduler determines if harvesting is needed based on last update time
    logger.info("Checking if metadata harvesting is needed...")
    run_harvesting_scheduler(
        force_update=args.force_harvesting, 
        run_continuously=False,
        arxiv_id=args.arxiv_id
    )
    
    # Step 1: Content and Metadata Embedding (always run, but only processes new papers)
    logger.info("Running content and metadata embedding...")
    embedding_success = content_embedding.run_content_embedding(
        force_update=args.force_harvesting,
        specific_arxiv_id=args.arxiv_id
    )
    if embedding_success:
        logger.info("Content and metadata embedding completed successfully.")
    else:
        logger.warning("Content and metadata embedding encountered issues. Check logs for details.")

    # Mode distinctive pipeline
    if search_mode == "recommendation":

        # Step 2: Query Understanding & Decomposition
        logger.info("Running query understanding and decomposition for recommendation mode...")
        architecture_query, technical_implementation_query, algorithmic_approach_query, domain_specific_query, integration_pipeline_query = run_query_understanding_recommendation(user_query)
        logger.info(f"Architecture query: {architecture_query}")
        logger.info(f"Technical Implementation query: {technical_implementation_query}")
        logger.info(f"Algorithmic Approach query: {algorithmic_approach_query}")
        logger.info(f"Domain-Specific query: {domain_specific_query}")
        logger.info(f"Integration & Pipeline query: {integration_pipeline_query}")

        # Step 3: arXiv Category Prediction
        logger.info("Running arXiv category prediction for recommendation mode...")
        predicted_categories = run_arxiv_category_prediction_recommendation(
            architecture_query,
            technical_implementation_query,
            algorithmic_approach_query,
            domain_specific_query,
            integration_pipeline_query
        )
        logger.info(f"Predicted categories for recommendation mode: {predicted_categories}")
        
        # Step 4: Paper Preselection by Category
        logger.info("Running paper preselection by category for recommendation mode...")
        preselected_papers = run_category_paper_preselection_recommendation(predicted_categories)
        logger.info(f"Paper preselection by category for recommendation mode completed. Found {len(preselected_papers)} papers.")
        
        # Step 5: Chunk Similarity Selection
        logger.info("Running chunk similarity selection for recommendation mode...")
        top_chunks_architecture, top_chunks_technical, top_chunks_algorithmic, top_chunks_domain, top_chunks_integration, similarity_scores = run_chunk_similarity_selection_recommendation(
            architecture_query=architecture_query,
            technical_implementation_query=technical_implementation_query,
            algorithmic_approach_query=algorithmic_approach_query,
            domain_specific_query=domain_specific_query,
            integration_pipeline_query=integration_pipeline_query,
            preselected_papers=preselected_papers
        )
        logger.info(f"Chunk similarity selection for recommendation mode completed.")
        logger.info(f"Found {len(top_chunks_architecture)} architecture chunks, {len(top_chunks_technical)} technical chunks, ")
        logger.info(f"      {len(top_chunks_algorithmic)} algorithmic chunks, {len(top_chunks_domain)} domain chunks, ")
        logger.info(f"      {len(top_chunks_integration)} integration chunks.")

        # Step 6: Tweet generation
        logger.info("Running tweet recommendation generation...")
        
        # Check if this is for recent work recommendations
        is_recent_work = os.environ.get("IS_RECENT_WORK", "false").lower() == "true"
        file_suffix = "(1)" if is_recent_work else ""
        
        tweet_results = run_tweet_recommendation(
            architecture_query=architecture_query,
            technical_implementation_query=technical_implementation_query,
            algorithmic_approach_query=algorithmic_approach_query,
            domain_specific_query=domain_specific_query,
            integration_pipeline_query=integration_pipeline_query,
            top_chunks_architecture=top_chunks_architecture,
            top_chunks_technical=top_chunks_technical,
            top_chunks_algorithmic=top_chunks_algorithmic,
            top_chunks_domain=top_chunks_domain,
            top_chunks_integration=top_chunks_integration,
            file_suffix=file_suffix
        )
        logger.info("Tweet recommendation generation completed.")
        logger.info(f"Generated {len(tweet_results['architecture'])} architecture tweets, {len(tweet_results['technical'])} technical tweets,")
        logger.info(f"         {len(tweet_results['algorithmic'])} algorithmic tweets, {len(tweet_results['domain'])} domain tweets,")
        logger.info(f"         {len(tweet_results['integration'])} integration tweets.")

    elif search_mode == "answer":
        
        # Step 2: Query Understanding & Decomposition
        logger.info("Running query understanding and decomposition...")
        user_query = get_user_query()
        topic_query, subtopic_query, enhanced_query = run_query_understanding(user_query)
        logger.info(f"Topic query: {topic_query}")
        logger.info(f"Subtopic query: {subtopic_query}")
        logger.info(f"Enhanced query: {enhanced_query}")
        
        # Debugging
        """
        topic_query = "Event reconstruction for studying Higgs Self-Coupling at the XCC, an X-ray FEL-based gamma gamma Compton Collider Higgs Factory."
        subtopic_query = "Jet clustering algorithms for event reconstruction."
        enhanced_query = "Selection of jet clustering algorithms for event reconstruction in the study of Higgs Self-Coupling at the XCC, an X-ray FEL-based gamma gamma Compton Collider Higgs Factory, focusing on the reconstruction level analysis."
        """

        # Step 3: arXiv Category Prediction
        logger.info("Running arXiv category prediction...")
        predicted_categories = run_arxiv_category_prediction(topic_query, subtopic_query, enhanced_query)
        logger.info(f"Predicted categories: {predicted_categories}")

        # Debugging
        #predicted_categories = ['hep-ph', 'hep-ex', 'physics.acc-ph', 'cs.LG', 'nucl-ex', 'nucl-th', 'physics.ins-det']

        # Step 4: Paper Preselection by Category
        logger.info("Running paper preselection by category...")
        preselected_papers = run_category_paper_preselection(predicted_categories)
        logger.info(f"Paper preselection by category completed. Found {len(preselected_papers)} papers.")

        #################################
        ####    Begins Ramanujan -- A Knowledge Graph Based Agent    ####
        max_loops = MAX_LOOPS  # Using the parameter from main_parameters
        final_answer = None
        is_sufficient = False
        current_grade = 0.0
        
        for nth_loop in range(max_loops):
            logger.info(f"Running KGB agent loop {nth_loop + 1}/{max_loops}")
            
            # Create a list of queries in the specified order: subtopic, enhanced, topic
            if nth_loop == 0:
                # First loop: use all queries
                queries_list = [subtopic_query, enhanced_query, topic_query]
                logger.info(f"Queries list created with order: [subtopic, enhanced, topic]")
            
            # Log all queries in the list for this iteration
            logger.info(f"Queries for iteration {nth_loop + 1}/{max_loops}:")
            for i, query in enumerate(queries_list):
                logger.info(f"  Query {i}: {query}")
            
            
            # Step 5: Chunk Similarity Selection KGB
            logger.info("Running KGB chunk similarity selection...")
            top_chunks_per_query, similarity_scores = run_chunk_similarity_selection_kgb(
                queries_list=queries_list,
                preselected_papers=preselected_papers,
                search_mode=search_mode
            )
            
            # Extract individual top chunks lists for backward compatibility
            top_chunks_subtopic = top_chunks_per_query[0]
            top_chunks_enhanced = top_chunks_per_query[1] if len(top_chunks_per_query) > 1 else []
            top_chunks_topic = top_chunks_per_query[-1] if len(top_chunks_per_query) > 0 else []
            top_chunks_test = top_chunks_per_query[2] if len(top_chunks_per_query) > 2 else []
            
            logger.info(f"KGB chunk similarity selection completed. Found {len(top_chunks_subtopic)} subtopic chunks, "
                        f"{len(top_chunks_enhanced)} enhanced chunks, and {len(top_chunks_topic)} topic chunks.")
            if top_chunks_test:
                logger.info(f"Also found {len(top_chunks_test)} test chunks.")


            # Step 6: Chunk Weighting
            logger.info("Running KGB chunk weight determination...")
            weighted_chunks_by_query, all_weighted_chunks = run_chunk_weight_determination_kgb(
                queries_list=queries_list
            )
            
            logger.info(f"KGB chunk weight determination completed. Processed {len(all_weighted_chunks)} chunks in total.")
            for i, query_chunks in enumerate(weighted_chunks_by_query):
                query_type = f"query_{i}"
                logger.info(f"  {query_type} query: {len(query_chunks)} weighted chunks")
            
            # Step 7: LLM-based Chunk Reweighting
            logger.info("Running KGB LLM-based chunk weight determination...")
            reweighted_chunks_by_query, all_reweighted_chunks = determine_chunk_weights_kgb_llm(
                enhanced_query=enhanced_query
            )
            
            logger.info(f"KGB LLM-based chunk weight determination completed. Processed {len(all_reweighted_chunks)} chunks in total.")
            for i, query_chunks in enumerate(reweighted_chunks_by_query):
                query_type = f"query_{i}"
                logger.info(f"  {query_type} query: {len(query_chunks)} reweighted chunks")
            
            
            
            # Step 8: Chunk Selection and Answer Generation
            logger.info("Running KGB-based answer generation...")
            answer = run_answer_generation_kgb(enhanced_query=enhanced_query)
            logger.info("KGB-based answer generation completed.")
            logger.info(f"Generated answer with {len(answer)} characters")
            
            # Debugging
            #answer = "Accurate jet reconstruction is a leading systematic in any determination of the Higgs self-coupling because the di-jet masses entering the fit are extremely sensitive to the mis-clustering and mis-pairing of final-state particles [2410.15323]. In the most recent full-simulation ZHH studies, this effect alone degrades the statistical precision on the self-coupling by almost a factor two, demonstrating that the choice of jet algorithm is a first-order design decision for any Higgs factory, including the XCC.\n1. Baseline algorithm \n a The Durham sequential-recombination scheme remains the standard benchmark. It is IRC-safe, fast, and already integrated in the existing ILC/ILD reconstruction chain that is being adapted to XCC studies. However, at 500 GeV ZHH it produces a noticeably broader di-jet mass distribution than the atrutha clustering obtained with perfect information, directly linking it to the observed loss in sensitivity [2410.15323].\n2. ML-enhanced alternative \n a A graph-neural-network plus spectral-clustering hybrid (GNNSC) has been demonstrated on full ILD simulation to recover a di-jet mass resolution comparable to the cheated atrue-jeta reference while retaining full particle-flow input information [2410.15323]. \n a The model generalises to processes beyond ZHH, e.g. ZZH, suggesting that it can be trained once and reused for the gamma-gamma environment of the XCC with only modest retuning [2410.15323]. \n a The same graph architecture can be extended to incorporate jet-flavour tagging, whose performance has already been benchmarked for several detector layouts and energies, confirming the suitability of GNNs for future collider detectors [2501.16584].\n3. Relevance for the XCC \n a The XCC will produce O(10^6) Higgs bosons in its first decade via resonant gamma-gamma collisions at 125 GeV; branching-ratio measurements therefore rely on jet-based final states to reach the quoted sub-percent coupling precisions [2306.10057]. \n a Although the self-coupling itself must be accessed through double-Higgs production channels (as already studied at e+ea colliders) [2411.01507][2410.15323], the same jet-clustering issues identified in ZHH analyses will control the attainable precision at the XCC. Transferring the GNNSC approach therefore offers a direct path to minimise this limiting systematic.\n4. Recommended strategy for reconstruction-level analyses at the XCC \n a) Use Durham as a baseline and systematic cross-check. \n b) Deploy GNNSC (or an equivalent fully differentiable, IRC-aware GNN) as the primary clustering algorithm; train on simulated XCC HH and single-Higgs samples with truth-level labels generated as in the ILD study. \n c) Monitor performance using the AaD purity/efficiency categorisation defined in Ref. [2410.15323] to quantify residual mis-clustering. \n d) Integrate the GNN outputs with flavour-tagging networks to exploit correlated information and reduce combinatorial ambiguities, following the methodology of Ref. [2501.16584]. \nAdopting this workflow transfers the demonstrated factor-of-two gain in self-coupling sensitivity from the ILD ZHH analysis to the XCC program while retaining a well-understood Durham reference for robustness and systematic evaluation."
            
            # Step 9: Answer Evaluation and KGB Agent Feedback loop
            logger.info("Running KGB-based answer sufficiency evaluation...")
            final_answer, is_sufficient, current_grade, queries_list = run_answer_sufficiency_kgb(
                answer=answer,
                enhanced_query=enhanced_query,
                nth_loop=nth_loop,
                max_loops=max_loops,
                queries_list=queries_list
                
            )
            
            # Debugging
            logger.info("Debugging: KGB-based answer sufficiency evaluation...")
            final_answer = answer
            if nth_loop != max_loops - 1:
                current_grade = 0.3
                is_sufficient = False
            else:
                current_grade = 1.0
                is_sufficient = True
            
            if nth_loop == 0:
                queries_list = [subtopic_query, topic_query]
            
            
            logger.info(f"Answer evaluation completed with grade: {current_grade}, sufficient: {is_sufficient}")
            
            # If we have a sufficient answer, we can stop the loop
            if is_sufficient:
                logger.info(f"Found sufficient answer with grade {current_grade}")
                break
            
            elif not is_sufficient and nth_loop != max_loops - 1:
                # Step 10: Midpoint Finding - Enhance the knowledge graph with intermediate nodes
                logger.info("Running KGB-based midpoint finding to enhance knowledge graph...")
                
                # Find midpoints between consecutive queries to create a more detailed knowledge graph path
                queries_list = run_midpoint_finding_kgb(queries_list)
                
                logger.info(f"Knowledge graph enhanced with midpoints. New query count: {len(queries_list)}")
        
        #### KGB Agent done
    ##### Answer mode done
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the paper search pipeline")
    parser.add_argument("--force-harvesting", action="store_true", help="Force metadata harvesting regardless of schedule")
    parser.add_argument("--run-embedding", action="store_true", help="Run metadata embedding step")
    parser.add_argument("--db-path", type=str, help="Path to the SQLite database")
    parser.add_argument("--arxiv-id", type=str, help="arXiv ID of the paper to download")
    parser.add_argument("--max-papers", type=int, help="Maximum number of papers to process for content embedding")
    
    args = parser.parse_args()
    main(args)
