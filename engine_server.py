from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import traceback
import json
import sys
import subprocess
import logging
import datetime
import re
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path

# Set environment variable to disable tokenizers parallelism
# This prevents the warning: "huggingface/tokenizers: The current process just got forked..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from module_query_obtention.submodule_github_query.subsubmodule_recent_work.main import generate_recent_work_query
from module_query_obtention.submodule_github_query.subsubmodule_general_understanding.main import generate_repo_based_query

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"engine_server_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("engine_server")

# Add global variables to store the latest query and recommendation query
latest_query = None
latest_recommendation_query = None

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/query")
async def query_endpoint(request: Request):
    try:
        # Get the raw request body first
        raw_body = await request.body()
        
        # Try to decode and parse as JSON with explicit encoding handling
        try:
            body_str = raw_body.decode('utf-8')
            data = json.loads(body_str)
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try with latin-1 which accepts any byte value
            body_str = raw_body.decode('latin-1')
            data = json.loads(body_str)
            
        repo = data.get("repo")
        github_token = data.get("github_token")
        
        if not repo:
            raise HTTPException(status_code=400, detail="Repository name is required")
        
        print(f"Received query for repo: {repo}")
        
        # Set default parameters
        commit_limit = data.get("commit_limit", 10)
        hot_files_limit = data.get("hot_files_limit", 5)
        
        # Set GitHub token in environment if provided
        if github_token:
            os.environ["GITHUB_TOKEN"] = github_token
            print("Using GitHub token from request")
        else:
            print("No GitHub token provided in request")
            
        # Use OpenAI API key from environment variables
        if os.environ.get("OPENAI_API_KEY"):
            print("Using OpenAI API key from environment variables")
        else:
            print("Warning: OpenAI API key not found in environment variables")
        
        # Generate a query based on recent work in the repository
        try:
            # Use our new generate_recent_work_query function
            query = generate_recent_work_query(repo, github_token)
            logger.info(f"Generated recent work query: {query}")
            
            # Return the query result
            return {
                "repo": repo,
                "query": query,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error generating recent work query: {str(e)}")
            # Return a generic query if there's an error
            fallback_query = f"Recent developments in software projects similar to {repo}"
            
            return {
                "repo": repo,
                "query": fallback_query,
                "status": "fallback",
                "error": str(e)
            }
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/research-query")
async def research_query_endpoint(request: Request):
    try:
        # Get the raw request body
        raw_body = await request.body()
        
        # Decode and parse as JSON
        try:
            body_str = raw_body.decode('utf-8')
            data = json.loads(body_str)
        except UnicodeDecodeError:
            body_str = raw_body.decode('latin-1')
            data = json.loads(body_str)
        
        # Extract the query from the request
        query = data.get("query")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Check if a specific mode is requested (smart-search or smart-answer)
        requested_mode = data.get("mode")
        
        print(f"Received research query: {query}")
        if requested_mode:
            print(f"Requested mode: {requested_mode}")
        
        # Store the query globally so it can be accessed by the query_input module
        global latest_query
        latest_query = query
        
        # Make sure the output directory exists
        output_dir = Path("module_paper_search/submodule_answer_generation/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir.absolute()}")
        
        # Run the paper search pipeline in a separate process
        print(f"\n\n===== RUNNING PAPER SEARCH PIPELINE WITH QUERY: '{query}' =====\n")
        
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["USER_QUERY"] = query  # Set the query as an environment variable
        
        # Set the search mode based on the requested mode parameter
        if requested_mode == "smart-search":
            env["SEARCH_MODE"] = "recommendation"
            print("Setting search mode to 'recommendation' for smart-search request")
        else:
            # Default to answer mode for smart-answer or if no mode specified
            env["SEARCH_MODE"] = "answer"
            print("Setting search mode to 'answer' for smart-answer request")
        
        
        try:
            # Run the pipeline
            result = subprocess.run(
                [sys.executable, "-m", "module_paper_search.main"],
                capture_output=False,  # Don't capture output, let it print to terminal
                text=True,
                check=True,
                env=env  # Pass the environment variables
            )
            print("\n===== PIPELINE EXECUTION COMPLETED SUCCESSFULLY =====\n")
        except subprocess.CalledProcessError as e:
            print(f"Pipeline execution failed: {e}")
            if hasattr(e, 'stdout'):
                print(f"STDOUT: {e.stdout}")
            if hasattr(e, 'stderr'):
                print(f"STDERR: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")
        

        #print("\n===== MANUAL JSON =====\n")
        
        # Use the search mode we explicitly set in the environment variable
        # This ensures consistency with what was actually used in the subprocess
        search_mode = env.get("SEARCH_MODE", "answer")
        print(f"Using search mode from environment: {search_mode}")
        
        # Determine the correct output file path based on the search mode
        if search_mode == "answer":
            answer_file_json = Path("results/answer_mode/answer_kgb.json")
            answer_file_txt = Path("results/answer_mode/answer_kgb.txt")
            print(f"Looking for answer file at: {answer_file_json.absolute()} or {answer_file_txt.absolute()}")
        else:  # recommendation mode
            answer_file_json = Path("results/recommendation_mode/tweets.json")
            answer_file_txt = Path("results/recommendation_mode/tweets.txt")
            print(f"Looking for recommendation file at: {answer_file_json.absolute()} or {answer_file_txt.absolute()}")
        
        # Check if the JSON file exists
        if answer_file_json.exists():
            print(f"Answer file found at: {answer_file_json.absolute()}")
            try:
                with open(answer_file_json, "r") as f:
                    answer_data = json.load(f)
                print("Answer file loaded successfully")
                print(f"JSON answer data: {json.dumps(answer_data, indent=2)}")
                
                # Extract relevant information based on search mode
                papers = []
                answer_text = ""
                
                if search_mode == "answer":
                    # Extract answer text for answer mode
                    answer_text = answer_data.get("answer", answer_data.get("answer_text", "No answer was generated."))
                    print(f"Extracted answer text from JSON: '{answer_text}'")
                    
                    # If the answer was truncated (ends with ...), check if we need to handle it
                    if answer_text.endswith("...") or answer_text.endswith("\n"):
                        print("Answer appears to be truncated in the log, but we have the full content in the variable")
                    
                    # Extract paper information for answer mode
                    for chunk in answer_data.get("top_chunks", []):
                        paper_id = chunk.get("paper_id", "unknown")
                        paper_title = chunk.get("paper_title", "Unknown Title")
                        paper_authors = chunk.get("paper_authors", "Unknown Authors")
                        paper_year = chunk.get("paper_year", "")
                        paper_journal = chunk.get("paper_venue", "")
                        paper_abstract = chunk.get("paper_abstract", "")
                        paper_url = f"https://arxiv.org/abs/{paper_id}" if paper_id.startswith("arXiv:") else ""
                        
                        # Check if this paper is already in the list
                        if not any(p.get("id") == paper_id for p in papers):
                            papers.append({
                                "id": paper_id,
                                "title": paper_title,
                                "authors": paper_authors,
                                "year": paper_year,
                                "journal": paper_journal,
                                "abstract": paper_abstract,
                                "url": paper_url,
                                "citations": 0  # We don't have citation data
                            })
                else:  # recommendation mode
                    # For recommendation mode, the structure is different
                    # Combine all recommendations into a single text
                    recommendations = answer_data.get("recommendations", [])
                    recommendation_texts = []
                    
                    for i, rec in enumerate(recommendations):
                        tweet_text = rec.get("tweet_text", "")
                        if tweet_text:
                            recommendation_texts.append(f"Recommendation {i+1}: {tweet_text}")
                            
                        # Extract paper information from recommendations
                        paper_ids = rec.get("paper_ids", [])
                        for paper_id in paper_ids:
                            # Try to extract paper details from the tweet text or other fields
                            # This is a simplified version - in a real implementation, you'd want to extract more details
                            if not any(p.get("id") == paper_id for p in papers):
                                papers.append({
                                    "id": paper_id,
                                    "title": f"Paper {paper_id}",  # Basic placeholder
                                    "authors": "Authors not available in recommendation mode",
                                    "year": "",
                                    "journal": "",
                                    "abstract": "",
                                    "url": f"https://arxiv.org/abs/{paper_id}" if paper_id.startswith("arXiv:") else "",
                                    "citations": 0
                                })
                    
                    # Join all recommendation texts
                    answer_text = "\n\n".join(recommendation_texts) if recommendation_texts else "No recommendations were generated."
                
                response_data = {
                    "text": answer_text,
                    "papers": papers
                }
                print(f"Returning response: {json.dumps(response_data, indent=2)}")
                return response_data
            except Exception as e:
                print(f"Error loading answer file: {e}")
                raise HTTPException(status_code=500, detail=f"Error loading answer file: {e}")
        # If JSON file doesn't exist, try to read the TXT file
        elif answer_file_txt.exists():
            print(f"Text answer file found at: {answer_file_txt.absolute()}")
            try:
                with open(answer_file_txt, "r") as f:
                    file_content = f.read()
                print("Text answer file loaded successfully")
                print(f"Raw file content: {file_content}")
                
                # Extract text content based on search mode
                answer_text = ""
                lines = file_content.split('\n')
                
                if search_mode == "answer":
                    # Extract just the answer text from the file for answer mode
                    # The format is typically "Query: ...", then "Answer: ...", followed by multiple lines of answer
                    in_answer_section = False
                    answer_lines = []
                    
                    for i, line in enumerate(lines):
                        print(f"Line {i}: {line}")
                        
                        # Start capturing when we find the Answer: line
                        if line.startswith("Answer:"):
                            in_answer_section = True
                            # Get the text after "Answer:"
                            initial_answer = line.replace("Answer:", "").strip()
                            if initial_answer:  # If there's text on the same line as "Answer:"
                                answer_lines.append(initial_answer)
                            continue
                        
                        # Stop capturing when we hit the next section (Quality Score, etc.)
                        if in_answer_section and (line.startswith("Quality Score:") or not line.strip()):
                            in_answer_section = False
                            continue
                        
                        # Capture all lines while in the answer section
                        if in_answer_section and line.strip():
                            answer_lines.append(line.strip())
                    
                    # Join all the answer lines
                    answer_text = " ".join(answer_lines)
                    print(f"Found answer section with {len(answer_lines)} lines: '{answer_text}'")
                    
                    # If the answer is just "Insufficient relevant context", provide a better message
                    if answer_text == "Insufficient relevant context":
                        answer_text = "I couldn't find enough relevant information to answer your query. Please try a more specific question or a different topic."
                else:  # recommendation mode
                    # Extract recommendations from the text file
                    # For recommendation mode, the format is typically "Query: ...", followed by "Recommendations:"
                    # and then multiple recommendation entries
                    in_recommendations_section = False
                    recommendation_lines = []
                    current_recommendation = ""
                    recommendations = []
                    
                    for i, line in enumerate(lines):
                        print(f"Line {i}: {line}")
                        
                        # Start capturing when we find the Recommendations: line
                        if "Recommendations:" in line:
                            in_recommendations_section = True
                            continue
                        
                        # Process lines in the recommendations section
                        if in_recommendations_section:
                            # Check if this is the start of a new recommendation
                            if line.strip().startswith("Recommendation") or line.strip().startswith("Tweet"):
                                # Save the previous recommendation if it exists
                                if current_recommendation:
                                    recommendations.append(current_recommendation)
                                    current_recommendation = ""
                                # Start a new recommendation
                                current_recommendation = line.strip()
                            elif line.strip() and current_recommendation:  # Continue current recommendation
                                current_recommendation += " " + line.strip()
                    
                    # Add the last recommendation if it exists
                    if current_recommendation:
                        recommendations.append(current_recommendation)
                    
                    # Join all recommendations with newlines
                    answer_text = "\n\n".join(recommendations) if recommendations else "No recommendations were found in the file."
                    print(f"Found {len(recommendations)} recommendations")
                    
                    # If we couldn't find any recommendations, use a default message
                    if not recommendations:
                        answer_text = "No paper recommendations were generated for your query. Please try a different search term."
                
                # If we couldn't find an answer line, use the whole content
                if not answer_text:
                    print("Warning: Could not extract answer from text file, using full content")
                    answer_text = file_content
                
                print(f"Final extracted answer text: '{answer_text}'")
                
                # For text files, we don't have structured paper information
                # So we return just the text without paper recommendations
                response_data = {
                    "text": answer_text,
                    "papers": [],
                    "debug_info": {
                        "file_path": str(answer_file_txt.absolute()),
                        "file_content": file_content
                    }
                }
                print(f"Returning response: {json.dumps(response_data, indent=2)}")
                return response_data
            except Exception as e:
                print(f"Error loading text answer file: {e}")
                raise HTTPException(status_code=500, detail=f"Error loading text answer file: {e}")
        else:
            # Check if the results directory exists and list its contents
            results_dir = Path("results")
            if results_dir.exists():
                print(f"Results directory exists: {results_dir.absolute()}")
                print("Directories in results:")
                for directory in results_dir.iterdir():
                    if directory.is_dir():
                        print(f"  {directory.name}:")
                        for file in directory.iterdir():
                            print(f"    {file.name}")
            else:
                print(f"Results directory does not exist: {results_dir.absolute()}")
            
            return {
                "text": "The pipeline ran but no answer was generated. Please try a different query.",
                "papers": []
            }
    
    except Exception as e:
        print(f"Error processing research query: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing research query: {str(e)}")

def get_latest_query():
    """
    Returns the latest query received from the API.
    This function is called by the query_input module.
    """
    global latest_query
    global latest_recommendation_query
    
    # If we have a recommendation query, prioritize it
    if latest_recommendation_query:
        logger.info(f"Returning latest recommendation query: {latest_recommendation_query}")
        return latest_recommendation_query
    
    logger.info(f"Returning latest regular query: {latest_query}")
    return latest_query

@app.post("/github-recommendations")
async def github_recommendations_endpoint(request: Request):
    """
    Endpoint for generating paper recommendations based on GitHub repository content.
    This endpoint directly calls the module_paper_search/main.py script in recommendation mode.
    """
    try:
        # Get the raw request body
        raw_body = await request.body()
        
        # Decode and parse as JSON
        try:
            body_str = raw_body.decode('utf-8')
            data = json.loads(body_str)
        except UnicodeDecodeError:
            body_str = raw_body.decode('latin-1')
            data = json.loads(body_str)
        
        # Extract the repository name from the request
        repo_name = data.get("repoName")
        
        if not repo_name:
            logger.error("Repository name is required")
            raise HTTPException(status_code=400, detail="Repository name is required")
        
        logger.info(f"Received GitHub recommendation request for repo: {repo_name}")
        
        # Get GitHub token from various possible sources
        # First check if it's directly in the request data
        github_token = data.get("github_token")
        
        # If not in request data, check for access_token in the request
        if not github_token:
            github_token = data.get("access_token")
        
        # If still not found, check environment variables
        if not github_token:
            # Check various possible environment variable names for GitHub tokens
            for env_var in ["GITHUB_TOKEN", "GITHUB_ACCESS_TOKEN", "GITHUB_OAUTH_TOKEN", "GITHUB_SECRET"]:
                if os.environ.get(env_var):
                    github_token = os.environ.get(env_var)
                    logger.info(f"Using GitHub token from environment variable: {env_var}")
                    break
        
        # If we have a token, make sure it's properly formatted
        if github_token:
            # Remove any leading/trailing whitespace
            github_token = github_token.strip()
            logger.info("GitHub token is available for repository access")
        else:
            logger.warning("No GitHub token found in request or environment variables")
        
        # Generate a query based on the repository using our new module
        
        try:
            query = generate_repo_based_query(repo_name, github_token)
            logger.info(f"Generated query using repo analysis: {query}")
            
            # Store the query globally so it can be accessed by the query_input module
            global latest_recommendation_query
            latest_recommendation_query = query
        except ValueError as e:
            # If we couldn't generate a query, return an error
            error_message = str(e)
            logger.error(f"Failed to generate query: {error_message}")
            return JSONResponse(
                content={
                    "error": "Failed to access repository",
                    "message": error_message,
                    "status": "error"
                },
                status_code=400
            )
        
        #print("\n===== MANUAL query recs=====\n")
        #query = "The repository in question is primarily focused on the analysis of di-Higgs boson production through photon-photon collisions at a hypothetical 380 GeV collider, utilizing a sophisticated simulation and analysis framework. The core of the repository is structured around the deployment of Delphes, a highly configurable tool for fast simulation of a multipurpose detector in a collider experiment. Delphes is employed here to process Monte Carlo (MC) samples, which are provided in the HepMC format and subsequently converted into ROOT files for further analysis. This conversion facilitates the extraction of relevant physical observables from the simulated data. The repository features a complex pipeline integrating various custom modules and configuration settings tailored for the gamma-gamma collider scenario, indicative of its focus on high-energy physics (HEP) research. These modules, likely implemented in C++ given the presence of .cxx and .pcm files, extend the Delphes framework to enforce specific constraints and adapt the simulation to the unique experimental conditions dictated by photon-photon interactions. The presence of TCL configuration files such as 'delphes_card_SiD_2024_XCC.tcl' suggests customization of the Delphes cards to specify detector settings, which are presumably aligned with the SiD detector's specifications as outlined in referenced academic literature. Further down the analysis pipeline, the repository implements a machine learning approach to optimize signal-background separation. This is achieved through the use of XGBoost, an ensemble learning algorithm based on decision trees, combined with a genetic algorithm to enhance the feature selection process. This methodology is directed towards calculating the sensitivity of the di-Higgs cross-section measurement, a key metric in validating theoretical models of Higgs boson interactions. The feature extraction from ROOT files is converted into a more manageable CSV format, likely serving as input to the machine learning models. Overall, the technical ecosystem of this repository is emblematic of modern computational physics research, integrating fast detector simulation, advanced statistical analysis, and machine learning to address complex problems in particle physics. The repository not only utilizes but also contributes to the evolving landscape of tools and techniques in high-energy physics, making it a pertinent subject for studies on simulation methodologies, detector optimization, and machine learning applications in particle physics."
                
        # Set environment variables
        env = os.environ.copy()
        env["USER_QUERY"] = query
        # EXPLICITLY set the search mode to "recommendation" for GitHub endpoints
        env["SEARCH_MODE"] = "recommendation"
        logger.info(f"Set USER_QUERY environment variable: {query}")
        logger.info("Setting search mode to 'recommendation' for GitHub endpoint")
        
        # Run the paper search pipeline
        logger.info("Running paper search pipeline in recommendation mode")
        pipeline_script = Path("module_paper_search/main.py").absolute()
        try:
            
            # Run the pipeline without capturing output to allow real-time logging
            logger.info("\n\n===== RUNNING PAPER SEARCH PIPELINE FOR OVERALL REPO RECOMMENDATIONS =====\n")
            result = subprocess.run(
                ["python", str(pipeline_script)],
                capture_output=False,  # Don't capture output, let it print to terminal in real-time
                text=True,
                env=env,  # Use our environment with SEARCH_MODE set to recommendation
                check=True
            )
            logger.info("\n===== PIPELINE EXECUTION COMPLETED SUCCESSFULLY =====\n")
            
            #print("\n===== MANUAL JSON RECS=====\n")

            # Define the query types and their corresponding JSON files
            query_types = [
                {
                    "name": "Check these out based on your repo's architecture",
                    "description": "System architecture, design patterns, and overall structure",
                    "file": "tweets_architecture.json"
                },
                {
                    "name": "Check these out based on your repo's technical implementation",
                    "description": "Specific technologies, libraries, and frameworks",
                    "file": "tweets_technical.json"
                },
                {
                    "name": "Check these out based on your repo's algorithmic approach",
                    "description": "Algorithms, mathematical models, and computational techniques",
                    "file": "tweets_algorithmic.json"
                },
                {
                    "name": "Check these out based on your repo's domain-specific aspect",
                    "description": "Specific academic domain and research methodologies",
                    "file": "tweets_domain.json"
                },
                {
                    "name": "Check these out based on your repo's integration pipeline",
                    "description": "Component interactions and pipeline structure",
                    "file": "tweets_integration.json"
                }
            ]
            
            # Create a structured output with categories
            output_data = {
                "query": query,
                "timestamp": datetime.datetime.now().isoformat(),
                "categories": []
            }
            
            # Process each query type
            for query_type in query_types:
                json_path = Path(f"results/recommendation_mode/{query_type['file']}")
                
                if not json_path.exists():
                    logger.warning(f"File not found: {json_path}")
                    continue
                
                try:
                    with open(json_path, "r") as f:
                        tweets_data = json.load(f)
                    
                    # Create a category object
                    category = {
                        "name": query_type["name"],
                        "description": query_type["description"],
                        "tweets": []
                    }
                    
                    # Extract tweets
                    for tweet_text in tweets_data.get("tweets", []):
                        # Extract paper IDs from the tweet text
                        paper_ids = []
                        # Match both [ID: id] and [id] formats (where id is typically a number or arXiv identifier)
                        for match in re.finditer(r'\[(?:ID: )?([0-9]+(?:\.[0-9]+)?(?:v[0-9]+)?|(?:arXiv:)?[0-9]+\.[0-9]+(?:v[0-9]+)?)\]', tweet_text):
                            paper_id = match.group(1)
                            # Add arXiv: prefix if not already present and it looks like an arXiv ID
                            if not paper_id.startswith('arXiv:') and re.match(r'^[0-9]+\.[0-9]+(?:v[0-9]+)?$', paper_id):
                                paper_id = f'arXiv:{paper_id}'
                            if paper_id not in paper_ids:
                                paper_ids.append(paper_id)
                        
                        # Add the tweet to the category
                        category["tweets"].append({
                            "text": tweet_text,
                            "paper_ids": paper_ids
                        })
                    
                    # Add the category to the output data
                    output_data["categories"].append(category)
                    
                    logger.info(f"Loaded {len(category['tweets'])} tweets for {query_type['name']}")
                    
                except Exception as e:
                    logger.error(f"Error reading {json_path}: {str(e)}")
                    # Continue with other files instead of failing completely
            
            # Log what we're returning for debugging
            total_tweets = sum(len(category['tweets']) for category in output_data['categories'])
            logger.info(f"Total tweets: {total_tweets}")
            
            # Log the breakdown by category
            for category in output_data['categories']:
                logger.info(f"  {category['name']}: {len(category['tweets'])} tweets")
            
            # Add a flag to indicate this is for GitHub recommendations
            output_data["isGitHubRecommendation"] = True
            
            logger.info("Successfully read recommendations from individual tweet files")
            return JSONResponse(content=output_data)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running paper search pipeline: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return JSONResponse(
                content={
                    "error": "Failed to run paper search pipeline",
                    "details": e.stderr
                },
                status_code=500
            )
        finally:
            # No need to restore files since we're using environment variables now
            pass
            # Removed backup file operations since we're not creating backups anymore
            
            # Reset the recommendation query
            # The variable is already declared as global earlier in this function
            latest_recommendation_query = None
        
        # Read the recommendation results
        recommendation_file = Path("results/recommendation_mode/tweets.json")
        logger.info(f"Reading recommendation results from: {recommendation_file.absolute()}")
        
        if not recommendation_file.exists():
            logger.error(f"Recommendation file not found at: {recommendation_file.absolute()}")
            raise HTTPException(
                status_code=404,
                detail="Recommendation results not found. Pipeline may have failed to generate output."
            )
        
        try:
            with open(recommendation_file, "r") as f:
                recommendation_data = json.load(f)
                logger.info(f"Successfully loaded recommendation data with {len(recommendation_data.get('recommendations', []))} recommendations")
                logger.debug(f"Recommendation data: {json.dumps(recommendation_data, indent=2)}")
                
                return recommendation_data
        except Exception as e:
            logger.exception(f"Error reading recommendation results: {e}")
            raise HTTPException(status_code=500, detail=f"Error reading recommendation results: {str(e)}")
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in github-recommendations endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/github-recommendations-recent-work")
async def github_recommendations_recent_work_endpoint(request: Request):
    """
    Endpoint for generating paper recommendations based on recent work in a GitHub repository.
    This endpoint uses a different query focused on recent work but still uses recommendation mode.
    """
    try:
        # Get the raw request body
        raw_body = await request.body()
        
        # Decode and parse as JSON
        try:
            body_str = raw_body.decode('utf-8')
            data = json.loads(body_str)
        except UnicodeDecodeError:
            body_str = raw_body.decode('latin-1')
            data = json.loads(body_str)
        
        # Extract the repository name from the request
        repo_name = data.get("repoName")
        
        if not repo_name:
            logger.error("Repository name is required")
            raise HTTPException(status_code=400, detail="Repository name is required")
        
        logger.info(f"Received GitHub recommendation request for recent work in repo: {repo_name}")
        
        # Get GitHub token from various possible sources
        # First check if it's directly in the request data
        github_token = data.get("github_token")
        
        # If not in request data, check for access_token in the request
        if not github_token:
            github_token = data.get("access_token")
        
        # If still not found, check environment variables
        if not github_token:
            # Check various possible environment variable names for GitHub tokens
            for env_var in ["GITHUB_TOKEN", "GITHUB_ACCESS_TOKEN", "GITHUB_OAUTH_TOKEN", "GITHUB_SECRET"]:
                if os.environ.get(env_var):
                    github_token = os.environ.get(env_var)
                    logger.info(f"Using GitHub token from environment variable: {env_var}")
                    break
        
        # If we have a token, make sure it's properly formatted
        if github_token:
            # Remove any leading/trailing whitespace
            github_token = github_token.strip()
            logger.info("GitHub token is available for repository access")
        else:
            logger.warning("No GitHub token found in request or environment variables")
        
        # Import the recent work query generation module
        
        try:
            from module_query_obtention.submodule_github_query.subsubmodule_recent_work import generate_recent_work_query
            
            # Generate a query based on recent work in the repository
            logger.info(f"Generating recent work query using the recent work module")
            query = generate_recent_work_query(repo_name, github_token)
            logger.info(f"Successfully generated recent work query")
            
        except Exception as e:
            logger.error(f"Error using recent work module: {str(e)}")
            # Fallback to hardcoded queries if the module fails
            logger.warning("Falling back to hardcoded queries")
            if repo_name == "XCC_gammagamma_HH_bbbb":
                query = "kk"
            elif repo_name == "ML_Fundamental_Notes":
                query = "kk"
            else:
                query = f"Recent developments and cutting-edge techniques related to {repo_name} technologies and methodologies"
            
        
        logger.info(f"Generated recent work query: {query[:100]}..." if len(query) > 100 else f"Generated recent work query: {query}")
         
        #print("\n===== MANUAL query recent recs=====\n")
        #query = "The recent development activities within the repository are concentrated on advancing machine learning techniques, particularly using XGBoost, for signal-background separation and the sensitivity measurement of the di-Higgs cross-section. This involves a comprehensive pipeline set up for data handling, feature extraction, model training, and sensitivity calculations, all tailored to high-energy physics data analysis. The development focus includes the implementation of a genetic algorithm to optimize the parameters used in the machine learning models, which is indicative of a push towards refining the accuracy of the predictive models. This algorithm operates in conjunction with XGBoost models trained for classification tasks, where features of particle events are used to distinguish between signal and background datasets. The use of genetic algorithms suggests an adaptive approach to optimize the selection criteria automatically based on evolutionary strategies, enhancing the model's performance iteratively. Moreover, the integration of Numba for JIT compilation in the sensitivity calculation scripts highlights an optimization effort aimed at improving computational efficiency. This is crucial in scenarios where the computation of statistical significance needs to be fast and efficient across large datasets. The Numba-optimized functions are designed to perform array operations at high speeds, essential for real-time data processing in physics experiments. The codebase also shows a robust data handling and preprocessing framework using pandas and NumPy, preparing datasets for machine learning applications. The use of standard libraries for data manipulation and preparation underscores the project's reliance on proven data science tools to handle complex and voluminous datasets typical in particle physics. These changes are part of a broader effort to enhance the precision and efficiency of particle physics experiments through advanced computational techniques, reflecting a significant evolution in the repository towards integrating more sophisticated data analysis and machine learning methods to solve high-stakes physics problems."
        

        
        
        # Store the query globally
        global latest_recommendation_query
        latest_recommendation_query = query
        
        # Set environment variables
        env = os.environ.copy()
        env["USER_QUERY"] = query
        # EXPLICITLY set the search mode to "recommendation" for GitHub endpoints
        env["SEARCH_MODE"] = "recommendation"
        # Set a flag to indicate this is for recent work recommendations
        env["IS_RECENT_WORK"] = "true"
        logger.info(f"Set USER_QUERY environment variable: {query}")
        logger.info("Setting search mode to 'recommendation' for GitHub endpoint")
        logger.info("Setting IS_RECENT_WORK flag to 'true' for recent work recommendations")
        
        # Modify the search_mode.py file to use recommendation mode
        search_mode_path = Path("module_query_obtention/search_mode.py")
        logger.info(f"Modifying search mode file at: {search_mode_path.absolute()}")
        
        # Create a backup of the original file
        backup_path = search_mode_path.with_suffix(".py.bak")
        with open(search_mode_path, "r") as f:
            original_content = f.read()
            logger.debug(f"Original search_mode.py content:\n{original_content}")
        
        with open(backup_path, "w") as f:
            f.write(original_content)
            logger.info(f"Created backup at: {backup_path.absolute()}")
        
        # Modify the file to use recommendation mode
        updated_content = original_content
        if "mode = \"answer\"" in original_content:
            updated_content = original_content.replace(
                'mode = "answer"',
                'mode = "recommendation"  # Set to recommendation mode for GitHub integration'
            )
        elif "#mode = \"recommendation\"" in original_content:
            updated_content = original_content.replace(
                '#mode = "recommendation"',
                'mode = "recommendation"  # Set to recommendation mode for GitHub integration'
            )
        else:
            logger.warning("Could not find mode line in search_mode.py, creating a new one")
            updated_content = original_content.replace(
                "def get_search_mode():",
                'def get_search_mode():\n    # Hardcoded to recommendation mode for GitHub integration\n    mode = "recommendation"'
            )
        
        with open(search_mode_path, "w") as f:
            f.write(updated_content)
            logger.info("Updated search_mode.py to use recommendation mode")
            logger.debug(f"Updated search_mode.py content:\n{updated_content}")
        
        # Run the paper search pipeline
        logger.info("Running paper search pipeline in recommendation mode for recent work")
        pipeline_script = Path("module_paper_search/main.py").absolute()
        try:
            
            # Run the pipeline without capturing output to allow real-time logging
            logger.info("\n\n===== RUNNING PAPER SEARCH PIPELINE FOR RECENT WORK RECOMMENDATIONS =====\n")
            result = subprocess.run(
                ["python", str(pipeline_script)],
                capture_output=False,  # Don't capture output, let it print to terminal in real-time
                text=True,
                env=env,  # Use our environment with SEARCH_MODE set to recommendation
                check=True
            )
            logger.info("\n===== PIPELINE EXECUTION COMPLETED SUCCESSFULLY =====\n")
            
            #print("\n===== MANUAL JSON RECENT RECS=====\n")
            
            # Define the query types and their corresponding JSON files for recent work
            query_types = [
                {
                    "name": "Check these out based on your repo's architecture",
                    "description": "System architecture, design patterns, and overall structure",
                    "file": "tweets_architecture(1).json"
                },
                {
                    "name": "Check these out based on your repo's technical implementation",
                    "description": "Specific technologies, libraries, and frameworks",
                    "file": "tweets_technical(1).json"
                },
                {
                    "name": "Check these out based on your repo's algorithmic approach",
                    "description": "Algorithms, mathematical models, and computational techniques",
                    "file": "tweets_algorithmic(1).json"
                },
                {
                    "name": "Check these out based on your repo's domain-specific aspect",
                    "description": "Specific academic domain and research methodologies",
                    "file": "tweets_domain(1).json"
                },
                {
                    "name": "Check these out based on your repo's integration pipeline",
                    "description": "Component interactions and pipeline structure",
                    "file": "tweets_integration(1).json"
                }
            ]
            
            # Create a structured output with categories
            output_data = {
                "query": query,
                "timestamp": datetime.datetime.now().isoformat(),
                "categories": [],
                "isRecentWork": True
            }
            
            # Process each query type
            for query_type in query_types:
                json_path = Path(f"results/recommendation_mode/{query_type['file']}")
                
                if not json_path.exists():
                    logger.warning(f"File not found: {json_path}")
                    continue
                
                try:
                    with open(json_path, "r") as f:
                        tweets_data = json.load(f)
                    
                    # Create a category object
                    category = {
                        "name": query_type["name"],
                        "description": query_type["description"],
                        "tweets": []
                    }
                    
                    # Extract tweets
                    for tweet_text in tweets_data.get("tweets", []):
                        # Extract paper IDs from the tweet text
                        paper_ids = []
                        # Match both [ID: id] and [id] formats (where id is typically a number or arXiv identifier)
                        for match in re.finditer(r'\[(?:ID: )?([0-9]+(?:\.[0-9]+)?(?:v[0-9]+)?|(?:arXiv:)?[0-9]+\.[0-9]+(?:v[0-9]+)?)\]', tweet_text):
                            paper_id = match.group(1)
                            # Add arXiv: prefix if not already present and it looks like an arXiv ID
                            if not paper_id.startswith('arXiv:') and re.match(r'^[0-9]+\.[0-9]+(?:v[0-9]+)?$', paper_id):
                                paper_id = f'arXiv:{paper_id}'
                            if paper_id not in paper_ids:
                                paper_ids.append(paper_id)
                        
                        # Add the tweet to the category
                        category["tweets"].append({
                            "text": tweet_text,
                            "paper_ids": paper_ids
                        })
                    
                    # Add the category to the output data
                    output_data["categories"].append(category)
                    
                    logger.info(f"Loaded {len(category['tweets'])} tweets for {query_type['name']}")
                    
                except Exception as e:
                    logger.error(f"Error reading {json_path}: {str(e)}")
                    # Continue with other files instead of failing completely
            
            # Log what we're returning for debugging
            total_tweets = sum(len(category['tweets']) for category in output_data['categories'])
            logger.info(f"Total tweets: {total_tweets}")
            
            # Log the breakdown by category
            for category in output_data['categories']:
                logger.info(f"  {category['name']}: {len(category['tweets'])} tweets")
            
            # Add a flag to indicate this is for GitHub recommendations
            output_data["isGitHubRecommendation"] = True
            
            logger.info("Successfully read recommendations from individual tweet files")
            return JSONResponse(content=output_data)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running paper search pipeline: {e}")
            logger.error(f"Stderr: {e.stderr}")
            return JSONResponse(
                content={
                    "error": "Failed to run paper search pipeline",
                    "details": e.stderr
                },
                status_code=500
            )
        finally:
            # No need to restore files since we're using environment variables now
            pass
            # Removed backup file operations since we're not creating backups anymore
            
            # Reset the recommendation query
            latest_recommendation_query = None
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in github-recommendations-recent-work endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    # Directly read and set environment variables from .env file
    try:
        with open('.env', 'r') as env_file:
            for line in env_file:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        logger.info("Environment variables loaded from .env file")
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
    
    # Check if required environment variables are set
    if not os.environ.get("GITHUB_TOKEN"):
        logger.warning("GITHUB_TOKEN environment variable not set. API rate limits may apply.")
    else:
        logger.info("GITHUB_TOKEN is set successfully.")
    
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set. Summary generation will fail.")
    else:
        logger.info("OPENAI_API_KEY is set successfully.")
    # Start the server
    logger.info("Starting the FastAPI server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
