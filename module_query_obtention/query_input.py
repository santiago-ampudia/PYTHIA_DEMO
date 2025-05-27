# module_query_obtention/query_input.py

"""
Module 1: Query Obtention
-------------------------
This module is responsible for obtaining the user's query as a plain string.
It now supports getting queries from the API or using a default query if none is provided.

The module interfaces with the engine_server.py API to get queries submitted by users
through the web interface.
"""

# Import the function to get the latest query from the API
try:
    from engine_server import get_latest_query
except ImportError:
    # If we can't import from engine_server, create a dummy function
    def get_latest_query():
        return None

def get_user_query():
    """
    Returns the user query as a plain string.
    
    This function first checks for a USER_QUERY environment variable.
    If that's not available, it tries to get the latest query from the API.
    If no query is available from either source, it falls back to a default query.
    
    Returns:
        str: The user's query in natural language, as if typed or spoken by the user.
    """
    # First, check for the USER_QUERY environment variable
    import os
    env_query = os.environ.get("USER_QUERY")
    if env_query:
        print(f"Using query from environment variable: {env_query}")
        return env_query
    
    # If no environment variable, try to get the query from the API
    api_query = get_latest_query()
    if api_query:
        print(f"Using query from API: {api_query}")
        return api_query
    
    # If no query is available from the API or environment, use a default query
    print("No query available from API or environment, using default query")
    default_query = "which jet clustering algorithm should I run for event reconstruction (so on reconstruction level, not truth level) for studying the Higgs Self-Coupling at the XCC, an X-ray FEL-based gamma gamma Compton Collider Higgs Factory"
    return default_query

if __name__ == "__main__":
    # If this script is run directly, obtain the user query and print it to the console.
    user_query = get_user_query()
    print("User Query:", user_query)  # Output the query for verification
