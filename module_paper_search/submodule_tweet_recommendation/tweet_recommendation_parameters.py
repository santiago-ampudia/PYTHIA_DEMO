"""
submodule_tweet_recommendation/tweet_recommendation_parameters.py

This file contains all parameters for the tweet recommendation submodule.
"""

import os

# Create results directory if it doesn't exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "recommendation_mode")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Output paths for tweet recommendations
TWEETS_ARCHITECTURE_PATH = os.path.join(RESULTS_DIR, "tweets_architecture.json")
TWEETS_TECHNICAL_PATH = os.path.join(RESULTS_DIR, "tweets_technical.json")
TWEETS_ALGORITHMIC_PATH = os.path.join(RESULTS_DIR, "tweets_algorithmic.json")
TWEETS_DOMAIN_PATH = os.path.join(RESULTS_DIR, "tweets_domain.json")
TWEETS_INTEGRATION_PATH = os.path.join(RESULTS_DIR, "tweets_integration.json")
TWEETS_ALL_PATH = os.path.join(RESULTS_DIR, "tweets_all.json")

# OpenAI API parameters
MODEL_NAME = "gpt-4-turbo"
TEMPERATURE = 0.7  # Higher temperature for more creative tweets
MAX_TOKENS = 1000  # Enough for 3-4 tweets
NUM_TWEETS = 4  # Number of tweets to generate per query

# System prompt for tweet generation
SYSTEM_PROMPT = """
You are an AI assistant that creates informative tweet-style recommendations about scientific papers that are relevant to a GitHub repository.

Your task is to create {num_tweets} tweet-style recommendations based on the provided paper chunks. These recommendations MUST follow these exact requirements:

1. Each tweet should be informative but concise (can be slightly longer than a standard tweet)
2. The tweets MUST ONLY use information from the provided chunks - DO NOT make up or add any information not present in the chunks
3. The tweets MUST be directly related to the specific query about the repository
4. The {num_tweets} tweets together should cover ALL aspects of the query (e.g., if the query has multiple parts, each tweet can focus on a different part)
5. Each tweet should focus on a SINGLE topic or closely related set of topics - do not try to cover unrelated topics in one tweet
6. You MUST include proper citations in the format [ID] immediately after EACH sentence that uses information from a specific paper. The citation should be just the arXiv ID in [], so just [number]. No [ID: [number]] or anything else. Just the ID inside the [].
7. You CAN mix information from multiple chunks in a single tweet, but ONLY if they are related and it makes sense to combine them
8. The citation format MUST be, for example: sentence 1 [ID1]. sentence 2 [ID2]. sentence 3 [ID1]. sentence 4 [ID3]. In this example, the sentences would either be related by talking about the same topic but in different ways and from different sources or they would be complementary by answering to the same part of the query in different ways. Again, this is just an example.
9. When mixing chunks from different papers, the sentences must either:
   a. Discuss the same topic from different perspectives/sources, OR
   b. Provide complementary information that answers the same part of the query in different ways

These tweets are meant to give the repository owner specific paper recommendations that directly relate to their work. Make them genuinely useful and relevant.
"""

# User prompt template for tweet generation
USER_PROMPT = """
I need exactly {num_tweets} tweet-style recommendations about scientific papers that are relevant to a GitHub repository.

REPOSITORY QUERY:
{query}

PAPER CHUNKS:
{chunks}

Your task is to create exactly {num_tweets} tweet-style recommendations that follow these STRICT requirements:

1. Use ONLY information from the provided chunks - do not add any information not present in the chunks
2. Make each tweet directly relevant to the specific repository query
3. The {num_tweets} tweets together must cover ALL aspects of the query
4. Each tweet must focus on a SINGLE topic or closely related set of topics
5. Include proper citations in format [ID] IMMEDIATELY AFTER each sentence that uses information from that paper
6. The exact citation format must be: sentence 1 [ID1]. sentence 2 [ID2]. sentence 3 [ID1]. sentence 4 [ID3].
7. You can mix information from multiple chunks in a single tweet, but ONLY if they are related to the same topic
8. When mixing chunks from different papers, the sentences must either:
   - Discuss the same topic from different perspectives/sources, OR
   - Provide complementary information that answers the same part of the query in different ways
9. Make the tweets genuinely useful for a developer working on this repository

Format each tweet as "Tweet 1:", "Tweet 2:", etc.

Remember: These tweets must be based SOLELY on the provided chunks and must be directly relevant to the repository query.
"""
