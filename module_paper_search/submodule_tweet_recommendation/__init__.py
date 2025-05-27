"""
submodule_tweet_recommendation/__init__.py

This module implements Tweet Recommendation generation for GitHub repositories.
It exposes the main function to run the tweet recommendation process.
"""

from .tweet_recommendation import run_tweet_recommendation

__all__ = ['run_tweet_recommendation']
