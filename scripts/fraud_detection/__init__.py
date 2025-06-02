"""
NYC DOB Fraud Detection System

A Graphiti-powered pattern recognition system for detecting complex fraud schemes
in NYC construction data, with focus on corporate network analysis and temporal
relationship tracking.
"""

from .episode_design import EpisodeDesigner, NYCDOBEntityTypes

__version__ = "0.1.0"
__all__ = [
    "EpisodeDesigner",
    "NYCDOBEntityTypes"
]

# Future modules to be implemented:
# from .pattern_queries import PatternRecognitionQueries
# from .community_analysis import CommunityAnalyzer
# from .temporal_analysis import TemporalAnalyzer
# from .explorer import GraphExplorer
# from .alerts import AlertManager 