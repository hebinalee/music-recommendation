"""
Spotify 음악 추천 시스템 소스 코드 패키지
"""

__version__ = "1.0.0"
__author__ = "Music Recommender Team"

# 주요 클래스들을 패키지 레벨에서 import 가능하게 함
from .music_recommender import MusicRecommender
from .vector_database import MusicVectorDatabase
from .spotify_collector import SpotifyMusicCollector
from .ann_index import ANNIndex

__all__ = [
    'MusicRecommender',
    'MusicVectorDatabase', 
    'SpotifyMusicCollector',
    'ANNIndex'
]
