"""
Spotify 음악 추천 시스템 소스 코드 패키지
"""

__version__ = "1.0.0"
__author__ = "Music Recommender Team"

# 주요 클래스들을 패키지 레벨에서 import 가능하게 함
# 무거운 의존성(faiss, tensorflow, torch 등)이 없는 환경에서도
# 가벼운 모듈만 사용할 수 있도록 try/except로 감쌉니다.

from .spotify_collector import SpotifyMusicCollector

try:
    from .ann_index import ANNIndex
except ImportError:
    ANNIndex = None  # type: ignore

try:
    from .vector_database import MusicVectorDatabase
except ImportError:
    MusicVectorDatabase = None  # type: ignore

try:
    from .music_recommender import MusicRecommender
except ImportError:
    MusicRecommender = None  # type: ignore

__all__ = [
    'ANNIndex',
    'MusicRecommender',
    'MusicVectorDatabase',
    'SpotifyMusicCollector',
]
