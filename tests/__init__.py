"""
테스트 설정 및 유틸리티
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager
from src.exceptions import MusicRecommenderError, ValidationError
from src.validators import DataValidator, DataSanitizer
from src.logger import get_logger


class TestBase(unittest.TestCase):
    """테스트 기본 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # 테스트용 환경 변수 설정
        os.environ['SPOTIFY_CLIENT_ID'] = 'test_client_id'
        os.environ['SPOTIFY_CLIENT_SECRET'] = 'test_client_secret'
        os.environ['CHROMA_PERSIST_DIRECTORY'] = str(self.test_data_dir / "chroma_db")
        os.environ['MODEL_SAVE_DIR'] = str(self.test_data_dir / "models")
        
        # 로거 설정 (테스트 중에는 로그 출력 최소화)
        self.logger = get_logger("test")
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_track_data(self, count: int = 5) -> pd.DataFrame:
        """테스트용 트랙 데이터 생성"""
        data = []
        for i in range(count):
            track = {
                'id': f'test_track_{i}',
                'name': f'Test Track {i}',
                'artists': [f'Test Artist {i}'],
                'album': f'Test Album {i}',
                'release_date': '2023-01-01',
                'popularity': 50 + i,
                'duration_ms': 180000 + i * 1000,
                'explicit': False,
                'external_urls': f'https://open.spotify.com/track/test_track_{i}',
                'preview_url': f'https://p.scdn.co/mp3-preview/test_track_{i}',
                'danceability': 0.5 + i * 0.1,
                'energy': 0.6 + i * 0.05,
                'valence': 0.7 + i * 0.03,
                'tempo': 120.0 + i * 5,
                'instrumentalness': 0.1 + i * 0.02,
                'acousticness': 0.2 + i * 0.01,
                'liveness': 0.3 + i * 0.01,
                'speechiness': 0.05 + i * 0.01,
                'key': i % 12,
                'mode': i % 2,
                'time_signature': 4
            }
            data.append(track)
        
        return pd.DataFrame(data)
    
    def create_test_user_preferences(self) -> dict:
        """테스트용 사용자 선호도 생성"""
        return {
            'user1': {
                'test_track_0': 5.0,
                'test_track_1': 4.0,
                'test_track_2': 3.0
            },
            'user2': {
                'test_track_1': 4.5,
                'test_track_2': 3.5,
                'test_track_3': 4.0
            }
        }


class MockSpotifyAPI:
    """Spotify API 모킹 클래스"""
    
    def __init__(self):
        self.search_results = {
            'tracks': {
                'items': [
                    {
                        'id': 'mock_track_1',
                        'name': 'Mock Track 1',
                        'artists': [{'name': 'Mock Artist 1'}],
                        'album': {'name': 'Mock Album 1', 'release_date': '2023-01-01'},
                        'popularity': 80,
                        'duration_ms': 180000,
                        'explicit': False,
                        'external_urls': {'spotify': 'https://open.spotify.com/track/mock_track_1'},
                        'preview_url': 'https://p.scdn.co/mp3-preview/mock_track_1'
                    }
                ]
            }
        }
        
        self.audio_features = [
            {
                'id': 'mock_track_1',
                'danceability': 0.8,
                'energy': 0.7,
                'valence': 0.6,
                'tempo': 120.0,
                'instrumentalness': 0.1,
                'acousticness': 0.2,
                'liveness': 0.3,
                'speechiness': 0.05,
                'key': 0,
                'mode': 1,
                'time_signature': 4
            }
        ]
    
    def search(self, q, type, limit):
        return self.search_results
    
    def audio_features(self, track_ids):
        return self.audio_features
    
    def playlist_tracks(self, playlist_id):
        return {'items': [{'track': self.search_results['tracks']['items'][0]}]}
    
    def artist_top_tracks(self, artist_id, country='US'):
        return {'tracks': self.search_results['tracks']['items']}
    
    def audio_analysis(self, track_id):
        return {'track': {'analysis': 'mock_analysis'}}


def run_tests():
    """모든 테스트 실행"""
    # 테스트 디스커버리
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

