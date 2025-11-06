"""
Spotify 수집기 테스트
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from tests import TestBase, MockSpotifyAPI
from src.spotify_collector import SpotifyMusicCollector
from src.exceptions import SpotifyAPIError, ValidationError


class TestSpotifyMusicCollector(TestBase):
    """SpotifyMusicCollector 테스트"""
    
    def setUp(self):
        super().setUp()
        self.mock_spotify = MockSpotifyAPI()
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_initialization_success(self, mock_spotify_class):
        """초기화 성공 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        
        self.assertIsNotNone(collector.client_id)
        self.assertIsNotNone(collector.client_secret)
        self.assertIsNotNone(collector.sp)
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_initialization_missing_credentials(self, mock_spotify_class):
        """인증 정보 누락 초기화 테스트"""
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(SpotifyAPIError):
                SpotifyMusicCollector()
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_search_tracks_success(self, mock_spotify_class):
        """트랙 검색 성공 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        results = collector.search_tracks("jazz", limit=10)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['name'], 'Mock Track 1')
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_search_tracks_invalid_query(self, mock_spotify_class):
        """잘못된 검색 쿼리 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        
        with self.assertRaises(ValidationError):
            collector.search_tracks("", limit=10)
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_search_tracks_invalid_limit(self, mock_spotify_class):
        """잘못된 제한값 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        
        with self.assertRaises(ValidationError):
            collector.search_tracks("jazz", limit=0)
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_get_track_features_success(self, mock_spotify_class):
        """트랙 특성 조회 성공 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        track_ids = ['mock_track_1']
        features = collector.get_track_features(track_ids)
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0]['id'], 'mock_track_1')
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_get_track_features_empty_list(self, mock_spotify_class):
        """빈 트랙 ID 목록 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        features = collector.get_track_features([])
        
        self.assertEqual(features, [])
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_get_playlist_tracks_success(self, mock_spotify_class):
        """플레이리스트 트랙 조회 성공 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        playlist_id = '37i9dQZF1DXcBWIGoYBM5M'
        tracks = collector.get_playlist_tracks(playlist_id)
        
        self.assertIsInstance(tracks, list)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['name'], 'Mock Track 1')
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_get_playlist_tracks_invalid_id(self, mock_spotify_class):
        """잘못된 플레이리스트 ID 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        
        with self.assertRaises(ValidationError):
            collector.get_playlist_tracks("invalid_id")
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_get_artist_top_tracks_success(self, mock_spotify_class):
        """아티스트 인기 트랙 조회 성공 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        artist_id = '0TnOYISbd1XYRBk9myaseg'
        tracks = collector.get_artist_top_tracks(artist_id)
        
        self.assertIsInstance(tracks, list)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['name'], 'Mock Track 1')
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_get_artist_top_tracks_invalid_id(self, mock_spotify_class):
        """잘못된 아티스트 ID 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        
        with self.assertRaises(ValidationError):
            collector.get_artist_top_tracks("invalid_id")
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_collect_music_data_search_queries(self, mock_spotify_class):
        """검색어 기반 음악 데이터 수집 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        search_queries = ['jazz', 'rock']
        df = collector.collect_music_data(search_queries=search_queries)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('id', df.columns)
        self.assertIn('name', df.columns)
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_collect_music_data_playlist_ids(self, mock_spotify_class):
        """플레이리스트 기반 음악 데이터 수집 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        playlist_ids = ['37i9dQZF1DXcBWIGoYBM5M']
        df = collector.collect_music_data(playlist_ids=playlist_ids)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_collect_music_data_artist_ids(self, mock_spotify_class):
        """아티스트 기반 음악 데이터 수집 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        artist_ids = ['0TnOYISbd1XYRBk9myaseg']
        df = collector.collect_music_data(artist_ids=artist_ids)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_collect_music_data_no_sources(self, mock_spotify_class):
        """수집 소스 없음 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        df = collector.collect_music_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_save_to_csv_success(self, mock_spotify_class):
        """CSV 저장 성공 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        df = self.create_test_track_data(3)
        
        filename = collector.save_to_csv(df, 'test_output.csv')
        
        self.assertEqual(filename, 'test_output.csv')
        
        # 파일이 생성되었는지 확인
        import os
        self.assertTrue(os.path.exists('test_output.csv'))
        
        # 테스트 후 파일 삭제
        os.remove('test_output.csv')
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_save_to_csv_empty_dataframe(self, mock_spotify_class):
        """빈 DataFrame CSV 저장 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        df = pd.DataFrame()
        
        filename = collector.save_to_csv(df, 'test_empty.csv')
        
        self.assertEqual(filename, 'test_empty.csv')
    
    @patch('src.spotify_collector.spotipy.Spotify')
    def test_save_to_csv_auto_extension(self, mock_spotify_class):
        """자동 확장자 추가 테스트"""
        mock_spotify_class.return_value = self.mock_spotify
        
        collector = SpotifyMusicCollector()
        df = self.create_test_track_data(2)
        
        filename = collector.save_to_csv(df, 'test_output')
        
        self.assertEqual(filename, 'test_output.csv')
        
        # 파일이 생성되었는지 확인
        import os
        self.assertTrue(os.path.exists('test_output.csv'))
        
        # 테스트 후 파일 삭제
        os.remove('test_output.csv')


if __name__ == '__main__':
    unittest.main()

