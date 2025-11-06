"""
데이터 검증 테스트
"""

import unittest
import pandas as pd
from tests import TestBase
from src.validators import DataValidator, DataSanitizer
from src.exceptions import ValidationError


class TestDataValidator(TestBase):
    """DataValidator 테스트"""
    
    def test_validate_track_data_success(self):
        """트랙 데이터 검증 성공 테스트"""
        track_data = {
            'id': 'test_track_1234567890',
            'name': 'Test Track',
            'artists': ['Test Artist'],
            'album': 'Test Album'
        }
        
        result = DataValidator.validate_track_data(track_data)
        self.assertTrue(result)
    
    def test_validate_track_data_missing_field(self):
        """트랙 데이터 필수 필드 누락 테스트"""
        track_data = {
            'id': 'test_track_1234567890',
            'name': 'Test Track',
            # 'artists' 필드 누락
            'album': 'Test Album'
        }
        
        with self.assertRaises(ValidationError):
            DataValidator.validate_track_data(track_data)
    
    def test_validate_track_data_invalid_id(self):
        """트랙 데이터 잘못된 ID 테스트"""
        track_data = {
            'id': 'short',  # 너무 짧은 ID
            'name': 'Test Track',
            'artists': ['Test Artist'],
            'album': 'Test Album'
        }
        
        with self.assertRaises(ValidationError):
            DataValidator.validate_track_data(track_data)
    
    def test_validate_audio_features_success(self):
        """오디오 특성 검증 성공 테스트"""
        features = {
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
        
        result = DataValidator.validate_audio_features(features)
        self.assertTrue(result)
    
    def test_validate_audio_features_out_of_range(self):
        """오디오 특성 범위 초과 테스트"""
        features = {
            'danceability': 1.5,  # 범위 초과
            'energy': 0.7
        }
        
        with self.assertRaises(ValidationError):
            DataValidator.validate_audio_features(features)
    
    def test_validate_user_rating_success(self):
        """사용자 평점 검증 성공 테스트"""
        valid_ratings = [1.0, 2.5, 3.0, 4.5, 5.0]
        
        for rating in valid_ratings:
            result = DataValidator.validate_user_rating(rating)
            self.assertTrue(result)
    
    def test_validate_user_rating_invalid(self):
        """사용자 평점 검증 실패 테스트"""
        invalid_ratings = [0.5, 5.5, -1.0, 'invalid', None]
        
        for rating in invalid_ratings:
            with self.assertRaises(ValidationError):
                DataValidator.validate_user_rating(rating)
    
    def test_validate_user_id_success(self):
        """사용자 ID 검증 성공 테스트"""
        valid_ids = ['user123', 'test_user', 'user_123']
        
        for user_id in valid_ids:
            result = DataValidator.validate_user_id(user_id)
            self.assertTrue(result)
    
    def test_validate_user_id_invalid(self):
        """사용자 ID 검증 실패 테스트"""
        invalid_ids = ['ab', 'a' * 51, '', None, 123]
        
        for user_id in invalid_ids:
            with self.assertRaises(ValidationError):
                DataValidator.validate_user_id(user_id)
    
    def test_validate_search_query_success(self):
        """검색 쿼리 검증 성공 테스트"""
        valid_queries = ['jazz', 'rock music', 'k-pop']
        
        for query in valid_queries:
            result = DataValidator.validate_search_query(query)
            self.assertTrue(result)
    
    def test_validate_search_query_invalid(self):
        """검색 쿼리 검증 실패 테스트"""
        invalid_queries = ['', 'a' * 101, None, 123]
        
        for query in invalid_queries:
            with self.assertRaises(ValidationError):
                DataValidator.validate_search_query(query)
    
    def test_validate_dataframe_success(self):
        """DataFrame 검증 성공 테스트"""
        df = self.create_test_track_data(3)
        required_columns = ['id', 'name', 'artists', 'album']
        
        result = DataValidator.validate_dataframe(df, required_columns)
        self.assertTrue(result)
    
    def test_validate_dataframe_empty(self):
        """빈 DataFrame 검증 테스트"""
        df = pd.DataFrame()
        required_columns = ['id', 'name']
        
        with self.assertRaises(ValidationError):
            DataValidator.validate_dataframe(df, required_columns)
    
    def test_validate_dataframe_missing_columns(self):
        """필수 컬럼 누락 DataFrame 검증 테스트"""
        df = pd.DataFrame({'id': ['1', '2'], 'name': ['A', 'B']})
        required_columns = ['id', 'name', 'artists']  # 'artists' 누락
        
        with self.assertRaises(ValidationError):
            DataValidator.validate_dataframe(df, required_columns)
    
    def test_validate_n_results_success(self):
        """결과 개수 검증 성공 테스트"""
        valid_counts = [1, 10, 100, 1000]
        
        for count in valid_counts:
            result = DataValidator.validate_n_results(count)
            self.assertTrue(result)
    
    def test_validate_n_results_invalid(self):
        """결과 개수 검증 실패 테스트"""
        invalid_counts = [0, -1, 1001, 'invalid']
        
        for count in invalid_counts:
            with self.assertRaises(ValidationError):
                DataValidator.validate_n_results(count)
    
    def test_validate_playlist_id_success(self):
        """플레이리스트 ID 검증 성공 테스트"""
        valid_ids = ['37i9dQZF1DXcBWIGoYBM5M', '37i9dQZF1DX0XUsuxWHRQd']
        
        for playlist_id in valid_ids:
            result = DataValidator.validate_playlist_id(playlist_id)
            self.assertTrue(result)
    
    def test_validate_playlist_id_invalid(self):
        """플레이리스트 ID 검증 실패 테스트"""
        invalid_ids = ['short', 'too_long_id_123456789012345678901', '', None]
        
        for playlist_id in invalid_ids:
            with self.assertRaises(ValidationError):
                DataValidator.validate_playlist_id(playlist_id)
    
    def test_validate_artist_id_success(self):
        """아티스트 ID 검증 성공 테스트"""
        valid_ids = ['0TnOYISbd1XYRBk9myaseg', '1dfeR4HaWDbWqFHLkxsg1d']
        
        for artist_id in valid_ids:
            result = DataValidator.validate_artist_id(artist_id)
            self.assertTrue(result)
    
    def test_validate_artist_id_invalid(self):
        """아티스트 ID 검증 실패 테스트"""
        invalid_ids = ['short', 'too_long_id_123456789012345678901', '', None]
        
        for artist_id in invalid_ids:
            with self.assertRaises(ValidationError):
                DataValidator.validate_artist_id(artist_id)


class TestDataSanitizer(TestBase):
    """DataSanitizer 테스트"""
    
    def test_sanitize_track_data(self):
        """트랙 데이터 정제 테스트"""
        track_data = {
            'id': 'test_track_1234567890',
            'name': '  Test Track  ',
            'artists': ['  Test Artist  ', ''],
            'album': 'Test Album',
            'popularity': '80.5',
            'duration_ms': '180000.0'
        }
        
        sanitized = DataSanitizer.sanitize_track_data(track_data)
        
        self.assertEqual(sanitized['name'], 'Test Track')
        self.assertEqual(sanitized['artists'], ['Test Artist'])
        self.assertEqual(sanitized['popularity'], 80)
        self.assertEqual(sanitized['duration_ms'], 180000)
    
    def test_sanitize_audio_features(self):
        """오디오 특성 정제 테스트"""
        features = {
            'danceability': '0.8',
            'energy': 'invalid',
            'valence': 0.6
        }
        
        sanitized = DataSanitizer.sanitize_audio_features(features)
        
        self.assertEqual(sanitized['danceability'], 0.8)
        self.assertEqual(sanitized['energy'], 0.5)  # 기본값
        self.assertEqual(sanitized['valence'], 0.6)
    
    def test_sanitize_search_query(self):
        """검색 쿼리 정제 테스트"""
        queries = [
            '  jazz music  ',
            'rock; DROP TABLE users;',
            'k-pop -- comment',
            'electronic /* comment */'
        ]
        
        expected = [
            'jazz music',
            'rock DROP TABLE users',
            'k-pop comment',
            'electronic comment'
        ]
        
        for query, expected_result in zip(queries, expected):
            result = DataSanitizer.sanitize_search_query(query)
            self.assertEqual(result, expected_result)
    
    def test_sanitize_search_query_non_string(self):
        """비문자열 검색 쿼리 정제 테스트"""
        result = DataSanitizer.sanitize_search_query(123)
        self.assertEqual(result, '')


if __name__ == '__main__':
    unittest.main()

