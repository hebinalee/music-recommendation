"""
에러 처리 테스트
"""

import unittest
from tests import TestBase
from src.exceptions import (
    MusicRecommenderError, SpotifyAPIError, DatabaseError, ModelError,
    ValidationError, ConfigError, ErrorCode, handle_exception, create_user_friendly_message
)


class TestExceptions(TestBase):
    """예외 클래스 테스트"""
    
    def test_music_recommender_error_basic(self):
        """기본 MusicRecommenderError 테스트"""
        error = MusicRecommenderError("Test error message")
        
        self.assertEqual(error.message, "Test error message")
        self.assertEqual(error.error_code, ErrorCode.UNKNOWN_ERROR)
        self.assertEqual(str(error), "[UNKNOWN_ERROR] Test error message")
    
    def test_music_recommender_error_with_details(self):
        """상세 정보가 있는 MusicRecommenderError 테스트"""
        details = {'user_id': 'test_user', 'track_id': 'test_track'}
        original_exception = ValueError("Original error")
        
        error = MusicRecommenderError(
            "Test error message",
            ErrorCode.SPOTIFY_API_ERROR,
            details,
            original_exception
        )
        
        self.assertEqual(error.error_code, ErrorCode.SPOTIFY_API_ERROR)
        self.assertEqual(error.details, details)
        self.assertEqual(error.original_exception, original_exception)
    
    def test_music_recommender_error_to_dict(self):
        """MusicRecommenderError 딕셔너리 변환 테스트"""
        error = MusicRecommenderError(
            "Test error message",
            ErrorCode.DATABASE_ERROR,
            {'key': 'value'},
            ValueError("Original")
        )
        
        error_dict = error.to_dict()
        
        self.assertEqual(error_dict['error_code'], 'DATABASE_ERROR')
        self.assertEqual(error_dict['message'], 'Test error message')
        self.assertEqual(error_dict['details'], {'key': 'value'})
        self.assertEqual(error_dict['original_exception'], 'Original error')
    
    def test_spotify_api_error(self):
        """SpotifyAPIError 테스트"""
        error = SpotifyAPIError("Spotify API error")
        
        self.assertEqual(error.error_code, ErrorCode.SPOTIFY_API_ERROR)
        self.assertEqual(error.message, "Spotify API error")
    
    def test_database_error(self):
        """DatabaseError 테스트"""
        error = DatabaseError("Database error")
        
        self.assertEqual(error.error_code, ErrorCode.DATABASE_ERROR)
        self.assertEqual(error.message, "Database error")
    
    def test_model_error(self):
        """ModelError 테스트"""
        error = ModelError("Model error")
        
        self.assertEqual(error.error_code, ErrorCode.MODEL_ERROR)
        self.assertEqual(error.message, "Model error")
    
    def test_validation_error(self):
        """ValidationError 테스트"""
        error = ValidationError("Validation error")
        
        self.assertEqual(error.error_code, ErrorCode.VALIDATION_ERROR)
        self.assertEqual(error.message, "Validation error")
    
    def test_config_error(self):
        """ConfigError 테스트"""
        error = ConfigError("Config error")
        
        self.assertEqual(error.error_code, ErrorCode.CONFIG_ERROR)
        self.assertEqual(error.message, "Config error")


class TestErrorHandling(TestBase):
    """에러 처리 데코레이터 테스트"""
    
    def test_handle_exception_success(self):
        """정상 실행 테스트"""
        @handle_exception
        def test_function():
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")
    
    def test_handle_exception_music_recommender_error(self):
        """MusicRecommenderError 전파 테스트"""
        @handle_exception
        def test_function():
            raise MusicRecommenderError("Test error")
        
        with self.assertRaises(MusicRecommenderError):
            test_function()
    
    def test_handle_exception_generic_error(self):
        """일반 예외 래핑 테스트"""
        @handle_exception
        def test_function():
            raise ValueError("Original error")
        
        with self.assertRaises(MusicRecommenderError) as context:
            test_function()
        
        self.assertEqual(context.exception.error_code, ErrorCode.UNKNOWN_ERROR)
        self.assertIn("예상치 못한 에러가 발생했습니다", context.exception.message)
        self.assertIsInstance(context.exception.original_exception, ValueError)


class TestUserFriendlyMessages(TestBase):
    """사용자 친화적 메시지 테스트"""
    
    def test_create_user_friendly_message_spotify(self):
        """Spotify API 에러 메시지 테스트"""
        error = SpotifyAPIError("API error")
        message = create_user_friendly_message(error)
        
        self.assertIn("음악 정보를 가져오는 중", message)
    
    def test_create_user_friendly_message_database(self):
        """데이터베이스 에러 메시지 테스트"""
        error = DatabaseError("Database error")
        message = create_user_friendly_message(error)
        
        self.assertIn("데이터베이스에 문제가", message)
    
    def test_create_user_friendly_message_model(self):
        """모델 에러 메시지 테스트"""
        error = ModelError("Model error")
        message = create_user_friendly_message(error)
        
        self.assertIn("추천 모델에 문제가", message)
    
    def test_create_user_friendly_message_validation(self):
        """검증 에러 메시지 테스트"""
        error = ValidationError("Validation error")
        message = create_user_friendly_message(error)
        
        self.assertIn("입력한 정보에 문제가", message)
    
    def test_create_user_friendly_message_config(self):
        """설정 에러 메시지 테스트"""
        error = ConfigError("Config error")
        message = create_user_friendly_message(error)
        
        self.assertIn("시스템 설정에 문제가", message)
    
    def test_create_user_friendly_message_unknown(self):
        """알 수 없는 에러 메시지 테스트"""
        error = MusicRecommenderError("Unknown error", ErrorCode.UNKNOWN_ERROR)
        message = create_user_friendly_message(error)
        
        self.assertIn("예상치 못한 문제가", message)


if __name__ == '__main__':
    unittest.main()

