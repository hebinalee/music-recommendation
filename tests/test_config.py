"""
설정 관리 테스트
"""

import os
import unittest
from unittest.mock import patch
from tests import TestBase
from src.config import ConfigManager, SpotifyConfig, DatabaseConfig, ModelConfig, LoggingConfig


class TestConfigManager(TestBase):
    """ConfigManager 테스트"""
    
    def test_spotify_config_creation(self):
        """Spotify 설정 생성 테스트"""
        config = SpotifyConfig(
            client_id='test_id',
            client_secret='test_secret',
            redirect_uri='http://localhost:8080'
        )
        
        self.assertEqual(config.client_id, 'test_id')
        self.assertEqual(config.client_secret, 'test_secret')
        self.assertEqual(config.redirect_uri, 'http://localhost:8080')
    
    def test_database_config_defaults(self):
        """데이터베이스 설정 기본값 테스트"""
        config = DatabaseConfig()
        
        self.assertEqual(config.persist_directory, './chroma_db')
        self.assertEqual(config.collection_name, 'music_collection')
    
    def test_model_config_defaults(self):
        """모델 설정 기본값 테스트"""
        config = ModelConfig()
        
        self.assertEqual(config.save_dir, './models')
        self.assertEqual(config.embedding_dim, 128)
        self.assertEqual(config.hidden_dim, 256)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epochs, 100)
    
    def test_logging_config_defaults(self):
        """로깅 설정 기본값 테스트"""
        config = LoggingConfig()
        
        self.assertEqual(config.level, 'INFO')
        self.assertEqual(config.log_dir, './logs')
    
    def test_config_manager_initialization(self):
        """ConfigManager 초기화 테스트"""
        config_manager = ConfigManager()
        
        # 환경 변수가 설정되어 있는지 확인
        spotify_config = config_manager.get_spotify_config()
        self.assertEqual(spotify_config.client_id, 'test_client_id')
        self.assertEqual(spotify_config.client_secret, 'test_client_secret')
    
    def test_config_manager_missing_credentials(self):
        """인증 정보 누락 테스트"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                ConfigManager()
    
    def test_get_all_config(self):
        """모든 설정 반환 테스트"""
        config_manager = ConfigManager()
        all_config = config_manager.get_all_config()
        
        self.assertIn('spotify', all_config)
        self.assertIn('database', all_config)
        self.assertIn('model', all_config)
        self.assertIn('logging', all_config)
        
        self.assertIsInstance(all_config['spotify'], SpotifyConfig)
        self.assertIsInstance(all_config['database'], DatabaseConfig)
        self.assertIsInstance(all_config['model'], ModelConfig)
        self.assertIsInstance(all_config['logging'], LoggingConfig)


if __name__ == '__main__':
    unittest.main()

