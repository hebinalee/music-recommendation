"""
설정 관리 모듈
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import logging


@dataclass
class SpotifyConfig:
    """Spotify API 설정"""
    client_id: str
    client_secret: str
    redirect_uri: str = "http://localhost:8888/callback"


@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    persist_directory: str = "./chroma_db"
    collection_name: str = "music_collection"


@dataclass
class ModelConfig:
    """모델 설정"""
    save_dir: str = "./models"
    embedding_dim: int = 128
    hidden_dim: int = 256
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100


@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    log_dir: str = "./logs"


class ConfigManager:
    """설정 관리자"""
    
    def __init__(self, env_file: Optional[str] = None):
        self.env_file = env_file or ".env"
        self._load_environment()
        self._validate_config()
    
    def _load_environment(self):
        """환경 변수 로드"""
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
        else:
            logging.warning(f"환경 파일 {self.env_file}을 찾을 수 없습니다.")
    
    def _validate_config(self):
        """설정 검증"""
        required_vars = ['SPOTIFY_CLIENT_ID', 'SPOTIFY_CLIENT_SECRET']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
    
    def get_spotify_config(self) -> SpotifyConfig:
        """Spotify 설정 반환"""
        return SpotifyConfig(
            client_id=os.getenv('SPOTIFY_CLIENT_ID'),
            client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
            redirect_uri=os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8888/callback')
        )
    
    def get_database_config(self) -> DatabaseConfig:
        """데이터베이스 설정 반환"""
        return DatabaseConfig(
            persist_directory=os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db'),
            collection_name=os.getenv('COLLECTION_NAME', 'music_collection')
        )
    
    def get_model_config(self) -> ModelConfig:
        """모델 설정 반환"""
        return ModelConfig(
            save_dir=os.getenv('MODEL_SAVE_DIR', './models'),
            embedding_dim=int(os.getenv('EMBEDDING_DIM', '128')),
            hidden_dim=int(os.getenv('HIDDEN_DIM', '256')),
            learning_rate=float(os.getenv('LEARNING_RATE', '0.001')),
            batch_size=int(os.getenv('BATCH_SIZE', '32')),
            epochs=int(os.getenv('EPOCHS', '100'))
        )
    
    def get_logging_config(self) -> LoggingConfig:
        """로깅 설정 반환"""
        return LoggingConfig(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            log_dir=os.getenv('LOG_DIR', './logs')
        )
    
    def get_all_config(self) -> Dict[str, Any]:
        """모든 설정 반환"""
        return {
            'spotify': self.get_spotify_config(),
            'database': self.get_database_config(),
            'model': self.get_model_config(),
            'logging': self.get_logging_config()
        }


# 전역 설정 관리자 인스턴스
config_manager = ConfigManager()

