"""
로깅 시스템 모듈
"""

import logging
import os
from datetime import datetime
from typing import Optional
from pathlib import Path


class MusicRecommenderLogger:
    """음악 추천 시스템용 로거 클래스"""
    
    def __init__(self, name: str = "music_recommender", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 중복 핸들러 방지
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """로거 핸들러 설정"""
        # 로그 디렉토리 생성
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 파일 핸들러 (상세 로그)
        log_file = log_dir / f"music_recommender_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러 (간단한 로그)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터 설정
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, **kwargs):
        """디버그 로그"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """정보 로그"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """경고 로그"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """에러 로그"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}", exc_info=True, extra=kwargs)
        else:
            self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """치명적 에러 로그"""
        if exception:
            self.logger.critical(f"{message}: {str(exception)}", exc_info=True, extra=kwargs)
        else:
            self.logger.critical(message, extra=kwargs)


# 전역 로거 인스턴스
logger = MusicRecommenderLogger()


def get_logger(name: Optional[str] = None) -> MusicRecommenderLogger:
    """로거 인스턴스 반환"""
    if name:
        return MusicRecommenderLogger(name)
    return logger

