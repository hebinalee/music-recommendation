"""
에러 처리 모듈
"""

from typing import Optional, Dict, Any
from enum import Enum
import logging


class ErrorCode(Enum):
    """에러 코드 열거형"""
    SPOTIFY_API_ERROR = "SPOTIFY_API_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    FILE_ERROR = "FILE_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class MusicRecommenderError(Exception):
    """음악 추천 시스템 기본 예외 클래스"""
    
    def __init__(self, 
                 message: str, 
                 error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
                 details: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
    
    def __str__(self):
        return f"[{self.error_code.value}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """에러 정보를 딕셔너리로 변환"""
        return {
            'error_code': self.error_code.value,
            'message': self.message,
            'details': self.details,
            'original_exception': str(self.original_exception) if self.original_exception else None
        }


class SpotifyAPIError(MusicRecommenderError):
    """Spotify API 관련 에러"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCode.SPOTIFY_API_ERROR, details, original_exception)


class DatabaseError(MusicRecommenderError):
    """데이터베이스 관련 에러"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCode.DATABASE_ERROR, details, original_exception)


class ModelError(MusicRecommenderError):
    """모델 관련 에러"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCode.MODEL_ERROR, details, original_exception)


class ValidationError(MusicRecommenderError):
    """데이터 검증 관련 에러"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCode.VALIDATION_ERROR, details, original_exception)


class ConfigError(MusicRecommenderError):
    """설정 관련 에러"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCode.CONFIG_ERROR, details, original_exception)


def handle_exception(func):
    """예외 처리를 위한 데코레이터"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MusicRecommenderError:
            # 이미 처리된 에러는 그대로 전파
            raise
        except Exception as e:
            # 예상치 못한 에러는 래핑
            logging.error(f"예상치 못한 에러 발생: {str(e)}", exc_info=True)
            raise MusicRecommenderError(
                f"예상치 못한 에러가 발생했습니다: {str(e)}",
                ErrorCode.UNKNOWN_ERROR,
                {'function': func.__name__},
                e
            )
    return wrapper


def create_user_friendly_message(error: MusicRecommenderError) -> str:
    """사용자 친화적 에러 메시지 생성"""
    messages = {
        ErrorCode.SPOTIFY_API_ERROR: "음악 정보를 가져오는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
        ErrorCode.DATABASE_ERROR: "데이터베이스에 문제가 발생했습니다. 관리자에게 문의해주세요.",
        ErrorCode.MODEL_ERROR: "추천 모델에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
        ErrorCode.VALIDATION_ERROR: "입력한 정보에 문제가 있습니다. 다시 확인해주세요.",
        ErrorCode.CONFIG_ERROR: "시스템 설정에 문제가 있습니다. 관리자에게 문의해주세요.",
        ErrorCode.NETWORK_ERROR: "네트워크 연결에 문제가 있습니다. 인터넷 연결을 확인해주세요.",
        ErrorCode.FILE_ERROR: "파일 처리 중 문제가 발생했습니다. 파일을 확인해주세요.",
        ErrorCode.UNKNOWN_ERROR: "예상치 못한 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    }
    
    return messages.get(error.error_code, messages[ErrorCode.UNKNOWN_ERROR])

