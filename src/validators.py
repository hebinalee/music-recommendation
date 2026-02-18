"""
데이터 검증 모듈
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from .exceptions import ValidationError


class DataValidator:
    """데이터 검증 클래스"""
    
    @staticmethod
    def validate_track_data(track_data: Dict[str, Any]) -> bool:
        """트랙 데이터 검증"""
        required_fields = ['id', 'name', 'artists', 'album']
        
        for field in required_fields:
            if field not in track_data or not track_data[field]:
                raise ValidationError(f"필수 필드 '{field}'가 누락되었습니다.")
        
        # ID 형식 검증
        if not isinstance(track_data['id'], str) or len(track_data['id']) < 10:
            raise ValidationError("트랙 ID 형식이 올바르지 않습니다.")
        
        # 아티스트 리스트 검증
        if not isinstance(track_data['artists'], list) or not track_data['artists']:
            raise ValidationError("아티스트 정보가 올바르지 않습니다.")
        
        return True
    
    @staticmethod
    def validate_audio_features(features: Dict[str, Any]) -> bool:
        """오디오 특성 데이터 검증"""
        valid_ranges = {
            'danceability': (0.0, 1.0),
            'energy': (0.0, 1.0),
            'valence': (0.0, 1.0),
            'tempo': (0.0, 300.0),
            'instrumentalness': (0.0, 1.0),
            'acousticness': (0.0, 1.0),
            'liveness': (0.0, 1.0),
            'speechiness': (0.0, 1.0),
            'key': (0, 11),
            'mode': (0, 1),
            'time_signature': (3, 7)
        }
        
        for feature, value in features.items():
            if feature in valid_ranges:
                min_val, max_val = valid_ranges[feature]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    raise ValidationError(f"'{feature}' 값이 유효 범위({min_val}-{max_val})를 벗어났습니다.")
        
        return True
    
    @staticmethod
    def validate_user_rating(rating: Union[int, float]) -> bool:
        """사용자 평점 검증"""
        if not isinstance(rating, (int, float)):
            raise ValidationError("평점은 숫자여야 합니다.")
        
        if not (1.0 <= rating <= 5.0):
            raise ValidationError("평점은 1.0-5.0 범위여야 합니다.")
        
        return True
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """사용자 ID 검증"""
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValidationError("사용자 ID는 비어있지 않은 문자열이어야 합니다.")
        
        if len(user_id) < 3 or len(user_id) > 50:
            raise ValidationError("사용자 ID는 3-50자 사이여야 합니다.")
        
        return True
    
    @staticmethod
    def validate_search_query(query: str) -> bool:
        """검색 쿼리 검증"""
        if not isinstance(query, str) or not query.strip():
            raise ValidationError("검색어는 비어있지 않은 문자열이어야 합니다.")
        
        if len(query) > 100:
            raise ValidationError("검색어는 100자를 초과할 수 없습니다.")
        
        return True
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """DataFrame 검증"""
        if df.empty:
            raise ValidationError("데이터프레임이 비어있습니다.")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValidationError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
        
        # 중복 ID 검증
        if 'id' in df.columns and df['id'].duplicated().any():
            raise ValidationError("중복된 ID가 있습니다.")
        
        return True
    
    @staticmethod
    def validate_n_results(n_results: int) -> bool:
        """결과 개수 검증"""
        if not isinstance(n_results, int) or n_results <= 0:
            raise ValidationError("결과 개수는 양의 정수여야 합니다.")
        
        if n_results > 1000:
            raise ValidationError("결과 개수는 1000을 초과할 수 없습니다.")
        
        return True
    
    @staticmethod
    def validate_playlist_id(playlist_id: str) -> bool:
        """플레이리스트 ID 검증"""
        if not isinstance(playlist_id, str) or not playlist_id.strip():
            raise ValidationError("플레이리스트 ID는 비어있지 않은 문자열이어야 합니다.")
        
        # Spotify 플레이리스트 ID 형식 검증 (22자리 영숫자)
        if len(playlist_id) != 22 or not playlist_id.isalnum():
            raise ValidationError("플레이리스트 ID 형식이 올바르지 않습니다.")
        
        return True
    
    @staticmethod
    def validate_artist_id(artist_id: str) -> bool:
        """아티스트 ID 검증"""
        if not isinstance(artist_id, str) or not artist_id.strip():
            raise ValidationError("아티스트 ID는 비어있지 않은 문자열이어야 합니다.")
        
        # Spotify 아티스트 ID 형식 검증 (22자리 영숫자)
        if len(artist_id) != 22 or not artist_id.isalnum():
            raise ValidationError("아티스트 ID 형식이 올바르지 않습니다.")
        
        return True


class DataSanitizer:
    """데이터 정제 클래스"""
    
    @staticmethod
    def sanitize_track_data(track_data: Dict[str, Any]) -> Dict[str, Any]:
        """트랙 데이터 정제"""
        sanitized = track_data.copy()
        
        # 문자열 필드 정제
        string_fields = ['name', 'album']
        for field in string_fields:
            if field in sanitized:
                sanitized[field] = str(sanitized[field]).strip()
        
        # 아티스트 리스트 정제
        if 'artists' in sanitized and isinstance(sanitized['artists'], list):
            sanitized['artists'] = [str(artist).strip() for artist in sanitized['artists'] if artist]
        
        # 숫자 필드 정제
        numeric_fields = ['popularity', 'duration_ms']
        for field in numeric_fields:
            if field in sanitized:
                try:
                    sanitized[field] = int(float(sanitized[field]))
                except (ValueError, TypeError):
                    sanitized[field] = 0
        
        return sanitized
    
    @staticmethod
    def sanitize_audio_features(features: Dict[str, Any]) -> Dict[str, Any]:
        """오디오 특성 정제"""
        sanitized = {}
        
        for key, value in features.items():
            try:
                sanitized[key] = float(value)
            except (ValueError, TypeError):
                # 기본값 설정
                default_values = {
                    'danceability': 0.5,
                    'energy': 0.5,
                    'valence': 0.5,
                    'tempo': 120.0,
                    'instrumentalness': 0.0,
                    'acousticness': 0.0,
                    'liveness': 0.0,
                    'speechiness': 0.0,
                    'key': 0,
                    'mode': 0,
                    'time_signature': 4
                }
                sanitized[key] = default_values.get(key, 0.0)
        
        return sanitized
    
    @staticmethod
    def sanitize_search_query(query: str) -> str:
        """검색 쿼리 정제"""
        if not isinstance(query, str):
            return ""
        
        # 특수 문자 제거 및 공백 정리
        sanitized = query.strip()
        # SQL 인젝션 방지를 위한 기본적인 정제
        dangerous_chars = [';', '--', '/*', '*/', 'xp_', 'sp_']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized

