import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

from .config import ConfigManager
from .exceptions import SpotifyAPIError, handle_exception, ValidationError
from .validators import DataValidator, DataSanitizer
from .logger import get_logger

load_dotenv()
logger = get_logger(__name__)

class SpotifyMusicCollector:
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Spotify API 클라이언트를 초기화합니다."""
        try:
            self.config_manager = config_manager or ConfigManager()
            spotify_config = self.config_manager.get_spotify_config()
            
            self.client_id = spotify_config.client_id
            self.client_secret = spotify_config.client_secret
            self.redirect_uri = spotify_config.redirect_uri
            
            if not self.client_id or not self.client_secret:
                raise SpotifyAPIError("Spotify API 인증 정보가 설정되지 않았습니다.")
            
            # Client Credentials Flow (public data access)
            self.sp = spotipy.Spotify(
                client_credentials_manager=SpotifyClientCredentials(
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
            )
            
            logger.info("Spotify API 클라이언트 초기화 완료")
            
        except Exception as e:
            logger.error("Spotify API 클라이언트 초기화 실패", exception=e)
            raise SpotifyAPIError("Spotify API 클라이언트 초기화에 실패했습니다.", original_exception=e)
    
    @handle_exception
    def search_tracks(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """검색어로 트랙을 검색합니다."""
        try:
            # 입력 검증
            DataValidator.validate_search_query(query)
            DataValidator.validate_n_results(limit)
            
            # 쿼리 정제
            sanitized_query = DataSanitizer.sanitize_search_query(query)
            
            logger.info(f"트랙 검색 시작: '{sanitized_query}', 제한: {limit}")
            
            results = self.sp.search(q=sanitized_query, type='track', limit=limit)
            tracks = []
            
            for track in results['tracks']['items']:
                track_info = {
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'release_date': track['album']['release_date'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms'],
                    'explicit': track['explicit'],
                    'external_urls': track['external_urls']['spotify'],
                    'preview_url': track['preview_url']
                }
                
                # 데이터 검증 및 정제
                DataValidator.validate_track_data(track_info)
                sanitized_track = DataSanitizer.sanitize_track_data(track_info)
                tracks.append(sanitized_track)
            
            logger.info(f"트랙 검색 완료: {len(tracks)}개 결과")
            return tracks
            
        except Exception as e:
            logger.error(f"트랙 검색 실패: '{query}'", exception=e)
            raise SpotifyAPIError(f"트랙 검색에 실패했습니다: {str(e)}", original_exception=e)
    
    @handle_exception
    def get_track_features(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """트랙의 오디오 특성을 가져옵니다."""
        try:
            if not track_ids:
                logger.warning("트랙 ID 목록이 비어있습니다.")
                return []
            
            logger.info(f"오디오 특성 조회 시작: {len(track_ids)}개 트랙")
            features = self.sp.audio_features(track_ids)
            
            # None 값 필터링 및 정제
            valid_features = []
            for feature in features:
                if feature:
                    sanitized_feature = DataSanitizer.sanitize_audio_features(feature)
                    DataValidator.validate_audio_features(sanitized_feature)
                    valid_features.append(sanitized_feature)
            
            logger.info(f"오디오 특성 조회 완료: {len(valid_features)}개 결과")
            return valid_features
            
        except Exception as e:
            logger.error(f"오디오 특성 조회 실패", exception=e)
            raise SpotifyAPIError(f"오디오 특성 조회에 실패했습니다: {str(e)}", original_exception=e)
    
    @handle_exception
    def get_track_analysis(self, track_id: str) -> Dict[str, Any]:
        """트랙의 상세 분석 정보를 가져옵니다."""
        try:
            if not track_id or not isinstance(track_id, str):
                raise ValidationError("트랙 ID가 유효하지 않습니다.")
            
            logger.info(f"트랙 분석 조회 시작: {track_id}")
            analysis = self.sp.audio_analysis(track_id)
            
            if analysis:
                logger.info(f"트랙 분석 조회 완료: {track_id}")
                return analysis
            else:
                logger.warning(f"트랙 분석 정보가 없습니다: {track_id}")
                return {}
                
        except Exception as e:
            logger.error(f"트랙 분석 조회 실패: {track_id}", exception=e)
            raise SpotifyAPIError(f"트랙 분석 조회에 실패했습니다: {str(e)}", original_exception=e)
    
    @handle_exception
    def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """플레이리스트의 모든 트랙을 가져옵니다."""
        try:
            DataValidator.validate_playlist_id(playlist_id)
            
            logger.info(f"플레이리스트 트랙 조회 시작: {playlist_id}")
            results = self.sp.playlist_tracks(playlist_id)
            tracks = []
            
            for item in results['items']:
                track = item['track']
                if track:
                    track_info = {
                        'id': track['id'],
                        'name': track['name'],
                        'artists': [artist['name'] for artist in track['artists']],
                        'album': track['album']['name'],
                        'release_date': track['album']['release_date'],
                        'popularity': track['popularity'],
                        'duration_ms': track['duration_ms'],
                        'explicit': track['explicit'],
                        'external_urls': track['external_urls']['spotify'],
                        'preview_url': track['preview_url']
                    }
                    
                    # 데이터 검증 및 정제
                    DataValidator.validate_track_data(track_info)
                    sanitized_track = DataSanitizer.sanitize_track_data(track_info)
                    tracks.append(sanitized_track)
            
            logger.info(f"플레이리스트 트랙 조회 완료: {len(tracks)}개 결과")
            return tracks
            
        except Exception as e:
            logger.error(f"플레이리스트 트랙 조회 실패: {playlist_id}", exception=e)
            raise SpotifyAPIError(f"플레이리스트 트랙 조회에 실패했습니다: {str(e)}", original_exception=e)
    
    @handle_exception
    def get_artist_top_tracks(self, artist_id: str, country: str = 'US') -> List[Dict[str, Any]]:
        """아티스트의 인기 트랙을 가져옵니다."""
        try:
            DataValidator.validate_artist_id(artist_id)
            
            logger.info(f"아티스트 인기 트랙 조회 시작: {artist_id}, 국가: {country}")
            results = self.sp.artist_top_tracks(artist_id, country=country)
            tracks = []
            
            for track in results['tracks']:
                track_info = {
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'release_date': track['album']['release_date'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms'],
                    'explicit': track['explicit'],
                    'external_urls': track['external_urls']['spotify'],
                    'preview_url': track['preview_url']
                }
                
                # 데이터 검증 및 정제
                DataValidator.validate_track_data(track_info)
                sanitized_track = DataSanitizer.sanitize_track_data(track_info)
                tracks.append(sanitized_track)
            
            logger.info(f"아티스트 인기 트랙 조회 완료: {len(tracks)}개 결과")
            return tracks
            
        except Exception as e:
            logger.error(f"아티스트 인기 트랙 조회 실패: {artist_id}", exception=e)
            raise SpotifyAPIError(f"아티스트 인기 트랙 조회에 실패했습니다: {str(e)}", original_exception=e)
    
    @handle_exception
    def collect_music_data(self, search_queries: Optional[List[str]] = None, 
                          playlist_ids: Optional[List[str]] = None,
                          artist_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """다양한 소스에서 음악 데이터를 수집합니다."""
        try:
            all_tracks = []
            
            logger.info("음악 데이터 수집 시작")
            
            # 검색어로 트랙 검색
            if search_queries:
                logger.info(f"검색어 기반 수집: {len(search_queries)}개 쿼리")
                for query in search_queries:
                    tracks = self.search_tracks(query, limit=50)
                    all_tracks.extend(tracks)
            
            # 플레이리스트에서 트랙 수집
            if playlist_ids:
                logger.info(f"플레이리스트 기반 수집: {len(playlist_ids)}개 플레이리스트")
                for playlist_id in playlist_ids:
                    DataValidator.validate_playlist_id(playlist_id)
                    tracks = self.get_playlist_tracks(playlist_id)
                    all_tracks.extend(tracks)
            
            # 아티스트의 인기 트랙 수집
            if artist_ids:
                logger.info(f"아티스트 기반 수집: {len(artist_ids)}개 아티스트")
                for artist_id in artist_ids:
                    DataValidator.validate_artist_id(artist_id)
                    tracks = self.get_artist_top_tracks(artist_id)
                    all_tracks.extend(tracks)
            
            if not all_tracks:
                logger.warning("수집된 트랙이 없습니다.")
                return pd.DataFrame()
            
            # 중복 제거
            unique_tracks = {track['id']: track for track in all_tracks}.values()
            logger.info(f"중복 제거 후 {len(unique_tracks)}개 트랙")
            
            # DataFrame으로 변환
            df = pd.DataFrame(list(unique_tracks))
            
            if not df.empty:
                # 오디오 특성 가져오기
                track_ids = df['id'].tolist()
                logger.info(f"오디오 특성 조회 시작: {len(track_ids)}개 트랙")
                
                features = self.get_track_features(track_ids)
                
                if features:
                    features_df = pd.DataFrame(features)
                    # ID로 병합
                    df = df.merge(features_df, left_on='id', right_on='id', how='left')
                    logger.info(f"오디오 특성 병합 완료: {len(df)}개 트랙")
            
            logger.info(f"음악 데이터 수집 완료: {len(df)}개 트랙")
            return df
            
        except Exception as e:
            logger.error("음악 데이터 수집 실패", exception=e)
            raise SpotifyAPIError(f"음악 데이터 수집에 실패했습니다: {str(e)}", original_exception=e)
    
    @handle_exception
    def save_to_csv(self, df: pd.DataFrame, filename: str = 'spotify_music_data.csv') -> str:
        """수집된 데이터를 CSV 파일로 저장합니다."""
        try:
            if df.empty:
                logger.warning("저장할 데이터가 비어있습니다.")
                return filename
            
            # 파일명 검증
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            logger.info(f"CSV 파일 저장 시작: {filename}")
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"CSV 파일 저장 완료: {filename} ({len(df)}개 행)")
            
            return filename
            
        except Exception as e:
            logger.error(f"CSV 파일 저장 실패: {filename}", exception=e)
            raise SpotifyAPIError(f"CSV 파일 저장에 실패했습니다: {str(e)}", original_exception=e)
