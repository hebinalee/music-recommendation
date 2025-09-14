import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class SpotifyMusicCollector:
    def __init__(self):
        """Spotify API 클라이언트를 초기화합니다."""
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify API credentials not found in environment variables")
        
        # Client Credentials Flow (public data access)
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
        )
    
    def search_tracks(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """검색어로 트랙을 검색합니다."""
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
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
                tracks.append(track_info)
            
            return tracks
        except Exception as e:
            print(f"Error searching tracks: {e}")
            return []
    
    def get_track_features(self, track_ids: List[str]) -> List[Dict[str, Any]]:
        """트랙의 오디오 특성을 가져옵니다."""
        try:
            features = self.sp.audio_features(track_ids)
            return features
        except Exception as e:
            print(f"Error getting track features: {e}")
            return []
    
    def get_track_analysis(self, track_id: str) -> Dict[str, Any]:
        """트랙의 상세 분석 정보를 가져옵니다."""
        try:
            analysis = self.sp.audio_analysis(track_id)
            return analysis
        except Exception as e:
            print(f"Error getting track analysis: {e}")
            return {}
    
    def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """플레이리스트의 모든 트랙을 가져옵니다."""
        try:
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
                    tracks.append(track_info)
            
            return tracks
        except Exception as e:
            print(f"Error getting playlist tracks: {e}")
            return []
    
    def get_artist_top_tracks(self, artist_id: str, country: str = 'US') -> List[Dict[str, Any]]:
        """아티스트의 인기 트랙을 가져옵니다."""
        try:
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
                tracks.append(track_info)
            
            return tracks
        except Exception as e:
            print(f"Error getting artist top tracks: {e}")
            return []
    
    def collect_music_data(self, search_queries: List[str] = None, 
                          playlist_ids: List[str] = None,
                          artist_ids: List[str] = None) -> pd.DataFrame:
        """다양한 소스에서 음악 데이터를 수집합니다."""
        all_tracks = []
        
        # 검색어로 트랙 검색
        if search_queries:
            for query in search_queries:
                tracks = self.search_tracks(query, limit=50)
                all_tracks.extend(tracks)
        
        # 플레이리스트에서 트랙 수집
        if playlist_ids:
            for playlist_id in playlist_ids:
                tracks = self.get_playlist_tracks(playlist_id)
                all_tracks.extend(tracks)
        
        # 아티스트의 인기 트랙 수집
        if artist_ids:
            for artist_id in artist_ids:
                tracks = self.get_artist_top_tracks(artist_id)
                all_tracks.extend(tracks)
        
        # 중복 제거
        unique_tracks = {track['id']: track for track in all_tracks}.values()
        
        # DataFrame으로 변환
        df = pd.DataFrame(list(unique_tracks))
        
        if not df.empty:
            # 오디오 특성 가져오기
            track_ids = df['id'].tolist()
            features = self.get_track_features(track_ids)
            
            if features:
                features_df = pd.DataFrame(features)
                # ID로 병합
                df = df.merge(features_df, left_on='id', right_on='id', how='left')
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = 'spotify_music_data.csv'):
        """수집된 데이터를 CSV 파일로 저장합니다."""
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        return filename
