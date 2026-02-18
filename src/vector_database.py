import chromadb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from .config import ConfigManager
from .exceptions import DatabaseError, handle_exception, ValidationError
from .validators import DataValidator, DataSanitizer
from .logger import get_logger

load_dotenv()
logger = get_logger(__name__)

class MusicVectorDatabase:
    def __init__(self, persist_directory: Optional[str] = None, config_manager: Optional[ConfigManager] = None):
        """Chroma vector DB를 초기화합니다."""
        try:
            self.config_manager = config_manager or ConfigManager()
            database_config = self.config_manager.get_database_config()
            
            self.persist_directory = persist_directory or database_config.persist_directory
            self.collection_name = database_config.collection_name
            
            logger.info(f"Vector DB 초기화 시작: {self.persist_directory}")
            
            # 디렉토리 생성
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Chroma 클라이언트 초기화
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # 임베딩 모델 초기화
            logger.info("임베딩 모델 로딩 중...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 컬렉션 생성 또는 가져오기
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Spotify music tracks with embeddings"}
            )
            
            logger.info(f"Vector DB 초기화 완료: {self.collection_name}")
            
        except Exception as e:
            logger.error("Vector DB 초기화 실패", exception=e)
            raise DatabaseError("Vector DB 초기화에 실패했습니다.", original_exception=e)
    
    @handle_exception
    def create_music_embedding(self, music_data: pd.DataFrame) -> List[str]:
        """음악 데이터로부터 텍스트 임베딩을 생성합니다."""
        try:
            if music_data.empty:
                logger.warning("임베딩을 생성할 음악 데이터가 없습니다.")
                return []
            
            logger.info(f"텍스트 임베딩 생성 시작: {len(music_data)}개 트랙")
            embeddings = []
            
            for _, row in music_data.iterrows():
                # 음악 정보를 텍스트로 결합
                text_parts = [
                    str(row['name']),
                    ' '.join(row['artists']) if isinstance(row['artists'], list) else str(row['artists']),
                    str(row['album'])
                ]
                
                # 장르나 태그가 있다면 추가
                if 'genres' in row and pd.notna(row['genres']):
                    text_parts.append(str(row['genres']))
                
                # 오디오 특성 정보 추가
                audio_features = ['danceability', 'energy', 'valence', 'tempo', 'instrumentalness']
                for feature in audio_features:
                    if feature in row and pd.notna(row[feature]):
                        text_parts.append(f"{feature}:{row[feature]:.2f}")
                
                text = ' '.join(text_parts)
                embeddings.append(text)
            
            logger.info(f"텍스트 임베딩 생성 완료: {len(embeddings)}개")
            return embeddings
            
        except Exception as e:
            logger.error("텍스트 임베딩 생성 실패", exception=e)
            raise DatabaseError(f"텍스트 임베딩 생성에 실패했습니다: {str(e)}", original_exception=e)
    
    @handle_exception
    def add_music_to_database(self, music_data: pd.DataFrame) -> bool:
        """음악 데이터를 vector DB에 추가합니다."""
        try:
            if music_data.empty:
                logger.warning("추가할 음악 데이터가 없습니다.")
                return False
            
            # 데이터 검증
            required_columns = ['id', 'name', 'artists', 'album']
            DataValidator.validate_dataframe(music_data, required_columns)
            
            logger.info(f"음악 데이터 추가 시작: {len(music_data)}개 트랙")
            
            # 중복 제거 (이미 존재하는 ID는 제외)
            existing_ids = set()
            if self.collection.count() > 0:
                existing_ids = set(self.collection.get()['ids'])
            
            # 새로운 데이터만 필터링
            new_data = music_data[~music_data['id'].isin(existing_ids)]
            
            if new_data.empty:
                logger.info("모든 음악 데이터가 이미 데이터베이스에 존재합니다.")
                return True
            
            logger.info(f"새로운 데이터: {len(new_data)}개 트랙")
            
            # 텍스트 임베딩 생성
            text_embeddings = self.create_music_embedding(new_data)
            
            # 벡터 임베딩 생성
            logger.info("벡터 임베딩 생성 중...")
            embeddings = self.embedding_model.encode(text_embeddings)
            
            # 메타데이터 준비
            metadatas = []
            for _, row in new_data.iterrows():
                metadata = {
                    'name': str(row['name']),
                    'artists': str(row['artists']),
                    'album': str(row['album']),
                    'release_date': str(row['release_date']),
                    'popularity': int(row['popularity']) if pd.notna(row['popularity']) else 0,
                    'duration_ms': int(row['duration_ms']) if pd.notna(row['duration_ms']) else 0,
                    'external_urls': str(row['external_urls']) if 'external_urls' in row else '',
                    'preview_url': str(row['preview_url']) if 'preview_url' in row else ''
                }
                
                # 오디오 특성 추가
                audio_features = ['danceability', 'energy', 'valence', 'tempo', 'instrumentalness', 
                                'acousticness', 'liveness', 'speechiness', 'key', 'mode', 'time_signature']
                for feature in audio_features:
                    if feature in row and pd.notna(row[feature]):
                        metadata[feature] = float(row[feature])
                
                metadatas.append(metadata)
            
            # DB에 추가
            logger.info("데이터베이스에 데이터 추가 중...")
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=text_embeddings,
                metadatas=metadatas,
                ids=new_data['id'].tolist()
            )
            
            logger.info(f"음악 데이터 추가 완료: {len(new_data)}개 트랙")
            return True
            
        except Exception as e:
            logger.error("음악 데이터 추가 실패", exception=e)
            raise DatabaseError(f"음악 데이터 추가에 실패했습니다: {str(e)}", original_exception=e)
    
    @handle_exception
    def search_similar_music(self, query: str, n_results: int = 10, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """유사한 음악을 검색합니다."""
        try:
            # 입력 검증
            DataValidator.validate_search_query(query)
            DataValidator.validate_n_results(n_results)
            
            logger.info(f"음악 검색 시작: '{query}', 결과 수: {n_results}")
            
            # 쿼리 정제
            sanitized_query = DataSanitizer.sanitize_search_query(query)
            
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([sanitized_query])
            
            # 검색 실행
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=filters
            )
            
            # 결과 정리
            similar_music = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    music_info = {
                        'id': results['ids'][0][i],
                        'name': results['metadatas'][0][i]['name'],
                        'artists': results['metadatas'][0][i]['artists'],
                        'album': results['metadatas'][0][i]['album'],
                        'similarity_score': results['distances'][0][i],
                        'metadata': results['metadatas'][0][i]
                    }
                    similar_music.append(music_info)
            
            logger.info(f"음악 검색 완료: {len(similar_music)}개 결과")
            return similar_music
            
        except Exception as e:
            logger.error(f"음악 검색 실패: '{query}'", exception=e)
            raise DatabaseError(f"음악 검색에 실패했습니다: {str(e)}", original_exception=e)
    
    @handle_exception
    def search_by_audio_features(self, target_features: Dict[str, float], 
                               n_results: int = 10) -> List[Dict[str, Any]]:
        """오디오 특성 기반으로 음악을 검색합니다."""
        try:
            # 입력 검증
            DataValidator.validate_audio_features(target_features)
            DataValidator.validate_n_results(n_results)
            
            logger.info(f"오디오 특성 기반 검색 시작: {len(target_features)}개 특성, 결과 수: {n_results}")
            
            # 모든 음악 검색 후 특성 기반 정렬
            all_results = self.collection.get()
            
            if not all_results['ids']:
                logger.warning("데이터베이스에 음악이 없습니다.")
                return []
            
            logger.info(f"전체 {len(all_results['ids'])}개 트랙에서 특성 기반 검색 수행")
            
            # 특성 기반 점수 계산
            scores = []
            for i, metadata in enumerate(all_results['metadatas']):
                score = 0
                for feature, target_value in target_features.items():
                    if feature in metadata:
                        feature_value = metadata[feature]
                        # 유클리드 거리 기반 점수 (낮을수록 좋음)
                        score += (feature_value - target_value) ** 2
                
                scores.append((score, i))
            
            # 점수로 정렬
            scores.sort()
            
            # 상위 결과 반환
            top_results = []
            for score, idx in scores[:n_results]:
                music_info = {
                    'id': all_results['ids'][idx],
                    'name': all_results['metadatas'][idx]['name'],
                    'artists': all_results['metadatas'][idx]['artists'],
                    'album': all_results['metadatas'][idx]['album'],
                    'feature_score': score,
                    'metadata': all_results['metadatas'][idx]
                }
                top_results.append(music_info)
            
            logger.info(f"오디오 특성 기반 검색 완료: {len(top_results)}개 결과")
            return top_results
            
        except Exception as e:
            logger.error("오디오 특성 기반 검색 실패", exception=e)
            raise DatabaseError(f"오디오 특성 기반 검색에 실패했습니다: {str(e)}", original_exception=e)
    
    @handle_exception
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보를 반환합니다."""
        try:
            count = self.collection.count()
            stats = {
                'total_tracks': count,
                'persist_directory': self.persist_directory,
                'collection_name': self.collection_name
            }
            
            logger.info(f"데이터베이스 통계: {count}개 트랙")
            return stats
            
        except Exception as e:
            logger.error("데이터베이스 통계 조회 실패", exception=e)
            raise DatabaseError(f"데이터베이스 통계 조회에 실패했습니다: {str(e)}", original_exception=e)
    
    @handle_exception
    def clear_database(self) -> bool:
        """데이터베이스를 초기화합니다."""
        try:
            logger.warning("데이터베이스 초기화 시작")
            
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Spotify music tracks with embeddings"}
            )
            
            logger.info("데이터베이스 초기화 완료")
            return True
            
        except Exception as e:
            logger.error("데이터베이스 초기화 실패", exception=e)
            raise DatabaseError(f"데이터베이스 초기화에 실패했습니다: {str(e)}", original_exception=e)
