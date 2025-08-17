import chromadb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class MusicVectorDatabase:
    def __init__(self, persist_directory: str = None):
        """Chroma vector DB를 초기화합니다."""
        if persist_directory is None:
            persist_directory = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
        
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 임베딩 모델 초기화
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 컬렉션 생성 또는 가져오기
        self.collection = self.client.get_or_create_collection(
            name="music_collection",
            metadata={"description": "Spotify music tracks with embeddings"}
        )
    
    def create_music_embedding(self, music_data: pd.DataFrame) -> List[str]:
        """음악 데이터로부터 텍스트 임베딩을 생성합니다."""
        embeddings = []
        
        for _, row in music_data.iterrows():
            # 음악 정보를 텍스트로 결합
            text = f"{row['name']} {' '.join(row['artists'])} {row['album']}"
            
            # 장르나 태그가 있다면 추가
            if 'genres' in row and pd.notna(row['genres']):
                text += f" {row['genres']}"
            
            # 오디오 특성 정보 추가
            audio_features = ['danceability', 'energy', 'valence', 'tempo', 'instrumentalness']
            for feature in audio_features:
                if feature in row and pd.notna(row[feature]):
                    text += f" {feature}:{row[feature]:.2f}"
            
            embeddings.append(text)
        
        return embeddings
    
    def add_music_to_database(self, music_data: pd.DataFrame) -> bool:
        """음악 데이터를 vector DB에 추가합니다."""
        try:
            if music_data.empty:
                print("No music data to add")
                return False
            
            # 중복 제거 (이미 존재하는 ID는 제외)
            existing_ids = set()
            if self.collection.count() > 0:
                existing_ids = set(self.collection.get()['ids'])
            
            # 새로운 데이터만 필터링
            new_data = music_data[~music_data['id'].isin(existing_ids)]
            
            if new_data.empty:
                print("All music data already exists in database")
                return True
            
            # 텍스트 임베딩 생성
            text_embeddings = self.create_music_embedding(new_data)
            
            # 벡터 임베딩 생성
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
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=text_embeddings,
                metadatas=metadatas,
                ids=new_data['id'].tolist()
            )
            
            print(f"Added {len(new_data)} new tracks to database")
            return True
            
        except Exception as e:
            print(f"Error adding music to database: {e}")
            return False
    
    def search_similar_music(self, query: str, n_results: int = 10, 
                           filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """유사한 음악을 검색합니다."""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query])
            
            # 검색 실행
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=filters
            )
            
            # 결과 정리
            similar_music = []
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
            
            return similar_music
            
        except Exception as e:
            print(f"Error searching similar music: {e}")
            return []
    
    def search_by_audio_features(self, target_features: Dict[str, float], 
                               n_results: int = 10) -> List[Dict[str, Any]]:
        """오디오 특성 기반으로 음악을 검색합니다."""
        try:
            # 특성 기반 필터 생성
            filters = {}
            for feature, value in target_features.items():
                if feature in ['danceability', 'energy', 'valence', 'acousticness', 
                             'instrumentalness', 'liveness', 'speechiness']:
                    # 0-1 범위의 특성에 대해 근사값 검색
                    filters[f"$and"] = [
                        {feature: {"$gte": max(0, value - 0.1)}},
                        {feature: {"$lte": min(1, value + 0.1)}}
                    ]
            
            # 모든 음악 검색 후 특성 기반 정렬
            all_results = self.collection.get()
            
            if not all_results['ids']:
                return []
            
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
            
            return top_results
            
        except Exception as e:
            print(f"Error searching by audio features: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보를 반환합니다."""
        try:
            count = self.collection.count()
            return {
                'total_tracks': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def clear_database(self) -> bool:
        """데이터베이스를 초기화합니다."""
        try:
            self.client.delete_collection("music_collection")
            self.collection = self.client.create_collection(
                name="music_collection",
                metadata={"description": "Spotify music tracks with embeddings"}
            )
            print("Database cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False

if __name__ == "__main__":
    # 사용 예시
    db = MusicVectorDatabase()
    
    # 데이터베이스 통계 확인
    stats = db.get_database_stats()
    print(f"Database stats: {stats}")
    
    # 유사한 음악 검색 예시
    similar = db.search_similar_music("k-pop dance music", n_results=5)
    print(f"Found {len(similar)} similar tracks")
    
    for track in similar[:3]:
        print(f"- {track['name']} by {track['artists']} (Score: {track['similarity_score']:.3f})")
