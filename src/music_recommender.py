import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from .vector_database import MusicVectorDatabase
from .spotify_collector import SpotifyMusicCollector

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .ann_index import ANNIndex

class MusicRecommender:
    def __init__(self, vector_db: MusicVectorDatabase, spotify_collector: SpotifyMusicCollector):
        """음악 추천 시스템을 초기화합니다."""
        self.vector_db = vector_db
        self.spotify_collector = spotify_collector
        self.user_preferences = {}
        
        # Two-stage 추천 시스템 초기화
        load_dotenv()
        self.model_save_dir = os.getenv('MODEL_SAVE_DIR', './models')
        self.embedding_dim = int(os.getenv('EMBEDDING_DIM', '128'))
        self.hidden_dim = int(os.getenv('HIDDEN_DIM', '256'))
        self.learning_rate = float(os.getenv('LEARNING_RATE', '0.001'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '32'))
        self.epochs = int(os.getenv('EPOCHS', '100'))
        
        # 모델 디렉토리 생성
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Two-Tower 모델 (후보 생성용)
        self.two_tower_model = None
        self.user_encoder = None
        self.item_encoder = None
        
        # Wide&Deep 모델 (랭킹용)
        self.wide_deep_model = None
        
        # 데이터 전처리기
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # 모델 로드 시도
        self._load_models()
        
        # ANN 인덱스 초기화
        self.ann_index = ANNIndex(self.embedding_dim, 
                                   index_path=os.path.join(self.model_save_dir, 'faiss_index.bin'),
                                   mapping_path=os.path.join(self.model_save_dir, 'faiss_id_map.json'))
        self.ann_index.load()
    
    def _load_models(self):
        """저장된 모델들을 로드합니다."""
        try:
            # Two-Tower 모델 로드
            two_tower_path = os.path.join(self.model_save_dir, 'two_tower_model.pth')
            if os.path.exists(two_tower_path):
                self.two_tower_model = TwoTowerModel(self.embedding_dim, self.hidden_dim)
                self.two_tower_model.load_state_dict(torch.load(two_tower_path))
                self.two_tower_model.eval()
                print("✅ Two-Tower 모델 로드 완료")
            
            # Wide&Deep 모델 로드
            wide_deep_path = os.path.join(self.model_save_dir, 'wide_deep_model')
            if os.path.exists(wide_deep_path):
                self.wide_deep_model = tf.keras.models.load_model(wide_deep_path)
                print("✅ Wide&Deep 모델 로드 완료")
                
        except Exception as e:
            print(f"⚠️ 모델 로드 중 오류 발생: {e}")
            print("새로운 모델을 생성합니다.")
    
    def add_user_preference(self, user_id: str, track_id: str, rating: float = 1.0):
        """사용자의 음악 선호도를 추가합니다."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        self.user_preferences[user_id][track_id] = rating
        print(f"Added preference for user {user_id}: track {track_id} with rating {rating}")
    
    def get_user_profile(self, user_id: str) -> Dict[str, float]:
        """사용자의 음악 프로필을 생성합니다."""
        if user_id not in self.user_preferences:
            return {}
        
        # 사용자가 평가한 트랙들의 오디오 특성 평균 계산
        user_tracks = self.user_preferences[user_id]
        audio_features = []
        weights = []
        
        for track_id, rating in user_tracks.items():
            # 트랙의 메타데이터에서 오디오 특성 추출
            track_data = self.vector_db.collection.get(ids=[track_id])
            if track_data['metadatas']:
                metadata = track_data['metadatas'][0]
                features = {}
                
                # 오디오 특성 수집
                audio_feature_names = ['danceability', 'energy', 'valence', 'tempo', 
                                     'instrumentalness', 'acousticness', 'liveness', 'speechiness']
                
                for feature in audio_feature_names:
                    if feature in metadata:
                        features[feature] = metadata[feature]
                
                if features:
                    audio_features.append(features)
                    weights.append(rating)
        
        if not audio_features:
            return {}
        
        # 가중 평균 계산
        user_profile = {}
        for feature in audio_features[0].keys():
            weighted_sum = sum(features[feature] * weight for features, weight in zip(audio_features, weights))
            total_weight = sum(weights)
            user_profile[feature] = weighted_sum / total_weight
        
        return user_profile
    
    def recommend_music(self, user_id: str, n_recommendations: int = 10, 
                       method: str = 'hybrid') -> List[Dict[str, Any]]:
        """사용자에게 음악을 추천합니다."""
        if method == 'two_stage':
            return self._two_stage_recommendation(user_id, n_recommendations)
        elif method == 'collaborative':
            return self._collaborative_filtering(user_id, n_recommendations)
        elif method == 'content_based':
            return self._content_based_filtering(user_id, n_recommendations)
        elif method == 'hybrid':
            return self._hybrid_recommendation(user_id, n_recommendations)
        else:
            raise ValueError("Method must be 'two_stage', 'collaborative', 'content_based', or 'hybrid'")
    
    def _content_based_filtering(self, user_id: str, n_recommendations: int) -> List[Dict[str, Any]]:
        """콘텐츠 기반 필터링을 사용한 추천"""
        user_profile = self.get_user_profile(user_id)
        
        if not user_profile:
            return []
        
        # 사용자 프로필과 유사한 오디오 특성을 가진 음악 검색
        recommendations = self.vector_db.search_by_audio_features(
            user_profile, n_results=n_recommendations
        )
        
        # 이미 사용자가 평가한 트랙은 제외
        user_rated_tracks = set(self.user_preferences.get(user_id, {}).keys())
        filtered_recommendations = [
            rec for rec in recommendations 
            if rec['id'] not in user_rated_tracks
        ]
        
        return filtered_recommendations[:n_recommendations]
    
    def _collaborative_filtering(self, user_id: str, n_recommendations: int) -> List[Dict[str, Any]]:
        """협업 필터링을 사용한 추천"""
        if user_id not in self.user_preferences:
            return []
        
        # 다른 사용자들과의 유사도 계산
        user_similarities = {}
        current_user_tracks = set(self.user_preferences[user_id].keys())
        
        for other_user_id, other_preferences in self.user_preferences.items():
            if other_user_id == user_id:
                continue
            
            other_user_tracks = set(other_preferences.keys())
            
            # Jaccard 유사도 계산
            intersection = len(current_user_tracks & other_user_tracks)
            union = len(current_user_tracks | other_user_tracks)
            
            if union > 0:
                similarity = intersection / union
                user_similarities[other_user_id] = similarity
        
        # 유사도가 높은 사용자들의 선호도 기반 추천
        recommendations = {}
        
        for other_user_id, similarity in sorted(user_similarities.items(), 
                                              key=lambda x: x[1], reverse=True)[:5]:
            other_preferences = self.user_preferences[other_user_id]
            
            for track_id, rating in other_preferences.items():
                if track_id not in current_user_tracks:
                    if track_id not in recommendations:
                        recommendations[track_id] = 0
                    recommendations[track_id] += similarity * rating
        
        # 추천 점수로 정렬
        sorted_recommendations = sorted(recommendations.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        # 상위 추천 결과 반환
        top_recommendations = []
        for track_id, score in sorted_recommendations[:n_recommendations]:
            track_data = self.vector_db.collection.get(ids=[track_id])
            if track_data['metadatas']:
                metadata = track_data['metadatas'][0]
                recommendation = {
                    'id': track_id,
                    'name': metadata['name'],
                    'artists': metadata['artists'],
                    'album': metadata['album'],
                    'recommendation_score': score,
                    'metadata': metadata
                }
                top_recommendations.append(recommendation)
        
        return top_recommendations
    
    def _hybrid_recommendation(self, user_id: str, n_recommendations: int) -> List[Dict[str, Any]]:
        """하이브리드 추천 (콘텐츠 기반 + 협업 필터링)"""
        content_based = self._content_based_filtering(user_id, n_recommendations // 2)
        collaborative = self._collaborative_filtering(user_id, n_recommendations // 2)
        
        # 결과 병합 및 정렬
        all_recommendations = []
        
        for rec in content_based:
            rec['method'] = 'content_based'
            all_recommendations.append(rec)
        
        for rec in collaborative:
            rec['method'] = 'collaborative'
            all_recommendations.append(rec)
        
        # 중복 제거 (ID 기준)
        unique_recommendations = {}
        for rec in all_recommendations:
            if rec['id'] not in unique_recommendations:
                unique_recommendations[rec['id']] = rec
            else:
                # 중복된 경우 점수 평균
                existing = unique_recommendations[rec['id']]
                if 'recommendation_score' in existing and 'feature_score' in rec:
                    existing['hybrid_score'] = (existing.get('recommendation_score', 0) + 
                                              (1.0 / (1.0 + rec['feature_score']))) / 2
                elif 'feature_score' in existing and 'recommendation_score' in rec:
                    existing['hybrid_score'] = (existing['feature_score'] + 
                                              rec['recommendation_score']) / 2
        
        # 하이브리드 점수로 정렬
        final_recommendations = list(unique_recommendations.values())
        final_recommendations.sort(key=lambda x: x.get('hybrid_score', 0) or x.get('feature_score', 0) or x.get('recommendation_score', 0), reverse=True)
        
        return final_recommendations[:n_recommendations]
    
    def _two_stage_recommendation(self, user_id: str, n_recommendations: int) -> List[Dict[str, Any]]:
        """Two-stage 추천 시스템: Two-Tower로 후보 생성, Wide&Deep으로 랭킹"""
        try:
            # Stage 1: Two-Tower 모델로 후보 생성 (더 많은 후보 생성)
            candidate_count = min(n_recommendations * 10, 1000)  # 충분한 후보 생성
            candidates = self._generate_candidates_two_tower(user_id, candidate_count)
            
            if not candidates:
                print("⚠️ Two-Tower 모델로 후보를 생성할 수 없습니다. 기본 방법으로 대체합니다.")
                return self._hybrid_recommendation(user_id, n_recommendations)
            
            # Stage 2: Wide&Deep 모델로 랭킹
            ranked_candidates = self._rank_candidates_wide_deep(user_id, candidates)
            
            # 상위 결과 반환
            final_recommendations = []
            for candidate in ranked_candidates[:n_recommendations]:
                track_data = self.vector_db.collection.get(ids=[candidate['id']])
                if track_data['metadatas']:
                    metadata = track_data['metadatas'][0]
                    recommendation = {
                        'id': candidate['id'],
                        'name': metadata['name'],
                        'artists': metadata['artists'],
                        'album': metadata['album'],
                        'two_stage_score': candidate['score'],
                        'method': 'two_stage',
                        'metadata': metadata
                    }
                    final_recommendations.append(recommendation)
            
            return final_recommendations
            
        except Exception as e:
            print(f"❌ Two-stage 추천 중 오류 발생: {e}")
            print("기본 하이브리드 방법으로 대체합니다.")
            return self._hybrid_recommendation(user_id, n_recommendations)
    
    def _generate_candidates_two_tower(self, user_id: str, n_candidates: int) -> List[Dict[str, Any]]:
        """Two-Tower 모델을 사용하여 후보를 생성합니다."""
        try:
            if self.two_tower_model is None:
                print("⚠️ Two-Tower 모델이 없습니다. 모델을 훈련합니다.")
                self._train_two_tower_model()
            
            # 사용자 임베딩 생성
            user_embedding = self._get_user_embedding(user_id)
            if user_embedding is None:
                return []
            
            # ANN 인덱스가 준비되지 않았다면 빌드 시도
            if not self.ann_index.is_ready():
                self._rebuild_ann_index()
                if not self.ann_index.is_ready():
                    print("⚠️ ANN 인덱스를 사용할 수 없어 전수 검색으로 대체합니다.")
                    return self._fallback_bruteforce_candidates(user_embedding, n_candidates)
            
            # ANN 검색
            user_vec = user_embedding.detach().cpu().numpy().reshape(1, -1)
            scores, ids = self.ann_index.search(user_vec, n_candidates)
            decoded_ids = self.ann_index.decode_ids(ids)
            results = []
            for i, item_id in enumerate(decoded_ids[0]):
                if not item_id:
                    continue
                results.append({'id': item_id, 'similarity': float(scores[0][i])})
            return results
        except Exception as e:
            print(f"❌ Two-Tower 후보 생성 중 오류: {e}")
            return []

    def _fallback_bruteforce_candidates(self, user_embedding: torch.Tensor, n_candidates: int) -> List[Dict[str, Any]]:
        all_items = self.vector_db.collection.get()
        if not all_items['ids']:
            return []
        similarities = []
        for item_id in all_items['ids']:
            item_embedding = self._get_item_embedding(item_id)
            if item_embedding is None:
                continue
            similarity = F.cosine_similarity(user_embedding.unsqueeze(0), item_embedding.unsqueeze(0)).item()
            similarities.append({'id': item_id, 'similarity': similarity})
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:n_candidates]
    
    def _rank_candidates_wide_deep(self, user_id: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Wide&Deep 모델을 사용하여 후보들을 랭킹합니다."""
        try:
            if self.wide_deep_model is None:
                print("⚠️ Wide&Deep 모델이 없습니다. 모델을 훈련합니다.")
                self._train_wide_deep_model()
            
            # 사용자가 이미 평가한 트랙은 제외
            user_rated_tracks = set(self.user_preferences.get(user_id, {}).keys())
            filtered_candidates = [c for c in candidates if c['id'] not in user_rated_tracks]
            
            if not filtered_candidates:
                return candidates
            
            # Wide&Deep 모델로 점수 예측
            ranked_candidates = []
            for candidate in filtered_candidates:
                score = self._predict_wide_deep_score(user_id, candidate['id'])
                ranked_candidates.append({
                    'id': candidate['id'],
                    'score': score,
                    'similarity': candidate['similarity']
                })
            
            # 점수로 정렬
            ranked_candidates.sort(key=lambda x: x['score'], reverse=True)
            return ranked_candidates
            
        except Exception as e:
            print(f"❌ Wide&Deep 랭킹 중 오류: {e}")
            # 기본 유사도로 정렬
            return sorted(candidates, key=lambda x: x['similarity'], reverse=True)
    
    def _get_user_embedding(self, user_id: str) -> Optional[torch.Tensor]:
        """사용자의 임베딩을 생성합니다."""
        try:
            user_profile = self.get_user_profile(user_id)
            if not user_profile:
                return None
            
            # 사용자 특성을 텐서로 변환
            user_features = []
            feature_names = ['danceability', 'energy', 'valence', 'tempo', 
                           'instrumentalness', 'acousticness', 'liveness', 'speechiness']
            
            for feature in feature_names:
                if feature in user_profile:
                    user_features.append(user_profile[feature])
                else:
                    user_features.append(0.0)
            
            user_tensor = torch.tensor(user_features, dtype=torch.float32)
            
            # Two-Tower 모델의 사용자 인코더를 통과
            if self.two_tower_model:
                with torch.no_grad():
                    user_embedding = self.two_tower_model.user_tower(user_tensor.unsqueeze(0))
                    return user_embedding.squeeze(0)
            
            return None
            
        except Exception as e:
            print(f"❌ 사용자 임베딩 생성 중 오류: {e}")
            return None
    
    def _get_item_embedding(self, item_id: str) -> Optional[torch.Tensor]:
        """아이템의 임베딩을 생성합니다."""
        try:
            item_data = self.vector_db.collection.get(ids=[item_id])
            if not item_data['metadatas']:
                return None
            
            metadata = item_data['metadatas'][0]
            
            # 아이템 특성을 텐서로 변환
            item_features = []
            feature_names = ['danceability', 'energy', 'valence', 'tempo', 
                           'instrumentalness', 'acousticness', 'liveness', 'speechiness']
            
            for feature in feature_names:
                if feature in metadata:
                    item_features.append(metadata[feature])
                else:
                    item_features.append(0.0)
            
            item_tensor = torch.tensor(item_features, dtype=torch.float32)
            
            # Two-Tower 모델의 아이템 인코더를 통과
            if self.two_tower_model:
                with torch.no_grad():
                    item_embedding = self.two_tower_model.item_tower(item_tensor.unsqueeze(0))
                    return item_embedding.squeeze(0)
            
            return None
            
        except Exception as e:
            print(f"❌ 아이템 임베딩 생성 중 오류: {e}")
            return None
    
    def _predict_wide_deep_score(self, user_id: str, item_id: str) -> float:
        """Wide&Deep 모델을 사용하여 점수를 예측합니다."""
        try:
            # 사용자 특성
            user_profile = self.get_user_profile(user_id)
            user_features = []
            feature_names = ['danceability', 'energy', 'valence', 'tempo', 
                           'instrumentalness', 'acousticness', 'liveness', 'speechiness']
            
            for feature in feature_names:
                if feature in user_profile:
                    user_features.append(user_profile[feature])
                else:
                    user_features.append(0.0)
            
            # 아이템 특성
            item_data = self.vector_db.collection.get(ids=[item_id])
            if not item_data['metadatas']:
                return 0.0
            
            metadata = item_data['metadatas'][0]
            item_features = []
            
            for feature in feature_names:
                if feature in metadata:
                    item_features.append(metadata[feature])
                else:
                    item_features.append(0.0)
            
            # 특성 결합
            combined_features = user_features + item_features
            
            # 모델 예측
            if self.wide_deep_model:
                prediction = self.wide_deep_model.predict(
                    np.array([combined_features]), verbose=0
                )
                return float(prediction[0][0])
            
            return 0.0
            
        except Exception as e:
            print(f"❌ Wide&Deep 예측 중 오류: {e}")
            return 0.0
    
    def _train_two_tower_model(self):
        """Two-Tower 모델을 훈련합니다."""
        try:
            print("🔄 Two-Tower 모델 훈련을 시작합니다...")
            
            # 모델 생성
            self.two_tower_model = TwoTowerModel(self.embedding_dim, self.hidden_dim)
            
            # 훈련 데이터 준비
            train_data = self._prepare_two_tower_training_data()
            if not train_data:
                print("❌ 훈련 데이터가 부족합니다.")
                return
            
            # 훈련 설정
            optimizer = torch.optim.Adam(self.two_tower_model.parameters(), lr=self.learning_rate)
            criterion = nn.BCEWithLogitsLoss()
            
            # 훈련 루프
            self.two_tower_model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for batch in train_data:
                    user_features, item_features, labels = batch
                    
                    optimizer.zero_grad()
                    outputs = self.two_tower_model(user_features, item_features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(train_data):.4f}")
            
            # 모델 저장
            model_path = os.path.join(self.model_save_dir, 'two_tower_model.pth')
            torch.save(self.two_tower_model.state_dict(), model_path)
            print(f"✅ Two-Tower 모델 훈련 완료 및 저장: {model_path}")
            # 모델이 갱신되었으므로 ANN 인덱스 재빌드
            self._rebuild_ann_index()
            
        except Exception as e:
            print(f"❌ Two-Tower 모델 훈련 중 오류: {e}")

    def _rebuild_ann_index(self) -> None:
        try:
            # 모든 아이템 임베딩 계산
            all_items = self.vector_db.collection.get()
            if not all_items['ids']:
                return
            item_ids: List[str] = []
            item_emb_list: List[np.ndarray] = []
            for item_id in all_items['ids']:
                emb = self._get_item_embedding(item_id)
                if emb is None:
                    continue
                item_ids.append(item_id)
                item_emb_list.append(emb.detach().cpu().numpy())
            if not item_emb_list:
                return
            item_emb_matrix = np.vstack(item_emb_list)
            self.ann_index.build_from_embeddings(item_ids, item_emb_matrix, use_ivf=False)
            # IVF가 필요한 대규모일 경우 use_ivf=True로 전환 가능
            self.ann_index.save()
            print(f"✅ ANN index built with {len(item_ids)} items")
        except Exception as e:
            print(f"❌ ANN 인덱스 빌드 실패: {e}")
    
    def _train_wide_deep_model(self):
        """Wide&Deep 모델을 훈련합니다."""
        try:
            print("🔄 Wide&Deep 모델 훈련을 시작합니다...")
            
            # 훈련 데이터 준비
            train_data = self._prepare_wide_deep_training_data()
            if not train_data:
                print("❌ 훈련 데이터가 부족합니다.")
                return
            
            # 모델 생성
            input_dim = len(train_data[0][0])  # 특성 차원
            self.wide_deep_model = self._build_wide_deep_model(input_dim)
            
            # 훈련
            X, y = zip(*train_data)
            X = np.array(X)
            y = np.array(y)
            
            history = self.wide_deep_model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=1
            )
            
            # 모델 저장
            model_path = os.path.join(self.model_save_dir, 'wide_deep_model')
            self.wide_deep_model.save(model_path)
            print(f"✅ Wide&Deep 모델 훈련 완료 및 저장: {model_path}")
            
        except Exception as e:
            print(f"❌ Wide&Deep 모델 훈련 중 오류: {e}")
    
    def _prepare_two_tower_training_data(self):
        """Two-Tower 모델 훈련을 위한 데이터를 준비합니다."""
        try:
            training_data = []
            
            # 사용자-아이템 상호작용 데이터 생성
            for user_id, preferences in self.user_preferences.items():
                user_profile = self.get_user_profile(user_id)
                if not user_profile:
                    continue
                
                for item_id, rating in preferences.items():
                    item_data = self.vector_db.collection.get(ids=[item_id])
                    if not item_data['metadatas']:
                        continue
                    
                    metadata = item_data['metadatas'][0]
                    
                    # 사용자 특성
                    user_features = []
                    feature_names = ['danceability', 'energy', 'valence', 'tempo', 
                                   'instrumentalness', 'acousticness', 'liveness', 'speechiness']
                    
                    for feature in feature_names:
                        if feature in user_profile:
                            user_features.append(user_profile[feature])
                        else:
                            user_features.append(0.0)
                    
                    # 아이템 특성
                    item_features = []
                    for feature in feature_names:
                        if feature in metadata:
                            item_features.append(metadata[feature])
                        else:
                            item_features.append(0.0)
                    
                    # 긍정적 샘플 (rating >= 3.0)
                    if rating >= 3.0:
                        training_data.append((
                            torch.tensor(user_features, dtype=torch.float32),
                            torch.tensor(item_features, dtype=torch.float32),
                            torch.tensor([1.0], dtype=torch.float32)
                        ))
                    
                    # 부정적 샘플 (rating < 3.0)
                    if rating < 3.0:
                        training_data.append((
                            torch.tensor(user_features, dtype=torch.float32),
                            torch.tensor(item_features, dtype=torch.float32),
                            torch.tensor([0.0], dtype=torch.float32)
                        ))
            
            return training_data
            
        except Exception as e:
            print(f"❌ Two-Tower 훈련 데이터 준비 중 오류: {e}")
            return []
    
    def _prepare_wide_deep_training_data(self):
        """Wide&Deep 모델 훈련을 위한 데이터를 준비합니다."""
        try:
            training_data = []
            
            # 사용자-아이템 상호작용 데이터 생성
            for user_id, preferences in self.user_preferences.items():
                user_profile = self.get_user_profile(user_id)
                if not user_profile:
                    continue
                
                for item_id, rating in preferences.items():
                    item_data = self.vector_db.collection.get(ids=[item_id])
                    if not item_data['metadatas']:
                        continue
                    
                    metadata = item_data['metadatas'][0]
                    
                    # 사용자 특성
                    user_features = []
                    feature_names = ['danceability', 'energy', 'valence', 'tempo', 
                                   'instrumentalness', 'acousticness', 'liveness', 'speechiness']
                    
                    for feature in feature_names:
                        if feature in user_profile:
                            user_features.append(user_profile[feature])
                        else:
                            user_features.append(0.0)
                    
                    # 아이템 특성
                    item_features = []
                    for feature in feature_names:
                        if feature in metadata:
                            item_features.append(metadata[feature])
                        else:
                            item_features.append(0.0)
                    
                    # 특성 결합
                    combined_features = user_features + item_features
                    
                    # 정규화된 평점 (0-1 범위)
                    normalized_rating = min(rating / 5.0, 1.0)
                    
                    training_data.append((combined_features, normalized_rating))
            
            return training_data
            
        except Exception as e:
            print(f"❌ Wide&Deep 훈련 데이터 준비 중 오류: {e}")
            return []
    
    def _build_wide_deep_model(self, input_dim: int):
        """Wide&Deep 모델을 구축합니다."""
        # Wide 부분 (선형 모델)
        wide_input = keras.layers.Input(shape=(input_dim,))
        wide_output = keras.layers.Dense(1, activation='linear')(wide_input)
        
        # Deep 부분 (신경망)
        deep_input = keras.layers.Input(shape=(input_dim,))
        deep_output = keras.layers.Dense(self.hidden_dim, activation='relu')(deep_input)
        deep_output = keras.layers.Dropout(0.3)(deep_output)
        deep_output = keras.layers.Dense(self.hidden_dim // 2, activation='relu')(deep_output)
        deep_output = keras.layers.Dropout(0.3)(deep_output)
        deep_output = keras.layers.Dense(1, activation='sigmoid')(deep_output)
        
        # Wide와 Deep 결합
        combined_output = keras.layers.Add()([wide_output, deep_output])
        final_output = keras.layers.Activation('sigmoid')(combined_output)
        
        model = keras.Model(
            inputs=[wide_input, deep_input],
            outputs=final_output
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_recommendation_explanation(self, user_id: str, track_id: str) -> str:
        """추천 이유를 설명합니다."""
        if user_id not in self.user_preferences:
            return "사용자 선호도 정보가 없습니다."
        
        user_profile = self.get_user_profile(user_id)
        track_data = self.vector_db.collection.get(ids=[track_id])
        
        if not track_data['metadatas']:
            return "트랙 정보를 찾을 수 없습니다."
        
        metadata = track_data['metadatas'][0]
        explanation_parts = []
        
        # 오디오 특성 기반 설명
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
        for feature in audio_features:
            if feature in user_profile and feature in metadata:
                user_value = user_profile[feature]
                track_value = metadata[feature]
                
                if abs(user_value - track_value) < 0.2:
                    feature_name = {
                        'danceability': '춤추기 좋은',
                        'energy': '활기찬',
                        'valence': '긍정적인',
                        'acousticness': '어쿠스틱한',
                        'instrumentalness': '기악적인'
                    }.get(feature, feature)
                    
                    explanation_parts.append(f"{feature_name} 특성")
        
        # 장르/아티스트 기반 설명
        if 'artists' in metadata:
            user_artists = set()
            for track_id in self.user_preferences[user_id].keys():
                track_info = self.vector_db.collection.get(ids=[track_id])
                if track_info['metadatas']:
                    user_artists.update(track_info['metadatas'][0].get('artists', '').split(', '))
            
            track_artists = set(metadata['artists'].split(', '))
            if user_artists & track_artists:
                explanation_parts.append("좋아하는 아티스트와 유사")
        
        if explanation_parts:
            return f"이 곡을 추천하는 이유: {', '.join(explanation_parts)}"
        else:
            return "새로운 장르의 음악을 시도해보세요!"
    
    def save_user_preferences(self, filename: str = 'user_preferences.json'):
        """사용자 선호도를 파일로 저장합니다."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.user_preferences, f, ensure_ascii=False, indent=2)
        print(f"User preferences saved to {filename}")
    
    def load_user_preferences(self, filename: str = 'user_preferences.json'):
        """파일에서 사용자 선호도를 로드합니다."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.user_preferences = json.load(f)
            print(f"User preferences loaded from {filename}")
        except FileNotFoundError:
            print(f"Preferences file {filename} not found. Starting with empty preferences.")
            self.user_preferences = {}


class TwoTowerModel(nn.Module):
    """Two-Tower 모델: 사용자와 아이템을 별도의 인코더로 처리"""
    
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super(TwoTowerModel, self).__init__()
        
        # 사용자 타워 (User Tower)
        self.user_tower = nn.Sequential(
            nn.Linear(8, hidden_dim),  # 8개 오디오 특성
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # 아이템 타워 (Item Tower)
        self.item_tower = nn.Sequential(
            nn.Linear(8, hidden_dim),  # 8개 오디오 특성
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # 출력 레이어
        self.output_layer = nn.Linear(embedding_dim, 1)
    
    def forward(self, user_features, item_features):
        # 사용자와 아이템 임베딩 생성
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)
        
        # 코사인 유사도 계산
        user_norm = F.normalize(user_embedding, p=2, dim=1)
        item_norm = F.normalize(item_embedding, p=2, dim=1)
        
        # 내적 계산 (코사인 유사도와 동일)
        similarity = torch.sum(user_norm * item_norm, dim=1, keepdim=True)
        
        # 출력 레이어를 통과하여 최종 예측
        output = self.output_layer(similarity)
        
        return output


if __name__ == "__main__":
    # 사용 예시
    from .vector_database import MusicVectorDatabase
    from .spotify_collector import SpotifyMusicCollector
    
    # 초기화
    vector_db = MusicVectorDatabase()
    spotify_collector = SpotifyMusicCollector()
    recommender = MusicRecommender(vector_db, spotify_collector)
    
    # 사용자 선호도 추가 (예시)
    recommender.add_user_preference("user1", "track_id_1", 5.0)
    recommender.add_user_preference("user1", "track_id_2", 4.0)
    
    # 추천 받기
    print("\n=== 기본 하이브리드 추천 ===")
    recommendations = recommender.recommend_music("user1", n_recommendations=5, method="hybrid")
    
    print(f"Found {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"- {rec['name']} by {rec['artists']}")
        if 'method' in rec:
            print(f"  Method: {rec['method']}")
    
    print("\n=== Two-Stage 추천 시스템 ===")
    two_stage_recommendations = recommender.recommend_music("user1", n_recommendations=5, method="two_stage")
    
    print(f"Found {len(two_stage_recommendations)} two-stage recommendations:")
    for rec in two_stage_recommendations:
        print(f"- {rec['name']} by {rec['artists']}")
        if 'two_stage_score' in rec:
            print(f"  Two-Stage Score: {rec['two_stage_score']:.3f}")
        if 'method' in rec:
            print(f"  Method: {rec['method']}")
    
    # 선호도 저장
    recommender.save_user_preferences()
