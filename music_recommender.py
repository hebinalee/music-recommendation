import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from vector_database import MusicVectorDatabase
from spotify_collector import SpotifyMusicCollector
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from dotenv import load_dotenv

class MusicRecommender:
    def __init__(self, vector_db: MusicVectorDatabase, spotify_collector: SpotifyMusicCollector):
        """음악 추천 시스템을 초기화합니다."""
        self.vector_db = vector_db
        self.spotify_collector = spotify_collector
        self.user_preferences = {}
    
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
        if method == 'collaborative':
            return self._collaborative_filtering(user_id, n_recommendations)
        elif method == 'content_based':
            return self._content_based_filtering(user_id, n_recommendations)
        elif method == 'hybrid':
            return self._hybrid_recommendation(user_id, n_recommendations)
        else:
            raise ValueError("Method must be 'collaborative', 'content_based', or 'hybrid'")
    
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

if __name__ == "__main__":
    # 사용 예시
    from vector_database import MusicVectorDatabase
    from spotify_collector import SpotifyMusicCollector
    
    # 초기화
    vector_db = MusicVectorDatabase()
    spotify_collector = SpotifyMusicCollector()
    recommender = MusicRecommender(vector_db, spotify_collector)
    
    # 사용자 선호도 추가 (예시)
    recommender.add_user_preference("user1", "track_id_1", 5.0)
    recommender.add_user_preference("user1", "track_id_2", 4.0)
    
    # 추천 받기
    recommendations = recommender.recommend_music("user1", n_recommendations=5, method="hybrid")
    
    print(f"Found {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"- {rec['name']} by {rec['artists']}")
        if 'method' in rec:
            print(f"  Method: {rec['method']}")
    
    # 선호도 저장
    recommender.save_user_preferences()
