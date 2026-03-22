#!/usr/bin/env python3
"""
Spotify 음악 추천 시스템 메인 실행 파일
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from dotenv import load_dotenv

from src.music_recommender import MusicRecommender
from src.vector_database import MusicVectorDatabase
from src.spotify_collector import SpotifyMusicCollector

def main():
    """메인 실행 함수"""
    print("🎵 Spotify 음악 추천 시스템을 시작합니다...")
    
    # 환경 변수 로드
    load_dotenv()
    
    # Spotify API 인증 확인
    if not os.getenv('SPOTIFY_CLIENT_ID') or not os.getenv('SPOTIFY_CLIENT_SECRET'):
        print("❌ Spotify API 인증 정보가 설정되지 않았습니다.")
        print("env_example.txt 파일을 참고하여 .env 파일을 생성해주세요.")
        return
    
    try:
        # 컴포넌트 초기화
        print("🔧 시스템 컴포넌트를 초기화합니다...")
        spotify_collector = SpotifyMusicCollector()
        vector_db = MusicVectorDatabase()
        recommender = MusicRecommender(vector_db, spotify_collector)
        
        # 사용자 선호도 로드
        recommender.load_user_preferences()
        
        print("✅ 시스템 초기화 완료!")
        
        # 메인 메뉴
        while True:
            print("\n" + "="*50)
            print("🎵 Spotify 음악 추천 시스템")
            print("="*50)
            print("1. 음악 데이터 수집")
            print("2. 음악 검색")
            print("3. 음악 추천 받기")
            print("4. 사용자 선호도 관리")
            print("5. 데이터베이스 통계")
            print("6. ANN 인덱스 (재)빌드")
            print("7. Two-Stage 모델 훈련")
            print("8. 종료")
            print("="*50)
            
            choice = input("원하는 기능을 선택하세요 (1-8): ").strip()
            
            if choice == '1':
                collect_music_data(spotify_collector, vector_db)
            elif choice == '2':
                search_music(vector_db)
            elif choice == '3':
                get_recommendations(recommender)
            elif choice == '4':
                manage_user_preferences(recommender)
            elif choice == '5':
                show_database_stats(vector_db)
            elif choice == '6':
                rebuild_ann_index(recommender)
            elif choice == '7':
                train_two_stage_models(recommender)
            elif choice == '8':
                print("👋 시스템을 종료합니다. 감사합니다!")
                break
            else:
                print("❌ 잘못된 선택입니다. 1-8 중에서 선택해주세요.")
    
    except Exception as e:
        print(f"❌ 시스템 오류가 발생했습니다: {e}")
        return

def collect_music_data(spotify_collector, vector_db):
    """음악 데이터 수집"""
    print("\n🎵 음악 데이터 수집")
    print("-" * 30)
    
    print("수집 방법을 선택하세요:")
    print("1. 검색어로 수집")
    print("2. 플레이리스트에서 수집")
    print("3. 아티스트의 인기곡에서 수집")
    
    method = input("선택 (1-3): ").strip()
    
    if method == '1':
        query = input("검색어를 입력하세요 (예: k-pop, jazz): ").strip()
        if query:
            print(f"'{query}'로 음악을 검색하고 있습니다...")
            music_data = spotify_collector.collect_music_data(search_queries=[query])
            
            if not music_data.empty:
                print(f"✅ {len(music_data)}개의 트랙을 찾았습니다.")
                
                # Vector DB에 저장
                if vector_db.add_music_to_database(music_data):
                    print("✅ Vector DB에 저장되었습니다.")
                    
                    # CSV로도 저장
                    filename = f"spotify_data_{query.replace(' ', '_')}.csv"
                    spotify_collector.save_to_csv(music_data, filename)
                else:
                    print("❌ Vector DB 저장에 실패했습니다.")
            else:
                print("❌ 검색 결과가 없습니다.")
    
    elif method == '2':
        playlist_id = input("플레이리스트 ID를 입력하세요: ").strip()
        if playlist_id:
            print(f"플레이리스트에서 음악을 수집하고 있습니다...")
            music_data = spotify_collector.collect_music_data(playlist_ids=[playlist_id])
            
            if not music_data.empty:
                print(f"✅ {len(music_data)}개의 트랙을 찾았습니다.")
                vector_db.add_music_to_database(music_data)
            else:
                print("❌ 플레이리스트를 찾을 수 없습니다.")
    
    elif method == '3':
        artist_id = input("아티스트 ID를 입력하세요: ").strip()
        if artist_id:
            print(f"아티스트의 인기곡을 수집하고 있습니다...")
            music_data = spotify_collector.collect_music_data(artist_ids=[artist_id])
            
            if not music_data.empty:
                print(f"✅ {len(music_data)}개의 트랙을 찾았습니다.")
                vector_db.add_music_to_database(music_data)
            else:
                print("❌ 아티스트를 찾을 수 없습니다.")

def search_music(vector_db):
    """음악 검색"""
    print("\n🔍 음악 검색")
    print("-" * 30)
    
    print("검색 방법을 선택하세요:")
    print("1. 텍스트 검색")
    print("2. 오디오 특성 기반 검색")
    
    method = input("선택 (1-2): ").strip()
    
    if method == '1':
        query = input("검색어를 입력하세요: ").strip()
        if query:
            n_results = int(input("결과 개수를 입력하세요 (기본값: 10): ") or "10")
            results = vector_db.search_similar_music(query, n_results)
            
            if results:
                print(f"\n✅ {len(results)}개의 결과를 찾았습니다:")
                for i, track in enumerate(results, 1):
                    print(f"{i}. {track['name']} - {track['artists']}")
                    print(f"   앨범: {track['album']}")
                    print(f"   유사도 점수: {track['similarity_score']:.3f}")
                    print()
            else:
                print("❌ 검색 결과가 없습니다.")
    
    elif method == '2':
        print("오디오 특성 값을 입력하세요 (0.0-1.0, 엔터로 건너뛰기):")
        features = {}
        
        feature_names = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
        for feature in feature_names:
            value = input(f"{feature}: ").strip()
            if value:
                try:
                    features[feature] = float(value)
                except ValueError:
                    print(f"❌ {feature}에 유효하지 않은 값입니다.")
        
        if features:
            n_results = int(input("결과 개수를 입력하세요 (기본값: 10): ") or "10")
            results = vector_db.search_by_audio_features(features, n_results)
            
            if results:
                print(f"\n✅ {len(results)}개의 결과를 찾았습니다:")
                for i, track in enumerate(results, 1):
                    print(f"{i}. {track['name']} - {track['artists']}")
                    print(f"   앨범: {track['album']}")
                    print(f"   특성 점수: {track['feature_score']:.3f}")
                    print()
            else:
                print("❌ 검색 결과가 없습니다.")

def get_recommendations(recommender):
    """음악 추천 받기"""
    print("\n🎯 음악 추천 받기")
    print("-" * 30)
    
    user_id = input("사용자 ID를 입력하세요: ").strip()
    if not user_id:
        print("❌ 사용자 ID가 필요합니다.")
        return
    
    print("추천 방법을 선택하세요:")
    print("1. 콘텐츠 기반 추천")
    print("2. 협업 필터링")
    print("3. 하이브리드 추천")
    print("4. Two-Stage 추천 (Two-Tower + Wide&Deep)")
    
    method = input("선택 (1-4): ").strip()
    
    method_map = {'1': 'content_based', '2': 'collaborative', '3': 'hybrid', '4': 'two_stage'}
    if method in method_map:
        n_results = int(input("추천 개수를 입력하세요 (기본값: 10): ") or "10")
        
        print(f"\n{method_map[method]} 방식으로 추천을 생성하고 있습니다...")
        recommendations = recommender.recommend_music(user_id, n_results, method_map[method])
        
        if recommendations:
            print(f"\n✅ {len(recommendations)}개의 추천을 찾았습니다:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['name']} - {rec['artists']}")
                print(f"   앨범: {rec['album']}")
                
                if 'method' in rec:
                    print(f"   추천 방식: {rec['method']}")
                
                if 'two_stage_score' in rec:
                    print(f"   Two-Stage 점수: {rec['two_stage_score']:.3f}")
                elif 'recommendation_score' in rec:
                    print(f"   추천 점수: {rec['recommendation_score']:.3f}")
                elif 'feature_score' in rec:
                    print(f"   특성 점수: {rec['feature_score']:.3f}")
                elif 'hybrid_score' in rec:
                    print(f"   하이브리드 점수: {rec['hybrid_score']:.3f}")
                
                # 추천 이유 설명
                explanation = recommender.get_recommendation_explanation(user_id, rec['id'])
                print(f"   추천 이유: {explanation}")
                print()
        else:
            print("❌ 추천할 음악이 없습니다. 더 많은 음악을 평가해보세요.")
    else:
        print("❌ 잘못된 선택입니다.")

def manage_user_preferences(recommender):
    """사용자 선호도 관리"""
    print("\n👤 사용자 선호도 관리")
    print("-" * 30)
    
    print("1. 음악 평가하기")
    print("2. 선호도 저장")
    print("3. 선호도 로드")
    
    choice = input("선택 (1-3): ").strip()
    
    if choice == '1':
        user_id = input("사용자 ID를 입력하세요: ").strip()
        track_id = input("트랙 ID를 입력하세요: ").strip()
        rating = input("평점을 입력하세요 (1.0-5.0): ").strip()
        
        if user_id and track_id and rating:
            try:
                rating = float(rating)
                if 1.0 <= rating <= 5.0:
                    recommender.add_user_preference(user_id, track_id, rating)
                    print("✅ 평가가 저장되었습니다.")
                else:
                    print("❌ 평점은 1.0-5.0 사이여야 합니다.")
            except ValueError:
                print("❌ 유효하지 않은 평점입니다.")
        else:
            print("❌ 모든 정보를 입력해주세요.")
    
    elif choice == '2':
        recommender.save_user_preferences()
    
    elif choice == '3':
        recommender.load_user_preferences()

def show_database_stats(vector_db):
    """데이터베이스 통계 표시"""
    print("\n📊 데이터베이스 통계")
    print("-" * 30)
    
    stats = vector_db.get_database_stats()
    if stats:
        print(f"총 트랙 수: {stats['total_tracks']}")
        print(f"저장 경로: {stats['persist_directory']}")
    else:
        print("❌ 통계 정보를 가져올 수 없습니다.")


def train_two_stage_models(recommender):
    """Two-Stage 추천 시스템 모델들을 훈련합니다."""
    print("\n🤖 Two-Stage 모델 훈련")
    print("-" * 30)
    
    print("훈련할 모델을 선택하세요:")
    print("1. Two-Tower 모델 (후보 생성용)")
    print("2. Wide&Deep 모델 (랭킹용)")
    print("3. 두 모델 모두 훈련")
    
    choice = input("선택 (1-3): ").strip()
    
    if choice == '1':
        print("🔄 Two-Tower 모델 훈련을 시작합니다...")
        recommender._train_two_tower_model()
    elif choice == '2':
        print("🔄 Wide&Deep 모델 훈련을 시작합니다...")
        recommender._train_wide_deep_model()
    elif choice == '3':
        print("🔄 두 모델 모두 훈련을 시작합니다...")
        recommender._train_two_tower_model()
        recommender._train_wide_deep_model()
        print("✅ 모든 모델 훈련 완료!")
    else:
        print("❌ 잘못된 선택입니다.")

def rebuild_ann_index(recommender):
    """ANN 인덱스를 (재)구축합니다."""
    print("\n⚙️ ANN 인덱스 (재)빌드")
    print("-" * 30)
    recommender._rebuild_ann_index()

if __name__ == "__main__":
    main()
