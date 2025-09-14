#!/usr/bin/env python3
"""
Spotify 음악 추천 시스템 사용 예시
"""

import os
from dotenv import load_dotenv
from src.spotify_collector import SpotifyMusicCollector
from src.vector_database import MusicVectorDatabase
from src.music_recommender import MusicRecommender
import pandas as pd

def main():
    """메인 예시 함수"""
    print("🎵 Spotify 음악 추천 시스템 사용 예시")
    print("=" * 50)
    
    # 환경 변수 로드
    load_dotenv()
    
    # Spotify API 인증 확인
    if not os.getenv('SPOTIFY_CLIENT_ID') or not os.getenv('SPOTIFY_CLIENT_SECRET'):
        print("❌ Spotify API 인증 정보가 설정되지 않았습니다.")
        print("env_example.txt 파일을 참고하여 .env 파일을 생성해주세요.")
        return
    
    try:
        # 1. 컴포넌트 초기화
        print("\n1️⃣ 시스템 컴포넌트 초기화...")
        spotify_collector = SpotifyMusicCollector()
        vector_db = MusicVectorDatabase()
        recommender = MusicRecommender(vector_db, spotify_collector)
        print("✅ 초기화 완료!")
        
        # 2. 음악 데이터 수집 예시
        print("\n2️⃣ 음악 데이터 수집 예시...")
        print("다양한 장르의 음악을 수집하고 있습니다...")
        
        # 여러 장르의 음악을 한 번에 수집
        genres = ['k-pop', 'jazz', 'rock', 'classical']
        total_tracks = 0
        
        for genre in genres:
            print(f"  📥 {genre} 장르 음악 수집 중...")
            music_data = spotify_collector.collect_music_data(search_queries=[genre])
            
            if not music_data.empty:
                print(f"    ✅ {len(music_data)}개 트랙 발견")
                
                # Vector DB에 저장
                if vector_db.add_music_to_database(music_data):
                    print(f"    💾 Vector DB에 저장 완료")
                    total_tracks += len(music_data)
                else:
                    print(f"    ❌ Vector DB 저장 실패")
                
                # CSV로도 저장
                filename = f"spotify_data_{genre.replace('-', '_')}.csv"
                spotify_collector.save_to_csv(music_data, filename)
                print(f"    📄 CSV 파일 저장: {filename}")
            else:
                print(f"    ❌ {genre} 장르에서 트랙을 찾을 수 없습니다")
        
        print(f"\n🎉 총 {total_tracks}개의 트랙을 수집했습니다!")
        
        # 3. 데이터베이스 통계 확인
        print("\n3️⃣ 데이터베이스 통계 확인...")
        stats = vector_db.get_database_stats()
        print(f"📊 총 저장된 트랙 수: {stats['total_tracks']}")
        print(f"📁 저장 경로: {stats['persist_directory']}")
        
        # 4. 음악 검색 예시
        print("\n4️⃣ 음악 검색 예시...")
        
        # 텍스트 기반 검색
        print("🔍 'energetic dance music' 검색 중...")
        search_results = vector_db.search_similar_music("energetic dance music", n_results=5)
        
        if search_results:
            print(f"✅ {len(search_results)}개의 결과를 찾았습니다:")
            for i, track in enumerate(search_results[:3], 1):
                print(f"  {i}. {track['name']} - {track['artists']}")
                print(f"     앨범: {track['album']}")
                print(f"     유사도 점수: {track['similarity_score']:.3f}")
        else:
            print("❌ 검색 결과가 없습니다")
        
        # 오디오 특성 기반 검색
        print("\n🎵 오디오 특성 기반 검색...")
        target_features = {
            'danceability': 0.8,
            'energy': 0.9,
            'valence': 0.7
        }
        print(f"목표 특성: {target_features}")
        
        feature_results = vector_db.search_by_audio_features(target_features, n_results=5)
        
        if feature_results:
            print(f"✅ {len(feature_results)}개의 결과를 찾았습니다:")
            for i, track in enumerate(feature_results[:3], 1):
                print(f"  {i}. {track['name']} - {track['artists']}")
                print(f"     앨범: {track['album']}")
                print(f"     특성 점수: {track['feature_score']:.3f}")
        else:
            print("❌ 특성 기반 검색 결과가 없습니다")
        
        # 5. 사용자 선호도 및 추천 예시
        print("\n5️⃣ 사용자 선호도 및 추천 예시...")
        
        # 가상의 사용자 선호도 추가
        print("👤 사용자 'music_lover'의 선호도 추가 중...")
        
        # 실제 트랙 ID가 필요하므로, 첫 번째 검색 결과 사용
        if search_results:
            sample_track_id = search_results[0]['id']
            recommender.add_user_preference("music_lover", sample_track_id, 5.0)
            print(f"✅ 트랙 '{search_results[0]['name']}'에 5점 평가 추가")
            
            # 더 많은 선호도 추가 (가상)
            if len(search_results) > 1:
                recommender.add_user_preference("music_lover", search_results[1]['id'], 4.0)
                print(f"✅ 트랙 '{search_results[1]['name']}'에 4점 평가 추가")
            
            # 사용자 프로필 생성
            print("\n👤 사용자 프로필 생성 중...")
            user_profile = recommender.get_user_profile("music_lover")
            if user_profile:
                print("사용자 음악 프로필:")
                for feature, value in user_profile.items():
                    print(f"  {feature}: {value:.3f}")
            
            # 추천 받기
            print("\n🎯 음악 추천 생성 중...")
            recommendations = recommender.recommend_music("music_lover", n_recommendations=5, method="hybrid")
            
            if recommendations:
                print(f"✅ {len(recommendations)}개의 추천을 생성했습니다:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec['name']} - {rec['artists']}")
                    print(f"     앨범: {rec['album']}")
                    
                    if 'method' in rec:
                        print(f"     추천 방식: {rec['method']}")
                    
                    # 추천 이유 설명
                    explanation = recommender.get_recommendation_explanation("music_lover", rec['id'])
                    print(f"     추천 이유: {explanation}")
            else:
                print("❌ 추천할 음악이 없습니다")
        
        # 6. 사용자 선호도 저장
        print("\n6️⃣ 사용자 선호도 저장...")
        recommender.save_user_preferences("example_user_preferences.json")
        print("✅ 사용자 선호도가 저장되었습니다")
        
        # 7. 시스템 요약
        print("\n" + "=" * 50)
        print("🎉 시스템 사용 예시 완료!")
        print("=" * 50)
        print("✅ 음악 데이터 수집 및 저장")
        print("✅ Vector DB 검색 기능")
        print("✅ 사용자 선호도 기반 추천")
        print("✅ 하이브리드 추천 알고리즘")
        print("✅ 추천 이유 설명")
        print("✅ 데이터 지속성 (저장/로드)")
        
        print("\n🚀 이제 main.py를 실행하여 전체 시스템을 사용해보세요!")
        print("📚 README.md에서 더 자세한 사용법을 확인할 수 있습니다")
        
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

def demo_simple_search():
    """간단한 검색 데모"""
    print("\n🔍 간단한 검색 데모")
    print("-" * 30)
    
    try:
        vector_db = MusicVectorDatabase()
        
        # 간단한 검색
        query = input("검색어를 입력하세요 (예: jazz, rock): ").strip()
        if query:
            results = vector_db.search_similar_music(query, n_results=5)
            
            if results:
                print(f"\n✅ '{query}'에 대한 {len(results)}개 결과:")
                for i, track in enumerate(results, 1):
                    print(f"{i}. {track['name']} - {track['artists']}")
                    print(f"   앨범: {track['album']}")
                    print(f"   점수: {track['similarity_score']:.3f}")
            else:
                print(f"❌ '{query}'에 대한 검색 결과가 없습니다")
        else:
            print("❌ 검색어를 입력해주세요")
    
    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")

if __name__ == "__main__":
    print("Spotify 음악 추천 시스템 예시")
    print("1. 전체 시스템 예시 실행")
    print("2. 간단한 검색 데모")
    
    choice = input("선택하세요 (1-2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        demo_simple_search()
    else:
        print("❌ 잘못된 선택입니다")
