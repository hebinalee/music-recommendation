# 🎵 Spotify 음악 추천 시스템

Spotify API를 사용하여 음악 정보를 수집하고, Vector Database에 저장하여 사용자에게 개인화된 음악을 추천해주는 시스템입니다.

## ✨ 주요 기능

- **🎵 음악 데이터 수집**: Spotify API를 통한 다양한 소스에서 음악 정보 수집
- **🔍 Vector 검색**: Chroma DB를 사용한 의미 기반 음악 검색
- **🎯 개인화 추천**: 사용자 선호도 기반 음악 추천 시스템
- **📊 오디오 특성 분석**: Spotify의 오디오 특성을 활용한 정확한 추천

## 🏗️ 시스템 아키텍처

```
Spotify API → 음악 수집기 → Vector DB → 추천 엔진 → 사용자
     ↓              ↓           ↓         ↓
  음악 정보    오디오 특성   임베딩    추천 알고리즘
```

## 주요 파일들
1. spotify_collector.py - Spotify API를 통한 음악 데이터 수집
2. vector_database.py - Chroma DB를 사용한 Vector 데이터베이스 관리
3. music_recommender.py - 사용자 선호도 기반 음악 추천 엔진
4. main.py - 통합된 메인 실행 프로그램
5. example_usage.py - 시스템 사용 예시 및 데모
6. requirements.txt - 필요한 Python 패키지 목록
7. env_example.txt - 환경 변수 설정 예시
8. README.md - 상세한 사용법 및 문서

## 📋 요구사항

- Python 3.8+
- Spotify Developer Account
- 인터넷 연결

## 🚀 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd recommender
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. Spotify API 설정

1. [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)에 접속
2. 새 앱 생성
3. Client ID와 Client Secret 복사
4. `.env` 파일 생성:

```env
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

## 🎮 사용 방법

### 기본 실행
```bash
python main.py
```

### 프로그램 기능

1. **음악 데이터 수집**
   - 검색어로 음악 검색
   - 플레이리스트에서 음악 수집
   - 아티스트의 인기곡 수집

2. **음악 검색**
   - 텍스트 기반 의미 검색
   - 오디오 특성 기반 검색

3. **음악 추천 받기**
   - 콘텐츠 기반 추천
   - 협업 필터링
   - 하이브리드 추천

4. **사용자 선호도 관리**
   - 음악 평가 및 평점
   - 선호도 저장/로드

5. **데이터베이스 통계**
   - 저장된 음악 수 확인
   - 시스템 상태 모니터링

## 🔧 고급 사용법

### Python 코드에서 직접 사용

```python
from spotify_collector import SpotifyMusicCollector
from vector_database import MusicVectorDatabase
from music_recommender import MusicRecommender

# 컴포넌트 초기화
collector = SpotifyMusicCollector()
vector_db = MusicVectorDatabase()
recommender = MusicRecommender(vector_db, collector)

# 음악 데이터 수집
music_data = collector.collect_music_data(search_queries=['k-pop', 'jazz'])

# Vector DB에 저장
vector_db.add_music_to_database(music_data)

# 사용자 선호도 추가
recommender.add_user_preference("user1", "track_id", 5.0)

# 추천 받기
recommendations = recommender.recommend_music("user1", method="hybrid")
```

### 배치 데이터 수집

```python
# 다양한 장르의 음악을 한 번에 수집
genres = ['pop', 'rock', 'jazz', 'classical', 'electronic', 'hip-hop']
for genre in genres:
    music_data = collector.collect_music_data(search_queries=[genre])
    vector_db.add_music_to_database(music_data)
    print(f"Collected {len(music_data)} {genre} tracks")
```

## 📊 추천 알고리즘

### 1. 콘텐츠 기반 필터링
- 사용자가 좋아한 음악의 오디오 특성 분석
- 유사한 특성을 가진 음악 추천
- 장르, 아티스트, 음악적 특성 고려

### 2. 협업 필터링
- 다른 사용자들과의 유사도 계산
- 유사한 취향을 가진 사용자가 좋아한 음악 추천
- Jaccard 유사도 기반

### 3. 하이브리드 추천
- 콘텐츠 기반 + 협업 필터링 결합
- 더 정확하고 다양한 추천 결과
- 중복 제거 및 점수 통합

## 🗄️ 데이터베이스 구조

### Chroma Vector DB
- **Collection**: `music_collection`
- **Embeddings**: Sentence Transformers 기반 텍스트 임베딩
- **Metadata**: 음악 정보 + 오디오 특성
- **Persistence**: 로컬 디스크에 저장

### 저장되는 정보
- 기본 정보: 제목, 아티스트, 앨범, 발매일
- 오디오 특성: danceability, energy, valence, tempo 등
- 외부 링크: Spotify URL, 미리듣기 URL

## 🔍 검색 기능

### 텍스트 검색
```python
# 의미 기반 검색
results = vector_db.search_similar_music("energetic dance music", n_results=10)
```

### 오디오 특성 검색
```python
# 특정 특성을 가진 음악 검색
target_features = {
    'danceability': 0.8,
    'energy': 0.9,
    'valence': 0.7
}
results = vector_db.search_by_audio_features(target_features)
```

## 📈 성능 최적화

### 임베딩 모델
- **모델**: `all-MiniLM-L6-v2`
- **차원**: 384
- **속도**: 빠른 추론
- **정확도**: 높은 의미 이해

### 데이터베이스 최적화
- 중복 제거
- 배치 처리
- 인덱싱 최적화

## 🚨 주의사항

1. **API 제한**: Spotify API 호출 제한 준수
2. **저장 공간**: Vector DB는 시간이 지날수록 커질 수 있음
3. **인터넷 연결**: 실시간 데이터 수집을 위해 필요
4. **개인정보**: 사용자 선호도는 로컬에 저장

## 🐛 문제 해결

### 일반적인 오류

1. **Spotify API 인증 실패**
   - `.env` 파일 확인
   - Client ID/Secret 재확인

2. **의존성 설치 실패**
   - Python 버전 확인 (3.8+)
   - 가상환경 활성화 확인

3. **메모리 부족**
   - 배치 크기 줄이기
   - 불필요한 데이터 정리

### 디버깅

```python
# 상세 로그 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# 데이터베이스 상태 확인
stats = vector_db.get_database_stats()
print(stats)
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- [Spotify Web API](https://developer.spotify.com/documentation/web-api/)
- [Chroma DB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

## 📞 지원

문제가 있거나 질문이 있으시면 이슈를 생성해주세요.

---

**즐거운 음악 탐험 되세요! 🎵✨**
