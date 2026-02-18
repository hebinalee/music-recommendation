# 🎵 Spotify 음악 추천 시스템

Spotify API를 사용하여 음악 정보를 수집하고, Vector Database에 저장하여 사용자에게 개인화된 음악을 추천해주는 시스템입니다.

## ✨ 주요 기능

- **🎵 음악 데이터 수집**: Spotify API를 통한 다양한 소스에서 음악 정보 수집
- **🔍 Vector 검색**: Chroma DB를 사용한 의미 기반 음악 검색
- **🎯 개인화 추천**: 사용자 선호도 기반 음악 추천 시스템
- **📊 오디오 특성 분석**: Spotify의 오디오 특성을 활용한 정확한 추천
- **🚀 성능 최적화**: 캐싱 시스템, 배치 처리, 메모리 최적화
- **📈 모니터링**: 실시간 성능 메트릭 및 사용자 행동 추적
- **🧪 테스트**: 포괄적인 단위 테스트 및 통합 테스트
- **🔧 설정 관리**: 환경별 설정 파일 및 검증 로직

## 🏗️ 시스템 아키텍처

```
Spotify API → 음악 수집기 → Vector DB → 추천 엔진 → 사용자
     ↓              ↓           ↓         ↓
  음악 정보    오디오 특성   임베딩    추천 알고리즘
     ↓              ↓           ↓         ↓
  캐싱 시스템    성능 모니터링  검증 시스템  로깅 시스템
```

## 📁 프로젝트 구조

```
music/
├── main.py                    # 메인 실행 파일 (진입점)
├── example_usage.py           # 사용 예시 및 데모
├── run_tests.py              # 테스트 실행 스크립트
├── README.md                  # 프로젝트 문서
├── requirements.txt           # Python 의존성 목록
├── env_example.txt           # 환경 변수 설정 예시
├── src/                      # 소스 코드 패키지
│   ├── __init__.py           # 패키지 초기화
│   ├── music_recommender.py  # 메인 추천 시스템 클래스
│   ├── vector_database.py    # Chroma DB 벡터 데이터베이스 관리
│   ├── spotify_collector.py  # Spotify API 음악 데이터 수집
│   ├── ann_index.py          # ANN 인덱스 관리 (Two-Stage 추천용)
│   ├── config.py             # 설정 관리 모듈
│   ├── exceptions.py         # 에러 처리 모듈
│   ├── validators.py         # 데이터 검증 모듈
│   ├── logger.py             # 로깅 시스템 모듈
│   ├── cache.py              # 캐싱 시스템 모듈
│   └── monitoring.py         # 성능 모니터링 모듈
└── tests/                    # 테스트 코드
    ├── __init__.py           # 테스트 설정
    ├── test_config.py        # 설정 관리 테스트
    ├── test_validators.py    # 데이터 검증 테스트
    ├── test_exceptions.py    # 에러 처리 테스트
    └── test_spotify_collector.py # Spotify 수집기 테스트
```

### 주요 모듈 설명

- **`main.py`**: 통합된 메인 실행 프로그램 (CLI 인터페이스)
- **`src/music_recommender.py`**: 핵심 추천 시스템 클래스
  - 콘텐츠 기반 필터링
  - 협업 필터링  
  - 하이브리드 추천
  - Two-Stage 추천 (Two-Tower + Wide&Deep)
- **`src/vector_database.py`**: Chroma DB를 사용한 벡터 데이터베이스 관리
- **`src/spotify_collector.py`**: Spotify API를 통한 음악 데이터 수집
- **`src/ann_index.py`**: ANN 인덱스 관리 (Two-Stage 추천 시스템용)
- **`src/config.py`**: 설정 관리 및 환경 변수 검증
- **`src/exceptions.py`**: 사용자 친화적 에러 처리
- **`src/validators.py`**: 데이터 검증 및 정제
- **`src/logger.py`**: 구조화된 로깅 시스템
- **`src/cache.py`**: 메모리 및 파일 기반 캐싱
- **`src/monitoring.py`**: 성능 모니터링 및 사용자 행동 추적
- **`example_usage.py`**: 시스템 사용 예시 및 데모 코드

## 📋 요구사항

- Python 3.8+
- Spotify Developer Account
- 인터넷 연결
- 최소 4GB RAM (권장 8GB+)

## 🚀 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd recommender/music
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
# Spotify API 설정
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback

# 데이터베이스 설정
CHROMA_PERSIST_DIRECTORY=./chroma_db
COLLECTION_NAME=music_collection

# 모델 설정
MODEL_SAVE_DIR=./models
EMBEDDING_DIM=128
HIDDEN_DIM=256
LEARNING_RATE=0.001
BATCH_SIZE=32
EPOCHS=100

# 로깅 설정
LOG_LEVEL=INFO
LOG_DIR=./logs
```

## 🎮 사용 방법

### 기본 실행
```bash
python main.py
```

### 테스트 실행
```bash
python run_tests.py
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
   - Two-Stage 추천 (Two-Tower + Wide&Deep)

4. **사용자 선호도 관리**
   - 음악 평가 및 평점
   - 선호도 저장/로드

5. **데이터베이스 통계**
   - 저장된 음악 수 확인
   - 시스템 상태 모니터링

6. **ANN 인덱스 관리**
   - Two-Stage 모델용 인덱스 (재)빌드
   - FAISS 기반 빠른 검색 최적화

7. **Two-Stage 모델 훈련**
   - Two-Tower 모델 훈련
   - Wide&Deep 모델 훈련

8. **성능 모니터링**
   - 실시간 성능 메트릭
   - 사용자 행동 추적
   - 캐시 히트율 모니터링

## 🔧 고급 사용법

### Python 코드에서 직접 사용

```python
from src.spotify_collector import SpotifyMusicCollector
from src.vector_database import MusicVectorDatabase
from src.music_recommender import MusicRecommender
from src.config import ConfigManager
from src.monitoring import get_performance_monitor, get_user_behavior_tracker

# 설정 관리자 초기화
config_manager = ConfigManager()

# 컴포넌트 초기화
collector = SpotifyMusicCollector(config_manager)
vector_db = MusicVectorDatabase(config_manager=config_manager)
recommender = MusicRecommender(vector_db, collector)

# 성능 모니터링 시작
performance_monitor = get_performance_monitor()
performance_monitor.start_monitoring()

# 사용자 행동 추적
behavior_tracker = get_user_behavior_tracker()

# 음악 데이터 수집
music_data = collector.collect_music_data(search_queries=['k-pop', 'jazz'])

# Vector DB에 저장
vector_db.add_music_to_database(music_data)

# 사용자 선호도 추가
recommender.add_user_preference("user1", "track_id", 5.0)

# 추천 받기
recommendations = recommender.recommend_music("user1", method="hybrid")

# 성능 통계 확인
stats = performance_monitor.get_performance_summary()
print(f"평균 응답 시간: {stats['average_duration']:.3f}초")
print(f"성공률: {stats['success_rate']:.2%}")
```

### 캐싱 시스템 사용

```python
from src.cache import get_cache_manager, cached

cache_manager = get_cache_manager()

@cached(cache_manager)
def expensive_computation(data):
    # 시간이 오래 걸리는 계산
    return process_data(data)

# 첫 번째 호출: 계산 수행
result1 = expensive_computation("test_data")

# 두 번째 호출: 캐시에서 반환
result2 = expensive_computation("test_data")

# 캐시 통계 확인
cache_stats = cache_manager.get_stats()
print(f"캐시 히트율: {cache_stats['hit_rate']:.2%}")
```

### 성능 모니터링

```python
from src.monitoring import monitor_performance, get_performance_monitor

performance_monitor = get_performance_monitor()

@monitor_performance(performance_monitor)
def my_function():
    # 모니터링되는 함수
    return "result"

# 함수 실행 후 성능 메트릭 확인
result = my_function()
function_stats = performance_monitor.get_function_stats("my_function")
print(f"평균 실행 시간: {function_stats['avg_duration']:.3f}초")
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

### 4. Two-Stage 추천 시스템 (고급)
- **Two-Tower 모델**: 사용자와 아이템을 별도 인코더로 처리
- **Wide&Deep 모델**: 선형 모델과 딥러닝 모델 결합
- **ANN 인덱스**: FAISS를 사용한 빠른 유사도 검색
- 대규모 데이터베이스에서 효율적인 추천 성능

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

### 캐싱 시스템
- **메모리 캐시**: LRU 알고리즘 기반 빠른 접근
- **파일 캐시**: 영구 저장 및 만료 관리
- **자동 정리**: 만료된 캐시 자동 삭제

### 임베딩 모델
- **모델**: `all-MiniLM-L6-v2`
- **차원**: 384
- **속도**: 빠른 추론
- **정확도**: 높은 의미 이해

### 데이터베이스 최적화
- 중복 제거
- 배치 처리
- 인덱싱 최적화

## 🧪 테스트

### 테스트 실행
```bash
# 모든 테스트 실행
python run_tests.py

# 특정 테스트 모듈 실행
python -m unittest tests.test_config
python -m unittest tests.test_validators
python -m unittest tests.test_exceptions
python -m unittest tests.test_spotify_collector
```

### 테스트 커버리지
- 설정 관리 테스트
- 데이터 검증 테스트
- 에러 처리 테스트
- Spotify API 모킹 테스트
- 통합 테스트

## 📊 모니터링 및 로깅

### 성능 모니터링
- 함수별 실행 시간 추적
- 메모리 사용량 모니터링
- CPU 사용률 추적
- 시스템 리소스 모니터링

### 사용자 행동 추적
- 검색 패턴 분석
- 추천 클릭률 추적
- 사용자 선호도 변화 모니터링
- 인기 검색어 통계

### 로깅 시스템
- 구조화된 로그 포맷
- 파일 및 콘솔 출력
- 로그 레벨 관리
- 자동 로그 로테이션

## 🚨 주의사항

1. **API 제한**: Spotify API 호출 제한 준수
2. **저장 공간**: Vector DB는 시간이 지날수록 커질 수 있음
3. **인터넷 연결**: 실시간 데이터 수집을 위해 필요
4. **개인정보**: 사용자 선호도는 로컬에 저장
5. **메모리 사용량**: 대용량 데이터 처리 시 메모리 모니터링 필요

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
   - 캐시 크기 조정

4. **성능 저하**
   - 캐시 히트율 확인
   - 데이터베이스 인덱스 재구축
   - 시스템 리소스 모니터링

### 디버깅

```python
# 상세 로그 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# 데이터베이스 상태 확인
stats = vector_db.get_database_stats()
print(stats)

# 성능 통계 확인
performance_stats = performance_monitor.get_performance_summary()
print(performance_stats)

# 캐시 통계 확인
cache_stats = cache_manager.get_stats()
print(cache_stats)
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Commit your changes
6. Push to the branch
7. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- [Spotify Web API](https://developer.spotify.com/documentation/web-api/)
- [Chroma DB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)

## 📞 지원

문제가 있거나 질문이 있으시면 이슈를 생성해주세요.

---

**즐거운 음악 탐험 되세요! 🎵✨**

