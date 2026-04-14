#!/usr/bin/env python3
"""
자연어 입력 기반 음악 추천 리포트 생성기

사용법:
  python scripts/generate_report.py "이번 브루노마스 신곡 너무 좋더라 비슷한 거 없어?"
  python scripts/generate_report.py "잔잔한 재즈 감성" --n 15
  python scripts/generate_report.py          # 대화형 모드

Spotify Web API 제한 사항 (2024년 11월~):
  /audio-features, /related-artists → 일반 앱 차단
  → 아티스트 장르 Jaccard + 인기도 + 다양성 패널티 기반 Multi-signal Scoring 사용
"""

import os
import sys
import re
import json
import argparse
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import logging
from dotenv import load_dotenv
load_dotenv()

logging.getLogger("spotipy").setLevel(logging.CRITICAL)

from src.spotify_collector import SpotifyMusicCollector
from src.logger import get_logger

logger = get_logger(__name__)

REPORTS_DIR = _ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ── Two-Tower 선택적 import ──────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

# 지원 모델 목록
MODELS: Dict[str, Dict[str, str]] = {
    "multi-signal": {
        "name": "Multi-Signal Genre Ranker",
        "desc": "Seed-Based · Genre Jaccard + Popularity + Recency · Artist Diversity Penalty",
        "slug": "MultiSignal",
    },
    "genre-jaccard": {
        "name": "Genre-Jaccard Ranker",
        "desc": "Seed-Based · Genre Jaccard 0.6 + Popularity 0.4 (legacy)",
        "slug": "GenreJaccard",
    },
    "two-tower": {
        "name": "Two-Tower Item Ranker",
        "desc": "Item-Tower Embedding · FAISS ANN · Proxy Audio Features · Artist Diversity Penalty",
        "slug": "TwoTower",
    },
}
DEFAULT_MODEL = "multi-signal"


# ════════════════════════════════════════════════════
# 1. 자연어 파싱  (개선: 장르·무드·최신곡 시그널 분리)
# ════════════════════════════════════════════════════

# 한국어 장르명 → Spotify genre 쿼리 목록
_KO_TO_GENRE: Dict[str, List[str]] = {
    "재즈":        ["jazz", "smooth jazz", "jazz piano"],
    "클래식":      ["classical"],
    "힙합":        ["hip-hop", "rap"],
    "알앤비":      ["r&b", "soul"],
    "알비":        ["r&b"],
    "팝":          ["pop"],
    "케이팝":      ["k-pop"],
    "케팝":        ["k-pop"],
    "록":          ["rock"],
    "락":          ["rock"],
    "인디":        ["indie", "indie pop"],
    "발라드":      ["ballad"],
    "댄스":        ["dance pop"],
    "일렉":        ["electronic", "edm"],
    "일렉트로닉":  ["electronic", "edm"],
    "어쿠스틱":    ["acoustic"],
    "로파이":      ["lo-fi", "lofi hip hop"],
    "소울":        ["soul", "r&b"],
    "블루스":      ["blues"],
    "컨트리":      ["country"],
    "레게":        ["reggae"],
    "메탈":        ["metal"],
    "펑크":        ["funk"],
    "보사노바":    ["bossa nova"],
    "뉴에이지":    ["new age"],
    "앰비언트":    ["ambient"],
}

# 한국어 무드 → 영문 검색 어구
_KO_TO_MOOD: Dict[str, str] = {
    "잔잔한":   "calm mellow",
    "잔잔하고": "calm mellow",
    "신나는":   "upbeat energetic",
    "신나고":   "upbeat energetic",
    "슬픈":     "sad melancholy",
    "슬프고":   "sad melancholy",
    "감성적인": "emotional",
    "힐링":     "relaxing healing",
    "편안한":   "relaxing calm",
    "강렬한":   "intense powerful",
    "몽환적인": "dreamy ambient",
    "따뜻한":   "warm cozy",
    "차분한":   "calm chill",
    "업비트":   "upbeat",
    "에너지넘치는": "energetic",
}

# 최신곡 선호 감지 단어
_RECENCY_KW = {"신곡", "새로운", "최신", "최근", "요즘", "새 곡", "새곡"}

# 일반 불용어 (장르·무드 단어는 위 딕셔너리로 분리)
_STOPWORDS = {
    "이번", "노래", "음악", "곡", "좀", "좋은", "있어", "없어",
    "추천", "추천해", "추천해줘", "해줘", "해", "줘", "이랑", "랑",
    "같은", "비슷한", "느낌", "너무", "정말", "진짜", "완전",
    "좋더라", "좋아", "들어봤어", "들었는데",
    "아는", "알아", "찾아줘", "찾아", "주세요", "부탁해", "어때", "어떤",
    "듣고", "싶어", "싶은", "뭐", "뭔가", "스타일", "좀더", "더", "많이",
    "그냥", "한번", "한 번", "같이", "처럼", "없나", "있나", "없을까",
}


def parse_natural_query(text: str) -> Dict:
    """
    자연어에서 4가지 시그널을 분리합니다.

    Returns:
        keyword      : Spotify 기본 검색어 (아티스트명·곡명 등 고유명사)
        genre_queries: ['genre:"jazz"', ...] 형태의 Spotify 장르 쿼리
        mood_terms   : "calm mellow" 등 영문 무드 어구
        prefer_recent: True면 최신 발매 보너스 적용
    """
    cleaned = re.sub(r"[^\w\s가-힣]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens  = cleaned.split()

    prefer_recent = any(t in _RECENCY_KW for t in tokens)

    genre_queries: List[str] = []
    mood_parts:    List[str] = []
    ko_genre_words = set(_KO_TO_GENRE.keys())
    ko_mood_words  = set(_KO_TO_MOOD.keys())

    for tok in tokens:
        if tok in ko_genre_words:
            genre_queries.extend(_KO_TO_GENRE[tok])
        if tok in ko_mood_words:
            mood_parts.append(_KO_TO_MOOD[tok])

    # 중복 제거 (순서 유지)
    seen: set = set()
    genre_queries = [g for g in genre_queries if not (g in seen or seen.add(g))]  # type: ignore
    mood_terms = " ".join(dict.fromkeys(mood_parts))

    # 키워드: 불용어·장르어·무드어·최신어 제거 후 남은 토큰
    filtered = []
    for tok in tokens:
        if re.search(r"[가-힣]", tok):
            if (tok not in _STOPWORDS
                    and tok not in ko_genre_words
                    and tok not in ko_mood_words
                    and tok not in _RECENCY_KW
                    and len(tok) >= 2):
                filtered.append(tok)
        else:
            if len(tok) >= 2:
                filtered.append(tok)

    keyword = " ".join(filtered) if filtered else ""

    return {
        "keyword":      keyword,
        "genre_queries": genre_queries,
        "mood_terms":   mood_terms,
        "prefer_recent": prefer_recent,
    }


def make_filename_slug(text: str, max_len: int = 40) -> str:
    slug = re.sub(r'[\\/:*?"<>|]', "", text)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:max_len] if slug else "report"


# ════════════════════════════════════════════════════
# 2. Spotify 데이터 수집 헬퍼
# ════════════════════════════════════════════════════

def _get_artist_info(sp, artist_id: str) -> Dict:
    try:
        return sp.artist(artist_id)
    except Exception:
        return {}


def _get_artist_top_tracks(sp, artist_id: str, market: str = "US") -> List[Dict]:
    try:
        return sp.artist_top_tracks(artist_id, country=market).get("tracks", [])
    except Exception:
        return []


def _normalize_track(raw: Dict) -> Dict:
    artists_raw = raw.get("artists", [])
    album_raw   = raw.get("album", {})
    album_name  = album_raw.get("name", "") if isinstance(album_raw, dict) else str(album_raw)
    release     = album_raw.get("release_date", "") if isinstance(album_raw, dict) else ""
    return {
        "id":            raw.get("id", ""),
        "name":          raw.get("name", "Unknown"),
        "artists":       [a["name"] for a in artists_raw],
        "artist_ids":    [a["id"]   for a in artists_raw],
        "album":         album_name,
        "release_date":  release,
        "popularity":    raw.get("popularity", 0),
        "duration_ms":   raw.get("duration_ms", 0),
        "explicit":      raw.get("explicit", False),
        "external_urls": raw.get("external_urls", {}).get("spotify", "#"),
        "preview_url":   raw.get("preview_url", None),
        "genres":        [],
    }


def _normalize_title(name: str) -> str:
    """곡명에서 버전 표시·특수문자 제거 후 소문자 반환 (씨앗 중복 판단용)"""
    name = re.sub(r"\(.*?\)|\[.*?\]", "", name)
    name = re.sub(r"[^\w\s]", "", name)
    return name.lower().strip()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ════════════════════════════════════════════════════
# 3. 씨앗 탐색  (개선: 중복 제거 + 아티스트 다양성)
# ════════════════════════════════════════════════════

def find_seed_tracks(
    collector: SpotifyMusicCollector,
    parsed: Dict,
    n_seeds: int = 3,
) -> Tuple[List[Dict], Set[str], List[str]]:
    """
    씨앗 트랙 선택 전략:
    1. 키워드 + 장르 쿼리로 Spotify 검색
    2. (정규화 제목 + 주아티스트) 기준 중복 제거  ← Dynamite 버전 중복 방지
    3. 아티스트 다양성 우선 선택
    """
    sp      = collector.sp
    keyword = parsed["keyword"]
    genre_q = parsed["genre_queries"]
    mood    = parsed["mood_terms"]

    # 검색 쿼리 우선순위: 키워드 → 장르 → 무드+장르명(텍스트) → 무드만
    # ※ genre:"X" 필터와 mood를 동시 사용하면 결과가 극히 적어짐 → 분리
    candidate_search_qs = []
    if keyword:
        candidate_search_qs.append(keyword)
    if genre_q:
        # 장르 필터만 단독 사용 (결과 풍부)
        candidate_search_qs.append(f'genre:"{genre_q[0]}"')
        # 무드 어구 + 장르명 텍스트 (필터 아닌 일반 검색)
        if mood:
            candidate_search_qs.append(f"{mood} {genre_q[0]}")
    if mood and not candidate_search_qs:
        candidate_search_qs.append(mood)

    if not candidate_search_qs:
        raise ValueError("검색어를 추출할 수 없습니다.")

    raw_tracks: List[Dict] = []
    for search_q in candidate_search_qs:
        print(f"\n[1/3] Spotify 검색 중: '{search_q}'")
        raw_result = sp.search(q=search_q, type="track", limit=20)
        raw_tracks = raw_result.get("tracks", {}).get("items", [])
        if raw_tracks:
            break

    if not raw_tracks:
        raise ValueError(f"검색 결과가 없습니다. 다른 키워드를 시도해보세요.")

    raw_tracks.sort(key=lambda t: t.get("popularity", 0), reverse=True)

    # ── 중복 제거: (정규화 제목, 주아티스트) 기준 ──
    seen_keys: Set[tuple] = set()
    deduped: List[Dict] = []
    for raw in raw_tracks:
        artists = raw.get("artists", [])
        primary = artists[0]["name"].lower() if artists else ""
        key = (_normalize_title(raw.get("name", "")), primary)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(raw)

    # ── 아티스트 다양성 우선 선택 ──
    seen_artists: Set[str] = set()
    selected: List[Dict] = []
    remainder: List[Dict] = []
    for raw in deduped:
        artists = raw.get("artists", [])
        primary = artists[0]["name"].lower() if artists else ""
        if primary not in seen_artists:
            seen_artists.add(primary)
            selected.append(raw)
        else:
            remainder.append(raw)
        if len(selected) == n_seeds:
            break
    if len(selected) < n_seeds:
        selected.extend(remainder[:n_seeds - len(selected)])

    # ── 장르 정보 수집 ──
    seeds:           List[Dict] = []
    seed_genres:     Set[str]   = set()
    seed_artist_ids: List[str]  = []

    for raw in selected:
        track = _normalize_track(raw)
        if track["artist_ids"]:
            pid = track["artist_ids"][0]
            if pid not in seed_artist_ids:
                seed_artist_ids.append(pid)
            info   = _get_artist_info(sp, pid)
            genres = info.get("genres", [])
            track["genres"] = genres
            seed_genres.update(genres)
        seeds.append(track)

    print(f"  씨앗 곡 {len(seeds)}개 선택:")
    for t in seeds:
        g_str = ", ".join(t["genres"][:3]) or "장르 정보 없음"
        print(f"  - {t['name']} by {', '.join(t['artists'])}  [{g_str}]")
    print(f"  씨앗 장르: {', '.join(sorted(seed_genres)[:8]) or '없음'}")

    return seeds, seed_genres, seed_artist_ids


# ════════════════════════════════════════════════════
# 4. 후보 수집 + Multi-signal Scoring
# ════════════════════════════════════════════════════

def _build_candidate_queries(parsed: Dict, seed_genres: Set[str]) -> List[str]:
    """
    후보 트랙 수집용 검색 쿼리를 구성합니다.

    우선순위:
    1. 명시적 장르 쿼리  genre:"jazz"          ← 가장 정확
    2. 무드 + 장르 조합  calm mellow jazz
    3. 씨앗 아티스트 장르 genre:"k-pop"
    4. 키워드 폴백
    """
    queries: List[str] = []
    genre_qs  = parsed["genre_queries"]
    mood      = parsed["mood_terms"]
    keyword   = parsed["keyword"]

    # 1) 명시적 장르 (단독 - 결과 풍부)
    for g in genre_qs[:3]:
        queries.append(f'genre:"{g}"')
    # 무드 + 장르명 텍스트 조합 (genre: 필터 없이)
    if mood and genre_qs:
        queries.append(f"{mood} {genre_qs[0]}")

    # 2) 씨앗 장르 (명시 장르와 겹치지 않는 것만)
    explicit_genres = set(genre_qs)
    for g in sorted(seed_genres):
        if g not in explicit_genres:
            queries.append(f'genre:"{g}"')
        if len(queries) >= 6:
            break

    # 3) 키워드 + 무드 폴백 (장르 쿼리가 없거나 키워드가 있을 때)
    if keyword:
        queries.append(f"{keyword} {mood}".strip() if mood else keyword)

    # 4) 장르·키워드가 전혀 없으면 무드만이라도
    if not queries and mood:
        queries.append(mood)

    return queries


def _score_track(
    track: Dict,
    seed_genres: Set[str],
    seed_artist_names: Set[str],
    prefer_recent: bool,
) -> float:
    """
    Multi-signal Scoring (개선):

    Score = 0.50 × genre_jaccard
          + 0.30 × popularity_norm
          + 0.10 × recency_bonus      (prefer_recent=True & 2024년 이후)
          - 0.40 × seed_artist_penalty (씨앗 아티스트 본인 곡)

    씨앗 아티스트 패널티가 핵심: "브루노마스" 쿼리 시 Bruno Mars 곡이
    추천 상위를 독점하는 현상 방지.
    """
    track_genres = set(track.get("genres", []))
    genre_sim    = _jaccard(seed_genres, track_genres) if seed_genres else 0.0
    pop_norm     = track.get("popularity", 0) / 100.0

    # 씨앗 아티스트 패널티
    track_artists = {a.lower() for a in track.get("artists", [])}
    artist_penalty = 0.40 if track_artists & seed_artist_names else 0.0

    # 최신곡 보너스
    recency_bonus = 0.0
    if prefer_recent:
        release = track.get("release_date", "")
        if release and release[:4] >= "2024":
            recency_bonus = 0.10

    score = 0.50 * genre_sim + 0.30 * pop_norm + recency_bonus - artist_penalty
    return max(0.0, round(score, 4))


def _enforce_artist_diversity(tracks: List[Dict], max_per_artist: int = 2) -> List[Dict]:
    """
    최종 추천 목록에서 아티스트당 최대 max_per_artist 곡으로 하드캡을 적용합니다.
    씨앗 아티스트 패널티 이후에도 남은 동일 아티스트 과잉을 제거합니다.
    """
    counts: Dict[str, int] = {}
    result: List[Dict] = []
    for t in tracks:
        primary = t["artists"][0].lower() if t.get("artists") else "unknown"
        if counts.get(primary, 0) < max_per_artist:
            counts[primary] = counts.get(primary, 0) + 1
            result.append(t)
    return result


def find_similar_tracks(
    collector: SpotifyMusicCollector,
    parsed: Dict,
    seed_genres: Set[str],
    seed_artist_ids: List[str],
    seed_ids: Set[str],
    n_results: int = 20,
) -> List[Dict]:
    """
    후보 수집 → 장르 정보 보강 → Multi-signal Scoring → 아티스트 다양성 적용
    """
    print(f"\n[2/3] 유사 트랙 탐색 중...")
    sp = collector.sp

    seed_artist_names: Set[str] = set()
    for aid in seed_artist_ids:
        info = _get_artist_info(sp, aid)
        if info.get("name"):
            seed_artist_names.add(info["name"].lower())

    candidate_map: Dict[str, Dict] = {}
    queries = _build_candidate_queries(parsed, seed_genres)
    print(f"  검색 쿼리 {len(queries)}개: {queries}")

    for q in queries:
        raw_result = sp.search(q=q, type="track", limit=30)
        for raw in raw_result.get("tracks", {}).get("items", []):
            t = _normalize_track(raw)
            if t["id"] and t["id"] not in seed_ids and t["id"] not in candidate_map:
                candidate_map[t["id"]] = t

    print(f"  후보 트랙 {len(candidate_map)}개 수집 완료, 장르 정보 보강 중...")

    # 장르 정보 없는 후보에 대해 아티스트 API 호출
    for t in candidate_map.values():
        if not t["genres"] and t["artist_ids"]:
            info = _get_artist_info(sp, t["artist_ids"][0])
            t["genres"] = info.get("genres", [])

    # Scoring
    prefer_recent = parsed.get("prefer_recent", False)
    scored: List[Dict] = []
    for track in candidate_map.values():
        s = _score_track(track, seed_genres, seed_artist_names, prefer_recent)
        scored.append({**track, "similarity_score": s})

    scored.sort(key=lambda x: x["similarity_score"], reverse=True)

    # 아티스트 다양성 하드캡 적용
    diverse = _enforce_artist_diversity(scored, max_per_artist=2)
    return diverse[:n_results]


# ════════════════════════════════════════════════════
# 5-a. Genre-Jaccard 모델 (legacy)
# ════════════════════════════════════════════════════

def _score_track_genre_jaccard(track: Dict, seed_genres: Set[str]) -> float:
    """Legacy scoring: 0.6 × Jaccard + 0.4 × popularity_norm (아티스트 패널티 없음)"""
    track_genres = set(track.get("genres", []))
    genre_sim    = _jaccard(seed_genres, track_genres) if seed_genres else 0.0
    pop_norm     = track.get("popularity", 0) / 100.0
    return max(0.0, round(0.6 * genre_sim + 0.4 * pop_norm, 4))


def find_similar_tracks_genre_jaccard(
    collector: SpotifyMusicCollector,
    parsed: Dict,
    seed_genres: Set[str],
    seed_artist_ids: List[str],
    seed_ids: Set[str],
    n_results: int = 20,
) -> List[Dict]:
    """Genre-Jaccard 모델: 씨앗 장르 Jaccard 0.6 + 인기도 0.4, 아티스트 다양성 처리 없음"""
    print(f"\n[2/3] [Genre-Jaccard] 유사 트랙 탐색 중...")
    sp = collector.sp

    candidate_map: Dict[str, Dict] = {}
    queries = _build_candidate_queries(parsed, seed_genres)
    print(f"  검색 쿼리 {len(queries)}개: {queries}")

    for q in queries:
        raw_result = sp.search(q=q, type="track", limit=30)
        for raw in raw_result.get("tracks", {}).get("items", []):
            t = _normalize_track(raw)
            if t["id"] and t["id"] not in seed_ids and t["id"] not in candidate_map:
                candidate_map[t["id"]] = t

    print(f"  후보 트랙 {len(candidate_map)}개 수집, 장르 정보 보강 중...")
    for t in candidate_map.values():
        if not t["genres"] and t["artist_ids"]:
            info = _get_artist_info(sp, t["artist_ids"][0])
            t["genres"] = info.get("genres", [])

    scored: List[Dict] = []
    for track in candidate_map.values():
        s = _score_track_genre_jaccard(track, seed_genres)
        scored.append({**track, "similarity_score": s})

    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored[:n_results]


# ════════════════════════════════════════════════════
# 5-b. Two-Tower 모델 (고급 — 학습된 모델 필요)
# ════════════════════════════════════════════════════

# 장르 → 프록시 오디오 피처 (8차원)
# 순서: [danceability, energy, valence, tempo_norm, instrumentalness, acousticness, liveness, speechiness]
_GENRE_PROXY: Dict[str, List[float]] = {
    "pop":            [0.75, 0.70, 0.75, 0.55, 0.02, 0.20, 0.12, 0.06],
    "k-pop":          [0.75, 0.75, 0.70, 0.60, 0.02, 0.15, 0.12, 0.07],
    "dance pop":      [0.85, 0.80, 0.75, 0.65, 0.02, 0.10, 0.10, 0.05],
    "hip-hop":        [0.80, 0.70, 0.55, 0.55, 0.05, 0.10, 0.10, 0.25],
    "rap":            [0.80, 0.70, 0.50, 0.55, 0.03, 0.08, 0.10, 0.35],
    "r&b":            [0.72, 0.60, 0.65, 0.50, 0.03, 0.25, 0.12, 0.08],
    "soul":           [0.65, 0.55, 0.65, 0.45, 0.05, 0.30, 0.15, 0.07],
    "jazz":           [0.55, 0.50, 0.60, 0.45, 0.35, 0.45, 0.20, 0.05],
    "smooth jazz":    [0.45, 0.40, 0.60, 0.40, 0.50, 0.50, 0.10, 0.03],
    "jazz piano":     [0.40, 0.35, 0.55, 0.38, 0.60, 0.55, 0.08, 0.02],
    "classical":      [0.25, 0.30, 0.50, 0.40, 0.85, 0.75, 0.10, 0.02],
    "rock":           [0.55, 0.82, 0.55, 0.65, 0.05, 0.15, 0.18, 0.06],
    "indie":          [0.58, 0.65, 0.60, 0.55, 0.10, 0.30, 0.15, 0.05],
    "indie pop":      [0.65, 0.65, 0.65, 0.55, 0.08, 0.28, 0.13, 0.05],
    "electronic":     [0.78, 0.82, 0.60, 0.70, 0.40, 0.08, 0.10, 0.05],
    "edm":            [0.85, 0.90, 0.65, 0.75, 0.35, 0.05, 0.12, 0.04],
    "metal":          [0.40, 0.95, 0.35, 0.80, 0.10, 0.05, 0.15, 0.06],
    "ballad":         [0.40, 0.35, 0.45, 0.38, 0.05, 0.55, 0.10, 0.04],
    "acoustic":       [0.45, 0.40, 0.60, 0.40, 0.10, 0.80, 0.12, 0.05],
    "lo-fi":          [0.55, 0.35, 0.55, 0.40, 0.45, 0.60, 0.08, 0.03],
    "lofi hip hop":   [0.60, 0.38, 0.55, 0.40, 0.50, 0.55, 0.08, 0.05],
    "ambient":        [0.30, 0.25, 0.50, 0.30, 0.75, 0.65, 0.08, 0.02],
    "new age":        [0.30, 0.25, 0.60, 0.32, 0.70, 0.70, 0.07, 0.02],
    "bossa nova":     [0.55, 0.40, 0.70, 0.40, 0.30, 0.60, 0.15, 0.04],
    "blues":          [0.50, 0.55, 0.45, 0.45, 0.15, 0.40, 0.20, 0.06],
    "country":        [0.55, 0.55, 0.65, 0.48, 0.05, 0.55, 0.18, 0.07],
    "reggae":         [0.70, 0.55, 0.75, 0.45, 0.10, 0.40, 0.18, 0.08],
    "funk":           [0.80, 0.70, 0.75, 0.58, 0.10, 0.20, 0.15, 0.08],
}
_DEFAULT_PROXY = [0.55, 0.55, 0.55, 0.50, 0.10, 0.30, 0.12, 0.06]


def _make_proxy_features(track: Dict) -> List[float]:
    """
    장르 태그 + 인기도로 8차원 오디오 피처를 추정합니다.
    (Spotify /audio-features API가 2024년 11월부터 일반 앱 차단되어 프록시 사용)

    반환 순서: [danceability, energy, valence, tempo_norm,
                instrumentalness, acousticness, liveness, speechiness]
    """
    genres  = [g.lower() for g in track.get("genres", [])]
    pop     = track.get("popularity", 50) / 100.0

    matched: List[List[float]] = []
    for g in genres:
        if g in _GENRE_PROXY:
            matched.append(_GENRE_PROXY[g])
        else:
            # 부분 일치 (예: "korean jazz")
            for key, vec in _GENRE_PROXY.items():
                if key in g or g in key:
                    matched.append(vec)
                    break

    if not matched:
        base = list(_DEFAULT_PROXY)
    else:
        n = len(matched)
        base = [sum(matched[j][i] for j in range(n)) / n for i in range(8)]

    # 인기도가 높을수록 danceability·energy 약간 상향 (보정)
    base[0] = min(1.0, base[0] * (0.85 + 0.30 * pop))
    base[1] = min(1.0, base[1] * (0.85 + 0.30 * pop))
    return base


if _TORCH_AVAILABLE:
    import torch.nn as _nn

    class _TwoTowerModel(_nn.Module):
        """
        music_recommender.py의 TwoTowerModel과 동일한 아키텍처 (추론 전용).
        학습 없음 — 저장된 가중치(.pth) 로드용.
        """
        def __init__(self, input_dim: int = 8, embedding_dim: int = 128, hidden_dim: int = 256):
            super().__init__()
            drop = 0.2

            def _tower(in_dim: int) -> _nn.Sequential:
                return _nn.Sequential(
                    _nn.Linear(in_dim,         hidden_dim),    _nn.ReLU(), _nn.Dropout(drop),
                    _nn.Linear(hidden_dim,     hidden_dim // 2), _nn.ReLU(), _nn.Dropout(drop),
                    _nn.Linear(hidden_dim // 2, embedding_dim),
                )

            self.user_tower = _tower(input_dim)
            self.item_tower = _tower(input_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":   # type: ignore[name-defined]
            return F.normalize(self.item_tower(x), dim=-1)

        def encode_item(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            self.eval()
            with torch.no_grad():
                return F.normalize(self.item_tower(x), dim=-1)


_MODELS_DIR = _ROOT / "models"


def _load_two_tower() -> Optional[object]:
    """학습된 Two-Tower 모델과 FAISS 인덱스를 로드합니다. 실패 시 None 반환."""
    if not _TORCH_AVAILABLE:
        print("  [Two-Tower] PyTorch 미설치 → Multi-Signal 폴백")
        return None
    if not _FAISS_AVAILABLE:
        print("  [Two-Tower] FAISS 미설치 → Multi-Signal 폴백")
        return None

    model_path   = _MODELS_DIR / "two_tower_model.pth"
    index_path   = _MODELS_DIR / "faiss_index.bin"
    mapping_path = _MODELS_DIR / "faiss_id_map.json"

    for p in (model_path, index_path, mapping_path):
        if not p.exists():
            print(f"  [Two-Tower] 파일 없음: {p.name} → Multi-Signal 폴백")
            print("  힌트: python scripts/main.py 에서 메뉴 6(Two-Tower 학습) 먼저 실행하세요.")
            return None

    try:
        model = _TwoTowerModel()
        state = torch.load(str(model_path), map_location="cpu")
        # 저장된 state_dict에 'two_tower_model.' 접두사가 있을 수 있음
        if any(k.startswith("two_tower_model.") for k in state):
            state = {k.replace("two_tower_model.", ""): v for k, v in state.items()
                     if k.startswith("two_tower_model.")}
        model.load_state_dict(state, strict=False)
        model.eval()

        index = _faiss.read_index(str(index_path))
        with open(mapping_path, "r", encoding="utf-8") as f:
            id_map: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}

        print("  [Two-Tower] 모델 + FAISS 인덱스 로드 완료")
        return {"model": model, "index": index, "id_map": id_map}
    except Exception as e:
        print(f"  [Two-Tower] 로드 실패: {e} → Multi-Signal 폴백")
        return None


def find_similar_tracks_two_tower(
    collector: SpotifyMusicCollector,
    parsed: Dict,
    seed_tracks: List[Dict],
    seed_genres: Set[str],
    seed_artist_ids: List[str],
    seed_ids: Set[str],
    n_results: int = 20,
) -> List[Dict]:
    """
    Two-Tower Item Ranker:
    1. 씨앗 트랙의 프록시 오디오 피처 → item_tower 인코딩 → 평균 쿼리 벡터
    2. FAISS ANN 검색 → 후보 track_id 목록
    3. Spotify 메타데이터 Fetch → 장르 보강
    4. Artist Diversity Penalty 적용
    학습된 모델이 없으면 Multi-Signal로 자동 폴백.
    """
    print(f"\n[2/3] [Two-Tower] 유사 트랙 탐색 중...")
    artifacts = _load_two_tower()

    # 폴백: 학습된 모델 없으면 Multi-Signal 실행
    if artifacts is None:
        print("  → Multi-Signal 폴백 실행")
        return find_similar_tracks(
            collector, parsed, seed_genres, seed_artist_ids, seed_ids, n_results
        )

    model: _TwoTowerModel = artifacts["model"]    # type: ignore[assignment]
    index = artifacts["index"]
    id_map: Dict[int, str] = artifacts["id_map"]
    sp = collector.sp

    # ── 씨앗 임베딩 생성 ──
    seed_vecs: List[List[float]] = []
    for t in seed_tracks:
        feat = _make_proxy_features(t)
        seed_vecs.append(feat)

    if not seed_vecs:
        return find_similar_tracks(
            collector, parsed, seed_genres, seed_artist_ids, seed_ids, n_results
        )

    feat_tensor = torch.tensor(seed_vecs, dtype=torch.float32)
    embeddings  = model.encode_item(feat_tensor)           # (n_seeds, embed_dim)
    query_vec   = embeddings.mean(dim=0, keepdim=True).numpy()  # (1, embed_dim)

    # ── FAISS 검색 ──
    top_k    = min(n_results * 5, index.ntotal)
    scores_arr, indices_arr = index.search(query_vec, top_k)
    scores_list  = scores_arr[0].tolist()
    indices_list = indices_arr[0].tolist()

    candidate_ids: List[Tuple[str, float]] = []
    for idx, score in zip(indices_list, scores_list):
        if idx < 0:
            continue
        track_id = id_map.get(idx)
        if track_id and track_id not in seed_ids:
            candidate_ids.append((track_id, float(score)))

    print(f"  FAISS 후보 {len(candidate_ids)}개 검색 완료")

    # ── Spotify 메타데이터 Fetch (50개 단위 batch) ──
    tracks_meta: List[Dict] = []
    batch_size = 50
    id_score_map: Dict[str, float] = {tid: s for tid, s in candidate_ids}

    ids_only = [tid for tid, _ in candidate_ids[:n_results * 3]]
    for i in range(0, len(ids_only), batch_size):
        batch = ids_only[i:i + batch_size]
        try:
            result = sp.tracks(batch)
            for raw in result.get("tracks", []):
                if raw:
                    t = _normalize_track(raw)
                    t["similarity_score"] = round(id_score_map.get(t["id"], 0.0), 4)
                    tracks_meta.append(t)
        except Exception as e:
            logger.warning(f"Batch fetch 실패: {e}")

    # 장르 정보 보강
    for t in tracks_meta:
        if not t["genres"] and t["artist_ids"]:
            info = _get_artist_info(sp, t["artist_ids"][0])
            t["genres"] = info.get("genres", [])

    # FAISS score 내림차순 정렬
    tracks_meta.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

    # 씨앗 아티스트 패널티 + 다양성 하드캡
    seed_artist_names: Set[str] = set()
    for aid in seed_artist_ids:
        info = _get_artist_info(sp, aid)
        if info.get("name"):
            seed_artist_names.add(info["name"].lower())

    penalized: List[Dict] = []
    for t in tracks_meta:
        track_artists = {a.lower() for a in t.get("artists", [])}
        if track_artists & seed_artist_names:
            t["similarity_score"] = max(0.0, t["similarity_score"] - 0.4)
        penalized.append(t)

    penalized.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
    diverse = _enforce_artist_diversity(penalized, max_per_artist=2)

    # FAISS 결과가 부족하면 Multi-Signal로 보충
    if len(diverse) < n_results // 2:
        print(f"  FAISS 결과 부족({len(diverse)}개) → Multi-Signal로 보충")
        extra = find_similar_tracks(
            collector, parsed, seed_genres, seed_artist_ids,
            seed_ids | {t["id"] for t in diverse}, n_results - len(diverse)
        )
        diverse.extend(extra)

    return diverse[:n_results]


# ════════════════════════════════════════════════════
# 5-c. 모델 디스패처
# ════════════════════════════════════════════════════

def find_recommendations(
    collector: SpotifyMusicCollector,
    parsed: Dict,
    seed_tracks: List[Dict],
    seed_genres: Set[str],
    seed_artist_ids: List[str],
    seed_ids: Set[str],
    n_results: int = 20,
    model_key: str = DEFAULT_MODEL,
) -> List[Dict]:
    """--model 파라미터에 따라 올바른 추천 함수로 라우팅합니다."""
    if model_key == "genre-jaccard":
        return find_similar_tracks_genre_jaccard(
            collector, parsed, seed_genres, seed_artist_ids, seed_ids, n_results
        )
    elif model_key == "two-tower":
        return find_similar_tracks_two_tower(
            collector, parsed, seed_tracks, seed_genres, seed_artist_ids, seed_ids, n_results
        )
    else:  # "multi-signal" (기본값)
        return find_similar_tracks(
            collector, parsed, seed_genres, seed_artist_ids, seed_ids, n_results
        )


# ════════════════════════════════════════════════════
# 6. HTML 리포트 생성
# ════════════════════════════════════════════════════

def _genre_tags_html(genres: List[str]) -> str:
    if not genres:
        return '<span class="genre-tag muted">-</span>'
    return "".join(f'<span class="genre-tag">{g}</span>' for g in genres[:5])


def _track_card_html(track: Dict, rank: int, is_seed: bool = False) -> str:
    name        = track.get("name", "Unknown")
    artists     = track.get("artists", [])
    if isinstance(artists, str):
        artists = [artists]
    artists_str = ", ".join(artists)
    album       = track.get("album", "")
    url         = track.get("external_urls", "#")
    pop         = track.get("popularity", 0)
    sim         = track.get("similarity_score")
    dur_ms      = track.get("duration_ms", 0)
    dur_str     = f"{int(dur_ms/60000)}:{int((dur_ms%60000)/1000):02d}" if dur_ms else "-"
    genres      = track.get("genres", [])
    explicit    = track.get("explicit", False)
    release     = track.get("release_date", "")

    sim_badge = ""
    if sim is not None and not is_seed:
        sim_pct = round(sim * 100, 1)
        color   = "var(--green)" if sim_pct >= 50 else ("var(--orange)" if sim_pct >= 30 else "var(--muted)")
        sim_badge = f'<span class="sim-badge" style="color:{color};border-color:{color}">유사도 {sim_pct:.0f}%</span>'

    new_badge = ""
    if release and release[:4] >= "2024":
        new_badge = '<span class="new-badge">NEW</span>'

    seed_class   = "seed-card" if is_seed else ""
    rank_label   = "SEED" if is_seed else f"#{rank}"
    explicit_tag = '<span class="explicit-tag">E</span>' if explicit else ""
    preview_btn  = ""
    if track.get("preview_url"):
        preview_btn = f'<a class="preview-btn" href="{track["preview_url"]}" target="_blank" rel="noopener">▶ 미리듣기</a>'

    genre_html = _genre_tags_html(genres)

    return f"""
    <div class="track-card {seed_class}">
      <div class="track-rank">{rank_label}</div>
      <div class="track-info">
        <div class="track-name">
          {explicit_tag}{new_badge}
          <a href="{url}" target="_blank" rel="noopener">{name}</a>
          {sim_badge}
        </div>
        <div class="track-meta artist">{artists_str}</div>
        <div class="track-meta album">{album} &nbsp;·&nbsp; {dur_str}</div>
        <div class="pop-row">
          <span class="pop-label">인기도</span>
          <div class="pop-bar-wrap"><div class="pop-bar" style="width:{pop}%"></div></div>
          <span class="pop-num">{pop}</span>
        </div>
        <div class="genre-row">{genre_html}</div>
        {preview_btn}
      </div>
    </div>"""


def _genre_chart_data(tracks: List[Dict]) -> str:
    counter: Counter = Counter()
    for t in tracks:
        for g in t.get("genres", [])[:2]:
            counter[g] += 1
    top = counter.most_common(8)
    if not top:
        return "null"
    labels = [x[0] for x in top]
    data   = [x[1] for x in top]
    colors = ["#1DB954","#ff6b35","#4fc3f7","#f7c59f","#ab47bc","#ef5350","#26a69a","#ffa726"]
    return json.dumps({"labels": labels, "data": data, "colors": colors[:len(labels)]})


def generate_html_report(
    original_query: str,
    parsed: Dict,
    model_info: Dict,
    seed_tracks: List[Dict],
    seed_genres: Set[str],
    recommendations: List[Dict],
    output_path: Path,
) -> None:
    ts_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_seed  = len(seed_tracks)
    n_rec   = len(recommendations)
    avg_pop = round(sum(t.get("popularity", 0) for t in recommendations) / n_rec) if n_rec else 0

    seed_html = "".join(_track_card_html(t, i+1, is_seed=True) for i, t in enumerate(seed_tracks))
    rec_html  = "".join(_track_card_html(t, i+1)               for i, t in enumerate(recommendations))
    genre_cloud      = _genre_tags_html(sorted(seed_genres))
    genre_chart_json = _genre_chart_data(recommendations)

    # 헤더 시그널 뱃지
    signal_badges = []
    if parsed.get("genre_queries"):
        genres_str = " · ".join(parsed["genre_queries"][:3])
        signal_badges.append(f'<span class="sig-pill">🎵 장르: {genres_str}</span>')
    if parsed.get("mood_terms"):
        signal_badges.append(f'<span class="sig-pill">✨ 무드: {parsed["mood_terms"]}</span>')
    if parsed.get("prefer_recent"):
        signal_badges.append('<span class="sig-pill new">🆕 최신곡 우선</span>')
    if parsed.get("keyword"):
        signal_badges.append(f'<span class="sig-pill">🔍 키워드: {parsed["keyword"]}</span>')
    signals_html = "".join(signal_badges)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>[{model_info['name']}] 음악 추천 리포트 · {ts_str}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --green:  #1DB954;
      --orange: #ff6b35;
      --dark:   #111111;
      --dark2:  #181818;
      --dark3:  #222222;
      --border: #2a2a2a;
      --light:  #FFFFFF;
      --muted:  #9A9A9A;
      --font: 'Segoe UI','Apple SD Gothic Neo',sans-serif;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}
    body {{ font-family: var(--font); background: var(--dark); color: var(--light); }}

    header {{
      background: linear-gradient(160deg, #0a2342 0%, #0d3b2e 100%);
      padding: 52px 40px 40px; border-bottom: 2px solid var(--green);
      position: relative; overflow: hidden;
    }}
    header::after {{
      content:''; position:absolute; top:-60px; right:-60px;
      width:300px; height:300px; border-radius:50%;
      background: radial-gradient(circle, rgba(29,185,84,.15) 0%, transparent 70%);
    }}
    .header-eyebrow {{
      font-size:11px; font-weight:700; letter-spacing:3px;
      text-transform:uppercase; color:var(--green); margin-bottom:14px;
    }}
    .model-name {{
      font-size:13px; font-weight:700; color:var(--light);
      margin-bottom:4px; letter-spacing:.3px;
    }}
    .model-desc {{
      font-size:11px; color:var(--muted); margin-bottom:16px;
    }}
    header h1 {{
      font-size: clamp(20px, 4vw, 30px); font-weight:800; line-height:1.35; margin-bottom:10px;
    }}
    header h1 .hl {{ color: var(--green); }}
    .header-meta {{ font-size:13px; color:var(--muted); margin-bottom:14px; }}
    .signals {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .sig-pill {{
      display:inline-flex; align-items:center;
      background:rgba(29,185,84,.12); border:1px solid rgba(29,185,84,.35);
      color:var(--green); padding:4px 12px; border-radius:20px; font-size:12px;
    }}
    .sig-pill.new {{
      background:rgba(255,107,53,.12); border-color:rgba(255,107,53,.4); color:var(--orange);
    }}

    .stats-bar {{
      display:grid; grid-template-columns:repeat(4,1fr);
      background: var(--dark3); border-bottom:1px solid var(--border);
    }}
    .stat {{
      padding:22px 20px; text-align:center; border-right:1px solid var(--border);
    }}
    .stat:last-child {{ border-right:none; }}
    .stat-num {{ font-size:34px; font-weight:800; color:var(--green); }}
    .stat-lbl {{ font-size:11px; color:var(--muted); margin-top:4px; letter-spacing:.5px; }}

    .page {{ max-width:1100px; margin:0 auto; padding:0 24px 80px; }}
    section {{ margin-top:52px; }}
    .section-title {{
      font-size:17px; font-weight:700;
      border-left:4px solid var(--green); padding-left:13px;
      margin-bottom:24px; display:flex; align-items:center; gap:10px;
    }}
    .section-count {{
      font-size:12px; font-weight:500; color:var(--muted);
      background:var(--dark3); padding:2px 10px; border-radius:12px;
    }}
    .genre-cloud {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px; }}
    .cards-grid {{
      display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:14px;
    }}

    .track-card {{
      background:var(--dark2); border-radius:10px; padding:16px 18px;
      display:flex; gap:12px; border:1px solid var(--border);
      transition: border-color .18s, transform .18s, box-shadow .18s;
    }}
    .track-card:hover {{
      border-color: var(--green); transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(29,185,84,.12);
    }}
    .seed-card {{ border-color: rgba(29,185,84,.35); background: rgba(29,185,84,.04); }}
    .track-rank {{ font-size:12px; font-weight:800; min-width:38px; color:var(--muted); padding-top:2px; flex-shrink:0; }}
    .seed-card .track-rank {{ color: var(--green); }}
    .track-info {{ flex:1; min-width:0; }}
    .track-name {{
      font-size:14px; font-weight:600; line-height:1.4;
      display:flex; flex-wrap:wrap; align-items:center; gap:5px; margin-bottom:4px;
    }}
    .track-name a {{ color:var(--light); text-decoration:none; }}
    .track-name a:hover {{ color:var(--green); }}
    .track-meta {{ font-size:12px; color:var(--muted); margin-bottom:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .track-meta.artist {{ color:#ccc; font-size:13px; }}
    .pop-row {{ display:flex; align-items:center; gap:7px; margin:8px 0 6px; }}
    .pop-label {{ font-size:10px; color:var(--muted); width:36px; flex-shrink:0; }}
    .pop-bar-wrap {{ flex:1; background:#2e2e2e; border-radius:3px; height:5px; overflow:hidden; }}
    .pop-bar {{ height:100%; background:var(--green); border-radius:3px; }}
    .pop-num {{ font-size:10px; color:var(--muted); width:22px; text-align:right; flex-shrink:0; }}
    .genre-row {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:6px; }}
    .genre-tag {{
      font-size:10px; padding:2px 7px; border-radius:10px;
      background:rgba(29,185,84,.12); color:var(--green); white-space:nowrap;
    }}
    .genre-tag.muted {{ background:var(--dark3); color:var(--muted); }}
    .sim-badge {{ font-size:10px; font-weight:700; padding:2px 8px; border-radius:10px; border:1px solid; white-space:nowrap; }}
    .new-badge {{ font-size:9px; font-weight:700; padding:1px 5px; background:var(--orange); color:#fff; border-radius:3px; }}
    .explicit-tag {{ font-size:9px; font-weight:700; padding:1px 5px; background:#aaa; color:#000; border-radius:3px; }}
    .preview-btn {{
      display:inline-block; margin-top:8px; font-size:11px; color:var(--green);
      text-decoration:none; border:1px solid rgba(29,185,84,.4); padding:3px 10px; border-radius:10px;
    }}
    .preview-btn:hover {{ background:rgba(29,185,84,.1); }}

    .chart-row {{ display:grid; grid-template-columns:280px 1fr; gap:32px; align-items:center; }}
    @media(max-width:680px) {{ .chart-row {{ grid-template-columns:1fr; }} }}
    .chart-box {{ background:var(--dark2); border-radius:12px; padding:24px; border:1px solid var(--border); }}
    .chart-legend {{ display:flex; flex-direction:column; gap:10px; }}
    .legend-item {{ display:flex; align-items:center; gap:10px; font-size:13px; }}
    .legend-dot {{ width:12px; height:12px; border-radius:50%; flex-shrink:0; }}
    .legend-label {{ flex:1; color:var(--light); }}
    .legend-count {{ color:var(--muted); }}

    footer {{
      text-align:center; color:var(--muted); font-size:12px;
      margin-top:60px; padding:24px 24px 40px; border-top:1px solid var(--border);
    }}
    footer a {{ color:var(--green); text-decoration:none; }}
  </style>
</head>
<body>

<header>
  <div class="header-eyebrow">Music Recommender · Report</div>
  <div class="model-name">{model_info['name']}</div>
  <div class="model-desc">{model_info['desc']}</div>
  <h1>"<span class="hl">{original_query}</span>"</h1>
  <div class="header-meta">생성 시각 {ts_str}</div>
  <div class="signals">{signals_html}</div>
</header>

<div class="stats-bar">
  <div class="stat"><div class="stat-num">{n_seed}</div><div class="stat-lbl">씨앗 트랙</div></div>
  <div class="stat"><div class="stat-num">{n_rec}</div><div class="stat-lbl">추천 트랙</div></div>
  <div class="stat"><div class="stat-num">{len(seed_genres)}</div><div class="stat-lbl">장르 수</div></div>
  <div class="stat"><div class="stat-num">{avg_pop}</div><div class="stat-lbl">평균 인기도</div></div>
</div>

<div class="page">

  <section>
    <div class="section-title">씨앗 트랙 (Seed Tracks)<span class="section-count">{n_seed}개</span></div>
    <div class="genre-cloud">{genre_cloud}</div>
    <div class="cards-grid" style="margin-top:16px;">{seed_html}</div>
  </section>

  <section>
    <div class="section-title">추천 트랙 장르 분포</div>
    <div class="chart-row">
      <div class="chart-box"><canvas id="genreChart" width="240" height="240"></canvas></div>
      <div class="chart-legend" id="chartLegend"></div>
    </div>
  </section>

  <section>
    <div class="section-title">추천 트랙 (Recommendations)<span class="section-count">{n_rec}개</span></div>
    <div class="cards-grid">{rec_html}</div>
  </section>

</div>

<footer>
  Generated by <strong>Music Recommender</strong> &nbsp;·&nbsp;
  Powered by <a href="https://developer.spotify.com/" target="_blank">Spotify Web API</a>
  &nbsp;·&nbsp; {ts_str}
</footer>

<script>
(function() {{
  var raw = {genre_chart_json};
  if (!raw) return;
  var ctx = document.getElementById('genreChart').getContext('2d');
  new Chart(ctx, {{
    type: 'doughnut',
    data: {{
      labels: raw.labels,
      datasets: [{{ data: raw.data, backgroundColor: raw.colors, borderWidth: 2, borderColor: '#111' }}]
    }},
    options: {{
      responsive: true, cutout: '62%',
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: function(ctx) {{ return ' ' + ctx.label + ': ' + ctx.parsed + '곡'; }} }} }}
      }}
    }}
  }});
  var legend = document.getElementById('chartLegend');
  raw.labels.forEach(function(lbl, i) {{
    var div = document.createElement('div');
    div.className = 'legend-item';
    div.innerHTML = '<div class="legend-dot" style="background:' + raw.colors[i] + '"></div>'
      + '<span class="legend-label">' + lbl + '</span>'
      + '<span class="legend-count">' + raw.data[i] + '곡</span>';
    legend.appendChild(div);
  }});
}})();
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"\n[3/3] 리포트 저장 완료: {output_path}")


# ════════════════════════════════════════════════════
# 6. 진입점
# ════════════════════════════════════════════════════

def build_output_path(original_query: str, keyword: str, model_slug: str) -> Path:
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = make_filename_slug(keyword) or make_filename_slug(original_query) or "report"
    return REPORTS_DIR / f"report_{ts}_{model_slug}_{query_slug}.html"


def main():
    parser = argparse.ArgumentParser(
        description="자연어 입력으로 음악 추천 HTML 리포트를 생성합니다."
    )
    parser.add_argument("query", nargs="?", help="자연어 입력")
    parser.add_argument("--n",     dest="n_results", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--model", dest="model", default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"추천 모델 (기본값: {DEFAULT_MODEL}). 선택지: {list(MODELS.keys())}",
    )
    args = parser.parse_args()

    if args.query:
        original_query = args.query.strip()
    else:
        print("=" * 55)
        print("  음악 추천 리포트 생성기")
        print("=" * 55)
        print('예시: "브루노마스 신곡 너무 좋더라 비슷한 거 없어?"')
        print('      "잔잔한 재즈 감성"')
        print('      "BTS Dynamite 같은 신나는 팝"')
        print("-" * 55)
        original_query = input("입력: ").strip()

    if not original_query:
        print("입력이 없습니다.")
        sys.exit(1)

    model_info = MODELS[args.model]
    parsed = parse_natural_query(original_query)
    print(f"\n모델: {model_info['name']}")
    print(f"원본 입력: {original_query!r}")
    print(f"  키워드:      {parsed['keyword']!r}")
    print(f"  장르 쿼리:   {parsed['genre_queries']}")
    print(f"  무드:        {parsed['mood_terms']!r}")
    print(f"  최신곡 선호: {parsed['prefer_recent']}")

    try:
        collector = SpotifyMusicCollector()
    except Exception as e:
        print(f"\n[오류] Spotify 초기화 실패: {e}")
        sys.exit(1)

    try:
        seed_tracks, seed_genres, seed_artist_ids = find_seed_tracks(
            collector, parsed, n_seeds=args.seeds
        )
    except ValueError as e:
        print(f"\n[오류] {e}")
        sys.exit(1)

    seed_ids = {t["id"] for t in seed_tracks}
    recommendations = find_recommendations(
        collector, parsed, seed_tracks, seed_genres, seed_artist_ids, seed_ids,
        n_results=args.n_results,
        model_key=args.model,
    )

    if not recommendations:
        print("[경고] 유사 트랙을 찾지 못했습니다.")

    slug = parsed["keyword"] or original_query
    output_path = build_output_path(original_query, slug, model_info["slug"])
    generate_html_report(
        original_query=original_query,
        parsed=parsed,
        model_info=model_info,
        seed_tracks=seed_tracks,
        seed_genres=seed_genres,
        recommendations=recommendations,
        output_path=output_path,
    )

    print(f"\n완료! 브라우저에서 열기:")
    print(f"  {output_path.resolve()}")


if __name__ == "__main__":
    main()
