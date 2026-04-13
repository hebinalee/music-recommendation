#!/usr/bin/env python3
"""
자연어 입력 기반 음악 추천 리포트 생성기

사용법:
  # 대화형 모드
  python scripts/generate_report.py

  # CLI 모드
  python scripts/generate_report.py "이번 브루노마스 신곡 너무 좋더라 비슷한 거 없어?"

  # 추천 개수 지정
  python scripts/generate_report.py "재즈 분위기 잔잔한 곡 추천" --n 15

참고:
  Spotify Web API는 2024년 11월부터 일반 앱에서 /audio-features 엔드포인트를
  제한합니다. 이 스크립트는 아티스트 장르·관련 아티스트 기반 유사도를 사용합니다.
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

# 프로젝트 루트를 sys.path에 추가
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import logging

from dotenv import load_dotenv
load_dotenv()

# spotipy 내부 HTTP 에러 로그 억제 (예상된 404 등 노이즈 방지)
logging.getLogger("spotipy").setLevel(logging.CRITICAL)

from src.spotify_collector import SpotifyMusicCollector
from src.logger import get_logger

logger = get_logger(__name__)

REPORTS_DIR = _ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ────────────────────────────────────────────────
# 1. 자연어 파싱
# ────────────────────────────────────────────────

_KO_STOPWORDS = {
    "이번", "신곡", "노래", "음악", "곡", "좀", "좋은", "있어", "없어",
    "추천", "해줘", "해", "줘", "이랑", "랑", "같은", "비슷한", "느낌",
    "너무", "정말", "진짜", "완전", "좋더라", "좋아", "들어봤어", "들었는데",
    "아는", "알아", "찾아줘", "찾아", "주세요", "부탁해", "어때", "어떤",
    "듣고", "싶어", "싶은", "뭐", "뭔가", "분위기", "스타일", "장르",
    "좀더", "더", "많이", "그냥", "한번", "한 번", "같이", "처럼",
}


def parse_natural_query(text: str) -> str:
    """자연어 문장에서 Spotify 검색용 핵심 키워드를 추출합니다."""
    cleaned = re.sub(r"[^\w\s가-힣]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = cleaned.split()
    filtered = []
    for tok in tokens:
        if re.search(r"[가-힣]", tok):
            if tok not in _KO_STOPWORDS and len(tok) >= 2:
                filtered.append(tok)
        else:
            if len(tok) >= 2:
                filtered.append(tok)
    return " ".join(filtered) if filtered else text.strip()


def make_filename_slug(text: str, max_len: int = 40) -> str:
    """파일명용 슬러그 생성 (한글·영문 포함, 공백→하이픈)"""
    # 파일명 불가 특수문자만 제거
    slug = re.sub(r'[\\/:*?"<>|]', "", text)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:max_len] if slug else "report"


# ────────────────────────────────────────────────
# 2. Spotify 데이터 수집 헬퍼
# ────────────────────────────────────────────────

def _get_artist_info(sp, artist_id: str) -> Dict:
    """아티스트 정보 (장르, 팔로워, 인기도) 조회"""
    try:
        return sp.artist(artist_id)
    except Exception:
        return {}


def _get_related_artists(sp, artist_id: str, limit: int = 10) -> List[Dict]:
    """Spotify 관련 아티스트 조회"""
    try:
        result = sp.artist_related_artists(artist_id)
        return result.get("artists", [])[:limit]
    except Exception:
        return []


def _get_artist_top_tracks(sp, artist_id: str, market: str = "US") -> List[Dict]:
    """아티스트 인기 트랙 조회"""
    try:
        result = sp.artist_top_tracks(artist_id, country=market)
        return result.get("tracks", [])
    except Exception:
        return []


def _normalize_track(raw: Dict) -> Dict:
    """Spotify 원시 트랙 딕셔너리를 정규화된 형태로 변환"""
    artists_raw = raw.get("artists", [])
    return {
        "id":           raw.get("id", ""),
        "name":         raw.get("name", "Unknown"),
        "artists":      [a["name"] for a in artists_raw],
        "artist_ids":   [a["id"] for a in artists_raw],
        "album":        raw.get("album", {}).get("name", "") if isinstance(raw.get("album"), dict) else raw.get("album", ""),
        "release_date": raw.get("album", {}).get("release_date", "") if isinstance(raw.get("album"), dict) else "",
        "popularity":   raw.get("popularity", 0),
        "duration_ms":  raw.get("duration_ms", 0),
        "explicit":     raw.get("explicit", False),
        "external_urls": raw.get("external_urls", {}).get("spotify", "#"),
        "preview_url":  raw.get("preview_url", None),
        "genres":       [],   # 아티스트 조회 후 채움
    }


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ────────────────────────────────────────────────
# 3. 씨앗 곡 탐색 & 추천
# ────────────────────────────────────────────────

def find_seed_tracks(
    collector: SpotifyMusicCollector,
    search_keyword: str,
    n_seeds: int = 3,
) -> tuple:
    """
    검색어로 씨앗 트랙과 아티스트 장르 정보를 반환합니다.
    Spotify API를 직접 호출해 artist_id 정보를 보존합니다.
    Returns: (seed_tracks, seed_genres: Set[str], seed_artist_ids: List[str])
    """
    print(f"\n[1/3] Spotify 검색 중: '{search_keyword}'")
    sp = collector.sp

    # sp.search() 원시 결과 사용 (artist ID 보존)
    raw_result = sp.search(q=search_keyword, type="track", limit=10)
    raw_tracks  = raw_result.get("tracks", {}).get("items", [])
    if not raw_tracks:
        raise ValueError(f"'{search_keyword}'에 해당하는 트랙을 찾지 못했습니다.")

    raw_tracks.sort(key=lambda t: t.get("popularity", 0), reverse=True)
    raw_seeds = raw_tracks[:n_seeds]

    seeds: List[Dict] = []
    seed_genres: Set[str] = set()
    seed_artist_ids: List[str] = []

    for raw in raw_seeds:
        track = _normalize_track(raw)
        if track["artist_ids"]:
            primary_id = track["artist_ids"][0]
            if primary_id not in seed_artist_ids:
                seed_artist_ids.append(primary_id)
            artist_info = _get_artist_info(sp, primary_id)
            genres = artist_info.get("genres", [])
            track["genres"] = genres
            seed_genres.update(genres)
        seeds.append(track)

    print(f"  씨앗 곡 {len(seeds)}개 선택:")
    for t in seeds:
        artists_str = ", ".join(t["artists"])
        genres_str  = ", ".join(t["genres"][:3]) if t["genres"] else "장르 정보 없음"
        print(f"  - {t['name']} by {artists_str}  [{genres_str}]")
    print(f"  씨앗 장르: {', '.join(list(seed_genres)[:8]) or '없음'}")

    return seeds, seed_genres, seed_artist_ids


def find_similar_tracks(
    collector: SpotifyMusicCollector,
    seed_genres: Set[str],
    seed_artist_ids: List[str],
    search_keyword: str,
    n_results: int = 20,
) -> List[Dict]:
    """
    씨앗 아티스트의 관련 아티스트 + 장르 검색으로 유사 트랙을 탐색합니다.

    유사도 점수 = 0.6 * 장르 Jaccard + 0.4 * 정규화된 인기도
    """
    print(f"\n[2/3] 유사 트랙 탐색 중...")
    sp = collector.sp

    candidate_map: Dict[str, Dict] = {}

    # 전략 A: 관련 아티스트의 인기 트랙
    for artist_id in seed_artist_ids[:2]:
        related = _get_related_artists(sp, artist_id, limit=8)
        print(f"  관련 아티스트 {len(related)}명 발견")
        for rel_artist in related:
            rel_id = rel_artist["id"]
            top_tracks = _get_artist_top_tracks(sp, rel_id)
            rel_genres = set(rel_artist.get("genres", []))
            for raw in top_tracks:
                t = _normalize_track(raw)
                t["genres"] = list(rel_genres)
                if t["id"] and t["id"] not in candidate_map:
                    candidate_map[t["id"]] = t

    # 전략 B: 키워드 + 무드 변형 검색
    extra_queries = [search_keyword]
    if seed_genres:
        genre_sample = " ".join(list(seed_genres)[:2])
        extra_queries.append(genre_sample)

    for q in extra_queries:
        # sp.search() 원시 결과로 artist_id 보존
        raw_result = sp.search(q=q, type="track", limit=30)
        raw_list   = raw_result.get("tracks", {}).get("items", [])
        for raw in raw_list:
            t = _normalize_track(raw)
            if t["id"] and t["id"] not in candidate_map:
                if not t["genres"] and t["artist_ids"]:
                    info = _get_artist_info(sp, t["artist_ids"][0])
                    t["genres"] = info.get("genres", [])
                candidate_map[t["id"]] = t

    print(f"  후보 트랙 총 {len(candidate_map)}개")

    # 유사도 계산
    scored: List[Dict] = []
    for track in candidate_map.values():
        track_genres = set(track.get("genres", []))
        genre_sim   = _jaccard(seed_genres, track_genres) if seed_genres else 0.0
        pop_norm    = track.get("popularity", 0) / 100.0
        score       = round(0.6 * genre_sim + 0.4 * pop_norm, 4)
        scored.append({**track, "similarity_score": score, "genre_sim": genre_sim})

    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored[:n_results]


# ────────────────────────────────────────────────
# 4. HTML 리포트 생성
# ────────────────────────────────────────────────

def _genre_tags_html(genres: List[str]) -> str:
    if not genres:
        return '<span class="genre-tag muted">-</span>'
    return "".join(
        f'<span class="genre-tag">{g}</span>'
        for g in genres[:5]
    )


def _track_card_html(track: Dict, rank: int, is_seed: bool = False) -> str:
    name      = track.get("name", "Unknown")
    artists   = track.get("artists", [])
    if isinstance(artists, str):
        artists = [artists]
    artists_str = ", ".join(artists)
    album     = track.get("album", "")
    url       = track.get("external_urls", "#")
    pop       = track.get("popularity", 0)
    sim       = track.get("similarity_score")
    genre_sim = track.get("genre_sim")
    dur_ms    = track.get("duration_ms", 0)
    dur_str   = f"{int(dur_ms/60000)}:{int((dur_ms%60000)/1000):02d}" if dur_ms else "-"
    genres    = track.get("genres", [])
    explicit  = track.get("explicit", False)

    # 배지
    sim_badge = ""
    if sim is not None and not is_seed:
        sim_pct = round(sim * 100, 1)
        color   = "var(--green)" if sim_pct >= 50 else ("var(--orange)" if sim_pct >= 30 else "var(--muted)")
        sim_badge = f'<span class="sim-badge" style="color:{color};border-color:{color}">유사도 {sim_pct:.0f}%</span>'

    seed_class  = "seed-card" if is_seed else ""
    rank_label  = "SEED" if is_seed else f"#{rank}"
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
          {explicit_tag}
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
    """추천 트랙 장르 분포 데이터 (Chart.js 도넛 차트용)"""
    from collections import Counter
    counter: Counter = Counter()
    for t in tracks:
        for g in t.get("genres", [])[:2]:
            counter[g] += 1
    top = counter.most_common(8)
    if not top:
        return "null"
    labels = [x[0] for x in top]
    data   = [x[1] for x in top]
    colors = [
        "#1DB954", "#ff6b35", "#4fc3f7", "#f7c59f",
        "#ab47bc", "#ef5350", "#26a69a", "#ffa726",
    ]
    return json.dumps({"labels": labels, "data": data, "colors": colors[:len(labels)]})


def generate_html_report(
    original_query: str,
    parsed_keyword: str,
    seed_tracks: List[Dict],
    seed_genres: Set[str],
    recommendations: List[Dict],
    output_path: Path,
) -> None:
    ts_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_seed   = len(seed_tracks)
    n_rec    = len(recommendations)

    # 씨앗 카드
    seed_html = "".join(_track_card_html(t, i + 1, is_seed=True) for i, t in enumerate(seed_tracks))

    # 추천 카드
    rec_html = "".join(_track_card_html(t, i + 1) for i, t in enumerate(recommendations))

    # 장르 태그 클라우드
    genre_cloud = _genre_tags_html(sorted(seed_genres))

    # 차트 데이터
    genre_chart_json = _genre_chart_data(recommendations)

    # 평균 인기도
    avg_pop = round(sum(t.get("popularity", 0) for t in recommendations) / n_rec) if n_rec else 0

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>음악 추천 리포트 · {ts_str}</title>
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

    /* ── HEADER ── */
    header {{
      background: linear-gradient(160deg, #0a2342 0%, #0d3b2e 100%);
      padding: 52px 40px 40px;
      border-bottom: 2px solid var(--green);
      position: relative;
      overflow: hidden;
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
    header h1 {{
      font-size: clamp(20px, 4vw, 30px); font-weight:800; line-height:1.35;
      margin-bottom:10px;
    }}
    header h1 .hl {{ color: var(--green); }}
    .header-meta {{ font-size:13px; color:var(--muted); }}
    .kw-pill {{
      display:inline-flex; align-items:center; gap:6px; margin-top:16px;
      background:rgba(29,185,84,.12); border:1px solid rgba(29,185,84,.4);
      color:var(--green); padding:5px 14px; border-radius:20px; font-size:13px;
    }}

    /* ── STATS BAR ── */
    .stats-bar {{
      display:grid; grid-template-columns:repeat(4,1fr);
      background: var(--dark3); border-bottom:1px solid var(--border);
    }}
    .stat {{
      padding:22px 20px; text-align:center;
      border-right:1px solid var(--border);
    }}
    .stat:last-child {{ border-right:none; }}
    .stat-num {{ font-size:34px; font-weight:800; color:var(--green); }}
    .stat-lbl {{ font-size:11px; color:var(--muted); margin-top:4px; letter-spacing:.5px; }}

    /* ── LAYOUT ── */
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

    /* ── GENRE CLOUD ── */
    .genre-cloud {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px; }}

    /* ── GRID ── */
    .cards-grid {{
      display:grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap:14px;
    }}

    /* ── TRACK CARD ── */
    .track-card {{
      background:var(--dark2); border-radius:10px;
      padding:16px 18px; display:flex; gap:12px;
      border:1px solid var(--border);
      transition: border-color .18s, transform .18s, box-shadow .18s;
    }}
    .track-card:hover {{
      border-color: var(--green);
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(29,185,84,.12);
    }}
    .seed-card {{
      border-color: rgba(29,185,84,.35);
      background: rgba(29,185,84,.04);
    }}
    .track-rank {{
      font-size:12px; font-weight:800; min-width:38px;
      color:var(--muted); padding-top:2px; flex-shrink:0;
    }}
    .seed-card .track-rank {{ color: var(--green); }}
    .track-info {{ flex:1; min-width:0; }}
    .track-name {{
      font-size:14px; font-weight:600; line-height:1.4;
      display:flex; flex-wrap:wrap; align-items:center; gap:6px;
      margin-bottom:4px;
    }}
    .track-name a {{ color:var(--light); text-decoration:none; }}
    .track-name a:hover {{ color:var(--green); }}
    .track-meta {{
      font-size:12px; color:var(--muted); margin-bottom:2px;
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    }}
    .track-meta.artist {{ color:#ccc; font-size:13px; }}

    /* Popularity bar */
    .pop-row {{
      display:flex; align-items:center; gap:7px; margin:8px 0 6px;
    }}
    .pop-label {{ font-size:10px; color:var(--muted); width:36px; flex-shrink:0; }}
    .pop-bar-wrap {{ flex:1; background:#2e2e2e; border-radius:3px; height:5px; overflow:hidden; }}
    .pop-bar {{ height:100%; background:var(--green); border-radius:3px; }}
    .pop-num {{ font-size:10px; color:var(--muted); width:22px; text-align:right; flex-shrink:0; }}

    /* Genre tags */
    .genre-row {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:6px; }}
    .genre-tag {{
      font-size:10px; padding:2px 7px; border-radius:10px;
      background:rgba(29,185,84,.12); color:var(--green); white-space:nowrap;
    }}
    .genre-tag.muted {{ background:var(--dark3); color:var(--muted); }}

    /* Badges */
    .sim-badge {{
      font-size:10px; font-weight:700; padding:2px 8px;
      border-radius:10px; border:1px solid; white-space:nowrap;
    }}
    .explicit-tag {{
      font-size:9px; font-weight:700; padding:1px 5px;
      background:#aaa; color:#000; border-radius:3px;
    }}
    .preview-btn {{
      display:inline-block; margin-top:8px;
      font-size:11px; color:var(--green);
      text-decoration:none; border:1px solid rgba(29,185,84,.4);
      padding:3px 10px; border-radius:10px;
    }}
    .preview-btn:hover {{ background:rgba(29,185,84,.1); }}

    /* ── CHART SECTION ── */
    .chart-row {{
      display:grid; grid-template-columns:280px 1fr; gap:32px; align-items:center;
    }}
    @media(max-width:680px) {{ .chart-row {{ grid-template-columns:1fr; }} }}
    .chart-box {{
      background:var(--dark2); border-radius:12px;
      padding:24px; border:1px solid var(--border);
    }}
    .chart-legend {{
      display:flex; flex-direction:column; gap:10px;
    }}
    .legend-item {{
      display:flex; align-items:center; gap:10px; font-size:13px;
    }}
    .legend-dot {{
      width:12px; height:12px; border-radius:50%; flex-shrink:0;
    }}
    .legend-label {{ flex:1; color:var(--light); }}
    .legend-count {{ color:var(--muted); }}

    /* ── FOOTER ── */
    footer {{
      text-align:center; color:var(--muted); font-size:12px;
      margin-top:60px; padding:24px 24px 40px;
      border-top:1px solid var(--border);
    }}
    footer a {{ color:var(--green); text-decoration:none; }}
    footer a:hover {{ text-decoration:underline; }}
  </style>
</head>
<body>

<header>
  <div class="header-eyebrow">Music Recommender · Report</div>
  <h1>"<span class="hl">{original_query}</span>"</h1>
  <div class="header-meta">생성 시각 {ts_str}</div>
  <div class="kw-pill">🔍 검색 키워드: {parsed_keyword}</div>
</header>

<div class="stats-bar">
  <div class="stat">
    <div class="stat-num">{n_seed}</div>
    <div class="stat-lbl">씨앗 트랙</div>
  </div>
  <div class="stat">
    <div class="stat-num">{n_rec}</div>
    <div class="stat-lbl">추천 트랙</div>
  </div>
  <div class="stat">
    <div class="stat-num">{len(seed_genres)}</div>
    <div class="stat-lbl">장르 수</div>
  </div>
  <div class="stat">
    <div class="stat-num">{avg_pop}</div>
    <div class="stat-lbl">평균 인기도</div>
  </div>
</div>

<div class="page">

  <!-- ① 씨앗 트랙 -->
  <section>
    <div class="section-title">
      씨앗 트랙 (Seed Tracks)
      <span class="section-count">{n_seed}개</span>
    </div>
    <div class="genre-cloud">{genre_cloud}</div>
    <div class="cards-grid" style="margin-top:16px;">
      {seed_html}
    </div>
  </section>

  <!-- ② 장르 분포 차트 -->
  <section>
    <div class="section-title">추천 트랙 장르 분포</div>
    <div class="chart-row">
      <div class="chart-box">
        <canvas id="genreChart" width="240" height="240"></canvas>
      </div>
      <div class="chart-legend" id="chartLegend"></div>
    </div>
  </section>

  <!-- ③ 추천 트랙 -->
  <section>
    <div class="section-title">
      추천 트랙 (Recommendations)
      <span class="section-count">{n_rec}개</span>
    </div>
    <div class="cards-grid">
      {rec_html}
    </div>
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
      datasets: [{{
        data: raw.data,
        backgroundColor: raw.colors,
        borderWidth: 2,
        borderColor: '#111',
      }}]
    }},
    options: {{
      responsive: true,
      cutout: '62%',
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              return ' ' + ctx.label + ': ' + ctx.parsed + '곡';
            }}
          }}
        }}
      }}
    }}
  }});

  // 커스텀 범례
  var legend = document.getElementById('chartLegend');
  raw.labels.forEach(function(lbl, i) {{
    var div = document.createElement('div');
    div.className = 'legend-item';
    div.innerHTML =
      '<div class="legend-dot" style="background:' + raw.colors[i] + '"></div>' +
      '<span class="legend-label">' + lbl + '</span>' +
      '<span class="legend-count">' + raw.data[i] + '곡</span>';
    legend.appendChild(div);
  }});
}})();
</script>

</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"\n[3/3] 리포트 저장 완료: {output_path}")


# ────────────────────────────────────────────────
# 5. 진입점
# ────────────────────────────────────────────────

def build_output_path(original_query: str, parsed_keyword: str) -> Path:
    """reports/report_YYYYMMDD_HHMMSS_<slug>.html"""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = make_filename_slug(parsed_keyword) or make_filename_slug(original_query) or "report"
    return REPORTS_DIR / f"report_{ts}_{slug}.html"


def main():
    parser = argparse.ArgumentParser(
        description="자연어 입력으로 음악 추천 HTML 리포트를 생성합니다."
    )
    parser.add_argument(
        "query", nargs="?",
        help='자연어 입력 (예: "브루노마스 신곡 너무 좋던데 비슷한 거 없어?")',
    )
    parser.add_argument(
        "--n", dest="n_results", type=int, default=20,
        help="추천 트랙 수 (기본값: 20)",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="씨앗 트랙 수 (기본값: 3)",
    )
    args = parser.parse_args()

    if args.query:
        original_query = args.query.strip()
    else:
        print("=" * 55)
        print("  음악 추천 리포트 생성기")
        print("=" * 55)
        print("어떤 음악이 듣고 싶으세요?")
        print('예시: "브루노마스 신곡 너무 좋더라 비슷한 거 없어?"')
        print('      "잔잔하고 감성적인 재즈 추천해줘"')
        print('      "BTS Dynamite 같은 신나는 팝"')
        print("-" * 55)
        original_query = input("입력: ").strip()

    if not original_query:
        print("입력이 없습니다.")
        sys.exit(1)

    print(f"\n원본 입력: {original_query!r}")

    parsed_keyword = parse_natural_query(original_query)
    if not parsed_keyword:
        parsed_keyword = original_query
    print(f"추출 키워드: {parsed_keyword!r}")

    # Spotify 클라이언트
    try:
        collector = SpotifyMusicCollector()
    except Exception as e:
        print(f"\n[오류] Spotify 초기화 실패: {e}")
        print(".env 파일에 SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET 설정 필요")
        sys.exit(1)

    # 씨앗 탐색
    try:
        seed_tracks, seed_genres, seed_artist_ids = find_seed_tracks(
            collector, parsed_keyword, n_seeds=args.seeds
        )
    except ValueError as e:
        print(f"\n[오류] {e}")
        sys.exit(1)

    # 유사 트랙 추천
    seed_ids = {t["id"] for t in seed_tracks}
    recommendations = find_similar_tracks(
        collector,
        seed_genres,
        seed_artist_ids,
        parsed_keyword,
        n_results=args.n_results,
    )
    # 씨앗 곡 제거
    recommendations = [r for r in recommendations if r["id"] not in seed_ids]

    if not recommendations:
        print("[경고] 유사 트랙을 찾지 못했습니다.")

    # HTML 생성
    output_path = build_output_path(original_query, parsed_keyword)
    generate_html_report(
        original_query=original_query,
        parsed_keyword=parsed_keyword,
        seed_tracks=seed_tracks,
        seed_genres=seed_genres,
        recommendations=recommendations,
        output_path=output_path,
    )

    print(f"\n완료! 브라우저에서 열기:")
    print(f"  {output_path.resolve()}")


if __name__ == "__main__":
    main()
