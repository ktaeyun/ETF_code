# -*- coding: utf-8 -*-
"""
Google Trends 3개월 단위 자동 수집
- 글로벌 Bitcoin  → merged_bitcoin_all.csv  (date, value)
- 한국  비트코인  → merged_bitcoin_kr.csv   (date, value)

3개월 구간별로 일별 데이터를 수집한 뒤 단순 연결.
(ASVI 전처리가 스케일 차이를 흡수하므로 구간 간 정규화 생략)
"""

import time
import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from pytrends.request import TrendReq


# ── 설정 ──────────────────────────────────────────────────────────────
START_DATE  = date(2022, 5, 1)   # 수집 시작일 (약 3년 전)
END_DATE    = date(2025, 5, 23)  # 수집 종료일
CHUNK_MONTHS = 3                 # 구간 단위 (개월)
SLEEP_SEC    = 10                # 구간 간 대기 시간 (Google 요청 제한 회피)

OUT_DIR = "."  # trend 폴더 내에서 실행 기준


def build_chunks(start: date, end: date, months: int):
    """start~end 를 months 개월 단위로 분할한 (chunk_start, chunk_end) 리스트 반환"""
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + relativedelta(months=months) - timedelta(days=1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def fetch_chunk(pt: TrendReq, keyword: str, geo: str, start: date, end: date) -> pd.Series:
    """단일 구간 일별 검색량 지수 수집"""
    timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
    pt.build_payload([keyword], timeframe=timeframe, geo=geo)
    df = pt.interest_over_time()
    if df.empty or keyword not in df.columns:
        return pd.Series(dtype=float)
    return df[keyword].rename("value")


def _chunk_cache_path(out_dir: str, label_slug: str, idx: int) -> str:
    return f"{out_dir}/_cache_{label_slug}_{idx:02d}.csv"


def collect(keyword: str, geo: str, label: str, out_dir: str) -> pd.DataFrame:
    """
    3개월 단위로 전체 기간 수집 후 병합.
    - 구간별로 캐시 파일에 즉시 저장 → 중단 후 재실행 시 완료된 구간 건너뜀.
    - KeyboardInterrupt 발생 시 그때까지 수집한 결과를 저장하고 종료.
    """
    pt = TrendReq(hl='en-US', tz=0)
    chunks = build_chunks(START_DATE, END_DATE, CHUNK_MONTHS)
    label_slug = label.replace(" ", "_").replace("/", "_")
    parts = []

    print(f"\n[{label}] 총 {len(chunks)}개 구간 수집 시작 (keyword='{keyword}', geo='{geo or 'global'}')")
    try:
        for i, (s, e) in enumerate(chunks, 1):
            cache_path = _chunk_cache_path(out_dir, label_slug, i)

            # 이미 수집된 구간이면 캐시에서 로드
            try:
                cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)["value"]
                parts.append(cached)
                print(f"  [{i:02d}/{len(chunks)}] {s} ~ {e} ... 캐시 로드 ({len(cached)}행)")
                continue
            except FileNotFoundError:
                pass

            print(f"  [{i:02d}/{len(chunks)}] {s} ~ {e} ...", end=" ", flush=True)
            try:
                series = fetch_chunk(pt, keyword, geo, s, e)
                if series.empty:
                    print("빈 응답, 건너뜀")
                else:
                    series.to_frame("value").to_csv(cache_path, encoding="utf-8-sig")
                    parts.append(series)
                    print(f"{len(series)}행")
            except Exception as ex:
                print(f"오류: {ex}")

            if i < len(chunks):
                time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print(f"\n[{label}] 중단됨. 수집된 {len(parts)}개 구간까지만 저장합니다.")

    if not parts:
        raise RuntimeError(f"{label} 수집 결과 없음")

    merged = pd.concat(parts)
    merged = merged[~merged.index.duplicated(keep='first')].sort_index()
    merged.index.name = "date"
    df_out = merged.reset_index()
    df_out["date"] = pd.to_datetime(df_out["date"]).dt.normalize()
    df_out = df_out.sort_values("date").reset_index(drop=True)
    print(f"  → 최종 {len(df_out)}행  ({df_out['date'].min().date()} ~ {df_out['date'].max().date()})")
    return df_out


def _cleanup_cache(out_dir: str, label: str):
    """수집 완료 후 캐시 파일 제거"""
    import glob, os
    label_slug = label.replace(" ", "_").replace("/", "_")
    for f in glob.glob(f"{out_dir}/_cache_{label_slug}_*.csv"):
        os.remove(f)


def main():
    try:
        from dateutil.relativedelta import relativedelta  # noqa
    except ImportError:
        raise ImportError("pip install python-dateutil 필요")

    # 글로벌 Bitcoin
    label_all = "글로벌 Bitcoin"
    df_all = collect(keyword="Bitcoin", geo="", label=label_all, out_dir=OUT_DIR)
    df_all.to_csv(f"{OUT_DIR}/bitcoin_all_ver1.csv", index=False, encoding="utf-8-sig")
    _cleanup_cache(OUT_DIR, label_all)
    print(f"저장 완료: bitcoin_all_ver1.csv")

    time.sleep(SLEEP_SEC)

    # 한국 비트코인
    label_kr = "한국 비트코인"
    df_kr = collect(keyword="비트코인", geo="KR", label=label_kr, out_dir=OUT_DIR)
    df_kr.to_csv(f"{OUT_DIR}/bitcoin_kr_ver1.csv", index=False, encoding="utf-8-sig")
    _cleanup_cache(OUT_DIR, label_kr)
    print(f"저장 완료: bitcoin_kr_ver1.csv")


if __name__ == "__main__":
    main()
