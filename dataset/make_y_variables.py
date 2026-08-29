# -*- coding: utf-8 -*-
"""
dataset/raw/y_variables.csv 생성 (Log Return 재현 경로 고정)

기존 y_variables.csv는 생성 코드가 없는 수동 산물이었고, 그 탓에
"로그차분 → 주말 필터" 순서 결함(주말 수익률 소실, 실현 드리프트의 71%만 반영)이
장기간 발견되지 않았다. 이 스크립트는 올바른 순서를 코드로 고정한다.

  Log Return      : 거래일 필터 → 로그차분  (data_variables.py:202-213 순서)
                    blockchain.info market-price 원천
  etf_premium     : 수준 비율 (etf_true/nav_true - 1). 결함 없음 → 기존 값 보존
  Kimchi Premium  : 수준 비율. 결함 없음 → 기존 값 보존

보존 컬럼은 재계산하지 않고 원본 텍스트를 그대로 옮겨 bit-exact 동일성을 보장한다.
"""

import sys
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

_ROOT = Path(__file__).resolve().parent.parent
Y_PATH = _ROOT / "dataset" / "raw" / "y_variables.csv"

START = pd.Timestamp("2024-01-12")
END = pd.Timestamp("2025-05-23")
EXPECTED_ROWS = 343
MARKET_PRICE_URL = (
    "https://api.blockchain.info/charts/market-price?timespan=1826days&format=json"
)


def trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """미국 영업일 그리드에서 연방공휴일을 제외한 거래일."""
    grid = pd.date_range(start, end, freq="B")
    holidays = USFederalHolidayCalendar().holidays(start, end)
    return grid[~grid.isin(holidays)]


def previous_trading_day(day: pd.Timestamp) -> pd.Timestamp:
    """day 직전 거래일. 첫 행의 로그차분을 정의하기 위해 1행 확장용."""
    back = trading_days(day - pd.Timedelta(days=10), day)
    return back[back < day][-1]


def fetch_market_price() -> pd.DataFrame:
    """blockchain.info Market Price (USD) 일별 시계열."""
    with urllib.request.urlopen(MARKET_PRICE_URL, timeout=60) as resp:
        payload = json.load(resp)
    df = pd.DataFrame(payload["values"])
    df["Date"] = pd.to_datetime(df["x"], unit="s").dt.normalize()
    df = df[["Date", "y"]].rename(columns={"y": "Market Price (USD)"})
    return df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


def build_log_return(price: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.Series:
    """(a) 거래일 필터 → (b) 로그차분. 순서를 뒤집으면 주말 수익률이 소실된다."""
    prev = previous_trading_day(grid[0])
    keep = pd.DatetimeIndex([prev]).append(grid)

    filtered = price[price["Date"].isin(keep)].sort_values("Date").reset_index(drop=True)
    missing = set(keep) - set(filtered["Date"])
    if missing:
        raise RuntimeError(f"가격 결측 {len(missing)}일: {sorted(missing)[:5]}")

    filtered["Log Return"] = np.log(
        filtered["Market Price (USD)"] / filtered["Market Price (USD)"].shift(1)
    )
    out = filtered[filtered["Date"].isin(grid)].set_index("Date")["Log Return"]
    if out.isna().any():
        raise RuntimeError(f"Log Return NaN {int(out.isna().sum())}건")
    return out


def main() -> int:
    grid = trading_days(START, END)
    assert len(grid) == EXPECTED_ROWS, f"거래일 {len(grid)}개, 기대 {EXPECTED_ROWS}개"

    raw = Y_PATH.read_bytes().decode("utf-8")
    lines = raw.split("\r\n")
    header, body = lines[0], [ln for ln in lines[1:] if ln]
    assert header == "Date,etf_premium,Log Return,Kimchi Premium", f"헤더 불일치: {header}"
    assert len(body) == EXPECTED_ROWS, f"기존 파일 {len(body)}행"

    print("[1] blockchain.info market-price 수집 중...", end=" ", flush=True)
    price = fetch_market_price()
    print(f"{len(price)}행")

    print("[2] 거래일 필터 → 로그차분")
    log_return = build_log_return(price, grid)

    print("[3] 보존 컬럼 대조 및 교체")
    new_body = []
    for line in body:
        date_s, etf_premium, old_lr, kimchi = line.split(",")
        date = pd.Timestamp(date_s)
        assert date in log_return.index, f"거래일 그리드에 없는 날짜: {date_s}"
        new_body.append(f"{date_s},{etf_premium},{log_return[date]:.9f},{kimchi}")

    # 보존 컬럼(Date, etf_premium, Kimchi Premium) bit-exact 동일성 검증
    for old, new in zip(body, new_body):
        o, n = old.split(","), new.split(",")
        assert (o[0], o[1], o[3]) == (n[0], n[1], n[3]), f"보존 컬럼 변경됨: {old} -> {new}"
    print(f"    보존 컬럼 bit-exact 확인: {len(body)}행 x 3컬럼")

    old_sum = sum(float(ln.split(",")[2]) for ln in body[1:])
    new_sum = sum(float(ln.split(",")[2]) for ln in new_body[1:])
    print(f"    누적 로그수익: {old_sum:.6f} -> {new_sum:.6f}")

    Y_PATH.write_bytes(("\r\n".join([header] + new_body) + "\r\n").encode("utf-8"))
    print(f"[4] 저장: {Y_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
