# -*- coding: utf-8 -*-
"""
dataset/train_ver1/ 학습 데이터 생성 (3년 확장 버전)
- 수집 기간: 2022-05-01 ~ 2025-05-23
- nav_train.csv : Hash Rate (TH/s), Unique Addresses          ← blockchain.info
- gap_train.csv : Search Interest (value), btc_volatility     ← bitcoin_all_ver1.csv + blockchain.info
- kp_train.csv  : KOSPI_Volatility, volume_btc, bitcoin_kr   ← ^KS11(yfinance) + pyupbit + bitcoin_kr_ver1.csv

KOSPI_Volatility: ^VKOSPI가 yfinance 미지원 → ^KS11 일별 수익률 30일 Rolling Realized Volatility (연환산, %)
"""

import time
import numpy as np
import pandas as pd
import requests
import pyupbit
from pathlib import Path
from datetime import datetime

# ── 기간 설정 ──────────────────────────────────────────────────────────
START_DATE      = datetime(2022, 5, 1)
END_DATE        = datetime(2025, 5, 23)
# btc_volatility RV 계산을 위해 시작일보다 30일 앞서 가격 수집
PRICE_START     = datetime(2022, 3, 1)
RV_WINDOW       = 30

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR    = PROJECT_ROOT / "dataset" / "train_ver1"
TREND_ROOT   = PROJECT_ROOT  # bitcoin_*_ver1.csv 위치


# ── blockchain.info 수집 ───────────────────────────────────────────────
BLOCKCHAIN_CHARTS = {
    "hash-rate":          "Hash Rate (TH/s)",
    "n-unique-addresses": "Unique Addresses",
    "market-price":       "Market Price (USD)",
}

def fetch_blockchain(chart: str, label: str) -> pd.DataFrame:
    url = f"https://api.blockchain.info/charts/{chart}?timespan=1826days&format=json"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"blockchain.info {chart} 실패: HTTP {resp.status_code}")
    values = resp.json().get("values", [])
    df = pd.DataFrame(values)
    df["x"] = pd.to_datetime(df["x"], unit="s")
    df.columns = ["DateTime", label]
    df["Date"] = df["DateTime"].dt.floor("D")
    df = (
        df[["Date", label]]
        .drop_duplicates("Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df = df[(df["Date"] >= pd.Timestamp(PRICE_START)) & (df["Date"] <= pd.Timestamp(END_DATE))]
    return df


# ── KOSPI_Volatility: investing.com VKOSPI 다운로드 파일 ──────────────
def load_kospi_volatility() -> pd.DataFrame:
    path = PROJECT_ROOT / "dataset" / "raw" / "KOSPI Volatility_history.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 컬럼: 날짜, 종가, 시가, 고가, 저가, 거래량, 변동 %
    df.columns = ["Date", "KOSPI_Volatility", "Open", "High", "Low", "Volume", "Change"]
    df["Date"] = pd.to_datetime(df["Date"].str.replace(" ", ""), format="%Y-%m-%d")
    df["KOSPI_Volatility"] = pd.to_numeric(df["KOSPI_Volatility"], errors="coerce")
    df = (
        df[["Date", "KOSPI_Volatility"]]
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )
    df = df[
        (df["Date"] >= pd.Timestamp(START_DATE)) &
        (df["Date"] <= pd.Timestamp(END_DATE))
    ]
    return df


# ── Upbit volume_btc ───────────────────────────────────────────────────
def fetch_volume_btc() -> pd.DataFrame:
    ticker = "KRW-BTC"
    to = None
    all_df = pd.DataFrame()
    while True:
        df = pyupbit.get_ohlcv(ticker=ticker, interval="day", count=200, to=to)
        if df is None or df.empty:
            break
        all_df = pd.concat([df, all_df])
        earliest = df.index.min()
        to = earliest.strftime("%Y-%m-%d %H:%M:%S")
        if earliest <= pd.Timestamp(START_DATE):
            break
    all_df = all_df.reset_index()[["index", "volume"]]
    all_df.columns = ["Date", "volume_btc"]
    all_df["Date"] = pd.to_datetime(all_df["Date"]).dt.normalize()
    all_df = (
        all_df.drop_duplicates("Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    all_df = all_df[
        (all_df["Date"] >= pd.Timestamp(START_DATE)) &
        (all_df["Date"] <= pd.Timestamp(END_DATE))
    ]
    return all_df


# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. blockchain.info 수집 ──
    print("[1] blockchain.info 수집 중...")
    bc_data = {}
    for chart, label in BLOCKCHAIN_CHARTS.items():
        print(f"    {label} ...", end=" ", flush=True)
        bc_data[label] = fetch_blockchain(chart, label)
        print(f"{len(bc_data[label])}행")
        time.sleep(2)

    # ── 2. btc_volatility 계산 ──
    price_df = bc_data["Market Price (USD)"].copy()
    price_df["log_ret"] = np.log(
        price_df["Market Price (USD)"] / price_df["Market Price (USD)"].shift(1)
    )
    price_df["btc_volatility"] = np.sqrt(
        price_df["log_ret"].rolling(RV_WINDOW).apply(lambda x: (x**2).sum(), raw=True)
    )
    btc_rv = price_df[price_df["Date"] >= pd.Timestamp(START_DATE)][
        ["Date", "btc_volatility"]
    ].copy()

    # ── 3. nav_train.csv ──
    print("[2] nav_train.csv 생성 중...")
    hash_df  = bc_data["Hash Rate (TH/s)"][bc_data["Hash Rate (TH/s)"]["Date"] >= pd.Timestamp(START_DATE)]
    addr_df  = bc_data["Unique Addresses"][bc_data["Unique Addresses"]["Date"] >= pd.Timestamp(START_DATE)]
    nav_train = hash_df.merge(addr_df, on="Date", how="inner")
    nav_train = nav_train.sort_values("Date").reset_index(drop=True)
    nav_train.to_csv(TRAIN_DIR / "nav_train.csv", index=False, encoding="utf-8-sig")
    print(f"    저장: nav_train.csv  {len(nav_train)}행  {nav_train['Date'].min().date()} ~ {nav_train['Date'].max().date()}")

    # ── 4. gap_train.csv ──
    print("[3] gap_train.csv 생성 중...")
    bitcoin_all_path = TREND_ROOT / "bitcoin_all_ver1.csv"
    if not bitcoin_all_path.exists():
        raise FileNotFoundError(f"파일 없음: {bitcoin_all_path}\ntrend/fetch_trends.py 먼저 실행하세요.")
    bitcoin_all = pd.read_csv(bitcoin_all_path)
    bitcoin_all["date"] = pd.to_datetime(bitcoin_all["date"]).dt.normalize()
    bitcoin_all = bitcoin_all.rename(columns={"date": "Date", "value": "value"})

    nav_dates = set(nav_train["Date"].dt.normalize())
    gap_train = bitcoin_all[bitcoin_all["Date"].isin(nav_dates)].copy()
    gap_train["Date"] = pd.to_datetime(gap_train["Date"]).dt.normalize()
    btc_rv["Date"] = pd.to_datetime(btc_rv["Date"]).dt.normalize()
    gap_train = gap_train.merge(btc_rv, on="Date", how="left")
    gap_train = gap_train.sort_values("Date").reset_index(drop=True)
    gap_train.to_csv(TRAIN_DIR / "gap_train.csv", index=False, encoding="utf-8-sig")
    print(f"    저장: gap_train.csv  {len(gap_train)}행  {gap_train['Date'].min().date()} ~ {gap_train['Date'].max().date()}")

    # ── 5. KOSPI_Volatility ──
    print("[4] KOSPI_Volatility (investing.com VKOSPI) 로드 중...", end=" ", flush=True)
    kospi_df = load_kospi_volatility()
    print(f"{len(kospi_df)}행")

    # ── 6. volume_btc ──
    print("[5] volume_btc (Upbit) 수집 중...", end=" ", flush=True)
    vol_df = fetch_volume_btc()
    print(f"{len(vol_df)}행")

    # ── 7. bitcoin_kr ──
    print("[6] bitcoin_kr (한국 검색량) 로드 중...", end=" ", flush=True)
    bitcoin_kr_path = TREND_ROOT / "bitcoin_kr_ver1.csv"
    if not bitcoin_kr_path.exists():
        raise FileNotFoundError(f"파일 없음: {bitcoin_kr_path}\ntrend/fetch_trends.py 먼저 실행하세요.")
    bitcoin_kr = pd.read_csv(bitcoin_kr_path)
    bitcoin_kr["date"] = pd.to_datetime(bitcoin_kr["date"]).dt.normalize()
    bitcoin_kr = bitcoin_kr.rename(columns={"date": "Date", "value": "bitcoin_kr"})
    print(f"{len(bitcoin_kr)}행")

    # ── 8. kp_train.csv ──
    print("[7] kp_train.csv 생성 중...")
    kp_train = kospi_df.merge(vol_df, on="Date", how="inner")
    kp_train["Date"] = pd.to_datetime(kp_train["Date"]).dt.normalize()
    bitcoin_kr["Date"] = pd.to_datetime(bitcoin_kr["Date"]).dt.normalize()
    kp_train = kp_train.merge(bitcoin_kr, on="Date", how="left")
    kp_train = kp_train.sort_values("Date").reset_index(drop=True)
    kp_train.to_csv(TRAIN_DIR / "kp_train.csv", index=False, encoding="utf-8-sig")
    print(f"    저장: kp_train.csv   {len(kp_train)}행  {kp_train['Date'].min().date()} ~ {kp_train['Date'].max().date()}")

    # ── 9. 한국 영업일 기준 날짜 정렬 (kp_train 날짜 = 기준) ──
    print("\n[8] 한국 영업일 기준 날짜 정렬 (kp 날짜 기준)...")
    kp_dates = pd.to_datetime(kp_train["Date"]).dt.normalize()
    kp_date_set = set(kp_dates)

    def align_to_kp(df: pd.DataFrame, name: str) -> pd.DataFrame:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        # kp 날짜에 없는 행 제거, 없는 kp 날짜는 ffill로 보완
        df = df.set_index("Date").reindex(sorted(kp_date_set)).ffill().reset_index()
        df = df.rename(columns={"index": "Date"})
        print(f"    {name}: {len(df)}행  {df['Date'].min().date()} ~ {df['Date'].max().date()}")
        return df

    nav_train = align_to_kp(nav_train, "nav_train.csv")
    gap_train = align_to_kp(gap_train, "gap_train.csv")
    kp_train  = align_to_kp(kp_train,  "kp_train.csv")

    nav_train.to_csv(TRAIN_DIR / "nav_train.csv", index=False, encoding="utf-8-sig")
    gap_train.to_csv(TRAIN_DIR / "gap_train.csv", index=False, encoding="utf-8-sig")
    kp_train.to_csv(TRAIN_DIR  / "kp_train.csv",  index=False, encoding="utf-8-sig")

    print(f"\n완료. dataset/train_ver1/ 에 3개 파일 저장됨 (한국 영업일 기준 {len(kp_train)}행).")


if __name__ == "__main__":
    main()
