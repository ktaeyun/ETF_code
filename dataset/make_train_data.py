# -*- coding: utf-8 -*-
"""
dataset/train용 학습 데이터 생성
- nav_train.csv: nav_variables에서 Hash Rate, Unique Addresses (동일 기간 그대로)
- gap_train.csv: trend/merged_bitcoin_all.csv → nav_train과 동일 기간, + btc_volatility (data_Modeling.py와 동일한 수집·Log Return으로 Rolling Realized Volatility, window=30, 2024-01-12부터 유효)
- kp_train.csv: kp_variables의 KOSPI_Volatility, volume_btc + trend/merged_bitcoin_kr.csv, kp 기간에 맞춰 저장
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None

# Rolling Realized Volatility 윈도우 (data_Modeling.py의 Log Return 방식 사용, window=30)
# Log Return 정의: data_Modeling.py와 동일 = np.log(Price / Price.shift(1))
RV_WINDOW = 30
# 실현변동성 산출 시 2024-01-12부터 유효하도록, 그 이전부터 가격 수집 (최소 RV_WINDOW일 이상)
PRICE_START_DATE = datetime(2023, 11, 1)  # 2024-01-12 기준 약 42거래일 이전
ANALYSIS_START = datetime(2024, 1, 12)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TREND_DIR = PROJECT_ROOT / "trend"


def fetch_market_price_extended(end_date=None):
    """
    data_Modeling.py와 동일한 방식으로 blockchain.info에서 Market Price (USD) 수집.
    PRICE_START_DATE부터 수집하여 2024-01-12부터 30일 롤링 실현변동성이 나오도록 함.
    """
    if requests is None:
        raise ImportError("실현변동성 수집을 위해 pip install requests 필요")
    end_date = end_date or datetime(2025, 5, 23)
    url = "https://api.blockchain.info/charts/market-price?timespan=1826days&format=json"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"blockchain.info API 실패: HTTP {resp.status_code}")
    raw = resp.json()
    values = raw.get("values", [])
    if not values:
        raise RuntimeError("blockchain.info에서 market-price 값 없음")
    df = pd.DataFrame(values)
    df["x"] = pd.to_datetime(df["x"], unit="s")
    df.columns = ["DateTime", "Market Price (USD)"]
    df["Date"] = df["DateTime"].dt.floor("D")
    df = df[["Date", "Market Price (USD)"]]
    df = df[(df["Date"] >= pd.Timestamp(PRICE_START_DATE)) & (df["Date"] <= pd.Timestamp(end_date))]
    df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return df


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # ----- nav_train.csv: Hash Rate, Unique Addresses -----
    nav_raw = pd.read_csv(RAW_DIR / "nav_variables.csv")
    nav_raw["Date"] = pd.to_datetime(nav_raw["Date"])
    nav_train = nav_raw[["Date", "Hash Rate (TH/s)", "Unique Addresses"]].copy()
    nav_train = nav_train.sort_values("Date").dropna().reset_index(drop=True)
    nav_train.to_csv(TRAIN_DIR / "nav_train.csv", index=False, encoding="utf-8-sig")
    print(f"nav_train.csv: {len(nav_train)} rows, {nav_train['Date'].min()} ~ {nav_train['Date'].max()}")

    # ----- gap_train.csv: merged_bitcoin_all (nav와 동일 기간) + btc_volatility -----
    # btc_volatility = data_Modeling.py와 동일한 수집(blockchain.info Market Price) → Log Return → Rolling Realized Volatility (window=30)
    # 더 이전(PRICE_START_DATE)부터 가격 수집하여 2024-01-12부터 실현변동성 유효
    print("  Market Price (USD) 수집 (blockchain.info, data_Modeling.py와 동일)...")
    price_df = fetch_market_price_extended()
    price_df["Log Return"] = np.log(
        price_df["Market Price (USD)"] / price_df["Market Price (USD)"].shift(1)
    )
    rv = np.sqrt(
        price_df["Log Return"].rolling(RV_WINDOW).apply(lambda x: (x ** 2).sum(), raw=True)
    )
    price_df["btc_volatility"] = rv
    # 분석 시작일(2024-01-12) 이후만 사용 (이 시점부터 RV 유효)
    btc_rv_df = price_df[price_df["Date"] >= pd.Timestamp(ANALYSIS_START)][
        ["Date", "btc_volatility"]
    ].copy()

    bitcoin_all = pd.read_csv(TREND_DIR / "merged_bitcoin_all.csv")
    bitcoin_all["date"] = pd.to_datetime(bitcoin_all["date"])
    nav_dates = set(nav_train["Date"].dt.normalize())
    gap_train = bitcoin_all[bitcoin_all["date"].dt.normalize().isin(nav_dates)].copy()
    gap_train = gap_train.rename(columns={"date": "Date"})[["Date", "value"]]
    gap_train["Date"] = pd.to_datetime(gap_train["Date"]).dt.normalize()
    btc_rv_df["Date"] = pd.to_datetime(btc_rv_df["Date"]).dt.normalize()
    gap_train = gap_train.merge(btc_rv_df, on="Date", how="left")
    gap_train = gap_train.sort_values("Date").reset_index(drop=True)
    gap_train.to_csv(TRAIN_DIR / "gap_train.csv", index=False, encoding="utf-8-sig")
    print(f"gap_train.csv: {len(gap_train)} rows (nav와 동일 기간), btc_volatility 추가 (RV window={RV_WINDOW}d)")

    # ----- kp_train.csv: KOSPI_Volatility, volume_btc + merged_bitcoin_kr, kp 기간에 맞춤 -----
    kp_raw = pd.read_csv(RAW_DIR / "kp_variables.csv")
    kp_raw["Date"] = pd.to_datetime(kp_raw["Date"])
    bitcoin_kr = pd.read_csv(TREND_DIR / "merged_bitcoin_kr.csv")
    bitcoin_kr["date"] = pd.to_datetime(bitcoin_kr["date"])

    kp_train = kp_raw[["Date", "KOSPI_Volatility", "volume_btc"]].copy()
    kp_train = kp_train.merge(
        bitcoin_kr.rename(columns={"date": "Date", "value": "bitcoin_kr"})[["Date", "bitcoin_kr"]],
        on="Date",
        how="left",
    )
    kp_train = kp_train.sort_values("Date").reset_index(drop=True)
    kp_train.to_csv(TRAIN_DIR / "kp_train.csv", index=False, encoding="utf-8-sig")
    print(f"kp_train.csv: {len(kp_train)} rows (kp 기간 기준), {kp_train['Date'].min()} ~ {kp_train['Date'].max()}")
    print("Done. dataset/train/ with nav_train.csv, gap_train.csv, kp_train.csv")


if __name__ == "__main__":
    main()
