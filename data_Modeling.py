import requests
import pandas as pd
import os
from functools import reduce
from datetime import datetime
import glob
import numpy as np
import yfinance as yf
import pyupbit

start_date = datetime(2023, 12, 1)
analysis_start = datetime(2024, 1, 12)
end_date = datetime(2025, 5, 23)

chart_names = {
    "n-unique-addresses": "Unique Addresses",
    "market-price": "Market Price (USD)",
    "n-transactions": "Number of Transactions",
    "hash-rate": "Hash Rate (TH/s)",
    "total-bitcoins": "Total Bitcoins in Circulation",
    "output-volume": "Output Volume (BTC)"
}
chart_data = {}

print("\n🔗 Blockchain.com 지표 수집 시작")
for chart, label in chart_names.items():
    url = f"https://api.blockchain.info/charts/{chart}?timespan=1826days&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        raw = response.json()
        values = raw.get("values", [])
        if values:
            df = pd.DataFrame(values)
            df["x"] = pd.to_datetime(df["x"], unit="s")
            df.columns = ["DateTime", label]
            if chart != "total-bitcoins":
                df["Date"] = df["DateTime"].dt.floor("D")
                df = df[["Date", label]]
            else:
                df["Date"] = df["DateTime"].dt.date
                df = df.sort_values(["Date", "DateTime"])
                df = df.groupby("Date").tail(1)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df[["Date", label]]
            df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
            chart_data[label] = df
            print(f"✅ {label} 완료")
        else:
            print(f"⚠️ {label}: 값 없음")
    else:
        print(f"❌ {label} 실패 (HTTP {response.status_code})")

print("\n📂 Google Trends 병합 시작")
trend_files = sorted(glob.glob("data/trend_data/trend_*.csv"))
trends_list = []
for file in trend_files:
    df = pd.read_csv(file, skiprows=1)
    df.columns = ["Date", "Search Interest"]
    df["Date"] = pd.to_datetime(df["Date"])
    trends_list.append(df)
    print(f"✅ {os.path.basename(file)} 완료")
if trends_list:
    trends_df = pd.concat(trends_list)
    trends_df = trends_df.groupby("Date").mean().reset_index()
    chart_data["Google Trends"] = trends_df

print("\n💲 매크로 지표 수집 시작")
macro_tickers = {
    "VIX": "^VIX",
    "Gold": "GLD"
}
for name, ticker in macro_tickers.items():
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[["Close"]].reset_index()
    df.columns = ["Date", name]
    chart_data[name] = df
    print(f"✅ {name} 완료")

## ✅ 환율 데이터 (업로드 파일) 사용
print("\n💱 업로드한 원/달러 환율 데이터 사용")
usd_df = pd.read_csv("data/USD_KRW.csv")
usd_df["Date"] = pd.to_datetime(usd_df["Date"])
usd_df = usd_df[["Date", "USD_KRW"]]
chart_data["USD_KRW"] = usd_df
print("✅ 환율 데이터 대체 완료")

## ✅ Upbit BTC 거래량 슬라이딩 방식 수집
print("\n🇰🇷 Upbit BTC 거래량 수집 (슬라이딩, 날짜 단위 하루 한 건)")
ticker = "KRW-BTC"
to = None
all_df = pd.DataFrame()

while True:
    df = pyupbit.get_ohlcv(ticker=ticker, interval="day", count=200, to=to)
    if df is None or df.empty:
        break
    all_df = pd.concat([df, all_df])
    earliest_date = df.index.min()
    to = earliest_date.strftime("%Y-%m-%d %H:%M:%S")
    if earliest_date <= pd.Timestamp(start_date):
        break

# index 컬럼 = Date
all_df = all_df.reset_index()
all_df = all_df[["index", "volume", "value"]]
all_df.columns = ["Date", "Upbit_Volume_BTC", "Upbit_Volume_KRW"]

# 날짜만 남기기 (시간, 분, 초 제거)
all_df["Date"] = all_df["Date"].dt.date
all_df["Date"] = pd.to_datetime(all_df["Date"])

all_df = all_df.drop_duplicates("Date").sort_values("Date")
all_df = all_df[(all_df["Date"] >= start_date) & (all_df["Date"] <= end_date)]
chart_data["Upbit_Volume"] = all_df
print(f"✅ Upbit 거래량 완료: {len(all_df)}행")


print("\n🧪 병합 시작")
merged_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), chart_data.values())
merged_df.sort_values("Date", inplace=True)

# ✔️ 로그 수익률 및 변동성
merged_df["Log Return"] = np.log(merged_df["Market Price (USD)"] / merged_df["Market Price (USD)"].shift(1))
merged_df["VIX Volatility"] = np.log(merged_df["VIX"] / merged_df["VIX"].shift(1))
merged_df["Gold Volatility"] = np.log(merged_df["Gold"] / merged_df["Gold"].shift(1))

# ✔️ 모든 변수 forward fill
cols_to_fill = merged_df.columns.drop("Date")
merged_df[cols_to_fill] = merged_df[cols_to_fill].ffill()

# ✔️ 불필요한 컬럼 제거
merged_df.drop(columns=["Market Price (USD)"], inplace=True)
merged_df = merged_df[merged_df["Date"] >= analysis_start]

## ✅ IBIT Premium 종속변수 추가 (NAV, Price 기반 계산)
ibit_df = pd.read_csv("data/IBIT Premium.csv")
ibit_df["Date"] = pd.to_datetime(ibit_df["Date"])

# 실제 컬럼명 확인
print("📄 IBIT Premium 컬럼:", ibit_df.columns.tolist())

# NAV, Price 컬럼 자동 탐색
nav_col = [col for col in ibit_df.columns if "NAV" in col][0]
price_col = [col for col in ibit_df.columns if "Price" in col and "NAV" not in col][0]

# 프리미엄 계산
ibit_df["ETF_Premium"] = ((ibit_df[price_col] - ibit_df[nav_col]) / ibit_df[nav_col]) * 100

# 필요한 컬럼만
ibit_df = ibit_df[["Date", "ETF_Premium"]]

# 병합
merged_df = pd.merge(merged_df, ibit_df, on="Date", how="left")

# 병합 후 ETF_Premium 결측값 forward fill
merged_df["ETF_Premium"] = merged_df["ETF_Premium"].ffill()

print("✅ IBIT 프리미엄 계산 및 병합 완료")

print("\n💾 문자열 컬럼 불필요 문자 제거 및 숫자형 변환 (%, 따옴표, 쉼표, 공백)")

for col in merged_df.columns:
    if merged_df[col].dtype == "object":
        merged_df[col] = merged_df[col].str.replace('"', '', regex=False)
        merged_df[col] = merged_df[col].str.replace(",", '', regex=False)
        merged_df[col] = merged_df[col].str.strip()
        if merged_df[col].str.contains("%").any():
            merged_df[col] = merged_df[col].str.replace("%", '', regex=False)
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce") / 100
        else:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

print("✅ 문자열 정리 및 숫자형 변환 완료")

## ✅ 종속변수 파일
target_df = merged_df[["Date", "Log Return", "ETF_Premium"]].copy()
target_df.to_csv("data/target_variables.csv", index=False)
print("✅ 종속변수 저장 완료")

## ✅ 전체 독립변수 파일
feature_columns = [col for col in merged_df.columns if col not in ["Log Return", "ETF_Premium"]]
features_df = merged_df[["Date"] + feature_columns]
features_df.to_csv("data/all_features.csv", index=False)
print("✅ 전체 독립변수 저장 완료")

## ✅ 모델별 독립변수 분리 저장
# NAV 모델
nav_features = ["Hash Rate (TH/s)", "Total Bitcoins in Circulation", "Number of Transactions", 
                "Unique Addresses", "Output Volume (BTC)", "Google Trends", "Gold", "USD_KRW"]
nav_df = merged_df[["Date"] + [col for col in nav_features if col in merged_df.columns]]
nav_df.to_csv("data/nav_features.csv", index=False)
print("✅ NAV 모델 저장 완료")

# 괴리율 모델
gap_features = ["Output Volume (BTC)", "Log Return", "Google Trends", "VIX", "Upbit_Volume_BTC", "Upbit_Volume_KRW"]
gap_df = merged_df[["Date"] + [col for col in gap_features if col in merged_df.columns]]
gap_df.to_csv("data/gap_features.csv", index=False)
print("✅ 괴리율 모델 저장 완료")

# 김치 프리미엄 모델
kimchi_features = ["USD_KRW", "Google Trends", "Upbit_Volume_BTC", "Upbit_Volume_KRW"]
kimchi_df = merged_df[["Date"] + [col for col in kimchi_features if col in merged_df.columns]]
kimchi_df.to_csv("data/kimchi_features.csv", index=False)
print("✅ 김치 프리미엄 모델 저장 완료")

print("\n📁 모든 데이터셋 저장 완료!")
