import requests
import pandas as pd
import os
from functools import reduce
from datetime import datetime
import glob
import numpy as np
import yfinance as yf
import holidays

# 날짜 범위 설정
start_date = datetime(2024, 1, 11) # 2022년 1월 1일 수익률 계산을 위해 2021년 12월 31일 부터 데이터 수집
analysis_start = datetime(2024, 1, 12) # 실제 분석하고자 하는 날짜
end_date = datetime(2025, 5, 23)

# Blockchain.com 차트 설정
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

# 1. Blockchain.com 데이터 수집
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
            chart_data[chart] = df
            print(f"✅ {label} 수집 및 필터링 완료")
        else:
            print(f"⚠️ {label}: 값 없음")
    else:
        print(f"❌ {label} 수집 실패 (HTTP {response.status_code})")

# 2. Google Trends 수동 다운로드 병합
print("\n📂 수동 다운로드 Google Trends 병합 시작")

# trend_1.csv ~ trend_13.csv 경로 지정
trend_files = sorted(glob.glob("data/trend_data/trend_**.csv"))  # 폴더에 trend_1.csv 등 존재해야 함
trends_list = []

for file in trend_files:
    try:
        df = pd.read_csv(file, skiprows=1)  # 첫 줄은 메타데이터라 건너뜀
        df.columns = ["Date", "Search Interest"]
        df["Date"] = pd.to_datetime(df["Date"])
        trends_list.append(df)
        print(f"✅ {os.path.basename(file)} 병합 완료")
    except Exception as e:
        print(f"❌ {file} 병합 실패 - {e}")

if trends_list:
    trends_df = pd.concat(trends_list)
    trends_df = trends_df.groupby("Date").mean().reset_index()  # 날짜 기준 평균 (혹은 다른 방법)
    chart_data["Investor Attention"] = trends_df
    print("✅ Investor Attention 전체 병합 완료")
else:
    print("⚠️ Investor Attention 데이터 없음")    

# 2-1. ETF 검색량 데이터 처리 (etf_*.csv)
print("\n📂 ETF 검색량 데이터 병합 시작")

etf_files = sorted(glob.glob("data/etf_trend_data/etf_*.csv"))  # etf_*.csv 위치 확인
etf_list = []

for file in etf_files:
    try:
        df = pd.read_csv(file, skiprows=1)
        df.columns = ["Date", "ETF Search Interest"]
        df["Date"] = pd.to_datetime(df["Date"])
        etf_list.append(df)
        print(f"✅ {os.path.basename(file)} 병합 완료")
    except Exception as e:
        print(f"❌ {file} 병합 실패 - {e}")

if etf_list:
    etf_df = pd.concat(etf_list)
    etf_df = etf_df.groupby("Date").mean().reset_index()
    chart_data["ETF Attention"] = etf_df
    print("✅ ETF Attention 전체 병합 완료")
else:
    print("⚠️ ETF Attention 데이터 없음")

print("\n📂 김치프리미엄 데이터 처리 시작")

def clean_price_df(path, date_col="Date", value_col=None, new_col=None):
    df = pd.read_csv(path)
    df.replace({",": "", '"': ""}, regex=True, inplace=True)
    if value_col:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        if new_col:
            df = df.rename(columns={value_col: new_col})
    df[date_col] = pd.to_datetime(df[date_col])
    return df[[date_col, new_col if new_col else value_col]]

# 파일 경로
upbit_path = "data/BTC_KRW Upbit.csv"
binance_path = "data/BTC_USD Binance.csv"
usdkrw_path = "data/USD_KRW.csv"

# 데이터 정리
btc_upbit = clean_price_df(upbit_path, value_col="종가", new_col="BTC_KRW_Upbit")
btc_usd = clean_price_df(binance_path, value_col="종가", new_col="BTC_USD_Binance")
usdkrw = clean_price_df(usdkrw_path, value_col="USD_KRW", new_col="USD_KRW")

# 병합
kimchi_df = btc_upbit.merge(btc_usd, on="Date", how="inner")
kimchi_df = kimchi_df.merge(usdkrw, on="Date", how="inner")

# 김치 프리미엄 계산
kimchi_df["Kimchi Premium"] = kimchi_df["BTC_KRW_Upbit"] / (kimchi_df["BTC_USD_Binance"] * kimchi_df["USD_KRW"]) - 1

# chart_data에 추가
chart_data["Kimchi Premium"] = kimchi_df[["Date", "Kimchi Premium"]]

print("✅ 김치프리미엄 계산 및 chart_data 병합 완료")

# 유가 - EIA API 요청
print("\n🛢️ 유가 현물 가격 수집 (EIA v2)")
EIA_API_KEY = "FpNg8LsFhnTzQ6Dlio1nnAXiWvhxDaG4xi0bGEwN"  # 본인의 EIA API 키 입력
url = f"https://api.eia.gov/v2/seriesid/PET.RWTC.D?api_key={EIA_API_KEY}"

response = requests.get(url)

if response.status_code == 200:
    raw = response.json()
    records = raw.get("response", {}).get("data", [])

    if not records:
        print("⚠️ 응답은 성공했지만 데이터가 없습니다.")
    else:
        df_oil = pd.DataFrame(records)

        # 컬럼 확인
        if "period" not in df_oil.columns or "value" not in df_oil.columns:
            print("❌ 'period' 또는 'value' 컬럼이 없습니다:", df_oil.columns.tolist())
        else:
            df_oil["period"] = pd.to_datetime(df_oil["period"])
            df_oil = df_oil.rename(columns={"period": "Date", "value": "WTI_Spot_Price"})
            df_oil = df_oil[["Date", "WTI_Spot_Price"]]
            df_oil = df_oil.sort_values("Date")

            # 날짜 필터링
            df_oil = df_oil[(df_oil["Date"] >= start_date) & (df_oil["Date"] <= end_date)]

            print(f"✅ 유가 데이터 수집 성공: {len(df_oil)}개 행")
            print(df_oil.head())

            # 병합 대상에 추가
            chart_data["oil"] = df_oil
else:
    print("❌ 유가 데이터 수집 실패")
    print("📩 응답 코드:", response.status_code)
    print("📩 응답 내용:", response.text)
# 4. 달러 인덱스 & VIX 수집 (yfinance)
print("\n💲 미국 달러 지수 및 VIX 수집 시작")
macro_tickers = {
    "USD_Index": "DX-Y.NYB",
    "VIX": "^VIX",
    "Gold": "GLD"
}
for name, ticker in macro_tickers.items():
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[["Close"]].reset_index()
    df.columns = ["Date", name]
    chart_data[name] = df
    print(f"✅ {name} 수집 완료")

print("\n🧪 병합 전 각 데이터프레임의 컬럼 확인:")
for key, df in chart_data.items():
    print(f"🔍 {key}: {df.columns.tolist()}")
    if "Date" not in df.columns:
        print(f"❌ '{key}' 데이터프레임에 'Date' 컬럼이 없습니다.")

# 5. 병합 및 정리
merged_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), chart_data.values())
merged_df.sort_values("Date", inplace=True)
# 1) 주말 제거 (토: 5, 일: 6)
merged_df = merged_df[merged_df["Date"].dt.dayofweek < 5]

# 2) 미국 증시 공휴일 제거
us_holidays = holidays.US(years=range(start_date.year, end_date.year + 1))
merged_df = merged_df[~merged_df["Date"].isin(us_holidays)]

# 3) 거래일인데 일부 값이 NaN → 보간
merged_df = merged_df.interpolate(method="linear")

# ✔️ 로그 수익률 계산
merged_df["Log Return"] = np.log(
    merged_df["Market Price (USD)"] / merged_df["Market Price (USD)"].shift(1)
) 
# VIX 변동성 지수
merged_df["VIX Volatility"] = np.log(
    merged_df["VIX"] / merged_df["VIX"].shift(1)
)
# 달러 인덱스 변동성
merged_df["USD Volatility"] = np.log(
    merged_df["USD_Index"] / merged_df["USD_Index"].shift(1)
)
# 금 가격 변동성
merged_df["Gold Volatility"] = np.log(
    merged_df["Gold"] / merged_df["Gold"].shift(1)
)
# 유가 변동성
merged_df["WTI Volatility"] = np.log(
    merged_df["WTI_Spot_Price"] / merged_df["WTI_Spot_Price"].shift(1)
)

# 가격 데이터는 수익률 구하기 위해서만 사용
merged_df.drop(columns=[
    "WTI_Spot_Price",
    "USD_Index",
    "VIX",
    "Gold"], inplace=True)

# ✔️ 저장 시 필터링
merged_df = merged_df[merged_df["Date"] >= analysis_start]

# 6. 저장
save_folder = "data"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, "ETF_variables_v5.csv")
merged_df.to_csv(save_path, index=False)

print(f"\n📁 저장 완료: {save_path}")
print(f"📆 총 데이터 일 수: {len(merged_df)}일")

