import pandas as pd

# 1. 글로벌 비트코인 가격 (USD 기준) 불러오기 및 정리
def load_and_clean_global_btc(filepath):
    df = pd.read_csv(filepath)  # CSV 파일 불러오기
    df["Date"] = pd.to_datetime(df["Date"])  # 문자열을 datetime으로 변환
    df = df[["Date", "Market Price (USD)"]]  # 필요한 컬럼만 선택
    df.rename(columns={"Market Price (USD)": "BTC_USD"}, inplace=True)  # 컬럼명 통일
    return df

# 2. USD/KRW 환율 데이터 전처리
def load_and_clean_usd_krw(filepath):
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df["USD_KRW"] = df["USD_KRW"].str.replace(",", "")  # 쉼표 제거
    df["USD_KRW"] = df["USD_KRW"].astype(float)  # 숫자형으로 변환
    return df[["Date", "USD_KRW"]]

# 3. 국내 비트코인 가격 (KRW 기준) 데이터 정리
def load_and_clean_kr_btc(filepath):
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df["BTC_KRW"] = df["Close"].str.replace(",", "")  # 쉼표 제거
    df["BTC_KRW"] = df["BTC_KRW"].astype(float)  # 숫자형 변환
    return df[["Date", "BTC_KRW"]]

# 4. 김치프리미엄 계산: (한국 BTC / (글로벌 BTC * 환율)) - 1
def calculate_kimchi_premium(global_btc_df, usd_krw_df, kr_btc_df):
    merged = pd.merge(global_btc_df, usd_krw_df, on="Date", how="inner")  # 글로벌 BTC + 환율
    merged = pd.merge(merged, kr_btc_df, on="Date", how="inner")  # 위 결과 + 국내 BTC
    merged["Kimchi_Premium"] = (merged["BTC_KRW"] / (merged["BTC_USD"] * merged["USD_KRW"])) - 1
    return merged

# 5. 실행 및 저장
if __name__ == "__main__":
    # 파일 경로에 맞게 설정
    global_btc = load_and_clean_global_btc("data/ETF_variables_v2.csv")
    usd_krw = load_and_clean_usd_krw("data/USD_KRW.csv")
    kr_btc = load_and_clean_kr_btc("data/BTC_KRW.csv")

    # 김치프리미엄 계산
    kimchi_premium_df = calculate_kimchi_premium(global_btc, usd_krw, kr_btc)

    # 결과 출력
    print(kimchi_premium_df.head())

    # 날짜와 김치프리미엄만 저장
    kimchi_only = kimchi_premium_df[["Date", "Kimchi_Premium"]]
    kimchi_only.to_csv("data/kimchi_premium_only.csv", index=False)  # 원하는 경로로 저장
