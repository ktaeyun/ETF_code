# =========================================================
# main.py — 데이터 수집(①~⑪) + 보조지표 생성 + 마스터 병합
# 요구: pandas, numpy, requests, yfinance
# 실행 예:
#   python main.py --days 365 --save_csv 1 --out_csv merged_features.csv
# 환경변수(옵션): BOK_API_KEY, FRED_API_KEY
# =========================================================

import os, sys, argparse, time, math
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from typing import List, Optional

# ---------------------------
# 설정(원하면 여기만 바꿔서 씀)
# ---------------------------


CFG = {
    # ETF 티커(미국) — 필요시 추가
    "ETF_TICKER": "IBIT",            # IBIT / FBTC / ARKB 등
    # BTC USD/USDT 소스
    "BTC_CG_ID": "bitcoin",          # CoinGecko id
    # 국내/해외 거래소 거래량(비중 계산용)
    "UPBIT_MARKET": "KRW-BTC",
    "BINANCE_SYMBOL": "BTCUSDT",
    # 환율·금리: FRED 기본, 한국 기준금리는 BOK API 사용
    "FRED_USD_KRW_SERIES": "DEXKOUS",
    "FRED_FEDFUNDS_SERIES": "FEDFUNDS",
    # 수집 기간 (직접 지정)
    "START_DATE": "2024-01-12",
    "END_DATE":   "2025-05-23"
}

# =========================================================
# 유틸
# =========================================================
def _ensure_date(df: pd.DataFrame, col="Date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col]).dt.date
    return out

def _ffill_numeric(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy().sort_values("Date")
    for c in z.columns:
        if c != "Date":
            z[c] = pd.to_numeric(z[c], errors="coerce")
            z[c] = z[c].ffill()
    return z

def _merge_on_date(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    from functools import reduce
    valids = [d for d in dfs if d is not None and len(d)]
    if not valids:
        raise ValueError("병합할 데이터프레임 없음")
    z = reduce(lambda l, r: pd.merge(l, r, on="Date", how="outer"), valids)
    return z.sort_values("Date")

# =========================================================
# ①②③ ETF 종가/거래량 (yfinance) + ② NAV(운용사 페이지 표 파싱)
# =========================================================
def fetch_etf_ohlcv_yf(ticker: str, days: int) -> pd.DataFrame:
    period = f"{max(days, 1)}d"
    df = yf.Ticker(ticker).history(period=period, interval="1d")
    df = df.reset_index()
    df = df.rename(columns={"Date":"Date","Close":"ETF_Close_USD","Volume":"ETF_Volume",
                            "High":"High","Low":"Low"})
    df = df[["Date","ETF_Close_USD","ETF_Volume","High","Low"]]
    return _ensure_date(df)

def fetch_etf_nav_ishares_ibit() -> Optional[pd.DataFrame]:
    """
    IBIT NAV: iShares 공식 페이지의 표를 read_html로 파싱.
    형식이 바뀌면 None을 반환할 수 있음(그땐 운용사 CSV를 직접 내려받아 넣어줘).
    """
    url = "https://www.ishares.com/us/products/333011/ishares-bitcoin-trust-etf"
    try:
        tables = pd.read_html(url)  # 동적으로 바뀌면 실패할 수 있음
        # NAV가 들어있는 테이블 찾기 (열 이름 휴리스틱)
        nav_df = None
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if any("nav" in c for c in cols) and any("date" in c for c in cols):
                nav_df = t.copy()
                break
        if nav_df is None:
            return None
        # 대표적인 컬럼명 정규화
        cdate = [c for c in nav_df.columns if "Date" in str(c) or "date" in str(c)][0]
        cnav  = [c for c in nav_df.columns if "NAV" in str(c).upper()][0]
        nav_df = nav_df.rename(columns={cdate:"Date", cnav:"ETF_NAV_USD"})
        nav_df = nav_df[["Date","ETF_NAV_USD"]]
        nav_df["ETF_NAV_USD"] = pd.to_numeric(nav_df["ETF_NAV_USD"].astype(str).str.replace("[,$]","",regex=True), errors="coerce")
        return _ensure_date(nav_df).dropna()
    except Exception:
        return None

# =========================================================
# ④ BTC/USD (CoinGecko) & ⑤ BTC/KRW (Upbit)
# =========================================================
def fetch_btc_usd_coingecko(days: int) -> pd.DataFrame:
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency":"usd", "days": str(days), "interval":"daily"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    prices = pd.DataFrame(js["prices"], columns=["ts","BTC_USD"])
    prices["Date"] = pd.to_datetime(prices["ts"], unit="ms").dt.date
    return prices[["Date","BTC_USD"]]

def fetch_upbit_daily_krw_btc(days: int, market="KRW-BTC") -> pd.DataFrame:
    # Upbit 일봉은 최대 200개/호출 → 루프
    out = []
    url = "https://api.upbit.com/v1/candles/days"
    remaining = days
    to = None
    while remaining > 0:
        cnt = min(200, remaining)
        params = {"market": market, "count": cnt}
        if to: params["to"] = to
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        arr = r.json()
        if not arr: break
        df = pd.DataFrame(arr)
        df["Date"] = pd.to_datetime(df["candle_date_time_utc"]).dt.date
        df = df.rename(columns={"trade_price":"BTC_KRW"})
        out.append(df[["Date","BTC_KRW"]])
        # 다음 페이지 anchor
        oldest = arr[-1]["candle_date_time_utc"]
        to = oldest
        remaining -= cnt
        time.sleep(0.15)  # rate-limit 완화
    z = pd.concat(out, ignore_index=True).drop_duplicates().sort_values("Date")
    return z

# =========================================================
# ⑥ 환율(USD/KRW) & ⑩ 한-미 금리차(미국 금리)
#    - FRED 기본(키 필요), 한국 기준금리는 BOK API로
# =========================================================
def fetch_fred_series(series_id: str, api_key: str, days: int) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    obs = pd.DataFrame(r.json()["observations"])
    obs["Date"] = pd.to_datetime(obs["date"]).dt.date
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    # 최근 days만 슬라이싱
    obs = obs[["Date","value"]].dropna().sort_values("Date")
    if days:
        obs = obs.iloc[-days:]
    obs = obs.rename(columns={"value": series_id})
    return obs

def fetch_bok_base_rate(api_key: str, start: str, end: str) -> pd.DataFrame:
    """
    한국은행 ECOS: 기준금리(예: 통계표 코드 ‘722Y001’ 등)를 사용하는 것이 일반적.
    실제 사용 통계표 코드는 계정에 따라 다를 수 있어, 여기선 예시 엔드포인트를 제시하고
    불일치 시 코드만 바꿔주면 됨.
    """
    # 아래 통계코드는 계정 환경에 맞게 교체 필요. (예시는 월별 기준금리 가정)
    stat_code = "722Y001"   # 예시(월), 실제 계정에서 확인 필요
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/9999/{stat_code}/M/{start}/{end}/0101000"
    r = requests.get(url, timeout=30); r.raise_for_status()
    js = r.json()
    rows = js.get("StatisticSearch", {}).get("row", [])
    if not rows: return pd.DataFrame(columns=["Date","KR_PolicyRate"])
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["TIME"]).dt.date
    df["KR_PolicyRate"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    return df[["Date","KR_PolicyRate"]].dropna()

# =========================================================
# ⑧ ETF 자금 유출입(Farside) — HTML 표 파싱
# =========================================================
def fetch_btc_etf_flows_farside() -> Optional[pd.DataFrame]:
    try:
        url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
        tables = pd.read_html(url)
        # 첫 테이블이 날짜+티커별 flows
        df = tables[0].copy()
        df = df.rename(columns={"Date":"Date"})
        # 총합 혹은 특정 티커 합산
        flow_cols = [c for c in df.columns if c not in ["Date","BTC","Total"]]
        df["Flow_USD"] = df[flow_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df[["Date","Flow_USD"]].dropna()
    except Exception:
        return None

# =========================================================
# ⑨ 국내·해외 거래소 거래대금(비중 계산용)
#    - Upbit: 위에서 KRW-BTC 캔들 응답에 거래대금이 없음 → 추가 엔드포인트가 필요.
#      간편화를 위해 Binance의 BTCUSDT 일일 캔들에서 거래대금 quoteVolume 사용.
# =========================================================
def fetch_binance_daily_quote_volume(symbol="BTCUSDT", days=365) -> pd.DataFrame:
    # Binance 현물 public endpoint: /api/v3/klines
    # weight 고려: 한 번 호출로 1000개까지
    limit = min(1000, days)
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval":"1d", "limit": limit}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    arr = r.json()
    cols = ["open_time","o","h","l","c","volume","close_time","quote_volume","n","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(arr, columns=cols)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms").dt.date
    df["Vol_USD"] = pd.to_numeric(df["quote_volume"], errors="coerce")  # USDT≈USD 가정
    return df[["Date","Vol_USD"]].dropna().sort_values("Date")

def fetch_upbit_daily_value(market="KRW-BTC", days=365) -> Optional[pd.DataFrame]:
    # Upbit 일별 거래대금은 별도 제공이 제한적. 간단히 일봉의 거래량*종가 근사로 생성.
    url = "https://api.upbit.com/v1/candles/days"
    out = []; remaining = days; to = None
    while remaining > 0:
        cnt = min(200, remaining)
        params = {"market": market, "count": cnt}
        if to: params["to"] = to
        r = requests.get(url, params=params, timeout=30); r.raise_for_status()
        arr = r.json()
        if not arr: break
        df = pd.DataFrame(arr)
        df["Date"] = pd.to_datetime(df["candle_date_time_utc"]).dt.date
        # 거래대금 근사: 누적 거래금액(trade_price * candle_acc_trade_volume) 등 제공시 직접 사용
        if "candle_acc_trade_price" in df.columns:
            df["Vol_KRW"] = pd.to_numeric(df["candle_acc_trade_price"], errors="coerce")
        else:
            # 보수적 근사(정보 없을 때는 None)
            df["Vol_KRW"] = np.nan
        out.append(df[["Date","Vol_KRW"]])
        oldest = arr[-1]["candle_date_time_utc"]; to = oldest; remaining -= cnt
        time.sleep(0.15)
    if not out: return None
    z = pd.concat(out, ignore_index=True).drop_duplicates().sort_values("Date")
    return z

# =========================================================
# ⑦ 변동성, ⑪ 유동성 지표
# =========================================================
def compute_realized_vol(df_btc: pd.DataFrame, price_col="BTC_USD", window=30) -> pd.DataFrame:
    x = _ensure_date(df_btc)
    x = x.sort_values("Date").copy()
    x["ret"] = np.log(x[price_col]).diff()
    x[f"RV_{window}d"] = x["ret"].rolling(window).std() * np.sqrt(365)
    return x[["Date", f"RV_{window}d"]]

def compute_turnover(etf_df: pd.DataFrame, shares_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if shares_df is None: return None
    x = etf_df.merge(shares_df, on="Date", how="left")
    if "Shares_Outstanding" not in x.columns: return None
    mcap = x["ETF_Close_USD"] * x["Shares_Outstanding"]
    turnover = (x["ETF_Close_USD"] * x["ETF_Volume"]) / (mcap + 1e-12)
    return pd.DataFrame({"Date": x["Date"], "Turnover": turnover})

def compute_amihud(etf_df: pd.DataFrame) -> pd.DataFrame:
    x = etf_df.copy().sort_values("Date")
    x["ret_abs"] = x["ETF_Close_USD"].pct_change().abs()
    traded_value = x["ETF_Close_USD"] * x["ETF_Volume"]
    x["Amihud"] = x["ret_abs"] / (traded_value + 1e-12)
    return x[["Date","Amihud"]]

def approximate_spread(etf_df: pd.DataFrame) -> pd.DataFrame:
    x = etf_df.copy().sort_values("Date")
    H, L = x["High"].astype(float), x["Low"].astype(float)
    beta = (np.log(H/L))**2
    beta1 = (np.log(H.shift(-1)/L.shift(-1)))**2
    gamma = beta + beta1
    alpha = (np.sqrt(2*beta) - np.sqrt(gamma))**2
    spread = 2*(np.exp(alpha)-1)/(1+np.exp(alpha))
    return pd.DataFrame({"Date": x["Date"], "Approx_Spread": spread})

def compute_volume_share(df_kr: Optional[pd.DataFrame], df_us: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df_kr is None or df_us is None: return None
    x = df_kr.merge(df_us, on="Date", how="inner")
    x["Vol_Share_KR"] = x["Vol_KRW"] / (x["Vol_KRW"] + x["Vol_USD"] + 1e-12)
    return x[["Date","Vol_Share_KR"]]

# =========================================================
# GAP 모델용 거래량/유동성 파생 피처
# =========================================================
def build_etf_liquidity_features(etf_df: pd.DataFrame,
                                 win_short: int = 5,
                                 win_long: int = 20) -> pd.DataFrame:
    """
    입력: fetch_etf_ohlcv_yf() 결과 (Date, ETF_Close_USD, ETF_Volume, High, Low)
    출력: GAP 모델 투입용 거래량/유동성 파생 피처
    """
    x = etf_df.copy().sort_values("Date")
    # 1) 달러 거래대금
    x["ETF_DollarValue"] = x["ETF_Close_USD"] * x["ETF_Volume"]

    # 2) 거래량 정규화(회전/레짐 감지에 유용)
    x["ETF_Volume_MA_s"] = x["ETF_Volume"].rolling(win_short, min_periods=1).mean()
    x["ETF_Volume_MA_l"] = x["ETF_Volume"].rolling(win_long,  min_periods=1).mean()
    x["ETF_Volume_Z"]    = (x["ETF_Volume"] - x["ETF_Volume_MA_l"]) / (x["ETF_Volume"].rolling(win_long, min_periods=1).std() + 1e-12)
    x["ETF_DollarValue_MA_l"] = x["ETF_DollarValue"].rolling(win_long, min_periods=1).mean()
    x["ETF_DollarValue_Z"]    = (x["ETF_DollarValue"] - x["ETF_DollarValue_MA_l"]) / (x["ETF_DollarValue"].rolling(win_long, min_periods=1).std() + 1e-12)

    # 3) 변동성 대비 거래대금(가격 급변 시 유동성 취약성 포착)
    x["ETF_Return_abs"] = x["ETF_Close_USD"].pct_change().abs()
    x["ILLIQ_proxy"]    = x["ETF_Return_abs"] / (x["ETF_DollarValue"] + 1e-12)  # Amihud 대용량 지표

    # 4) (선택) 스프레드 근사 추가 결합
    # 이미 approximate_spread(etf_df) 가 있으므로 메인에서 merge

    keep = ["Date", "ETF_Volume", "ETF_DollarValue",
            "ETF_Volume_MA_s", "ETF_Volume_MA_l", "ETF_Volume_Z",
            "ETF_DollarValue_MA_l", "ETF_DollarValue_Z",
            "ETF_Return_abs", "ILLIQ_proxy"]
    return x[keep]


# =========================================================
# 메인
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_csv", type=int, default=1)
    ap.add_argument("--out_csv", type=str, default="merged_features.csv")
    ap.add_argument("--etf_ticker", type=str, default=CFG["ETF_TICKER"])
    args = ap.parse_args()

    # ✅ 기간 계산 (CFG 기반)
    start = pd.to_datetime(CFG["START_DATE"]).date()
    end   = pd.to_datetime(CFG["END_DATE"]).date()
    days  = (end - start).days

    fred_key = os.getenv("FRED_API_KEY", "")
    bok_key  = os.getenv("BOK_API_KEY", "")

    # 1) ETF: 가격/거래량/고저
    etf = fetch_etf_ohlcv_yf(args.etf_ticker, days)

    # 2) ETF NAV (가능하면 iShares 파싱; 실패시 None)
    nav = fetch_etf_nav_ishares_ibit()

    # 5) 환율(USD/KRW), 6) 미 기준금리(FEDFUNDS)
    fx = None
    fed = None
    if fred_key:
        fx  = fetch_fred_series(CFG["FRED_USD_KRW_SERIES"], fred_key, days)
        fed = fetch_fred_series(CFG["FRED_FEDFUNDS_SERIES"], fred_key, days)
        if fx is not None: fx = fx.rename(columns={CFG["FRED_USD_KRW_SERIES"]:"USD_KRW"})
        if fed is not None: fed = fed.rename(columns={CFG["FRED_FEDFUNDS_SERIES"]:"US_FedFunds"})
    else:
        print("[WARN] FRED_API_KEY 없음 → 환율/미금리 생략")

    # 7) 한국 기준금리(BOK)
    kr_rate = None
    if bok_key:
        # 날짜 범위(월지표) — 대략 days 범위를 커버
        end = pd.Timestamp.today().strftime("%Y%m%d")
        start = (pd.Timestamp.today() - pd.Timedelta(days=days*2)).strftime("%Y%m%d")
        kr_rate = fetch_bok_base_rate(bok_key, start, end)
    else:
        print("[WARN] BOK_API_KEY 없음 → 한국 기준금리 생략")

    # 8) ETF 자금 유출입(Farside)
    flows = fetch_btc_etf_flows_farside()

    # 9) 국내/해외 거래대금
    kr_val = fetch_upbit_daily_value(days=days)
    us_val = fetch_binance_daily_quote_volume(days=days)

    # === 보조지표(변동성/유동성/비중) ===

    amihud = compute_amihud(etf)
    spread = approximate_spread(etf)
    vol_share = compute_volume_share(kr_val, us_val)
    etf_liq = build_etf_liquidity_features(etf)

    # === 병합 ===
    pieces = [etf, nav, fx, fed, kr_rate, flows, kr_val, us_val, amihud, spread, vol_share, etf_liq]
    Z = _merge_on_date([p for p in pieces if p is not None])
    Z = _ffill_numeric(Z)

    # 파생: 괴리율/김치프리미엄
    if {"ETF_Close_USD","ETF_NAV_USD"}.issubset(Z.columns):
        Z["GAP_pct"] = (Z["ETF_Close_USD"] - Z["ETF_NAV_USD"]) / (Z["ETF_NAV_USD"] + 1e-12) * 100
    if {"BTC_KRW","USD_KRW","BTC_USD"}.issubset(Z.columns):
        Z["Kimchi_Premium_pct"] = ((Z["BTC_KRW"]/(Z["USD_KRW"]+1e-12)) - Z["BTC_USD"]) / (Z["BTC_USD"]+1e-12) * 100

    if args.save_csv:
        Z.to_csv(args.out_csv, index=False)
        print(f"[OK] 저장: {args.out_csv}  rows={len(Z)}")
    else:
        print(Z.tail(5).to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
