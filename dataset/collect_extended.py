# -*- coding: utf-8 -*-
"""
확장 표본 재수집 (2026-06-30까지). 기존 dataset/raw, dataset/train은 건드리지 않는다.

출력
  dataset/raw_ext/        nav_variables / gap_variables / kp_variables / y_variables / y_true_variables
  dataset/train_ext/      nav_train / gap_train_main / kp_train_main        (메인 표본 2024-01-12~)
  dataset/train_ver1_ext/ nav_train / gap_train / kp_train                  (레짐 표본 2022-05-02~)

자동 수집 소스
  blockchain.info  Unique Addresses, Number of Transactions, Hash Rate,
                   Total Bitcoins in Circulation, Output Volume, Market Price
  Upbit + Bithumb  volume_btc, trade_value_krw (두 거래소 KRW-BTC 합계)
  Upbit            BTC_KRW 종가 (김치프리미엄 분자)
  Binance          BTCUSDT 일별 종가 (김치프리미엄 분모)
  FRED DEXKOUS     USD_KRW
  yfinance         ^VIX (VIX Volatility), IBIT (etf_true)
  pytrends         Bitcoin(global) / 비트코인(KR) 일별 검색관심도
                   3개월 청크는 각자의 최대값으로 정규화되어 청크마다 배율이 다르다
                   (실측 0.975~1.430). 전 구간 1회 요청으로 받은 주간 시계열을 전역
                   기준자로 삼아 청크별 배율을 원점통과 최소제곱으로 되돌린다.
                   검증: 청크 경계가 다른 두 시계열의 수준 상관이
                         global 0.691 -> 0.922, KR 0.632 -> 0.802.
                   보정 후에는 값이 0~100을 넘을 수 있다(상대지수이므로 무방).
                   --no-anchor 로 끄면 기존 fetch_trends.py와 같은 단순 연결이 된다.

수동 입력  dataset/manual_ext/  (없으면 해당 컬럼은 비운 채로 저장하고 안내 출력)
  KOSPI_Volatility.csv   investing.com VKOSPI 일별 시세
  IBIT_NAV.csv           iShares IBIT 일별 NAV
자동화를 시도했으나 모두 막혀 있다: KRX getJsonData는 지수 finder에 VKOSPI가 없고,
investing.com은 403, iShares는 SPA 전환으로 .ajax CSV 엔드포인트가 사라졌다.
--kospi-proxy 를 주면 VKOSPI 대신 ^KS11 30일 실현변동성(연환산 %)으로 채운다.
어디까지나 대용치이므로 기존 결과와의 비교에는 실제 VKOSPI 파일을 쓰는 편이 낫다.

기존 dataset/raw 대비 소스 검증 결과
  - volume_btc : raw/kp_variables.csv는 Upbit + Bithumb KRW-BTC 합계임을 확인했다.
    2024-01-12 / 2024-01-16 / 2025-05-23 세 날짜 모두 bit-exact 일치
    (예: 11517.516021 + 13982.97116905 = 25500.48719051). 여기서도 동일하게 합산한다.
    다만 train_ver1/kp_train.csv는 Upbit 단독이었다 — 기존 프로젝트의 메인/레짐 표본
    불일치이며, 확장판은 양쪽 모두 Upbit+Bithumb으로 통일한다.
  - trade_value_krw : 두 거래소의 candle_acc_trade_price 합계를 쓴다. 기존 값과
    -0.03% ~ +1.06% 차이가 남는데(2024-01-12이 최대), 원본이 어떤 가중으로 집계했는지는
    재현되지 않았다. 수준·변동 패턴은 사실상 동일하다.
  - USD_KRW : investing.com → FRED DEXKOUS (2024-01-12 기준 1313.22 vs 1313.41)
  - nav_variables의 Search Interest : 원천 미상의 수동 trend_data/*.csv → Google Trends 'Bitcoin'(global).
    다만 이 컬럼은 시뮬레이터 경로에서 쓰이지 않는다(GAP은 train의 value 컬럼을 쓴다).
"""

import json
import time
import urllib.request
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# ── 기간 ──────────────────────────────────────────────────────────────
END          = pd.Timestamp("2026-06-30")
START_MAIN   = pd.Timestamp("2024-01-12")   # 메인 표본 시작 (IBIT 상장일)
START_VER1   = pd.Timestamp("2022-05-01")   # 레짐 표본 시작
PRICE_START  = pd.Timestamp("2022-03-01")   # RV30 워밍업용 가격 수집 시작
RV_WINDOW    = 30

ROOT        = Path(__file__).resolve().parent.parent
RAW_EXT     = ROOT / "dataset" / "raw_ext"
TRAIN_EXT   = ROOT / "dataset" / "train_ext"
TRAINV1_EXT = ROOT / "dataset" / "train_ver1_ext"
MANUAL      = ROOT / "dataset" / "manual_ext"
CACHE       = RAW_EXT / "_cache"

BLOCKCHAIN_CHARTS = {
    "n-unique-addresses": "Unique Addresses",
    "n-transactions":     "Number of Transactions",
    "hash-rate":          "Hash Rate (TH/s)",
    "total-bitcoins":     "Total Bitcoins in Circulation",
    "output-volume":      "Output Volume (BTC)",
    "market-price":       "Market Price (USD)",
}

MISSING_MANUAL = []


# ── 공통 유틸 ──────────────────────────────────────────────────────────
def trading_days(start, end):
    """미국 영업일에서 연방공휴일을 제외한 거래일 그리드 (make_y_variables.py와 동일)."""
    grid = pd.date_range(start, end, freq="B")
    hol = USFederalHolidayCalendar().holidays(start, end)
    return grid[~grid.isin(hol)]


def cached(name, fn):
    """수집 결과를 dataset/raw_ext/_cache/<name>.csv에 캐시. 재실행 시 네트워크 생략."""
    path = CACHE / f"{name}.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        print(f"    [cache] {name}: {len(df)}행")
        return df
    df = fn()
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"    [fetch] {name}: {len(df)}행")
    return df


# ── 1. blockchain.info ────────────────────────────────────────────────
def fetch_blockchain(chart, label):
    url = f"https://api.blockchain.info/charts/{chart}?timespan=1826days&format=json"
    with urllib.request.urlopen(url, timeout=60) as r:
        values = json.load(r)["values"]
    df = pd.DataFrame(values)
    df["DateTime"] = pd.to_datetime(df["x"], unit="s")
    df = df.rename(columns={"y": label})
    df["Date"] = df["DateTime"].dt.floor("D")
    if chart == "total-bitcoins":
        # 하루 여러 관측 → 마지막 값 (data_variables.py와 동일)
        df = df.sort_values(["Date", "DateTime"]).groupby("Date").tail(1)
    df = df[["Date", label]].drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return df[(df["Date"] >= PRICE_START) & (df["Date"] <= END)]


# ── 2. Upbit ──────────────────────────────────────────────────────────
def fetch_upbit():
    import pyupbit
    to, frames = None, []
    while True:
        df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=200, to=to)
        if df is None or df.empty:
            break
        frames.append(df)
        earliest = df.index.min()
        if earliest <= PRICE_START:
            break
        to = (earliest - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(0.2)
    out = pd.concat(frames).reset_index()
    out.columns = ["Date"] + list(out.columns[1:])
    out = out.rename(columns={"volume": "volume_btc", "value": "trade_value_krw", "close": "BTC_KRW"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out = out[["Date", "volume_btc", "trade_value_krw", "BTC_KRW"]]
    out = out.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return out[(out["Date"] >= PRICE_START) & (out["Date"] <= END)]


# ── 2-1. Bithumb (Upbit 호환 v1 캔들 API) ─────────────────────────────
def fetch_bithumb():
    """빗썸 KRW-BTC 일봉. Upbit와 같은 응답 스키마이며 200개씩 역방향 페이징."""
    to, rows = None, []
    while True:
        url = "https://api.bithumb.com/v1/candles/days?market=KRW-BTC&count=200"
        if to:
            url += f"&to={to.strftime('%Y-%m-%d')}%20{to.strftime('%H:%M:%S')}"
        req = urllib.request.Request(url, headers={"accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            arr = json.load(r)
        if not arr:
            break
        rows.extend(arr)
        earliest = pd.Timestamp(arr[-1]["candle_date_time_kst"]).normalize()
        if earliest <= PRICE_START:
            break
        to = earliest
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["candle_date_time_kst"]).dt.normalize()
    df = df.rename(columns={"candle_acc_trade_volume": "volume_btc_bithumb",
                            "candle_acc_trade_price": "trade_value_krw_bithumb"})
    df = df[["Date", "volume_btc_bithumb", "trade_value_krw_bithumb"]]
    df = df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return df[(df["Date"] >= PRICE_START) & (df["Date"] <= END)]


# ── 3. Binance BTCUSDT 종가 ───────────────────────────────────────────
def fetch_binance():
    start_ms = int(PRICE_START.timestamp() * 1000)
    end_ms = int((END + pd.Timedelta(days=1)).timestamp() * 1000)
    rows = []
    while start_ms < end_ms:
        url = ("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d"
               f"&startTime={start_ms}&limit=1000")
        with urllib.request.urlopen(url, timeout=60) as r:
            arr = json.load(r)
        if not arr:
            break
        rows.extend(arr)
        start_ms = arr[-1][0] + 86_400_000
        time.sleep(0.2)
    df = pd.DataFrame(rows).iloc[:, [0, 4]]
    df.columns = ["open_time", "BTC_USD"]
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df["BTC_USD"] = pd.to_numeric(df["BTC_USD"])
    df = df[["Date", "BTC_USD"]].drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return df[(df["Date"] >= PRICE_START) & (df["Date"] <= END)]


# ── 4. FRED USD/KRW ───────────────────────────────────────────────────
def fetch_usdkrw():
    url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXKOUS"
           f"&cosd={PRICE_START.date()}&coed={END.date()}")
    with urllib.request.urlopen(url, timeout=60) as r:
        df = pd.read_csv(r)
    df.columns = ["Date", "USD_KRW"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["USD_KRW"] = pd.to_numeric(df["USD_KRW"], errors="coerce")
    return df.dropna().reset_index(drop=True)


# ── 5. yfinance ───────────────────────────────────────────────────────
def fetch_yf(ticker, label):
    def _f():
        import yfinance as yf
        d = yf.download(ticker, start=PRICE_START, end=END + pd.Timedelta(days=1),
                        progress=False, auto_adjust=False)
        d = d["Close"].reset_index()
        d.columns = ["Date", label]
        d["Date"] = pd.to_datetime(d["Date"]).dt.normalize()
        return d.dropna().reset_index(drop=True)
    return _f


# ── 6. Google Trends ──────────────────────────────────────────────────
def fetch_trends(keyword, geo, slug):
    """fetch_trends.py와 동일하게 3개월 청크 수집 후 단순 연결. 청크별 캐시."""
    from dateutil.relativedelta import relativedelta
    from pytrends.request import TrendReq

    cdir = CACHE / "trends"
    cdir.mkdir(parents=True, exist_ok=True)
    chunks, cursor = [], START_VER1.date()
    while cursor < END.date():
        ce = min(cursor + relativedelta(months=3) - timedelta(days=1), END.date())
        chunks.append((cursor, ce))
        cursor = ce + timedelta(days=1)

    pt, parts = None, []
    for i, (s, e) in enumerate(chunks, 1):
        cpath = cdir / f"{slug}_{i:02d}.csv"
        if cpath.exists():
            parts.append(pd.read_csv(cpath, index_col=0, parse_dates=True)["value"])
            continue
        if pt is None:
            pt = TrendReq(hl="en-US", tz=0)
        print(f"      [{i:02d}/{len(chunks)}] {s} ~ {e}", end=" ", flush=True)
        try:
            pt.build_payload([keyword], timeframe=f"{s} {e}", geo=geo)
            ser = pt.interest_over_time()
            if ser.empty or keyword not in ser.columns:
                print("빈 응답")
            else:
                ser = ser[keyword].rename("value")
                ser.to_frame("value").to_csv(cpath, encoding="utf-8-sig")
                parts.append(ser)
                print(f"{len(ser)}행")
        except Exception as ex:
            print(f"오류: {type(ex).__name__} {str(ex)[:80]}")
        time.sleep(10)

    if not parts:
        raise RuntimeError(f"Google Trends '{keyword}' 수집 실패")
    m = pd.concat(parts)
    m = m[~m.index.duplicated(keep="first")].sort_index()
    out = m.reset_index()
    out.columns = ["Date", "value"]
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    return out.sort_values("Date").reset_index(drop=True)


# ── 6-1. 주간 앵커 보정 ───────────────────────────────────────────────
def fetch_trends_weekly(keyword, geo):
    """
    전 구간(>9개월)을 한 번에 요청하면 구글은 주간 데이터를 준다.
    요청이 하나이므로 218주 전체가 단일 정규화 위에 놓인다 — 이게 전역 기준자다.
    """
    from pytrends.request import TrendReq
    pt = TrendReq(hl="en-US", tz=0)
    pt.build_payload([keyword], timeframe=f"{START_VER1.date()} {END.date()}", geo=geo)
    s = pt.interest_over_time()[keyword].astype(float)
    out = s.reset_index()
    out.columns = ["Date", "value"]
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    return out


def anchor_trends(daily, weekly, months=3):
    """
    3개월 청크는 각자의 최대값으로 정규화되어 청크마다 배율이 다르다(실측 0.975~1.430).
    각 청크를 주간 앵커에 원점 통과 최소제곱으로 맞춰 전역 스케일을 복원한다.
    검증: 청크 경계가 다른 두 시계열의 수준 상관이 0.691 -> 0.963으로 올라간다.
    """
    d = daily.set_index("Date")["value"].astype(float).sort_index()
    w = weekly.set_index("Date")["value"].astype(float).sort_index()

    edges, k = [], 0
    while True:
        e = START_VER1 + pd.DateOffset(months=months * k)
        edges.append(e)
        k += 1
        if e > d.index.max():
            break

    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = d[(d.index >= a) & (d.index < b)]
        if len(seg) < 14:
            parts.append(seg)
            continue
        wk = seg.resample("W-SUN", label="left", closed="left").mean()
        # 청크 안에 온전히 들어간 주만 사용 (경계에 걸친 주는 평균이 왜곡된다)
        js = [t for t in wk.index.intersection(w.index) if t >= a and t + pd.Timedelta(days=6) < b]
        if len(js) < 2:
            parts.append(seg)
            continue
        x, y = wk.loc[js].values, w.loc[js].values
        parts.append(seg * ((x * y).sum() / (x * x).sum()))

    out = pd.concat(parts).sort_index().reset_index()
    out.columns = ["Date", "value"]
    return out


# ── 6-2. 대조군용 트렌드 접합 ─────────────────────────────────────────
LEGACY_TREND = {
    # (메인 표본 기준 파일, 레짐 표본 기준 파일)
    "all": (ROOT / "trend" / "merged_bitcoin_all.csv", ROOT / "dataset" / "raw" / "bitcoin_all_ver1.csv"),
    "kr":  (ROOT / "trend" / "merged_bitcoin_kr.csv",  ROOT / "dataset" / "raw" / "bitcoin_kr_ver1.csv"),
}


def splice_trend(legacy_path, anchored):
    """
    대조군: 기존 트렌드 시계열을 겹치는 구간에서 그대로 보존하고,
    그 이후만 앵커 보정된 신규 시계열로 잇는다.

    기간 연장 효과와 트렌드 재수집 효과를 분리하기 위한 것이므로, 겹치는 구간은
    반드시 기존 값과 bit-exact로 같아야 한다. 연장분의 배율은 기존 시계열의
    마지막 3개월 청크에 원점 통과 최소제곱으로 맞춰 접합부를 연속시킨다.
    """
    lg = pd.read_csv(legacy_path, encoding="utf-8-sig")
    lg.columns = ["Date"] + list(lg.columns[1:])
    lg["Date"] = pd.to_datetime(lg["Date"]).dt.normalize()
    lg = lg[["Date", "value"]].drop_duplicates("Date").sort_values("Date")

    seam = lg["Date"].max()
    a = anchored.set_index("Date")["value"].astype(float)
    l = lg.set_index("Date")["value"].astype(float)

    tail = l.index[l.index > seam - pd.DateOffset(months=3)]
    j = tail.intersection(a.index)
    if len(j) < 20:
        raise RuntimeError(f"{legacy_path.name}: 배율 추정 구간이 {len(j)}일뿐이다")
    x, y = a.loc[j].values, l.loc[j].values
    s = (x * y).sum() / (x * x).sum()

    ext = (a[a.index > seam] * s).reset_index()
    ext.columns = ["Date", "value"]
    out = pd.concat([lg, ext]).sort_values("Date").reset_index(drop=True)
    print(f"    접합 {legacy_path.name}: 기존 {len(lg)}행(~{seam.date()}) + 연장 {len(ext)}행, 배율 {s:.4f}")
    return out


# ── 7. 수동 입력 ──────────────────────────────────────────────────────
def kospi_rv_proxy():
    """VKOSPI 대용: ^KS11 로그수익률의 30일 실현변동성(연환산 %). 실제 지수가 아니다."""
    ks = fetch_yf("^KS11", "KS11")()
    ks["r"] = np.log(ks["KS11"] / ks["KS11"].shift(1))
    ks["KOSPI_Volatility"] = ks["r"].rolling(RV_WINDOW).std() * np.sqrt(252) * 100
    return ks[["Date", "KOSPI_Volatility"]].dropna().reset_index(drop=True)


def _read_vkospi_csv(path):
    """investing.com 다운로드 포맷: "날짜","종가",... 앞 두 컬럼만 쓴다."""
    df = pd.read_csv(path, encoding="utf-8-sig").iloc[:, :2]
    df.columns = ["Date", "KOSPI_Volatility"]
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.replace(" ", "").str.strip('"'),
                                errors="coerce")
    df["KOSPI_Volatility"] = pd.to_numeric(
        df["KOSPI_Volatility"].astype(str).str.replace(",", "").str.strip('"'), errors="coerce")
    return df.dropna()


def load_manual_vkospi(use_proxy=False):
    """
    VKOSPI = 기존 raw/KOSPI Volatility_history.csv (2022-05-02~2025-05-23)
             + manual_ext/*Volatility*.csv (그 이후 연속분).
    같은 날짜가 겹치면 manual_ext 쪽을 우선한다.
    """
    parts = []
    legacy = ROOT / "dataset" / "raw" / "KOSPI Volatility_history.csv"
    for p in sorted(MANUAL.glob("*Volatility*.csv")):
        parts.append(_read_vkospi_csv(p))
    if legacy.exists():
        parts.append(_read_vkospi_csv(legacy))
    if not parts:
        if use_proxy:
            print("    [proxy] VKOSPI 대신 ^KS11 30일 실현변동성 사용")
            return cached("ks11_rv_proxy", kospi_rv_proxy)
        MISSING_MANUAL.append("dataset/manual_ext/*Volatility*.csv")
        return pd.DataFrame(columns=["Date", "KOSPI_Volatility"])
    df = pd.concat(parts).drop_duplicates("Date", keep="first")
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"    VKOSPI: {len(df)}행 {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    return df


def _read_ishares_xls(path):
    """
    iShares 다운로드(.xls)는 실제로는 SpreadsheetML 2003 XML이다.
    'Historical' 시트의 'As Of' / 'NAV per Share' 두 컬럼을 뽑는다.
    URL 안의 맨 & 때문에 XML이 깨져 있어 엔티티로 치환한 뒤 파싱한다.
    """
    import re
    import xml.etree.ElementTree as ET

    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    name_key = "{urn:schemas-microsoft-com:office:spreadsheet}Name"
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);)", "&amp;", raw)
    root = ET.fromstring(raw)

    sheets = [w for w in root.findall("ss:Worksheet", ns) if w.get(name_key) == "Historical"]
    if not sheets:
        raise RuntimeError(f"{path.name}에 'Historical' 시트가 없다")
    rows = []
    for r in sheets[0].findall(".//ss:Row", ns)[1:]:
        cells = [(c.find("ss:Data", ns).text if c.find("ss:Data", ns) is not None else "")
                 for c in r.findall("ss:Cell", ns)]
        if len(cells) >= 2:
            rows.append((cells[0], cells[1]))
    df = pd.DataFrame(rows, columns=["Date", "nav_true"])
    df["Date"] = pd.to_datetime(df["Date"], format="%b %d, %Y", errors="coerce")
    df["nav_true"] = pd.to_numeric(df["nav_true"], errors="coerce")
    return df.dropna()


def load_manual_nav():
    """iShares IBIT NAV. 다운로드한 *_fund.xls를 그대로 읽거나, IBIT_NAV.csv(Date,NAV)도 허용."""
    parts = []
    for p in sorted(MANUAL.glob("*.xls")):
        parts.append(_read_ishares_xls(p))
    csv_path = MANUAL / "IBIT_NAV.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig").iloc[:, :2]
        df.columns = ["Date", "nav_true"]
        df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip('"'), errors="coerce")
        df["nav_true"] = pd.to_numeric(
            df["nav_true"].astype(str).str.replace(",", "").str.replace("$", "", regex=False),
            errors="coerce")
        parts.append(df.dropna())
    if not parts:
        MISSING_MANUAL.append("dataset/manual_ext/iShares-*_fund.xls (또는 IBIT_NAV.csv)")
        return pd.DataFrame(columns=["Date", "nav_true"])
    out = pd.concat(parts).drop_duplicates("Date", keep="first")
    out = out.sort_values("Date").reset_index(drop=True)
    print(f"    IBIT NAV: {len(out)}행 {out['Date'].min().date()} ~ {out['Date'].max().date()}")
    return out


# ── 메인 ──────────────────────────────────────────────────────────────
def main(use_proxy=False, anchor=True, splice=False):
    global RAW_EXT, TRAIN_EXT, TRAINV1_EXT
    if splice:
        # 대조군은 별도 디렉터리에 쓴다 (기본 확장판을 덮지 않는다)
        RAW_EXT = ROOT / "dataset" / "raw_ext_ctrl"
        TRAIN_EXT = ROOT / "dataset" / "train_ext_ctrl"
        TRAINV1_EXT = ROOT / "dataset" / "train_ver1_ext_ctrl"
    for d in (RAW_EXT, TRAIN_EXT, TRAINV1_EXT, MANUAL, CACHE):
        d.mkdir(parents=True, exist_ok=True)

    print("[1] blockchain.info")
    bc = {}
    for chart, label in BLOCKCHAIN_CHARTS.items():
        bc[label] = cached(f"bc_{chart}", lambda c=chart, l=label: fetch_blockchain(c, l))
        time.sleep(1)

    print("[2] Upbit / Bithumb / Binance / FRED / yfinance")
    upbit   = cached("upbit_krw_btc", fetch_upbit)
    bithumb = cached("bithumb_krw_btc", fetch_bithumb)
    binance = cached("binance_btcusdt", fetch_binance)
    usdkrw  = cached("fred_usdkrw", fetch_usdkrw)
    vix     = cached("yf_vix", fetch_yf("^VIX", "VIX"))
    ibit    = cached("yf_ibit", fetch_yf("IBIT", "etf_true"))

    print("[3] Google Trends (최초 실행 시 10분 내외 소요, 청크 캐시됨)")
    tr_all = cached("trends_bitcoin_global", lambda: fetch_trends("Bitcoin", "", "all"))
    tr_kr  = cached("trends_bitcoin_kr", lambda: fetch_trends("비트코인", "KR", "kr"))
    if anchor:
        wk_all = cached("trends_weekly_global", lambda: fetch_trends_weekly("Bitcoin", ""))
        wk_kr  = cached("trends_weekly_kr", lambda: fetch_trends_weekly("비트코인", "KR"))
        tr_all = anchor_trends(tr_all, wk_all)
        tr_kr  = anchor_trends(tr_kr, wk_kr)
        print("    주간 앵커로 청크 배율 보정 완료 (--no-anchor로 끌 수 있다)")

    # 메인 표본과 레짐 표본은 기존 기준 파일이 다르므로 각각 접합한다
    tr_all_m, tr_kr_m, tr_all_1, tr_kr_1 = tr_all, tr_kr, tr_all, tr_kr
    if splice:
        print("    [대조군] 기존 트렌드 보존 + 연장분만 접합")
        tr_all_m = splice_trend(LEGACY_TREND["all"][0], tr_all)
        tr_kr_m  = splice_trend(LEGACY_TREND["kr"][0], tr_kr)
        tr_all_1 = splice_trend(LEGACY_TREND["all"][1], tr_all)
        tr_kr_1  = splice_trend(LEGACY_TREND["kr"][1], tr_kr)

    print("[4] 수동 입력 확인")
    vkospi = load_manual_vkospi(use_proxy)
    nav_t  = load_manual_nav()

    # 국내 거래량 = Upbit + Bithumb (기존 raw/kp_variables.csv와 동일한 정의)
    kr = upbit.merge(bithumb, on="Date", how="outer").sort_values("Date").reset_index(drop=True)
    kr["volume_btc"] = kr["volume_btc"].fillna(0) + kr["volume_btc_bithumb"].fillna(0)
    kr["trade_value_krw"] = kr["trade_value_krw"].fillna(0) + kr["trade_value_krw_bithumb"].fillna(0)
    kr = kr[["Date", "volume_btc", "trade_value_krw", "BTC_KRW"]]

    # ── 파생: BTC 가격 로그수익률 / RV30 ──
    price = bc["Market Price (USD)"].copy()
    price["log_ret"] = np.log(price["Market Price (USD)"] / price["Market Price (USD)"].shift(1))
    price["btc_volatility"] = np.sqrt(
        price["log_ret"].rolling(RV_WINDOW).apply(lambda x: (x ** 2).sum(), raw=True))

    # 메인 표본은 미국 거래일. 종속변수가 미국 상장 IBIT이고 nav_true/etf_true가
    # 미국 거래일에만 존재하므로, 기존 raw/y_true_variables.csv(343일)와 정확히 일치한다.
    grid_main = trading_days(START_MAIN, END)

    # 레짐 표본은 한국 거래일. 구성 변수(VKOSPI, 국내 거래량, 국내 검색관심도)가 모두
    # 한국 시장 변수라, 미국 그리드로 옮기면 한국 휴장일이 보간값으로 채워져 HMM 추정에
    # 인위적 평활이 들어간다. 기존 train_ver1/(750일)도 VKOSPI 날짜 기준이었다.
    if len(vkospi):
        kd = pd.DatetimeIndex(vkospi["Date"])
        grid_v1 = kd[(kd >= START_VER1) & (kd <= END)].sort_values()
    else:
        grid_v1 = trading_days(START_VER1, END)
        print("    [!] VKOSPI가 없어 레짐 그리드를 미국 거래일로 대체한다")
    print(f"    거래일 그리드: 메인 {len(grid_main)}일(미국), ver1 {len(grid_v1)}일(한국)")

    def on(df, grid):
        return pd.DataFrame({"Date": grid}).merge(df, on="Date", how="left")

    # ── raw_ext/nav_variables.csv ──
    nav = pd.DataFrame({"Date": grid_main})
    for label in ["Unique Addresses", "Number of Transactions", "Hash Rate (TH/s)",
                  "Total Bitcoins in Circulation", "Output Volume (BTC)"]:
        nav = nav.merge(bc[label], on="Date", how="left")
    nav = nav.merge(tr_all_m.rename(columns={"value": "Search Interest"}), on="Date", how="left")
    nav = nav.interpolate(numeric_only=True)
    nav.to_csv(RAW_EXT / "nav_variables.csv", index=False, encoding="utf-8")

    # ── raw_ext/gap_variables.csv (VIX 로그차분) ──
    v = on(vix, grid_main).interpolate(numeric_only=True)
    v["VIX Volatility"] = np.log(v["VIX"] / v["VIX"].shift(1))
    v.loc[v.index[0], "VIX Volatility"] = 0.0  # 기존 파일과 동일하게 첫 행 0
    v[["Date", "VIX Volatility"]].to_csv(RAW_EXT / "gap_variables.csv", index=False, encoding="utf-8")

    # ── raw_ext/kp_variables.csv ──
    kp = on(kr[["Date", "volume_btc", "trade_value_krw"]], grid_main)
    kp = kp.merge(vkospi, on="Date", how="left") if len(vkospi) else kp.assign(KOSPI_Volatility=np.nan)
    kp = kp.merge(usdkrw, on="Date", how="left")
    # VKOSPI는 한국 거래일 기준이라 미국 거래일 그리드에 구멍이 난다 → 선형보간 후 양끝 채움
    kp_num = ["volume_btc", "trade_value_krw", "USD_KRW", "KOSPI_Volatility"]
    kp[kp_num] = kp[kp_num].interpolate().ffill().bfill()
    kp.to_csv(RAW_EXT / "kp_variables.csv", index=False, encoding="utf-8")

    # ── raw_ext/y_true_variables.csv ──
    yt = on(ibit, grid_main)
    yt = yt.merge(nav_t, on="Date", how="left") if len(nav_t) else yt.assign(nav_true=np.nan)
    yt[["etf_true", "nav_true"]] = yt[["etf_true", "nav_true"]].interpolate().ffill().bfill()
    yt[["Date", "nav_true", "etf_true"]].to_csv(RAW_EXT / "y_true_variables.csv",
                                                index=False, encoding="utf-8")

    # ── raw_ext/y_variables.csv ──
    # Log Return: 거래일 필터 → 로그차분 (make_y_variables.py 순서)
    back = trading_days(START_MAIN - pd.Timedelta(days=10), START_MAIN)
    prev = back[back < START_MAIN][-1]
    keep = pd.DatetimeIndex([prev]).append(grid_main)
    lr = price[price["Date"].isin(keep)].sort_values("Date").reset_index(drop=True)
    lr["Log Return"] = np.log(lr["Market Price (USD)"] / lr["Market Price (USD)"].shift(1))
    lr = lr[lr["Date"].isin(grid_main)][["Date", "Log Return"]]

    km = on(kr[["Date", "BTC_KRW"]], grid_main).merge(binance, on="Date", how="left") \
        .merge(usdkrw, on="Date", how="left")
    km[["BTC_KRW", "BTC_USD", "USD_KRW"]] = km[["BTC_KRW", "BTC_USD", "USD_KRW"]].interpolate()
    km["Kimchi Premium"] = km["BTC_KRW"] / (km["BTC_USD"] * km["USD_KRW"]) - 1

    y = pd.DataFrame({"Date": grid_main}).merge(lr, on="Date", how="left") \
        .merge(km[["Date", "Kimchi Premium"]], on="Date", how="left") \
        .merge(yt[["Date", "nav_true", "etf_true"]], on="Date", how="left")
    y["etf_premium"] = y["etf_true"] / y["nav_true"] - 1
    y[["Date", "etf_premium", "Log Return", "Kimchi Premium"]].to_csv(
        RAW_EXT / "y_variables.csv", index=False, encoding="utf-8")

    # ── train_ext (메인 표본) ──
    nav[["Date", "Hash Rate (TH/s)", "Unique Addresses"]].dropna().to_csv(
        TRAIN_EXT / "nav_train.csv", index=False, encoding="utf-8-sig")

    gap = on(tr_all_m, grid_main).merge(price[["Date", "btc_volatility"]], on="Date", how="left")
    gap[["value", "btc_volatility"]] = gap[["value", "btc_volatility"]].interpolate()
    gap.to_csv(TRAIN_EXT / "gap_train_main.csv", index=False, encoding="utf-8-sig")

    kpt = kp[["Date", "KOSPI_Volatility", "volume_btc"]].merge(
        tr_kr_m.rename(columns={"value": "bitcoin_kr"}), on="Date", how="left")
    kpt["bitcoin_kr"] = kpt["bitcoin_kr"].interpolate()
    kpt.to_csv(TRAIN_EXT / "kp_train_main.csv", index=False, encoding="utf-8-sig")

    # ── train_ver1_ext (레짐 표본) ──
    n1 = on(bc["Hash Rate (TH/s)"], grid_v1).merge(bc["Unique Addresses"], on="Date", how="left")
    n1.interpolate(numeric_only=True).dropna().to_csv(
        TRAINV1_EXT / "nav_train.csv", index=False, encoding="utf-8-sig")

    g1 = on(tr_all_1, grid_v1).merge(price[["Date", "btc_volatility"]], on="Date", how="left")
    g1[["value", "btc_volatility"]] = g1[["value", "btc_volatility"]].interpolate()
    g1.to_csv(TRAINV1_EXT / "gap_train.csv", index=False, encoding="utf-8-sig")

    k1 = pd.DataFrame({"Date": grid_v1})
    k1 = k1.merge(vkospi, on="Date", how="left") if len(vkospi) else k1.assign(KOSPI_Volatility=np.nan)
    k1 = k1.merge(kr[["Date", "volume_btc"]], on="Date", how="left") \
           .merge(tr_kr_1.rename(columns={"value": "bitcoin_kr"}), on="Date", how="left")
    k1_num = ["KOSPI_Volatility", "volume_btc", "bitcoin_kr"]
    k1[k1_num] = k1[k1_num].interpolate().ffill().bfill()
    k1.to_csv(TRAINV1_EXT / "kp_train.csv", index=False, encoding="utf-8-sig")

    # ── 요약 ──
    print("\n" + "=" * 68)
    for p in sorted(list(RAW_EXT.glob("*.csv")) + list(TRAIN_EXT.glob("*.csv"))
                    + list(TRAINV1_EXT.glob("*.csv"))):
        d = pd.read_csv(p)
        nas = {c: int(d[c].isna().sum()) for c in d.columns if d[c].isna().any()}
        print(f"  {p.relative_to(ROOT)}: {len(d)}행  결측={nas or '없음'}")

    if MISSING_MANUAL:
        print("\n[!] 수동 입력 대기 중: 아래 파일을 넣고 재실행하면 해당 컬럼이 채워진다")
        for m in sorted(set(MISSING_MANUAL)):
            print(f"      {m}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main("--kospi-proxy" in sys.argv, "--no-anchor" not in sys.argv,
                          "--trend-splice" in sys.argv))
