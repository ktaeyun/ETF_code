"""
NAV/GAP/KP 시뮬레이터용 데이터 로더
- 기본값: dataset/train (nav_train, gap_train, kp_train) + dataset/raw (y_variables: Log Return, etf_premium, Kimchi Premium)
- Date 기준으로 병합
"""

import numpy as np
import pandas as pd
from pathlib import Path

TRAIN_DIR_NAME = "train"
RAW_DIR_NAME = "raw"

# NAV 계열의 정의.
#
#   "btc_aligned" (기본) : blockchain.info BTC 현물가격 + 1거래일 시차보정.
#       NAV는 "기초자산의 순수한 가치"여야 하므로 운용보수 등 펀드 단위 비용이
#       섞이지 않은 BTC 현물을 쓴다. 다만 blockchain.info의 market-price는
#       00:00 UTC 스냅샷(= 전일 종가)이라 GAP이 기준하는 IBIT NAV 산정 시각
#       (16:00 ET)보다 1거래일 늦다. 실측으로 확인한 정렬 근거는 다음과 같다
#       (연속 거래일 265쌍, nav_true 로그수익률과의 상관):
#           같은 날 -0.031  /  BTC를 1거래일 당김 +0.739
#       따라서 수익률 계열을 한 칸 당겨 맞추고, 대응물이 없는 마지막 거래일을 버린다.
#       부분적 보정이다 — market-price가 시점 스냅샷이 아니라 일별 집계값이라
#       상관이 0.95 근처까지는 올라가지 않는다. 근본 해결은 16:00 ET 기준
#       BTC 레퍼런스 레이트를 별도 수집하는 것이다.
#
#   "btc"      : 위 보정 없이 원본 그대로 (구 사양). GAP과 1거래일 어긋난다.
#
#   "nav_true" : IBIT의 주당 NAV. GAP이 etf_true/nav_true - 1 로 정의되므로
#                nav_true x (1+GAP) = etf_true 가 항등식으로 성립한다. 대신 운용보수
#                등 펀드 단위 비용이 NAV에 섞여 들어간다.
NAV_SOURCE_DEFAULT = "btc_aligned"
NAV_SOURCES = ("btc_aligned", "btc", "nav_true")


def align_to_nav_grid(nav_df: pd.DataFrame, *dfs: pd.DataFrame):
    """GAP/KP를 NAV 계열의 날짜 그리드에 맞춘다.

    NAV는 사양에 따라 앞(nav_true: 로그차분) 또는 뒤(btc_aligned: 시프트)에서
    거래일 하나를 잃는다. 결합 시 날짜가 어긋나지 않도록 GAP/KP를 NAV의
    날짜 집합으로 필터링한다.
    """
    keep = set(nav_df["Date"])
    out = tuple(d[d["Date"].isin(keep)].sort_values("Date").reset_index(drop=True)
                for d in dfs)
    for d in out:
        if len(d) != len(nav_df):
            raise ValueError(f"NAV 그리드 정렬 실패: {len(d)} != {len(nav_df)}")
    return out[0] if len(out) == 1 else out


def load_nav_exog_and_returns(
    nav_path: str = None,
    y_path: str = None,
    base_dir: str = None,
    nav_source: str = NAV_SOURCE_DEFAULT,
) -> pd.DataFrame:
    """
    NAV 독립변수(Hash Rate, Unique Addresses)와 종속변수(Log Return)를
    Date 기준으로 병합하여 반환. 기본값: train/nav_train.csv + raw/y_variables.csv

    Args:
        nav_path: nav 경로 (None이면 base_dir/dataset/train/nav_train.csv)
        y_path: y_variables 경로 (None이면 base_dir/dataset/raw/y_variables.csv)
        base_dir: 프로젝트 루트 (None이면 이 파일 기준 상위 디렉터리)
        nav_source: "btc_aligned"(기본) / "btc" / "nav_true". NAV_SOURCE_DEFAULT 주석 참조.

    Returns:
        DataFrame with columns: Date, Log Return, Hash Rate (TH/s), Unique Addresses,
        nav_true, etf_true (뒤 둘은 결합 검증 Step 6-A의 기준 계열)
    """
    if nav_source not in NAV_SOURCES:
        raise ValueError(f"nav_source는 {NAV_SOURCES} 중 하나: {nav_source}")
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent)
    train_dir = Path(base_dir) / "dataset" / TRAIN_DIR_NAME
    raw_dir = Path(base_dir) / "dataset" / RAW_DIR_NAME
    if nav_path is None:
        nav_path = train_dir / "nav_train.csv"
    if y_path is None:
        y_path = raw_dir / "y_variables.csv"
    nav_path = Path(nav_path)
    y_path = Path(y_path)

    nav = pd.read_csv(nav_path)
    y_df = pd.read_csv(y_path)
    nav["Date"] = pd.to_datetime(nav["Date"])
    y_df["Date"] = pd.to_datetime(y_df["Date"])

    # 독립변수: Hash Rate + Unique Addresses
    exog_cols = ["Hash Rate (TH/s)", "Unique Addresses"]
    for c in exog_cols:
        if c not in nav.columns:
            raise ValueError(f"nav_variables에 컬럼 없음: {c}")

    # 병합: Date 기준
    merged = y_df[["Date", "Log Return"]].merge(
        nav[["Date"] + exog_cols],
        on="Date",
        how="inner",
    )
    merged = merged.sort_values("Date").reset_index(drop=True)

    # 결합 검증(Step 6-A)의 기준 계열을 항상 함께 싣는다.
    true_df = load_etf_true(base_dir=base_dir)
    merged = merged.merge(true_df, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    if len(merged) < 2:
        raise ValueError("y_true_variables 병합 결과가 2행 미만")

    if nav_source == "btc":
        return merged

    if nav_source == "btc_aligned":
        # BTC 수익률을 1거래일 당겨 IBIT NAV 산정 시각에 맞춘다.
        # 마지막 거래일은 당겨올 값이 없으므로 버린다 (343 -> 342행).
        merged["Log Return"] = merged["Log Return"].shift(-1)
        return merged.iloc[:-1].reset_index(drop=True)

    # nav_true: Log Return을 IBIT 주당 NAV의 로그차분으로 교체한다.
    # 첫 행은 로그차분이 정의되지 않으므로 버린다 (343 -> 342행).
    merged["Log Return"] = np.concatenate(
        [[np.nan], np.diff(np.log(merged["nav_true"].to_numpy(dtype=float)))]
    )
    return merged.iloc[1:].reset_index(drop=True)


def load_gap_exog(
    gap_path: str = None,
    y_path: str = None,
    base_dir: str = None,
) -> pd.DataFrame:
    """
    GAP 독립변수와 종속변수(etf_premium)를 Date 기준으로 병합하여 반환.
    기본값: train/gap_train.csv (value→Search Interest, btc_volatility→VIX Volatility) + raw/y_variables.csv

    Args:
        gap_path: gap 경로 (None이면 base_dir/dataset/train/gap_train.csv)
        y_path: y_variables 경로 (None이면 base_dir/dataset/raw/y_variables.csv)
        base_dir: 프로젝트 루트 (None이면 이 파일 기준 상위 디렉터리)

    Returns:
        DataFrame with columns: Date, etf_premium (GAP), value, btc_volatility
    """
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent)
    train_dir = Path(base_dir) / "dataset" / TRAIN_DIR_NAME
    raw_dir = Path(base_dir) / "dataset" / RAW_DIR_NAME
    if gap_path is None:
        gap_path = train_dir / "gap_train_main.csv"
    if y_path is None:
        y_path = raw_dir / "y_variables.csv"
    gap_path = Path(gap_path)
    y_path = Path(y_path)

    gap_df = pd.read_csv(gap_path)
    y_df = pd.read_csv(y_path)
    gap_df["Date"] = pd.to_datetime(gap_df["Date"])
    y_df["Date"] = pd.to_datetime(y_df["Date"])

    if "value" not in gap_df.columns or "btc_volatility" not in gap_df.columns:
        raise ValueError("gap 데이터에 'value', 'btc_volatility' 컬럼 필요")

    merged = y_df[["Date", "etf_premium"]].merge(
        gap_df[["Date", "value", "btc_volatility"]],
        on="Date",
        how="inner",
    )
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def load_kp_exog(
    kp_path: str = None,
    y_path: str = None,
    base_dir: str = None,
) -> pd.DataFrame:
    """
    KP 독립변수(volume_btc, KOSPI_Volatility)와 종속변수(Kimchi Premium)를
    Date 기준으로 병합하여 반환. 기본값: train/kp_train.csv + raw/y_variables.csv

    Args:
        kp_path: kp 경로 (None이면 base_dir/dataset/train/kp_train.csv)
        y_path: y_variables 경로 (None이면 base_dir/dataset/raw/y_variables.csv)
        base_dir: 프로젝트 루트 (None이면 이 파일 기준 상위 디렉터리)

    Returns:
        DataFrame with columns: Date, Kimchi Premium, volume_btc, KOSPI_Volatility
    """
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent)
    train_dir = Path(base_dir) / "dataset" / TRAIN_DIR_NAME
    raw_dir = Path(base_dir) / "dataset" / RAW_DIR_NAME
    if kp_path is None:
        kp_path = train_dir / "kp_train_main.csv"
    if y_path is None:
        y_path = raw_dir / "y_variables.csv"
    kp_path = Path(kp_path)
    y_path = Path(y_path)

    kp_df = pd.read_csv(kp_path)
    y_df = pd.read_csv(y_path)
    kp_df["Date"] = pd.to_datetime(kp_df["Date"])
    y_df["Date"] = pd.to_datetime(y_df["Date"])

    # 독립변수: volume_btc + KOSPI_Volatility + bitcoin_kr
    exog_cols = ["volume_btc", "KOSPI_Volatility", "bitcoin_kr"]
    for c in exog_cols:
        if c not in kp_df.columns:
            raise ValueError(f"kp_variables에 컬럼 없음: {c}")

    # 병합: Date 기준
    merged = y_df[["Date", "Kimchi Premium"]].merge(
        kp_df[["Date"] + exog_cols],
        on="Date",
        how="inner",
    )
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def load_etf_true(
    y_true_path: str = None,
    base_dir: str = None,
) -> pd.DataFrame:
    """
    실제 ETF 가격 및 NAV (etf_true, nav_true) 로드 (raw 유지)
    
    Args:
        y_true_path: y_true_variables.csv 경로 (None이면 base_dir/dataset/raw/y_true_variables.csv)
        base_dir: 프로젝트 루트 (None이면 이 파일 기준 상위 디렉터리)
    
    Returns:
        DataFrame with columns: Date, etf_true, nav_true
    """
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent)
    raw_dir = Path(base_dir) / "dataset" / RAW_DIR_NAME
    if y_true_path is None:
        y_true_path = raw_dir / "y_true_variables.csv"
    y_true_path = Path(y_true_path)
    
    y_true_df = pd.read_csv(y_true_path)
    y_true_df["Date"] = pd.to_datetime(y_true_df["Date"])
    
    for col in ["etf_true", "nav_true"]:
        if col not in y_true_df.columns:
            raise ValueError(f"y_true_variables에 {col} 컬럼 없음")
    
    y_true_df = y_true_df.sort_values("Date").reset_index(drop=True)
    return y_true_df[["Date", "etf_true", "nav_true"]]
