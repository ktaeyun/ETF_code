"""
NAV/GAP/KP 시뮬레이터용 데이터 로더
- 기본값: dataset/train (nav_train, gap_train, kp_train) + dataset/raw (y_variables: Log Return, etf_premium, Kimchi Premium)
- Date 기준으로 병합
"""

import pandas as pd
from pathlib import Path

TRAIN_DIR_NAME = "train"
RAW_DIR_NAME = "raw"


def load_nav_exog_and_returns(
    nav_path: str = None,
    y_path: str = None,
    base_dir: str = None,
) -> pd.DataFrame:
    """
    NAV 독립변수(Hash Rate, Unique Addresses)와 종속변수(Log Return)를
    Date 기준으로 병합하여 반환. 기본값: train/nav_train.csv + raw/y_variables.csv

    Args:
        nav_path: nav 경로 (None이면 base_dir/dataset/train/nav_train.csv)
        y_path: y_variables 경로 (None이면 base_dir/dataset/raw/y_variables.csv)
        base_dir: 프로젝트 루트 (None이면 이 파일 기준 상위 디렉터리)

    Returns:
        DataFrame with columns: Date, Log Return, Hash Rate (TH/s), Unique Addresses
    """
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
    return merged


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
        DataFrame with columns: Date, etf_premium (GAP), Search Interest, VIX Volatility
    """
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent)
    train_dir = Path(base_dir) / "dataset" / TRAIN_DIR_NAME
    raw_dir = Path(base_dir) / "dataset" / RAW_DIR_NAME
    if gap_path is None:
        gap_path = train_dir / "gap_train.csv"
    if y_path is None:
        y_path = raw_dir / "y_variables.csv"
    gap_path = Path(gap_path)
    y_path = Path(y_path)

    gap_df = pd.read_csv(gap_path)
    y_df = pd.read_csv(y_path)
    gap_df["Date"] = pd.to_datetime(gap_df["Date"])
    y_df["Date"] = pd.to_datetime(y_df["Date"])

    # gap_train: value, btc_volatility → 시뮬레이터 호환명 Search Interest, VIX Volatility
    if "value" in gap_df.columns and "btc_volatility" in gap_df.columns:
        gap_df = gap_df.rename(columns={"value": "Search Interest", "btc_volatility": "VIX Volatility"})
    if "Search Interest" not in gap_df.columns or "VIX Volatility" not in gap_df.columns:
        raise ValueError("gap 데이터에 'Search Interest', 'VIX Volatility' (또는 value, btc_volatility) 컬럼 필요")

    merged = y_df[["Date", "etf_premium"]].merge(
        gap_df[["Date", "Search Interest", "VIX Volatility"]],
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
        kp_path = train_dir / "kp_train.csv"
    if y_path is None:
        y_path = raw_dir / "y_variables.csv"
    kp_path = Path(kp_path)
    y_path = Path(y_path)

    kp_df = pd.read_csv(kp_path)
    y_df = pd.read_csv(y_path)
    kp_df["Date"] = pd.to_datetime(kp_df["Date"])
    y_df["Date"] = pd.to_datetime(y_df["Date"])

    # 독립변수: volume_btc + KOSPI_Volatility
    exog_cols = ["volume_btc", "KOSPI_Volatility"]
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
