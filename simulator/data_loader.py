"""
NAV 시뮬레이터용 데이터 로더
- nav_variables.csv: Hash Rate (TH/s), Unique Addresses → 독립변수
- y_variables.csv: Log Return → 종속변수, etf_premium → GAP
- Date 기준으로 병합
"""

import pandas as pd
from pathlib import Path


def load_nav_exog_and_returns(
    nav_path: str = None,
    y_path: str = None,
    base_dir: str = None,
) -> pd.DataFrame:
    """
    NAV 독립변수(Hash Rate, Unique Addresses)와 종속변수(Log Return)를
    Date 기준으로 병합하여 반환.

    Args:
        nav_path: nav_variables.csv 경로 (None이면 base_dir/dataset/nav_variables.csv)
        y_path: y_variables.csv 경로 (None이면 base_dir/dataset/y_variables.csv)
        base_dir: 프로젝트 루트 (None이면 이 파일 기준 상위 디렉터리)

    Returns:
        DataFrame with columns: Date, Log Return, Hash Rate (TH/s), Unique Addresses
        (그 외 nav/y 컬럼도 포함)
    """
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent)
    dataset_dir = Path(base_dir) / "dataset"
    if nav_path is None:
        nav_path = dataset_dir / "nav_variables.csv"
    if y_path is None:
        y_path = dataset_dir / "y_variables.csv"
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
    nav_path: str = None,
    gap_path: str = None,
    y_path: str = None,
    base_dir: str = None,
) -> pd.DataFrame:
    """
    GAP 독립변수(Search Interest, VIX Volatility)와 종속변수(etf_premium)를
    Date 기준으로 병합하여 반환.

    Args:
        nav_path: nav_variables.csv 경로 (None이면 base_dir/dataset/nav_variables.csv)
        gap_path: gap_variables.csv 경로 (None이면 base_dir/dataset/gap_variables.csv)
        y_path: y_variables.csv 경로 (None이면 base_dir/dataset/y_variables.csv)
        base_dir: 프로젝트 루트 (None이면 이 파일 기준 상위 디렉터리)

    Returns:
        DataFrame with columns: Date, etf_premium (GAP), Search Interest, VIX Volatility
    """
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent)
    dataset_dir = Path(base_dir) / "dataset"
    if nav_path is None:
        nav_path = dataset_dir / "nav_variables.csv"
    if gap_path is None:
        gap_path = dataset_dir / "gap_variables.csv"
    if y_path is None:
        y_path = dataset_dir / "y_variables.csv"
    nav_path = Path(nav_path)
    gap_path = Path(gap_path)
    y_path = Path(y_path)

    nav = pd.read_csv(nav_path)
    gap_df = pd.read_csv(gap_path)
    y_df = pd.read_csv(y_path)
    nav["Date"] = pd.to_datetime(nav["Date"])
    gap_df["Date"] = pd.to_datetime(gap_df["Date"])
    y_df["Date"] = pd.to_datetime(y_df["Date"])

    # 독립변수: Search Interest (nav_variables에서) + VIX Volatility (gap_variables에서)
    if "Search Interest" not in nav.columns:
        raise ValueError("nav_variables에 'Search Interest' 컬럼 없음")
    if "VIX Volatility" not in gap_df.columns:
        raise ValueError("gap_variables에 'VIX Volatility' 컬럼 없음")

    # 병합: Date 기준
    # 먼저 nav와 gap을 병합한 후, y와 병합
    exog_merged = nav[["Date", "Search Interest"]].merge(
        gap_df[["Date", "VIX Volatility"]],
        on="Date",
        how="inner",
    )
    merged = y_df[["Date", "etf_premium"]].merge(
        exog_merged,
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
    Date 기준으로 병합하여 반환.

    Args:
        kp_path: kp_variables.csv 경로 (None이면 base_dir/dataset/kp_variables.csv)
        y_path: y_variables.csv 경로 (None이면 base_dir/dataset/y_variables.csv)
        base_dir: 프로젝트 루트 (None이면 이 파일 기준 상위 디렉터리)

    Returns:
        DataFrame with columns: Date, Kimchi Premium, volume_btc, KOSPI_Volatility
    """
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent)
    dataset_dir = Path(base_dir) / "dataset"
    if kp_path is None:
        kp_path = dataset_dir / "kp_variables.csv"
    if y_path is None:
        y_path = dataset_dir / "y_variables.csv"
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
    실제 ETF 가격 및 NAV (etf_true, nav_true) 로드
    
    Args:
        y_true_path: y_true_variables.csv 경로 (None이면 base_dir/dataset/y_true_variables.csv)
        base_dir: 프로젝트 루트 (None이면 이 파일 기준 상위 디렉터리)
    
    Returns:
        DataFrame with columns: Date, etf_true, nav_true
    """
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent.parent)
    dataset_dir = Path(base_dir) / "dataset"
    if y_true_path is None:
        y_true_path = dataset_dir / "y_true_variables.csv"
    y_true_path = Path(y_true_path)
    
    y_true_df = pd.read_csv(y_true_path)
    y_true_df["Date"] = pd.to_datetime(y_true_df["Date"])
    
    for col in ["etf_true", "nav_true"]:
        if col not in y_true_df.columns:
            raise ValueError(f"y_true_variables에 {col} 컬럼 없음")
    
    y_true_df = y_true_df.sort_values("Date").reset_index(drop=True)
    return y_true_df[["Date", "etf_true", "nav_true"]]
