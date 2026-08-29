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


def load_nav_data(base_dir: str = None, S0: float = 100.0):
    """
    모형 후보 비교(compare/, Step 3)용 NAV 시계열 로더.

    Step 3의 모형 선택 근거는 Step 4 본모델과 같은 데이터 위에서 나와야 하므로,
    load_nav_exog_and_returns()가 읽는 것과 동일한 `Log Return`(거래일 그리드로
    필터링한 뒤 로그차분한 계열)을 쓴다. 과거 구현은 y_true_variables의 `nav_true`를
    직접 로그차분했는데, 그 경로는 커밋 eb97732에서 고친 주말 수익률 손실 문제를
    그대로 안고 있었다.

    Args:
        base_dir: 프로젝트 루트
        S0: 초기 NAV (simulator/main.py 기본값과 동일하게 100.0)

    Returns:
        tuple: (nav_series, returns_series) — 둘 다 Date 인덱스를 갖는 pd.Series
    """
    from simulator.arima_garch_t_nav_simulator import log_returns_to_nav

    df = load_nav_exog_and_returns(base_dir=base_dir)
    df = df.dropna(subset=["Log Return"]).reset_index(drop=True)

    idx = pd.to_datetime(df["Date"])
    returns_series = pd.Series(df["Log Return"].to_numpy(dtype=float), index=idx, name="log_return")

    # nav_series는 returns_series보다 한 점 길다(S0, S1, ..., ST).
    # 호출부(compare_main)가 S0 = nav_series.iloc[0],
    # actual_nav_aligned = nav_series.iloc[1:] 로 쓰는 계약을 따른다.
    nav_path = np.asarray(log_returns_to_nav(returns_series, S0=S0)).flatten()
    nav_idx = idx.iloc[:1] - (idx.iloc[1] - idx.iloc[0]) if len(idx) > 1 else idx.iloc[:1]
    nav_series = pd.Series(
        np.concatenate([[S0], nav_path]),
        index=pd.DatetimeIndex(list(nav_idx) + list(idx)), name="nav",
    )

    print("NAV 데이터 로드 (Step 3 비교용, Step 4와 동일 계열)")
    print(f"  - 수익률 샘플 수: {len(returns_series)}, NAV 샘플 수: {len(nav_series)}")
    print(f"  - 기간: {idx.iloc[0].date()} ~ {idx.iloc[-1].date()}")
    return nav_series, returns_series


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
