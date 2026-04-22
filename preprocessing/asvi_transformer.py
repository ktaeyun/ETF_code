"""
ASVI (Abnormal Search Volume Index) 변환 모듈
=============================================
외생변수 구간 분할을 위한 전처리 단계 - 변수별 ASVI 변환

대상 변수:
  - Global Bitcoin SVI   (Google Trends, 글로벌)
  - Domestic Bitcoin SVI (Google Trends, 한국)
  - 국내 Bitcoin 거래량   (원화 기준 주별 합산)

변환 공식:
  ASVI_t = (X_t - mean(X[t-k : t-1])) / std(X[t-k : t-1])

  - 롤링 윈도우 k = 4주 (파라미터 조정 가능)
  - look-ahead bias 없이 과거 k개 관측치만 참조
  - 분모 == 0 → NaN 처리
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# 상수: 대상 컬럼 논리명 → 실제 컬럼명 매핑
# 실제 데이터를 삽입할 때 아래 딕셔너리의 값만 수정하면 됩니다.
# ──────────────────────────────────────────────
ASVI_COLUMN_MAP: Dict[str, str] = {
    "global_svi":   "global_btc_svi",    # Global Bitcoin SVI (Google Trends)
    "domestic_svi": "domestic_btc_svi",  # Domestic Bitcoin SVI (한국, Google Trends)
    "volume_krw":   "btc_volume_krw",    # 국내 Bitcoin 거래량 (원화 기준 주별 합산)
}

# 기본 롤링 윈도우 크기 (주 단위)
DEFAULT_WINDOW: int = 4


# ══════════════════════════════════════════════
# 핵심 변환 함수
# ══════════════════════════════════════════════

def compute_asvi(
    series: pd.Series,
    window: int = DEFAULT_WINDOW,
) -> pd.Series:
    """단일 시계열에 ASVI 변환을 적용한다.

    Parameters
    ----------
    series : pd.Series
        Weekly frequency 원시 시계열. index는 DatetimeIndex 권장.
    window : int
        롤링 윈도우 크기 k (기본값 4주).
        look-ahead bias 방지를 위해 현재 시점 t 는 제외하고
        [t-k, t-1] 구간(k 개)만 사용한다.

    Returns
    -------
    pd.Series
        ASVI 변환된 시계열. 최초 window 개 시점은 NaN.
    """
    # shift(1) : 현재 시점 t 를 포함하지 않고 t-1 부터 k 개를 참조
    # min_periods=window : 윈도우가 채워지지 않으면 NaN 반환
    rolling_obj  = series.shift(1).rolling(window=window, min_periods=window)
    rolling_mean = rolling_obj.mean()
    rolling_std  = rolling_obj.std(ddof=1)   # 표본 표준편차 (불편 추정량)

    # 분모 == 0 → NaN (상수 구간 보호)
    rolling_std_safe = rolling_std.where(rolling_std != 0, other=np.nan)

    asvi = (series - rolling_mean) / rolling_std_safe
    asvi.name = f"asvi_{series.name}" if series.name else "asvi"
    return asvi


def transform_asvi(
    df: pd.DataFrame,
    column_map: Dict[str, str] = ASVI_COLUMN_MAP,
    window: int = DEFAULT_WINDOW,
    drop_original: bool = False,
) -> pd.DataFrame:
    """DataFrame 내 여러 컬럼에 ASVI 변환을 일괄 적용한다.

    Parameters
    ----------
    df : pd.DataFrame
        원시 데이터. column_map 값에 해당하는 컬럼을 포함해야 함.
    column_map : dict
        논리명 → 실제 컬럼명 매핑 (기본값: ``ASVI_COLUMN_MAP``).
    window : int
        롤링 윈도우 크기 k (기본값 4주).
    drop_original : bool
        True 이면 원본 컬럼을 결과 DataFrame 에서 제거.

    Returns
    -------
    pd.DataFrame
        원본 컬럼 + ASVI 컬럼이 추가된 DataFrame.
    """
    result = df.copy()

    for _logical, col in column_map.items():
        if col not in df.columns:
            # 실제 데이터 삽입 전 컬럼이 없는 경우 건너뜀
            continue
        asvi_series = compute_asvi(df[col], window=window)
        result[asvi_series.name] = asvi_series

    if drop_original:
        result = result.drop(columns=list(column_map.values()), errors="ignore")

    return result


# ══════════════════════════════════════════════
# 기술통계 비교 테이블
# ══════════════════════════════════════════════

def build_stats_table(
    df_raw: pd.DataFrame,
    df_asvi: pd.DataFrame,
    column_map: Dict[str, str] = ASVI_COLUMN_MAP,
) -> pd.DataFrame:
    """변환 전/후 기술통계량 비교 테이블을 생성한다.

    Parameters
    ----------
    df_raw : pd.DataFrame
        원시 데이터프레임.
    df_asvi : pd.DataFrame
        ASVI 변환이 적용된 데이터프레임.
    column_map : dict
        논리명 → 실제 컬럼명 매핑.

    Returns
    -------
    pd.DataFrame
        MultiIndex 컬럼 (Before / After) × 통계량으로 구성된 테이블.
    """
    stat_funcs: Dict[str, callable] = {
        "count": lambda s: s.count(),
        "mean":  lambda s: s.mean(),
        "std":   lambda s: s.std(),
        "min":   lambda s: s.min(),
        "25%":   lambda s: s.quantile(0.25),
        "50%":   lambda s: s.median(),
        "75%":   lambda s: s.quantile(0.75),
        "max":   lambda s: s.max(),
        "skew":  lambda s: s.skew(),
        "kurt":  lambda s: s.kurt(),
        "nan%":  lambda s: s.isna().mean() * 100,
    }

    rows: List[Dict] = []
    for logical, col in column_map.items():
        if col not in df_raw.columns:
            continue

        asvi_col = f"asvi_{col}"
        before   = df_raw[col].dropna()
        after    = df_asvi[asvi_col].dropna() if asvi_col in df_asvi.columns else pd.Series(dtype=float)

        row: Dict = {"variable": logical, "column": col}
        for stat_name, func in stat_funcs.items():
            row[("Before", stat_name)] = func(before) if len(before) else np.nan
            row[("After",  stat_name)] = func(after)  if len(after)  else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    flat = pd.DataFrame(rows).set_index(["variable", "column"])
    flat.columns = pd.MultiIndex.from_tuples(flat.columns)
    return flat


# ══════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════

# 논리명 → 사람이 읽기 좋은 레이블 매핑
_LABEL_MAP: Dict[str, str] = {
    "global_svi":   "Global Bitcoin SVI",
    "domestic_svi": "Domestic Bitcoin SVI (KR)",
    "volume_krw":   "BTC Volume (KRW)",
}


def plot_asvi_transformation(
    df_raw: pd.DataFrame,
    df_asvi: pd.DataFrame,
    column_map: Dict[str, str] = ASVI_COLUMN_MAP,
    window: int = DEFAULT_WINDOW,
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """각 변수에 대해 변환 전/후 시계열을 나란히 시각화한다.

    Parameters
    ----------
    df_raw : pd.DataFrame
        원시 데이터프레임.
    df_asvi : pd.DataFrame
        ASVI 변환이 적용된 데이터프레임.
    column_map : dict
        논리명 → 실제 컬럼명 매핑.
    window : int
        실제 사용된 롤링 윈도우 크기 (플롯 제목 표기용).
    figsize : tuple
        서브플롯 한 행의 (width, height). 전체 height = height × 변수 수.
    save_path : str | None
        지정하면 해당 경로에 PNG 로 저장.

    Returns
    -------
    matplotlib.figure.Figure
    """
    valid_pairs = [(logical, col) for logical, col in column_map.items() if col in df_raw.columns]
    n = len(valid_pairs)
    if n == 0:
        raise ValueError("column_map 에 지정된 컬럼이 DataFrame 에 없습니다.")

    fig, axes = plt.subplots(
        nrows=n, ncols=2,
        figsize=(figsize[0], figsize[1] * n),
        sharex=False,
    )
    # n == 1 이면 axes 가 1-D 배열 → 2-D 형태로 통일
    if n == 1:
        axes = [axes]

    for i, (logical, col) in enumerate(valid_pairs):
        asvi_col  = f"asvi_{col}"
        label     = _LABEL_MAP.get(logical, col)
        ax_before = axes[i][0]
        ax_after  = axes[i][1]

        # ── Before ──────────────────────────────────
        ax_before.plot(df_raw.index, df_raw[col], color="#2c7bb6", linewidth=1.2)
        ax_before.set_title(f"{label}  |  Raw (Before)", fontsize=10)
        ax_before.set_ylabel("Value")
        ax_before.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )
        ax_before.grid(True, linestyle="--", alpha=0.4)

        # ── After (ASVI) ─────────────────────────────
        if asvi_col in df_asvi.columns:
            s = df_asvi[asvi_col]
            ax_after.plot(s.index, s, color="#d7191c", linewidth=1.2)
            # ±2σ 참조선
            ax_after.axhline(y= 0, color="black",   linewidth=0.8, linestyle="-",  label="0")
            ax_after.axhline(y= 2, color="#fdae61",  linewidth=0.8, linestyle="--", label="+2σ")
            ax_after.axhline(y=-2, color="#fdae61",  linewidth=0.8, linestyle="--", label="-2σ")
            ax_after.legend(fontsize=8, loc="upper right")

        ax_after.set_title(f"{label}  |  ASVI (After, k={window}w)", fontsize=10)
        ax_after.set_ylabel("ASVI")
        ax_after.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("ASVI Transformation: Before vs After", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[ASVI] Plot saved -> {save_path}")

    return fig


# ══════════════════════════════════════════════
# 편의 파이프라인 함수
# ══════════════════════════════════════════════

def run_asvi_pipeline(
    df: pd.DataFrame,
    column_map: Dict[str, str] = ASVI_COLUMN_MAP,
    window: int = DEFAULT_WINDOW,
    plot: bool = True,
    save_plot_path: Optional[str] = None,
    drop_original: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ASVI 변환 → 통계 테이블 → 시각화를 한 번에 실행하는 파이프라인.

    Parameters
    ----------
    df : pd.DataFrame
        원시 주별 데이터프레임.
    column_map : dict
        논리명 → 실제 컬럼명 매핑 (기본값: ``ASVI_COLUMN_MAP``).
    window : int
        롤링 윈도우 크기 k (기본값 4주).
    plot : bool
        True 이면 변환 전/후 시계열 시각화 실행.
    save_plot_path : str | None
        시각화 저장 경로 (None 이면 저장 안 함).
    drop_original : bool
        True 이면 결과 DataFrame 에서 원본 컬럼 제거.

    Returns
    -------
    df_asvi : pd.DataFrame
        ASVI 컬럼이 추가된 DataFrame.
    stats_table : pd.DataFrame
        변환 전/후 기술통계량 비교 테이블.
    """
    # 1. ASVI 변환
    df_asvi = transform_asvi(df, column_map=column_map, window=window, drop_original=drop_original)

    # 2. 기술통계 비교 테이블
    stats_table = build_stats_table(df, df_asvi, column_map=column_map)

    # 3. 시각화
    if plot:
        plot_asvi_transformation(
            df_raw=df,
            df_asvi=df_asvi,
            column_map=column_map,
            window=window,
            save_path=save_plot_path,
        )
        plt.show()

    return df_asvi, stats_table


# ══════════════════════════════════════════════
# 독립 실행 예시
# ══════════════════════════════════════════════

if __name__ == "__main__":
    from pathlib import Path
    import sys

    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    # ── 실제 데이터 로드 (dataset/train_ver1) ────────────────────────
    _DATA = _ROOT / "dataset" / "train_ver1"

    kp_df  = pd.read_csv(_DATA / "kp_train.csv",  parse_dates=["Date"]).set_index("Date").sort_index()
    gap_df = pd.read_csv(_DATA / "gap_train.csv", parse_dates=["Date"]).set_index("Date").sort_index()

    # 일별 → 주별 리샘플링
    # SVI  : 주 마지막 관측값 (Google Trends 특성)
    # Volume: 주 합산 (거래량 누적)
    df_weekly = pd.DataFrame(
        {
            "global_btc_svi":   gap_df["value"].resample("W-MON").last(),
            "domestic_btc_svi": kp_df["bitcoin_kr"].resample("W-MON").last(),
            "btc_volume_btc":   kp_df["volume_btc"].resample("W-MON").sum(),
        }
    ).dropna()
    # ─────────────────────────────────────────────────────────────────

    # 실제 데이터용 컬럼 매핑
    REAL_MAP = {
        "global_svi":   "global_btc_svi",    # gap_train["value"]
        "domestic_svi": "domestic_btc_svi",  # kp_train["bitcoin_kr"]
        "volume_btc":   "btc_volume_btc",    # kp_train["volume_btc"] 주합산
    }

    df_asvi_result, stats_result = run_asvi_pipeline(
        df            = df_weekly,
        column_map    = REAL_MAP,
        window        = DEFAULT_WINDOW,  # k=4주
        plot          = True,
        save_plot_path= None,
        drop_original = False,
    )

    print("\n=== ASVI Transformed Series (top 10 rows) ===")
    asvi_cols = [c for c in df_asvi_result.columns if c.startswith("asvi_")]
    print(df_asvi_result[asvi_cols].head(10).to_string())

    print("\n=== Descriptive Statistics: Before vs After ===")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(stats_result.T.to_string())
