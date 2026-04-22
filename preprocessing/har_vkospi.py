"""
HAR-VKOSPI 모형 모듈
=====================
VKOSPI 시계열에 HAR(Heterogeneous Autoregressive) 모형을 적합하고
평균회귀 사이클을 제거한 표준화 잔차(비정상적 변동성 충격)를 추출한다.

모형:
  VKOSPI_t = α
            + β_d · VKOSPI_{t-1}
            + β_w · mean(VKOSPI[t-5 : t-1])
            + β_m · mean(VKOSPI[t-22 : t-1])
            + ε_t

잔차 표준화:
  z_t = (ε_t - mean(ε)) / std(ε)   ← 전체 표본 기준
  (rolling 기준으로 변경 시 look-ahead bias 없이 구현 가능 — 주의사항 참조)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller


# ──────────────────────────────────────────────
# 파라미터 기본값
# ──────────────────────────────────────────────
LAG_D:  int = 1    # 일간 (daily) lag
LAG_W:  int = 5    # 주간 (weekly) rolling 길이
LAG_M:  int = 22   # 월간 (monthly) rolling 길이


# ══════════════════════════════════════════════
# 데이터 클래스 — 적합 결과 보관
# ══════════════════════════════════════════════

@dataclass
class HARResult:
    """HAR 모형 적합 결과 컨테이너."""

    # OLS 결과 객체 (statsmodels RegressionResultsWrapper)
    ols_result: object

    # 계수
    alpha: float
    beta_d: float
    beta_w: float
    beta_m: float

    # 적합도
    r_squared: float
    adj_r_squared: float

    # 잔차
    residuals: pd.Series           # 원시 잔차 ε_t
    residuals_z: pd.Series         # 표준화 잔차 z_t

    # 표준화 모수
    resid_mean: float
    resid_std: float

    # ADF 검정 결과 (잔차)
    adf_stat: float     = field(default=float("nan"))
    adf_pval: float     = field(default=float("nan"))
    adf_crit: dict      = field(default_factory=dict)

    # 설계 행렬 (X) — 진단용
    X: Optional[pd.DataFrame] = field(default=None, repr=False)

    # ──────────────────────────────────────────
    def summary_dict(self) -> dict:
        """계수 + 적합도 + ADF 요약 딕셔너리 반환."""
        return {
            "alpha":         self.alpha,
            "beta_d":        self.beta_d,
            "beta_w":        self.beta_w,
            "beta_m":        self.beta_m,
            "R²":            self.r_squared,
            "Adj-R²":        self.adj_r_squared,
            "resid_mean":    self.resid_mean,
            "resid_std":     self.resid_std,
            "ADF stat":      self.adf_stat,
            "ADF p-value":   self.adf_pval,
        }

    def print_summary(self) -> None:
        """콘솔에 요약 출력."""
        print("=" * 52)
        print("  HAR-VKOSPI Estimation Results")
        print("=" * 52)
        print(f"  alpha (const): {self.alpha:>10.4f}")
        print(f"  beta_d (lag1): {self.beta_d:>10.4f}")
        print(f"  beta_w (5d)  : {self.beta_w:>10.4f}")
        print(f"  beta_m (22d) : {self.beta_m:>10.4f}")
        print(f"  R^2          : {self.r_squared:>10.4f}")
        print(f"  Adj-R^2      : {self.adj_r_squared:>10.4f}")
        print("-" * 52)
        print(f"  Resid mean   : {self.resid_mean:>10.4f}")
        print(f"  Resid std    : {self.resid_std:>10.4f}")
        print("-" * 52)
        crit_str = "  |  ".join(f"{k}: {v:.2f}" for k, v in self.adf_crit.items())
        print(f"  ADF stat     : {self.adf_stat:>10.4f}")
        print(f"  ADF p-value  : {self.adf_pval:>10.4f}  "
              f"({'Stationary' if self.adf_pval < 0.05 else 'Non-Stationary'} @ 5%)")
        print(f"  ADF crit.val : {crit_str}")
        print("=" * 52)


# ══════════════════════════════════════════════
# 특징 행렬 생성
# ══════════════════════════════════════════════

def build_har_features(
    series: pd.Series,
    lag_d: int = LAG_D,
    lag_w: int = LAG_W,
    lag_m: int = LAG_M,
) -> pd.DataFrame:
    """HAR 설계 행렬 X 를 생성한다 (look-ahead bias 없음).

    Parameters
    ----------
    series : pd.Series
        VKOSPI 원시 시계열 (daily frequency).
    lag_d : int
        일간 lag 크기 (기본 1).
    lag_w : int
        주간 rolling 창 크기 (기본 5).
    lag_m : int
        월간 rolling 창 크기 (기본 22).

    Returns
    -------
    pd.DataFrame
        컬럼: ["vkospi_d", "vkospi_w", "vkospi_m"]
        인덱스는 원본 series 와 동일.
        rolling 미달 구간은 NaN.
    """
    # 일간: t-1
    vkospi_d = series.shift(lag_d)

    # 주간: mean(t-lag_w, ..., t-1) → shift(1) 후 rolling(lag_w)
    vkospi_w = (
        series.shift(1)
        .rolling(window=lag_w, min_periods=lag_w)
        .mean()
    )

    # 월간: mean(t-lag_m, ..., t-1)
    vkospi_m = (
        series.shift(1)
        .rolling(window=lag_m, min_periods=lag_m)
        .mean()
    )

    return pd.DataFrame(
        {"vkospi_d": vkospi_d, "vkospi_w": vkospi_w, "vkospi_m": vkospi_m},
        index=series.index,
    )


# ══════════════════════════════════════════════
# HAR OLS 적합
# ══════════════════════════════════════════════

def fit_har(
    series: pd.Series,
    lag_d: int = LAG_D,
    lag_w: int = LAG_W,
    lag_m: int = LAG_M,
    standardize_resid: bool = True,
) -> HARResult:
    """VKOSPI 시계열에 HAR 모형을 OLS 로 적합하고 결과를 반환한다.

    Parameters
    ----------
    series : pd.Series
        VKOSPI 원시 시계열 (daily frequency).
        index 는 DatetimeIndex 권장.
    lag_d, lag_w, lag_m : int
        HAR 구성 lag 크기.
    standardize_resid : bool
        True(기본)이면 잔차를 전체 표본 mean/std 로 z-score 표준화.

    Returns
    -------
    HARResult
        계수, 잔차, ADF 검정 결과 등을 담은 컨테이너.
    """
    # 1. 설계 행렬 생성
    X_df = build_har_features(series, lag_d=lag_d, lag_w=lag_w, lag_m=lag_m)

    # 2. 유효 행(NaN 없는 행)만 사용
    combined = pd.concat([series.rename("y"), X_df], axis=1).dropna()
    y = combined["y"]
    X = sm.add_constant(combined[["vkospi_d", "vkospi_w", "vkospi_m"]])

    # 3. OLS 적합
    ols_res = sm.OLS(y, X).fit()

    # 4. 계수 추출
    alpha  = ols_res.params["const"]
    beta_d = ols_res.params["vkospi_d"]
    beta_w = ols_res.params["vkospi_w"]
    beta_m = ols_res.params["vkospi_m"]

    # 5. 잔차 (원본 인덱스 정렬)
    resid = pd.Series(ols_res.resid, index=y.index, name="resid_har")

    # 6. 잔차 표준화 (전체 표본 기준)
    resid_mean = resid.mean()
    resid_std  = resid.std(ddof=1)
    if resid_std == 0:
        resid_z = resid * np.nan
    else:
        resid_z = (resid - resid_mean) / resid_std if standardize_resid else resid.copy()
    resid_z.name = "resid_z_har"

    # 7. ADF 정상성 검정 (표준화 잔차)
    adf_target = resid_z.dropna()
    adf_out    = adfuller(adf_target, autolag="AIC")
    adf_stat, adf_pval, _, _, adf_crit, _ = adf_out

    return HARResult(
        ols_result    = ols_res,
        alpha         = alpha,
        beta_d        = beta_d,
        beta_w        = beta_w,
        beta_m        = beta_m,
        r_squared     = ols_res.rsquared,
        adj_r_squared = ols_res.rsquared_adj,
        residuals     = resid,
        residuals_z   = resid_z,
        resid_mean    = resid_mean,
        resid_std     = resid_std,
        adf_stat      = adf_stat,
        adf_pval      = adf_pval,
        adf_crit      = adf_crit,
        X             = X,
    )


# ══════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════

def plot_har_result(
    series: pd.Series,
    result: HARResult,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """원시 VKOSPI vs HAR 잔차 비교 플롯 (4-패널).

    패널 구성:
      [1] 원시 VKOSPI 시계열
      [2] HAR 적합값 vs 실제값 오버레이
      [3] 원시 잔차 ε_t
      [4] 표준화 잔차 z_t (±2σ 참조선 포함)

    Parameters
    ----------
    series : pd.Series
        원시 VKOSPI 시계열.
    result : HARResult
        ``fit_har`` 반환값.
    figsize : tuple
        Figure 크기.
    save_path : str | None
        지정하면 PNG 저장.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    fitted   = result.ols_result.fittedvalues
    resid    = result.residuals
    resid_z  = result.residuals_z

    # ── 패널 1: 원시 VKOSPI ──────────────────────
    ax1.plot(series.index, series, color="#2c7bb6", linewidth=1.1, label="VKOSPI")
    ax1.set_title("(1) Raw VKOSPI Series", fontsize=10)
    ax1.set_ylabel("VKOSPI")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # ── 패널 2: 실제 vs 적합값 ────────────────────
    ax2.plot(fitted.index, series.reindex(fitted.index),
             color="#2c7bb6", linewidth=1.0, alpha=0.7, label="Actual")
    ax2.plot(fitted.index, fitted,
             color="#d7191c", linewidth=1.0, linestyle="--", label="HAR Fitted")
    ax2.set_title(
        f"(2) Actual vs HAR Fitted  "
        f"(R²={result.r_squared:.3f}, Adj-R²={result.adj_r_squared:.3f})",
        fontsize=10,
    )
    ax2.set_ylabel("VKOSPI")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.35)

    # ── 패널 3: 원시 잔차 ε_t ─────────────────────
    ax3.plot(resid.index, resid, color="#756bb1", linewidth=0.9, label="ε_t")
    ax3.axhline(y=0, color="black", linewidth=0.8)
    ax3.set_title("(3) HAR Residuals epsilon_t (Raw)", fontsize=10)
    ax3.set_ylabel("Residual")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.35)

    # ── 패널 4: 표준화 잔차 z_t ───────────────────
    ax4.plot(resid_z.index, resid_z, color="#31a354", linewidth=0.9, label="z_t")
    ax4.axhline(y= 0, color="black",   linewidth=0.8, linestyle="-")
    ax4.axhline(y= 2, color="#fdae61", linewidth=0.8, linestyle="--", label="+2σ")
    ax4.axhline(y=-2, color="#fdae61", linewidth=0.8, linestyle="--", label="-2σ")
    ax4.axhline(y= 3, color="#d7191c", linewidth=0.7, linestyle=":",  label="+3σ")
    ax4.axhline(y=-3, color="#d7191c", linewidth=0.7, linestyle=":",  label="-3σ")

    adf_label = (
        f"ADF p={result.adf_pval:.3f} "
        f"({'Stationary' if result.adf_pval < 0.05 else 'Non-Stationary'} @ 5%)"
    )
    ax4.set_title(f"(4) Standardized Residuals z_t  |  {adf_label}", fontsize=10)
    ax4.set_ylabel("z-score")
    ax4.legend(fontsize=8, ncol=3, loc="upper right")
    ax4.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("HAR-VKOSPI : Raw Series vs Residuals", fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[HAR] Plot saved -> {save_path}")

    return fig


# ══════════════════════════════════════════════
# 편의 파이프라인 함수
# ══════════════════════════════════════════════

def run_har_pipeline(
    series: pd.Series,
    lag_d: int = LAG_D,
    lag_w: int = LAG_W,
    lag_m: int = LAG_M,
    plot: bool = True,
    save_plot_path: Optional[str] = None,
) -> HARResult:
    """HAR 적합 → 요약 출력 → 시각화를 한 번에 실행하는 파이프라인.

    Parameters
    ----------
    series : pd.Series
        VKOSPI 원시 시계열 (daily frequency).
    lag_d, lag_w, lag_m : int
        HAR lag 파라미터.
    plot : bool
        True 이면 4-패널 비교 플롯 표시.
    save_plot_path : str | None
        시각화 저장 경로 (None 이면 저장 안 함).

    Returns
    -------
    HARResult
        적합 결과 컨테이너.
    """
    # 1. 적합
    result = fit_har(series, lag_d=lag_d, lag_w=lag_w, lag_m=lag_m)

    # 2. 요약 콘솔 출력
    result.print_summary()

    # 3. statsmodels 상세 요약
    print("\n[OLS Detailed Results]")
    print(result.ols_result.summary())

    # 4. 시각화
    if plot:
        plot_har_result(series, result, save_path=save_plot_path)
        plt.show()

    return result


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

    kp_df = pd.read_csv(
        _DATA / "kp_train.csv", parse_dates=["Date"]
    ).set_index("Date").sort_index()

    # VKOSPI: kp_train["KOSPI_Volatility"] (일별, 750행)
    vkospi_series = kp_df["KOSPI_Volatility"].copy()
    vkospi_series.name = "VKOSPI"
    # ─────────────────────────────────────────────────────────────────

    result = run_har_pipeline(
        series         = vkospi_series,
        lag_d          = LAG_D,   # 1일
        lag_w          = LAG_W,   # 5 거래일
        lag_m          = LAG_M,   # 22 거래일
        plot           = True,
        save_plot_path = None,
    )

    print("\n=== Standardized Residuals (top 10 rows) ===")
    print(result.residuals_z.head(10).to_string())

    jb_stat, jb_pval = stats.jarque_bera(result.residuals_z.dropna())
    print(f"\nJarque-Bera  stat={jb_stat:.3f},  p={jb_pval:.4f}")
