"""
전처리 통합 파이프라인
======================
dataset/train_ver1/ 의 실제 데이터를 사용하여
HAR-VKOSPI → Bai-Perron(raw) → Gaussian HMM 을 순차 실행한다.

데이터 출처:
  kp_train.csv  : KOSPI_Volatility (VKOSPI), volume_btc, bitcoin_kr (국내 SVI)
  gap_train.csv : value (글로벌 BTC SVI), btc_volatility (Global_RV)

컬럼 매핑:
  global_btc_svi   ← gap_train["value"]           (Google Trends 글로벌, 0~100)
  domestic_btc_svi ← kp_train["bitcoin_kr"]        (Google Trends 한국,   0~100)
  btc_volume_btc   ← kp_train["volume_btc"]        (국내 BTC 일별 거래량, BTC 단위)
  VKOSPI           ← kp_train["KOSPI_Volatility"]  (일별 VKOSPI)
  Global_RV        ← gap_train["btc_volatility"]   (BTC 일별 실현 변동성)

[구간 분할 설계]
  Step 2. Bai-Perron — raw 주별 SVI/Volume 원시값 → 수준 변화 탐지
  Step 3. Gaussian HMM — Global_RV / VKOSPI_resid → 변동성 regime 분류
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.har_vkospi import run_har_pipeline
from preprocessing.bai_perron import run_bai_perron_pipeline
from preprocessing.gaussian_hmm import run_hmm_pipeline, HMMComparison


# ══════════════════════════════════════════════
# 경로 설정
# ══════════════════════════════════════════════

DATA_DIR    = _ROOT / "dataset" / "train_ver1"
RESULTS_DIR = _ROOT / "results" / "preprocessing"

# Bai-Perron 대상 컬럼 (raw 주별 원시값)
BP_RAW_COLS = ["global_btc_svi", "domestic_btc_svi", "btc_volume_btc"]


# ══════════════════════════════════════════════
# Step 1 — 데이터 로드 및 병합
# ══════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """kp_train / gap_train 로드 후 일별(daily) + 주별(weekly) DataFrame 반환.

    Returns
    -------
    df_daily  : pd.DataFrame  (daily, Global_RV 포함)
    df_weekly : pd.DataFrame  (W-MON 기준, SVI/Volume 원시값)
    """
    kp_df  = pd.read_csv(DATA_DIR / "kp_train.csv",
                          parse_dates=["Date"]).set_index("Date").sort_index()
    gap_df = pd.read_csv(DATA_DIR / "gap_train.csv",
                          parse_dates=["Date"]).set_index("Date").sort_index()

    df_daily = pd.DataFrame(
        {
            "global_btc_svi":   gap_df["value"],
            "domestic_btc_svi": kp_df["bitcoin_kr"],
            "btc_volume_btc":   kp_df["volume_btc"],
            "VKOSPI":           kp_df["KOSPI_Volatility"],
            "Global_RV":        gap_df["btc_volatility"],
        }
    ).dropna()

    # SVI: 주 마지막 관측값, Volume: 주 합산
    df_weekly = pd.DataFrame(
        {
            "global_btc_svi":   df_daily["global_btc_svi"].resample("W-MON").last(),
            "domestic_btc_svi": df_daily["domestic_btc_svi"].resample("W-MON").last(),
            "btc_volume_btc":   df_daily["btc_volume_btc"].resample("W-MON").sum(),
        }
    ).dropna()

    print(f"[Data] daily  : {df_daily.shape}  "
          f"({df_daily.index[0].date()} ~ {df_daily.index[-1].date()})")
    print(f"[Data] weekly : {df_weekly.shape}  "
          f"({df_weekly.index[0].date()} ~ {df_weekly.index[-1].date()})")
    return df_daily, df_weekly


# ══════════════════════════════════════════════
# Step 2 — HAR-VKOSPI (일별)
# ══════════════════════════════════════════════

def step_har(df_daily: pd.DataFrame, save: bool = True):
    """HAR 모형 VKOSPI 적합 및 표준화 잔차 추출.

    Returns
    -------
    har_result : HARResult
    """
    print("\n" + "═" * 60)
    print("  Step 2: HAR-VKOSPI")
    print("═" * 60)

    vkospi      = df_daily["VKOSPI"].copy()
    vkospi.name = "VKOSPI"

    return run_har_pipeline(
        series         = vkospi,
        lag_d          = 1,
        lag_w          = 5,
        lag_m          = 22,
        plot           = True,
        save_plot_path = str(RESULTS_DIR / "har_vkospi.png") if save else None,
    )


# ══════════════════════════════════════════════
# Step 3 — Bai-Perron: raw 주별 데이터
# ══════════════════════════════════════════════

def step_bai_perron(df_weekly: pd.DataFrame, save: bool = True) -> dict:
    """Bai-Perron 검정을 raw 주별 SVI/Volume 에 적용한다.

    Returns
    -------
    dict : {컬럼명 → BaiPerronResult}
    """
    print("\n" + "═" * 60)
    print("  Step 3: Bai-Perron (Raw Weekly Series — Level Breaks)")
    print("═" * 60)

    return run_bai_perron_pipeline(
        df           = df_weekly,
        asvi_columns = BP_RAW_COLS,
        m_max        = 3,
        trim         = 0.10,
        sig_level    = 0.05,
        hac_bw       = None,
        plot         = True,
        save_dir     = str(RESULTS_DIR / "bai_perron") if save else None,
    )


# ══════════════════════════════════════════════
# Step 4 — Gaussian HMM
# ══════════════════════════════════════════════

def step_hmm(
    df_daily:   pd.DataFrame,
    har_result,
    n_init:     int  = 10,
    B:          int  = 1000,
    save:       bool = True,
) -> dict[str, HMMComparison]:
    """Global_RV 및 VKOSPI_resid 에 Gaussian HMM K=2/3 를 적합한다.

    Returns
    -------
    dict : {"Global_RV": HMMComparison, "VKOSPI_resid": HMMComparison}
    """
    print("\n" + "═" * 60)
    print("  Step 4: Gaussian HMM (K=2 vs K=3)")
    print("  [Global_RV  &  VKOSPI_resid]")
    print("═" * 60)

    global_rv      = df_daily["Global_RV"].dropna().copy()
    global_rv.name = "Global_RV"

    vkospi_idx = df_daily["VKOSPI"].dropna().index
    resid_z    = pd.Series(
        har_result.residuals_z,
        index=vkospi_idx[22:22 + len(har_result.residuals_z)],
        name="VKOSPI_resid",
    )

    hmm_results = {}
    for series in [global_rv, resid_z]:
        hmm_results[series.name] = run_hmm_pipeline(
            series   = series,
            n_init   = n_init,
            B        = B,
            plot     = True,
            save_dir = str(RESULTS_DIR / "gaussian_hmm") if save else None,
        )
    return hmm_results


# ══════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════

def main(
    hmm_n_init: int  = 10,
    hmm_B:      int  = 1000,
    save_plots: bool = True,
) -> dict:
    """전처리 파이프라인 전체 실행.

    Returns
    -------
    {
      "df_daily"   : pd.DataFrame,
      "df_weekly"  : pd.DataFrame,
      "har_result" : HARResult,
      "bp"         : dict[str, BaiPerronResult],
      "hmm"        : dict[str, HMMComparison],
    }
    """
    if save_plots:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for sub in ["bai_perron", "gaussian_hmm"]:
            (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  Step 1: Load Data")
    print("═" * 60)
    df_daily, df_weekly = load_data()

    har_result  = step_har(df_daily, save=save_plots)
    bp_results  = step_bai_perron(df_weekly, save=save_plots)
    hmm_results = step_hmm(df_daily, har_result,
                            n_init=hmm_n_init, B=hmm_B, save=save_plots)

    # ── 최종 요약 ────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Preprocessing Pipeline Complete")
    print("═" * 60)
    print(f"  HAR R²    : {har_result.r_squared:.4f}")
    print(f"  HAR ADF p : {har_result.adf_pval:.4f}  "
          f"({'Stationary' if har_result.adf_pval < 0.05 else 'Non-Stationary'})")

    print("\n  [Bai-Perron — Raw Weekly]")
    for col, res in bp_results.items():
        dates_str = [str(d.date()) for d in res.break_dates]
        print(f"    {col}: m={res.optimal_m}  breaks={dates_str}")

    print("\n  [Gaussian HMM]")
    for name, comp in hmm_results.items():
        r = comp.result_k2 if comp.optimal_K == 2 else comp.result_k3
        print(f"    {name}: optimal K={comp.optimal_K}  "
              f"BIC_k2={comp.result_k2.bic:.2f}  BIC_k3={comp.result_k3.bic:.2f}  "
              f"LRT_p={comp.bootstrap_pvalue:.3f}")
        for k in range(comp.optimal_K):
            print(f"      Regime {k}: μ={r.mu[k]:.4f}  σ={r.sigma[k]:.4f}  "
                  f"occ={r.occupancy[k]:.1%}")

    return {
        "df_daily":   df_daily,
        "df_weekly":  df_weekly,
        "har_result": har_result,
        "bp":         bp_results,
        "hmm":        hmm_results,
    }


if __name__ == "__main__":
    outputs = main(hmm_n_init=10, hmm_B=1000, save_plots=True)
