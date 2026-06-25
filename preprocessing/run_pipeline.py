"""
전처리 통합 파이프라인
======================
dataset/train_ver1/ 의 실제 데이터를 사용하여
HAR-VKOSPI → Gaussian HMM (전 변수) 을 순차 실행한다.

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
  Step 2. HAR-VKOSPI   — 일별 VKOSPI → 표준화 잔차 추출
  Step 3. HMM(SVI)     — 주별 SVI/Volume → Gaussian HMM 레짐 분류
  Step 4. HMM(변동성)  — 일별 Global_RV / VKOSPI_resid → Gaussian HMM 레짐 분류

[로그 변환 설계]
  global_btc_svi   : log1p  (0~100 bounded, 0 포함 가능)
  domestic_btc_svi : log1p  (동일)
  btc_volume_btc   : log1p  (0 거래 주 가능)
  Global_RV        : log    (양수 보장 실현 변동성, log-normal)
  VKOSPI_resid     : None   (표준화 잔차, 음수 포함)
"""

from __future__ import annotations

import pickle
from pathlib import Path
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.har_vkospi import run_har_pipeline
from preprocessing.gaussian_hmm import run_hmm_pipeline, HMMComparison


# ══════════════════════════════════════════════
# 경로 설정
# ══════════════════════════════════════════════

DATA_DIR    = _ROOT / "dataset" / "train_ver1"
RESULTS_DIR = _ROOT / "results" / "preprocessing"
HMM_CACHE   = RESULTS_DIR / "hmm_cache.pkl"

# HMM 대상 컬럼 (주별)
HMM_SVI_COLS = ["global_btc_svi", "domestic_btc_svi", "btc_volume_btc"]

# 로그 변환 설정
HMM_SVI_TRANSFORM: dict[str, str | None] = {
    "global_btc_svi":   "log1p",
    "domestic_btc_svi": "log1p",
    "btc_volume_btc":   "log1p",
}
HMM_VOL_TRANSFORM: dict[str, str | None] = {
    "Global_RV":    "log",
    "VKOSPI_resid": None,
}


# ══════════════════════════════════════════════
# Step 1 — 데이터 로드 및 병합
# ══════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """kp_train / gap_train 로드 후 일별(daily) + 주별(weekly) DataFrame 반환.

    Returns
    -------
    df_daily  : pd.DataFrame  (daily)
    df_weekly : pd.DataFrame  (W-MON 기준, SVI/Volume)
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
    """HAR 모형 VKOSPI 적합 및 표준화 잔차 추출."""
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
# Step 3 — Gaussian HMM: 주별 SVI/Volume
# ══════════════════════════════════════════════

def step_hmm_svi(
    df_weekly: pd.DataFrame,
    n_init:    int  = 10,
    B:         int  = 1000,
    save:      bool = True,
) -> dict[str, HMMComparison]:
    """주별 SVI/Volume 3개 시리즈에 Gaussian HMM K=2/3 을 적합한다.

    Returns
    -------
    dict : {컬럼명 → HMMComparison}
    """
    print("\n" + "═" * 60)
    print("  Step 3: Gaussian HMM — SVI/Volume (Weekly)")
    print("  [global_btc_svi / domestic_btc_svi / btc_volume_btc]")
    print("═" * 60)

    save_dir = str(RESULTS_DIR / "hmm_svi") if save else None
    results: dict[str, HMMComparison] = {}

    for col in HMM_SVI_COLS:
        if col not in df_weekly.columns:
            print(f"[HMM-SVI] Column not found, skipping: {col}")
            continue
        series      = df_weekly[col].dropna().copy()
        series.name = col
        results[col] = run_hmm_pipeline(
            series    = series,
            n_init    = n_init,
            B         = B,
            plot      = True,
            save_dir  = save_dir,
            transform = HMM_SVI_TRANSFORM.get(col),
        )

    return results


# ══════════════════════════════════════════════
# Step 4 — Gaussian HMM: 변동성 (일별)
# ══════════════════════════════════════════════

def step_hmm_vol(
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
    print("  Step 4: Gaussian HMM — Volatility (Daily)")
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
            series    = series,
            n_init    = n_init,
            B         = B,
            plot      = True,
            save_dir  = str(RESULTS_DIR / "hmm_vol") if save else None,
            transform = HMM_VOL_TRANSFORM.get(series.name),
        )
    return hmm_results


# ══════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════

def main(
    hmm_n_init:   int  = 10,
    hmm_B:        int  = 1000,
    save_plots:   bool = True,
    force_refit:  bool = False,
) -> dict:
    """전처리 파이프라인 전체 실행.

    Parameters
    ----------
    force_refit : True 이면 캐시를 무시하고 HMM을 재실행한다.
                  False(기본)이면 캐시가 있을 때 HMM 단계를 건너뛴다.

    Returns
    -------
    {
      "df_daily"   : pd.DataFrame,
      "df_weekly"  : pd.DataFrame,
      "har_result" : HARResult,
      "hmm_svi"    : dict[str, HMMComparison],
      "hmm_vol"    : dict[str, HMMComparison],
    }
    """
    if save_plots:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for sub in ["hmm_svi", "hmm_vol"]:
            (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  Step 1: Load Data")
    print("═" * 60)
    df_daily, df_weekly = load_data()

    har_result = step_har(df_daily, save=save_plots)

    if not force_refit and HMM_CACHE.exists():
        print("\n" + "═" * 60)
        print(f"  [HMM Cache] 캐시 로드: {HMM_CACHE}")
        print("  (재실행하려면 force_refit=True 또는 --force-refit 사용)")
        print("═" * 60)
        with open(HMM_CACHE, "rb") as f:
            cached = pickle.load(f)
        hmm_svi_results = cached["hmm_svi"]
        hmm_vol_results = cached["hmm_vol"]
    else:
        hmm_svi_results = step_hmm_svi(df_weekly, n_init=hmm_n_init, B=hmm_B, save=save_plots)
        hmm_vol_results = step_hmm_vol(df_daily, har_result,
                                        n_init=hmm_n_init, B=hmm_B, save=save_plots)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(HMM_CACHE, "wb") as f:
            pickle.dump({"hmm_svi": hmm_svi_results, "hmm_vol": hmm_vol_results}, f)
        print(f"\n  [HMM Cache] 저장 완료: {HMM_CACHE}")

    # ── 최종 요약 ────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Preprocessing Pipeline Complete")
    print("═" * 60)
    print(f"  HAR R²    : {har_result.r_squared:.4f}")
    print(f"  HAR ADF p : {har_result.adf_pval:.4f}  "
          f"({'Stationary' if har_result.adf_pval < 0.05 else 'Non-Stationary'})")

    def _hmm_summary_line(comp):
        k = comp.optimal_K
        r = {1: comp.result_k1, 2: comp.result_k2, 3: comp.result_k3}[k]
        print(f"    optimal K={k}  "
              f"BIC_k1={comp.result_k1.bic:.2f}  "
              f"BIC_k2={comp.result_k2.bic:.2f}  "
              f"BIC_k3={comp.result_k3.bic:.2f}  "
              f"LRT_k1k2_p={comp.bootstrap_pvalue_k1k2:.3f}  "
              f"LRT_k2k3_p={comp.bootstrap_pvalue:.3f}")
        for i in range(k):
            print(f"      Regime {i}: μ={r.mu[i]:.4f}  σ={r.sigma[i]:.4f}  "
                  f"occ={r.occupancy[i]:.1%}")

    print("\n  [Gaussian HMM — SVI/Volume Weekly]")
    for col, comp in hmm_svi_results.items():
        print(f"    {col}:", end="  ")
        _hmm_summary_line(comp)

    print("\n  [Gaussian HMM — Volatility Daily]")
    for name, comp in hmm_vol_results.items():
        print(f"    {name}:", end="  ")
        _hmm_summary_line(comp)

    return {
        "df_daily":   df_daily,
        "df_weekly":  df_weekly,
        "har_result": har_result,
        "hmm_svi":    hmm_svi_results,
        "hmm_vol":    hmm_vol_results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-refit", action="store_true",
                        help="캐시 무시하고 HMM 재실행")
    args = parser.parse_args()
    outputs = main(hmm_n_init=10, hmm_B=1000, save_plots=True,
                   force_refit=args.force_refit)
