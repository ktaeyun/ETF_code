"""
simulator/scenario_main.py
==========================
시나리오별 GAP / KP 모수 재추정 + 시뮬레이션

흐름
----
1. final_scenarios_latest.csv 에서 시나리오 로드
2. 전처리 파이프라인 실행 → HMM 결과
3. 각 시나리오마다
   a. 시나리오 레짐에서 외생변수 시계열 생성
   b. 실제 GAP / KP 시계열은 유지, 외생변수만 교체해 모수 재추정
   c. Monte Carlo 시뮬레이션
   d. 검증 지표 계산
4. 시나리오 간 지표 비교표 저장

외생변수 매핑 (HMM 변수 → 모델 입력)
--------------------------------------
  GAP 모델:
    global_btc_svi  → Search Interest
    Global_RV       → VIX Volatility
  KP 모델:
    btc_volume_btc  → volume_btc
    VKOSPI_resid    → KOSPI_Volatility
    bitcoin_kr      → 실제 데이터 유지 (HMM 외 변수)

실행 예시
---------
  python simulator/scenario_main.py
  python simulator/scenario_main.py --n-simulations 500 --T 252
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from preprocessing.run_pipeline import main as run_pipeline
from preprocessing.scenario_generator import (
    from_pipeline_results,
    generate_scenario_series,
    load_hmm_results_cache,
    save_hmm_results_cache,
    scenarios_from_csv,
)
from simulator.data_loader import load_gap_exog, load_kp_exog
from simulator.gap_ou_simulator import GapOUSimulator
from simulator.kp_threshold_ou_simulator import KPThresholdOUSimulator
from simulator.visualizer import create_all_visualizations
from compare.metrics import calculate_statistical_tests, calculate_all_metrics

# ── 경로 상수 ─────────────────────────────────────────────────
SCENARIO_CSV   = _ROOT / "results" / "scenario_selection" / "final_scenarios_latest.csv"
OUT_BASE       = _ROOT / "results" / "scenario_simulator"

# HMM 변수명 → GAP/KP 외생변수명
_GAP_EXOG_MAP = {
    "global_btc_svi": "value",
    "Global_RV":      "btc_volatility",
}
_KP_EXOG_MAP = {
    "btc_volume_btc": "volume_btc",
    "VKOSPI_resid":   "KOSPI_Volatility",
}


# ══════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════

def _to_serializable(obj):
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)



# ══════════════════════════════════════════════════════════════
# 핵심 함수: 단일 시나리오 실행
# ══════════════════════════════════════════════════════════════

def run_single_scenario(
    scenario_id:   str,
    scenario:      dict,
    hmm_results:   dict,
    base_dir:      str,
    out_dir:       Path,
    T_gen:         int   = 252,
    n_simulations: int   = 1000,
    seed:          int   = 42,
    save_plots:    bool  = True,
    verbose:       bool  = True,
) -> dict:
    """단일 시나리오에 대해 GAP / KP 모수 재추정 + 시뮬레이션.

    Parameters
    ----------
    scenario_id  : 시나리오 식별자 (P01, P02, …)
    scenario     : {var_name → int} 레짐 번호
    hmm_results  : from_pipeline_results() 반환값
    T_gen        : 시나리오 외생변수 시계열 생성 길이
                   (None 이면 실제 데이터 길이에 맞춤)
    """
    def _pr(*args):
        if verbose:
            print(*args)

    _pr(f"\n{'='*60}")
    _pr(f"  시나리오: {scenario_id}")
    _pr(f"{'='*60}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if save_plots:
        for sub in ("gap", "kp"):
            (out_dir / "plots" / sub).mkdir(parents=True, exist_ok=True)

    # ── 1. 실제 GAP / KP 시계열 로드 ──────────────────────────
    df_gap_hist = load_gap_exog(base_dir=base_dir)
    df_kp_hist  = load_kp_exog(base_dir=base_dir)

    gap_series    = df_gap_hist["etf_premium"]
    kp_series     = df_kp_hist["Kimchi Premium"]
    bitcoin_kr    = df_kp_hist["bitcoin_kr"]   # HMM 외 변수 → 실제 데이터 유지

    T_gap = len(gap_series)
    T_kp  = len(kp_series)
    actual_gap = np.asarray(gap_series).flatten()
    actual_kp  = np.asarray(kp_series).flatten()

    # ── 2. 시나리오 외생변수 시계열 생성 ───────────────────────
    T_gap_gen = T_gen if T_gen else T_gap
    T_kp_gen  = T_gen if T_gen else T_kp

    df_gap_exog_sc = generate_scenario_series(
        hmm_results, scenario, T=T_gap_gen, seed=seed, inverse=True
    )
    df_kp_exog_sc = generate_scenario_series(
        hmm_results, scenario, T=T_kp_gen, seed=seed, inverse=True
    )

    # 실제 데이터 길이에 맞춰 trim/pad
    def _align(sc_arr: np.ndarray, hist_len: int) -> np.ndarray:
        if len(sc_arr) >= hist_len:
            return sc_arr[:hist_len]
        return np.pad(sc_arr, (0, hist_len - len(sc_arr)), mode="edge")

    si_sc  = pd.Series(_align(df_gap_exog_sc["global_btc_svi"].values, T_gap))
    vix_sc = pd.Series(_align(df_gap_exog_sc["Global_RV"].values,      T_gap))
    vol_sc = pd.Series(_align(df_kp_exog_sc["btc_volume_btc"].values,  T_kp))
    kv_sc  = pd.Series(_align(df_kp_exog_sc["VKOSPI_resid"].values,    T_kp))

    # ── 3. Base 모수 로드 (고정) ───────────────────────────────
    _sim_meta_path = _ROOT / "results" / "simulator" / "cache" / "sim_meta.json"
    if not _sim_meta_path.exists():
        raise FileNotFoundError(
            f"Base 시뮬레이터 캐시 없음: {_sim_meta_path}\n"
            "simulator/main.py 를 먼저 실행하세요."
        )
    with open(_sim_meta_path, encoding="utf-8") as _f:
        _meta = json.load(_f)

    gap_p = _meta["gap_params"]
    kp_p  = _meta["kp_params"]

    # 정규화 통계는 실제 데이터 기준으로 계산
    _si_raw      = df_gap_hist["value"].values
    _vix_raw     = df_gap_hist["btc_volatility"].values
    _log_vol_raw = np.log(df_kp_hist["volume_btc"].values + 1e-8)
    _kospi_raw   = df_kp_hist["KOSPI_Volatility"].values
    _bkr_raw     = df_kp_hist["bitcoin_kr"].values

    gap_sim = GapOUSimulator(
        kappa=gap_p["kappa"],
        mu=gap_p["mu"],
        sigma0=gap_p["sigma0"],
        delta1=gap_p["delta1"],
        delta2=gap_p["delta2"],
        si_mean=float(np.mean(_si_raw)),
        si_std=float(np.std(_si_raw)),
        vix_mean=float(np.mean(_vix_raw)),
        vix_std=float(np.std(_vix_raw)),
        clip=3.0,
    )
    _pr(f"\n  [GAP] Base 모수 고정  "
        f"κ={gap_sim.kappa:.6f}  μ={gap_sim.mu:.6f}  "
        f"σ0={gap_sim.sigma0:.6f}  δ1={gap_sim.delta1:.6f}  δ2={gap_sim.delta2:.6f}")

    # ── 4. Base KP 모수 로드 (고정) ────────────────────────────
    _kp_rp = kp_p["regime_params"]
    kp_sim = KPThresholdOUSimulator({
        "threshold": kp_p["threshold"],
        "regime_params": {
            int(r): {"kappa": v["kappa"], "mu": v["mu"], "sigma0": v["sigma0"]}
            for r, v in _kp_rp.items()
        },
        "delta1_regime": {int(r): v["delta1"] for r, v in _kp_rp.items()},
        "delta2_regime": {int(r): v["delta2"] for r, v in _kp_rp.items()},
        "delta3_regime": {int(r): v["delta3"] for r, v in _kp_rp.items()},
        "vol_btc_mean":     float(np.mean(_log_vol_raw)),
        "vol_btc_std":      float(np.std(_log_vol_raw)),
        "kospi_mean":       float(np.mean(_kospi_raw)),
        "kospi_std":        float(np.std(_kospi_raw)),
        "bitcoin_kr_mean":  float(np.mean(_bkr_raw)),
        "bitcoin_kr_std":   float(np.std(_bkr_raw)),
    }, clip=3.0)
    _pr(f"  [KP]  Base 모수 고정  threshold={kp_sim.threshold:.6f}")

    # ── 5. GAP Monte Carlo ─────────────────────────────────────
    _pr(f"\n  [GAP] Monte Carlo (n={n_simulations}, T={T_gap})")
    g0 = actual_gap[0]
    mc_gap = np.array([
        np.asarray(gap_sim.simulate_gap(
            T=T_gap, g0=g0,
            si_future=si_sc.values,
            vix_future=vix_sc.values,
            seed=seed + 10000 + i,
        )).flatten()
        for i in range(n_simulations)
    ])
    rep_gap = np.median(mc_gap, axis=0)

    # ── 6. KP Monte Carlo ──────────────────────────────────────
    _pr(f"\n  [KP] Monte Carlo (n={n_simulations}, T={T_kp})")
    kp0 = actual_kp[0]
    mc_kp = np.array([
        np.asarray(kp_sim.simulate_kp(
            T=T_kp, kp0=kp0,
            volume_btc_future=vol_sc.values,
            kospi_vol_future=kv_sc.values,
            bitcoin_kr_future=bitcoin_kr.values,
            seed=seed + 20000 + i,
        )).flatten()
        for i in range(n_simulations)
    ])
    rep_kp = np.median(mc_kp, axis=0)

    # ── 7. 검증 지표 ───────────────────────────────────────────
    actual_gap_ch = np.diff(actual_gap)
    mc_gap_ch     = np.diff(mc_gap, axis=1)
    rep_gap_ch    = np.diff(rep_gap)

    gap_stat  = calculate_statistical_tests(actual_gap_ch, mc_gap_ch)
    gap_metr  = calculate_all_metrics(
        actual_nav=actual_gap, simulated_nav=rep_gap,
        actual_returns=actual_gap_ch, simulated_returns=rep_gap_ch,
        monte_carlo_nav_paths=mc_gap, monte_carlo_returns_paths=mc_gap_ch,
    )
    actual_kp_ch = np.diff(actual_kp)
    mc_kp_ch     = np.diff(mc_kp, axis=1)
    rep_kp_ch    = np.diff(rep_kp)

    kp_stat   = calculate_statistical_tests(actual_kp_ch, mc_kp_ch)
    kp_metr   = calculate_all_metrics(
        actual_nav=actual_kp, simulated_nav=rep_kp,
        actual_returns=actual_kp_ch, simulated_returns=rep_kp_ch,
        monte_carlo_nav_paths=mc_kp, monte_carlo_returns_paths=mc_kp_ch,
    )
    _pr(f"\n  [GAP 결과]  WMCR_price={gap_metr.get('wmcr_price', float('nan')):.4f}  "
        f"DTW={gap_metr.get('dtw_price', float('nan')):.4f}  "
        f"PIT-KS p={gap_stat['pit_ks']['ks_pvalue']:.4f}")
    _pr(f"  [KP  결과]  WMCR_price={kp_metr.get('wmcr_price', float('nan')):.4f}  "
        f"DTW={kp_metr.get('dtw_price', float('nan')):.4f}  "
        f"PIT-KS p={kp_stat['pit_ks']['ks_pvalue']:.4f}")

    # ── 8. 저장 ────────────────────────────────────────────────
    # 파라미터 JSON
    params_out = {
        "scenario_id": scenario_id,
        "scenario_regimes": scenario,
        "gap_params": {
            "kappa":  gap_sim.kappa,
            "mu":     gap_sim.mu,
            "sigma0": gap_sim.sigma0,
            "delta1": gap_sim.delta1,
            "delta2": gap_sim.delta2,
        },
        "kp_params": {
            "threshold": kp_sim.threshold,
            "regime_params": {
                str(r): {
                    "kappa":  kp_sim.regime_params[r]["kappa"],
                    "mu":     kp_sim.regime_params[r]["mu"],
                    "sigma0": kp_sim.regime_params[r]["sigma0"],
                    "delta1": kp_sim.delta1_regime[r],
                    "delta2": kp_sim.delta2_regime[r],
                    "delta3": kp_sim.delta3_regime[r],
                }
                for r in [0, 1, 2]
            },
        },
        "gap_metrics": {
            k: float(v) for k, v in gap_metr.items()
            if isinstance(v, (int, float, np.floating))
        },
        "kp_metrics": {
            k: float(v) for k, v in kp_metr.items()
            if isinstance(v, (int, float, np.floating))
        },
        "gap_pit_ks_pvalue": float(gap_stat["pit_ks"]["ks_pvalue"]),
        "kp_pit_ks_pvalue":  float(kp_stat["pit_ks"]["ks_pvalue"]),
    }
    with open(out_dir / "params_and_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_serializable(params_out), f, ensure_ascii=False, indent=2)

    # 시뮬레이션 결과 CSV
    pd.DataFrame({
        "actual_gap": actual_gap,
        "simulated_gap": rep_gap,
    }).to_csv(out_dir / "gap_results.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "actual_kp": actual_kp,
        "simulated_kp": rep_kp,
    }).to_csv(out_dir / "kp_results.csv", index=False, encoding="utf-8-sig")

    # 시각화
    if save_plots:
        create_all_visualizations(
            actual_nav=actual_gap, simulated_nav=rep_gap,
            actual_returns=actual_gap_ch, simulated_returns=rep_gap_ch,
            output_dir=str(out_dir / "plots" / "gap"),
            monte_carlo_nav_paths=mc_gap, monte_carlo_returns_paths=mc_gap_ch,
            module="gap",
        )
        create_all_visualizations(
            actual_nav=actual_kp, simulated_nav=rep_kp,
            actual_returns=actual_kp_ch, simulated_returns=rep_kp_ch,
            output_dir=str(out_dir / "plots" / "kp"),
            monte_carlo_nav_paths=mc_kp, monte_carlo_returns_paths=mc_kp_ch,
            module="kp",
        )

    return {
        "scenario_id":   scenario_id,
        "gap_sim":       gap_sim,
        "kp_sim":        kp_sim,
        "mc_gap":        mc_gap,
        "mc_kp":         mc_kp,
        "rep_gap":       rep_gap,
        "rep_kp":        rep_kp,
        "actual_gap":    actual_gap,
        "actual_kp":     actual_kp,
        "gap_metrics":   gap_metr,
        "kp_metrics":    kp_metr,
        "gap_stat":      gap_stat,
        "kp_stat":       kp_stat,
        "params_out":    params_out,
    }


# ══════════════════════════════════════════════════════════════
# Base 결과 로드 및 비교
# ══════════════════════════════════════════════════════════════

_BASE_DIR     = _ROOT / "results" / "simulator"
_BASE_CACHE   = _BASE_DIR / "cache" / "sim_arrays.npz"


def load_base_mc_arrays() -> dict | None:
    """Base MC 배열 로드 (results/simulator/cache/sim_arrays.npz).

    Keys: mc_nav, mc_nav_ret, mc_gap, mc_kp
    """
    if not _BASE_CACHE.exists():
        return None
    arrs = dict(np.load(str(_BASE_CACHE)))
    print(f"  [Base MC] 캐시 로드: {_BASE_CACHE.name}")
    return arrs


def compute_risk_metrics(mc_array: np.ndarray, actual: np.ndarray,
                         label: str, alpha: float = 0.05) -> dict:
    """가격경로 MC 배열에서 리스크 지표 산출.

    Parameters
    ----------
    mc_array : (N, T) - N개 경로, T 시점
    actual   : (T,)  - 실제 관측값
    alpha    : VaR/CVaR 신뢰수준 (0.05 = 95%)
    """
    N, T = mc_array.shape

    # 종단 분포 (terminal distribution)
    terminal = mc_array[:, -1]

    # 경로별 변화량 (수익률)
    returns  = np.diff(mc_array, axis=1)          # (N, T-1)

    # ── 분포 지표 ──────────────────────────────────────────
    var_95   = float(np.percentile(terminal, alpha * 100))
    cvar_95  = float(terminal[terminal <= var_95].mean()) if (terminal <= var_95).any() else var_95
    var_99   = float(np.percentile(terminal, 1.0))
    cvar_99  = float(terminal[terminal <= var_99].mean()) if (terminal <= var_99).any() else var_99

    # ── 경로 지표 ──────────────────────────────────────────
    # 최대 낙폭(Max Drawdown): 각 경로의 peak-to-trough
    cummax   = np.maximum.accumulate(mc_array, axis=1)
    drawdowns = (mc_array - cummax) / (np.abs(cummax) + 1e-12)
    max_dd   = float(drawdowns.min(axis=1).mean())           # 평균 MDD

    # 변동성 (경로별 변화량의 표준편차 평균)
    vol_paths = returns.std(axis=1)
    vol_mean  = float(vol_paths.mean())

    # 실제값과의 median 괴리
    median_path = np.median(mc_array, axis=0)
    mae_vs_actual = float(np.mean(np.abs(median_path - actual[:T])))

    # 분포 형태
    from scipy import stats as sp_stats
    skew = float(sp_stats.skew(terminal))
    kurt = float(sp_stats.kurtosis(terminal))

    # 5/25/50/75/95th 분위 밴드
    p5,  p25, p50, p75, p95 = [
        float(np.percentile(terminal, q)) for q in [5, 25, 50, 75, 95]
    ]

    return {
        "label":       label,
        "VaR_95":      var_95,
        "CVaR_95":     cvar_95,
        "VaR_99":      var_99,
        "CVaR_99":     cvar_99,
        "Max_DD_mean": max_dd,
        "Volatility":  vol_mean,
        "MAE_actual":  mae_vs_actual,
        "Skewness":    skew,
        "Kurtosis":    kurt,
        "p5":  p5,  "p25": p25, "p50": p50, "p75": p75, "p95": p95,
    }


def _build_korean_etf_paths(mc_nav: np.ndarray, mc_gap: np.ndarray,
                             mc_kp: np.ndarray, anchor: float = 10000.0) -> np.ndarray:
    """NAV*(1+GAP)*(1+KP) 결합 경로 생성 후 anchor(원)로 정규화.

    N(경로 수)과 T(시간) 모두 최솟값으로 정렬.
    """
    min_N = min(mc_nav.shape[0], mc_gap.shape[0], mc_kp.shape[0])
    min_T = min(mc_nav.shape[1], mc_gap.shape[1], mc_kp.shape[1])
    raw = (mc_nav[:min_N, :min_T]
           * (1 + mc_gap[:min_N, :min_T])
           * (1 + mc_kp[:min_N, :min_T]))
    init = raw[:, 0:1]
    init = np.where(init == 0, 1.0, init)
    return raw * (anchor / init)


def plot_price_path_comparison(
    base_mc:     dict,
    all_results: dict,
    out_dir:     Path,
    anchor:      float = 10000.0,
) -> None:
    """Base vs 시나리오: 결합 가격경로(NAV*(1+GAP)*(1+KP)) Fan Chart + 리스크 지표."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib 없음 -- 플롯 생략")
        return

    mc_nav = base_mc.get("mc_nav")
    mc_gap = base_mc.get("mc_gap")
    mc_kp  = base_mc.get("mc_kp")
    if mc_nav is None or mc_gap is None or mc_kp is None:
        print("  [!] Base MC 배열 불완전 -- 플롯 생략")
        return

    # ── 실제 가격 로드 ────────────────────────────────────────
    etf_csv = _BASE_DIR / "korean_etf_price.csv"
    actual_etf = None
    if etf_csv.exists():
        df_etf = pd.read_csv(etf_csv)
        actual_etf = df_etf["actual_etf_krw"].values

    # ── Base 결합 경로 ────────────────────────────────────────
    base_combined = _build_korean_etf_paths(mc_nav, mc_gap, mc_kp, anchor)  # (N, T)

    n_sc   = len(all_results)
    COLORS = ["#d6604d", "#4dac26", "#762a83", "#e66101"]

    fig, axes = plt.subplots(n_sc + 1, 2, figsize=(14, 4 * (n_sc + 1)))
    if n_sc == 0:
        axes = axes.reshape(1, 2)

    def _fan_ax(ax, mc, label, color, actual=None):
        x   = np.arange(mc.shape[1])
        p5  = np.percentile(mc, 5,  axis=0)
        p25 = np.percentile(mc, 25, axis=0)
        p50 = np.median(mc, axis=0)
        p75 = np.percentile(mc, 75, axis=0)
        p95 = np.percentile(mc, 95, axis=0)
        ax.fill_between(x, p5,  p95, alpha=0.12, color=color, label="5-95%")
        ax.fill_between(x, p25, p75, alpha=0.28, color=color, label="25-75%")
        ax.plot(x, p50, color=color, lw=1.8, label=f"{label} median")
        if actual is not None:
            ax.plot(x[:len(actual)], actual[:len(x)],
                    color="#111111", lw=1.3, label="Actual", zorder=5)
        ax.set_ylabel("ETF Price (KRW)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
        )

    # ── 행 0: Base Fan Chart + Terminal 분포 ─────────────────
    _fan_ax(axes[0, 0], base_combined, "Base", "#2166ac", actual_etf)
    axes[0, 0].set_title("Base -- ETF 가격경로 Fan Chart (KRW)", fontsize=9)

    ax_hist = axes[0, 1]
    ax_hist.hist(base_combined[:, -1], bins=50, density=True,
                 color="#2166ac", alpha=0.7, label="Base terminal")
    for q, ls in [(5, "--"), (50, "-"), (95, ":")]:
        v = np.percentile(base_combined[:, -1], q)
        ax_hist.axvline(v, color="#2166ac", lw=1.5, linestyle=ls,
                        label=f"p{q}={v:,.0f}")
    ax_hist.set_title("Base -- Terminal Price 분포", fontsize=9)
    ax_hist.set_xlabel("ETF Price (KRW)", fontsize=8)
    ax_hist.legend(fontsize=7)
    ax_hist.grid(True, linestyle="--", alpha=0.4)
    ax_hist.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )

    # ── 행 1~: 시나리오 ──────────────────────────────────────
    risk_rows = []
    for row_i, (sid, res) in enumerate(all_results.items(), 1):
        color = COLORS[(row_i - 1) % len(COLORS)]

        # 시나리오 결합 경로: Base NAV + Scenario GAP/KP
        sc_combined = _build_korean_etf_paths(
            mc_nav, res["mc_gap"], res["mc_kp"], anchor
        )
        min_T = sc_combined.shape[1]

        # Fan Chart: Base median + Scenario band 겹쳐 표시
        ax = axes[row_i, 0]
        _fan_ax(ax, sc_combined, sid, color, actual_etf)
        p50_base = np.median(base_combined[:, :min_T], axis=0)
        ax.plot(np.arange(min_T), p50_base,
                color="#2166ac", lw=1.2, linestyle=":", label="Base median", alpha=0.8)
        ax.set_title(f"{sid} vs Base -- ETF 가격경로 Fan Chart (KRW)", fontsize=9)
        ax.legend(fontsize=7)

        # Terminal 분포 비교
        ax_h = axes[row_i, 1]
        t_base = base_combined[:, -1]
        t_sc   = sc_combined[:, -1]
        ax_h.hist(t_base, bins=50, density=True, color="#2166ac", alpha=0.5, label="Base")
        ax_h.hist(t_sc,   bins=50, density=True, color=color,     alpha=0.5, label=sid)
        for t_arr, col in [(t_base, "#2166ac"), (t_sc, color)]:
            ax_h.axvline(np.percentile(t_arr, 5),  color=col, lw=1.5, linestyle="--")
            ax_h.axvline(np.percentile(t_arr, 50), color=col, lw=1.2, linestyle="-")
        ax_h.set_title(f"{sid} vs Base -- Terminal Price 분포", fontsize=9)
        ax_h.set_xlabel("ETF Price (KRW)", fontsize=8)
        ax_h.legend(fontsize=7)
        ax_h.grid(True, linestyle="--", alpha=0.4)
        ax_h.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
        )

        # 리스크 지표
        act_dummy = np.full(min_T, anchor)  # 실제 가격이 없으면 anchor 기준
        if actual_etf is not None:
            act_dummy = actual_etf
        risk_rows.append({"Scenario": "Base",
                          **compute_risk_metrics(base_combined, act_dummy, "Base")})
        risk_rows.append({"Scenario": sid,
                          **compute_risk_metrics(sc_combined, act_dummy, sid)})

    fig.suptitle("Base vs Scenario: Korean ETF 결합 가격경로 비교\n"
                 "(NAV * (1+GAP) * (1+KP), 기준 10,000 KRW)",
                 fontsize=11, fontweight="bold")
    out_path = out_dir / "korean_etf_price_comparison.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ETF 가격경로 비교 플롯: {out_path}")

    # ── 리스크 지표 테이블 ────────────────────────────────────
    if risk_rows:
        risk_df = (pd.DataFrame(risk_rows)
                   .drop_duplicates(subset=["Scenario"])
                   .reset_index(drop=True))
        disp_cols = ["Scenario", "VaR_95", "CVaR_95", "VaR_99", "CVaR_99",
                     "Max_DD_mean", "Volatility", "Skewness", "Kurtosis",
                     "p5", "p50", "p95"]
        risk_df = risk_df[[c for c in disp_cols if c in risk_df.columns]]
        risk_df[risk_df.select_dtypes(float).columns] = \
            risk_df.select_dtypes(float).round(2)

        risk_path = out_dir / "korean_etf_risk_metrics.csv"
        risk_df.to_csv(risk_path, index=False, encoding="utf-8-sig")

        print("\n  [리스크 지표 비교]")
        print(risk_df.to_string(index=False))
        print(f"\n  리스크 지표 저장: {risk_path}")


def load_base_results() -> dict | None:
    """results/simulator/ 에서 base 시뮬레이션 결과 로드.

    Returns None if base results don't exist.
    """
    val_path = _BASE_DIR / "validation_results.json"
    gap_path = _BASE_DIR / "gap_simulation_results.csv"
    kp_path  = _BASE_DIR / "kp_simulation_results.csv"

    if not val_path.exists():
        return None

    with open(val_path, encoding="utf-8") as f:
        val = json.load(f)

    result = {"validation": val}
    if gap_path.exists():
        result["gap_df"] = pd.read_csv(gap_path)
    if kp_path.exists():
        result["kp_df"] = pd.read_csv(kp_path)
    return result


def _base_row(base: dict) -> dict:
    """base 결과에서 비교표 행 생성."""
    v   = base["validation"]
    gp  = v.get("gap", {})
    kp  = v.get("kp", {})
    ou  = gp.get("ou_params", {})
    kpp = kp.get("threshold_ou_params", {})
    gm  = gp.get("validation_metrics", {})
    km  = kp.get("validation_metrics", {})
    gs  = gp.get("statistical_tests", {})
    ks  = kp.get("statistical_tests", {})

    return {
        "Scenario_ID":   "Base",
        "gap_kappa":     ou.get("kappa"),
        "gap_mu":        ou.get("mu"),
        "gap_sigma0":    ou.get("sigma0"),
        "gap_delta1_SI": ou.get("delta1"),
        "gap_delta2_VIX":ou.get("delta2"),
        "gap_wmcr_price":gm.get("wmcr_price"),
        "gap_dtw_price": gm.get("dtw_price"),
        "gap_pmc":       gm.get("pmc"),
        "gap_pit_ks_p":  gs.get("pit_ks", {}).get("ks_pvalue"),
        "kp_threshold":  kpp.get("threshold"),
        "kp_kappa_r0":   kpp.get("regime_params", {}).get("0", {}).get("kappa"),
        "kp_kappa_r1":   kpp.get("regime_params", {}).get("1", {}).get("kappa"),
        "kp_kappa_r2":   kpp.get("regime_params", {}).get("2", {}).get("kappa"),
        "kp_wmcr_price": km.get("wmcr_price"),
        "kp_dtw_price":  km.get("dtw_price"),
        "kp_pmc":        km.get("pmc"),
        "kp_pit_ks_p":   ks.get("pit_ks", {}).get("ks_pvalue"),
    }


def plot_base_vs_scenarios(
    base:        dict,
    all_results: dict,
    out_dir:     Path,
) -> None:
    """Base와 시나리오별 GAP / KP 시뮬레이션 비교 시각화."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  [!] matplotlib 없음 — 비교 플롯 생략")
        return

    n_sc = len(all_results)
    fig = plt.figure(figsize=(14, 5 * (n_sc + 1)))
    gs  = gridspec.GridSpec(n_sc + 1, 2, figure=fig, hspace=0.45, wspace=0.3)

    def _plot_one(ax, actual, base_sim, sc_sim, title, ylabel):
        T = len(actual)
        x = np.arange(T)
        ax.plot(x, actual,   color="#222222", lw=1.2, label="Actual",   zorder=3)
        ax.plot(x, base_sim, color="#2166ac", lw=1.0, label="Base sim", zorder=2, linestyle="--")
        if sc_sim is not None:
            ax.plot(x, sc_sim, color="#d6604d", lw=1.0, label="Scenario sim", zorder=2, linestyle="-.")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    # ── Base 행 ──────────────────────────────────────────────
    gap_df = base.get("gap_df")
    kp_df  = base.get("kp_df")
    if gap_df is not None:
        ax = fig.add_subplot(gs[0, 0])
        _plot_one(ax, gap_df["actual_gap"].values, gap_df["simulated_gap"].values,
                  None, "Base — GAP", "ETF Premium")
    if kp_df is not None:
        ax = fig.add_subplot(gs[0, 1])
        _plot_one(ax, kp_df["actual_kp"].values, kp_df["simulated_kp"].values,
                  None, "Base — KP (Kimchi Premium)", "KP")

    # ── 시나리오별 행 ────────────────────────────────────────
    for row_i, (sid, res) in enumerate(all_results.items(), 1):
        actual_gap = res["actual_gap"]
        rep_gap    = res["rep_gap"]
        actual_kp  = res["actual_kp"]
        rep_kp     = res["rep_kp"]

        base_gap_sim = gap_df["simulated_gap"].values if gap_df is not None else None
        base_kp_sim  = kp_df["simulated_kp"].values  if kp_df  is not None else None

        ax = fig.add_subplot(gs[row_i, 0])
        _plot_one(ax, actual_gap, base_gap_sim, rep_gap,
                  f"{sid} — GAP", "ETF Premium")

        ax = fig.add_subplot(gs[row_i, 1])
        _plot_one(ax, actual_kp, base_kp_sim, rep_kp,
                  f"{sid} — KP (Kimchi Premium)", "KP")

    fig.suptitle("Base vs Scenario 시뮬레이션 비교", fontsize=12, fontweight="bold")
    out_path = out_dir / "base_vs_scenario_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  비교 플롯 저장: {out_path}")


# ══════════════════════════════════════════════════════════════
# 비교 요약 테이블
# ══════════════════════════════════════════════════════════════

def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """시나리오 간 주요 지표 비교 DataFrame 생성."""
    rows = []
    for sid, res in all_results.items():
        p = res["params_out"]
        rows.append({
            "Scenario_ID":        sid,
            # GAP 파라미터
            "gap_kappa":          p["gap_params"]["kappa"],
            "gap_mu":             p["gap_params"]["mu"],
            "gap_sigma0":         p["gap_params"]["sigma0"],
            "gap_delta1_SI":      p["gap_params"]["delta1"],
            "gap_delta2_VIX":     p["gap_params"]["delta2"],
            # GAP 검증
            "gap_wmcr_price":     p["gap_metrics"].get("wmcr_price"),
            "gap_dtw_price":      p["gap_metrics"].get("dtw_price"),
            "gap_pmc":            p["gap_metrics"].get("pmc"),
            "gap_pit_ks_p":       p["gap_pit_ks_pvalue"],
            # KP 임계값
            "kp_threshold":       p["kp_params"]["threshold"],
            # KP 레짐별 kappa
            "kp_kappa_r0":        p["kp_params"]["regime_params"]["0"]["kappa"],
            "kp_kappa_r1":        p["kp_params"]["regime_params"]["1"]["kappa"],
            "kp_kappa_r2":        p["kp_params"]["regime_params"]["2"]["kappa"],
            # KP 검증
            "kp_wmcr_price":      p["kp_metrics"].get("wmcr_price"),
            "kp_dtw_price":       p["kp_metrics"].get("dtw_price"),
            "kp_pmc":             p["kp_metrics"].get("pmc"),
            "kp_pit_ks_p":        p["kp_pit_ks_pvalue"],
        })
    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    df[numeric_cols] = df[numeric_cols].round(6)
    return df


# ══════════════════════════════════════════════════════════════
# 레짐 캐시에서 K를 읽어 고정 K로 HMM 학습
# ══════════════════════════════════════════════════════════════

def _build_hmm_with_fixed_k(regime_cache_csv: Path) -> dict:
    """regime_df_cache.csv에서 변수별 K를 읽고 고정 K로 HMM EM만 실행.

    부트스트랩 LRT(K 선택)를 생략하므로 run_pipeline()보다 빠름.
    """
    from preprocessing.run_pipeline import load_data, step_har
    from preprocessing.scenario_generator import fit_scenario_hmm

    # ── K 읽기 ────────────────────────────────────────────────
    df_cache = pd.read_csv(regime_cache_csv, index_col=0, parse_dates=True)
    _REGIME_TO_VAR = {
        "global_btc_svi_regime":   "global_btc_svi",
        "domestic_btc_svi_regime": "domestic_btc_svi",
        "btc_volume_btc_regime":   "btc_volume_btc",
        "Global_RV_regime":        "Global_RV",
        "VKOSPI_resid_regime":     "VKOSPI_resid",
    }
    k_per_var = {
        var: int(df_cache[col].dropna().nunique())
        for col, var in _REGIME_TO_VAR.items()
        if col in df_cache.columns
    }
    print(f"  변수별 K: {k_per_var}")

    # ── 데이터 로드 ───────────────────────────────────────────
    df_daily, df_weekly = load_data()
    har_result = step_har(df_daily, save=False)

    results = {}

    # 주별 SVI / Volume
    for col in ["global_btc_svi", "domestic_btc_svi", "btc_volume_btc"]:
        if col not in k_per_var or col not in df_weekly.columns:
            continue
        series = df_weekly[col].dropna().copy()
        series.name = col
        results[col] = fit_scenario_hmm(series, n_states=k_per_var[col])
        print(f"    {col}: K={k_per_var[col]}  μ={results[col].mu}")

    # 일별 Global_RV
    if "Global_RV" in k_per_var and "Global_RV" in df_daily.columns:
        series = df_daily["Global_RV"].dropna().copy()
        series.name = "Global_RV"
        results["Global_RV"] = fit_scenario_hmm(series, n_states=k_per_var["Global_RV"])
        print(f"    Global_RV: K={k_per_var['Global_RV']}  μ={results['Global_RV'].mu}")

    # 일별 VKOSPI_resid (HAR 잔차)
    if "VKOSPI_resid" in k_per_var:
        vkospi_idx = df_daily["VKOSPI"].dropna().index
        resid = pd.Series(
            har_result.residuals_z,
            index=vkospi_idx[22:22 + len(har_result.residuals_z)],
            name="VKOSPI_resid",
        )
        results["VKOSPI_resid"] = fit_scenario_hmm(resid, n_states=k_per_var["VKOSPI_resid"])
        print(f"    VKOSPI_resid: K={k_per_var['VKOSPI_resid']}  μ={results['VKOSPI_resid'].mu}")

    return results


# ══════════════════════════════════════════════════════════════
# 인터랙티브 시나리오 선택
# ══════════════════════════════════════════════════════════════

def _prompt_scenario_selection_csv(df_sc: pd.DataFrame) -> set[str] | None:
    """시나리오 CSV를 보여주고 실행할 ID 집합을 반환. None이면 전체 실행."""
    ids = df_sc["Scenario_ID"].tolist()
    label_col = "combo_label" if "combo_label" in df_sc.columns else None

    print("\n" + "=" * 60)
    print("  실행할 시나리오를 선택하세요 (HMM 실행 전)")
    print("=" * 60)
    for i, row in enumerate(df_sc.itertuples(), 1):
        sid = row.Scenario_ID
        label = getattr(row, "combo_label", "") if label_col else ""
        print(f"  [{i:>2}] {sid}  --  {label}")
    print()
    print("  입력 예시:")
    print("    전체 실행  : all 또는 엔터")
    print("    단일 선택  : 1")
    print("    복수 선택  : 1,3,5  또는  S01,S03")
    print()

    while True:
        raw = input("  선택: ").strip()

        if raw == "" or raw.lower() == "all":
            print(f"  → 전체 {len(ids)}개 시나리오 실행")
            return None

        tokens = [t.strip() for t in raw.replace(" ", ",").split(",") if t.strip()]
        selected: list[str] = []
        valid = True
        for tok in tokens:
            if tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(ids):
                    selected.append(ids[idx])
                else:
                    print(f"  [오류] 번호 범위 초과: {tok} (1~{len(ids)})")
                    valid = False
                    break
            elif tok in ids:
                selected.append(tok)
            else:
                print(f"  [오류] 알 수 없는 시나리오: {tok}")
                valid = False
                break

        if not valid:
            continue

        result = list(dict.fromkeys(selected))  # 순서 유지 + 중복 제거
        print(f"  → 선택된 시나리오: {result}")
        return set(result)


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="시나리오별 GAP/KP 모수 재추정 + 시뮬레이션")
    parser.add_argument("--scenario-csv", type=str, default=str(SCENARIO_CSV))
    parser.add_argument("--out-dir",      type=str, default=str(OUT_BASE))
    parser.add_argument("--T",            type=int, default=252,
                        help="시나리오 외생변수 생성 길이 (기본 252)")
    parser.add_argument("--n-simulations",type=int, default=1000)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--no-plots",     action="store_true")
    parser.add_argument("--scenarios",    type=str, default=None,
                        help="실행할 시나리오 ID 목록 (쉼표 구분, 예: P01,P03)")
    parser.add_argument("--force-refit",  action="store_true",
                        help="HMM 캐시 무시하고 재실행")
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # ── Step 0: 시나리오 선택 (HMM 실행 전) ───────────────────
    scenario_csv = Path(args.scenario_csv)
    if not scenario_csv.exists():
        raise FileNotFoundError(
            f"시나리오 CSV 없음: {scenario_csv}\n"
            "analysis/1_scenario_selection.py 를 먼저 실행하세요."
        )
    df_sc = pd.read_csv(scenario_csv)

    if args.scenarios:
        selected_ids = set(args.scenarios.split(","))
    else:
        selected_ids = _prompt_scenario_selection_csv(df_sc)

    # ── Step 1: HMM 파라미터 로드 (캐시 우선) ────────────────
    print("\n" + "=" * 60)
    print("  [Step 1] HMM 파라미터 로드")
    print("=" * 60)
    _hmm_cache_dir = _ROOT / "results" / "cache"
    hmm_results = None
    if not args.force_refit:
        hmm_results = load_hmm_results_cache(_hmm_cache_dir, n_init=10, B=1000)

    if hmm_results is None:
        _regime_cache = _ROOT / "results" / "cache" / "regime_df_cache.csv"
        if _regime_cache.exists() and not args.force_refit:
            print("  [Cache MISS] 레짐 캐시에서 K 읽어 HMM 학습 (부트스트랩 LRT 생략)")
            hmm_results = _build_hmm_with_fixed_k(_regime_cache)
        else:
            print("  [Cache MISS] 전처리 파이프라인 실행 (K 선택 포함)")
            pipeline_out = run_pipeline(hmm_n_init=10, hmm_B=1000, save_plots=False,
                                        force_refit=args.force_refit)
            hmm_results = from_pipeline_results(pipeline_out)
        save_hmm_results_cache(hmm_results, _hmm_cache_dir, n_init=10, B=1000)

    # ── Step 2: 시나리오 로드 및 필터링 ───────────────────────
    print("\n" + "=" * 60)
    print("  [Step 2] 시나리오 로드")
    print("=" * 60)
    all_scenarios = scenarios_from_csv(scenario_csv, hmm_results)

    if selected_ids is not None:
        all_scenarios = {k: v for k, v in all_scenarios.items() if k in selected_ids}
    print(f"  실행 대상: {list(all_scenarios.keys())}")

    base_dir = str(_ROOT)

    # ── Step 3: 시나리오별 실행 ────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  [Step 3] 시나리오별 GAP/KP 모수 재추정 + 시뮬레이션")
    print(f"           총 {len(all_scenarios)}개 시나리오")
    print("=" * 60)

    all_results: dict = {}
    for sid, scenario in all_scenarios.items():
        scenario_out = out_base / sid
        try:
            res = run_single_scenario(
                scenario_id   = sid,
                scenario      = scenario,
                hmm_results   = hmm_results,
                base_dir      = base_dir,
                out_dir       = scenario_out,
                T_gen         = args.T,
                n_simulations = args.n_simulations,
                seed          = args.seed,
                save_plots    = not args.no_plots,
                verbose       = True,
            )
            all_results[sid] = res
        except Exception as e:
            print(f"  [!] {sid} 실패: {e}")

    if not all_results:
        print("  실행된 시나리오 없음.")
        return {}

    # ── Step 4: Base vs 시나리오 비교 ────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 4] Base vs 시나리오 비교")
    print("=" * 60)

    base = load_base_results()
    if base is None:
        print("  [!] Base 결과 없음 (results/simulator/validation_results.json)")
        print("      simulator/main.py 를 먼저 실행하세요.")

    comparison_df = build_comparison_table(all_results)

    # Base 행을 맨 위에 추가
    if base is not None:
        base_row_df = pd.DataFrame([_base_row(base)])
        numeric_cols = base_row_df.select_dtypes(include=[float, int]).columns
        base_row_df[numeric_cols] = base_row_df[numeric_cols].round(6)
        comparison_df = pd.concat([base_row_df, comparison_df], ignore_index=True)

    comparison_path = out_base / "comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    print(comparison_df[[
        "Scenario_ID",
        "gap_kappa", "gap_mu", "gap_sigma0",
        "gap_wmcr_price", "gap_pit_ks_p",
        "kp_threshold",
        "kp_wmcr_price", "kp_pit_ks_p",
    ]].to_string(index=False))
    print(f"\n  비교 요약 저장: {comparison_path}")

    # 가격경로 Fan Chart + 리스크 지표 비교
    base_mc = load_base_mc_arrays()
    if base_mc is not None:
        plot_price_path_comparison(base_mc, all_results, out_base)
    elif not args.no_plots and base is not None:
        plot_base_vs_scenarios(base, all_results, out_base)

    print(f"  시나리오별 결과: {out_base}/<Scenario_ID>/")

    return all_results


if __name__ == "__main__":
    main()
