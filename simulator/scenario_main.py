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
    scenarios_from_csv,
)
from simulator.data_loader import load_gap_exog, load_kp_exog
from simulator.gap_ou_simulator import fit_gap_ou
from simulator.kp_threshold_ou_simulator import fit_kp_threshold_ou
from simulator.visualizer import create_all_visualizations
from compare.metrics import calculate_statistical_tests, calculate_all_metrics
from simulator.wmcr_test import wmcr_binomial_test

# ── 경로 상수 ─────────────────────────────────────────────────
SCENARIO_CSV   = _ROOT / "results" / "scenario_selection" / "final_scenarios_latest.csv"
OUT_BASE       = _ROOT / "results" / "scenario_simulator"

# HMM 변수명 → GAP/KP 외생변수명
_GAP_EXOG_MAP = {
    "global_btc_svi": "Search Interest",
    "Global_RV":      "VIX Volatility",
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


def _wmcr_bands(mc_array: np.ndarray, actual: np.ndarray,
                p_targets: np.ndarray) -> dict:
    """Monte Carlo 경로에서 WMCR 검정 결과 반환."""
    T = mc_array.shape[1]
    bands_L = np.zeros((len(p_targets), T))
    bands_U = np.zeros((len(p_targets), T))
    for k, p in enumerate(p_targets):
        a = (1 - p) / 2
        for t in range(T):
            bands_L[k, t] = np.percentile(mc_array[:, t], a * 100)
            bands_U[k, t] = np.percentile(mc_array[:, t], (1 - a) * 100)
    C_obs = np.array([
        np.mean((bands_L[k] <= actual) & (actual <= bands_U[k]))
        for k in range(len(p_targets))
    ])
    return wmcr_binomial_test(T=T, p_targets=p_targets, C_obs=C_obs,
                               alpha=0.05, verbose=False)


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

    _pr(f"\n{'═'*60}")
    _pr(f"  시나리오: {scenario_id}")
    _pr(f"{'═'*60}")

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

    # ── 3. GAP 모수 재추정 ─────────────────────────────────────
    _pr(f"\n  [GAP] 모수 추정 (외생변수: global_btc_svi → SI, Global_RV → VIX)")
    gap_sim = fit_gap_ou(
        gap_series=gap_series,
        si_series=si_sc,
        vix_series=vix_sc,
        clip=3.0,
        regularization=0.01,
    )
    _pr(f"    κ={gap_sim.kappa:.6f}  μ={gap_sim.mu:.6f}  "
        f"σ0={gap_sim.sigma0:.6f}  δ1={gap_sim.delta1:.6f}  δ2={gap_sim.delta2:.6f}")

    # ── 4. KP 모수 재추정 ──────────────────────────────────────
    _pr(f"\n  [KP] 모수 추정 (외생변수: btc_volume_btc, VKOSPI_resid; bitcoin_kr 실제 유지)")
    kp_sim = fit_kp_threshold_ou(
        kp_series=kp_series,
        volume_btc=vol_sc,
        kospi_vol=kv_sc,
        bitcoin_kr=bitcoin_kr,
        threshold=None,
        clip=3.0,
        regularization=0.01,
    )
    _pr(f"    threshold={kp_sim.threshold:.6f}")

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
    p_targets = np.array([0.50, 0.80, 0.95])

    actual_gap_ch = np.diff(actual_gap)
    mc_gap_ch     = np.diff(mc_gap, axis=1)
    rep_gap_ch    = np.diff(rep_gap)

    gap_stat  = calculate_statistical_tests(actual_gap_ch, mc_gap_ch)
    gap_metr  = calculate_all_metrics(
        actual_nav=actual_gap, simulated_nav=rep_gap,
        actual_returns=actual_gap_ch, simulated_returns=rep_gap_ch,
        monte_carlo_nav_paths=mc_gap, monte_carlo_returns_paths=mc_gap_ch,
    )
    gap_wmcr  = _wmcr_bands(mc_gap, actual_gap, p_targets)

    actual_kp_ch = np.diff(actual_kp)
    mc_kp_ch     = np.diff(mc_kp, axis=1)
    rep_kp_ch    = np.diff(rep_kp)

    kp_stat   = calculate_statistical_tests(actual_kp_ch, mc_kp_ch)
    kp_metr   = calculate_all_metrics(
        actual_nav=actual_kp, simulated_nav=rep_kp,
        actual_returns=actual_kp_ch, simulated_returns=rep_kp_ch,
        monte_carlo_nav_paths=mc_kp, monte_carlo_returns_paths=mc_kp_ch,
    )
    kp_wmcr   = _wmcr_bands(mc_kp, actual_kp, p_targets)

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
            "kappa": gap_sim.kappa, "mu": gap_sim.mu, "sigma0": gap_sim.sigma0,
            "delta1": gap_sim.delta1, "delta2": gap_sim.delta2,
        },
        "kp_params": {
            "threshold": kp_sim.threshold,
            "regime_params": {
                str(r): {
                    "kappa": kp_sim.regime_params[r]["kappa"],
                    "mu":    kp_sim.regime_params[r]["mu"],
                    "sigma0":kp_sim.regime_params[r]["sigma0"],
                    "delta1":kp_sim.delta1_regime[r],
                    "delta2":kp_sim.delta2_regime[r],
                    "delta3":kp_sim.delta3_regime[r],
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
        "gap_wmcr_all_acceptable": bool(gap_wmcr["all_acceptable"]),
        "kp_wmcr_all_acceptable":  bool(kp_wmcr["all_acceptable"]),
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
        "gap_wmcr":      gap_wmcr,
        "kp_wmcr":       kp_wmcr,
        "params_out":    params_out,
    }


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
            "gap_wmcr_pass":      p["gap_wmcr_all_acceptable"],
            # KP 임계값
            "kp_threshold":       p["kp_params"]["threshold"],
            # KP 레짐별 kappa (레짐0)
            "kp_kappa_r0":        p["kp_params"]["regime_params"]["0"]["kappa"],
            "kp_kappa_r1":        p["kp_params"]["regime_params"]["1"]["kappa"],
            "kp_kappa_r2":        p["kp_params"]["regime_params"]["2"]["kappa"],
            # KP 검증
            "kp_wmcr_price":      p["kp_metrics"].get("wmcr_price"),
            "kp_dtw_price":       p["kp_metrics"].get("dtw_price"),
            "kp_pmc":             p["kp_metrics"].get("pmc"),
            "kp_pit_ks_p":        p["kp_pit_ks_pvalue"],
            "kp_wmcr_pass":       p["kp_wmcr_all_acceptable"],
        })
    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    df[numeric_cols] = df[numeric_cols].round(6)
    return df


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

    # ── Step 1: 파이프라인 실행 ────────────────────────────────
    print("=" * 60)
    print("  [Step 1] 전처리 파이프라인 실행")
    print("=" * 60)
    pipeline_out = run_pipeline(hmm_n_init=10, hmm_B=1000, save_plots=False,
                                force_refit=args.force_refit)
    hmm_results  = from_pipeline_results(pipeline_out)

    # ── Step 2: 시나리오 로드 ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 2] 시나리오 로드")
    print("=" * 60)
    scenario_csv = Path(args.scenario_csv)
    if not scenario_csv.exists():
        raise FileNotFoundError(
            f"시나리오 CSV 없음: {scenario_csv}\n"
            "analysis/scenario_selection.py 를 먼저 실행하세요."
        )
    all_scenarios = scenarios_from_csv(scenario_csv, hmm_results)

    # 특정 시나리오만 실행하는 경우 필터링
    if args.scenarios:
        selected = set(args.scenarios.split(","))
        all_scenarios = {k: v for k, v in all_scenarios.items() if k in selected}
        print(f"  선택 실행: {list(all_scenarios.keys())}")

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

    # ── Step 4: 비교 요약 저장 ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  [Step 4] 시나리오 간 비교 요약")
    print("=" * 60)
    comparison_df = build_comparison_table(all_results)
    comparison_path = out_base / "comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    print(comparison_df[[
        "Scenario_ID",
        "gap_kappa", "gap_mu", "gap_sigma0",
        "gap_wmcr_price", "gap_pit_ks_p", "gap_wmcr_pass",
        "kp_threshold",
        "kp_wmcr_price", "kp_pit_ks_p", "kp_wmcr_pass",
    ]].to_string(index=False))
    print(f"\n  비교 요약 저장: {comparison_path}")
    print(f"  시나리오별 결과: {out_base}/<Scenario_ID>/")

    return all_results


if __name__ == "__main__":
    main()
