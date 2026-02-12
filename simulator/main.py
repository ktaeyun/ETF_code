"""
NAV 및 GAP 시뮬레이터 실행 스크립트
- NAV: ARIMAX-GARCH-t (Hash Rate, Unique Addresses)
- GAP: OU with exog (Search Interest, VIX Volatility)
- 몬테카를로 시뮬레이션 → 검증(통계검정 + WMCR 등) + 시각화
"""

import sys
import json
import numpy as np
from pathlib import Path

# 프로젝트 루트를 path에 추가 (simulator, compare 패키지 인식)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import pandas as pd

from simulator.data_loader import load_nav_exog_and_returns, load_gap_exog, load_kp_exog, load_etf_true
from simulator.arima_garch_t_nav_simulator import (
    fit_arimax_garch_t,
    log_returns_to_nav,
    EXOG_COLS,
)
from simulator.gap_ou_simulator import fit_gap_ou
from simulator.kp_threshold_ou_simulator import fit_kp_threshold_ou
from simulator.visualizer import create_all_visualizations
from simulator.wmcr_test import wmcr_pvalue_calibration, compute_wmcr, wmcr_binomial_test
from compare.metrics import calculate_statistical_tests, calculate_all_metrics


def _to_serializable(obj):
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.int64, np.int32, np.float64, np.float32)):
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


def main():
    parser = argparse.ArgumentParser(description="NAV ARIMAX-GARCH-t 시뮬레이터 (검증·시각화 포함)")
    parser.add_argument("--base-dir", type=str, default=None, help="프로젝트 루트")
    parser.add_argument("--ar-order", type=int, nargs=3, default=[1, 0, 1], metavar=("p", "d", "q"), help="ARIMA 차수")
    parser.add_argument("--S0", type=float, default=100.0, help="초기 NAV (실제·시뮬 공통)")
    parser.add_argument("--n-simulations", type=int, default=1000, help="몬테카를로 시뮬레이션 횟수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--out-dir", type=str, default=None, help="결과 저장 디렉터리 (기본: results/simulator)")
    parser.add_argument("--no-save", action="store_true", help="파일 저장 안 함")
    parser.add_argument("--wmcr-alpha", type=float, default=0.05, help="WMCR 검정 유의수준")
    args = parser.parse_args()

    base_dir = args.base_dir or str(_root)
    out_dir = Path(args.out_dir or Path(base_dir) / "results" / "simulator")
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "plots" / "nav").mkdir(parents=True, exist_ok=True)
        (out_dir / "plots" / "gap").mkdir(parents=True, exist_ok=True)
        (out_dir / "plots" / "kp").mkdir(parents=True, exist_ok=True)
        (out_dir / "plots" / "combined").mkdir(parents=True, exist_ok=True)

    # 1) 데이터 로드
    print("\n[1단계] 데이터 로드")
    df = load_nav_exog_and_returns(base_dir=base_dir)
    log_returns = df["Log Return"]
    exog = df[EXOG_COLS]
    T = len(log_returns)
    actual_returns = np.asarray(log_returns).flatten()
    S0 = args.S0
    actual_nav = np.asarray(log_returns_to_nav(pd.Series(actual_returns), S0=S0)).flatten()

    # 2) ARIMAX-GARCH-t 적합
    print("\n[2단계] ARIMAX-GARCH-t 적합")
    ar_order = tuple(args.ar_order)
    sim = fit_arimax_garch_t(log_returns=log_returns, exog=exog, ar_order=ar_order)

    # 3) 몬테카를로 시뮬레이션 (실제 기간 T와 동일)
    print(f"\n[3단계] 몬테카를로 시뮬레이션 (n={args.n_simulations}, T={T})")
    all_returns = []
    all_nav = []
    for i in range(args.n_simulations):
        sim_ret = sim.simulate_returns(T=T, exog_future=None, seed=args.seed + i)
        ret_arr = np.asarray(sim_ret).flatten()
        nav_arr = np.asarray(log_returns_to_nav(pd.Series(ret_arr), S0=S0)).flatten()
        all_returns.append(ret_arr)
        all_nav.append(nav_arr)
    monte_carlo_returns_array = np.array(all_returns)
    monte_carlo_nav_array = np.array(all_nav)
    representative_returns = np.median(monte_carlo_returns_array, axis=0)
    representative_nav = np.median(monte_carlo_nav_array, axis=0)

    # 4) 통계적 검정 (PIT-KS, VaR-Kupiec, ES)
    print("\n[4단계] 통계적 검정 (수익률 기준)")
    statistical_tests = calculate_statistical_tests(
        actual_returns=actual_returns,
        simulated_returns_paths=monte_carlo_returns_array,
        alpha=0.05,
    )
    print(f"  PIT-KS: statistic={statistical_tests['pit_ks']['ks_statistic']:.4f}, p-value={statistical_tests['pit_ks']['ks_pvalue']:.4f}")
    print(f"  VaR-Kupiec: LR_uc={statistical_tests['kupiec']['lr_uc']:.4f}, p-value={statistical_tests['kupiec']['pvalue']:.4f}, exceedance_rate={statistical_tests['kupiec']['exceedance_rate']:.4f}")
    es = statistical_tests['es']
    print(f"  ES: tail_error={es.get('tail_error')}, n_violations={es.get('n_violations')}")
    print(f"  VALID: {statistical_tests['is_valid']}")

    # 5) 검증 지표 (WMCR Price/Vol, DTW, PMC, RVR, VVS, VPR, VJC, TWAD, KS, VaR/ES 등)
    print("\n[5단계] 검증 지표 계산 (Weighted Multi-band Capture Rate 등)")
    validation_metrics = calculate_all_metrics(
        actual_nav=actual_nav,
        simulated_nav=representative_nav,
        actual_returns=actual_returns,
        simulated_returns=representative_returns,
        monte_carlo_nav_paths=monte_carlo_nav_array,
        monte_carlo_returns_paths=monte_carlo_returns_array,
    )
    v = validation_metrics
    for name, key in [("WMCR Price", "wmcr_price"), ("WMCR Vol", "wmcr_vol"), ("DTW Price", "dtw_price"), ("PMC", "pmc")]:
        val = v.get(key)
        if isinstance(val, (int, float)):
            print(f"  {name}: {val:.4f}")
        else:
            print(f"  {name}: {val}")

    # 5-2) WMCR 이항비율 근사 검정 (캘리브레이션 검정)
    print("\n[5-2단계] WMCR 이항비율 근사 검정")
    
    # NAV에 대한 WMCR 검정
    # 밴드 구성: 각 시점 t에서 몬테카를로 경로의 분위수 사용
    # 예: 50% 밴드 = [25% 분위수, 75% 분위수], 80% 밴드 = [10% 분위수, 90% 분위수]
    p_targets_nav = np.array([0.50, 0.80, 0.95])  # 목표 커버리지
    
    bands_L_nav = np.zeros((len(p_targets_nav), T))
    bands_U_nav = np.zeros((len(p_targets_nav), T))
    
    for k, p_target in enumerate(p_targets_nav):
        alpha_band = (1 - p_target) / 2
        lower_q = alpha_band * 100
        upper_q = (1 - alpha_band) * 100
        for t in range(T):
            bands_L_nav[k, t] = np.percentile(monte_carlo_nav_array[:, t], lower_q)
            bands_U_nav[k, t] = np.percentile(monte_carlo_nav_array[:, t], upper_q)
    
    # 관측 캡처율 계산
    C_obs_nav = np.zeros(len(p_targets_nav))
    for k in range(len(p_targets_nav)):
        I_tk = (bands_L_nav[k, :] <= actual_nav) & (actual_nav <= bands_U_nav[k, :])
        C_obs_nav[k] = np.mean(I_tk)
    
    # 이항비율 근사 검정
    print("\n[NAV 검정]")
    wmcr_test_nav = wmcr_binomial_test(
        T=T,
        p_targets=p_targets_nav,
        C_obs=C_obs_nav,
        alpha=0.05,
        verbose=True
    )
    
    # 수익률에 대한 WMCR 검정
    p_targets_ret = np.array([0.50, 0.80, 0.95])
    
    bands_L_ret = np.zeros((len(p_targets_ret), T))
    bands_U_ret = np.zeros((len(p_targets_ret), T))
    
    for k, p_target in enumerate(p_targets_ret):
        alpha_band = (1 - p_target) / 2
        lower_q = alpha_band * 100
        upper_q = (1 - alpha_band) * 100
        for t in range(T):
            bands_L_ret[k, t] = np.percentile(monte_carlo_returns_array[:, t], lower_q)
            bands_U_ret[k, t] = np.percentile(monte_carlo_returns_array[:, t], upper_q)
    
    # 관측 캡처율 계산
    C_obs_ret = np.zeros(len(p_targets_ret))
    for k in range(len(p_targets_ret)):
        I_tk = (bands_L_ret[k, :] <= actual_returns) & (actual_returns <= bands_U_ret[k, :])
        C_obs_ret[k] = np.mean(I_tk)
    
    # 이항비율 근사 검정
    print("\n[Returns 검정]")
    wmcr_test_ret = wmcr_binomial_test(
        T=T,
        p_targets=p_targets_ret,
        C_obs=C_obs_ret,
        alpha=0.05,
        verbose=True
    )

    # 6) 시각화 (NAV)
    print("\n[6단계] NAV 시각화")
    create_all_visualizations(
        actual_nav=actual_nav,
        simulated_nav=representative_nav,
        actual_returns=actual_returns,
        simulated_returns=representative_returns,
        output_dir=str(out_dir / "plots" / "nav"),
        monte_carlo_nav_paths=monte_carlo_nav_array,
        monte_carlo_returns_paths=monte_carlo_returns_array,
    )

    # ============================================================================
    # GAP 시뮬레이션 (NAV와 동일한 형식)
    # ============================================================================
    print("\n" + "=" * 80)
    print("GAP OU 시뮬레이션 시작")
    print("=" * 80)
    
    # GAP-1) 데이터 로드
    print("\n[GAP-1단계] 데이터 로드")
    df_gap = load_gap_exog(base_dir=base_dir)
    gap_series = df_gap["etf_premium"]
    si_series = df_gap["Search Interest"]
    vix_series = df_gap["VIX Volatility"]
    T_gap = len(gap_series)
    actual_gap = np.asarray(gap_series).flatten()
    
    # GAP-2) OU 모델 적합
    print("\n[GAP-2단계] OU 모델 적합 (외생변수: SI, VIX)")
    gap_sim = fit_gap_ou(
        gap_series=gap_series,
        si_series=si_series,
        vix_series=vix_series,
        clip=3.0,
        regularization=0.01
    )
    print(f"  κ (kappa): {gap_sim.kappa:.6f}")
    print(f"  μ (mu): {gap_sim.mu:.6f}")
    print(f"  σ0 (sigma0): {gap_sim.sigma0:.6f}")
    print(f"  δ1 (delta1, SI): {gap_sim.delta1:.6f}")
    print(f"  δ2 (delta2, VIX): {gap_sim.delta2:.6f}")
    
    # GAP-3) 몬테카를로 시뮬레이션
    print(f"\n[GAP-3단계] 몬테카를로 시뮬레이션 (n={args.n_simulations}, T={T_gap})")
    all_gap = []
    g0 = actual_gap[0] if len(actual_gap) > 0 else gap_sim.mu
    for i in range(args.n_simulations):
        sim_gap = gap_sim.simulate_gap(
            T=T_gap,
            g0=g0,
            si_future=si_series.values,
            vix_future=vix_series.values,
            seed=args.seed + 10000 + i
        )
        all_gap.append(np.asarray(sim_gap).flatten())
    monte_carlo_gap_array = np.array(all_gap)
    representative_gap = np.median(monte_carlo_gap_array, axis=0)
    
    # GAP-4) 통계적 검정 (PIT-KS 등)
    print("\n[GAP-4단계] 통계적 검정")
    # GAP 변화량에 대해 검정
    actual_gap_changes = np.diff(actual_gap)
    simulated_gap_changes_array = np.diff(monte_carlo_gap_array, axis=1)
    representative_gap_changes = np.diff(representative_gap)
    
    gap_statistical_tests = calculate_statistical_tests(
        actual_returns=actual_gap_changes,
        simulated_returns_paths=simulated_gap_changes_array,
        alpha=0.05,
    )
    print(f"  PIT-KS: statistic={gap_statistical_tests['pit_ks']['ks_statistic']:.4f}, p-value={gap_statistical_tests['pit_ks']['ks_pvalue']:.4f}")
    print(f"  VaR-Kupiec: LR_uc={gap_statistical_tests['kupiec']['lr_uc']:.4f}, p-value={gap_statistical_tests['kupiec']['pvalue']:.4f}")
    print(f"  VALID: {gap_statistical_tests['is_valid']}")
    
    # GAP-5) 검증 지표 및 WMCR 검정
    print("\n[GAP-5단계] 검증 지표 계산")
    gap_validation_metrics = calculate_all_metrics(
        actual_nav=actual_gap,
        simulated_nav=representative_gap,
        actual_returns=actual_gap_changes,
        simulated_returns=representative_gap_changes,
        monte_carlo_nav_paths=monte_carlo_gap_array,
        monte_carlo_returns_paths=simulated_gap_changes_array,
    )
    v_gap = gap_validation_metrics
    for name, key in [("WMCR Price", "wmcr_price"), ("WMCR Vol", "wmcr_vol"), ("DTW Price", "dtw_price"), ("PMC", "pmc")]:
        val = v_gap.get(key)
        if isinstance(val, (int, float)):
            print(f"  {name}: {val:.4f}")
    
    # GAP-5-2) WMCR 이항비율 근사 검정
    print("\n[GAP-5-2단계] WMCR 이항비율 근사 검정")
    p_targets_gap = np.array([0.50, 0.80, 0.95])
    
    bands_L_gap = np.zeros((len(p_targets_gap), T_gap))
    bands_U_gap = np.zeros((len(p_targets_gap), T_gap))
    
    for k, p_target in enumerate(p_targets_gap):
        alpha_band = (1 - p_target) / 2
        lower_q = alpha_band * 100
        upper_q = (1 - alpha_band) * 100
        for t in range(T_gap):
            bands_L_gap[k, t] = np.percentile(monte_carlo_gap_array[:, t], lower_q)
            bands_U_gap[k, t] = np.percentile(monte_carlo_gap_array[:, t], upper_q)
    
    C_obs_gap = np.zeros(len(p_targets_gap))
    for k in range(len(p_targets_gap)):
        I_tk = (bands_L_gap[k, :] <= actual_gap) & (actual_gap <= bands_U_gap[k, :])
        C_obs_gap[k] = np.mean(I_tk)
    
    print("\n[GAP 검정]")
    wmcr_test_gap = wmcr_binomial_test(
        T=T_gap,
        p_targets=p_targets_gap,
        C_obs=C_obs_gap,
        alpha=0.05,
        verbose=True
    )
    
    # GAP-6) 시각화
    print("\n[GAP-6단계] 시각화")
    create_all_visualizations(
        actual_nav=actual_gap,
        simulated_nav=representative_gap,
        actual_returns=actual_gap_changes,
        simulated_returns=representative_gap_changes,
        output_dir=str(out_dir / "plots" / "gap"),
        monte_carlo_nav_paths=monte_carlo_gap_array,
        monte_carlo_returns_paths=simulated_gap_changes_array,
    )

    # ============================================================================
    # KP 시뮬레이션 (NAV/GAP와 동일한 형식)
    # ============================================================================
    print("\n" + "=" * 80)
    print("KP Threshold-OU 시뮬레이션 시작")
    print("=" * 80)
    
    # KP-1) 데이터 로드
    print("\n[KP-1단계] 데이터 로드")
    df_kp = load_kp_exog(base_dir=base_dir)
    kp_series = df_kp["Kimchi Premium"]
    volume_btc_series = df_kp["volume_btc"]
    kospi_vol_series = df_kp["KOSPI_Volatility"]
    T_kp = len(kp_series)
    actual_kp = np.asarray(kp_series).flatten()
    
    # KP-2) Threshold-OU 모델 적합
    print("\n[KP-2단계] Threshold-OU 모델 적합 (외생변수: volume_btc, KOSPI_Volatility)")
    kp_sim = fit_kp_threshold_ou(
        kp_series=kp_series,
        volume_btc=volume_btc_series,
        kospi_vol=kospi_vol_series,
        threshold=None,  # 자동 선택
        clip=3.0,
        regularization=0.01
    )
    print(f"  최적 임계값 τ: {kp_sim.threshold:.6f}")
    print(f"  레짐별 파라미터:")
    for r in [0, 1, 2]:
        regime_name = ["|KP| ≤ τ", "KP > τ", "KP < -τ"][r]
        print(f"    레짐 {r} ({regime_name}):")
        print(f"      κ_{r}: {kp_sim.regime_params[r]['kappa']:.6f}")
        print(f"      μ_{r}: {kp_sim.regime_params[r]['mu']:.6f}")
        print(f"      σ0_{r}: {kp_sim.regime_params[r]['sigma0']:.6f}")
        print(f"      δ1_{r} (volume_btc): {kp_sim.delta1_regime[r]:.6f}")
        print(f"      δ2_{r} (KOSPI_Vol): {kp_sim.delta2_regime[r]:.6f}")
    
    # KP-3) 몬테카를로 시뮬레이션
    print(f"\n[KP-3단계] 몬테카를로 시뮬레이션 (n={args.n_simulations}, T={T_kp})")
    all_kp = []
    kp0 = actual_kp[0] if len(actual_kp) > 0 else kp_sim.regime_params[0]['mu']
    for i in range(args.n_simulations):
        sim_kp = kp_sim.simulate_kp(
            T=T_kp,
            kp0=kp0,
            volume_btc_future=volume_btc_series.values,
            kospi_vol_future=kospi_vol_series.values,
            seed=args.seed + 20000 + i
        )
        all_kp.append(np.asarray(sim_kp).flatten())
    monte_carlo_kp_array = np.array(all_kp)
    representative_kp = np.median(monte_carlo_kp_array, axis=0)
    
    # KP-4) 통계적 검정 (PIT-KS 등)
    print("\n[KP-4단계] 통계적 검정")
    # KP 변화량에 대해 검정
    actual_kp_changes = np.diff(actual_kp)
    simulated_kp_changes_array = np.diff(monte_carlo_kp_array, axis=1)
    representative_kp_changes = np.diff(representative_kp)
    
    kp_statistical_tests = calculate_statistical_tests(
        actual_returns=actual_kp_changes,
        simulated_returns_paths=simulated_kp_changes_array,
        alpha=0.05,
    )
    print(f"  PIT-KS: statistic={kp_statistical_tests['pit_ks']['ks_statistic']:.4f}, p-value={kp_statistical_tests['pit_ks']['ks_pvalue']:.4f}")
    print(f"  VaR-Kupiec: LR_uc={kp_statistical_tests['kupiec']['lr_uc']:.4f}, p-value={kp_statistical_tests['kupiec']['pvalue']:.4f}")
    print(f"  VALID: {kp_statistical_tests['is_valid']}")
    
    # KP-5) 검증 지표 및 WMCR 검정
    print("\n[KP-5단계] 검증 지표 계산")
    kp_validation_metrics = calculate_all_metrics(
        actual_nav=actual_kp,
        simulated_nav=representative_kp,
        actual_returns=actual_kp_changes,
        simulated_returns=representative_kp_changes,
        monte_carlo_nav_paths=monte_carlo_kp_array,
        monte_carlo_returns_paths=simulated_kp_changes_array,
    )
    v_kp = kp_validation_metrics
    for name, key in [("WMCR Price", "wmcr_price"), ("WMCR Vol", "wmcr_vol"), ("DTW Price", "dtw_price"), ("PMC", "pmc")]:
        val = v_kp.get(key)
        if isinstance(val, (int, float)):
            print(f"  {name}: {val:.4f}")
    
    # KP-5-2) WMCR 이항비율 근사 검정
    print("\n[KP-5-2단계] WMCR 이항비율 근사 검정")
    p_targets_kp = np.array([0.50, 0.80, 0.95])
    
    bands_L_kp = np.zeros((len(p_targets_kp), T_kp))
    bands_U_kp = np.zeros((len(p_targets_kp), T_kp))
    
    for k, p_target in enumerate(p_targets_kp):
        alpha_band = (1 - p_target) / 2
        lower_q = alpha_band * 100
        upper_q = (1 - alpha_band) * 100
        for t in range(T_kp):
            bands_L_kp[k, t] = np.percentile(monte_carlo_kp_array[:, t], lower_q)
            bands_U_kp[k, t] = np.percentile(monte_carlo_kp_array[:, t], upper_q)
    
    C_obs_kp = np.zeros(len(p_targets_kp))
    for k in range(len(p_targets_kp)):
        I_tk = (bands_L_kp[k, :] <= actual_kp) & (actual_kp <= bands_U_kp[k, :])
        C_obs_kp[k] = np.mean(I_tk)
    
    print("\n[KP 검정]")
    wmcr_test_kp = wmcr_binomial_test(
        T=T_kp,
        p_targets=p_targets_kp,
        C_obs=C_obs_kp,
        alpha=0.05,
        verbose=True
    )
    
    # KP-6) 시각화
    print("\n[KP-6단계] 시각화")
    create_all_visualizations(
        actual_nav=actual_kp,
        simulated_nav=representative_kp,
        actual_returns=actual_kp_changes,
        simulated_returns=representative_kp_changes,
        output_dir=str(out_dir / "plots" / "kp"),
        monte_carlo_nav_paths=monte_carlo_kp_array,
        monte_carlo_returns_paths=simulated_kp_changes_array,
    )

    # ============================================================================
    # NAV*(1+GAP) 결합 시뮬레이션 및 검정
    # ============================================================================
    print("\n" + "=" * 80)
    print("NAV*(1+GAP) 결합 시뮬레이션 및 검정")
    print("=" * 80)
    
    # 실제 ETF 가격 및 NAV 로드 (etf_true, nav_true)
    print("\n[결합-1단계] 실제 ETF 가격 및 NAV 로드 (etf_true, nav_true)")
    y_true_df = load_etf_true(base_dir=base_dir)
    actual_etf_true = np.asarray(y_true_df["etf_true"]).flatten()
    actual_nav_true = np.asarray(y_true_df["nav_true"]).flatten()
    T_etf = len(actual_etf_true)
    
    # NAV, GAP, ETF 길이 확인 및 정렬
    min_T = min(T, T_gap, T_etf)
    print(f"  데이터 정렬 (T={min_T})")
    
    # 실제 관측값: etf_true 사용
    actual_combined = actual_etf_true[:min_T]
    actual_nav_true_aligned = actual_nav_true[:min_T]
    
    # NAV와 GAP 시뮬레이션 정렬
    actual_nav_aligned = actual_nav[:min_T]
    actual_gap_aligned = actual_gap[:min_T]
    
    # 시뮬레이션 결합값 계산 (몬테카를로)
    # 스케일 조정: nav_true의 초기값에 맞춤
    nav_true_initial = actual_nav_true_aligned[0] if len(actual_nav_true_aligned) > 0 else S0
    nav_initial = actual_nav_aligned[0] if len(actual_nav_aligned) > 0 else S0
    scale_factor = nav_true_initial / nav_initial if nav_initial != 0 else 1.0
    
    monte_carlo_combined_array = np.zeros((args.n_simulations, min_T))
    for i in range(args.n_simulations):
        nav_path = monte_carlo_nav_array[i, :min_T] * scale_factor  # 스케일 조정
        gap_path = monte_carlo_gap_array[i, :min_T]
        monte_carlo_combined_array[i] = nav_path * (1 + gap_path)
    
    representative_combined = np.median(monte_carlo_combined_array, axis=0)
    
    print(f"  실제 NAV 초기값 (nav_true): {nav_true_initial:.4f}")
    print(f"  실제 ETF 초기값 (etf_true): {actual_combined[0]:.4f}")
    print(f"  시뮬 NAV 초기값: {nav_initial:.4f}")
    print(f"  스케일 조정 계수: {scale_factor:.4f}")
    
    # 결합-2) 통계적 검정 (수익률 기준: NAV 기준 상대 수익률)
    print("\n[결합-2단계] 통계적 검정 (수익률 기준: NAV 기준 상대 수익률)")
    # 실제: (etf_true[t] - nav_true[t]) / nav_true[t] = etf_true[t] / nav_true[t] - 1
    actual_combined_returns = (actual_combined / actual_nav_true_aligned) - 1.0
    
    # 시뮬레이션: gap_sim과 동일 (nav_sim * (1 + gap_sim) - nav_sim) / nav_sim = gap_sim
    simulated_combined_returns_array = np.zeros((args.n_simulations, min_T))
    for i in range(args.n_simulations):
        nav_path = monte_carlo_nav_array[i, :min_T] * scale_factor
        gap_path = monte_carlo_gap_array[i, :min_T]
        combined_path = nav_path * (1 + gap_path)
        # NAV 기준 상대 수익률: (ETF - NAV) / NAV = gap
        simulated_combined_returns_array[i] = (combined_path / nav_path) - 1.0
    
    representative_combined_returns = (representative_combined / (actual_nav_aligned * scale_factor)) - 1.0
    
    # 통계적 검정은 수익률 시계열에 대해 수행 (길이 min_T)
    combined_statistical_tests = calculate_statistical_tests(
        actual_returns=actual_combined_returns,
        simulated_returns_paths=simulated_combined_returns_array,
        alpha=0.05,
    )
    
    print(f"  PIT-KS: statistic={combined_statistical_tests['pit_ks']['ks_statistic']:.4f}, p-value={combined_statistical_tests['pit_ks']['ks_pvalue']:.4f}")
    print(f"  VaR-Kupiec: LR_uc={combined_statistical_tests['kupiec']['lr_uc']:.4f}, p-value={combined_statistical_tests['kupiec']['pvalue']:.4f}, exceedance_rate={combined_statistical_tests['kupiec']['exceedance_rate']:.4f}")
    es_combined = combined_statistical_tests['es']
    print(f"  ES: tail_error={es_combined.get('tail_error')}, n_violations={es_combined.get('n_violations')}")
    print(f"  VALID: {combined_statistical_tests['is_valid']}")
    
    # 결합-3) 검증 지표 (가격 경로 기준)
    print("\n[결합-3단계] 검증 지표 계산 (가격 경로 기준)")
    combined_validation_metrics = calculate_all_metrics(
        actual_nav=actual_combined,
        simulated_nav=representative_combined,
        actual_returns=actual_combined_returns,
        simulated_returns=representative_combined_returns,
        monte_carlo_nav_paths=monte_carlo_combined_array,
        monte_carlo_returns_paths=simulated_combined_returns_array,
    )
    v_combined = combined_validation_metrics
    for name, key in [("WMCR Price", "wmcr_price"), ("WMCR Vol", "wmcr_vol"), ("DTW Price", "dtw_price"), ("PMC", "pmc")]:
        val = v_combined.get(key)
        if isinstance(val, (int, float)):
            print(f"  {name}: {val:.4f}")
    
    # 결합-4) WMCR 이항비율 근사 검정
    print("\n[결합-4단계] WMCR 이항비율 근사 검정")
    p_targets_combined = np.array([0.50, 0.80, 0.95])
    
    bands_L_combined = np.zeros((len(p_targets_combined), min_T))
    bands_U_combined = np.zeros((len(p_targets_combined), min_T))
    
    for k, p_target in enumerate(p_targets_combined):
        alpha_band = (1 - p_target) / 2
        lower_q = alpha_band * 100
        upper_q = (1 - alpha_band) * 100
        for t in range(min_T):
            bands_L_combined[k, t] = np.percentile(monte_carlo_combined_array[:, t], lower_q)
            bands_U_combined[k, t] = np.percentile(monte_carlo_combined_array[:, t], upper_q)
    
    C_obs_combined = np.zeros(len(p_targets_combined))
    for k in range(len(p_targets_combined)):
        I_tk = (bands_L_combined[k, :] <= actual_combined) & (actual_combined <= bands_U_combined[k, :])
        C_obs_combined[k] = np.mean(I_tk)
    
    print("\n[NAV*(1+GAP) 검정]")
    wmcr_test_combined = wmcr_binomial_test(
        T=min_T,
        p_targets=p_targets_combined,
        C_obs=C_obs_combined,
        alpha=0.05,
        verbose=True
    )
    
    # 결합-5) 시각화
    print("\n[결합-5단계] 시각화")
    create_all_visualizations(
        actual_nav=actual_combined,
        simulated_nav=representative_combined,
        actual_returns=actual_combined_returns,
        simulated_returns=representative_combined_returns,
        output_dir=str(out_dir / "plots" / "combined"),
        monte_carlo_nav_paths=monte_carlo_combined_array,
        monte_carlo_returns_paths=simulated_combined_returns_array,
    )

    # 7) 결과 저장
    validation_results = {
        "nav": {
            "statistical_tests": statistical_tests,
            "validation_metrics": validation_metrics,
            "wmcr_test_nav": {
                "summary_table": wmcr_test_nav['summary_table'].to_dict('records'),
                "n_acceptable": wmcr_test_nav['n_acceptable'],
                "n_total": wmcr_test_nav['n_total'],
                "all_acceptable": wmcr_test_nav['all_acceptable'],
                "problematic_bands": wmcr_test_nav['problematic_bands'],
                "summary_text": wmcr_test_nav['summary_text'],
            },
            "wmcr_test_returns": {
                "summary_table": wmcr_test_ret['summary_table'].to_dict('records'),
                "n_acceptable": wmcr_test_ret['n_acceptable'],
                "n_total": wmcr_test_ret['n_total'],
                "all_acceptable": wmcr_test_ret['all_acceptable'],
                "problematic_bands": wmcr_test_ret['problematic_bands'],
                "summary_text": wmcr_test_ret['summary_text'],
            },
            "T": T,
            "S0": S0,
        },
        "gap": {
            "statistical_tests": gap_statistical_tests,
            "validation_metrics": gap_validation_metrics,
            "wmcr_test": {
                "summary_table": wmcr_test_gap['summary_table'].to_dict('records'),
                "n_acceptable": wmcr_test_gap['n_acceptable'],
                "n_total": wmcr_test_gap['n_total'],
                "all_acceptable": wmcr_test_gap['all_acceptable'],
                "problematic_bands": wmcr_test_gap['problematic_bands'],
                "summary_text": wmcr_test_gap['summary_text'],
            },
            "ou_params": {
                "kappa": gap_sim.kappa,
                "mu": gap_sim.mu,
                "sigma0": gap_sim.sigma0,
                "delta1": gap_sim.delta1,
                "delta2": gap_sim.delta2,
            },
            "T": T_gap,
        },
        "kp": {
            "statistical_tests": kp_statistical_tests,
            "validation_metrics": kp_validation_metrics,
            "wmcr_test": {
                "summary_table": wmcr_test_kp['summary_table'].to_dict('records'),
                "n_acceptable": wmcr_test_kp['n_acceptable'],
                "n_total": wmcr_test_kp['n_total'],
                "all_acceptable": wmcr_test_kp['all_acceptable'],
                "problematic_bands": wmcr_test_kp['problematic_bands'],
                "summary_text": wmcr_test_kp['summary_text'],
            },
            "threshold_ou_params": {
                "threshold": kp_sim.threshold,
                "regime_params": {
                    r: {
                        "kappa": kp_sim.regime_params[r]['kappa'],
                        "mu": kp_sim.regime_params[r]['mu'],
                        "sigma0": kp_sim.regime_params[r]['sigma0'],
                        "delta1": kp_sim.delta1_regime[r],
                        "delta2": kp_sim.delta2_regime[r],
                    }
                    for r in [0, 1, 2]
                },
            },
            "T": T_kp,
        },
        "combined": {
            "statistical_tests": combined_statistical_tests,
            "validation_metrics": combined_validation_metrics,
            "wmcr_test": {
                "summary_table": wmcr_test_combined['summary_table'].to_dict('records'),
                "n_acceptable": wmcr_test_combined['n_acceptable'],
                "n_total": wmcr_test_combined['n_total'],
                "all_acceptable": wmcr_test_combined['all_acceptable'],
                "problematic_bands": wmcr_test_combined['problematic_bands'],
                "summary_text": wmcr_test_combined['summary_text'],
            },
            "T": min_T,
        },
        "n_simulations": args.n_simulations,
    }
    if not args.no_save:
        with open(out_dir / "validation_results.json", "w", encoding="utf-8") as f:
            json.dump(_to_serializable(validation_results), f, ensure_ascii=False, indent=2)
        # NAV 결과 저장
        nav_results_df = pd.DataFrame({
            "actual_returns": actual_returns,
            "actual_nav": actual_nav,
            "simulated_returns": representative_returns,
            "simulated_nav": representative_nav,
        })
        nav_results_df.to_csv(out_dir / "nav_simulation_results.csv", index=False, encoding="utf-8-sig")
        
        # GAP 결과 저장
        gap_results_df = pd.DataFrame({
            "actual_gap": actual_gap,
            "simulated_gap": representative_gap,
            "actual_gap_changes": np.concatenate([[0], actual_gap_changes]),
            "simulated_gap_changes": np.concatenate([[0], representative_gap_changes]),
        })
        gap_results_df.to_csv(out_dir / "gap_simulation_results.csv", index=False, encoding="utf-8-sig")
        
        # KP 결과 저장
        kp_results_df = pd.DataFrame({
            "actual_kp": actual_kp,
            "simulated_kp": representative_kp,
            "actual_kp_changes": np.concatenate([[0], actual_kp_changes]),
            "simulated_kp_changes": np.concatenate([[0], representative_kp_changes]),
        })
        kp_results_df.to_csv(out_dir / "kp_simulation_results.csv", index=False, encoding="utf-8-sig")
        
        # NAV*(1+GAP) 결합 결과 저장
        # actual_combined_returns와 representative_combined_returns는 이미 길이 min_T
        combined_results_df = pd.DataFrame({
            "actual_combined": actual_combined,
            "simulated_combined": representative_combined,
            "actual_combined_returns": actual_combined_returns,
            "simulated_combined_returns": representative_combined_returns,
        })
        combined_results_df.to_csv(out_dir / "combined_simulation_results.csv", index=False, encoding="utf-8-sig")
        
        print(f"\n저장:")
        print(f"  - {out_dir / 'validation_results.json'}")
        print(f"  - {out_dir / 'nav_simulation_results.csv'}")
        print(f"  - {out_dir / 'gap_simulation_results.csv'}")
        print(f"  - {out_dir / 'kp_simulation_results.csv'}")
        print(f"  - {out_dir / 'combined_simulation_results.csv'}")
        print(f"  - {out_dir / 'plots'}")

    return {
        "nav": {
            "simulator": sim,
            "statistical_tests": statistical_tests,
            "validation_metrics": validation_metrics,
            "wmcr_test_nav": wmcr_test_nav,
            "wmcr_test_returns": wmcr_test_ret,
            "representative_returns": representative_returns,
            "representative_nav": representative_nav,
            "actual_nav": actual_nav,
            "actual_returns": actual_returns,
            "S0": S0,
            "T": T,
        },
        "gap": {
            "simulator": gap_sim,
            "statistical_tests": gap_statistical_tests,
            "validation_metrics": gap_validation_metrics,
            "wmcr_test": wmcr_test_gap,
            "representative_gap": representative_gap,
            "actual_gap": actual_gap,
            "T": T_gap,
        },
        "kp": {
            "simulator": kp_sim,
            "statistical_tests": kp_statistical_tests,
            "validation_metrics": kp_validation_metrics,
            "wmcr_test": wmcr_test_kp,
            "representative_kp": representative_kp,
            "actual_kp": actual_kp,
            "T": T_kp,
        },
        "combined": {
            "statistical_tests": combined_statistical_tests,
            "validation_metrics": combined_validation_metrics,
            "wmcr_test": wmcr_test_combined,
            "representative_combined": representative_combined,
            "actual_combined": actual_combined,
            "T": min_T,
        }
    }


if __name__ == "__main__":
    main()
