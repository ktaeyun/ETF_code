"""
NAV, GAP, KP 시뮬레이터 실행 스크립트
- NAV: ARIMAX-GARCH-t (Hash Rate, Unique Addresses)
- GAP: OU with exog (Search Interest, VIX Volatility)
- KP: Threshold-OU (volume_btc, KOSPI_Volatility)
- 결합: NAV*(1+GAP)*(1+KP) → 한국형 비트코인 ETF 가격 (초기 10,000원 앵커)
- 몬테카를로 시뮬레이션 → 검증 + 시각화
"""

import hashlib
import json
import sys
import numpy as np
from pathlib import Path

# 프로젝트 루트를 path에 추가 (simulator, compare 패키지 인식)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import pandas as pd

from simulator.data_loader import load_nav_exog_and_returns, load_gap_exog, load_kp_exog
from simulator.arima_garch_t_nav_simulator import (
    fit_arimax_garch_t,
    log_returns_to_nav,
    EXOG_COLS,
)
from simulator.gap_ou_simulator import fit_gap_ou
from simulator.kp_threshold_ou_simulator import fit_kp_threshold_ou
from simulator.visualizer import create_all_visualizations
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


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "missing"


def _sim_cache_key(base_dir: str, ar_order: tuple, S0: float, n_simulations: int, seed: int) -> str:
    data_files = [
        Path(base_dir) / "dataset" / "train" / "nav_train.csv",
        Path(base_dir) / "dataset" / "train" / "gap_train_main.csv",
        Path(base_dir) / "dataset" / "train" / "kp_train_main.csv",
        Path(base_dir) / "dataset" / "raw" / "y_variables.csv",
        Path(base_dir) / "dataset" / "raw" / "y_true_variables.csv",
    ]
    h = hashlib.md5()
    for p in data_files:
        h.update(_file_md5(str(p)).encode())
    h.update(json.dumps({
        "ar_order": list(ar_order), "S0": S0,
        "n_simulations": n_simulations, "seed": seed,
    }, sort_keys=True).encode())
    return h.hexdigest()


def _save_sim_cache(cache_dir: Path, key: str, arrays: dict, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_dir / "sim_arrays.npz"), **arrays)
    payload = {"cache_key": key}
    payload.update(_to_serializable(meta))
    with open(cache_dir / "sim_meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  [Cache] 저장 완료: {cache_dir / 'sim_arrays.npz'}")


def _load_sim_cache(cache_dir: Path, key: str):
    meta_path = cache_dir / "sim_meta.json"
    npz_path = cache_dir / "sim_arrays.npz"
    if not meta_path.exists() or not npz_path.exists():
        return None, None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("cache_key") != key:
        return None, None
    arrs = dict(np.load(str(npz_path)))
    return arrs, meta


def _print_coverage(m):
    """예측구간 커버리지 출력. 판정은 명목수준이 아니라 모형 내재 기준분포의 백분위로 한다."""
    for label, key, pct_key in [
        ("PICP50 ", "picp50_price", "picp50_pct_price"),
        ("PICP95 ", "picp95_price", "picp95_pct_price"),
        ("CovErr ", "coverage_error_price", "coverage_error_pct_price"),
    ]:
        val, pct = m.get(key), m.get(pct_key)
        if not isinstance(val, (int, float)):
            continue
        if isinstance(pct, (int, float)) and np.isfinite(pct):
            verdict = "정상" if 5.0 <= pct <= 95.0 else "이상"
            print(f"  {label}: {val:7.4f}   모형 기준분포의 {pct:5.1f} 백분위  [{verdict}]")
        else:
            print(f"  {label}: {val:7.4f}")
    for label, key in [("NMPIW95", "nmpiw95_price"), ("PICP95 Vol", "picp95_vol"),
                       ("CovErr Vol", "coverage_error_vol")]:
        val = m.get(key)
        if isinstance(val, (int, float)):
            print(f"  {label}: {val:7.4f}")


def main():
    parser = argparse.ArgumentParser(description="NAV ARIMAX-GARCH-t 시뮬레이터 (검증·시각화 포함)")
    parser.add_argument("--base-dir", type=str, default=None, help="프로젝트 루트")
    parser.add_argument("--ar-order", type=int, nargs=3, default=[1, 0, 1], metavar=("p", "d", "q"), help="ARIMA 차수")
    parser.add_argument("--S0", type=float, default=100.0, help="초기 NAV (실제·시뮬 공통)")
    parser.add_argument("--n-simulations", type=int, default=1000, help="몬테카를로 시뮬레이션 횟수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--out-dir", type=str, default=None, help="결과 저장 디렉터리 (기본: results/simulator)")
    parser.add_argument("--no-save", action="store_true", help="파일 저장 안 함")
    parser.add_argument("--no-cache", action="store_true", help="캐시 사용 안 함 (강제 재실행)")
    args = parser.parse_args()

    base_dir = args.base_dir or str(_root)
    out_dir = Path(args.out_dir or Path(base_dir) / "results" / "simulator")
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "plots" / "nav").mkdir(parents=True, exist_ok=True)
        (out_dir / "plots" / "gap").mkdir(parents=True, exist_ok=True)
        (out_dir / "plots" / "kp").mkdir(parents=True, exist_ok=True)
        (out_dir / "plots" / "combined").mkdir(parents=True, exist_ok=True)

    ar_order = tuple(args.ar_order)
    S0 = args.S0
    cache_dir = out_dir / "cache"

    # 1) 데이터 로드
    print("\n[1단계] 데이터 로드")
    df = load_nav_exog_and_returns(base_dir=base_dir)
    log_returns = df["Log Return"]
    exog = None
    T = len(log_returns)
    actual_returns = np.asarray(log_returns).flatten()
    actual_nav = np.asarray(log_returns_to_nav(pd.Series(actual_returns), S0=S0)).flatten()

    # 캐시 체크
    cache_key = _sim_cache_key(base_dir, ar_order, S0, args.n_simulations, args.seed)
    _skip_models = False
    cached_arrs, cached_meta = None, None
    if not args.no_cache:
        cached_arrs, cached_meta = _load_sim_cache(cache_dir, cache_key)
        if cached_arrs is not None:
            print("\n[Cache HIT] 저장된 시뮬레이션 결과 로드 → 모델/시뮬레이션 건너뜀")
            _skip_models = True

    if _skip_models:
        monte_carlo_returns_array = cached_arrs["mc_nav_ret"]
        monte_carlo_nav_array = cached_arrs["mc_nav"]
    else:
        # 2) ARIMAX-GARCH-t 적합
        print("\n[2단계] ARIMAX-GARCH-t 적합")
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

    # 5) 검증 지표 (예측구간 커버리지, DTW, PMC, RVR, VVS, VPR, VJC, TWAD, KS, VaR/ES 등)
    print("\n[5단계] 검증 지표 계산 (예측구간 커버리지 등)")
    validation_metrics = calculate_all_metrics(
        actual_nav=actual_nav,
        simulated_nav=representative_nav,
        actual_returns=actual_returns,
        simulated_returns=representative_returns,
        monte_carlo_nav_paths=monte_carlo_nav_array,
        monte_carlo_returns_paths=monte_carlo_returns_array,
    )
    v = validation_metrics
    _print_coverage(v)

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
        module="nav",
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
    si_series  = df_gap["value"]
    vix_series = df_gap["btc_volatility"]
    T_gap = len(gap_series)
    actual_gap = np.asarray(gap_series).flatten()
    g0 = actual_gap[0] if len(actual_gap) > 0 else 0.0

    if _skip_models:
        monte_carlo_gap_array = cached_arrs["mc_gap"]
        gap_params_dict = (cached_meta or {}).get("gap_params", {})
        print("\n[GAP-2,3단계] 캐시 로드 (모델 생략)")
        if gap_params_dict:
            print(f"  κ={gap_params_dict.get('kappa','N/A')}  μ={gap_params_dict.get('mu','N/A')}"
                  f"  σ0={gap_params_dict.get('sigma0','N/A')}")
            print(f"  δ1={gap_params_dict.get('delta1','N/A')}  δ2={gap_params_dict.get('delta2','N/A')}")
    else:
        # GAP-2) OU 모델 적합 (μ_t = μ_0 + γ1·SI,  σ_t = σ_0·exp(δ1·RV))
        print("\n[GAP-2단계] OU 모델 적합 (μ_t: SI 연동, σ_t: RV 연동)")
        gap_sim = fit_gap_ou(
            gap_series=gap_series,
            si_series=si_series,
            vix_series=vix_series,
            clip=3.0,
            regularization=0.01,
        )
        print(f"  κ={gap_sim.kappa:.6f}  μ={gap_sim.mu:.6f}  σ0={gap_sim.sigma0:.6f}")
        print(f"  δ1={gap_sim.delta1:.6f}  δ2={gap_sim.delta2:.6f}")
        gap_params_dict = {
            "kappa":  gap_sim.kappa,
            "mu":     gap_sim.mu,
            "sigma0": gap_sim.sigma0,
            "delta1": gap_sim.delta1,
            "delta2": gap_sim.delta2,
        }

        # GAP-3) 몬테카를로 시뮬레이션
        print(f"\n[GAP-3단계] 몬테카를로 시뮬레이션 (n={args.n_simulations}, T={T_gap})")
        all_gap = []
        for i in range(args.n_simulations):
            sim_gap = gap_sim.simulate_gap(
                T=T_gap,
                g0=g0,
                si_future=si_series.values,
                vix_future=vix_series.values,
                seed=args.seed + 10000 + i,
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
    
    # GAP-5) 검증 지표 및 예측구간 커버리지
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
    _print_coverage(v_gap)
    
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
        module="gap",
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
    kp_series         = df_kp["Kimchi Premium"]
    volume_btc_series = df_kp["volume_btc"]
    kospi_vol_series  = df_kp["KOSPI_Volatility"]
    bitcoin_kr_series = df_kp["bitcoin_kr"]
    T_kp = len(kp_series)
    actual_kp = np.asarray(kp_series).flatten()
    kp0 = actual_kp[0] if len(actual_kp) > 0 else 0.0

    if _skip_models:
        monte_carlo_kp_array = cached_arrs["mc_kp"]
        kp_params_dict = (cached_meta or {}).get("kp_params", {})
        print("\n[KP-2,3단계] 캐시 로드 (모델 생략)")
        if kp_params_dict:
            print(f"  최적 임계값 τ: {kp_params_dict.get('threshold', 'N/A')}")
            rp = kp_params_dict.get("regime_params", {})
            for r in [0, 1, 2]:
                regime_name = ["|KP| ≤ τ", "KP > τ", "KP < -τ"][r]
                rd = rp.get(str(r), {})
                print(f"    레짐 {r} ({regime_name}): "
                      f"κ={rd.get('kappa', 'N/A')}  μ={rd.get('mu', 'N/A')}")
    else:
        # KP-2) Threshold-OU 모델 적합
        print("\n[KP-2단계] Threshold-OU 모델 적합 (외생변수: volume_btc, KOSPI_Volatility, bitcoin_kr)")
        kp_sim = fit_kp_threshold_ou(
            kp_series=kp_series,
            volume_btc=volume_btc_series,
            kospi_vol=kospi_vol_series,
            bitcoin_kr=bitcoin_kr_series,
            threshold=None,
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
            print(f"      δ3_{r} (bitcoin_kr): {kp_sim.delta3_regime[r]:.6f}")
        kp_params_dict = {
            "threshold": kp_sim.threshold,
            "regime_params": {
                str(r): {
                    "kappa":  kp_sim.regime_params[r]['kappa'],
                    "mu":     kp_sim.regime_params[r]['mu'],
                    "sigma0": kp_sim.regime_params[r]['sigma0'],
                    "delta1": kp_sim.delta1_regime[r],
                    "delta2": kp_sim.delta2_regime[r],
                    "delta3": kp_sim.delta3_regime[r],
                }
                for r in [0, 1, 2]
            },
        }

        # KP-3) 몬테카를로 시뮬레이션
        print(f"\n[KP-3단계] 몬테카를로 시뮬레이션 (n={args.n_simulations}, T={T_kp})")
        all_kp = []
        for i in range(args.n_simulations):
            sim_kp = kp_sim.simulate_kp(
                T=T_kp,
                kp0=kp0,
                volume_btc_future=volume_btc_series.values,
                kospi_vol_future=kospi_vol_series.values,
                bitcoin_kr_future=bitcoin_kr_series.values,
                seed=args.seed + 20000 + i
            )
            all_kp.append(np.asarray(sim_kp).flatten())
        monte_carlo_kp_array = np.array(all_kp)

        # 캐시 저장 (최초 실행 시 모든 MC 완료 후)
        if not args.no_cache:
            _save_sim_cache(cache_dir, cache_key, {
                "mc_nav_ret": monte_carlo_returns_array,
                "mc_nav": monte_carlo_nav_array,
                "mc_gap": monte_carlo_gap_array,
                "mc_kp": monte_carlo_kp_array,
            }, {
                "gap_params": gap_params_dict,
                "kp_params": kp_params_dict,
            })

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
    
    # KP-5) 검증 지표 및 예측구간 커버리지
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
    _print_coverage(v_kp)
    
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
        module="kp",
    )

    # ============================================================================
    # [Step 5] 시뮬레이터 결합 + [Step 6] 통합 시뮬레이터 검증
    #   ETF_KR(t) = NAV(t) * (1+GAP(t)) * (1+KP(t))
    # ============================================================================
    print("\n" + "=" * 80)
    print("[Step 5-6] NAV*(1+GAP)*(1+KP) 통합 결합 및 검증")
    print("=" * 80)

    min_T = min(T, T_gap, T_kp)
    print(f"\n[Step 5] 세 컴포넌트 결합 (T={min_T})")

    # 실제 대응물: 한국형 ETF는 아직 존재하지 않으므로
    # 실제 NAV x (1+실제 GAP) x (1+실제 KP)로 구성한 대리 시계열을 기준으로 삼는다.
    actual_nav_c = actual_nav[:min_T]
    actual_gap_c = actual_gap[:min_T]
    actual_kp_c = actual_kp[:min_T]
    actual_combined = actual_nav_c * (1.0 + actual_gap_c) * (1.0 + actual_kp_c)

    # 몬테카를로 결합: 경로별로 세 컴포넌트를 곱한다
    monte_carlo_combined_array = np.zeros((args.n_simulations, min_T))
    for i in range(args.n_simulations):
        monte_carlo_combined_array[i] = (
            monte_carlo_nav_array[i, :min_T]
            * (1.0 + monte_carlo_gap_array[i, :min_T])
            * (1.0 + monte_carlo_kp_array[i, :min_T])
        )
    representative_combined = np.median(monte_carlo_combined_array, axis=0)

    print(f"  실제 결합 초기값: {actual_combined[0]:.4f}, 종료값: {actual_combined[-1]:.4f}")
    print(f"  시뮬 중앙값 초기: {representative_combined[0]:.4f}, 종료: {representative_combined[-1]:.4f}")

    # ------------------------------------------------------------------
    # [Step 6] 통합 검증 — 결합 가격의 로그수익률 기준
    #
    # NAV 기준 상대수익률 (ETF/NAV - 1)을 쓰면 대수적으로 GAP과 같아져
    # (nav*(1+gap)/nav - 1 = gap) NAV·KP 사양 변화에 반응하지 않는다.
    # 결합 가격 자체의 로그수익률을 쓰면
    #   d log ETF = d log NAV + d log(1+GAP) + d log(1+KP)
    # 이므로 세 컴포넌트가 모두 검정에 반영된다.
    # ------------------------------------------------------------------
    print("\n[Step 6] 통계적 검정 (결합 가격 로그수익률 기준)")

    actual_combined_returns = np.diff(np.log(actual_combined))
    simulated_combined_returns_array = np.diff(np.log(monte_carlo_combined_array), axis=1)
    representative_combined_returns = np.diff(np.log(representative_combined))

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

    # 보조 진단: 예측구간 커버리지
    print("\n[Step 6-보조] 검증 지표 계산 (예측구간 커버리지 등)")
    combined_validation_metrics = calculate_all_metrics(
        actual_nav=actual_combined,
        simulated_nav=representative_combined,
        actual_returns=actual_combined_returns,
        simulated_returns=representative_combined_returns,
        monte_carlo_nav_paths=monte_carlo_combined_array,
        monte_carlo_returns_paths=simulated_combined_returns_array,
    )
    v_combined = combined_validation_metrics
    _print_coverage(v_combined)

    # 시각화
    print("\n[Step 6-보조] 시각화")
    create_all_visualizations(
        actual_nav=actual_combined,
        simulated_nav=representative_combined,
        actual_returns=actual_combined_returns,
        simulated_returns=representative_combined_returns,
        output_dir=str(out_dir / "plots" / "combined"),
        monte_carlo_nav_paths=monte_carlo_combined_array,
        monte_carlo_returns_paths=simulated_combined_returns_array,
        module="combined",
    )

    # ============================================================================
    # [Step 7] KR Spot Bitcoin ETF 가격 경로 생성 (초기 10,000원 앵커링)
    #   Step 5에서 만든 결합 경로를 앵커링만 한다 (재계산하지 않음)
    # ============================================================================
    print("\n" + "=" * 80)
    print("[Step 7] KR Spot Bitcoin ETF 가격 경로 (10,000원 앵커)")
    print("=" * 80)
    ANCHOR_KRW = 10000.0
    min_T_korean = min_T

    def _anchor(path):
        return path * (ANCHOR_KRW / path[0]) if path[0] != 0 else path

    korean_etf_actual = _anchor(actual_combined)
    monte_carlo_korean_etf = np.vstack([_anchor(monte_carlo_combined_array[i])
                                        for i in range(args.n_simulations)])
    korean_etf_representative = np.median(monte_carlo_korean_etf, axis=0)
    korean_etf_p5 = np.percentile(monte_carlo_korean_etf, 5, axis=0)
    korean_etf_p95 = np.percentile(monte_carlo_korean_etf, 95, axis=0)

    print(f"  기간: {min_T_korean}일, 초기 가격(앵커): {ANCHOR_KRW:,.0f}원")
    print(f"  실제 한국형 ETF 초기: {korean_etf_actual[0]:,.2f}원, 종료: {korean_etf_actual[-1]:,.2f}원")
    print(f"  시뮬 중앙값 종료: {korean_etf_representative[-1]:,.2f}원")

    # 날짜 (NAV/공통 구간 기준)
    if min_T_korean <= len(df_gap):
        dates_korean = df_gap["Date"].iloc[:min_T_korean].values
    else:
        dates_korean = np.arange(min_T_korean, dtype=object)

    korean_etf_df = pd.DataFrame({
        "Date": dates_korean,
        "actual_etf_krw": korean_etf_actual,
        "simulated_median_krw": korean_etf_representative,
        "simulated_p5_krw": korean_etf_p5,
        "simulated_p95_krw": korean_etf_p95,
    })
    if not args.no_save:
        korean_etf_df.to_csv(out_dir / "korean_etf_price.csv", index=False, encoding="utf-8-sig")
        print(f"  저장: {out_dir / 'korean_etf_price.csv'}")

    # 7) 결과 저장
    validation_results = {
        "nav": {
            "statistical_tests": statistical_tests,
            "validation_metrics": validation_metrics,
            "T": T,
            "S0": S0,
        },
        "gap": {
            "statistical_tests": gap_statistical_tests,
            "validation_metrics": gap_validation_metrics,
            "ou_params": {
                "kappa":  gap_params_dict.get("kappa"),
                "mu":     gap_params_dict.get("mu"),
                "sigma0": gap_params_dict.get("sigma0"),
                "delta1": gap_params_dict.get("delta1"),
                "delta2": gap_params_dict.get("delta2"),
            },
            "T": T_gap,
        },
        "kp": {
            "statistical_tests": kp_statistical_tests,
            "validation_metrics": kp_validation_metrics,
            "threshold_ou_params": {
                "threshold": kp_params_dict.get("threshold"),
                "regime_params": {
                    r: kp_params_dict.get("regime_params", {}).get(str(r), {})
                    for r in [0, 1, 2]
                },
            },
            "T": T_kp,
        },
        "combined": {
            "statistical_tests": combined_statistical_tests,
            "validation_metrics": combined_validation_metrics,
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
        
        # [Step 5-7] 통합 결합 결과 저장
        # 로그수익률은 차분이라 길이가 min_T-1 이므로 앞에 0을 채워 가격 계열과 맞춘다
        combined_results_df = pd.DataFrame({
            "actual_combined": actual_combined,
            "simulated_combined": representative_combined,
            "actual_combined_returns": np.concatenate([[0], actual_combined_returns]),
            "simulated_combined_returns": np.concatenate([[0], representative_combined_returns]),
        })
        combined_results_df.to_csv(out_dir / "combined_simulation_results.csv", index=False, encoding="utf-8-sig")
        
        print(f"\n저장:")
        print(f"  - {out_dir / 'validation_results.json'}")
        print(f"  - {out_dir / 'nav_simulation_results.csv'}")
        print(f"  - {out_dir / 'gap_simulation_results.csv'}")
        print(f"  - {out_dir / 'kp_simulation_results.csv'}")
        print(f"  - {out_dir / 'combined_simulation_results.csv'}")
        print(f"  - {out_dir / 'korean_etf_price.csv'}")
        print(f"  - {out_dir / 'plots'}")

    return {
        "nav": {
            "simulator": sim if not _skip_models else None,
            "statistical_tests": statistical_tests,
            "validation_metrics": validation_metrics,
            "representative_returns": representative_returns,
            "representative_nav": representative_nav,
            "actual_nav": actual_nav,
            "actual_returns": actual_returns,
            "S0": S0,
            "T": T,
        },
        "gap": {
            "simulator": gap_sim if not _skip_models else None,
            "statistical_tests": gap_statistical_tests,
            "validation_metrics": gap_validation_metrics,
            "representative_gap": representative_gap,
            "actual_gap": actual_gap,
            "T": T_gap,
        },
        "kp": {
            "simulator": kp_sim if not _skip_models else None,
            "statistical_tests": kp_statistical_tests,
            "validation_metrics": kp_validation_metrics,
            "representative_kp": representative_kp,
            "actual_kp": actual_kp,
            "T": T_kp,
        },
        "combined": {
            "statistical_tests": combined_statistical_tests,
            "validation_metrics": combined_validation_metrics,
            "representative_combined": representative_combined,
            "actual_combined": actual_combined,
            "T": min_T,
        },
        "korean_etf": {
            "actual_etf_krw": korean_etf_actual,
            "representative_etf_krw": korean_etf_representative,
            "korean_etf_df": korean_etf_df,
            "anchor_krw": ANCHOR_KRW,
            "T": min_T_korean,
        },
    }


if __name__ == "__main__":
    main()
