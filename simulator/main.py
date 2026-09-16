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

from simulator.data_loader import (load_nav_exog_and_returns, load_gap_exog, load_kp_exog,
                                   align_to_nav_grid, NAV_SOURCE_DEFAULT, NAV_SOURCES)
from simulator.arima_garch_t_nav_simulator import (
    fit_arimax_garch_t,
    log_returns_to_nav,
    EXOG_COLS,
)
from simulator.gap_ou_simulator import fit_gap_ou
from simulator.kp_threshold_ou_simulator import fit_kp_threshold_ou
from simulator.visualizer import create_all_visualizations
from simulator.metrics import calculate_statistical_tests


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


def _sim_cache_key(base_dir: str, ar_order: tuple, S0: float, n_simulations: int, seed: int,
                   gap_dist: str = "t", kp_n_regimes: int = 2,
                   kp_dist: str = "t", nav_source: str = NAV_SOURCE_DEFAULT) -> str:
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
        # 사양이 바뀌면 경로가 달라지므로 캐시를 분리해야 한다
        "gap_dist": gap_dist,
        "kp_n_regimes": kp_n_regimes,
        "kp_dist": kp_dist,
        "nav_source": nav_source,
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


def _regime_label(r: int, n_regimes: int) -> str:
    """레짐 인덱스를 사람이 읽는 조건식으로. 2레짐과 3레짐의 레짐0 정의가 다르다."""
    if n_regimes == 2:
        return ["KP ≤ τ", "KP > τ"][r]
    return ["|KP| ≤ τ", "KP > τ", "KP < -τ"][r]


def _print_es(es):
    """ES 검정 결과 출력. tail_error 는 부호를 갖는 통계량이라 양측 검정이다."""
    te = es.get("tail_error")
    if te is None:
        print("  ES: 위반일 0건 -> tail_error 정의 불가 (판정 불가)")
        return
    p, pct = es.get("pvalue", float("nan")), es.get("percentile", float("nan"))
    verdict = "통과" if es.get("is_valid") else "기각"
    print(f"  ES: tail_error={te:+.6f}  p={p:.4f}  (기준분포 {pct:.1f} 백분위)  [{verdict}]")
    print(f"      기준분포 5/50/95 = {es.get('ref_p5', float('nan')):+.6f} / "
          f"{es.get('ref_p50', float('nan')):+.6f} / {es.get('ref_p95', float('nan')):+.6f}"
          f"   위반일 {es.get('n_violations')}건, 유효표본 {es.get('n_ref_effective')}")


def main():
    parser = argparse.ArgumentParser(description="NAV ARIMAX-GARCH-t 시뮬레이터 (검증·시각화 포함)")
    parser.add_argument("--base-dir", type=str, default=None, help="프로젝트 루트")
    parser.add_argument("--ar-order", type=int, nargs=3, default=[1, 0, 1], metavar=("p", "d", "q"), help="ARIMA 차수")
    parser.add_argument("--S0", type=float, default=None,
                        help="초기 NAV (실제·시뮬 공통). 기본: nav_true의 첫 거래일 값 "
                             "(--nav-source btc 이면 100.0)")
    parser.add_argument("--nav-source", choices=list(NAV_SOURCES), default=NAV_SOURCE_DEFAULT,
                        help="NAV 계열 정의 (기본 btc_aligned: BTC 현물 + 1거래일 시차보정. "
                             "운용보수 등 펀드 비용이 섞이지 않은 순수 기초자산 가치)")
    parser.add_argument("--n-simulations", type=int, default=1000, help="몬테카를로 시뮬레이션 횟수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--out-dir", type=str, default=None, help="결과 저장 디렉터리 (기본: results/simulator)")
    parser.add_argument("--no-save", action="store_true", help="파일 저장 안 함")
    parser.add_argument("--no-cache", action="store_true",
                        help="저장된 캐시를 읽지 않고 강제 재실행 (결과는 그대로 저장됨)")
    parser.add_argument("--gap-dist", choices=["t", "normal"], default="t",
                        help="GAP OU 혁신항 분포 (기본 t: 실제 괴리율의 두꺼운 꼬리 반영)")
    parser.add_argument("--kp-regimes", type=int, choices=[2, 3], default=2,
                        help="KP Threshold-OU 레짐 수 (기본 2: 역프리미엄 레짐 점유율 0%%)")
    parser.add_argument("--kp-dist", choices=["t", "normal"], default="t",
                        help="KP Threshold-OU 혁신항 분포 (기본 t: 정규에서는 ES 검정 기각)")
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
    nav_source = args.nav_source
    cache_dir = out_dir / "cache"

    # 1) 데이터 로드
    print("\n[1단계] 데이터 로드")
    df = load_nav_exog_and_returns(base_dir=base_dir, nav_source=nav_source)
    log_returns = df["Log Return"]
    exog = None
    T = len(log_returns)
    actual_returns = np.asarray(log_returns).flatten()

    # S0(초기 NAV) — 그리드 첫 거래일의 실제 nav_true를 앵커로 쓴다.
    # etf_true = nav_true x (1+GAP)이므로, 이 앵커에서 출발한 결합 경로
    # NAV x (1+GAP)이 실측 etf_true와 같은 수준(달러)에서 시작한다.
    # 그래야 Step 6-A에서 수익률뿐 아니라 가격 수준도 비교할 수 있다.
    S0 = args.S0 if args.S0 is not None else float(df["nav_true"].iloc[0])
    print(f"  NAV 계열: {nav_source}   T={T}   "
          f"기간 {df['Date'].iloc[0].date()} ~ {df['Date'].iloc[-1].date()}   S0={S0}")

    actual_nav = np.asarray(log_returns_to_nav(pd.Series(actual_returns), S0=S0)).flatten()
    # 실측 미국 ETF 가격 — 결합 검증(Step 6-A)의 기준 계열
    actual_etf_true = np.asarray(df["etf_true"], dtype=float)

    # 캐시 체크
    cache_key = _sim_cache_key(base_dir, ar_order, S0, args.n_simulations, args.seed,
                               args.gap_dist, args.kp_regimes, args.kp_dist, nav_source)
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
    _print_es(statistical_tests['es'])
    print(f"  VALID: {statistical_tests['is_valid']}")

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
    df_gap = align_to_nav_grid(df, load_gap_exog(base_dir=base_dir))
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
            dist=args.gap_dist,
        )
        print(f"  κ={gap_sim.kappa:.6f}  μ={gap_sim.mu:.6f}  σ0={gap_sim.sigma0:.6f}")
        print(f"  δ1={gap_sim.delta1:.6f}  δ2={gap_sim.delta2:.6f}")
        if gap_sim.nu is not None:
            print(f"  ν={gap_sim.nu:.4f}  (Student-t 혁신항, 초과첨도 {6/(gap_sim.nu-4):.2f})"
                  if gap_sim.nu > 4 else f"  ν={gap_sim.nu:.4f}  (Student-t 혁신항, 첨도 무한)")
        else:
            print("  혁신항: 정규분포")
        gap_params_dict = {
            "kappa":  gap_sim.kappa,
            "mu":     gap_sim.mu,
            "sigma0": gap_sim.sigma0,
            "delta1": gap_sim.delta1,
            "delta2": gap_sim.delta2,
            "nu":     gap_sim.nu,
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
    _print_es(gap_statistical_tests['es'])
    print(f"  VALID: {gap_statistical_tests['is_valid']}")
    
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
    df_kp = align_to_nav_grid(df, load_kp_exog(base_dir=base_dir))
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
            _n_reg = int(kp_params_dict.get("n_regimes", len(rp)) or len(rp))
            _kp_nu = kp_params_dict.get("nu")
            print(f"  ν: {_kp_nu:.4f}" if isinstance(_kp_nu, (int, float))
                  else "  (정규 혁신항)")
            for r in range(_n_reg):
                regime_name = _regime_label(r, _n_reg)
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
            regularization=0.01,
            n_regimes=args.kp_regimes,
            dist=args.kp_dist,
        )
        print(f"  최적 임계값 τ: {kp_sim.threshold:.6f}")
        print(f"  레짐 수: {kp_sim.n_regimes}")
        if kp_sim.nu is not None:
            print(f"  ν={kp_sim.nu:.4f}  (레짐 공통 Student-t 혁신항, 초과첨도 "
                  + (f"{6/(kp_sim.nu-4):.2f})" if kp_sim.nu > 4 else "무한)"))
        else:
            print("  (정규 혁신항)")
        print(f"  레짐별 파라미터:")
        for r in range(kp_sim.n_regimes):
            regime_name = _regime_label(r, kp_sim.n_regimes)
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
                for r in range(kp_sim.n_regimes)
            },
            "n_regimes": kp_sim.n_regimes,
            "nu": kp_sim.nu,
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

        # 캐시 저장.
        #
        # --no-cache는 "저장된 결과를 읽지 않는다"는 뜻이지 "저장하지 않는다"가 아니다.
        # 저장까지 막으면 sim_meta.json이 낡은 채로 남고, 이를 Base 모수의 출처로
        # 삼는 scenario_main.py / results_main.py가 옛 사양으로 Phase 2를 돌리게 된다.
        # (실제로 KP 2레짐 전환 직후 이 경로로 3레짐 모수가 시나리오에 흘러들었다.)
        if not args.no_save:
            _save_sim_cache(cache_dir, cache_key, {
                "mc_nav_ret": monte_carlo_returns_array,
                "mc_nav": monte_carlo_nav_array,
                "mc_gap": monte_carlo_gap_array,
                "mc_kp": monte_carlo_kp_array,
            }, {
                "gap_params": gap_params_dict,
                "kp_params": kp_params_dict,
                # Phase 2가 GAP/KP 계열을 같은 날짜 그리드로 맞추는 데 쓴다.
                "nav_source": nav_source,
                "S0": S0,
                "T": min(T, T_gap, T_kp),
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
    _print_es(kp_statistical_tests['es'])
    print(f"  VALID: {kp_statistical_tests['is_valid']}")
    
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
    # [Step 5] 시뮬레이터 결합 + [Step 6-A/6-B] 결합 시뮬레이터 검증
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

    # ==================================================================
    # [Step 6-A] 미국 ETF 결합 검증 — NAV x (1+GAP) 대 실측 etf_true
    #
    # 한국형 ETF는 실재하지 않으므로 3자 결합에는 대조할 실측 계열이 없다.
    # 그러나 2자 결합 NAV x (1+GAP)에는 실측 대응물이 있다 — 미국 현물 BTC ETF의
    # 시장가격 etf_true다. 따라서 "시뮬 NAV x (1+시뮬 GAP)이 실제 미국 ETF 가격을
    # 재현하는가"가 관측 가능한 검정이 된다. 여기를 통과한 결합기에 한국 고유
    # 프리미엄(KP)을 얹는 것이 Step 7이다.
    #
    # nav_true 사양에서는 GAP 정의상 실측끼리 항등식이 성립하므로 기준이 정확하다.
    # BTC 현물 사양에서는 항등식이 아니라 추적 관계이므로, 이 검정은 "BTC 현물 기반
    # 결합기가 실제 미국 ETF 가격을 재현하는가"라는 더 강한 물음이 된다.
    # ==================================================================
    print("\n[Step 6-A] 미국 ETF 결합 검증 (기준: 실측 etf_true)")
    actual_us_etf = actual_etf_true[:min_T]
    monte_carlo_us_etf_array = (monte_carlo_nav_array[:, :min_T]
                                * (1.0 + monte_carlo_gap_array[:, :min_T]))

    # nav_true 사양에서는 실측끼리 항등식이 성립한다. 어긋나면 정렬이 깨진 것이다.
    # BTC 현물 사양에서는 항등식이 아니라 추적오차이므로 참고값으로만 출력한다.
    identity_err = float(np.max(np.abs(actual_nav_c * (1.0 + actual_gap_c) - actual_us_etf)))
    if nav_source == "nav_true":
        print(f"  실측 항등식 max|NAV*(1+GAP) - etf_true| = {identity_err:.3e}")
        if identity_err > 1e-6:
            print("  [경고] 항등식이 어긋난다 - NAV/GAP 날짜 정렬을 확인할 것")
    else:
        rel = identity_err / float(np.max(np.abs(actual_us_etf)))
        print(f"  실측 추적오차 max|NAV*(1+GAP) - etf_true| = {identity_err:.4f} "
              f"(최대가 대비 {rel*100:.2f}%) - BTC 현물은 항등식이 아니다")

    actual_us_returns = np.diff(np.log(actual_us_etf))
    simulated_us_returns_array = np.diff(np.log(monte_carlo_us_etf_array), axis=1)
    us_etf_statistical_tests = calculate_statistical_tests(
        actual_returns=actual_us_returns,
        simulated_returns_paths=simulated_us_returns_array,
        alpha=0.05,
    )
    print(f"  PIT-KS: statistic={us_etf_statistical_tests['pit_ks']['ks_statistic']:.4f}, p-value={us_etf_statistical_tests['pit_ks']['ks_pvalue']:.4f}")
    print(f"  VaR-Kupiec: LR_uc={us_etf_statistical_tests['kupiec']['lr_uc']:.4f}, p-value={us_etf_statistical_tests['kupiec']['pvalue']:.4f}, exceedance_rate={us_etf_statistical_tests['kupiec']['exceedance_rate']:.4f}")
    _print_es(us_etf_statistical_tests['es'])
    print(f"  VALID: {us_etf_statistical_tests['is_valid']}")
    print(f"  최종값(달러): 실측 {actual_us_etf[-1]:.2f} / 시뮬 중앙 "
          f"{np.median(monte_carlo_us_etf_array[:, -1]):.2f} / "
          f"p5 {np.percentile(monte_carlo_us_etf_array[:, -1], 5):.2f} / "
          f"p95 {np.percentile(monte_carlo_us_etf_array[:, -1], 95):.2f}")

    # ------------------------------------------------------------------
    # [Step 6-B] 3자 결합 검증 — 실측 대리 계열 기준 (보조)
    #   기준이 실측 한국 ETF가 아니라 실측 세 성분의 곱이므로, 이것은 예측 정확도가
    #   아니라 "결합 절차의 정합성"에 대한 검정이다. 주검증은 Step 6-A다.
    #
    # NAV 기준 상대수익률 (ETF/NAV - 1)을 쓰면 대수적으로 GAP과 같아져
    # (nav*(1+gap)/nav - 1 = gap) NAV·KP 사양 변화에 반응하지 않는다.
    # 결합 가격 자체의 로그수익률을 쓰면
    #   d log ETF = d log NAV + d log(1+GAP) + d log(1+KP)
    # 이므로 세 컴포넌트가 모두 검정에 반영된다.
    # ------------------------------------------------------------------
    print("\n[Step 6-B] 3자 결합 검정 (기준: 실측 성분곱 대리 계열)")

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
    _print_es(combined_statistical_tests['es'])
    print(f"  VALID: {combined_statistical_tests['is_valid']}")

    # 시각화
    print("\n[Step 6-B] 시각화")
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
            "T": T,
            "S0": S0,
        },
        "gap": {
            "statistical_tests": gap_statistical_tests,
            "ou_params": {
                "kappa":  gap_params_dict.get("kappa"),
                "mu":     gap_params_dict.get("mu"),
                "sigma0": gap_params_dict.get("sigma0"),
                "delta1": gap_params_dict.get("delta1"),
                "delta2": gap_params_dict.get("delta2"),
                "nu":     gap_params_dict.get("nu"),
            },
            "T": T_gap,
        },
        "kp": {
            "statistical_tests": kp_statistical_tests,
            "threshold_ou_params": {
                "threshold": kp_params_dict.get("threshold"),
                "n_regimes": kp_params_dict.get("n_regimes"),
                "nu": kp_params_dict.get("nu"),
                "regime_params": {
                    r: kp_params_dict.get("regime_params", {}).get(str(r), {})
                    for r in range(int(kp_params_dict.get("n_regimes",
                                   len(kp_params_dict.get("regime_params", {}))) or 0))
                },
            },
            "T": T_kp,
        },
        "us_etf": {
            # Step 6-A 주검증: NAV x (1+GAP) 대 실측 미국 ETF 가격(etf_true)
            "statistical_tests": us_etf_statistical_tests,
            "T": min_T,
            "basis": "actual etf_true (observed US spot BTC ETF price)",
        },
        "combined": {
            # Step 6-B 보조: 3자 결합 대 실측 성분곱 대리 계열
            "statistical_tests": combined_statistical_tests,
            "T": min_T,
            "basis": "proxy = actual NAV x (1+actual GAP) x (1+actual KP)",
        },
        "nav_source": nav_source,
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
            "representative_gap": representative_gap,
            "actual_gap": actual_gap,
            "T": T_gap,
        },
        "kp": {
            "simulator": kp_sim if not _skip_models else None,
            "statistical_tests": kp_statistical_tests,
            "representative_kp": representative_kp,
            "actual_kp": actual_kp,
            "T": T_kp,
        },
        "us_etf": {
            "statistical_tests": us_etf_statistical_tests,
            "T": min_T,
        },
        "combined": {
            "statistical_tests": combined_statistical_tests,
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
