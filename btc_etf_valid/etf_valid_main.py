"""
BTC ETF 유효성 검정 메인 스크립트
NAV 모델 × Gap 모델 조합별로 ETF = NAV×(1+gap) 구성 후 etf_true와 3가지 통계적 검정
조합: NAV ∈ {Heston, ARIMA-GARCH}, Gap ∈ {Heston-SV, OU}
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
from simulator.data_loader import log_returns_to_nav
from compare.heston_simulator import HestonSimulator, fit_heston_parameters
from compare.arima_garch_simulator import fit_arima_garch
from compare_gap.heston_sv_simulator import create_heston_sv_simulator
from compare_gap.ou_simulator import create_ou_simulator
from compare.metrics import calculate_statistical_tests
from btc_etf_valid.etf_visualizer import create_all_etf_visualizations

# 조합: (NAV 모델, Gap 모델)
NAV_MODELS = ['Heston', 'ARIMA-GARCH']
GAP_MODELS = ['Heston-SV', 'OU']


def load_etf_validation_data(gap_unit='decimal'):
    """
    y_true_variables(nav_true, etf_true)와 y_variables(etf_premium) 로드 후 Date 기준 병합.
    
    gap_unit: 'decimal' = 괴리율이 비율(소수). 'percent' = 퍼센트 → /100 적용.
    
    Returns:
        tuple: (merged_df, S0, y0_gap, returns_nav, z_gap, y_gap_series, etf_true, T)
    """
    y_true_path = settings.get_y_true_variables_path()
    y_var_path = settings.RAW_DATASET_DIR / 'y_variables.csv'
    
    y_true_df = pd.read_csv(y_true_path)
    y_true_df['Date'] = pd.to_datetime(y_true_df['Date'])
    y_var_df = pd.read_csv(y_var_path)
    y_var_df['Date'] = pd.to_datetime(y_var_df['Date'])
    
    merged = pd.merge(
        y_true_df[['Date', 'nav_true', 'etf_true']],
        y_var_df[['Date', 'etf_premium']],
        on='Date',
        how='inner'
    ).sort_values('Date').reset_index(drop=True)
    
    nav = merged['nav_true'].values.astype(float)
    etf_true = merged['etf_true'].values.astype(float)
    gap_level = merged['etf_premium'].values.astype(float)
    if gap_unit == 'percent':
        gap_level = gap_level / 100.0
    
    S0 = float(nav[0])
    y0_gap = float(gap_level[0])
    T = len(nav) - 1
    
    returns_nav = np.diff(np.log(nav))
    z_gap = np.diff(gap_level)
    
    returns_nav_series = pd.Series(returns_nav)
    z_gap_series = pd.Series(z_gap)
    y_gap_series = pd.Series(gap_level)
    
    print(f"\n데이터 로드 완료 (Date 기준 병합):")
    print(f"  - 괴리율 단위: {gap_unit}")
    print(f"  - 일수: {len(merged)} (수익률/변화량 T={T})")
    print(f"  - S0(NAV): {S0:.4f}, y0(gap): {y0_gap:.6f}")
    print(f"  - 기간: {merged['Date'].iloc[0].date()} ~ {merged['Date'].iloc[-1].date()}")
    
    return merged, S0, y0_gap, returns_nav_series, z_gap_series, y_gap_series, etf_true, T


def _simulate_nav_paths(nav_model, returns_nav, S0, T, n_simulations):
    """NAV 모델별 몬테카를로 경로 생성 (N x T)."""
    if nav_model == 'Heston':
        params = fit_heston_parameters(returns_nav)
        sim = HestonSimulator(
            mu=params['mu'], kappa=params['kappa'], theta=params['theta'],
            sigma_v=params['sigma_v'], rho=params['rho'], v0=params['v0']
        )
        paths = []
        for i in range(n_simulations):
            ret = sim.simulate_returns(T, seed=42 + i)
            rv = ret.values if hasattr(ret, 'values') else np.asarray(ret).flatten()
            paths.append(log_returns_to_nav(S0, rv))
        return np.array(paths)
    
    if nav_model == 'ARIMA-GARCH':
        arima_garch_sim, _ = fit_arima_garch(
            returns_nav, ar_order=(1, 0, 1), garch_p=1, garch_q=1, dist='normal'
        )
        paths = []
        for i in range(n_simulations):
            ret = arima_garch_sim.simulate_returns(T, seed=42 + i)
            rv = ret.values if hasattr(ret, 'values') else np.asarray(ret).flatten()
            paths.append(log_returns_to_nav(S0, rv))
        return np.array(paths)
    
    raise ValueError(f"Unknown nav_model: {nav_model}")


def _simulate_gap_level_paths(gap_model, z_gap, y_gap_series, y0_gap, T, n_simulations):
    """Gap 모델별 괴리율 수준 몬테카를로 경로 생성 (N x T)."""
    if gap_model == 'Heston-SV':
        sim, _ = create_heston_sv_simulator(z_gap, method='simple')
        paths = []
        for i in range(n_simulations):
            ch = sim.simulate_changes(T, seed=42 + i)
            cv = ch.values if hasattr(ch, 'values') else np.asarray(ch).flatten()
            paths.append(y0_gap + np.cumsum(cv))
        return np.array(paths)
    
    if gap_model == 'OU':
        sim, _ = create_ou_simulator(y_gap_series, method='ols')
        sim.y0 = y0_gap
        paths = []
        for i in range(n_simulations):
            level = sim.simulate_level(T, dt=1.0, seed=42 + i)
            lv = level.values if hasattr(level, 'values') else np.asarray(level).flatten()
            if len(lv) < T:
                lv = np.resize(lv, T)
            paths.append(lv[:T])
        return np.array(paths)
    
    raise ValueError(f"Unknown gap_model: {gap_model}")


def _run_one_combination(nav_model, gap_model, merged, S0, y0_gap, returns_nav, z_gap, y_gap_series,
                         etf_true, T, n_simulations, results_dir, gap_unit='decimal'):
    """단일 조합에 대해 ETF 경로 생성, 검정, 시각화, 결과 저장."""
    combo_name = f"{nav_model}_{gap_model.replace('-', '_')}"
    combo_dir = results_dir / combo_name
    plots_dir = combo_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"조합: NAV={nav_model}, Gap={gap_model}")
    print("=" * 60)
    
    # NAV 경로
    print(f"\n  [{nav_model}] NAV 시뮬레이션 ({n_simulations}회)")
    nav_paths = _simulate_nav_paths(nav_model, returns_nav, S0, T, n_simulations)
    print(f"    NAV 경로: {nav_paths.shape}")
    
    # Gap 경로
    print(f"  [{gap_model}] gap 시뮬레이션 ({n_simulations}회)")
    gap_level_paths = _simulate_gap_level_paths(gap_model, z_gap, y_gap_series, y0_gap, T, n_simulations)
    print(f"    gap 수준 경로: {gap_level_paths.shape}")
    
    # ETF 경로: ETF[0]=S0*(1+y0_gap), ETF[t]=NAV[t-1]*(1+gap[t-1]), t=1..T
    etf_paths = np.zeros((n_simulations, T + 1))
    etf_paths[:, 0] = S0 * (1 + y0_gap)
    for t in range(1, T + 1):
        etf_paths[:, t] = nav_paths[:, t - 1] * (1 + gap_level_paths[:, t - 1])
    simulated_etf_returns = np.diff(np.log(etf_paths), axis=1)
    
    # 통계적 검정
    actual_etf_returns = np.diff(np.log(etf_true))
    statistical_tests = calculate_statistical_tests(
        actual_returns=actual_etf_returns,
        simulated_returns_paths=simulated_etf_returns,
        alpha=0.05
    )
    
    print(f"\n  통계적 검정 (수익률 기준):")
    print(f"    PIT-KS: p-value={statistical_tests['pit_ks']['ks_pvalue']:.4f}")
    print(f"    VaR-Kupiec: p-value={statistical_tests['kupiec']['pvalue']:.4f}, "
          f"exceedance_rate={statistical_tests['kupiec']['exceedance_rate']:.4f}")
    es = statistical_tests['es']
    nv = es['n_violations']
    print(f"    ES: n_violations={nv}, tail_error={es['tail_error']}")
    print(f"    is_valid={statistical_tests['is_valid']}")
    
    # 시각화
    rep_etf = np.median(etf_paths, axis=0)
    rep_returns = np.median(simulated_etf_returns, axis=0)
    create_all_etf_visualizations(
        actual_etf=etf_true,
        actual_etf_returns=actual_etf_returns,
        simulated_etf_median=rep_etf,
        simulated_etf_returns_median=rep_returns,
        monte_carlo_etf_paths=etf_paths,
        monte_carlo_etf_returns=simulated_etf_returns,
        output_dir=plots_dir,
    )
    
    def _serialize(obj):
        if obj is None:
            return None
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(x) for x in obj]
        return obj
    
    result = {
        'nav_model': nav_model,
        'gap_model': gap_model,
        'statistical_tests': _serialize(statistical_tests),
        'n_simulations': n_simulations,
        'T': T,
        'description': f'NAV({nav_model}) * (1 + gap({gap_model})) → ETF vs etf_true',
    }
    out_path = combo_dir / 'validation_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  결과 저장: {out_path}")
    
    return result


def run_etf_validation(n_simulations=1000, gap_unit='decimal', combinations=None):
    """
    조합별 ETF = NAV×(1+gap) 구성 후 etf_true와 3가지 통계적 검정 수행.
    
    combinations: None이면 전체 (Heston, ARIMA-GARCH) x (Heston-SV, OU).
                  예: [('Heston','Heston-SV'), ('ARIMA-GARCH','OU')]
    """
    print("=" * 60)
    print("BTC ETF 유효성 검정 (NAV×Gap 조합별)")
    print("=" * 60)
    
    if combinations is None:
        combinations = [(n, g) for n in NAV_MODELS for g in GAP_MODELS]
    
    print(f"\n조합 목록: {combinations}")
    
    merged, S0, y0_gap, returns_nav, z_gap, y_gap_series, etf_true, T = load_etf_validation_data(gap_unit=gap_unit)
    
    results_dir = Path(settings.PROJECT_ROOT) / 'results' / 'btc_etf_valid'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for nav_model, gap_model in combinations:
        if nav_model not in NAV_MODELS or gap_model not in GAP_MODELS:
            print(f"  [건너뜀] 알 수 없는 조합: ({nav_model}, {gap_model})")
            continue
        try:
            result = _run_one_combination(
                nav_model, gap_model,
                merged, S0, y0_gap, returns_nav, z_gap, y_gap_series,
                etf_true, T, n_simulations, results_dir, gap_unit
            )
            key = f"{nav_model}_{gap_model.replace('-', '_')}"
            all_results[key] = {
                'nav_model': nav_model,
                'gap_model': gap_model,
                'is_valid': result['statistical_tests']['is_valid'],
                'pit_ks_pvalue': result['statistical_tests']['pit_ks']['ks_pvalue'],
                'kupiec_pvalue': result['statistical_tests']['kupiec']['pvalue'],
                'kupiec_exceedance_rate': result['statistical_tests']['kupiec']['exceedance_rate'],
            }
        except Exception as e:
            print(f"  [오류] {nav_model}+{gap_model}: {e}")
            all_results[f"{nav_model}_{gap_model.replace('-', '_')}"] = {'error': str(e)}
    
    # 통합 결과 저장
    summary_path = results_dir / 'validation_results_all.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n통합 결과 저장: {summary_path}")
    print("=" * 60)
    print("BTC ETF 유효성 검정 완료")
    print("=" * 60)
    
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='BTC ETF 유효성 검정 (NAV×Gap 조합별)')
    parser.add_argument('--n_simulations', type=int, default=1000, help='몬테카를로 시뮬레이션 횟수')
    parser.add_argument('--gap_unit', choices=['decimal', 'percent'], default='decimal')
    parser.add_argument('--combinations', nargs='+', default=None,
                        help='예: Heston Heston-SV ARIMA-GARCH OU (2개씩 묶여 2조합) 또는 생략 시 전체 4조합')
    args = parser.parse_args()
    
    combos = None
    if args.combinations:
        tokens = args.combinations
        combos = [(tokens[i], tokens[i+1]) for i in range(0, len(tokens), 2) if i+1 < len(tokens)]
    
    run_etf_validation(n_simulations=args.n_simulations, gap_unit=args.gap_unit, combinations=combos)
