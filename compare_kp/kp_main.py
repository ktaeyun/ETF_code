"""
김치 프리미엄(KP) 모델링 메인 실행 스크립트
OU, Heston-SV 모델 비교 (gap과 동일 과정)
데이터: dataset/y_variables.csv Kimchi Premium 컬럼
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from compare_gap.ou_simulator import create_ou_simulator
from compare_gap.heston_sv_simulator import create_heston_sv_simulator
from compare_kp.kp_metrics import calculate_statistical_tests
from compare_kp.kp_visualizer import create_all_kp_visualizations


def load_kp_data():
    """
    김치 프리미엄(KP) 데이터 로드 (y_variables.csv Kimchi Premium 컬럼)
    
    Returns:
        tuple: (수준 시계열 y_t, 변화량 시계열 z_t)
    """
    data_path = Path(__file__).parent.parent / 'dataset' / 'y_variables.csv'
    df = pd.read_csv(data_path)
    
    y_series = pd.Series(df['Kimchi Premium'].values)
    y_series.index = pd.to_datetime(df['Date'])
    z_series = y_series.diff().dropna()
    
    print(f"\n데이터 로드 완료 (Kimchi Premium):")
    print(f"  - 전체 기간: {len(y_series)}일")
    print(f"  - 변화량 기간: {len(z_series)}일")
    print(f"  - 수준 평균: {y_series.mean():.6f}")
    print(f"  - 수준 표준편차: {y_series.std():.6f}")
    print(f"  - 변화량 평균: {z_series.mean():.6f}")
    print(f"  - 변화량 표준편차: {z_series.std():.6f}")
    
    return y_series, z_series


def run_kp_model_comparison(train_ratio=0.8, n_simulations=5000, selected_models=None):
    """
    김치 프리미엄(KP) 모델 비교 실행 (gap과 동일 과정)
    
    Args:
        train_ratio: 학습 구간 비율 (기본값: 0.8)
        n_simulations: 몬테카를로 시뮬레이션 횟수
        selected_models: 선택된 모델 리스트 (None이면 OU, Heston-SV 전체)
    """
    print("=" * 60)
    print("김치 프리미엄(KP) 모델 비교 분석 시작")
    print(f"학습 구간 비율: {train_ratio}")
    print(f"몬테카를로 시뮬레이션 횟수: {n_simulations}")
    print("=" * 60)
    
    print("\n[1단계] 데이터 로드")
    print("-" * 60)
    y_series, z_series = load_kp_data()
    
    T_total = len(z_series)
    T_train = int(T_total * train_ratio)
    y_train = y_series[:T_train + 1]
    y_val = y_series[T_train:]
    z_train = z_series[:T_train]
    z_val = z_series[T_train:]
    
    print(f"\n학습/검증 구간 분할:")
    print(f"  - 학습 구간: {len(z_train)}일")
    print(f"  - 검증 구간: {len(z_val)}일")
    
    results_dir = Path(__file__).parent.parent / 'results' / 'kp_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    available_models = ['OU', 'Heston-SV']
    if selected_models is None:
        models = available_models
    else:
        models = [m for m in selected_models if m in available_models]
        if not models:
            print(f"  [오류] 선택된 모델이 없습니다. 사용 가능: {available_models}")
            return
    
    print(f"  선택된 모델: {models}")
    all_results = {}
    
    for model_name in models:
        print(f"\n[2단계] {model_name} 모델 시뮬레이션")
        print("-" * 60)
        model_dir = results_dir / model_name.lower().replace('-', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if model_name == 'OU':
                simulator, params = create_ou_simulator(y_train, method='ols')
                has_level = True
            elif model_name == 'Heston-SV':
                simulator, params = create_heston_sv_simulator(z_train, method='simple')
                has_level = False
            else:
                continue
            
            print(f"\n[2-1단계] {model_name} 몬테카를로 시뮬레이션 ({n_simulations}회)")
            print("-" * 60)
            all_simulated_changes_list = []
            for sim_idx in range(n_simulations):
                if (sim_idx + 1) % 100 == 0:
                    print(f"  진행 중: {sim_idx + 1}/{n_simulations}")
                seed = 42 + sim_idx
                if model_name == 'OU':
                    simulated_changes = simulator.simulate_changes(len(z_val), dt=1.0, seed=seed)
                else:
                    simulated_changes = simulator.simulate_changes(len(z_val), seed=seed)
                all_simulated_changes_list.append(simulated_changes.values)
            
            print(f"  몬테카를로 시뮬레이션 완료: {n_simulations}회")
            monte_carlo_changes_array = np.array(all_simulated_changes_list)
            representative_changes = np.median(monte_carlo_changes_array, axis=0)
            print(f"  대표 경로 생성: 시점별 중앙값")
            
            print(f"\n[2-2단계] {model_name} 통계적 검정")
            print("-" * 60)
            statistical_tests = calculate_statistical_tests(
                actual_changes=z_val.values,
                simulated_changes_paths=monte_carlo_changes_array,
                alpha=0.05
            )
            print(f"    PIT-KS: statistic={statistical_tests['pit_ks']['ks_statistic']:.4f}, "
                  f"p-value={statistical_tests['pit_ks']['ks_pvalue']:.4f}")
            print(f"    VaR-Kupiec: LR_uc={statistical_tests['kupiec']['lr_uc']:.4f}, "
                  f"p-value={statistical_tests['kupiec']['pvalue']:.4f}, "
                  f"exceedance_rate={statistical_tests['kupiec']['exceedance_rate']:.4f} "
                  f"(expected={statistical_tests['kupiec']['expected_rate']:.4f})")
            es = statistical_tests['es']
            tail_err = f"{es['tail_error']:.6f}" if es['tail_error'] is not None else "N/A"
            mean_act = f"{es['mean_actual_es']:.6f}" if es['mean_actual_es'] is not None else "N/A"
            mean_sim = f"{es['mean_sim_es']:.6f}" if es['mean_sim_es'] is not None else "N/A"
            print(f"    ES: tail_error={tail_err}, mean_actual_es={mean_act}, "
                  f"mean_sim_es={mean_sim}, n_violations={es['n_violations']}")
            print(f"    VALID: {statistical_tests['is_valid']}")
            
            print(f"\n[3단계] {model_name} 모델 시각화")
            print("-" * 60)
            y0_validation = float(y_val.iloc[0])
            actual_level_val = None
            simulated_level_val = None
            monte_carlo_level_array = None
            closest_level_path_idx = None
            
            if model_name == 'OU':
                actual_level_val = y_val.values
                _y0_prev = getattr(simulator, 'y0', None)
                simulator.y0 = y0_validation
                all_simulated_level_list = []
                for sim_idx in range(n_simulations):
                    seed = 42 + sim_idx
                    simulated_level = simulator.simulate_level(len(z_val) + 1, dt=1.0, seed=seed)
                    all_simulated_level_list.append(simulated_level.values)
                if _y0_prev is not None:
                    simulator.y0 = _y0_prev
                monte_carlo_level_array = np.array(all_simulated_level_list)
                simulated_level_val = np.median(monte_carlo_level_array, axis=0)
            elif model_name == 'Heston-SV':
                actual_level_val = y_val.values
                all_simulated_level_list = []
                for path_changes in all_simulated_changes_list:
                    level_path = np.concatenate([[y0_validation], y0_validation + np.cumsum(path_changes)])
                    all_simulated_level_list.append(level_path)
                monte_carlo_level_array = np.array(all_simulated_level_list)
                simulated_level_val = np.median(monte_carlo_level_array, axis=0)
            
            if actual_level_val is not None and monte_carlo_level_array is not None:
                actual_level_val = np.asarray(actual_level_val)
                min_len = min(len(actual_level_val), monte_carlo_level_array.shape[1])
                actual_level_val = actual_level_val[:min_len]
                monte_carlo_level_array = monte_carlo_level_array[:, :min_len]
                simulated_level_val = simulated_level_val[:min_len]
                rmse_per_path = np.sqrt(np.mean((monte_carlo_level_array - actual_level_val) ** 2, axis=1))
                closest_level_path_idx = int(np.argmin(rmse_per_path))
                print(f"  수준 경로 초기값(검증 첫 KP) y0 = {y0_validation:.6f}")
                print(f"  실제와 가장 유사한 수준 경로: 시뮬레이션 #{closest_level_path_idx + 1} (RMSE={rmse_per_path[closest_level_path_idx]:.6f})")
            
            create_all_kp_visualizations(
                actual_changes=z_val.values,
                simulated_changes=representative_changes,
                actual_level=actual_level_val,
                simulated_level=simulated_level_val,
                output_dir=model_dir / 'plots',
                monte_carlo_changes_paths=monte_carlo_changes_array,
                monte_carlo_level_paths=monte_carlo_level_array,
                closest_level_path_idx=closest_level_path_idx
            )
            
            print(f"\n[4단계] {model_name} 모델 결과 저장")
            print("-" * 60)
            sim_results_df = pd.DataFrame({
                'simulated_changes': representative_changes,
                'actual_changes': z_val.values,
                'date': z_val.index.values
            })
            sim_results_df.to_csv(model_dir / 'simulation_results.csv', index=False, encoding='utf-8-sig')
            params_to_save = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()}
            with open(model_dir / 'estimated_parameters.json', 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, ensure_ascii=False, indent=2)
            
            def _serialize(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.floating, np.float64, np.float32, np.int64, np.int32)):
                    return float(obj)
                if isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                if isinstance(obj, dict):
                    return {k: _serialize(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_serialize(x) for x in obj]
                return obj
            
            validation_results = {
                'statistical_tests': statistical_tests,
                'monte_carlo_summary': {'n_simulations': n_simulations}
            }
            with open(model_dir / 'validation_results.json', 'w', encoding='utf-8') as f:
                json.dump(_serialize(validation_results), f, ensure_ascii=False, indent=2)
            print(f"  - {model_dir / 'validation_results.json'}")
            
            all_results[model_name] = {
                'statistical_tests': statistical_tests,
                'monte_carlo_summary': {'n_simulations': n_simulations}
            }
            
        except Exception as e:
            print(f"  [오류] {model_name} 모델 실행 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[5단계] 전체 비교 결과 저장")
    print("-" * 60)
    def _ser(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.float64, np.float32, np.int64, np.int32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_ser(x) for x in obj]
        return obj
    
    with open(results_dir / 'comparison_summary.json', 'w', encoding='utf-8') as f:
        json.dump(_ser(all_results), f, ensure_ascii=False, indent=2)
    final_comparison = {m: r['statistical_tests'] for m, r in all_results.items()}
    with open(results_dir / 'compare_final.json', 'w', encoding='utf-8') as f:
        json.dump(_ser(final_comparison), f, ensure_ascii=False, indent=2)
    print(f"  - {results_dir / 'compare_final.json'}")
    
    print(f"\n[통계적 검정 결과 요약]")
    print("-" * 60)
    for model_name in models:
        if model_name in final_comparison:
            tests = final_comparison[model_name]
            print(f"\n{model_name}:")
            print(f"  PIT-KS: p-value={tests['pit_ks']['ks_pvalue']:.4f}")
            print(f"  VaR-Kupiec: p-value={tests['kupiec']['pvalue']:.4f}, exceedance_rate={tests['kupiec']['exceedance_rate']:.4f}")
            print(f"  VALID: {tests['is_valid']}")
    
    print("\n" + "=" * 60)
    print("김치 프리미엄(KP) 모델 비교 분석 완료")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='김치 프리미엄(KP) 모델 비교')
    parser.add_argument('--models', nargs='+', choices=['OU', 'Heston-SV', 'all'], default=['all'])
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--n_simulations', type=int, default=5000)
    args = parser.parse_args()
    selected = None if 'all' in args.models else args.models
    run_kp_model_comparison(
        train_ratio=args.train_ratio,
        n_simulations=args.n_simulations,
        selected_models=selected
    )
