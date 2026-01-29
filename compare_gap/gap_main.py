"""
괴리율 모델링 메인 실행 스크립트
OU, Heston SV, GARCH 모델 비교
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from compare_gap.ou_simulator import create_ou_simulator
from compare_gap.heston_sv_simulator import create_heston_sv_simulator
from compare_gap.garch_simulator import create_garch_simulator
from compare_gap.gap_metrics import calculate_statistical_tests


def load_gap_data():
    """
    괴리율 데이터 로드
    
    Returns:
        tuple: (수준 시계열 y_t, 변화량 시계열 z_t)
    """
    data_path = Path(__file__).parent.parent / 'dataset' / 'y_variables.csv'
    df = pd.read_csv(data_path)
    
    # etf_premium 컬럼 사용
    y_series = pd.Series(df['etf_premium'].values)
    y_series.index = pd.to_datetime(df['Date'])
    
    # 변화량 계산: z_t = Δy_t = y_t - y_{t-1}
    z_series = y_series.diff().dropna()
    
    print(f"\n데이터 로드 완료:")
    print(f"  - 전체 기간: {len(y_series)}일")
    print(f"  - 변화량 기간: {len(z_series)}일")
    print(f"  - 수준 평균: {y_series.mean():.6f}")
    print(f"  - 수준 표준편차: {y_series.std():.6f}")
    print(f"  - 변화량 평균: {z_series.mean():.6f}")
    print(f"  - 변화량 표준편차: {z_series.std():.6f}")
    
    return y_series, z_series


def run_gap_model_comparison(train_ratio=0.7, n_simulations=1000, selected_models=None):
    """
    괴리율 모델 비교 실행
    
    Args:
        train_ratio: 학습 구간 비율 (기본값: 0.7)
        n_simulations: 몬테카를로 시뮬레이션 횟수 (기본값: 1000)
        selected_models: 선택된 모델 리스트 (None이면 모든 모델)
    """
    print("=" * 60)
    print("괴리율 모델 비교 분석 시작")
    print(f"학습 구간 비율: {train_ratio}")
    print(f"몬테카를로 시뮬레이션 횟수: {n_simulations}")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드")
    print("-" * 60)
    y_series, z_series = load_gap_data()
    
    # 학습/검증 구간 분할
    T_total = len(z_series)
    T_train = int(T_total * train_ratio)
    
    y_train = y_series[:T_train+1]  # 수준은 T_train+1개 필요 (변화량 계산용)
    y_val = y_series[T_train:]
    z_train = z_series[:T_train]
    z_val = z_series[T_train:]
    
    print(f"\n학습/검증 구간 분할:")
    print(f"  - 학습 구간: {len(z_train)}일")
    print(f"  - 검증 구간: {len(z_val)}일")
    
    # 결과 저장 디렉토리
    results_dir = Path(__file__).parent.parent / 'results' / 'gap_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 사용 가능한 모델 목록
    available_models = ['OU', 'Heston-SV', 'GARCH']
    
    # 모델 선택
    if selected_models is None:
        models = available_models
    else:
        models = [m for m in selected_models if m in available_models]
        if len(models) == 0:
            print(f"  [오류] 선택된 모델이 없습니다. 사용 가능한 모델: {available_models}")
            return
    
    print(f"  선택된 모델: {models}")
    all_results = {}
    
    # 2. 각 모델별 시뮬레이션
    for model_name in models:
        print(f"\n[2단계] {model_name} 모델 시뮬레이션")
        print("-" * 60)
        
        model_dir = results_dir / model_name.lower().replace('-', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 모델 파라미터 설정 (학습 구간에서 추정)
            if model_name == 'OU':
                # OU: 수준형 모델
                simulator, params = create_ou_simulator(y_train, method='ols')
                has_level = True
                
            elif model_name == 'Heston-SV':
                # Heston SV: 변화형 모델
                simulator, params = create_heston_sv_simulator(z_train, method='simple')
                has_level = False
                
            elif model_name == 'GARCH':
                # GARCH: 변화형 모델
                simulator, fitted_model = create_garch_simulator(
                    z_train, 
                    garch_p=1, 
                    garch_q=1, 
                    ar_order=0, 
                    dist='normal'
                )
                params = {'model_type': 'GARCH(1,1)-Normal'}
                has_level = False
            
            # 몬테카를로 시뮬레이션 실행
            print(f"\n[2-1단계] {model_name} 몬테카를로 시뮬레이션 ({n_simulations}회)")
            print("-" * 60)
            
            all_simulated_changes_list = []
            
            for sim_idx in range(n_simulations):
                if (sim_idx + 1) % 100 == 0:
                    print(f"  진행 중: {sim_idx + 1}/{n_simulations}")
                
                # 시뮬레이션 실행 (seed를 다르게)
                seed = 42 + sim_idx
                
                if model_name == 'OU':
                    # OU: 수준을 생성한 후 변화량으로 변환
                    simulated_level = simulator.simulate_level(len(z_val), dt=1.0, seed=seed)
                    simulated_changes = simulated_level.diff().dropna()
                else:
                    # Heston SV, GARCH: 변화량 직접 생성
                    simulated_changes = simulator.simulate_changes(len(z_val), seed=seed)
                
                # 시뮬레이션 경로 저장
                all_simulated_changes_list.append(simulated_changes.values)
            
            print(f"  몬테카를로 시뮬레이션 완료: {n_simulations}회")
            
            # 몬테카를로 배열 생성
            monte_carlo_changes_array = np.array(all_simulated_changes_list)
            
            # 대표 경로 선택 (중앙값 경로)
            representative_changes = np.median(monte_carlo_changes_array, axis=0)
            
            print(f"  대표 경로 생성: 시점별 중앙값")
            
            # 통계적 검정 수행
            print(f"\n[2-2단계] {model_name} 통계적 검정")
            print("-" * 60)
            print("  통계적 검정 수행 중...")
            
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
            print(f"    ES: tail_error={statistical_tests['es']['tail_error']:.6f}, "
                  f"mean_actual_es={statistical_tests['es']['mean_actual_es']:.6f}, "
                  f"mean_sim_es={statistical_tests['es']['mean_sim_es']:.6f}, "
                  f"n_violations={statistical_tests['es']['n_violations']}")
            print(f"    VALID: {statistical_tests['is_valid']}")
            
            # 3. 결과 저장
            print(f"\n[3단계] {model_name} 모델 결과 저장")
            print("-" * 60)
            
            # 시뮬레이션 결과 CSV 저장 (대표 경로)
            sim_results_df = pd.DataFrame({
                'simulated_changes': representative_changes,
                'actual_changes': z_val.values,
                'date': z_val.index.values
            })
            
            csv_path = model_dir / 'simulation_results.csv'
            sim_results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  - {csv_path}")
            
            # 파라미터 저장
            params_path = model_dir / 'estimated_parameters.json'
            params_to_save = {}
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    params_to_save[k] = v
                else:
                    params_to_save[k] = str(v)
            
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, ensure_ascii=False, indent=2)
            print(f"  - {params_path}")
            
            # 검증 결과 JSON 저장
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating, np.float64, np.float32, np.int64, np.int32)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    try:
                        return float(obj)
                    except (ValueError, TypeError):
                        return str(obj)
            
            json_path = model_dir / 'validation_results.json'
            validation_results = {
                'statistical_tests': statistical_tests,
                'monte_carlo_summary': {
                    'n_simulations': n_simulations
                }
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(convert_to_serializable(validation_results), f, ensure_ascii=False, indent=2)
            
            print(f"  - {json_path}")
            
            # 전체 결과에 추가
            all_results[model_name] = {
                'simulated_changes': representative_changes.tolist(),
                'validation': validation_results,
                'statistical_tests': statistical_tests,
                'monte_carlo_summary': {
                    'n_simulations': n_simulations
                }
            }
            
        except Exception as e:
            print(f"  [오류] {model_name} 모델 실행 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 4. 전체 비교 결과 저장
    print(f"\n[4단계] 전체 비교 결과 저장")
    print("-" * 60)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.float64, np.float32, np.int64, np.int32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            try:
                return float(obj)
            except (ValueError, TypeError):
                return str(obj)
    
    summary_path = results_dir / 'comparison_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(all_results), f, ensure_ascii=False, indent=2)
    
    print(f"  - {summary_path}")
    
    # 최종 비교 결과 저장 (검증 지표만 요약)
    print(f"\n[5단계] 최종 비교 결과 저장 (검증 지표)")
    print("-" * 60)
    
    final_comparison = {}
    for model_name, results in all_results.items():
        if 'statistical_tests' in results:
            final_comparison[model_name] = results['statistical_tests']
    
    final_path = results_dir / 'compare_final.json'
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(final_comparison), f, ensure_ascii=False, indent=2)
    
    print(f"  - {final_path}")
    
    # 통계적 검정 결과 출력
    print(f"\n[통계적 검정 결과 요약]")
    print("-" * 60)
    for model_name in models:
        if model_name in final_comparison:
            tests = final_comparison[model_name]
            print(f"\n{model_name}:")
            print(f"  PIT-KS: statistic={tests['pit_ks']['ks_statistic']:.4f}, "
                  f"p-value={tests['pit_ks']['ks_pvalue']:.4f}")
            print(f"  VaR-Kupiec: LR_uc={tests['kupiec']['lr_uc']:.4f}, "
                  f"p-value={tests['kupiec']['pvalue']:.4f}, "
                  f"exceedance_rate={tests['kupiec']['exceedance_rate']:.4f} "
                  f"(expected={tests['kupiec']['expected_rate']:.4f})")
            print(f"  ES: tail_error={tests['es']['tail_error']:.6f}, "
                  f"mean_actual_es={tests['es']['mean_actual_es']:.6f}, "
                  f"mean_sim_es={tests['es']['mean_sim_es']:.6f}, "
                  f"n_violations={tests['es']['n_violations']}")
            print(f"  VALID: {tests['is_valid']}")
    
    print("\n" + "=" * 60)
    print("괴리율 모델 비교 분석 완료")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='괴리율 모델 비교')
    parser.add_argument('--models', nargs='+', 
                       choices=['OU', 'Heston-SV', 'GARCH', 'all'],
                       default=['all'],
                       help='비교할 모델 선택 (예: --models OU Heston-SV 또는 --models all)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='학습 구간 비율 (기본값: 0.7)')
    parser.add_argument('--n_simulations', type=int, default=1000,
                       help='몬테카를로 시뮬레이션 횟수 (기본값: 1000)')
    
    args = parser.parse_args()
    
    # 모델 선택 처리
    if 'all' in args.models:
        selected_models = None  # 모든 모델
    else:
        selected_models = args.models
    
    run_gap_model_comparison(
        train_ratio=args.train_ratio,
        n_simulations=args.n_simulations,
        selected_models=selected_models
    )
