"""
시뮬레이터 비교 메인 실행 스크립트
GBM, Heston, GARCH, Poisson-Gaussian 모델 비교
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
from simulator.data_loader import load_nav_data
from simulator.visualizer import create_all_visualizations
from simulator.jump_detector import detect_jumps

from compare.gbm_simulator import GBMSimulator, fit_gbm_parameters
from compare.heston_simulator import HestonSimulator, fit_heston_parameters
from compare.garch_simulator import GARCHSimulator, fit_garch_model
from compare.poisson_gaussian_simulator import create_poisson_gaussian_simulator
from compare.merton_jd_simulator import fit_merton_jd_model, MertonJDSimulator
from compare.metrics import calculate_statistical_tests


def run_model_comparison(n_simulations=1000, selected_models=None):
    """
    모델 비교 실행 (몬테카를로 시뮬레이션)
    
    Args:
        n_simulations: 몬테카를로 시뮬레이션 횟수 (기본값: 100)
        selected_models: 선택된 모델 리스트 (None이면 모든 모델)
    """
    print("=" * 60)
    print("시뮬레이터 비교 분석 시작 (몬테카를로 시뮬레이션)")
    print(f"몬테카를로 시뮬레이션 횟수: {n_simulations}")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드")
    print("-" * 60)
    nav_series, returns_series = load_nav_data()
    
    T = len(returns_series)
    S0 = nav_series.iloc[0]
    actual_nav_aligned = nav_series.iloc[1:].values
    
    # 결과 저장 디렉토리
    results_dir = Path(settings.PROJECT_ROOT) / 'results' / 'compare_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 사용 가능한 모델 목록
    available_models = ['GBM', 'Poisson-Gaussian', 'Merton-JD']
    
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
            # 모델 파라미터 설정 (한 번만)
            if model_name == 'GBM':
                params = fit_gbm_parameters(returns_series)
                simulator_factory = lambda: GBMSimulator(mu=params['mu'], sigma=params['sigma'])
                has_jumps = False
            
            elif model_name == 'Heston':
                params = fit_heston_parameters(returns_series)
                simulator_factory = lambda: HestonSimulator(
                    mu=params['mu'],
                    kappa=params['kappa'],
                    theta=params['theta'],
                    sigma_v=params['sigma_v'],
                    rho=params['rho'],
                    v0=params['v0']
                )
                has_jumps = False
            
            elif model_name == 'GARCH':
                fitted_model = fit_garch_model(returns_series, garch_p=1, garch_q=1, dist='t')
                simulator_factory = lambda: GARCHSimulator(fitted_model)
                has_jumps = False
            
            elif model_name == 'Poisson-Gaussian':
                # jump_result는 한 번만 계산
                jump_result_actual = detect_jumps(returns_series, quantile=0.95)
                actual_jump_times = np.where(jump_result_actual['jump_mask'])[0]
                simulator_factory = lambda: create_poisson_gaussian_simulator(
                    returns_series,
                    continuous_model_type='garch'
                )[0]  # simulator만 반환
                has_jumps = True
            
            elif model_name == 'Merton-JD':
                # Merton Jump-Diffusion 모델
                merton_simulator, merton_params = fit_merton_jd_model(
                    returns_series,
                    jump_detection_method='quantile',
                    quantile=0.95
                )
                print(f"\nMerton JD 파라미터 추정 완료:")
                print(f"  μ = {merton_params['mu']:.6f}")
                print(f"  σ = {merton_params['sigma']:.6f}")
                print(f"  λ = {merton_params['lambda']:.6f}")
                print(f"  μ_J = {merton_params['mu_J']:.6f}")
                print(f"  σ_J = {merton_params['sigma_J']:.6f}")
                print(f"  κ = {merton_params['kappa']:.6f}")
                print(f"  점프일 수: {merton_params['n_jumps']}/{T} ({merton_params['n_jumps']/T*100:.2f}%)")
                # Merton JD는 매번 새 인스턴스 생성 (파라미터는 동일)
                simulator_factory = lambda: MertonJDSimulator(
                    mu=merton_params['mu'],
                    sigma=merton_params['sigma'],
                    lambda_param=merton_params['lambda'],
                    mu_J=merton_params['mu_J'],
                    sigma_J=merton_params['sigma_J'],
                    kappa=merton_params['kappa']
                )
                has_jumps = True
                # 파라미터 저장용
                model_params = merton_params
                # Merton JD의 점프 시점은 시뮬레이션 중에 결정되므로 None
                actual_jump_times = None
            
            # 실제 점프 시점 설정
            if model_name == 'Poisson-Gaussian':
                jump_result_actual = detect_jumps(returns_series, quantile=0.95)
                actual_jump_times = np.where(jump_result_actual['jump_mask'])[0]
            elif model_name == 'Merton-JD':
                # Merton JD의 점프 탐지 결과 사용
                from compare.merton_jd_simulator import detect_jumps_quantile
                jump_result_actual = detect_jumps_quantile(returns_series, quantile=0.95)
                actual_jump_times = np.where(jump_result_actual['jump_mask'])[0]
            else:
                actual_jump_times = None
            
            # 몬테카를로 시뮬레이션 실행
            print(f"\n[2-1단계] {model_name} 몬테카를로 시뮬레이션 ({n_simulations}회)")
            print("-" * 60)
            
            all_simulated_returns_list = []
            all_simulated_nav_list = []
            
            for sim_idx in range(n_simulations):
                if (sim_idx + 1) % 10 == 0:
                    print(f"  진행 중: {sim_idx + 1}/{n_simulations}")
                
                # 시뮬레이터 생성
                if model_name == 'Poisson-Gaussian':
                    simulator, _ = create_poisson_gaussian_simulator(
                        returns_series,
                        continuous_model_type='garch'
                    )
                else:
                    simulator = simulator_factory()
                
                # 시뮬레이션 실행 (seed를 다르게)
                seed = 42 + sim_idx
                simulated_returns = simulator.simulate_returns(T, seed=seed)
                simulated_nav = simulator.simulate_nav_path(S0, T, seed=seed)
                
                # 시뮬레이션 경로 저장 (통계적 검정용)
                all_simulated_returns_list.append(simulated_returns.values)
                all_simulated_nav_list.append(simulated_nav.values)
            
            print(f"  몬테카를로 시뮬레이션 완료: {n_simulations}회")
            
            # 대표 경로 선택 (중앙값 경로)
            # 각 시점별 중앙값으로 대표 경로 생성
            monte_carlo_returns_array = np.array(all_simulated_returns_list)
            monte_carlo_nav_array = np.array(all_simulated_nav_list)
            
            representative_returns = np.median(monte_carlo_returns_array, axis=0)
            representative_nav = np.median(monte_carlo_nav_array, axis=0)
            
            print(f"  대표 경로 생성: 시점별 중앙값")
            
            # numpy 배열로 변환 (통계적 검정용)
            monte_carlo_nav_array = np.array(all_simulated_nav_list)
            monte_carlo_returns_array = np.array(all_simulated_returns_list)
            
            # 통계적 검정 수행 (기존 검증 방식 제거)
            print(f"\n[2-2단계] {model_name} 통계적 검정")
            print("-" * 60)
            print("  통계적 검정 수행 중...")
            
            statistical_tests = calculate_statistical_tests(
                actual_returns=returns_series.values,
                simulated_returns_paths=monte_carlo_returns_array,
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
            
            # 3. 검증 결과 저장 (통계적 검정만)
            print(f"\n[3단계] {model_name} 모델 검증 결과 저장")
            print("-" * 60)
            
            validation_results = {
                'statistical_tests': statistical_tests,
                'monte_carlo_summary': {
                    'n_simulations': n_simulations
                }
            }
            
            # 4. 시각화 (몬테카를로 시뮬레이션 모든 경로 사용)
            print(f"\n[4단계] {model_name} 모델 시각화 (몬테카를로 시뮬레이션)")
            print("-" * 60)
            
            # 점프 시점 설정 (시각화용)
            jump_times = None
            if has_jumps and actual_jump_times is not None:
                if 'threshold' in jump_result_actual:
                    simulated_jump_mask = np.abs(representative_returns) >= jump_result_actual['threshold']
                    jump_times = np.where(simulated_jump_mask)[0]
            
            create_all_visualizations(
                actual_nav=actual_nav_aligned,
                simulated_nav=representative_nav,
                actual_returns=returns_series.values,
                simulated_returns=representative_returns,
                actual_jump_times=actual_jump_times,
                simulated_jump_times=jump_times,
                output_dir=model_dir / 'plots',
                monte_carlo_nav_paths=monte_carlo_nav_array,
                monte_carlo_returns_paths=monte_carlo_returns_array
            )
            
            # 5. 결과 저장
            print(f"\n[5단계] {model_name} 모델 결과 저장")
            print("-" * 60)
            
            # 시뮬레이션 결과 CSV 저장 (대표 경로)
            sim_results_df = pd.DataFrame({
                'simulated_returns': representative_returns,
                'simulated_nav': representative_nav,
                'actual_returns': returns_series.values,
                'actual_nav': actual_nav_aligned
            })
            
            csv_path = model_dir / 'simulation_results.csv'
            sim_results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  - {csv_path}")
            
            # 파라미터 저장 (Merton-JD 등)
            if model_name == 'Merton-JD':
                params_path = model_dir / 'estimated_parameters.json'
                params_to_save = {
                    'mu': float(model_params['mu']),
                    'sigma': float(model_params['sigma']),
                    'lambda': float(model_params['lambda']),
                    'mu_J': float(model_params['mu_J']),
                    'sigma_J': float(model_params['sigma_J']),
                    'kappa': float(model_params['kappa']),
                    'n_jumps': int(model_params['n_jumps']),
                    'n_total': T
                }
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
                elif hasattr(obj, '__dict__'):
                    return convert_to_serializable(obj.__dict__)
                else:
                    try:
                        return float(obj)
                    except (ValueError, TypeError):
                        return str(obj)
            
            json_path = model_dir / 'validation_results.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(convert_to_serializable(validation_results), f, ensure_ascii=False, indent=2)
            
            print(f"  - {json_path}")
            
            # 전체 결과에 추가 (대표 경로)
            all_results[model_name] = {
                'simulated_returns': representative_returns.tolist(),
                'simulated_nav': representative_nav.tolist(),
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
    
    # 6. 전체 비교 결과 저장
    print(f"\n[6단계] 전체 비교 결과 저장")
    print("-" * 60)
    
    # numpy 타입을 JSON 직렬화 가능하게 변환
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
        elif hasattr(obj, '__dict__'):
            return convert_to_serializable(obj.__dict__)
        else:
            try:
                return float(obj)
            except (ValueError, TypeError):
                return str(obj)
    
    summary_path = results_dir / 'comparison_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(all_results), f, ensure_ascii=False, indent=2)
    
    print(f"  - {summary_path}")
    
    # 7. 최종 비교 결과 저장 (금융공학 지표만 요약)
    print(f"\n[7단계] 최종 비교 결과 저장 (금융공학 지표)")
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
    print("시뮬레이터 비교 분석 완료")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='시뮬레이터 모델 비교')
    parser.add_argument('--models', nargs='+', 
                       choices=['GBM', 'Heston', 'GARCH', 'Poisson-Gaussian', 'Merton-JD', 'all'],
                       default=['all'],
                       help='비교할 모델 선택 (예: --models GBM Heston 또는 --models all)')
    parser.add_argument('--n_simulations', type=int, default=1000,
                       help='몬테카를로 시뮬레이션 횟수 (기본값: 100)')
    
    args = parser.parse_args()
    
    # 모델 선택 처리
    if 'all' in args.models:
        selected_models = None  # 모든 모델
    else:
        selected_models = args.models
    
    run_model_comparison(n_simulations=args.n_simulations, selected_models=selected_models)
