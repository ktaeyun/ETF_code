"""
NAV 시뮬레이터 메인 실행 스크립트
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
from simulator.data_loader import load_nav_data
from simulator.jump_detector import detect_jumps, estimate_jump_distribution
from simulator.continuous_component import fit_normal_model, fit_garch_model
from simulator.nav_simulator import NAVSimulator, create_simulator
from simulator.validator import validate_simulation
from simulator.visualizer import create_all_visualizations


def main():
    """
    메인 실행 함수
    """
    print("=" * 60)
    print("NAV 시뮬레이터 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드")
    print("-" * 60)
    nav_series, returns_series = load_nav_data()
    
    # 2. 점프 감지
    print("\n[2단계] 점프 감지")
    print("-" * 60)
    jump_result = detect_jumps(returns_series, quantile=0.95)
    
    # 3. 점프 분포 추정
    print("\n[3단계] 점프 분포 추정")
    print("-" * 60)
    jump_params = estimate_jump_distribution(jump_result['jump_returns'], dist_type='normal')
    
    # 4. 연속 성분 모델링
    print("\n[4단계] 연속 성분 모델링")
    print("-" * 60)
    
    # 모델 선택: 'normal' 또는 'garch'
    continuous_model_type = 'garch'  # 또는 'normal'
    
    if continuous_model_type == 'normal':
        continuous_params = fit_normal_model(jump_result['continuous_returns'])
    else:
        continuous_params = fit_garch_model(
            jump_result['continuous_returns'],
            garch_p=1,
            garch_q=1,
            dist='t'
        )
    
    # 5. 시뮬레이터 생성
    print("\n[5단계] 시뮬레이터 생성")
    print("-" * 60)
    simulator = NAVSimulator(
        jump_params=jump_params,
        continuous_params=continuous_params,
        lambda_param=jump_result['lambda']
    )
    
    # 6. 시뮬레이션 실행
    print("\n[6단계] 시뮬레이션 실행")
    print("-" * 60)
    T = len(returns_series)
    S0 = nav_series.iloc[0]
    
    simulated_returns = simulator.simulate_returns(T, seed=42)
    simulated_nav = simulator.simulate_nav_path(S0, T, seed=42)
    
    print(f"  시뮬레이션 완료: {T}일")
    print(f"  초기 NAV: {S0:.2f}")
    print(f"  최종 NAV: {simulated_nav.iloc[-1]:.2f}")
    
    # 7. 검증
    print("\n[7단계] 시뮬레이션 결과 검증")
    print("-" * 60)
    
    # 점프 시점 추출 (시뮬레이션)
    simulated_jump_mask = np.abs(simulated_returns) >= jump_result['threshold']
    simulated_jump_times = np.where(simulated_jump_mask)[0]
    actual_jump_times = np.where(jump_result['jump_mask'])[0]
    
    validation_results = validate_simulation(
        simulated_returns,
        returns_series.values,
        simulated_jump_times=simulated_jump_times,
        actual_jump_times=actual_jump_times
    )
    
    # 8. 시각화 생성
    print("\n[8단계] 시각화 생성")
    print("-" * 60)
    
    results_dir = Path(settings.PROJECT_ROOT) / 'results' / 'simulator_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 실제 NAV (수익률과 맞추기 위해 첫 번째 제외)
    actual_nav_aligned = nav_series.iloc[1:].values
    
    create_all_visualizations(
        actual_nav=actual_nav_aligned,
        simulated_nav=simulated_nav.values,
        actual_returns=returns_series.values,
        simulated_returns=simulated_returns.values,
        actual_jump_times=actual_jump_times,
        simulated_jump_times=simulated_jump_times,
        output_dir=results_dir / 'plots'
    )
    
    # 9. 결과 저장
    print("\n[9단계] 결과 저장")
    print("-" * 60)
    
    # 시뮬레이션 결과 저장
    sim_results_df = pd.DataFrame({
        'simulated_returns': simulated_returns.values,
        'simulated_nav': simulated_nav.values,
        'actual_returns': returns_series.values,
        'actual_nav': actual_nav_aligned
    })
    
    sim_results_path = results_dir / 'simulation_results.csv'
    sim_results_df.to_csv(sim_results_path, index=False, encoding='utf-8-sig')
    print(f"  - {sim_results_path}")
    
    # 검증 결과 저장
    import json
    validation_path = results_dir / 'validation_results.json'
    with open(validation_path, 'w', encoding='utf-8') as f:
        # numpy 타입을 JSON 직렬화 가능하게 변환
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_to_serializable(validation_results), f, ensure_ascii=False, indent=2)
    
    print(f"  - {validation_path}")
    
    print("\n" + "=" * 60)
    print("NAV 시뮬레이터 완료")
    print("=" * 60)


if __name__ == '__main__':
    main()
