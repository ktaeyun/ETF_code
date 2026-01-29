"""
Poisson-Gaussian 시뮬레이터
기존 simulator의 NAVSimulator를 재사용
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from simulator.nav_simulator import NAVSimulator
from simulator.jump_detector import detect_jumps, estimate_jump_distribution
from simulator.continuous_component import fit_normal_model, fit_garch_model


def create_poisson_gaussian_simulator(returns_series, continuous_model_type='garch'):
    """
    Poisson-Gaussian 시뮬레이터 생성
    
    Args:
        returns_series: 실제 수익률 시계열
        continuous_model_type: 연속 성분 모델 타입 ('normal' 또는 'garch')
    
    Returns:
        tuple: (NAVSimulator 인스턴스, jump_result)
    """
    # 점프 감지
    jump_result = detect_jumps(returns_series, quantile=0.95)
    
    # 점프 분포 추정
    jump_params = estimate_jump_distribution(jump_result['jump_returns'], dist_type='normal')
    
    # 연속 성분 모델링
    if continuous_model_type == 'normal':
        from simulator.continuous_component import fit_normal_model
        continuous_params = fit_normal_model(jump_result['continuous_returns'])
    else:
        from simulator.continuous_component import fit_garch_model
        continuous_params = fit_garch_model(
            jump_result['continuous_returns'],
            garch_p=1,
            garch_q=1,
            dist='t'
        )
    
    # 시뮬레이터 생성
    simulator = NAVSimulator(
        jump_params=jump_params,
        continuous_params=continuous_params,
        lambda_param=jump_result['lambda']
    )
    
    return simulator, jump_result
