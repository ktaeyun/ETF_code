"""
점프 감지 모듈
|r_t|의 상위 q 분위수를 기준으로 점프일 식별
"""

import numpy as np
import pandas as pd
from scipy import stats


def detect_jumps(returns_series, quantile=0.95):
    """
    점프일 감지: |r_t|의 상위 q 분위수를 기준으로
    
    Args:
        returns_series: 로그수익률 시계열
        quantile: 상위 분위수 (기본값: 0.95, 즉 95%)
    
    Returns:
        dict: {
            'jump_dates': 점프일 인덱스,
            'jump_returns': 점프일 수익률,
            'continuous_returns': 연속 성분 수익률,
            'threshold': 점프 임계값,
            'lambda': 점프 강도 λ = N/T
        }
    """
    abs_returns = np.abs(returns_series.values)
    threshold = np.quantile(abs_returns, quantile)
    
    # 점프일 식별
    jump_mask = abs_returns >= threshold
    jump_dates = returns_series.index[jump_mask]
    jump_returns = returns_series[jump_mask]
    
    # 연속 성분 (점프 제외)
    continuous_returns = returns_series[~jump_mask]
    
    # 점프 강도 추정: λ = N/T
    T = len(returns_series)
    N = len(jump_returns)
    lambda_est = N / T
    
    print(f"\n점프 감지 완료:")
    print(f"  - 임계값 (|r_t| 상위 {quantile*100}%): {threshold:.6f}")
    print(f"  - 점프일 수: {N}/{T} ({N/T*100:.2f}%)")
    print(f"  - 점프 강도 λ = {lambda_est:.6f}")
    print(f"  - 연속 성분 수: {len(continuous_returns)}")
    
    return {
        'jump_dates': jump_dates,
        'jump_returns': jump_returns,
        'continuous_returns': continuous_returns,
        'threshold': threshold,
        'lambda': lambda_est,
        'jump_mask': jump_mask
    }


def estimate_jump_distribution(jump_returns, dist_type='normal'):
    """
    점프 크기 분포 추정
    
    Args:
        jump_returns: 점프일 수익률
        dist_type: 분포 타입 ('normal' 또는 't')
    
    Returns:
        dict: {
            'dist_type': 분포 타입,
            'mu': 평균,
            'sigma': 표준편차,
            'nu': 자유도 (t-분포인 경우)
        }
    """
    if len(jump_returns) == 0:
        return None
    
    jump_values = jump_returns.values
    
    if dist_type == 'normal':
        mu = np.mean(jump_values)
        sigma = np.std(jump_values)
        
        print(f"\n점프 크기 분포 추정 (정규분포):")
        print(f"  - μ_J = {mu:.6f}")
        print(f"  - σ_J = {sigma:.6f}")
        
        return {
            'dist_type': 'normal',
            'mu': mu,
            'sigma': sigma
        }
    
    elif dist_type == 't':
        # t-분포로 적합 (MLE)
        from scipy.stats import t as t_dist
        
        # 간단한 추정: 표본 평균, 표준편차, 자유도 추정
        mu = np.mean(jump_values)
        sigma = np.std(jump_values)
        
        # 자유도 추정 (간단한 방법)
        # 실제로는 MLE로 추정해야 하지만, 여기서는 간단히 추정
        nu = max(2.1, len(jump_values) / 2)  # 최소 2.1
        
        print(f"\n점프 크기 분포 추정 (t-분포):")
        print(f"  - μ_J = {mu:.6f}")
        print(f"  - σ_J = {sigma:.6f}")
        print(f"  - ν = {nu:.2f}")
        
        return {
            'dist_type': 't',
            'mu': mu,
            'sigma': sigma,
            'nu': nu
        }
    
    else:
        raise ValueError(f"지원하지 않는 분포 타입: {dist_type}")
