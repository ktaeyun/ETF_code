"""
연속 성분 모델링 모듈
(a) Normal(μ,σ) 또는 (b) GARCH(1,1)-t로 생성
"""

import numpy as np
import pandas as pd
from arch import arch_model


def fit_normal_model(continuous_returns):
    """
    연속 성분을 정규분포로 모델링
    
    Args:
        continuous_returns: 연속 성분 수익률 시계열
    
    Returns:
        dict: {
            'model_type': 'normal',
            'mu': 평균,
            'sigma': 표준편차
        }
    """
    returns_values = continuous_returns.values
    mu = np.mean(returns_values)
    sigma = np.std(returns_values)
    
    print(f"\n연속 성분 모델링 (정규분포):")
    print(f"  - μ = {mu:.6f}")
    print(f"  - σ = {sigma:.6f}")
    
    return {
        'model_type': 'normal',
        'mu': mu,
        'sigma': sigma
    }


def fit_garch_model(continuous_returns, garch_p=1, garch_q=1, dist='t'):
    """
    연속 성분을 GARCH(1,1)-t로 모델링
    
    Args:
        continuous_returns: 연속 성분 수익률 시계열
        garch_p: GARCH p 파라미터
        garch_q: GARCH q 파라미터
        dist: 분포 ('t' 또는 'normal')
    
    Returns:
        dict: {
            'model_type': 'garch',
            'fitted_model': fitted GARCH 모델,
            'residuals': 표준화 잔차
        }
    """
    print(f"\n연속 성분 모델링 (GARCH({garch_p},{garch_q})-{dist}):")
    
    try:
        model = arch_model(
            continuous_returns * 100,  # arch 라이브러리는 백분율 단위 선호
            vol='GARCH',
            p=garch_p,
            q=garch_q,
            dist=dist
        )
        fitted = model.fit(disp='off')
        
        # 표준화 잔차 추출
        residuals = fitted.resid / fitted.conditional_volatility
        
        print(f"  - 모델 적합 완료")
        print(f"  - BIC: {fitted.bic:.4f}")
        
        return {
            'model_type': 'garch',
            'fitted_model': fitted,
            'residuals': residuals,
            'garch_p': garch_p,
            'garch_q': garch_q,
            'dist': dist
        }
        
    except Exception as e:
        print(f"  - GARCH 모델 적합 실패: {str(e)}")
        # 실패 시 정규분포로 대체
        return fit_normal_model(continuous_returns)
