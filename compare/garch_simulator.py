"""
GARCH 시뮬레이터
기존 continuous_component의 GARCH 모델을 활용
"""

import numpy as np
import pandas as pd
from arch import arch_model


class GARCHSimulator:
    """
    GARCH 시뮬레이터 클래스
    """
    
    def __init__(self, fitted_garch_model):
        """
        Args:
            fitted_garch_model: 적합된 GARCH 모델 (arch 라이브러리)
        """
        self.fitted_model = fitted_garch_model
    
    def simulate_returns(self, T, seed=None):
        """
        GARCH 모델에서 수익률 시뮬레이션
        
        Args:
            T: 시뮬레이션 기간 (일수)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 수익률
        """
        if seed is not None:
            np.random.seed(seed)
        
        fitted = self.fitted_model
        
        # GARCH 파라미터 추출
        omega = fitted.params['omega']
        alpha = fitted.params.get('alpha[1]', 0.1)
        beta = fitted.params.get('beta[1]', 0.8)
        
        # 분포 파라미터
        if 'nu' in fitted.params.index:
            nu = fitted.params['nu']
        else:
            nu = None
        
        # 초기값 (백분율 단위에서 원래 단위로 변환)
        initial_var = np.var(fitted.resid) / 10000
        sigma2 = initial_var
        returns = np.zeros(T)
        
        for t in range(T):
            # 분산 업데이트
            if t == 0:
                sigma2_t = sigma2
            else:
                sigma2_t = omega + alpha * (returns[t-1] ** 2) + beta * sigma2
            
            # 수익률 생성
            if nu is not None:
                # t-분포
                z = np.random.standard_t(nu)
            else:
                # 정규분포
                z = np.random.normal(0, 1)
            
            returns[t] = np.sqrt(sigma2_t) * z / 100  # 백분율 단위 조정
            
            # 다음 반복을 위한 분산 업데이트
            sigma2 = sigma2_t
        
        return pd.Series(returns)
    
    def simulate_nav_path(self, S0, T, seed=None):
        """
        GARCH NAV 경로 시뮬레이션
        
        Args:
            S0: 초기 NAV
            T: 시뮬레이션 기간 (일수)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 NAV 경로
        """
        returns = self.simulate_returns(T, seed)
        
        # NAV 경로 생성
        nav_path = np.zeros(T + 1)
        nav_path[0] = S0
        
        for t in range(T):
            nav_path[t + 1] = nav_path[t] * np.exp(returns.iloc[t])
        
        return pd.Series(nav_path[1:], index=range(T))


def fit_garch_model(returns_series, garch_p=1, garch_q=1, dist='t'):
    """
    실제 수익률 데이터로부터 GARCH 모델 적합
    
    Args:
        returns_series: 실제 수익률 시계열
        garch_p: GARCH p 파라미터
        garch_q: GARCH q 파라미터
        dist: 분포 ('t' 또는 'normal')
    
    Returns:
        fitted GARCH 모델
    """
    print(f"\nGARCH({garch_p},{garch_q})-{dist} 모델 적합 중...")
    
    try:
        model = arch_model(
            returns_series * 100,  # arch 라이브러리는 백분율 단위 선호
            vol='GARCH',
            p=garch_p,
            q=garch_q,
            dist=dist
        )
        fitted = model.fit(disp='off')
        
        print(f"  - 모델 적합 완료")
        print(f"  - BIC: {fitted.bic:.4f}")
        
        return fitted
        
    except Exception as e:
        print(f"  - GARCH 모델 적합 실패: {str(e)}")
        raise
