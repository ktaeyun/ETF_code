"""
GBM (Geometric Brownian Motion) 시뮬레이터
dS = μS dt + σS dW
"""

import numpy as np
import pandas as pd


class GBMSimulator:
    """
    GBM 시뮬레이터 클래스
    """
    
    def __init__(self, mu, sigma):
        """
        Args:
            mu: 드리프트 파라미터 (연간)
            sigma: 변동성 파라미터 (연간)
        """
        self.mu = mu
        self.sigma = sigma
    
    def simulate_returns(self, T, dt=1/252, seed=None):
        """
        GBM 수익률 시뮬레이션
        
        Args:
            T: 시뮬레이션 기간 (일수)
            dt: 시간 간격 (기본값: 1일 = 1/252년)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 수익률
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 일일 수익률: r_t = (μ - σ²/2)dt + σ√dt * Z
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        Z = np.random.normal(0, 1, T)
        returns = drift + diffusion * Z
        
        return pd.Series(returns)
    
    def simulate_nav_path(self, S0, T, dt=1/252, seed=None):
        """
        GBM NAV 경로 시뮬레이션: S_{t+1} = S_t * exp(r_t)
        
        Args:
            S0: 초기 NAV
            T: 시뮬레이션 기간 (일수)
            dt: 시간 간격
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 NAV 경로
        """
        returns = self.simulate_returns(T, dt, seed)
        
        # NAV 경로 생성
        nav_path = np.zeros(T + 1)
        nav_path[0] = S0
        
        for t in range(T):
            nav_path[t + 1] = nav_path[t] * np.exp(returns.iloc[t])
        
        return pd.Series(nav_path[1:], index=range(T))


def fit_gbm_parameters(returns_series):
    """
    실제 수익률 데이터로부터 GBM 파라미터 추정
    
    Args:
        returns_series: 실제 수익률 시계열
    
    Returns:
        dict: {'mu': 연간 드리프트, 'sigma': 연간 변동성}
    """
    returns_values = returns_series.values
    
    # 일일 평균 수익률과 표준편차
    daily_mean = np.mean(returns_values)
    daily_std = np.std(returns_values)
    
    # 연간화 (거래일 252일 가정)
    annual_mu = daily_mean * 252
    annual_sigma = daily_std * np.sqrt(252)
    
    print(f"\nGBM 파라미터 추정:")
    print(f"  - 일일 평균 수익률: {daily_mean:.6f}")
    print(f"  - 일일 표준편차: {daily_std:.6f}")
    print(f"  - 연간 드리프트 (μ): {annual_mu:.6f}")
    print(f"  - 연간 변동성 (σ): {annual_sigma:.6f}")
    
    return {
        'mu': annual_mu,
        'sigma': annual_sigma
    }
