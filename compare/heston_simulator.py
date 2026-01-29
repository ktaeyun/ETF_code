"""
Heston 확률변동성 모델 시뮬레이터
dS = μS dt + √V S dW1
dV = κ(θ - V) dt + σ_v √V dW2
"""

import numpy as np
import pandas as pd


class HestonSimulator:
    """
    Heston 확률변동성 모델 시뮬레이터 클래스
    """
    
    def __init__(self, mu, kappa, theta, sigma_v, rho, v0):
        """
        Args:
            mu: 드리프트 파라미터 (연간)
            kappa: 변동성 평균회귀 속도
            theta: 장기 변동성 평균
            sigma_v: 변동성의 변동성 (volatility of volatility)
            rho: 두 위너 과정의 상관계수
            v0: 초기 변동성
        """
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0
    
    def simulate_returns(self, T, dt=1/252, seed=None):
        """
        Heston 모델 수익률 시뮬레이션
        
        Args:
            T: 시뮬레이션 기간 (일수)
            dt: 시간 간격 (기본값: 1일 = 1/252년)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 수익률
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 변동성 경로 초기화
        V = np.zeros(T + 1)
        V[0] = self.v0
        
        # 수익률 초기화
        returns = np.zeros(T)
        
        # 상관관계가 있는 위너 과정 생성
        for t in range(T):
            # 상관관계가 있는 표준 정규 랜덤 변수
            Z1 = np.random.normal(0, 1)
            Z2 = np.random.normal(0, 1)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            # 변동성 업데이트 (Feller 조건 확인)
            V[t+1] = V[t] + self.kappa * (self.theta - V[t]) * dt + \
                     self.sigma_v * np.sqrt(max(V[t], 0)) * np.sqrt(dt) * W2
            
            # 변동성이 음수가 되지 않도록 제약
            V[t+1] = max(V[t+1], 0)
            
            # 수익률 생성
            drift = (self.mu - 0.5 * V[t]) * dt
            diffusion = np.sqrt(max(V[t], 0)) * np.sqrt(dt) * W1
            returns[t] = drift + diffusion
        
        return pd.Series(returns)
    
    def simulate_nav_path(self, S0, T, dt=1/252, seed=None):
        """
        Heston NAV 경로 시뮬레이션
        
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


def fit_heston_parameters(returns_series):
    """
    실제 수익률 데이터로부터 Heston 파라미터 추정 (간단한 방법)
    
    Args:
        returns_series: 실제 수익률 시계열
    
    Returns:
        dict: Heston 파라미터
    """
    returns_values = returns_series.values
    
    # 일일 평균 수익률과 표준편차
    daily_mean = np.mean(returns_values)
    daily_std = np.std(returns_values)
    
    # 연간화
    annual_mu = daily_mean * 252
    annual_sigma = daily_std * np.sqrt(252)
    
    # 변동성 시계열 (rolling window 사용)
    window = min(20, len(returns_values) // 10)
    squared_returns = returns_values ** 2
    volatility_series = pd.Series(squared_returns).rolling(window=window).mean() * 252
    
    # 변동성의 변동성 추정
    vol_of_vol = np.std(np.diff(volatility_series.dropna())) * np.sqrt(252)
    
    # Heston 파라미터 추정 (간단한 방법)
    kappa = 2.0  # 평균회귀 속도 (고정값 또는 추정)
    theta = annual_sigma**2  # 장기 변동성 평균
    sigma_v = vol_of_vol if not np.isnan(vol_of_vol) else annual_sigma * 0.5
    rho = -0.5  # 일반적인 음의 상관관계
    v0 = annual_sigma**2  # 초기 변동성
    
    print(f"\nHeston 파라미터 추정:")
    print(f"  - 연간 드리프트 (μ): {annual_mu:.6f}")
    print(f"  - 평균회귀 속도 (κ): {kappa:.4f}")
    print(f"  - 장기 변동성 평균 (θ): {theta:.6f}")
    print(f"  - 변동성의 변동성 (σ_v): {sigma_v:.6f}")
    print(f"  - 상관계수 (ρ): {rho:.4f}")
    print(f"  - 초기 변동성 (v0): {v0:.6f}")
    
    return {
        'mu': annual_mu,
        'kappa': kappa,
        'theta': theta,
        'sigma_v': sigma_v,
        'rho': rho,
        'v0': v0
    }
