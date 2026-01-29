"""
Merton Jump-Diffusion 시뮬레이터
dS/S = (μ - λκ)dt + σ dW + (exp(Y)-1)dN
Y ~ N(μ_J, σ_J^2), N ~ Poisson(λ)
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf


def detect_jumps_mad(returns, c=3.0):
    """
    MAD 기반 점프 탐지
    |r_t - median(r)| > c * MAD
    
    Args:
        returns: 로그수익률 시계열
        c: 임계값 배수 (기본값: 3.0)
    
    Returns:
        dict: 점프 탐지 결과
    """
    returns_array = np.array(returns)
    median_ret = np.median(returns_array)
    mad = np.median(np.abs(returns_array - median_ret))
    threshold = c * mad
    
    jump_mask = np.abs(returns_array - median_ret) > threshold
    
    return {
        'jump_mask': jump_mask,
        'jump_times': np.where(jump_mask)[0],
        'jump_returns': returns_array[jump_mask],
        'continuous_returns': returns_array[~jump_mask],
        'threshold': threshold,
        'median': median_ret,
        'mad': mad
    }


def detect_jumps_quantile(returns, quantile=0.95):
    """
    분위수 기반 점프 탐지
    상위 q% 꼬리로 점프일 식별
    
    Args:
        returns: 로그수익률 시계열
        quantile: 상위 분위수 (기본값: 0.95)
    
    Returns:
        dict: 점프 탐지 결과
    """
    returns_array = np.abs(returns)
    threshold = np.quantile(returns_array, quantile)
    jump_mask = returns_array >= threshold
    
    return {
        'jump_mask': jump_mask,
        'jump_times': np.where(jump_mask)[0],
        'jump_returns': returns[jump_mask],
        'continuous_returns': returns[~jump_mask],
        'threshold': threshold
    }


def estimate_merton_jd_parameters(returns, jump_detection_method='quantile', **kwargs):
    """
    Merton Jump-Diffusion 모델 파라미터 추정 (2-step)
    
    Args:
        returns: 로그수익률 시계열
        jump_detection_method: 'mad' 또는 'quantile'
        **kwargs: 점프 탐지 파라미터
    
    Returns:
        dict: 추정된 파라미터
    """
    returns_array = np.array(returns)
    T = len(returns_array)
    dt = 1.0  # 일간
    
    # 1) 점프 탐지
    if jump_detection_method == 'mad':
        c = kwargs.get('c', 3.0)
        jump_result = detect_jumps_mad(returns_array, c=c)
    else:  # quantile
        quantile = kwargs.get('quantile', 0.95)
        jump_result = detect_jumps_quantile(returns_array, quantile=quantile)
    
    jump_mask = jump_result['jump_mask']
    continuous_returns = jump_result['continuous_returns']
    jump_returns = jump_result['jump_returns']
    
    # 2) 연속부로 μ, σ 추정
    mu = np.mean(continuous_returns) / dt
    sigma = np.std(continuous_returns) / np.sqrt(dt)
    
    # 3) λ = (#점프일) / (T*dt)
    n_jumps = np.sum(jump_mask)
    lambda_param = n_jumps / (T * dt)
    
    # 4) 점프 크기: 점프일의 r_t를 점프 샘플로 보고 μ_J, σ_J 적합
    if len(jump_returns) > 0:
        mu_J = np.mean(jump_returns)
        sigma_J = np.std(jump_returns)
    else:
        # 점프가 없으면 기본값 사용
        mu_J = 0.0
        sigma_J = 0.01
    
    # 5) κ = E[exp(Y)-1] = exp(μ_J + 0.5 σ_J^2) - 1
    kappa = np.exp(mu_J + 0.5 * sigma_J**2) - 1
    
    return {
        'mu': mu,
        'sigma': sigma,
        'lambda': lambda_param,
        'mu_J': mu_J,
        'sigma_J': sigma_J,
        'kappa': kappa,
        'n_jumps': n_jumps,
        'jump_result': jump_result
    }


class MertonJDSimulator:
    """
    Merton Jump-Diffusion 시뮬레이터
    dS/S = (μ - λκ)dt + σ dW + (exp(Y)-1)dN
    """
    
    def __init__(self, mu, sigma, lambda_param, mu_J, sigma_J, kappa=None):
        """
        Args:
            mu: 드리프트
            sigma: 연속부 변동성
            lambda_param: 점프 강도
            mu_J: 점프 크기 평균
            sigma_J: 점프 크기 표준편차
            kappa: 점프 보정항 (None이면 자동 계산)
        """
        self.mu = mu
        self.sigma = sigma
        self.lambda_param = lambda_param
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        
        if kappa is None:
            self.kappa = np.exp(mu_J + 0.5 * sigma_J**2) - 1
        else:
            self.kappa = kappa
    
    def simulate_returns(self, T, dt=1.0, seed=None):
        """
        수익률 시뮬레이션
        
        Args:
            T: 시뮬레이션 기간
            dt: 시간 간격 (기본값: 1.0)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 수익률
        """
        if seed is not None:
            np.random.seed(seed)
        
        returns = np.zeros(T)
        
        for t in range(T):
            # 점프 개수: K_t ~ Poisson(λ dt)
            K_t = np.random.poisson(self.lambda_param * dt)
            
            # 점프 크기: J_t ~ Normal(K_t μ_J, K_t σ_J^2)
            if K_t > 0:
                J_t = np.random.normal(K_t * self.mu_J, np.sqrt(K_t) * self.sigma_J)
            else:
                J_t = 0.0
            
            # 연속부: ε_t ~ Normal(0,1)
            epsilon_t = np.random.normal(0, 1)
            
            # 수익률: r_t = (μ - λκ - 0.5 σ^2)dt + σ sqrt(dt) ε_t + J_t
            drift_corrected = (self.mu - self.lambda_param * self.kappa - 0.5 * self.sigma**2) * dt
            continuous_part = self.sigma * np.sqrt(dt) * epsilon_t
            
            returns[t] = drift_corrected + continuous_part + J_t
        
        return pd.Series(returns)
    
    def simulate_nav_path(self, S0, T, dt=1.0, seed=None):
        """
        NAV 경로 시뮬레이션
        
        Args:
            S0: 초기 가격
            T: 시뮬레이션 기간
            dt: 시간 간격 (기본값: 1.0)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 NAV 경로
        """
        returns = self.simulate_returns(T, dt=dt, seed=seed)
        
        # X_hat_t = X_hat_{t-1} * exp(r_hat_t)
        nav_path = np.zeros(T + 1)
        nav_path[0] = S0
        
        for t in range(T):
            nav_path[t + 1] = nav_path[t] * np.exp(returns.iloc[t])
        
        return pd.Series(nav_path[1:])  # t=1부터 반환


def fit_merton_jd_model(returns, jump_detection_method='quantile', **kwargs):
    """
    Merton JD 모델 파라미터 추정 및 시뮬레이터 생성
    
    Args:
        returns: 로그수익률 시계열
        jump_detection_method: 'mad' 또는 'quantile'
        **kwargs: 점프 탐지 파라미터
    
    Returns:
        tuple: (시뮬레이터 인스턴스, 파라미터 딕셔너리)
    """
    params = estimate_merton_jd_parameters(returns, jump_detection_method, **kwargs)
    
    simulator = MertonJDSimulator(
        mu=params['mu'],
        sigma=params['sigma'],
        lambda_param=params['lambda'],
        mu_J=params['mu_J'],
        sigma_J=params['sigma_J'],
        kappa=params['kappa']
    )
    
    return simulator, params
