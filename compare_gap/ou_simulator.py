"""
Ornstein-Uhlenbeck (OU) 모델 시뮬레이터
수준형 모델: dy_t = kappa*(theta - y_t)dt + sigma dW_t
이산화: y_{t+1} = theta + (y_t - theta)exp(-kappa) + eps_t
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats


class OUSimulator:
    """
    OU 모델 시뮬레이터
    수준형 모델로 y_t를 생성한 후 Δy_t로 변환
    """
    
    def __init__(self, kappa, theta, sigma, y0=None):
        """
        Args:
            kappa: 평균회귀 속도
            theta: 장기 평균
            sigma: 변동성
            y0: 초기값 (None이면 theta 사용)
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.y0 = y0 if y0 is not None else theta
    
    def simulate_level(self, T, dt=1.0, seed=None):
        """
        수준 y_t 시뮬레이션
        
        Args:
            T: 시뮬레이션 기간
            dt: 시간 간격 (기본값: 1.0)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 수준 y_t
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 이산화된 OU 과정
        # y_{t+1} = theta + (y_t - theta)exp(-kappa*dt) + eps_t
        # eps_t ~ N(0, sigma^2 * (1 - exp(-2*kappa*dt)) / (2*kappa))
        
        y = np.zeros(T + 1)
        y[0] = self.y0
        
        # 분산 계산
        var_eps = self.sigma**2 * (1 - np.exp(-2 * self.kappa * dt)) / (2 * self.kappa)
        std_eps = np.sqrt(var_eps)
        
        for t in range(T):
            # 평균회귀 항
            mean_reversion = self.theta + (y[t] - self.theta) * np.exp(-self.kappa * dt)
            # 확률적 항
            noise = np.random.normal(0, std_eps)
            y[t + 1] = mean_reversion + noise
        
        return pd.Series(y[1:])  # t=1부터 반환
    
    def simulate_changes(self, T, dt=1.0, seed=None):
        """
        변화량 Δy_t 시뮬레이션
        
        Args:
            T: 시뮬레이션 기간
            dt: 시간 간격
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 변화량 Δy_t
        """
        y_level = self.simulate_level(T, dt, seed)
        # Δy_t = y_t - y_{t-1}
        changes = y_level.diff().dropna()
        return changes


def fit_ou_parameters(y_series, method='ols'):
    """
    OU 모델 파라미터 추정
    
    Args:
        y_series: 수준 시계열 (pd.Series)
        method: 추정 방법 ('ols' 또는 'mle')
    
    Returns:
        dict: {'kappa': kappa, 'theta': theta, 'sigma': sigma}
    """
    y = np.array(y_series)
    T = len(y)
    
    if method == 'ols':
        # OLS 회귀: y_{t+1} = a + b*y_t + eps_t
        # a = theta*(1 - exp(-kappa)), b = exp(-kappa)
        # kappa = -log(b), theta = a/(1-b), sigma는 잔차로 추정
        
        y_lag = y[:-1]
        y_next = y[1:]
        
        # OLS 회귀
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        beta = np.linalg.lstsq(X, y_next, rcond=None)[0]
        a, b = beta[0], beta[1]
        
        # 파라미터 변환
        if b >= 1 or b <= 0:
            # 안정성 보장
            b = max(0.01, min(0.99, b))
        
        kappa = -np.log(b)
        theta = a / (1 - b) if abs(1 - b) > 1e-6 else np.mean(y)
        
        # 잔차로 sigma 추정
        residuals = y_next - (a + b * y_lag)
        sigma_sq = np.var(residuals)
        # 연속시간 변환: sigma^2 = 2*kappa*sigma_disc^2 / (1 - exp(-2*kappa))
        if kappa > 1e-6:
            sigma = np.sqrt(sigma_sq * 2 * kappa / (1 - np.exp(-2 * kappa)))
        else:
            sigma = np.sqrt(sigma_sq)
        
    elif method == 'mle':
        # 최대우도추정
        def neg_log_likelihood(params):
            kappa, theta, sigma = params
            if kappa <= 0 or sigma <= 0:
                return 1e10
            
            y_lag = y[:-1]
            y_next = y[1:]
            
            # OU 이산화: y_{t+1} = theta + (y_t - theta)exp(-kappa) + eps_t
            # eps_t ~ N(0, sigma^2 * (1 - exp(-2*kappa)) / (2*kappa))
            var_eps = sigma**2 * (1 - np.exp(-2 * kappa)) / (2 * kappa)
            if var_eps <= 0:
                return 1e10
            
            mean_t = theta + (y_lag - theta) * np.exp(-kappa)
            log_likelihood = np.sum(stats.norm.logpdf(y_next, loc=mean_t, scale=np.sqrt(var_eps)))
            
            return -log_likelihood
        
        # 초기값 설정
        y_mean = np.mean(y)
        y_std = np.std(y)
        initial_params = [0.1, y_mean, y_std]
        
        # 최적화
        result = minimize(neg_log_likelihood, initial_params, method='L-BFGS-B',
                         bounds=[(1e-6, 10), (None, None), (1e-6, None)])
        
        kappa, theta, sigma = result.x
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 안정성 검사
    if kappa <= 0:
        kappa = 0.1
    if sigma <= 0:
        sigma = np.std(y) * 0.1
    
    print(f"\nOU 파라미터 추정 ({method}):")
    print(f"  kappa (평균회귀 속도): {kappa:.6f}")
    print(f"  theta (장기 평균): {theta:.6f}")
    print(f"  sigma (변동성): {sigma:.6f}")
    
    return {
        'kappa': kappa,
        'theta': theta,
        'sigma': sigma
    }


def create_ou_simulator(y_series, method='ols'):
    """
    OU 시뮬레이터 생성
    
    Args:
        y_series: 수준 시계열
        method: 파라미터 추정 방법
    
    Returns:
        tuple: (시뮬레이터 인스턴스, 파라미터 딕셔너리)
    """
    params = fit_ou_parameters(y_series, method=method)
    simulator = OUSimulator(
        kappa=params['kappa'],
        theta=params['theta'],
        sigma=params['sigma'],
        y0=y_series.iloc[0] if len(y_series) > 0 else None
    )
    return simulator, params
