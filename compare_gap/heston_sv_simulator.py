"""
Heston Stochastic Volatility 모델 시뮬레이터
변화형 모델: z_t = mu dt + sqrt(v_t) dW1_t
dv_t = kappa*(theta - v_t)dt + xi*sqrt(v_t) dW2_t, corr(dW1,dW2)=rho
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats


class HestonSVSimulator:
    """
    Heston SV 모델 시뮬레이터
    변화형 모델로 z_t = Δy_t를 직접 생성
    """
    
    def __init__(self, mu, kappa, theta, xi, rho, v0=None):
        """
        Args:
            mu: 드리프트
            kappa: 변동성 평균회귀 속도
            theta: 장기 변동성 평균
            xi: 변동성의 변동성 (volatility of volatility)
            rho: 두 위너 과정의 상관계수
            v0: 초기 변동성 (None이면 theta 사용)
        """
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0 if v0 is not None else theta
    
    def simulate_changes(self, T, dt=1.0, seed=None):
        """
        변화량 z_t = Δy_t 시뮬레이션
        
        Args:
            T: 시뮬레이션 기간
            dt: 시간 간격 (기본값: 1.0)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 변화량 z_t
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 변동성 경로 초기화
        v = np.zeros(T + 1)
        v[0] = self.v0
        
        # 변화량 초기화
        z = np.zeros(T)
        
        # 상관관계가 있는 위너 과정 생성
        for t in range(T):
            # 독립 표준 정규 랜덤 변수
            Z1 = np.random.normal(0, 1)
            Z2 = np.random.normal(0, 1)
            
            # 상관관계가 있는 위너 과정
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            # 변동성 업데이트 (Full Truncation Scheme)
            v_prev = max(v[t], 0)  # 음수 방지
            v_new = v_prev + self.kappa * (self.theta - v_prev) * dt + \
                    self.xi * np.sqrt(v_prev) * np.sqrt(dt) * W2
            
            # 변동성이 음수가 되지 않도록 제약
            v[t + 1] = max(v_new, 0)
            
            # 변화량 생성: z_t = mu*dt + sqrt(v_t) * sqrt(dt) * W1
            sqrt_v = np.sqrt(max(v[t], 0))
            z[t] = self.mu * dt + sqrt_v * np.sqrt(dt) * W1
        
        return pd.Series(z)


def fit_heston_sv_parameters(z_series, method='simple'):
    """
    Heston SV 모델 파라미터 추정 (간단한 방법)
    
    Args:
        z_series: 변화량 시계열 (pd.Series)
        method: 추정 방법 ('simple' 또는 'mle')
    
    Returns:
        dict: {'mu': mu, 'kappa': kappa, 'theta': theta, 'xi': xi, 'rho': rho}
    """
    z = np.array(z_series)
    T = len(z)
    
    if method == 'simple':
        # 간단한 추정 방법
        # mu: 평균
        mu = np.mean(z)
        
        # 변동성 시계열 추정 (rolling window)
        window = min(20, T // 10)
        if window < 5:
            window = 5
        
        squared_z = z**2
        volatility_series = pd.Series(squared_z).rolling(window=window).mean()
        volatility_series = volatility_series.fillna(np.mean(squared_z))
        v_series = np.array(volatility_series)
        
        # 장기 변동성 평균
        theta = np.mean(v_series)
        
        # 변동성의 변동성 (volatility of volatility)
        v_diff = np.diff(v_series)
        xi = np.std(v_diff) * np.sqrt(T) if len(v_diff) > 0 else theta * 0.5
        
        # 평균회귀 속도 (간단한 추정)
        # v_t의 자기상관계수로부터 추정
        if len(v_series) > 1:
            v_lag = v_series[:-1]
            v_next = v_series[1:]
            if np.std(v_lag) > 1e-6:
                autocorr = np.corrcoef(v_lag, v_next)[0, 1]
                # exp(-kappa) ≈ autocorr
                kappa = -np.log(max(0.01, min(0.99, autocorr)))
            else:
                kappa = 0.5
        else:
            kappa = 0.5
        
        # 상관계수 (간단한 추정: z_t와 v_t의 상관관계)
        if len(v_series) == len(z):
            if np.std(z) > 1e-6 and np.std(v_series) > 1e-6:
                rho = np.corrcoef(z, v_series)[0, 1]
                rho = max(-0.99, min(0.99, rho))  # 범위 제한
            else:
                rho = -0.5  # 일반적인 음의 상관관계
        else:
            rho = -0.5
        
        # 초기 변동성
        v0 = v_series[0] if len(v_series) > 0 else theta
        
    elif method == 'mle':
        # 최대우도추정 (간단한 버전)
        # 실제로는 칼만 필터나 입자 필터가 필요하지만, 여기서는 근사 사용
        
        mu = np.mean(z)
        theta = np.var(z)
        
        # 초기값
        initial_params = [0.5, theta, theta * 0.5, -0.5]
        
        def neg_log_likelihood(params):
            kappa, theta_v, xi, rho = params
            if kappa <= 0 or theta_v <= 0 or xi <= 0:
                return 1e10
            
            # 간단한 근사 우도 (실제로는 더 복잡한 필터 필요)
            v_approx = theta_v  # 고정 변동성 근사
            log_likelihood = np.sum(stats.norm.logpdf(z, loc=mu, scale=np.sqrt(v_approx)))
            
            return -log_likelihood
        
        result = minimize(neg_log_likelihood, initial_params, method='L-BFGS-B',
                         bounds=[(1e-6, 10), (1e-6, None), (1e-6, None), (-0.99, 0.99)])
        
        kappa, theta, xi, rho = result.x
        v0 = theta
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 안정성 검사
    if kappa <= 0:
        kappa = 0.5
    if theta <= 0:
        theta = np.var(z)
    if xi <= 0:
        xi = theta * 0.5
    rho = max(-0.99, min(0.99, rho))
    
    print(f"\nHeston SV 파라미터 추정 ({method}):")
    print(f"  mu (드리프트): {mu:.6f}")
    print(f"  kappa (평균회귀 속도): {kappa:.6f}")
    print(f"  theta (장기 변동성 평균): {theta:.6f}")
    print(f"  xi (변동성의 변동성): {xi:.6f}")
    print(f"  rho (상관계수): {rho:.6f}")
    
    return {
        'mu': mu,
        'kappa': kappa,
        'theta': theta,
        'xi': xi,
        'rho': rho,
        'v0': v0 if 'v0' in locals() else theta
    }


def create_heston_sv_simulator(z_series, method='simple'):
    """
    Heston SV 시뮬레이터 생성
    
    Args:
        z_series: 변화량 시계열
        method: 파라미터 추정 방법
    
    Returns:
        tuple: (시뮬레이터 인스턴스, 파라미터 딕셔너리)
    """
    params = fit_heston_sv_parameters(z_series, method=method)
    simulator = HestonSVSimulator(
        mu=params['mu'],
        kappa=params['kappa'],
        theta=params['theta'],
        xi=params['xi'],
        rho=params['rho'],
        v0=params.get('v0', params['theta'])
    )
    return simulator, params
