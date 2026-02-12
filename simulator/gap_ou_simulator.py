"""
GAP OU(Ornstein-Uhlenbeck) 시뮬레이터
외생변수: Search Interest(SI), VIX Volatility(VIX) → σ_t에 결합
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Optional, Tuple


def zscore_clip(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """
    z-score 표준화 후 클리핑
    
    Args:
        x: 입력 배열
        clip: 클리핑 범위 (±clip)
    
    Returns:
        표준화 및 클리핑된 배열
    """
    x = np.asarray(x).flatten()
    mean_x = np.mean(x)
    std_x = np.std(x)
    if std_x == 0:
        return np.zeros_like(x)
    z = (x - mean_x) / std_x
    return np.clip(z, -clip, clip)


def fit_ou_basic(gap_series: pd.Series) -> dict:
    """
    기본 OU 모델 파라미터 추정 (OLS 또는 MLE)
    
    OU 과정: g_{t+1} = g_t + κ(μ - g_t) + σ0 * ε_t
    
    Args:
        gap_series: GAP 시계열 (etf_premium)
    
    Returns:
        dict: {'kappa': κ, 'mu': μ, 'sigma0': σ0}
    """
    g = np.asarray(gap_series).flatten()
    T = len(g)
    
    if T < 10:
        raise ValueError("최소 10개 관측 필요")
    
    # OLS 추정: g_{t+1} - g_t = κ(μ - g_t) + σ0 * ε_t
    # Δg_t = κ*μ - κ*g_t + σ0*ε_t
    # Δg_t = α + β*g_t + ε_t, 여기서 α = κ*μ, β = -κ
    delta_g = np.diff(g)
    g_lag = g[:-1]
    
    # 선형 회귀: Δg_t = α + β*g_t + ε_t
    X = np.column_stack([np.ones(len(g_lag)), g_lag])
    beta = np.linalg.lstsq(X, delta_g, rcond=None)[0]
    alpha = beta[0]
    beta_coef = beta[1]
    
    # κ = -β, μ = -α/β
    kappa = -beta_coef
    mu = -alpha / beta_coef if beta_coef != 0 else np.mean(g)
    
    # κ > 0 보장 (평균회귀)
    if kappa <= 0:
        kappa = 0.1  # 기본값
    
    # 잔차로 σ0 추정
    residuals = delta_g - (alpha + beta_coef * g_lag)
    sigma0 = np.std(residuals)
    
    # MLE로 미세 조정 (선택적)
    def neg_loglik(params):
        k, m, s = params
        if k <= 0 or s <= 0:
            return 1e10
        try:
            loglik = 0.0
            for t in range(len(delta_g)):
                mean_t = k * (m - g_lag[t])
                loglik += norm.logpdf(delta_g[t], loc=mean_t, scale=s)
            return -loglik
        except:
            return 1e10
    
    try:
        result = minimize(
            neg_loglik,
            x0=[kappa, mu, sigma0],
            method='L-BFGS-B',
            bounds=[(1e-6, 10), (None, None), (1e-6, None)]
        )
        if result.success:
            kappa, mu, sigma0 = result.x
    except:
        pass  # OLS 결과 사용
    
    return {
        'kappa': float(kappa),
        'mu': float(mu),
        'sigma0': float(sigma0)
    }


def fit_ou_with_exog(
    gap_series: pd.Series,
    si_series: pd.Series,
    vix_series: pd.Series,
    clip: float = 3.0,
    regularization: float = 0.01
) -> dict:
    """
    외생변수를 포함한 OU 모델 파라미터 추정
    
    σ_t = σ0 * exp(δ1 * SI_{t-1} + δ2 * VIX_{t-1})
    
    Args:
        gap_series: GAP 시계열
        si_series: Search Interest 시계열
        vix_series: VIX Volatility 시계열
        clip: z-score 클리핑 범위
        regularization: δ1, δ2에 대한 L2 규제 계수
    
    Returns:
        dict: {'kappa': κ, 'mu': μ, 'sigma0': σ0, 'delta1': δ1, 'delta2': δ2}
    """
    g = np.asarray(gap_series).flatten()
    si = np.asarray(si_series).flatten()
    vix = np.asarray(vix_series).flatten()
    
    T = len(g)
    if len(si) != T or len(vix) != T:
        raise ValueError("시계열 길이 불일치")
    
    # 기본 OU 파라미터 추정
    ou_basic = fit_ou_basic(pd.Series(g))
    kappa = ou_basic['kappa']
    mu = ou_basic['mu']
    sigma0_init = ou_basic['sigma0']
    
    # 외생변수 표준화 및 클리핑 (lag 1)
    si_z = zscore_clip(si, clip=clip)
    vix_z = zscore_clip(vix, clip=clip)
    
    # lag 1 정렬 (t-1 사용)
    si_lag = si_z[:-1]  # t=0..T-2
    vix_lag = vix_z[:-1]
    g_lag = g[:-1]  # t=0..T-2
    delta_g = np.diff(g)  # t=1..T-1
    
    # δ1, δ2 초기값: 단순 회귀로 추정
    # log(σ_t^2) ≈ log(σ0^2) + 2*δ1*SI_{t-1} + 2*δ2*VIX_{t-1}
    # 잔차의 절대값을 σ_t의 대리변수로 사용
    residuals_abs = np.abs(delta_g - kappa * (mu - g_lag))
    log_resid = np.log(residuals_abs + 1e-8)
    
    X_exog = np.column_stack([np.ones(len(si_lag)), si_lag, vix_lag])
    try:
        beta_exog = np.linalg.lstsq(X_exog, log_resid, rcond=None)[0]
        delta1_init = beta_exog[1] / 2.0
        delta2_init = beta_exog[2] / 2.0
    except:
        delta1_init = 0.0
        delta2_init = 0.0
    
    # MLE로 전체 파라미터 추정 (규제 포함)
    def neg_loglik(params):
        k, m, s0, d1, d2 = params
        if k <= 0 or s0 <= 0:
            return 1e10
        
        # 규제 항
        reg_penalty = regularization * (d1**2 + d2**2)
        
        try:
            loglik = 0.0
            for t in range(len(delta_g)):
                # σ_t 계산
                sigma_t = s0 * np.exp(d1 * si_lag[t] + d2 * vix_lag[t])
                if sigma_t <= 0:
                    return 1e10
                
                # OU 평균
                mean_t = k * (m - g_lag[t])
                
                # 로그 우도
                loglik += norm.logpdf(delta_g[t], loc=mean_t, scale=sigma_t)
            
            return -(loglik - reg_penalty)
        except:
            return 1e10
    
    try:
        result = minimize(
            neg_loglik,
            x0=[kappa, mu, sigma0_init, delta1_init, delta2_init],
            method='L-BFGS-B',
            bounds=[
                (1e-6, 10),      # kappa
                (None, None),    # mu
                (1e-6, None),    # sigma0
                (-2.0, 2.0),     # delta1 (제한)
                (-2.0, 2.0)      # delta2 (제한)
            ]
        )
        if result.success:
            kappa, mu, sigma0, delta1, delta2 = result.x
        else:
            # 실패 시 기본값 사용
            delta1 = delta1_init
            delta2 = delta2_init
    except:
        delta1 = delta1_init
        delta2 = delta2_init
    
    return {
        'kappa': float(kappa),
        'mu': float(mu),
        'sigma0': float(sigma0),
        'delta1': float(delta1),
        'delta2': float(delta2)
    }


class GapOUSimulator:
    """
    GAP OU 시뮬레이터
    """
    
    def __init__(
        self,
        kappa: float,
        mu: float,
        sigma0: float,
        delta1: float = 0.0,
        delta2: float = 0.0,
        si_mean: float = 0.0,
        si_std: float = 1.0,
        vix_mean: float = 0.0,
        vix_std: float = 1.0,
        clip: float = 3.0
    ):
        """
        Args:
            kappa: 평균회귀 속도
            mu: 장기 평균
            sigma0: 기본 변동성
            delta1: SI 계수
            delta2: VIX 계수
            si_mean, si_std: SI 표준화 파라미터 (시뮬레이션용)
            vix_mean, vix_std: VIX 표준화 파라미터 (시뮬레이션용)
            clip: 클리핑 범위
        """
        self.kappa = kappa
        self.mu = mu
        self.sigma0 = sigma0
        self.delta1 = delta1
        self.delta2 = delta2
        self.si_mean = si_mean
        self.si_std = si_std
        self.vix_mean = vix_mean
        self.vix_std = vix_std
        self.clip = clip
    
    def simulate_gap(
        self,
        T: int,
        g0: float = None,
        si_future: np.ndarray = None,
        vix_future: np.ndarray = None,
        seed: int = None
    ) -> pd.Series:
        """
        GAP 경로 시뮬레이션
        
        Args:
            T: 시뮬레이션 기간
            g0: 초기 GAP (None이면 mu 사용)
            si_future: 미래 SI 시계열 (None이면 평균 사용)
            vix_future: 미래 VIX 시계열 (None이면 평균 사용)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 GAP 경로 (길이 T)
        """
        if seed is not None:
            np.random.seed(seed)
        
        g = np.zeros(T)
        g[0] = g0 if g0 is not None else self.mu
        
        # 외생변수 처리
        if si_future is None:
            si_future = np.zeros(T)
        else:
            si_future = np.asarray(si_future).flatten()
            if len(si_future) < T:
                si_future = np.pad(si_future, (0, T - len(si_future)), mode='edge')
            else:
                si_future = si_future[:T]
            # 표준화 및 클리핑
            si_future = (si_future - self.si_mean) / (self.si_std + 1e-8)
            si_future = np.clip(si_future, -self.clip, self.clip)
        
        if vix_future is None:
            vix_future = np.zeros(T)
        else:
            vix_future = np.asarray(vix_future).flatten()
            if len(vix_future) < T:
                vix_future = np.pad(vix_future, (0, T - len(vix_future)), mode='edge')
            else:
                vix_future = vix_future[:T]
            # 표준화 및 클리핑
            vix_future = (vix_future - self.vix_mean) / (self.vix_std + 1e-8)
            vix_future = np.clip(vix_future, -self.clip, self.clip)
        
        # OU 시뮬레이션
        for t in range(T - 1):
            # σ_t 계산 (lag 1 사용)
            si_lag = si_future[t] if t > 0 else 0.0
            vix_lag = vix_future[t] if t > 0 else 0.0
            sigma_t = self.sigma0 * np.exp(self.delta1 * si_lag + self.delta2 * vix_lag)
            
            # OU 업데이트
            epsilon_t = np.random.standard_normal()
            g[t + 1] = g[t] + self.kappa * (self.mu - g[t]) + sigma_t * epsilon_t
        
        return pd.Series(g)


def fit_gap_ou(
    gap_series: pd.Series,
    si_series: pd.Series,
    vix_series: pd.Series,
    clip: float = 3.0,
    regularization: float = 0.01
) -> GapOUSimulator:
    """
    GAP OU 모델 적합 및 시뮬레이터 생성
    
    Args:
        gap_series: GAP 시계열 (etf_premium)
        si_series: Search Interest 시계열
        vix_series: VIX Volatility 시계열
        clip: z-score 클리핑 범위
        regularization: 규제 계수
    
    Returns:
        GapOUSimulator 인스턴스
    """
    params = fit_ou_with_exog(
        gap_series=gap_series,
        si_series=si_series,
        vix_series=vix_series,
        clip=clip,
        regularization=regularization
    )
    
    # 표준화 파라미터 저장 (시뮬레이션용)
    si_mean = np.mean(si_series)
    si_std = np.std(si_series)
    vix_mean = np.mean(vix_series)
    vix_std = np.std(vix_series)
    
    return GapOUSimulator(
        kappa=params['kappa'],
        mu=params['mu'],
        sigma0=params['sigma0'],
        delta1=params['delta1'],
        delta2=params['delta2'],
        si_mean=si_mean,
        si_std=si_std,
        vix_mean=vix_mean,
        vix_std=vix_std,
        clip=clip
    )
