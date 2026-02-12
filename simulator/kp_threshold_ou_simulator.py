"""
KP Threshold-OU 시뮬레이터
레짐별 OU 모델: 임계값 τ를 기준으로 레짐 분리
외생변수: volume_btc, KOSPI_Volatility → σ_t에 결합
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Optional, Tuple, Dict
from sklearn.metrics import mean_squared_error


def zscore_clip(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """z-score 표준화 후 클리핑"""
    x = np.asarray(x).flatten()
    mean_x = np.mean(x)
    std_x = np.std(x)
    if std_x == 0:
        return np.zeros_like(x)
    z = (x - mean_x) / std_x
    return np.clip(z, -clip, clip)


def get_regime(kp_t: float, threshold: float) -> int:
    """
    레짐 결정
    r=0: |KP_t| ≤ τ
    r=1: KP_t > τ
    r=2: KP_t < -τ
    """
    if abs(kp_t) <= threshold:
        return 0
    elif kp_t > threshold:
        return 1
    else:  # kp_t < -threshold
        return 2


def fit_ou_regime(
    kp_series: np.ndarray,
    regime_mask: np.ndarray,
    regime_id: int
) -> Dict:
    """
    특정 레짐에 대한 OU 파라미터 추정
    
    Args:
        kp_series: KP 시계열
        regime_mask: 레짐 마스크 (boolean array)
        regime_id: 레짐 ID (0, 1, 2)
    
    Returns:
        dict: {'kappa': κ, 'mu': μ, 'sigma0': σ0}
    """
    kp_regime = kp_series[regime_mask]
    if len(kp_regime) < 5:
        # 레짐 데이터가 부족하면 기본값 반환
        return {'kappa': 0.1, 'mu': np.mean(kp_series), 'sigma0': np.std(kp_series)}
    
    # 변화량 계산
    delta_kp = np.diff(kp_regime)
    kp_lag = kp_regime[:-1]
    
    if len(delta_kp) < 3:
        return {'kappa': 0.1, 'mu': np.mean(kp_regime), 'sigma0': np.std(delta_kp) if len(delta_kp) > 0 else 0.01}
    
    # OLS 추정
    X = np.column_stack([np.ones(len(kp_lag)), kp_lag])
    try:
        beta = np.linalg.lstsq(X, delta_kp, rcond=None)[0]
        alpha = beta[0]
        beta_coef = beta[1]
        
        kappa = -beta_coef
        mu = -alpha / beta_coef if beta_coef != 0 else np.mean(kp_regime)
        
        if kappa <= 0:
            kappa = 0.1
        
        residuals = delta_kp - (alpha + beta_coef * kp_lag)
        sigma0 = np.std(residuals)
        if sigma0 <= 0:
            sigma0 = 0.01
    except:
        kappa = 0.1
        mu = np.mean(kp_regime)
        sigma0 = np.std(delta_kp) if len(delta_kp) > 0 else 0.01
    
    # MLE 미세 조정
    def neg_loglik(params):
        k, m, s = params
        if k <= 0 or s <= 0:
            return 1e10
        try:
            loglik = 0.0
            for i in range(len(delta_kp)):
                mean_t = k * (m - kp_lag[i])
                loglik += norm.logpdf(delta_kp[i], loc=mean_t, scale=s)
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
        pass
    
    return {
        'kappa': float(kappa),
        'mu': float(mu),
        'sigma0': float(sigma0)
    }


def fit_threshold_ou(
    kp_series: pd.Series,
    volume_btc: pd.Series,
    kospi_vol: pd.Series,
    threshold: float,
    clip: float = 3.0,
    regularization: float = 0.01
) -> Dict:
    """
    Threshold-OU 모델 적합
    
    Args:
        kp_series: KP 시계열
        volume_btc: volume_btc 시계열
        kospi_vol: KOSPI_Volatility 시계열
        threshold: 임계값 τ
        clip: z-score 클리핑 범위
        regularization: 규제 계수
    
    Returns:
        dict: 레짐별 파라미터 및 외생변수 계수
    """
    kp = np.asarray(kp_series).flatten()
    vol_btc = np.asarray(volume_btc).flatten()
    kospi = np.asarray(kospi_vol).flatten()
    
    T = len(kp)
    if len(vol_btc) != T or len(kospi) != T:
        raise ValueError("시계열 길이 불일치")
    
    # 외생변수 전처리: log(volume_btc) 및 표준화
    log_vol_btc = np.log(vol_btc + 1e-8)
    vol_btc_z = zscore_clip(log_vol_btc, clip=clip)
    kospi_z = zscore_clip(kospi, clip=clip)
    
    # 레짐 마스크 생성
    regime_masks = {
        0: np.abs(kp) <= threshold,
        1: kp > threshold,
        2: kp < -threshold
    }
    
    # 레짐별 OU 파라미터 추정
    params_regime = {}
    for r in [0, 1, 2]:
        params_regime[r] = fit_ou_regime(kp, regime_masks[r], r)
    
    # 외생변수 계수 추정 (레짐별)
    # 각 레짐의 잔차에 대해 회귀
    delta_kp = np.diff(kp)
    kp_lag = kp[:-1]
    vol_btc_lag = vol_btc_z[:-1]
    kospi_lag = kospi_z[:-1]
    
    # 레짐별 잔차 계산
    delta1_regime = {}
    delta2_regime = {}
    
    for r in [0, 1, 2]:
        regime_mask_lag = np.array([get_regime(kp_lag[i], threshold) == r for i in range(len(kp_lag))])
        
        if np.sum(regime_mask_lag) < 5:
            delta1_regime[r] = 0.0
            delta2_regime[r] = 0.0
            continue
        
        # 해당 레짐의 예측값 및 잔차
        kappa_r = params_regime[r]['kappa']
        mu_r = params_regime[r]['mu']
        predicted = kappa_r * (mu_r - kp_lag[regime_mask_lag])
        residuals = delta_kp[regime_mask_lag] - predicted
        residuals_abs = np.abs(residuals)
        
        # log(|잔차|)를 외생변수로 회귀
        X_exog = np.column_stack([
            np.ones(np.sum(regime_mask_lag)),
            vol_btc_lag[regime_mask_lag],
            kospi_lag[regime_mask_lag]
        ])
        
        try:
            log_resid = np.log(residuals_abs + 1e-8)
            beta_exog = np.linalg.lstsq(X_exog, log_resid, rcond=None)[0]
            delta1_regime[r] = beta_exog[1] / 2.0  # exp(2*δ) = exp(δ)^2
            delta2_regime[r] = beta_exog[2] / 2.0
        except:
            delta1_regime[r] = 0.0
            delta2_regime[r] = 0.0
        
        # 범위 제한
        delta1_regime[r] = np.clip(delta1_regime[r], -2.0, 2.0)
        delta2_regime[r] = np.clip(delta2_regime[r], -2.0, 2.0)
    
    return {
        'threshold': threshold,
        'regime_params': params_regime,
        'delta1_regime': delta1_regime,
        'delta2_regime': delta2_regime,
        'vol_btc_mean': float(np.mean(log_vol_btc)),
        'vol_btc_std': float(np.std(log_vol_btc)),
        'kospi_mean': float(np.mean(kospi)),
        'kospi_std': float(np.std(kospi)),
    }


def select_optimal_threshold(
    kp_series: pd.Series,
    volume_btc: pd.Series,
    kospi_vol: pd.Series,
    threshold_candidates: np.ndarray = None,
    clip: float = 3.0
) -> Tuple[float, Dict]:
    """
    최적 임계값 선택 (AIC/BIC 기준)
    
    Args:
        kp_series: KP 시계열
        volume_btc: volume_btc 시계열
        kospi_vol: KOSPI_Volatility 시계열
        threshold_candidates: 임계값 후보 (None이면 자동 생성)
        clip: z-score 클리핑 범위
    
    Returns:
        (최적 임계값, 적합 결과)
    """
    kp = np.asarray(kp_series).flatten()
    
    if threshold_candidates is None:
        # |KP|의 70~90% 분위수 기반 후보 생성
        abs_kp = np.abs(kp)
        q70 = np.percentile(abs_kp, 70)
        q80 = np.percentile(abs_kp, 80)
        q85 = np.percentile(abs_kp, 85)
        q90 = np.percentile(abs_kp, 90)
        threshold_candidates = np.array([q70, q80, q85, q90])
    
    best_threshold = None
    best_bic = np.inf
    best_result = None
    
    for tau in threshold_candidates:
        try:
            result = fit_threshold_ou(kp_series, volume_btc, kospi_vol, tau, clip=clip)
            
            # BIC 계산 (간단한 근사)
            # 각 레짐의 데이터 포인트 수로 패널티
            kp_arr = np.asarray(kp_series)
            n_regime = {
                0: np.sum(np.abs(kp_arr) <= tau),
                1: np.sum(kp_arr > tau),
                2: np.sum(kp_arr < -tau)
            }
            
            # 파라미터 수: 각 레짐당 3개 (κ, μ, σ0) + 외생변수 계수 2개 * 3레짐 = 15개
            n_params = 3 * 3 + 2 * 3  # 15
            
            # 로그 우도 근사 (잔차 제곱합 기반)
            delta_kp = np.diff(kp_arr)
            kp_lag = kp_arr[:-1]
            ss_residual = 0.0
            
            for r in [0, 1, 2]:
                regime_mask = np.array([get_regime(kp_lag[i], tau) == r for i in range(len(kp_lag))])
                if np.sum(regime_mask) > 0:
                    kappa_r = result['regime_params'][r]['kappa']
                    mu_r = result['regime_params'][r]['mu']
                    predicted = kappa_r * (mu_r - kp_lag[regime_mask])
                    residuals = delta_kp[regime_mask] - predicted
                    ss_residual += np.sum(residuals ** 2)
            
            n_total = len(delta_kp)
            if n_total > 0 and ss_residual > 0:
                log_likelihood = -0.5 * n_total * (np.log(2 * np.pi) + np.log(ss_residual / n_total) + 1)
                bic = -2 * log_likelihood + n_params * np.log(n_total)
                
                if bic < best_bic:
                    best_bic = bic
                    best_threshold = tau
                    best_result = result
        except Exception as e:
            continue
    
    if best_threshold is None:
        # 기본값 사용
        abs_kp = np.abs(kp)
        best_threshold = np.percentile(abs_kp, 80)
        best_result = fit_threshold_ou(kp_series, volume_btc, kospi_vol, best_threshold, clip=clip)
    
    return best_threshold, best_result


class KPThresholdOUSimulator:
    """
    KP Threshold-OU 시뮬레이터
    """
    
    def __init__(self, fit_result: Dict, clip: float = 3.0):
        """
        Args:
            fit_result: fit_threshold_ou 또는 select_optimal_threshold의 결과
            clip: 클리핑 범위
        """
        self.threshold = fit_result['threshold']
        self.regime_params = fit_result['regime_params']
        self.delta1_regime = fit_result['delta1_regime']
        self.delta2_regime = fit_result['delta2_regime']
        self.vol_btc_mean = fit_result['vol_btc_mean']
        self.vol_btc_std = fit_result['vol_btc_std']
        self.kospi_mean = fit_result['kospi_mean']
        self.kospi_std = fit_result['kospi_std']
        self.clip = clip
    
    def simulate_kp(
        self,
        T: int,
        kp0: float = None,
        volume_btc_future: np.ndarray = None,
        kospi_vol_future: np.ndarray = None,
        seed: int = None
    ) -> pd.Series:
        """
        KP 경로 시뮬레이션
        
        Args:
            T: 시뮬레이션 기간
            kp0: 초기 KP (None이면 레짐 0의 평균 사용)
            volume_btc_future: 미래 volume_btc 시계열
            kospi_vol_future: 미래 KOSPI_Volatility 시계열
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 KP 경로 (길이 T)
        """
        if seed is not None:
            np.random.seed(seed)
        
        kp = np.zeros(T)
        kp[0] = kp0 if kp0 is not None else self.regime_params[0]['mu']
        
        # 외생변수 처리
        if volume_btc_future is None:
            vol_btc_z = np.zeros(T)
        else:
            vol_btc = np.asarray(volume_btc_future).flatten()
            if len(vol_btc) < T:
                vol_btc = np.pad(vol_btc, (0, T - len(vol_btc)), mode='edge')
            else:
                vol_btc = vol_btc[:T]
            log_vol_btc = np.log(vol_btc + 1e-8)
            vol_btc_z = (log_vol_btc - self.vol_btc_mean) / (self.vol_btc_std + 1e-8)
            vol_btc_z = np.clip(vol_btc_z, -self.clip, self.clip)
        
        if kospi_vol_future is None:
            kospi_z = np.zeros(T)
        else:
            kospi = np.asarray(kospi_vol_future).flatten()
            if len(kospi) < T:
                kospi = np.pad(kospi, (0, T - len(kospi)), mode='edge')
            else:
                kospi = kospi[:T]
            kospi_z = (kospi - self.kospi_mean) / (self.kospi_std + 1e-8)
            kospi_z = np.clip(kospi_z, -self.clip, self.clip)
        
        # Threshold-OU 시뮬레이션
        for t in range(T - 1):
            # 현재 레짐 결정
            r = get_regime(kp[t], self.threshold)
            
            # 레짐별 파라미터
            kappa_r = self.regime_params[r]['kappa']
            mu_r = self.regime_params[r]['mu']
            sigma0_r = self.regime_params[r]['sigma0']
            delta1_r = self.delta1_regime[r]
            delta2_r = self.delta2_regime[r]
            
            # σ_t 계산 (lag 1)
            vol_btc_lag = vol_btc_z[t] if t > 0 else 0.0
            kospi_lag = kospi_z[t] if t > 0 else 0.0
            sigma_t = sigma0_r * np.exp(delta1_r * vol_btc_lag + delta2_r * kospi_lag)
            
            # OU 업데이트
            epsilon_t = np.random.standard_normal()
            kp[t + 1] = kp[t] + kappa_r * (mu_r - kp[t]) + sigma_t * epsilon_t
        
        return pd.Series(kp)


def fit_kp_threshold_ou(
    kp_series: pd.Series,
    volume_btc: pd.Series,
    kospi_vol: pd.Series,
    threshold: float = None,
    clip: float = 3.0,
    regularization: float = 0.01
) -> KPThresholdOUSimulator:
    """
    KP Threshold-OU 모델 적합 및 시뮬레이터 생성
    
    Args:
        kp_series: KP 시계열 (Kimchi Premium)
        volume_btc: volume_btc 시계열
        kospi_vol: KOSPI_Volatility 시계열
        threshold: 임계값 (None이면 자동 선택)
        clip: z-score 클리핑 범위
        regularization: 규제 계수
    
    Returns:
        KPThresholdOUSimulator 인스턴스
    """
    if threshold is None:
        threshold, fit_result = select_optimal_threshold(
            kp_series, volume_btc, kospi_vol, clip=clip
        )
    else:
        fit_result = fit_threshold_ou(
            kp_series, volume_btc, kospi_vol, threshold, clip=clip, regularization=regularization
        )
    
    return KPThresholdOUSimulator(fit_result, clip=clip)
