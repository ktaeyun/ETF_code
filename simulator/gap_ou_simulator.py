"""
GAP OU(Ornstein-Uhlenbeck) 시뮬레이터
외생변수: Search Interest(SI), VIX Volatility(VIX) → σ_t에 결합

모델:
  σ_t = σ_0·exp(δ1·SI_{t-1} + δ2·VIX_{t-1})
  g[t+1] = g[t] + κ·(μ - g[t]) + σ_t·ε_t
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Optional, Tuple


def zscore_clip(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    x = np.asarray(x).flatten()
    mean_x = np.mean(x)
    std_x = np.std(x)
    if std_x == 0:
        return np.zeros_like(x)
    z = (x - mean_x) / std_x
    return np.clip(z, -clip, clip)


def fit_ou_basic(gap_series: pd.Series) -> dict:
    g = np.asarray(gap_series).flatten()
    T = len(g)
    if T < 10:
        raise ValueError("최소 10개 관측 필요")

    delta_g = np.diff(g)
    g_lag = g[:-1]
    X = np.column_stack([np.ones(len(g_lag)), g_lag])
    beta = np.linalg.lstsq(X, delta_g, rcond=None)[0]
    alpha = beta[0]
    beta_coef = beta[1]

    kappa = -beta_coef
    mu = -alpha / beta_coef if beta_coef != 0 else np.mean(g)

    if kappa <= 0:
        kappa = 0.1

    residuals = delta_g - (alpha + beta_coef * g_lag)
    sigma0 = np.std(residuals)

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
        pass

    return {'kappa': float(kappa), 'mu': float(mu), 'sigma0': float(sigma0)}


def fit_ou_with_exog(
    gap_series: pd.Series,
    si_series: pd.Series,
    vix_series: pd.Series,
    clip: float = 3.0,
    regularization: float = 0.01
) -> dict:
    """
    σ_t = σ_0·exp(δ1·SI_{t-1} + δ2·VIX_{t-1})
    """
    g = np.asarray(gap_series).flatten()
    si = np.asarray(si_series).flatten()
    vix = np.asarray(vix_series).flatten()

    T = len(g)
    if len(si) != T or len(vix) != T:
        raise ValueError("시계열 길이 불일치")

    ou_basic = fit_ou_basic(pd.Series(g))
    kappa = ou_basic['kappa']
    mu = ou_basic['mu']
    sigma0_init = ou_basic['sigma0']

    si_z = zscore_clip(si, clip=clip)
    vix_z = zscore_clip(vix, clip=clip)

    si_lag = si_z[:-1]
    vix_lag = vix_z[:-1]
    g_lag = g[:-1]
    delta_g = np.diff(g)

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

    def neg_loglik(params):
        k, m, s0, d1, d2 = params
        if k <= 0 or s0 <= 0:
            return 1e10
        reg_penalty = regularization * (d1**2 + d2**2)
        try:
            loglik = 0.0
            for t in range(len(delta_g)):
                sigma_t = s0 * np.exp(d1 * si_lag[t] + d2 * vix_lag[t])
                if sigma_t <= 0:
                    return 1e10
                mean_t = k * (m - g_lag[t])
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
                (1e-6, 10),
                (None, None),
                (1e-6, None),
                (-2.0, 2.0),
                (-2.0, 2.0)
            ]
        )
        if result.success:
            kappa, mu, sigma0, delta1, delta2 = result.x
        else:
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
    """GAP OU 시뮬레이터 (σ_t = σ_0·exp(δ1·SI + δ2·VIX))"""

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
        if seed is not None:
            np.random.seed(seed)

        g = np.zeros(T)
        g[0] = g0 if g0 is not None else self.mu

        if si_future is None:
            si_future = np.zeros(T)
        else:
            si_future = np.asarray(si_future).flatten()
            if len(si_future) < T:
                si_future = np.pad(si_future, (0, T - len(si_future)), mode='edge')
            else:
                si_future = si_future[:T]
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
            vix_future = (vix_future - self.vix_mean) / (self.vix_std + 1e-8)
            vix_future = np.clip(vix_future, -self.clip, self.clip)

        for t in range(T - 1):
            si_lag = si_future[t] if t > 0 else 0.0
            vix_lag = vix_future[t] if t > 0 else 0.0
            sigma_t = self.sigma0 * np.exp(self.delta1 * si_lag + self.delta2 * vix_lag)
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
    params = fit_ou_with_exog(
        gap_series=gap_series,
        si_series=si_series,
        vix_series=vix_series,
        clip=clip,
        regularization=regularization
    )

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
