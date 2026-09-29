"""
GAP OU(Ornstein-Uhlenbeck) 시뮬레이터
외생변수: Search Interest(SI), VIX Volatility(VIX) → σ_t에 결합

모델:
  σ_t = σ_0·exp(δ1·SI_{t-1} + δ2·VIX_{t-1})
  g[t+1] = g[t] + κ·(μ - g[t]) + σ_t·ε_t,   E[ε]=0, Var(ε)=1

혁신항 ε는 정규 또는 표준화 Student-t를 쓴다. 실제 ETF 괴리율은 정규보다
꼬리가 두꺼워(초과첨도 +2.15 대 모형 +0.13), 분산은 맞아도 분포 모양이
어긋난다. t 혁신항은 σ를 건드리지 않고 그 모양만 교정한다.

  ε = T_ν / sqrt(ν/(ν-2))   -> E[ε]=0, Var(ε)=1 (ν>2)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, t as student_t
from typing import Optional, Tuple

# 자유도 탐색 범위. 상한에 붙으면 사실상 정규분포와 구분되지 않는다.
NU_MIN, NU_MAX = 2.1, 50.0


def _t_scale(nu: float) -> float:
    """분산이 1이 되도록 표준화하는 스케일: T_ν 를 이 값으로 나누면 Var=1"""
    return np.sqrt(nu / (nu - 2.0))


def standardized_t(nu: float, size=None, rng=None):
    """평균 0, 분산 1로 표준화된 Student-t 난수"""
    draw = (rng.standard_t(nu, size=size) if rng is not None
            else np.random.standard_t(nu, size=size))
    return draw / _t_scale(nu)


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
    regularization: float = 0.01,
    dist: str = "t"
) -> dict:
    """
    σ_t = σ_0·exp(δ1·SI_{t-1} + δ2·VIX_{t-1})

    Args:
        dist: "t"이면 자유도 ν를 함께 최대우도 추정, "normal"이면 정규 혁신항.
              ν는 [NU_MIN, NU_MAX]에서 찾으며, 상한에 붙으면 정규와 사실상 같다.

    Returns:
        dict: kappa, mu, sigma0, delta1, delta2, nu (정규면 nu=None)
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

    use_t = (dist == "t")

    def _neg_loglik(params, nu):
        """5개 모수의 음의 로그우도. nu가 None이면 정규, 아니면 표준화 t."""
        k, m, s0, d1, d2 = params
        if k <= 0 or s0 <= 0:
            return 1e10
        sigma_t = s0 * np.exp(d1 * si_lag + d2 * vix_lag)
        if not np.all(np.isfinite(sigma_t)) or np.any(sigma_t <= 0):
            return 1e10
        mean_t = k * (m - g_lag)
        if nu is None:
            ll = norm.logpdf(delta_g, loc=mean_t, scale=sigma_t)
        else:
            # ε = T_ν / sqrt(ν/(ν-2)) 이므로 delta_g의 t-스케일은 σ_t/_t_scale(ν)
            ll = student_t.logpdf(delta_g, df=nu, loc=mean_t, scale=sigma_t / _t_scale(nu))
        if not np.all(np.isfinite(ll)):
            return 1e10
        return -(float(np.sum(ll)) - regularization * (d1 ** 2 + d2 ** 2))

    bounds = [(1e-6, 10), (None, None), (1e-6, None), (-2.0, 2.0), (-2.0, 2.0)]

    def _fit_params(nu, x0):
        try:
            r = minimize(_neg_loglik, x0=x0, args=(nu,), method='L-BFGS-B', bounds=bounds)
            if r.success and np.all(np.isfinite(r.x)):
                return list(r.x), float(r.fun)
        except Exception:
            pass
        return list(x0), _neg_loglik(x0, nu)

    def _fit_nu(params):
        """
        표준화 잔차에서 자유도만 1-D로 추정한다.
        6개 모수를 한꺼번에 L-BFGS-B에 넣으면 스케일 차이(κ~1 대 σ0~0.004 대 ν~8)
        때문에 ν 방향의 수치미분이 묻혀 초기값에서 움직이지 않는다.
        """
        k, m, s0, d1, d2 = params
        sigma_t = s0 * np.exp(d1 * si_lag + d2 * vix_lag)
        z = (delta_g - k * (m - g_lag)) / sigma_t
        z = z[np.isfinite(z)]
        if z.size < 20:
            return None

        def neg(nu):
            ll = student_t.logpdf(z, df=nu, scale=1.0 / _t_scale(nu))
            return 1e10 if not np.all(np.isfinite(ll)) else -float(np.sum(ll))

        try:
            r = minimize_scalar(neg, bounds=(NU_MIN, NU_MAX), method='bounded')
            return float(r.x) if r.success else None
        except Exception:
            return None

    # 1단계: 정규 가정으로 5개 모수를 잡는다
    x0 = [kappa, mu, sigma0_init, delta1_init, delta2_init]
    params, _ = _fit_params(None, x0)

    # 2단계: (ν | 모수) 와 (모수 | ν) 를 번갈아 갱신한다
    nu_hat = None
    if use_t:
        for _ in range(3):
            nu_new = _fit_nu(params)
            if nu_new is None:
                break
            params_new, _ = _fit_params(nu_new, params)
            converged = (nu_hat is not None and abs(nu_new - nu_hat) < 1e-3)
            nu_hat, params = nu_new, params_new
            if converged:
                break

    kappa, mu, sigma0, delta1, delta2 = params
    return {
        'kappa': float(kappa),
        'mu': float(mu),
        'sigma0': float(sigma0),
        'delta1': float(delta1),
        'delta2': float(delta2),
        'nu': (float(nu_hat) if nu_hat is not None else None),
    }


class GapOUSimulator:
    """
    GAP OU 시뮬레이터 (σ_t = σ_0·exp(δ1·SI + δ2·VIX))

    nu가 주어지면 혁신항으로 표준화 Student-t를, None이면 정규를 쓴다.
    표준화되어 있으므로 어느 쪽이든 Var(ε)=1이고 σ_t의 의미는 같다.
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
        clip: float = 3.0,
        nu: Optional[float] = None
    ):
        self.kappa = kappa
        self.mu = mu
        self.sigma0 = sigma0
        self.delta1 = delta1
        self.delta2 = delta2
        self.nu = nu
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

        # 혁신항을 한 번에 뽑는다. 표준화되어 있어 정규/​t 어느 쪽이든 Var=1이다.
        if self.nu is not None:
            eps = standardized_t(self.nu, size=T - 1)
        else:
            eps = np.random.standard_normal(T - 1)

        for t in range(T - 1):
            si_lag = si_future[t] if t > 0 else 0.0
            vix_lag = vix_future[t] if t > 0 else 0.0
            sigma_t = self.sigma0 * np.exp(self.delta1 * si_lag + self.delta2 * vix_lag)
            g[t + 1] = g[t] + self.kappa * (self.mu - g[t]) + sigma_t * eps[t]

        return pd.Series(g)


def fit_gap_ou(
    gap_series: pd.Series,
    si_series: pd.Series,
    vix_series: pd.Series,
    clip: float = 3.0,
    regularization: float = 0.01,
    dist: str = "t"
) -> GapOUSimulator:
    params = fit_ou_with_exog(
        gap_series=gap_series,
        si_series=si_series,
        vix_series=vix_series,
        clip=clip,
        regularization=regularization,
        dist=dist
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
        clip=clip,
        nu=params.get('nu')
    )
