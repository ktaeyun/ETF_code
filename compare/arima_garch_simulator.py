"""
ARIMA-GARCH 시뮬레이터
평균: ARIMA(p,d,q), 변동성: GARCH(1,1)
r_t = μ_t + ε_t,  ε_t = σ_t * z_t,  z_t ~ iid(0,1)
σ_t^2 = ω + α*ε_{t-1}^2 + β*σ_{t-1}^2
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


class ARIMAGARCHSimulator:
    """
    ARIMA-GARCH 시뮬레이터
    ARIMA로 조건부 평균, GARCH로 조건부 분산을 모델링하여 수익률 경로 생성
    """

    def __init__(self, arima_fitted, garch_fitted, last_returns=None, last_resid=None):
        """
        Args:
            arima_fitted: 적합된 ARIMA 모델 (statsmodels ARIMAResults)
            garch_fitted: 적합된 GARCH 모델 (arch, ARIMA 잔차에 적합)
            last_returns: 시뮬레이션 초기화용 마지막 관측 수익률 (길이 max(p,q) 이상)
            last_resid: 시뮬레이션 초기화용 마지막 잔차 (길이 max(p,q) 이상)
        """
        self.arima_fitted = arima_fitted
        self.garch_fitted = garch_fitted
        self._last_returns = np.atleast_1d(last_returns).flatten() if last_returns is not None else None
        self._last_resid = np.atleast_1d(last_resid).flatten() if last_resid is not None else None
        self._parse_arima_params()
        self._parse_garch_params()

    def _parse_arima_params(self):
        """ARIMA 계수 추출 (statsmodels ARIMAResults)"""
        res = self.arima_fitted
        param_names = list(getattr(res, 'param_names', []))
        if not param_names and hasattr(res.params, 'index'):
            param_names = list(res.params.index)
        if not param_names:
            param_names = [f'p{i}' for i in range(len(res.params))]
        # params가 ndarray일 수 있으므로 인덱스로 접근
        params_arr = np.atleast_1d(np.asarray(res.params).flatten())
        param_dict = {name: float(params_arr[i]) for i, name in enumerate(param_names) if i < len(params_arr)}
        self.ar_const = 0.0
        self.ar_lags = []
        self.ma_lags = []
        for name, val in param_dict.items():
            if name == 'const' or name == 'intercept':
                self.ar_const = val
            elif name.startswith('ar.'):
                lag_str = name.split('.')[1]
                lag = int(''.join(c for c in lag_str if c.isdigit()) or '1')
                self.ar_lags.append((lag, val))
            elif name.startswith('ma.'):
                lag_str = name.split('.')[1]
                lag = int(''.join(c for c in lag_str if c.isdigit()) or '1')
                self.ma_lags.append((lag, val))
        self.ar_lags.sort(key=lambda x: x[0])
        self.ma_lags.sort(key=lambda x: x[0])
        self.p = max([k for k, _ in self.ar_lags], default=0)
        self.q = max([k for k, _ in self.ma_lags], default=0)

    def _parse_garch_params(self):
        """GARCH 계수 추출 (arch는 백분율 단위로 적합된 경우 많음)"""
        g = self.garch_fitted
        self.g_omega = float(g.params['omega'])
        self.g_alpha = float(g.params.get('alpha[1]', 0.1))
        self.g_beta = float(g.params.get('beta[1]', 0.8))
        # arch가 백분율 단위(*100)로 적합됐으면 혁신을 소수로 쓰기 위해 0.01
        self._garch_scale = 1.0

    def _initial_history(self, seed=None):
        """시뮬레이션용 초기 이력 (r, epsilon, sigma^2)"""
        if seed is not None:
            np.random.seed(seed)
        p, q = self.p, self.q
        k = max(p, q, 1)
        if self._last_returns is not None and len(self._last_returns) >= k:
            r_history = np.array(self._last_returns[-k:], dtype=float)
        else:
            r_history = np.zeros(k)
            r_history[:] = np.mean(self.arima_fitted.fittedvalues) if hasattr(self.arima_fitted, 'fittedvalues') else 0.0
        if self._last_resid is not None and len(self._last_resid) >= k:
            eps_history = np.array(self._last_resid[-k:], dtype=float)
        else:
            eps_history = np.zeros(k)
        # GARCH 초기 분산: unconditional variance
        omega, alpha, beta = self.g_omega, self.g_alpha, self.g_beta
        if omega <= 0:
            omega = 1e-6
        sigma2_0 = omega / (1 - alpha - beta) if (1 - alpha - beta) > 0 else omega
        sigma2_0 = max(sigma2_0, 1e-12)
        return r_history, eps_history, sigma2_0

    def simulate_returns(self, T, seed=None):
        """
        ARIMA-GARCH 수익률 시뮬레이션
        
        Args:
            T: 시뮬레이션 기간 (일수)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 수익률 (길이 T)
        """
        if seed is not None:
            np.random.seed(seed)
        p, q = self.p, self.q
        r_hist, eps_hist, sigma2 = self._initial_history(seed=None)
        returns = np.zeros(T)
        # GARCH는 백분율 단위로 적합된 경우: sigma_t는 % 단위, 혁신을 소수로 쓰려면 * 0.01
        scale = self._garch_scale  # 0.01 if fit on resid*100
        g_omega = self.g_omega
        g_alpha = self.g_alpha
        g_beta = self.g_beta
        # eps_hist: ARIMA MA용은 소수 단위. GARCH 갱신용은 % 단위로 사용 (eps_prev_pct)
        eps_prev_pct = (eps_hist[-1] / scale) if len(eps_hist) > 0 and scale != 0 else 0.0

        for t in range(T):
            # GARCH: sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2 (단위: %^2)
            sigma2 = g_omega + g_alpha * (eps_prev_pct ** 2) + g_beta * sigma2
            sigma2 = max(sigma2, 1e-12)
            sigma_t_pct = np.sqrt(sigma2)  # % 단위
            z_t = np.random.standard_normal()
            eps_t_pct = sigma_t_pct * z_t  # % 단위
            eps_t = eps_t_pct * scale  # 소수 단위 (수익률)
            # ARIMA 조건부 평균 (소수 단위)
            mu_t = self.ar_const
            for lag, coef in self.ar_lags:
                if lag <= len(r_hist):
                    mu_t += coef * r_hist[-lag]
            for lag, coef in self.ma_lags:
                if lag <= len(eps_hist):
                    mu_t += coef * eps_hist[-lag]
            r_t = mu_t + eps_t
            returns[t] = r_t
            # 이력 갱신
            r_hist = np.roll(r_hist, -1)
            r_hist[-1] = r_t
            eps_hist = np.roll(eps_hist, -1)
            eps_hist[-1] = eps_t  # 소수 (MA용)
            eps_prev_pct = eps_t_pct  # 다음 GARCH 갱신용 (%)

        return pd.Series(returns)

    def simulate_nav_path(self, S0, T, seed=None):
        """
        NAV 경로 시뮬레이션: S_{t+1} = S_t * exp(r_t)
        
        Args:
            S0: 초기 NAV
            T: 시뮬레이션 기간 (일수)
            seed: 랜덤 시드
        
        Returns:
            pd.Series: 시뮬레이션된 NAV 경로 (길이 T)
        """
        returns = self.simulate_returns(T, seed=seed)
        nav_path = np.zeros(T + 1)
        nav_path[0] = S0
        for t in range(T):
            nav_path[t + 1] = nav_path[t] * np.exp(returns.iloc[t])
        return pd.Series(nav_path[1:], index=range(T))


def fit_arima_garch(returns_series, ar_order=(1, 0, 1), garch_p=1, garch_q=1, dist='normal'):
    """
    ARIMA-GARCH 모델 적합: ARIMA로 평균, GARCH로 변동성
    
    Args:
        returns_series: 수익률 시계열 (pd.Series)
        ar_order: ARIMA order (p, d, q). 기본 (1,0,1)
        garch_p: GARCH p
        garch_q: GARCH q
        dist: GARCH 혁신 분포 ('normal', 't' 등)
    
    Returns:
        tuple: (ARIMAGARCHSimulator 인스턴스, 파라미터 dict)
    """
    y = np.asarray(returns_series).flatten()
    T = len(y)
    if T < 30:
        raise ValueError("ARIMA-GARCH 적합을 위해 최소 30개 관측 필요")

    # 1) ARIMA 적합
    print(f"\nARIMA{ar_order} 적합 중...")
    try:
        arima_model = ARIMA(y, order=ar_order)
        arima_fitted = arima_model.fit()
        resid = np.asarray(arima_fitted.resid).flatten()
        print(f"  - ARIMA 적합 완료, AIC={arima_fitted.aic:.4f}")
    except Exception as e:
        print(f"  - ARIMA 적합 실패: {e}, ARIMA(0,0,0)+GARCH 사용")
        arima_model = ARIMA(y, order=(0, 0, 0))
        arima_fitted = arima_model.fit()
        resid = np.asarray(arima_fitted.resid).flatten()

    # 2) GARCH는 잔차의 분산 모델링. arch는 보통 백분율 단위 사용
    resid_pct = resid * 100.0
    print(f"GARCH({garch_p},{garch_q})-{dist} 적합 중 (ARIMA 잔차)...")
    try:
        garch_model = arch_model(
            resid_pct,
            vol='GARCH',
            p=garch_p,
            q=garch_q,
            dist=dist
        )
        garch_fitted = garch_model.fit(disp='off')
        print(f"  - GARCH 적합 완료")
    except Exception as e:
        print(f"  - GARCH 적합 실패: {e}")
        raise

    # 3) 시뮬레이터 생성 (초기 이력 = 마지막 관측)
    k = max(ar_order[0], ar_order[2], 1)
    last_r = y[-k:] if len(y) >= k else y
    last_eps = resid[-k:] if len(resid) >= k else resid
    # GARCH 잔차가 백분율 단위이면 원래 단위로
    if np.median(np.abs(last_eps)) > 0.5:
        last_eps = last_eps / 100.0
    simulator = ARIMAGARCHSimulator(
        arima_fitted=arima_fitted,
        garch_fitted=garch_fitted,
        last_returns=last_r,
        last_resid=last_eps
    )
    # GARCH scale: arch가 100배 데이터로 적합했으면 sigma를 1/100로
    simulator._garch_scale = 0.01

    params = {
        'ar_order': ar_order,
        'garch_order': (garch_p, garch_q),
        'arima_aic': float(arima_fitted.aic),
        'garch_bic': float(garch_fitted.bic),
    }
    return simulator, params
