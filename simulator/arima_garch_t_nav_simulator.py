"""
ARIMAX-GARCH-t NAV 시뮬레이터
- 조건부 평균: ARIMA(p,d,q) + β1*Hash Rate + β2*Unique Addresses (외생변수)
- 조건부 분산: GARCH(1,1)
- 혁신 분포: Student-t (GARCH-t)
- 종속변수: Log Return (y_variables.csv)
- 역변환: Log Return → NAV 경로 (log_returns_to_nav)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


# 독립변수 컬럼명 (nav_variables.csv)
EXOG_COLS = ["Hash Rate (TH/s)", "Unique Addresses"]


def fit_arimax_garch_t(
    log_returns: pd.Series,
    exog: pd.DataFrame = None,
    ar_order: tuple = (1, 0, 1),
    garch_p: int = 1,
    garch_q: int = 1,
) -> "NavArimaGarchTSimulator":
    """
    ARIMA-GARCH-t 적합. exog=None이면 순수 ARIMA(p,d,q)-GARCH-t.

    Args:
        log_returns: 종속변수 Log Return (y_variables)
        exog: 독립변수 DataFrame (None이면 외생변수 미사용)
        ar_order: ARIMA (p, d, q)
        garch_p, garch_q: GARCH 차수

    Returns:
        NavArimaGarchTSimulator 인스턴스
    """
    y = np.asarray(log_returns, dtype=float).flatten()
    if len(y) < 30:
        raise ValueError("최소 30개 관측 필요")

    if exog is not None:
        X = exog[EXOG_COLS].values.astype(float)
        if len(y) != len(X):
            raise ValueError("log_returns와 exog 길이 불일치")
        arima_model = ARIMA(y, exog=X, order=ar_order)
    else:
        X = None
        arima_model = ARIMA(y, order=ar_order)

    # 1) ARIMA(X) 적합
    arima_fitted = arima_model.fit()
    resid = np.asarray(arima_fitted.resid).flatten()

    # 2) GARCH(1,1)-t on residuals
    resid_pct = resid * 100.0
    garch_model = arch_model(
        resid_pct,
        vol="GARCH",
        p=garch_p,
        q=garch_q,
        dist="t",
    )
    garch_fitted = garch_model.fit(disp="off")

    k = max(ar_order[0], ar_order[2], 1)
    return NavArimaGarchTSimulator(
        arima_fitted=arima_fitted,
        garch_fitted=garch_fitted,
        exog_cols=EXOG_COLS if X is not None else [],
        last_returns=y[-k:],
        last_resid=resid[-k:],
        last_exog=X[-1:] if X is not None else None,
        resid_scale=0.01,
    )


class NavArimaGarchTSimulator:
    """
    ARIMAX-GARCH-t 기반 Log Return 시뮬레이터.
    시뮬레이션된 Log Return을 NAV 경로로 역변환하려면 log_returns_to_nav() 사용.
    """

    def __init__(
        self,
        arima_fitted,
        garch_fitted,
        exog_cols,
        last_returns=None,
        last_resid=None,
        last_exog=None,
        resid_scale=0.01,
    ):
        self.arima_fitted = arima_fitted
        self.garch_fitted = garch_fitted
        self.exog_cols = exog_cols
        self._last_returns = np.atleast_1d(last_returns).flatten() if last_returns is not None else None
        self._last_resid = np.atleast_1d(last_resid).flatten() if last_resid is not None else None
        self._last_exog = np.atleast_2d(last_exog) if last_exog is not None else None
        self._resid_scale = resid_scale
        self._parse_arima()
        self._parse_garch()

    def _parse_arima(self):
        res = self.arima_fitted
        param_names = list(
            getattr(res, "param_names", []) or (res.params.index.tolist() if hasattr(res.params, "index") else [])
        )
        if not param_names:
            param_names = [f"p{i}" for i in range(len(res.params))]
        params_arr = np.atleast_1d(np.asarray(res.params).flatten())
        param_dict = {name: float(params_arr[i]) for i, name in enumerate(param_names) if i < len(params_arr)}

        self.ar_const = 0.0
        self.ar_lags = []
        self.ma_lags = []
        self.exog_coef = []
        for name, val in param_dict.items():
            if name == "const" or name == "intercept":
                self.ar_const = val
            elif name.startswith("ar."):
                lag = int("".join(c for c in name.split(".")[-1] if c.isdigit()) or 1)
                self.ar_lags.append((lag, val))
            elif name.startswith("ma."):
                lag = int("".join(c for c in name.split(".")[-1] if c.isdigit()) or 1)
                self.ma_lags.append((lag, val))
        # exog 계수 순서 유지 (x0, x1, ...)
        exog_param_names = sorted([n for n in param_dict if n.startswith("x") and n[1:].isdigit()], key=lambda s: int(s[1:]))
        if exog_param_names:
            self.exog_coef = [param_dict[n] for n in exog_param_names]

        self.ar_lags.sort(key=lambda x: x[0])
        self.ma_lags.sort(key=lambda x: x[0])
        self.p = max([k for k, _ in self.ar_lags], default=0)
        self.q = max([k for k, _ in self.ma_lags], default=0)
        if not self.exog_coef and hasattr(res, "model") and res.model.exog is not None:
            n_exog = res.model.exog.shape[1]
            # statsmodels: const, ar*, ma*, exog*
            n_ar = len(self.ar_lags)
            n_ma = len(self.ma_lags)
            start = 1 + n_ar + n_ma
            for i in range(n_exog):
                if start + i < len(params_arr):
                    self.exog_coef.append(float(params_arr[start + i]))
                else:
                    self.exog_coef.append(0.0)

    def _parse_garch(self):
        g = self.garch_fitted
        self.g_omega = float(g.params["omega"])
        self.g_alpha = float(g.params.get("alpha[1]", 0.1))
        self.g_beta = float(g.params.get("beta[1]", 0.8))
        self._nu = float(g.params.get("nu", 10.0))  # Student-t 자유도

    def _initial_history(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        k = max(self.p, self.q, 1)
        r_hist = (
            np.array(self._last_returns[-k:], dtype=float)
            if self._last_returns is not None and len(self._last_returns) >= k
            else np.zeros(k)
        )
        eps_hist = (
            np.array(self._last_resid[-k:], dtype=float)
            if self._last_resid is not None and len(self._last_resid) >= k
            else np.zeros(k)
        )
        sigma2 = self.g_omega / (1 - self.g_alpha - self.g_beta) if (1 - self.g_alpha - self.g_beta) > 0 else self.g_omega
        sigma2 = max(sigma2, 1e-12)
        return r_hist, eps_hist, sigma2

    def simulate_returns(
        self,
        T: int,
        exog_future: pd.DataFrame = None,
        seed: int = None,
    ) -> pd.Series:
        """
        Log Return T기 시뮬레이션.

        Args:
            T: 시뮬레이션 기간(일수)
            exog_future: 미래 독립변수 (Hash Rate, Unique Addresses). 행 수 T, 컬럼 self.exog_cols.
                        None이면 학습 구간 마지막 행을 T기 동안 반복 사용.
            seed: 랜덤 시드

        Returns:
            pd.Series: 길이 T인 Log Return 시뮬레이션
        """
        if seed is not None:
            np.random.seed(seed)
        scale = self._resid_scale
        r_hist, eps_hist, sigma2 = self._initial_history(seed=None)

        if exog_future is None:
            # 마지막 관측 exog 반복
            if self._last_exog is not None and self._last_exog.size > 0:
                row = self._last_exog[-1]
                X_future = np.tile(row, (T, 1))
            else:
                X_future = np.zeros((T, len(self.exog_cols)))
        else:
            X_future = exog_future[self.exog_cols].values.astype(float)
            if len(X_future) < T:
                last = X_future[-1:]
                X_future = np.vstack([X_future, np.tile(last, (T - len(X_future), 1))])
            else:
                X_future = X_future[:T]

        returns = np.zeros(T)
        eps_prev_pct = (eps_hist[-1] / scale) if len(eps_hist) and scale != 0 else 0.0

        for t in range(T):
            sigma2 = self.g_omega + self.g_alpha * (eps_prev_pct ** 2) + self.g_beta * sigma2
            sigma2 = max(sigma2, 1e-12)
            sigma_t_pct = np.sqrt(sigma2)
            # Student-t 혁신
            z_t = np.random.standard_t(self._nu)
            # 표준화: E[z]=0, Var(z)=nu/(nu-2) for nu>2
            if self._nu > 2:
                z_t = z_t / np.sqrt(self._nu / (self._nu - 2))
            eps_t_pct = sigma_t_pct * z_t
            eps_t = eps_t_pct * scale

            mu_t = self.ar_const
            for lag, coef in self.ar_lags:
                if lag <= len(r_hist):
                    mu_t += coef * r_hist[-lag]
            for lag, coef in self.ma_lags:
                if lag <= len(eps_hist):
                    mu_t += coef * eps_hist[-lag]
            if self.exog_coef and len(self.exog_coef) == X_future.shape[1]:
                mu_t += np.dot(self.exog_coef, X_future[t])

            r_t = mu_t + eps_t
            returns[t] = r_t

            r_hist = np.roll(r_hist, -1)
            r_hist[-1] = r_t
            eps_hist = np.roll(eps_hist, -1)
            eps_hist[-1] = eps_t
            eps_prev_pct = eps_t_pct

        return pd.Series(returns)

    def simulate_nav_path(
        self,
        S0: float,
        T: int,
        exog_future: pd.DataFrame = None,
        seed: int = None,
    ) -> pd.Series:
        """
        Log Return 시뮬레이션 후 역변환하여 NAV 경로 반환. S_{t+1} = S_t * exp(r_t).

        Args:
            S0: 초기 NAV
            T: 기간
            exog_future: 미래 독립변수 (None이면 마지막 exog 반복)
            seed: 랜덤 시드

        Returns:
            pd.Series: 길이 T인 NAV 경로
        """
        log_ret = self.simulate_returns(T, exog_future=exog_future, seed=seed)
        return log_returns_to_nav(log_ret, S0)


def log_returns_to_nav(log_returns: pd.Series, S0: float) -> pd.Series:
    """
    Log Return 시계열을 NAV 경로로 역변환.
    r_t = log(S_t / S_{t-1}) => S_t = S_{t-1} * exp(r_t).

    Args:
        log_returns: Log Return 시계열 (길이 T)
        S0: 초기 NAV

    Returns:
        pd.Series: 길이 T인 NAV 경로 (S_1, ..., S_T)
    """
    r = np.asarray(log_returns).flatten()
    nav = np.zeros(len(r) + 1)
    nav[0] = S0
    for t in range(len(r)):
        nav[t + 1] = nav[t] * np.exp(r[t])
    return pd.Series(nav[1:], index=log_returns.index if hasattr(log_returns, "index") else range(len(r)))
